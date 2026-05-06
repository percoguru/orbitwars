"""Gymnasium wrapper for the Kaggle orbit_wars environment.

Game summary (from the competition getting-started notebook):
- 100×100 board with a sun at the centre (radius 10, destroys fleets)
- Planets produce ships every turn; inner planets rotate, outer planets are static
- Your agent returns a list of moves: [from_planet_id, angle_radians, num_ships]
- Fleets fly straight at the given angle; speed scales with fleet size (1–6 units/turn)
- Combat: arriving ships subtract from the defender's garrison; ownership flips when it goes negative
- Comets are temporary planets that drift through on elliptical paths
- Win condition: most total ships (planets + fleets) when time runs out, or last player standing

Observation space (Dict)
─────────────────────────────────────────────────────────────────────────────
  Key               Shape              Dtype     Description
  planets           (MAX_PLANETS, 7)  float32   [id, owner, x, y, radius, ships, production]
                                                  zero-padded; owner -1 = neutral
  n_planets         (1,)              int32     number of active planets
  fleets            (MAX_FLEETS, 7)   float32   [id, owner, x, y, angle, from_planet_id, ships]
                                                  zero-padded
  n_fleets          (1,)              int32     number of active fleets
  player            (1,)              int32     agent's player index (always 0)
  angular_velocity  (1,)              float32   inner-planet rotation speed (rad/turn)
  step              (1,)              int32     current episode step (0-based)

Action space: Box(0, 1, shape=(MAX_PLANETS, 3), dtype=float32)
─────────────────────────────────────────────────────────────────────────────
  Row i maps to the planet in obs["planets"][i].
  Column 0  send       — >0.5 launches a fleet from this planet this turn
  Column 1  angle      — launch direction normalised [0, 1] → multiplied by 2π internally
  Column 2  fraction   — fraction of available ships to send [0, 1]

  Rows for planets not owned by the agent, or empty padding rows, are ignored.

Rewards:
  +1.0 on win, -1.0 on loss/draw, 0.0 every non-terminal step.
  Pass reward_shaping=True to also add a small per-step signal:
      0.01 × (agent_ships − max_opponent_ships) / 1000
"""

import math
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import agents as _orbit_wars_agents

MAX_PLANETS = 64   # 40 regular + up to 20 comets (5 spawns × 4) + buffer
MAX_FLEETS = 256


class OrbitWarsEnv(gym.Env):
    """Single-agent Gymnasium interface to orbit_wars.

    The agent always plays as player 0. All other players are driven by
    opponent policies that are called automatically inside step().

    Args:
        opponent: A built-in agent name ("random" / "starter"), a callable
                  with signature fn(obs_dict) -> [[planet_id, angle, ships], ...],
                  or a list of either for multi-player games. Lists are
                  padded to num_players-1 using the last element.
        num_players: 2 or 4.
        render_mode: None, "ansi" (return string), or "human" (print to stdout).
        reward_shaping: If True, add a small intermediate reward each step
                        proportional to the agent's ship advantage.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 1}

    def __init__(
        self,
        opponent: Any = "random",
        num_players: int = 2,
        render_mode: str | None = None,
        reward_shaping: bool = False,
    ):
        super().__init__()
        assert num_players in (2, 4), "num_players must be 2 or 4"
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.num_players = num_players
        self.render_mode = render_mode
        self.reward_shaping = reward_shaping
        self.spec = None

        raw = opponent if isinstance(opponent, (list, tuple)) else [opponent]
        self._opponents: list = [self._resolve_agent(o) for o in raw]
        while len(self._opponents) < num_players - 1:
            self._opponents.append(self._opponents[-1])

        # Planets: [id, owner, x, y, radius, ships, production]
        # Fleets:  [id, owner, x, y, angle, from_planet_id, ships]
        self.observation_space = spaces.Dict({
            "planets": spaces.Box(
                low=-1.0, high=10_000.0, shape=(MAX_PLANETS, 7), dtype=np.float32
            ),
            "n_planets": spaces.Box(0, MAX_PLANETS, shape=(1,), dtype=np.int32),
            "fleets": spaces.Box(
                low=-1.0, high=10_000.0, shape=(MAX_FLEETS, 7), dtype=np.float32
            ),
            "n_fleets": spaces.Box(0, MAX_FLEETS, shape=(1,), dtype=np.int32),
            "player": spaces.Box(0, 3, shape=(1,), dtype=np.int32),
            "angular_velocity": spaces.Box(0.0, 0.1, shape=(1,), dtype=np.float32),
            "step": spaces.Box(0, 500, shape=(1,), dtype=np.int32),
        })

        # Per planet slot: [send, angle_norm, ships_fraction]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(MAX_PLANETS, 3), dtype=np.float32
        )

        self._env = None
        self._planet_slots: dict[int, int] = {}  # planet_id → obs row index

    # ------------------------------------------------------------------ #
    # Gymnasium core API                                                   #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        config = {"seed": seed} if seed is not None else {}
        self._env = make("orbit_wars", configuration=config, debug=False)
        self._env.reset()
        obs = self._build_obs()
        return obs, self._build_info()

    def step(self, action: np.ndarray):
        assert self._env is not None, "Call reset() before step()"

        # Player 0's action decoded from the Box array
        all_actions: list = [None] * self.num_players
        all_actions[0] = self._decode_action(action, player_id=0)

        # Opponents act on their own observations
        for opp_idx, opp_fn in enumerate(self._opponents):
            pid = opp_idx + 1
            opp_obs = self._env.state[pid].observation
            all_actions[pid] = opp_fn(self._obs_to_dict(opp_obs))

        self._env.step(all_actions)

        agent_state = self._env.state[0]
        terminated = agent_state.status == "DONE"
        reward = float(agent_state.reward) if terminated else 0.0

        if self.reward_shaping:
            reward += self._shaping_reward()

        return self._build_obs(), reward, terminated, False, self._build_info()

    def render(self):
        if self._env is None:
            return None
        text = self._env.render(mode="ansi")
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self):
        self._env = None

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_agent(agent):
        """Accept a named built-in string or any callable."""
        if callable(agent):
            return agent
        if agent in _orbit_wars_agents:
            return _orbit_wars_agents[agent]
        raise ValueError(
            f"Unknown agent '{agent}'. Available: {list(_orbit_wars_agents.keys())}"
        )

    def _build_obs(self) -> dict:
        raw = self._env.state[0].observation
        planets_raw = list(getattr(raw, "planets", []) or [])
        fleets_raw = list(getattr(raw, "fleets", []) or [])

        planet_arr = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
        self._planet_slots = {}
        n_planets = min(len(planets_raw), MAX_PLANETS)
        for i, p in enumerate(planets_raw[:MAX_PLANETS]):
            planet_arr[i] = p
            self._planet_slots[int(p[0])] = i

        fleet_arr = np.zeros((MAX_FLEETS, 7), dtype=np.float32)
        n_fleets = min(len(fleets_raw), MAX_FLEETS)
        for i, f in enumerate(fleets_raw[:MAX_FLEETS]):
            fleet_arr[i] = f

        return {
            "planets": planet_arr,
            "n_planets": np.array([n_planets], dtype=np.int32),
            "fleets": fleet_arr,
            "n_fleets": np.array([n_fleets], dtype=np.int32),
            "player": np.array([int(getattr(raw, "player", 0))], dtype=np.int32),
            "angular_velocity": np.array(
                [float(getattr(raw, "angular_velocity", 0.0))], dtype=np.float32
            ),
            "step": np.array([int(getattr(raw, "step", 0))], dtype=np.int32),
        }

    def _build_info(self) -> dict:
        s = self._env.state[0]
        return {"status": s.status, "reward": float(s.reward)}

    def _decode_action(self, action: np.ndarray, player_id: int) -> list:
        """Convert the Box action array to a kaggle move list.

        The kaggle action format is [[from_planet_id, angle_radians, num_ships], ...]
        See orbit_wars.py interpreter → process_moves() for validation rules:
          - from_planet must be owned by this player
          - ships must be > 0 and ≤ planet's current garrison
        """
        raw = self._env.state[player_id].observation
        planets_raw = list(getattr(raw, "planets", []) or [])
        moves = []
        for p in planets_raw:
            pid, owner, _, _, _, ships, _ = p
            pid, owner, ships = int(pid), int(owner), int(ships)
            if owner != player_id or ships <= 0:
                continue
            slot = self._planet_slots.get(pid)
            if slot is None or slot >= MAX_PLANETS:
                continue
            send, angle_norm, ships_frac = action[slot]
            if float(send) <= 0.5:
                continue
            # Map normalised angle [0, 1] → radians [0, 2π]
            angle = float(angle_norm) * 2.0 * math.pi
            # Map ships fraction [0, 1] → actual ship count
            n_ships = max(1, int(float(ships_frac) * ships))
            moves.append([pid, angle, n_ships])
        return moves

    def _shaping_reward(self) -> float:
        """Small intermediate reward: agent ship advantage over best opponent."""
        raw = self._env.state[0].observation
        planets_raw = list(getattr(raw, "planets", []) or [])
        fleets_raw = list(getattr(raw, "fleets", []) or [])

        ship_counts = {}
        for p in planets_raw:
            owner = int(p[1])
            if owner >= 0:
                ship_counts[owner] = ship_counts.get(owner, 0) + int(p[5])
        for f in fleets_raw:
            owner = int(f[1])
            ship_counts[owner] = ship_counts.get(owner, 0) + int(f[6])

        my_ships = ship_counts.get(0, 0)
        opp_ships = max((v for k, v in ship_counts.items() if k != 0), default=0)
        return 0.01 * (my_ships - opp_ships) / 1000.0

    @staticmethod
    def _obs_to_dict(obs) -> dict:
        """Convert a kaggle SimpleNamespace observation to a plain dict.

        Both built-in agents (random, starter) and custom agents written like
        the getting-started tutorial handle dict obs via obs.get("key", default).
        """
        return {
            "player": getattr(obs, "player", 0),
            "planets": list(getattr(obs, "planets", []) or []),
            "fleets": list(getattr(obs, "fleets", []) or []),
            "angular_velocity": getattr(obs, "angular_velocity", 0.0),
            "initial_planets": list(getattr(obs, "initial_planets", []) or []),
            "next_fleet_id": getattr(obs, "next_fleet_id", 0),
            "comets": list(getattr(obs, "comets", []) or []),
            "comet_planet_ids": list(getattr(obs, "comet_planet_ids", []) or []),
            "step": getattr(obs, "step", 0),
            "remainingOverageTime": getattr(obs, "remainingOverageTime", 60),
        }
