# src/combat_env.py
import logging
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.agent import SimpleAgent
from src.communicator import Communicator
from src.game_state import GameState
from src.run_tracker import RunTracker
from src.state_encoder import StateEncoder
from src.action_space import ActionSpace

logger = logging.getLogger(__name__)


class CombatEnv(gym.Env):
    """
    Gymnasium env for STS combat. One episode = one combat encounter.
    Non-combat screens are handled automatically by SimpleAgent.
    """
    metadata = {"render_modes": []}

    def __init__(self, communicator: Communicator,
                 run_tracker: Optional[RunTracker] = None):
        super().__init__()
        self.communicator = communicator
        self.run_tracker = run_tracker or RunTracker()
        self.simple_agent = SimpleAgent()
        self.encoder = StateEncoder()
        self._action_space = ActionSpace()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(StateEncoder.OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ActionSpace.TOTAL_ACTIONS)

        self._current_state: Optional[GameState] = None
        self._episode_steps: int = 0
        self._buffered_state: Optional[GameState] = None
        self._initialized: bool = False

    def action_masks(self) -> np.ndarray:
        if self._current_state is None:
            return np.ones(ActionSpace.TOTAL_ACTIONS, dtype=np.bool_)
        return self._action_space.get_action_mask(self._current_state)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_steps = 0

        if not self._initialized:
            self.communicator.send_ready()
            self._initialized = True

        state = self._advance_to_combat()
        self._current_state = state
        logger.debug("Combat started: floor=%d hp=%d/%d",
                     state.floor, state.current_hp, state.max_hp)
        return self.encoder.encode(state), {}

    def step(self, action: int):
        assert self._current_state is not None, "Call reset() first"
        self._episode_steps += 1

        command = self._action_space.action_to_command(action, self._current_state)
        logger.debug("Floor %d | HP %d/%d | action=%s",
                     self._current_state.floor, self._current_state.current_hp,
                     self._current_state.max_hp, command)
        self.communicator.send_command(command)

        state = self.communicator.receive_state()
        if state is None:
            # Connection closed mid-episode
            obs = self.encoder.encode(self._current_state)
            return obs, 0.0, True, False, {}

        if state.is_in_combat:
            self._current_state = state
            return self.encoder.encode(state), 0.0, False, False, {}

        # Combat ended — compute reward from post-combat state
        reward = self._compute_reward(state)
        terminal_obs = self.encoder.encode(self._current_state)
        info = {
            "episode": {
                "r": reward,
                "l": self._episode_steps,
                "hp": state.current_hp,
                "max_hp": state.max_hp,
                "floor": state.floor,
            }
        }

        # GAME_OVER is always handled here and never buffered — _buffered_state stays None.
        # _advance_to_combat() also handles GAME_OVER but only for deaths between episodes.
        if state.screen_type == "GAME_OVER":
            self.run_tracker.record_run(state)
            summary = self.run_tracker.summary()
            logger.info(
                "GAME_OVER | floor=%d | runs=%d | win_rate=%.1f%%",
                state.floor, summary["total_runs"], summary["win_rate"] * 100,
            )
            self.communicator.send_command("PROCEED")
            self._buffered_state = None
        else:
            self._buffered_state = state

        self._current_state = None
        return terminal_obs, reward, True, False, info

    def _compute_reward(self, state: GameState) -> float:
        if state.screen_type == "GAME_OVER" or state.current_hp <= 0:
            return -1.0
        return state.current_hp / max(state.max_hp, 1)

    def _advance_to_combat(self) -> GameState:
        """Loop through non-combat screens until combat is reached."""
        state = self._buffered_state
        self._buffered_state = None

        while True:
            if state is None:
                state = self.communicator.receive_state()
            if state is None:
                raise RuntimeError("Connection closed while waiting for combat")

            if state.error:
                logger.warning("Error from game: %s", state.error)
                state = None
                continue

            if not state.ready_for_command:
                logger.debug("Waiting for game to be ready (screen=%s)", state.screen_type)
                state = None
                continue

            if not state.in_game:
                if "START" in state.available_commands:
                    logger.info("Starting new run...")
                    self.communicator.send_command("START IRONCLAD 0")
                state = None
                continue

            if state.screen_type == "GAME_OVER":
                self.run_tracker.record_run(state)
                summary = self.run_tracker.summary()
                logger.info(
                    "GAME_OVER (between combats) | runs=%d | win_rate=%.1f%%",
                    summary["total_runs"], summary["win_rate"] * 100,
                )
                self.communicator.send_command("PROCEED")
                state = None
                continue

            if state.is_in_combat:
                return state

            action = self.simple_agent.act(state)
            logger.info("Floor %d | HP %d/%d | Screen: %s | Action: %s",
                        state.floor, state.current_hp, state.max_hp,
                        state.screen_type, action)
            self.communicator.send_command(action)
            state = None
