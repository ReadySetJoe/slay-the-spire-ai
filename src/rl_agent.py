import logging
import os
import numpy as np
from typing import Optional

from sb3_contrib import MaskablePPO
from gymnasium import spaces, Env

from src.agent import Agent, SimpleAgent
from src.game_state import GameState
from src.state_encoder import StateEncoder
from src.action_space import ActionSpace

logger = logging.getLogger(__name__)


class DummyCombatEnv(Env):
    """Minimal Gym env for initializing MaskablePPO. Not used for stepping."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(StateEncoder.OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ActionSpace.TOTAL_ACTIONS)
        self._current_mask = np.ones(ActionSpace.TOTAL_ACTIONS, dtype=np.bool_)

    def action_masks(self) -> np.ndarray:
        return self._current_mask

    def reset(self, seed=None, options=None):
        return np.zeros(StateEncoder.OBS_SIZE, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(StateEncoder.OBS_SIZE, dtype=np.float32), 0.0, True, False, {}


class RLAgent(Agent):
    """Hybrid agent: PPO for combat, SimpleAgent for everything else."""

    def __init__(self, model_path: str = "data/combat_model.zip",
                 learning_rate: float = 3e-4,
                 train: bool = True):
        self.model_path = model_path
        self.train = train
        self.simple_agent = SimpleAgent()
        self.encoder = StateEncoder()
        self.action_space = ActionSpace()

        # Combat episode tracking
        self.in_combat = False
        self.combat_start_hp: Optional[int] = None
        self.combat_observations: list = []
        self.combat_actions: list = []
        self.combat_rewards: list = []
        self.combat_masks: list = []

        # Training buffer
        self.episode_count = 0
        self.total_steps = 0

        # Initialize or load model
        self.env = DummyCombatEnv()
        if os.path.exists(model_path):
            logger.info("Loading existing model from %s", model_path)
            self.model = MaskablePPO.load(model_path, env=self.env)
        else:
            logger.info("Creating new model")
            self.model = MaskablePPO(
                "MlpPolicy",
                self.env,
                learning_rate=learning_rate,
                n_steps=256,
                batch_size=64,
                n_epochs=4,
                verbose=0,
            )

    def act(self, state: GameState) -> str:
        if not state.is_in_combat:
            # If we were in combat, combat just ended
            if self.in_combat:
                self._end_combat(state)
            return self.simple_agent.act(state)

        # Start tracking new combat
        if not self.in_combat:
            self._start_combat(state)

        # Get observation and mask
        obs = self.encoder.encode(state)
        mask = self.action_space.get_action_mask(state)

        # Set mask on dummy env for prediction
        self.env._current_mask = mask

        # Get action from PPO
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=not self.train)

        # Store experience
        self.combat_observations.append(obs)
        self.combat_actions.append(action)
        self.combat_masks.append(mask)

        # Convert to game command
        command = self.action_space.action_to_command(int(action), state)
        return command

    def _start_combat(self, state: GameState):
        self.in_combat = True
        self.combat_start_hp = state.current_hp
        self.combat_observations = []
        self.combat_actions = []
        self.combat_rewards = []
        self.combat_masks = []

    def _end_combat(self, state: GameState):
        self.in_combat = False

        if self.combat_start_hp is None:
            return

        # Calculate reward
        if state.current_hp <= 0 or state.screen_type == "GAME_OVER":
            reward = -1.0
        else:
            reward = state.current_hp / max(state.max_hp, 1)

        # Distribute reward to all steps in this combat
        n_steps = len(self.combat_observations)
        if n_steps > 0:
            self.episode_count += 1
            self.total_steps += n_steps
            logger.info(
                "Combat #%d ended: reward=%.2f, steps=%d, total_steps=%d",
                self.episode_count, reward, n_steps, self.total_steps,
            )

            if self.train:
                self._train_on_combat(reward)

        self.combat_start_hp = None

    def _train_on_combat(self, reward: float):
        # Accumulate experience and save periodically
        if self.episode_count % 10 == 0:
            self.save_model()

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        self.model.save(self.model_path)
        logger.info("Model saved to %s (episode %d)", self.model_path, self.episode_count)

    def on_game_over(self, state: GameState):
        """Called when the full run ends."""
        if self.in_combat:
            self._end_combat(state)
        if self.train:
            self.save_model()
