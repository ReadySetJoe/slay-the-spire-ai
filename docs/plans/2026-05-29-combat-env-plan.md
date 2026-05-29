# CombatEnv Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the stub `RLAgent` with a proper `gymnasium.Env` so `MaskablePPO.learn()` drives the game loop and actually trains.

**Architecture:** `CombatEnv` wraps `Communicator` directly; one episode = one combat. Non-combat screens are handled transparently by `SimpleAgent` inside `_advance_to_combat()`. A `_buffered_state` bridges the `step()`/`reset()` boundary when combat ends mid-screen-sequence. `EpisodeLoggerCallback` logs per-episode stats and periodic summaries. `main.py --rl` uses `model.learn()` with `CallbackList`.

**Tech Stack:** Python 3.11+, gymnasium, sb3-contrib (MaskablePPO), stable-baselines3 (CheckpointCallback), pytest, unittest.mock

---

### Task 1: Delete Dead Code

`src/rl_agent.py` and its tests are no longer needed.

**Files:**
- Delete: `src/rl_agent.py`
- Delete: `tests/test_rl_agent.py`

**Step 1: Delete the files**

```bash
rm src/rl_agent.py tests/test_rl_agent.py
```

**Step 2: Verify existing tests still pass**

Run: `source .venv/Scripts/activate && python -m pytest -v --ignore=tests/test_rl_agent.py`
Expected: All remaining tests PASS

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove rl_agent.py (replaced by CombatEnv)"
```

---

### Task 2: CombatEnv

**Files:**
- Create: `src/combat_env.py`
- Create: `tests/test_combat_env.py`

**Step 1: Write failing tests**

```python
# tests/test_combat_env.py
import json
from unittest.mock import MagicMock
import numpy as np
import pytest
from src.combat_env import CombatEnv
from src.game_state import GameState
from src.action_space import ActionSpace
from src.state_encoder import StateEncoder


def _combat(hp=70, max_hp=80, floor=1):
    return GameState.from_json(json.dumps({
        "available_commands": ["PLAY", "END"],
        "ready_for_command": True, "in_game": True,
        "game_state": {
            "screen_type": "NONE",
            "seed": 1, "floor": floor, "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": hp, "max_hp": max_hp, "gold": 99,
            "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
            "combat_state": {
                "hand": [{"id": "Strike_R", "name": "Strike", "cost": 1,
                           "type": "ATTACK", "is_playable": True, "has_target": True, "uuid": "a1"}],
                "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
                "monsters": [{"name": "Jaw Worm", "current_hp": 42, "max_hp": 42,
                               "block": 0, "intent": "ATTACK", "is_gone": False}],
                "player": {"current_hp": hp, "max_hp": max_hp, "block": 0,
                            "energy": 3, "powers": []},
                "turn": 1,
            },
        }
    }))


def _reward_screen(hp=65, max_hp=80, floor=1):
    return GameState.from_json(json.dumps({
        "available_commands": ["CHOOSE", "PROCEED"],
        "ready_for_command": True, "in_game": True,
        "game_state": {
            "screen_type": "COMBAT_REWARD",
            "screen_state": {"rewards": [{"reward_type": "GOLD", "gold": 10}]},
            "seed": 1, "floor": floor, "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": hp, "max_hp": max_hp, "gold": 99,
            "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
            "combat_state": None,
        }
    }))


def _game_over(hp=0, max_hp=80, floor=3):
    return GameState.from_json(json.dumps({
        "available_commands": ["PROCEED"],
        "ready_for_command": True, "in_game": True,
        "game_state": {
            "screen_type": "GAME_OVER",
            "seed": 1, "floor": floor, "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": hp, "max_hp": max_hp, "gold": 99,
            "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
            "combat_state": None,
        }
    }))


def test_action_masks_shape():
    env = CombatEnv(communicator=MagicMock())
    env._current_state = _combat()
    mask = env.action_masks()
    assert mask.shape == (ActionSpace.TOTAL_ACTIONS,)
    assert mask.dtype == np.bool_


def test_action_masks_all_ones_when_no_state():
    env = CombatEnv(communicator=MagicMock())
    assert env.action_masks().all()


def test_compute_reward_survival():
    env = CombatEnv(communicator=MagicMock())
    assert env._compute_reward(_reward_screen(hp=65, max_hp=80)) == pytest.approx(65 / 80)


def test_compute_reward_death():
    env = CombatEnv(communicator=MagicMock())
    assert env._compute_reward(_game_over(hp=0)) == -1.0


def test_step_continues_combat():
    comm = MagicMock()
    comm.receive_state.return_value = _combat(hp=65)
    env = CombatEnv(communicator=comm)
    env._current_state = _combat(hp=70)
    env._combat_start_hp = 70

    obs, reward, done, truncated, info = env.step(ActionSpace.END_TURN_ACTION)

    assert reward == 0.0
    assert done == False
    assert truncated == False
    assert obs.shape == (StateEncoder.OBS_SIZE,)


def test_step_combat_ends_buffers_state():
    comm = MagicMock()
    reward_state = _reward_screen(hp=65)
    comm.receive_state.return_value = reward_state
    env = CombatEnv(communicator=comm)
    env._current_state = _combat(hp=70)
    env._combat_start_hp = 70

    obs, reward, done, truncated, info = env.step(ActionSpace.END_TURN_ACTION)

    assert done == True
    assert reward == pytest.approx(65 / 80)
    assert env._buffered_state is reward_state
    assert obs.shape == (StateEncoder.OBS_SIZE,)


def test_step_combat_ends_includes_episode_info():
    comm = MagicMock()
    comm.receive_state.return_value = _reward_screen(hp=65, floor=2)
    env = CombatEnv(communicator=comm)
    env._current_state = _combat(hp=70, floor=2)
    env._combat_start_hp = 70

    _, _, _, _, info = env.step(ActionSpace.END_TURN_ACTION)

    assert "episode" in info
    assert info["episode"]["r"] == pytest.approx(65 / 80)
    assert info["episode"]["floor"] == 2


def test_step_game_over_sends_proceed():
    comm = MagicMock()
    comm.receive_state.return_value = _game_over(hp=0)
    env = CombatEnv(communicator=comm)
    env._current_state = _combat(hp=10)
    env._combat_start_hp = 10

    _, reward, done, _, _ = env.step(ActionSpace.END_TURN_ACTION)

    assert done == True
    assert reward == -1.0
    assert env._buffered_state is None
    comm.send_command.assert_called_with("PROCEED")


def test_reset_uses_buffered_state():
    comm = MagicMock()
    next_combat = _combat(hp=65)
    comm.receive_state.return_value = next_combat
    env = CombatEnv(communicator=comm)
    env._initialized = True
    env._buffered_state = _reward_screen(hp=65)

    obs, info = env.reset()

    # receive_state called once (for next_combat after SimpleAgent handles reward screen)
    assert comm.receive_state.call_count == 1
    assert env._buffered_state is None
    assert obs.shape == (StateEncoder.OBS_SIZE,)
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_combat_env.py -v`
Expected: ImportError — `No module named 'src.combat_env'`

**Step 3: Implement CombatEnv**

```python
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
        self._combat_start_hp: int = 0
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
        self._combat_start_hp = state.current_hp
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
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_combat_env.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/combat_env.py tests/test_combat_env.py
git commit -m "feat: CombatEnv gymnasium wrapper for STS combat"
```

---

### Task 3: EpisodeLoggerCallback

**Files:**
- Create: `src/callbacks.py`

No dedicated unit tests — callback behavior is logging/side-effect only and is validated via the full test suite passing.

**Step 1: Implement EpisodeLoggerCallback**

```python
# src/callbacks.py
import logging
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class EpisodeLoggerCallback(BaseCallback):
    """Logs per-episode stats and periodic summaries to the game log."""

    def __init__(self, summary_freq: int = 100):
        super().__init__(verbose=0)
        self.summary_freq = summary_freq
        self._episode_count = 0
        self._episode_stats: list = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done and "episode" in info:
                self._log_episode(info["episode"])
        return True

    def _log_episode(self, ep: dict):
        self._episode_count += 1
        self._episode_stats.append(ep)
        logger.info(
            "[Episode %d] reward=%.2f | steps=%d | hp=%d/%d | floor=%d | total_steps=%d",
            self._episode_count, ep["r"], ep["l"],
            ep.get("hp", 0), ep.get("max_hp", 0), ep.get("floor", 0),
            self.num_timesteps,
        )
        if self._episode_count % self.summary_freq == 0:
            recent = self._episode_stats[-self.summary_freq:]
            avg_r = sum(e["r"] for e in recent) / len(recent)
            avg_l = sum(e["l"] for e in recent) / len(recent)
            win_rate = sum(1 for e in recent if e["r"] > 0) / len(recent)
            avg_floor = sum(e.get("floor", 0) for e in recent) / len(recent)
            logger.info(
                "[Summary ep %d-%d] avg_reward=%.2f | avg_steps=%.1f | win_rate=%.1f%% | avg_floor=%.1f",
                self._episode_count - self.summary_freq + 1, self._episode_count,
                avg_r, avg_l, win_rate * 100, avg_floor,
            )
```

**Step 2: Verify imports work**

Run: `source .venv/Scripts/activate && python -c "from src.callbacks import EpisodeLoggerCallback; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/callbacks.py
git commit -m "feat: EpisodeLoggerCallback for RL training progress"
```

---

### Task 4: Update main.py + Full Test Suite

**Files:**
- Modify: `main.py`

**Step 1: Update main.py**

Replace the entire file:

```python
# main.py
import logging
import os
import sys

from src.communicator import Communicator
from src.run_tracker import RunTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("game.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def main():
    use_rl = "--rl" in sys.argv

    communicator = Communicator()
    tracker = RunTracker(log_path="data/run_log.jsonl")

    if use_rl:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from src.combat_env import CombatEnv
        from src.callbacks import EpisodeLoggerCallback

        env = CombatEnv(communicator=communicator, run_tracker=tracker)
        model_path = "data/combat_model.zip"

        if os.path.exists(model_path):
            model = MaskablePPO.load(model_path, env=env)
            logger.info("Loaded existing model from %s", model_path)
        else:
            model = MaskablePPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                verbose=1,
            )
            logger.info("Created new MaskablePPO model")

        os.makedirs("data/checkpoints", exist_ok=True)
        callbacks = CallbackList([
            EpisodeLoggerCallback(summary_freq=100),
            CheckpointCallback(
                save_freq=1000,
                save_path="data/checkpoints/",
                name_prefix="combat",
                verbose=1,
            ),
        ])

        logger.info("Starting RL training (MaskablePPO)...")
        model.learn(total_timesteps=10_000_000, callback=callbacks)
    else:
        from src.agent import SimpleAgent
        from src.game_loop import GameLoop

        agent = SimpleAgent()
        loop = GameLoop(communicator, agent, run_tracker=tracker)
        loop.run()


if __name__ == "__main__":
    main()
```

**Step 2: Run full test suite**

Run: `source .venv/Scripts/activate && python -m pytest -v`
Expected: All tests PASS (49 tests: original suite minus the 3 deleted rl_agent tests, plus 9 new combat_env tests)

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: wire CombatEnv into main.py with model.learn()"
```
