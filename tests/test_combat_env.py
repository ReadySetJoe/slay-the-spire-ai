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

    obs, reward, done, truncated, info = env.step(ActionSpace.END_TURN_ACTION)

    assert reward == pytest.approx(-5 / 80)
    assert done == False
    assert truncated == False
    assert obs.shape == (StateEncoder.OBS_SIZE,)


def test_step_combat_ends_buffers_state():
    comm = MagicMock()
    reward_state = _reward_screen(hp=65)
    comm.receive_state.return_value = reward_state
    env = CombatEnv(communicator=comm)
    env._current_state = _combat(hp=70)

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

    _, _, _, _, info = env.step(ActionSpace.END_TURN_ACTION)

    assert "episode" in info
    assert info["episode"]["r"] == pytest.approx(65 / 80)
    assert info["episode"]["floor"] == 2


def test_step_game_over_sends_proceed():
    comm = MagicMock()
    comm.receive_state.return_value = _game_over(hp=0)
    env = CombatEnv(communicator=comm)
    env._current_state = _combat(hp=10)

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


def _combat_energy(energy: int, hp=70, max_hp=80):
    """Combat state fixture with configurable remaining energy."""
    return GameState.from_json(json.dumps({
        "available_commands": ["PLAY", "END"],
        "ready_for_command": True, "in_game": True,
        "game_state": {
            "screen_type": "NONE",
            "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": hp, "max_hp": max_hp, "gold": 99,
            "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
            "combat_state": {
                "hand": [],
                "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
                "monsters": [{"name": "Jaw Worm", "current_hp": 42, "max_hp": 42,
                               "block": 0, "intent": "ATTACK", "is_gone": False}],
                "player": {"current_hp": hp, "max_hp": max_hp, "block": 0,
                            "energy": energy, "powers": []},
                "turn": 1,
            },
        }
    }))


def test_compute_step_reward_end_turn_full_energy_used():
    """Using all energy (leftover=0) gives the full bonus."""
    env = CombatEnv(communicator=MagicMock())
    post_state = _combat_energy(energy=3, hp=70)  # same hp, no damage exchanged
    reward = env._compute_step_reward(
        prev_hp=70, prev_monster_hp=42, prev_living=1,
        max_hp=80, state=post_state,
        action=ActionSpace.END_TURN_ACTION, prev_energy=0,
    )
    assert reward == pytest.approx(0.1)


def test_compute_step_reward_end_turn_partial_energy_used():
    """Using 2 of 3 energy gives a proportionally smaller bonus."""
    env = CombatEnv(communicator=MagicMock())
    post_state = _combat_energy(energy=3, hp=70)
    reward = env._compute_step_reward(
        prev_hp=70, prev_monster_hp=42, prev_living=1,
        max_hp=80, state=post_state,
        action=ActionSpace.END_TURN_ACTION, prev_energy=1,
    )
    assert reward == pytest.approx(0.1 * (1 - 1 / 3))


def test_compute_step_reward_end_turn_no_energy_used():
    """Wasting all energy gives no bonus."""
    env = CombatEnv(communicator=MagicMock())
    post_state = _combat_energy(energy=3, hp=70)
    reward = env._compute_step_reward(
        prev_hp=70, prev_monster_hp=42, prev_living=1,
        max_hp=80, state=post_state,
        action=ActionSpace.END_TURN_ACTION, prev_energy=3,
    )
    assert reward == pytest.approx(0.0)


def test_compute_step_reward_card_play_no_energy_bonus():
    """Energy bonus is NOT applied on card play, only on END_TURN."""
    env = CombatEnv(communicator=MagicMock())
    post_state = _combat_energy(energy=2, hp=70)
    # action=0 is a card-play action (not END_TURN_ACTION=60)
    reward = env._compute_step_reward(
        prev_hp=70, prev_monster_hp=42, prev_living=1,
        max_hp=80, state=post_state,
        action=0, prev_energy=1,
    )
    assert reward == pytest.approx(0.0)


def test_energy_efficiency_bonus_configurable():
    """energy_efficiency_bonus constructor param overrides the default weight."""
    env = CombatEnv(communicator=MagicMock(), energy_efficiency_bonus=0.2)
    post_state = _combat_energy(energy=3, hp=70)
    reward = env._compute_step_reward(
        prev_hp=70, prev_monster_hp=42, prev_living=1,
        max_hp=80, state=post_state,
        action=ActionSpace.END_TURN_ACTION, prev_energy=0,
    )
    assert reward == pytest.approx(0.2)


def test_step_end_turn_with_zero_energy_applies_bonus():
    """Integration: step() captures pre-command energy and applies bonus."""
    comm = MagicMock()
    comm.receive_state.return_value = _combat_energy(energy=3, hp=70)  # post-turn energy reset
    env = CombatEnv(communicator=comm)
    env._current_state = _combat_energy(energy=0, hp=70)  # agent spent all energy

    _, reward, done, _, _ = env.step(ActionSpace.END_TURN_ACTION)

    assert done is False
    assert reward == pytest.approx(0.1)  # no damage delta + full energy bonus


def test_combat_env_wraps_simple_agent_with_stuck_detector():
    from src.agent import StuckDetectorAgent
    env = CombatEnv(communicator=MagicMock())
    assert isinstance(env.simple_agent, StuckDetectorAgent)
