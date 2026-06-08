import json
import pytest
from unittest.mock import MagicMock
from src.v2.run_env import RunEnv
from src.v2.run_encoder import RunEncoder
from src.v2.run_action_space import RunActionSpace
from src.game_state import GameState
from tests.v2.helpers import make_state, make_game_over, make_card_reward, make_shop, make_rest


@pytest.fixture
def env():
    return RunEnv(communicator=MagicMock())


def test_observation_space_shape(env):
    assert env.observation_space.shape == (RunEncoder.OBS_SIZE,)


def test_action_space_size(env):
    assert env.action_space.n == RunActionSpace.TOTAL_ACTIONS


def test_action_masks_all_ones_before_reset(env):
    import numpy as np
    assert env.action_masks().all()


def test_reset_sends_ready_on_first_call(env):
    env.communicator.receive_state.return_value = make_state()
    env.reset()
    env.communicator.send_ready.assert_called_once()


def test_reset_does_not_send_ready_on_second_call(env):
    env.communicator.receive_state.return_value = make_state()
    env.reset()
    env.reset()
    env.communicator.send_ready.assert_called_once()


def test_reset_returns_obs_of_correct_shape(env):
    env.communicator.receive_state.return_value = make_state()
    obs, info = env.reset()
    assert obs.shape == (RunEncoder.OBS_SIZE,)
    assert isinstance(info, dict)


def test_reset_skips_not_ready_states(env):
    not_ready = GameState.from_json(json.dumps({
        "available_commands": [],
        "ready_for_command": False,
        "in_game": True,
        "game_state": {
            "screen_type": "NONE", "seed": 1, "floor": 1,
            "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": 70, "max_hp": 80, "gold": 99, "act": 1,
            "deck": [], "relics": [], "potions": [], "map": [],
            "combat_state": None,
        }
    }))
    env.communicator.receive_state.side_effect = [not_ready, make_state()]
    obs, _ = env.reset()
    assert env.communicator.receive_state.call_count == 2


def test_reset_sends_start_when_not_in_game(env):
    main_menu = GameState.from_json(json.dumps({
        "available_commands": ["START"],
        "ready_for_command": True,
        "in_game": False,
        "game_state": None,
    }))
    env.communicator.receive_state.side_effect = [main_menu, make_state()]
    env.reset()
    env.communicator.send_command.assert_any_call("START IRONCLAD 0")


def test_step_combat_continues(env):
    before = make_state(hp=70, max_hp=80, energy=3,
                        monsters=[{"name": "Worm", "current_hp": 42, "max_hp": 42,
                                   "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []}])
    after  = make_state(hp=65, max_hp=80, energy=2,
                        monsters=[{"name": "Worm", "current_hp": 36, "max_hp": 42,
                                   "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []}])
    env.communicator.receive_state.return_value = after
    env._current_state = before

    obs, reward, done, truncated, info = env.step(10)  # PLAY 1 0

    assert done is False
    assert truncated is False
    assert obs.shape == (RunEncoder.OBS_SIZE,)
    # damage_dealt = 6/80, damage_taken = 5/80
    assert reward == pytest.approx(6 / 80 - 5 / 80)


def test_step_end_with_energy_waste_penalty(env):
    before = make_state(hp=70, max_hp=80, energy=3)
    after  = make_state(hp=70, max_hp=80, energy=0)
    env.communicator.receive_state.return_value = after
    env._current_state = before

    _, reward, done, _, _ = env.step(60)  # END

    # energy_waste = 3/3 → penalty = -0.3 * 1.0
    assert reward == pytest.approx(-0.3)


def test_step_sends_correct_command(env):
    env.communicator.receive_state.return_value = make_state()
    env._current_state = make_state(
        hand=[{"id": "Strike_R", "cost": 1, "type": "ATTACK",
               "is_playable": True, "has_target": True}]
    )
    env.step(10)  # PLAY 1 0
    env.communicator.send_command.assert_called_with("PLAY 1 0")


def test_step_debuff_tracking_reset_on_new_turn(env):
    env._debuff_applied_this_turn = True
    env._current_turn = 1
    before = make_state(turn=2)
    after  = make_state(turn=2)
    env.communicator.receive_state.return_value = after
    env._current_state = before

    env.step(60)  # END

    assert env._debuff_applied_this_turn is False


def test_step_debuff_flag_set_when_debuff_card_played(env):
    before = make_state(
        hand=[{"id": "Bash", "cost": 2, "type": "ATTACK",
               "is_playable": True, "has_target": True}],
        monsters=[{"name": "Worm", "current_hp": 42, "max_hp": 42,
                   "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []}],
    )
    after = make_state(
        monsters=[{"name": "Worm", "current_hp": 34, "max_hp": 42,
                   "block": 0, "intent": "ATTACK", "is_gone": False,
                   "powers": [{"id": "Vulnerable", "amount": 2}]}],
    )
    env.communicator.receive_state.return_value = after
    env._current_state = before
    env._debuff_applied_this_turn = False

    env.step(10)  # PLAY 1 0 (Bash)

    assert env._debuff_applied_this_turn is True


def test_step_card_reward_pick_returns_shaped_reward(env):
    before = make_card_reward(
        cards=[{"id": "Inflame", "name": "Inflame", "type": "POWER"}]
    )
    after = make_state(screen_type="MAP", available_commands=["CHOOSE"],
                       screen_state={"next_nodes": [{"symbol": "M"}]}, combat=False)
    env.communicator.receive_state.return_value = after
    env._current_state = before

    _, reward, done, _, _ = env.step(91)  # CHOOSE 0 → pick Inflame

    # Inflame is A-tier (0.8), synergy=0
    assert reward == pytest.approx(0.8 * 0.05)
    assert done is False


def test_step_card_reward_proceed_zero_reward(env):
    before = make_card_reward()
    after  = make_state(screen_type="MAP", available_commands=["CHOOSE"],
                        screen_state={"next_nodes": [{"symbol": "M"}]}, combat=False)
    env.communicator.receive_state.return_value = after
    env._current_state = before

    _, reward, _, _, _ = env.step(99)  # PROCEED

    assert reward == pytest.approx(0.0)


def test_step_rest_heal_reward(env):
    before = make_rest(hp=50, max_hp=80)
    after  = make_state(hp=74, max_hp=80, combat=False,
                        screen_type="MAP", available_commands=["CHOOSE"],
                        screen_state={"next_nodes": [{"symbol": "M"}]})
    env.communicator.receive_state.return_value = after
    env._current_state = before

    _, reward, _, _, _ = env.step(101)  # CHOOSE rest

    # Heal = min(80*0.3=24, 80-50=30) = 24 → reward = 24/80 * 0.2
    assert reward == pytest.approx(24 / 80 * 0.2)


def test_step_game_over_terminal_reward(env):
    before = make_state()
    game_over = make_game_over(floor=10)
    env.communicator.receive_state.return_value = game_over
    env._current_state = before

    _, reward, done, _, info = env.step(60)  # END

    assert done is True
    assert reward == pytest.approx((10 / 55) * 3.0 - 1.0)
    assert "episode" in info
    assert info["episode"]["floor"] == 10


def test_step_game_over_sends_proceed(env):
    env.communicator.receive_state.return_value = make_game_over(floor=5)
    env._current_state = make_state()

    env.step(60)

    env.communicator.send_command.assert_called_with("PROCEED")


def test_step_game_over_records_run(env):
    env.communicator.receive_state.return_value = make_game_over(floor=5)
    env._current_state = make_state()

    env.step(60)

    assert env.run_tracker.summary()["total_runs"] == 1
