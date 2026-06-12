import time
import pytest
from unittest.mock import MagicMock, patch
from src.v3.run_env import V3RunEnv, HungEpisodeError
from src.v3.run_encoder import V3RunEncoder
from src.v3.card_scorer import CardScorer
from src.v2.run_action_space import RunActionSpace
from tests.v2.helpers import make_state, make_game_over, make_card_reward


def make_env(timeout=20.0, scorer=None):
    comm = MagicMock()
    scorer = scorer or CardScorer(path="data/card_scores_test.json")
    env = V3RunEnv(communicator=comm, card_scorer=scorer, timeout_seconds=timeout)
    return env, comm


# --- observation space ---

def test_obs_space_shape():
    env, _ = make_env()
    assert env.observation_space.shape == (V3RunEncoder.OBS_SIZE,)


def test_action_space_size():
    env, _ = make_env()
    assert env.action_space.n == RunActionSpace.TOTAL_ACTIONS


# --- reset ---

def test_reset_returns_correct_obs_shape():
    env, comm = make_env()
    comm.receive_state.return_value = make_state()
    obs, info = env.reset()
    assert obs.shape == (V3RunEncoder.OBS_SIZE,)


def test_reset_clears_turn_state():
    env, comm = make_env()
    comm.receive_state.return_value = make_state()
    env._turn_state["actions_taken"] = 5
    env.reset()
    assert env._turn_state["actions_taken"] == 0


# --- turn state tracking ---

def test_turn_state_updates_actions_taken_on_card_play():
    env, comm = make_env()
    comm.receive_state.return_value = make_state()
    env.reset()
    comm.receive_state.return_value = make_state()
    env.step(0)  # PLAY 1 (no-target slot 0)
    assert env._turn_state["actions_taken"] == 1


def test_turn_state_resets_on_end_action():
    env, comm = make_env()
    comm.receive_state.return_value = make_state()
    env.reset()
    env._turn_state["actions_taken"] = 3
    comm.receive_state.return_value = make_state()
    env.step(60)  # END
    assert env._turn_state["actions_taken"] == 0


def test_turn_state_resets_on_new_combat():
    """Transitioning from non-combat → combat resets turn state."""
    env, comm = make_env()
    comm.receive_state.return_value = make_state()
    env.reset()
    env._current_state = make_card_reward()  # simulate non-combat
    env._turn_state["attacks_played"] = 3
    comm.receive_state.return_value = make_state()  # back to combat
    env.step(99)  # PROCEED
    assert env._turn_state["attacks_played"] == 0


# --- hung watchdog ---

def test_hung_episode_returns_truncated():
    env, comm = make_env(timeout=0.1)

    comm.receive_state.return_value = make_state()
    env.reset()

    def _hang():
        time.sleep(10)
        return make_state()

    comm.receive_state.side_effect = _hang
    obs, reward, terminated, truncated, info = env.step(60)

    assert truncated is True
    assert terminated is False
    assert reward == 0.0
    assert info.get("hung") is True


def test_hung_episode_not_counted_as_death(tmp_path):
    from src.run_tracker import RunTracker
    tracker = RunTracker(log_path=str(tmp_path / "runs.jsonl"))
    env, comm = make_env(timeout=0.1)
    env.run_tracker = tracker

    comm.receive_state.return_value = make_state()
    env.reset()

    def _hang():
        time.sleep(10)
    comm.receive_state.side_effect = _hang

    env.step(60)
    assert tracker.hung_count == 1
    assert tracker.run_number == 0  # no normal run recorded


# --- CardScorer integration ---

def test_card_scorer_updated_on_combat_end(tmp_path):
    scorer = CardScorer(path=str(tmp_path / "scores.json"))
    env, comm = make_env(scorer=scorer)
    comm.receive_state.return_value = make_state()
    env.reset()

    # Simulate a card play that adds to _combat_cards_played
    env._combat_cards_played = ["Bash"]
    env._combat_total_damage = 20.0
    env._combat_total_enemy_max_hp = 42.0

    # Simulate combat→non-combat transition (triggers _on_combat_end)
    env._on_combat_end()

    assert scorer.score("Bash") != pytest.approx(0.5)


# --- normal game over ---

def test_normal_game_over_returns_terminated():
    env, comm = make_env()
    comm.receive_state.return_value = make_state()
    env.reset()
    comm.receive_state.return_value = make_game_over(floor=10)
    obs, reward, terminated, truncated, info = env.step(60)
    assert terminated is True
    assert truncated is False
