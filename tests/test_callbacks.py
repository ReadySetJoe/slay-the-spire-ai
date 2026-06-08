import pytest
from unittest.mock import MagicMock
from src.callbacks import EpisodeLoggerCallback


def test_callback_calls_write_training_step_after_episode():
    writer = MagicMock()
    cb = EpisodeLoggerCallback(summary_freq=100, live_state_writer=writer)
    cb.num_timesteps = 50
    cb._log_episode({"r": 2.5, "l": 30, "hp": 40, "max_hp": 80, "floor": 3})
    writer.write_training_step.assert_called_once()
    kwargs = writer.write_training_step.call_args.kwargs
    assert kwargs["total_episodes"] == 1
    assert kwargs["total_timesteps"] == 50
    assert kwargs["episodes"] == [{"ep": 1, "reward": 2.5, "steps": 30}]


def test_callback_without_writer_does_not_raise():
    cb = EpisodeLoggerCallback(summary_freq=100)
    cb.num_timesteps = 0
    cb._log_episode({"r": 1.0, "l": 10, "hp": 80, "max_hp": 80, "floor": 1})


def test_callback_rolling_window_caps_at_500():
    writer = MagicMock()
    cb = EpisodeLoggerCallback(summary_freq=100, live_state_writer=writer)
    cb.num_timesteps = 0
    for i in range(505):
        cb._log_episode({"r": float(i), "l": 10, "hp": 80, "max_hp": 80, "floor": 1})
    kwargs = writer.write_training_step.call_args.kwargs
    assert len(kwargs["episodes"]) == 500
    assert kwargs["total_episodes"] == 505
    assert kwargs["episodes"][-1]["reward"] == 504.0


def test_callback_tracks_total_episodes_across_calls():
    writer = MagicMock()
    cb = EpisodeLoggerCallback(summary_freq=100, live_state_writer=writer)
    cb.num_timesteps = 0
    for _ in range(3):
        cb._log_episode({"r": 1.0, "l": 10, "hp": 80, "max_hp": 80, "floor": 1})
    kwargs = writer.write_training_step.call_args.kwargs
    assert kwargs["total_episodes"] == 3
