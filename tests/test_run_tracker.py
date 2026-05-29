import json
import os
import tempfile
from src.run_tracker import RunTracker
from src.game_state import GameState

SAMPLE_GAME_OVER_WIN = json.dumps({
    "available_commands": ["PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "GAME_OVER",
        "screen_state": {"victory": True},
        "seed": 12345,
        "floor": 55,
        "ascension_level": 0,
        "class": "IRONCLAD",
        "current_hp": 32,
        "max_hp": 80,
        "gold": 150,
        "deck": [{"id": "Strike_R"}, {"id": "Defend_R"}, {"id": "Offering"}],
        "relics": [{"id": "Burning Blood"}, {"id": "Vajra"}],
        "potions": [],
        "map": [],
        "act": 3,
        "combat_state": None,
    }
})

SAMPLE_GAME_OVER_LOSS = json.dumps({
    "available_commands": ["PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "GAME_OVER",
        "screen_state": {"victory": False},
        "seed": 99999,
        "floor": 12,
        "ascension_level": 0,
        "class": "IRONCLAD",
        "current_hp": 0,
        "max_hp": 75,
        "gold": 50,
        "deck": [{"id": "Strike_R"}, {"id": "Defend_R"}],
        "relics": [{"id": "Burning Blood"}],
        "potions": [],
        "map": [],
        "act": 1,
        "combat_state": None,
    }
})


def test_record_run_win():
    tracker = RunTracker()
    state = GameState.from_json(SAMPLE_GAME_OVER_WIN)
    record = tracker.record_run(state)
    assert record["result"] == "win"
    assert record["seed"] == 12345
    assert record["floor_reached"] == 55
    assert record["deck_size"] == 3
    assert record["relic_count"] == 2
    assert record["run_number"] == 1


def test_record_run_loss():
    tracker = RunTracker()
    state = GameState.from_json(SAMPLE_GAME_OVER_LOSS)
    record = tracker.record_run(state)
    assert record["result"] == "loss"
    assert record["floor_reached"] == 12
    assert record["current_hp"] == 0


def test_run_number_increments():
    tracker = RunTracker()
    state_win = GameState.from_json(SAMPLE_GAME_OVER_WIN)
    state_loss = GameState.from_json(SAMPLE_GAME_OVER_LOSS)
    r1 = tracker.record_run(state_win)
    r2 = tracker.record_run(state_loss)
    assert r1["run_number"] == 1
    assert r2["run_number"] == 2


def test_save_to_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_log.jsonl")
        tracker = RunTracker(log_path=path)
        state = GameState.from_json(SAMPLE_GAME_OVER_WIN)
        tracker.record_run(state)
        tracker.record_run(GameState.from_json(SAMPLE_GAME_OVER_LOSS))

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["result"] == "win"
        second = json.loads(lines[1])
        assert second["result"] == "loss"


def test_summary():
    tracker = RunTracker()
    tracker.record_run(GameState.from_json(SAMPLE_GAME_OVER_WIN))
    tracker.record_run(GameState.from_json(SAMPLE_GAME_OVER_LOSS))
    tracker.record_run(GameState.from_json(SAMPLE_GAME_OVER_LOSS))
    summary = tracker.summary()
    assert summary["total_runs"] == 3
    assert summary["wins"] == 1
    assert summary["losses"] == 2
    assert summary["win_rate"] == 1 / 3
