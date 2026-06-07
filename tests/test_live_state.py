import json
import os
import tempfile
import pytest
from src.live_state import LiveStateWriter
from src.game_state import GameState

COMBAT_STATE = json.dumps({
    "available_commands": ["PLAY", "END"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "NONE",
        "seed": 1, "floor": 5, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 47, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": {
            "hand": [
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
                 "is_playable": True, "has_target": True, "uuid": "a1"},
                {"id": "Bash", "name": "Bash", "cost": 2, "type": "ATTACK",
                 "is_playable": True, "has_target": True, "uuid": "a2"},
            ],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [
                {"name": "Jaw Worm", "current_hp": 34, "max_hp": 44,
                 "block": 0, "intent": "ATTACK", "is_gone": False},
                {"name": "Cultist", "current_hp": 10, "max_hp": 50,
                 "block": 0, "intent": "BUFF", "is_gone": False},
            ],
            "player": {"current_hp": 47, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "turn": 1,
        },
    },
})

NON_COMBAT_STATE = json.dumps({
    "available_commands": ["CHOOSE", "PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "MAP",
        "seed": 1, "floor": 5, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 60, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": None,
    },
})


def test_write_creates_live_section():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        state = GameState.from_json(COMBAT_STATE)
        writer.write(state, "END")
        with open(path) as f:
            data = json.load(f)
        assert "live" in data
        live = data["live"]
        assert live["screen_type"] == "NONE"
        assert live["current_hp"] == 47
        assert live["max_hp"] == 80
        assert live["last_action"] == "END"
        assert len(live["monsters"]) == 2
        assert live["monsters"][0]["name"] == "Jaw Worm"
        assert live["monsters"][0]["current_hp"] == 34


def test_write_enriches_play_with_target():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        state = GameState.from_json(COMBAT_STATE)
        writer.write(state, "PLAY 2 0")
        with open(path) as f:
            data = json.load(f)
        assert data["live"]["last_action"] == "PLAY 2 0 (Bash → Jaw Worm)"


def test_write_enriches_play_without_target():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        state = GameState.from_json(COMBAT_STATE)
        writer.write(state, "PLAY 1")
        with open(path) as f:
            data = json.load(f)
        assert data["live"]["last_action"] == "PLAY 1 (Strike)"


def test_write_preserves_stats_section():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        existing = {"stats": {"run_number": 5, "wins": 2, "losses": 3,
                              "win_rate": 0.4, "avg_floor": 12.0}}
        with open(path, "w") as f:
            json.dump(existing, f)
        state = GameState.from_json(COMBAT_STATE)
        writer.write(state, "END")
        with open(path) as f:
            data = json.load(f)
        assert data["stats"]["run_number"] == 5


def test_write_empty_monsters_when_not_in_combat():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        state = GameState.from_json(NON_COMBAT_STATE)
        writer.write(state, "CHOOSE 0")
        with open(path) as f:
            data = json.load(f)
        assert data["live"]["monsters"] == []
        assert data["live"]["screen_type"] == "MAP"


def test_write_run_summary_updates_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        summary = {
            "total_runs": 10,
            "wins": 3,
            "losses": 7,
            "win_rate": 0.3,
            "avg_floor": 18.5,
        }
        writer.write_run_summary(summary)
        with open(path) as f:
            data = json.load(f)
        assert data["stats"]["run_number"] == 10
        assert data["stats"]["wins"] == 3
        assert data["stats"]["losses"] == 7
        assert data["stats"]["win_rate"] == 0.3
        assert data["stats"]["avg_floor"] == 18.5


def test_write_run_summary_preserves_live_section():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        existing = {"live": {"screen_type": "COMBAT", "current_hp": 40,
                             "max_hp": 80, "last_action": "END",
                             "monsters": [], "updated_at": "2026-01-01T00:00:00Z"}}
        with open(path, "w") as f:
            json.dump(existing, f)
        writer.write_run_summary({"total_runs": 1, "wins": 1, "losses": 0,
                                   "win_rate": 1.0, "avg_floor": 55.0})
        with open(path) as f:
            data = json.load(f)
        assert data["live"]["screen_type"] == "COMBAT"


def test_write_training_step_creates_training_section():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        episodes = [
            {"ep": 1, "reward": -2.0, "steps": 30},
            {"ep": 2, "reward": 1.5, "steps": 45},
        ]
        writer.write_training_step(episodes, total_episodes=2, total_timesteps=75)
        with open(path) as f:
            data = json.load(f)
        assert "training" in data
        assert data["training"]["total_episodes"] == 2
        assert data["training"]["total_timesteps"] == 75
        assert data["training"]["episodes"] == episodes


def test_write_training_step_preserves_other_sections():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        existing = {"stats": {"run_number": 5, "wins": 2, "losses": 3,
                              "win_rate": 0.4, "avg_floor": 12.0}}
        with open(path, "w") as f:
            json.dump(existing, f)
        writer.write_training_step([], total_episodes=0, total_timesteps=0)
        with open(path) as f:
            data = json.load(f)
        assert data["stats"]["run_number"] == 5
        assert "training" in data
