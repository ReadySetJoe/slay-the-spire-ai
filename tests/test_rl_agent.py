import json
import tempfile
import os
from src.rl_agent import RLAgent
from src.game_state import GameState

COMBAT_STATE = json.dumps({
    "available_commands": ["PLAY", "END", "POTION"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "NONE",
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 70, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": {
            "hand": [
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
                 "is_playable": True, "has_target": True, "uuid": "a1"},
                {"id": "Defend_R", "name": "Defend", "cost": 1, "type": "SKILL",
                 "is_playable": True, "has_target": False, "uuid": "a2"},
            ],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [
                {"name": "Jaw Worm", "current_hp": 42, "max_hp": 42,
                 "block": 0, "intent": "ATTACK", "is_gone": False},
            ],
            "player": {"current_hp": 70, "max_hp": 80, "block": 0,
                        "energy": 3, "powers": []},
            "turn": 1,
        },
    }
})

MAP_STATE = json.dumps({
    "available_commands": ["CHOOSE", "STATE"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "MAP",
        "screen_state": {
            "next_nodes": [{"x": 1, "y": 1, "symbol": "M"}],
        },
        "seed": 1, "floor": 0, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 80, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": None,
    }
})


def test_rl_agent_returns_valid_combat_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = RLAgent(model_path=os.path.join(tmpdir, "model.zip"))
        state = GameState.from_json(COMBAT_STATE)
        action = agent.act(state)
        # Should return a PLAY or END command
        assert action.startswith("PLAY") or action == "END"


def test_rl_agent_delegates_non_combat():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = RLAgent(model_path=os.path.join(tmpdir, "model.zip"))
        state = GameState.from_json(MAP_STATE)
        action = agent.act(state)
        assert action.startswith("CHOOSE")


def test_rl_agent_tracks_combat_hp():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = RLAgent(model_path=os.path.join(tmpdir, "model.zip"))
        state = GameState.from_json(COMBAT_STATE)
        agent.act(state)
        assert agent.combat_start_hp == 70
