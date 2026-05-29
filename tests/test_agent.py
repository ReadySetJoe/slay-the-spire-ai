# tests/test_agent.py
import json
from src.game_state import GameState
from src.agent import SimpleAgent

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
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK", "is_playable": True, "has_target": True, "uuid": "a1"},
                {"id": "Defend_R", "name": "Defend", "cost": 1, "type": "SKILL", "is_playable": True, "has_target": False, "uuid": "a2"},
                {"id": "Bash", "name": "Bash", "cost": 2, "type": "ATTACK", "is_playable": True, "has_target": True, "uuid": "a3"},
            ],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [
                {"name": "Jaw Worm", "current_hp": 42, "max_hp": 42, "block": 0, "intent": "ATTACK", "is_gone": False},
            ],
            "player": {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "turn": 1,
        },
    }
})

NO_ENERGY_STATE = json.dumps({
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
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK", "is_playable": False, "has_target": True, "uuid": "a1"},
            ],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [
                {"name": "Jaw Worm", "current_hp": 42, "max_hp": 42, "block": 0, "intent": "ATTACK", "is_gone": False},
            ],
            "player": {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 0, "powers": []},
            "turn": 1,
        },
    }
})

CARD_REWARD_STATE = json.dumps({
    "available_commands": ["CHOOSE", "PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "CARD_REWARD",
        "screen_state": {
            "cards": [
                {"id": "Cleave", "name": "Cleave", "cost": 1, "type": "ATTACK"},
                {"id": "Shrug_It_Off", "name": "Shrug It Off", "cost": 1, "type": "SKILL"},
                {"id": "Inflame", "name": "Inflame", "cost": 1, "type": "POWER"},
            ]
        },
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 70, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": None,
    }
})

REST_SITE_STATE = json.dumps({
    "available_commands": ["CHOOSE", "PROCEED", "STATE"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "REST",
        "screen_state": {
            "options": ["rest", "smith"],
        },
        "seed": 1, "floor": 6, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 50, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": None,
    }
})

MAP_STATE = json.dumps({
    "available_commands": ["CHOOSE", "STATE"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "MAP",
        "screen_state": {
            "next_nodes": [
                {"x": 1, "y": 1, "symbol": "M"},
                {"x": 3, "y": 1, "symbol": "?"},
            ],
        },
        "seed": 1, "floor": 0, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 80, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": None,
    }
})


def test_combat_plays_card():
    state = GameState.from_json(COMBAT_STATE)
    agent = SimpleAgent()
    action = agent.act(state)
    # Should play a card (PLAY command), not END turn with energy remaining
    assert action.startswith("PLAY ")


def test_combat_ends_turn_no_energy():
    state = GameState.from_json(NO_ENERGY_STATE)
    agent = SimpleAgent()
    action = agent.act(state)
    assert action == "END"


def test_card_reward_chooses_or_skips():
    state = GameState.from_json(CARD_REWARD_STATE)
    agent = SimpleAgent()
    action = agent.act(state)
    # Should either CHOOSE a card or PROCEED (skip)
    assert action.startswith("CHOOSE") or action == "PROCEED"


def test_rest_site():
    state = GameState.from_json(REST_SITE_STATE)
    agent = SimpleAgent()
    action = agent.act(state)
    assert action.startswith("CHOOSE")


def test_map_chooses_node():
    state = GameState.from_json(MAP_STATE)
    agent = SimpleAgent()
    action = agent.act(state)
    assert action.startswith("CHOOSE")
