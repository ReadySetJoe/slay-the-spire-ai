# tests/test_game_state.py
import json
from src.game_state import GameState

SAMPLE_COMBAT_STATE = json.dumps({
    "available_commands": ["PLAY", "END", "POTION", "STATE", "KEY", "CLICK", "WAIT"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "NONE",
        "seed": 12345,
        "floor": 3,
        "ascension_level": 0,
        "class": "IRONCLAD",
        "current_hp": 70,
        "max_hp": 80,
        "gold": 99,
        "deck": [
            {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK"},
            {"id": "Defend_R", "name": "Defend", "cost": 1, "type": "SKILL"},
        ],
        "relics": [{"id": "Burning Blood", "name": "Burning Blood"}],
        "potions": [{"id": "Potion Slot", "name": "Potion Slot"}],
        "combat_state": {
            "hand": [
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK", "is_playable": True, "has_target": True, "uuid": "a1"},
                {"id": "Defend_R", "name": "Defend", "cost": 1, "type": "SKILL", "is_playable": True, "has_target": False, "uuid": "a2"},
                {"id": "Bash", "name": "Bash", "cost": 2, "type": "ATTACK", "is_playable": True, "has_target": True, "uuid": "a3"},
            ],
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
            "monsters": [
                {"name": "Jaw Worm", "current_hp": 42, "max_hp": 42, "block": 0, "intent": "ATTACK", "is_gone": False},
            ],
            "player": {
                "current_hp": 70,
                "max_hp": 80,
                "block": 0,
                "energy": 3,
                "powers": [],
            },
            "turn": 1,
        },
        "map": [],
        "act": 1,
    }
})

SAMPLE_REWARD_SCREEN = json.dumps({
    "available_commands": ["CHOOSE", "PROCEED", "STATE", "KEY", "CLICK", "WAIT"],
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
        "seed": 12345,
        "floor": 3,
        "ascension_level": 0,
        "class": "IRONCLAD",
        "current_hp": 70,
        "max_hp": 80,
        "gold": 99,
        "deck": [],
        "relics": [],
        "potions": [],
        "combat_state": None,
        "map": [],
        "act": 1,
    }
})

SAMPLE_ERROR = json.dumps({
    "error": "Invalid command",
    "ready_for_command": True,
})


def test_parse_combat_state():
    state = GameState.from_json(SAMPLE_COMBAT_STATE)
    assert state.in_game is True
    assert state.ready_for_command is True
    assert state.screen_type == "NONE"
    assert state.current_hp == 70
    assert state.max_hp == 80
    assert state.floor == 3
    assert state.gold == 99
    assert len(state.hand) == 3
    assert state.hand[0]["name"] == "Strike"
    assert len(state.monsters) == 1
    assert state.monsters[0]["name"] == "Jaw Worm"
    assert state.energy == 3
    assert state.player_block == 0
    assert "PLAY" in state.available_commands
    assert "END" in state.available_commands


def test_parse_reward_screen():
    state = GameState.from_json(SAMPLE_REWARD_SCREEN)
    assert state.screen_type == "CARD_REWARD"
    assert state.combat_state is None
    assert state.hand == []
    assert state.monsters == []
    assert "CHOOSE" in state.available_commands


def test_parse_error():
    state = GameState.from_json(SAMPLE_ERROR)
    assert state.error == "Invalid command"
    assert state.ready_for_command is True
    assert state.in_game is False


def test_is_in_combat():
    combat = GameState.from_json(SAMPLE_COMBAT_STATE)
    reward = GameState.from_json(SAMPLE_REWARD_SCREEN)
    assert combat.is_in_combat is True
    assert reward.is_in_combat is False
