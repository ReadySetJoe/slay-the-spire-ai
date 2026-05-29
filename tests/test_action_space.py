import json
import numpy as np
from src.action_space import ActionSpace
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
                {"name": "Louse", "current_hp": 15, "max_hp": 15,
                 "block": 0, "intent": "ATTACK", "is_gone": False},
            ],
            "player": {"current_hp": 70, "max_hp": 80, "block": 0,
                        "energy": 3, "powers": []},
            "turn": 1,
        },
    }
})

NO_PLAY_STATE = json.dumps({
    "available_commands": ["END"],
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
                 "is_playable": False, "has_target": True, "uuid": "a1"},
            ],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [
                {"name": "Jaw Worm", "current_hp": 42, "max_hp": 42,
                 "block": 0, "intent": "ATTACK", "is_gone": False},
            ],
            "player": {"current_hp": 70, "max_hp": 80, "block": 0,
                        "energy": 0, "powers": []},
            "turn": 1,
        },
    }
})


def test_mask_shape():
    space = ActionSpace()
    state = GameState.from_json(COMBAT_STATE)
    mask = space.get_action_mask(state)
    assert mask.shape == (ActionSpace.TOTAL_ACTIONS,)
    assert mask.dtype == np.bool_


def test_end_turn_always_valid():
    space = ActionSpace()
    state = GameState.from_json(COMBAT_STATE)
    mask = space.get_action_mask(state)
    assert mask[ActionSpace.END_TURN_ACTION] == True


def test_targeted_card_needs_target():
    space = ActionSpace()
    state = GameState.from_json(COMBAT_STATE)
    mask = space.get_action_mask(state)
    # Strike (slot 0) has_target=True, 2 living monsters (targets 0, 1)
    assert mask[0] == False   # no-target version invalid for targeted card
    assert mask[10] == True   # slot 0 target 0
    assert mask[11] == True   # slot 0 target 1
    assert mask[12] == False  # slot 0 target 2 (no monster)


def test_untargeted_card_no_target():
    space = ActionSpace()
    state = GameState.from_json(COMBAT_STATE)
    mask = space.get_action_mask(state)
    # Defend (slot 1) has_target=False
    assert mask[1] == True    # no-target version valid
    assert mask[15] == False  # slot 1 target 0 invalid (not targeted)


def test_unplayable_card_masked():
    space = ActionSpace()
    state = GameState.from_json(NO_PLAY_STATE)
    mask = space.get_action_mask(state)
    # Strike not playable
    assert mask[0] == False
    assert mask[10] == False
    # End turn should be valid
    assert mask[ActionSpace.END_TURN_ACTION] == True


def test_action_to_command_end_turn():
    space = ActionSpace()
    state = GameState.from_json(COMBAT_STATE)
    cmd = space.action_to_command(ActionSpace.END_TURN_ACTION, state)
    assert cmd == "END"


def test_action_to_command_play_targeted():
    space = ActionSpace()
    state = GameState.from_json(COMBAT_STATE)
    cmd = space.action_to_command(10, state)  # slot 0, target 0
    assert cmd == "PLAY 1 0"


def test_action_to_command_play_untargeted():
    space = ActionSpace()
    state = GameState.from_json(COMBAT_STATE)
    cmd = space.action_to_command(1, state)  # slot 1, no target
    assert cmd == "PLAY 2"
