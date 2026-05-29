import json
import numpy as np
from src.state_encoder import StateEncoder
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
                {"id": "Bash", "name": "Bash", "cost": 2, "type": "ATTACK",
                 "is_playable": True, "has_target": True, "uuid": "a3"},
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

EMPTY_COMBAT = json.dumps({
    "available_commands": ["END"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "NONE",
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 50, "max_hp": 80, "gold": 0,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": {
            "hand": [],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [],
            "player": {"current_hp": 50, "max_hp": 80, "block": 0,
                        "energy": 0, "powers": []},
            "turn": 1,
        },
    }
})


def test_encode_returns_correct_shape():
    encoder = StateEncoder()
    state = GameState.from_json(COMBAT_STATE)
    obs = encoder.encode(state)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (StateEncoder.OBS_SIZE,)
    assert obs.dtype == np.float32


def test_encode_player_features():
    encoder = StateEncoder()
    state = GameState.from_json(COMBAT_STATE)
    obs = encoder.encode(state)
    # First 4 values: hp/max_hp, max_hp/max_hp, block/max_hp, energy/max_energy
    assert obs[0] == 70 / 80  # hp ratio
    assert obs[1] == 80 / 80  # max_hp ratio (always 1.0)
    assert obs[2] == 0 / 80   # block ratio
    assert obs[3] == 3 / 4    # energy ratio (3/4 base energy for Ironclad)


def test_encode_hand_cards():
    encoder = StateEncoder()
    state = GameState.from_json(COMBAT_STATE)
    obs = encoder.encode(state)
    # Card slot 0 starts at index 4, each card has 3 features
    card0_start = 4
    assert obs[card0_start] > 0      # cost (normalized)
    assert obs[card0_start + 1] > 0  # type encoded
    assert obs[card0_start + 2] == 1  # is_playable


def test_encode_empty_hand_is_zeros():
    encoder = StateEncoder()
    state = GameState.from_json(EMPTY_COMBAT)
    obs = encoder.encode(state)
    # All card slots should be zero
    for i in range(4, 4 + 10 * 3):
        assert obs[i] == 0.0


def test_encode_monsters():
    encoder = StateEncoder()
    state = GameState.from_json(COMBAT_STATE)
    obs = encoder.encode(state)
    # Monster slot 0 starts at index 4 + 30 = 34
    m0_start = 34
    assert obs[m0_start] == 42 / 42      # hp ratio
    assert obs[m0_start + 1] == 42 / 42  # max_hp ratio (always 1.0 for first)
    assert obs[m0_start + 2] == 0.0      # block ratio
    assert obs[m0_start + 3] > 0         # intent encoded


def test_encode_empty_monsters_is_zeros():
    encoder = StateEncoder()
    state = GameState.from_json(EMPTY_COMBAT)
    obs = encoder.encode(state)
    for i in range(34, 34 + 5 * 4):
        assert obs[i] == 0.0
