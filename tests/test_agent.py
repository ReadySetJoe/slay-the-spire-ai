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


COMBAT_REWARD_WITH_REWARDS = json.dumps({
    "available_commands": ["CHOOSE", "PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "COMBAT_REWARD",
        "screen_state": {
            "rewards": [
                {"reward_type": "GOLD", "gold": 25},
                {"reward_type": "POTION", "potion": {"id": "Fire Potion", "name": "Fire Potion"}},
                {"reward_type": "CARD"},
            ]
        },
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 70, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [],
        "potions": [
            {"id": "Potion Slot", "name": "Potion Slot"},
            {"id": "Potion Slot", "name": "Potion Slot"},
        ],
        "map": [], "act": 1,
        "combat_state": None,
    }
})

COMBAT_REWARD_POTIONS_FULL = json.dumps({
    "available_commands": ["CHOOSE", "PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "COMBAT_REWARD",
        "screen_state": {
            "rewards": [
                {"reward_type": "GOLD", "gold": 25},
                {"reward_type": "POTION", "potion": {"id": "Fire Potion", "name": "Fire Potion"}},
            ]
        },
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 70, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [],
        "potions": [
            {"id": "Fire Potion", "name": "Fire Potion"},
            {"id": "Block Potion", "name": "Block Potion"},
        ],
        "map": [], "act": 1,
        "combat_state": None,
    }
})

COMBAT_REWARD_NO_REWARDS = json.dumps({
    "available_commands": ["PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "COMBAT_REWARD",
        "screen_state": {"rewards": []},
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 70, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": None,
    }
})


def test_combat_reward_chooses_first_reward():
    state = GameState.from_json(COMBAT_REWARD_WITH_REWARDS)
    agent = SimpleAgent()
    action = agent.act(state)
    assert action == "CHOOSE 0"


def test_combat_reward_no_rewards_proceeds():
    state = GameState.from_json(COMBAT_REWARD_NO_REWARDS)
    agent = SimpleAgent()
    action = agent.act(state)
    assert action == "PROCEED"


def test_combat_reward_skips_potion_when_full():
    state = GameState.from_json(COMBAT_REWARD_POTIONS_FULL)
    agent = SimpleAgent()
    action = agent.act(state)
    # Should choose gold (index 0), not potion (index 1) since potions are full
    assert action == "CHOOSE 0"


CARD_REWARD_WITH_TIERS = json.dumps({
    "available_commands": ["CHOOSE", "PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "CARD_REWARD",
        "screen_state": {
            "cards": [
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK"},
                {"id": "Offering", "name": "Offering", "cost": 0, "type": "SKILL"},
                {"id": "Inflame", "name": "Inflame", "cost": 1, "type": "POWER"},
            ]
        },
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 70, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": None,
    }
})


def test_card_reward_picks_best_tier():
    state = GameState.from_json(CARD_REWARD_WITH_TIERS)
    agent = SimpleAgent()
    action = agent.act(state)
    # Offering is S tier (index 1), should pick it
    assert action == "CHOOSE 1"


COMBAT_LOW_HP_WITH_POTION = json.dumps({
    "available_commands": ["PLAY", "END", "POTION"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "NONE",
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 20, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [],
        "potions": [
            {"id": "Fruit Juice", "name": "Fruit Juice", "can_use": True, "can_discard": True, "requires_target": False},
        ],
        "map": [], "act": 1,
        "combat_state": {
            "hand": [
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
                 "is_playable": True, "has_target": True, "uuid": "a1"},
            ],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [
                {"name": "Jaw Worm", "current_hp": 42, "max_hp": 42, "block": 0, "intent": "ATTACK", "is_gone": False},
            ],
            "player": {"current_hp": 20, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "turn": 1,
        },
    }
})

COMBAT_FULL_HP_WITH_POTION = json.dumps({
    "available_commands": ["PLAY", "END", "POTION"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "NONE",
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 80, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [],
        "potions": [
            {"id": "Fruit Juice", "name": "Fruit Juice", "can_use": True, "can_discard": True, "requires_target": False},
        ],
        "map": [], "act": 1,
        "combat_state": {
            "hand": [
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
                 "is_playable": True, "has_target": True, "uuid": "a1"},
            ],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [
                {"name": "Jaw Worm", "current_hp": 42, "max_hp": 42, "block": 0, "intent": "ATTACK", "is_gone": False},
            ],
            "player": {"current_hp": 80, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "turn": 1,
        },
    }
})

COMBAT_ELITE_WITH_ATTACK_POTION = json.dumps({
    "available_commands": ["PLAY", "END", "POTION"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "NONE",
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 70, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [],
        "potions": [
            {"id": "Fire Potion", "name": "Fire Potion", "can_use": True, "can_discard": True, "requires_target": True},
        ],
        "map": [], "act": 1,
        "combat_state": {
            "hand": [
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
                 "is_playable": True, "has_target": True, "uuid": "a1"},
            ],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [
                {"name": "Gremlin Nob", "current_hp": 106, "max_hp": 106, "block": 0, "intent": "ATTACK", "is_gone": False},
            ],
            "player": {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "turn": 1,
        },
    }
})


def test_use_potion_when_low_hp():
    state = GameState.from_json(COMBAT_LOW_HP_WITH_POTION)
    agent = SimpleAgent()
    action = agent.act(state)
    assert action == "POTION Use 0"


def test_no_potion_when_full_hp():
    state = GameState.from_json(COMBAT_FULL_HP_WITH_POTION)
    agent = SimpleAgent()
    action = agent.act(state)
    assert action.startswith("PLAY")


def test_use_attack_potion_on_elite():
    state = GameState.from_json(COMBAT_ELITE_WITH_ATTACK_POTION)
    agent = SimpleAgent()
    action = agent.act(state)
    assert action == "POTION Use 0 0"


# ── GRID / HAND_SELECT fixtures ───────────────────────────────────────────────

_STRIKE = {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK", "uuid": "u1"}
_DEFEND = {"id": "Defend_R", "name": "Defend", "cost": 1, "type": "SKILL",  "uuid": "u2"}
_BASH   = {"id": "Bash",     "name": "Bash",   "cost": 2, "type": "ATTACK", "uuid": "u3"}


def _grid(cards, selected_cards=None, commands=("CHOOSE", "CANCEL")):
    return GameState.from_json(json.dumps({
        "available_commands": list(commands),
        "ready_for_command": True, "in_game": True,
        "game_state": {
            "screen_type": "GRID",
            "screen_state": {"cards": cards, "selected_cards": selected_cards or [], "num_cards": 2},
            "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": 70, "max_hp": 80, "gold": 99,
            "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
            "combat_state": None,
        }
    }))


def _hand_select(cards, selected=None, commands=("CHOOSE", "CANCEL")):
    return GameState.from_json(json.dumps({
        "available_commands": list(commands),
        "ready_for_command": True, "in_game": True,
        "game_state": {
            "screen_type": "HAND_SELECT",
            "screen_state": {"cards": cards, "selected": selected or [], "num_cards": 2},
            "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": 70, "max_hp": 80, "gold": 99,
            "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
            "combat_state": None,
        }
    }))


def test_grid_excludes_already_selected_card():
    """Agent must not re-pick a card already in selected_cards (avoids deselect oscillation)."""
    state = _grid(cards=[_STRIKE, _DEFEND, _BASH], selected_cards=[_STRIKE])
    agent = SimpleAgent()
    action = agent.act(state)
    # Strike (uuid u1) is in selected_cards — must not choose index 0
    assert action != "CHOOSE 0"


def test_hand_select_excludes_already_selected_card():
    """Agent skips cards in the 'selected' array on HAND_SELECT screens."""
    state = _hand_select(cards=[_STRIKE, _DEFEND], selected=[_STRIKE])
    agent = SimpleAgent()
    action = agent.act(state)
    # Strike (uuid u1) already selected — only Defend (index 1) is available
    assert action == "CHOOSE 1"


def test_grid_confirm_takes_priority():
    """CONFIRM is returned as soon as it appears in available_commands."""
    state = _grid(
        cards=[_STRIKE], selected_cards=[_STRIKE],
        commands=("CONFIRM", "CANCEL"),
    )
    agent = SimpleAgent()
    assert agent.act(state) == "CONFIRM"


def test_grid_cancel_when_all_selected_no_confirm():
    """Returns CANCEL (not CHOOSE 0) when all cards selected but CONFIRM not yet available."""
    state = _grid(cards=[_STRIKE], selected_cards=[_STRIKE])
    agent = SimpleAgent()
    assert agent.act(state) == "CANCEL"
