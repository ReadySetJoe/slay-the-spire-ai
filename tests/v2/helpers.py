import json
from src.game_state import GameState


def make_state(
    screen_type="NONE",
    available_commands=None,
    hp=70, max_hp=80, floor=1, gold=99, energy=3, block=0, act=1,
    hand=None, monsters=None, draw_pile=None, discard_pile=None,
    potions=None, relics=None, deck=None,
    screen_state=None,
    combat=True,
    powers=None,
    turn=1,
) -> GameState:
    if available_commands is None:
        available_commands = ["PLAY", "END"] if combat and screen_type == "NONE" else ["PROCEED"]
    if hand is None:
        hand = [{"id": "Strike_R", "name": "Strike", "cost": 1,
                 "type": "ATTACK", "is_playable": True, "has_target": True, "uuid": "a1"}]
    if monsters is None:
        monsters = [{"name": "Jaw Worm", "current_hp": 42, "max_hp": 42,
                     "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []}]
    if draw_pile is None:
        draw_pile = []
    if discard_pile is None:
        discard_pile = []
    if potions is None:
        potions = []
    if relics is None:
        relics = []
    if deck is None:
        deck = []
    if powers is None:
        powers = []

    combat_state = None
    if combat and screen_type == "NONE":
        combat_state = {
            "hand": hand,
            "draw_pile": draw_pile,
            "discard_pile": discard_pile,
            "exhaust_pile": [],
            "monsters": monsters,
            "player": {"current_hp": hp, "max_hp": max_hp, "block": block,
                       "energy": energy, "powers": powers},
            "turn": turn,
        }

    return GameState.from_json(json.dumps({
        "available_commands": available_commands,
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "screen_type": screen_type,
            "screen_state": screen_state,
            "seed": 1, "floor": floor, "ascension_level": 0,
            "class": "IRONCLAD",
            "current_hp": hp, "max_hp": max_hp, "gold": gold, "act": act,
            "deck": deck, "relics": relics, "potions": potions, "map": [],
            "combat_state": combat_state,
        }
    }))


def make_game_over(floor=3, hp=0, max_hp=80) -> GameState:
    return GameState.from_json(json.dumps({
        "available_commands": ["PROCEED"],
        "ready_for_command": True, "in_game": True,
        "game_state": {
            "screen_type": "GAME_OVER",
            "seed": 1, "floor": floor, "ascension_level": 0,
            "class": "IRONCLAD",
            "current_hp": hp, "max_hp": max_hp, "gold": 99, "act": 1,
            "deck": [], "relics": [], "potions": [], "map": [],
            "combat_state": None,
        }
    }))


def make_card_reward(cards=None, hp=70, max_hp=80, floor=1) -> GameState:
    if cards is None:
        cards = [
            {"id": "Shrug It Off", "name": "Shrug It Off", "type": "SKILL"},
            {"id": "Inflame", "name": "Inflame", "type": "POWER"},
            {"id": "Strike_R", "name": "Strike", "type": "ATTACK"},
        ]
    return make_state(
        screen_type="CARD_REWARD",
        available_commands=["CHOOSE", "PROCEED"],
        hp=hp, max_hp=max_hp, floor=floor,
        screen_state={"cards": cards},
        combat=False,
    )


def make_shop(cards=None, relics=None, gold=200, purge_available=True, hp=70, max_hp=80) -> GameState:
    if cards is None:
        cards = [{"id": "Inflame", "price": 75, "is_in_stock": True, "type": "POWER"}]
    if relics is None:
        relics = [{"id": "Anchor", "price": 150, "is_in_stock": True}]
    cmds = ["CHOOSE", "PROCEED"]
    if purge_available:
        cmds.append("PURGE")
    return make_state(
        screen_type="SHOP_SCREEN",
        available_commands=cmds,
        gold=gold, hp=hp, max_hp=max_hp,
        screen_state={"cards": cards, "relics": relics,
                      "purge_cost": 75, "purge_available": purge_available},
        combat=False,
    )


def make_rest(hp=50, max_hp=80) -> GameState:
    return make_state(
        screen_type="REST",
        available_commands=["CHOOSE"],
        hp=hp, max_hp=max_hp,
        combat=False,
    )


def make_map(nodes=None, hp=70, max_hp=80, floor=1) -> GameState:
    if nodes is None:
        nodes = [{"symbol": "M"}, {"symbol": "E"}, {"symbol": "?"}]
    return make_state(
        screen_type="MAP",
        available_commands=["CHOOSE"],
        hp=hp, max_hp=max_hp, floor=floor,
        screen_state={"next_nodes": nodes},
        combat=False,
    )
