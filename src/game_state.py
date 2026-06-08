# src/game_state.py
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GameState:
    raw: dict
    available_commands: list = field(default_factory=list)
    ready_for_command: bool = False
    in_game: bool = False
    error: Optional[str] = None

    # Game info
    screen_type: str = "NONE"
    screen_state: Optional[dict] = None
    seed: Optional[int] = None
    floor: int = 0
    ascension_level: int = 0
    player_class: str = ""
    current_hp: int = 0
    max_hp: int = 0
    gold: int = 0
    act: int = 1
    deck: list = field(default_factory=list)
    relics: list = field(default_factory=list)
    potions: list = field(default_factory=list)
    map_data: list = field(default_factory=list)

    # Combat info (empty when not in combat)
    combat_state: Optional[dict] = None
    hand: list = field(default_factory=list)
    draw_pile: list = field(default_factory=list)
    discard_pile: list = field(default_factory=list)
    exhaust_pile: list = field(default_factory=list)
    monsters: list = field(default_factory=list)
    energy: int = 0
    player_block: int = 0
    turn: int = 0

    @classmethod
    def from_json(cls, json_str: str) -> "GameState":
        data = json.loads(json_str)
        state = cls(raw=data)

        state.available_commands = [c.upper() for c in data.get("available_commands", [])]
        state.ready_for_command = data.get("ready_for_command", False)
        state.in_game = data.get("in_game", False)
        state.error = data.get("error")

        gs = data.get("game_state")
        if gs:
            state.screen_type = gs.get("screen_type", "NONE")
            state.screen_state = gs.get("screen_state")
            state.seed = gs.get("seed")
            state.floor = gs.get("floor", 0)
            state.ascension_level = gs.get("ascension_level", 0)
            state.player_class = gs.get("class", "")
            state.current_hp = gs.get("current_hp", 0)
            state.max_hp = gs.get("max_hp", 0)
            state.gold = gs.get("gold", 0)
            state.act = gs.get("act", 1)
            state.deck = gs.get("deck", [])
            state.relics = gs.get("relics", [])
            state.potions = gs.get("potions", [])
            state.map_data = gs.get("map", [])

            cs = gs.get("combat_state")
            if cs:
                state.combat_state = cs
                state.hand = cs.get("hand", [])
                state.draw_pile = cs.get("draw_pile", [])
                state.discard_pile = cs.get("discard_pile", [])
                state.exhaust_pile = cs.get("exhaust_pile", [])
                state.monsters = cs.get("monsters", [])
                state.turn = cs.get("turn", 0)
                player = cs.get("player", {})
                state.energy = player.get("energy", 0)
                state.player_block = player.get("block", 0)

        return state

    # Screens that appear mid-combat (e.g. Armaments → GRID, Dual Wield → HAND_SELECT).
    # combat_state is still populated; we must not treat these as combat-over.
    _IN_COMBAT_SCREENS = frozenset({"NONE", "GRID", "HAND_SELECT"})

    @property
    def is_in_combat(self) -> bool:
        return self.combat_state is not None and self.screen_type in self._IN_COMBAT_SCREENS
