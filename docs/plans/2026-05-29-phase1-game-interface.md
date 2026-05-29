# Phase 1 & 2: Game Interface + Rule-Based Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a working game loop that communicates with Slay the Spire via CommunicationMod and plays full runs with a simple rule-based Ironclad agent.

**Architecture:** A Python process launched by CommunicationMod via stdin/stdout. The process sends "ready", receives JSON game states, decides actions via an Agent interface, and sends commands back. The agent is swappable — starting with a rule-based agent, later replaced by RL.

**Tech Stack:** Python 3.11+, pytest, CommunicationMod (stdin/stdout JSON protocol)

---

## CommunicationMod Protocol Reference

- CommunicationMod launches our Python process via a configured command
- Our process sends `ready\n` on stdout to complete the handshake
- The mod sends JSON game state lines on our stdin
- We respond with command strings on stdout (e.g., `PLAY 1 0`, `END`, `CHOOSE 0`, `PROCEED`)
- Card indices are 1-indexed; monster target indices are 0-indexed
- `available_commands` in the JSON tells us what commands are valid
- `ready_for_command` tells us when the mod is waiting for input
- Errors come back as `{"error": "message", "ready_for_command": true}`

---

### Task 1: Project Scaffolding

**Files:**
- Create: `src/__init__.py`
- Create: `src/game_interface.py`
- Create: `tests/__init__.py`
- Create: `tests/test_game_interface.py`
- Create: `requirements.txt`
- Create: `main.py`
- Create: `.gitignore`

**Step 1: Create project structure and .gitignore**

```
# .gitignore
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
build/
.venv/
venv/
```

**Step 2: Create requirements.txt**

```
pytest>=7.0
```

**Step 3: Create empty __init__.py files**

`src/__init__.py` and `tests/__init__.py` — empty files.

**Step 4: Create virtual environment and install deps**

Run:
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

**Step 5: Commit**

```bash
git add .gitignore requirements.txt src/__init__.py tests/__init__.py
git commit -m "feat: project scaffolding with pytest"
```

---

### Task 2: Game State Parser

**Files:**
- Create: `src/game_state.py`
- Create: `tests/test_game_state.py`

**Step 1: Write failing tests for game state parsing**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_game_state.py -v`
Expected: FAIL with import errors

**Step 3: Implement GameState**

```python
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

        state.available_commands = data.get("available_commands", [])
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

    @property
    def is_in_combat(self) -> bool:
        return self.combat_state is not None and self.screen_type == "NONE"
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_game_state.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/game_state.py tests/test_game_state.py
git commit -m "feat: game state parser for CommunicationMod JSON"
```

---

### Task 3: Communication Layer

**Files:**
- Create: `src/communicator.py`
- Create: `tests/test_communicator.py`

**Step 1: Write failing tests**

```python
# tests/test_communicator.py
import io
import json
from src.communicator import Communicator

SAMPLE_STATE = {
    "available_commands": ["PLAY", "END"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "NONE",
        "seed": 1,
        "floor": 1,
        "ascension_level": 0,
        "class": "IRONCLAD",
        "current_hp": 80,
        "max_hp": 80,
        "gold": 99,
        "deck": [],
        "relics": [],
        "potions": [],
        "combat_state": {
            "hand": [],
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
            "monsters": [],
            "player": {"current_hp": 80, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "turn": 1,
        },
        "map": [],
        "act": 1,
    },
}


def test_send_ready():
    out = io.StringIO()
    comm = Communicator(input_stream=io.StringIO(), output_stream=out)
    comm.send_ready()
    assert out.getvalue() == "ready\n"


def test_send_command():
    out = io.StringIO()
    comm = Communicator(input_stream=io.StringIO(), output_stream=out)
    comm.send_command("PLAY 1 0")
    assert out.getvalue() == "PLAY 1 0\n"


def test_receive_state():
    json_line = json.dumps(SAMPLE_STATE) + "\n"
    inp = io.StringIO(json_line)
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    state = comm.receive_state()
    assert state.in_game is True
    assert state.ready_for_command is True


def test_receive_state_eof():
    inp = io.StringIO("")
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    state = comm.receive_state()
    assert state is None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_communicator.py -v`
Expected: FAIL with import errors

**Step 3: Implement Communicator**

```python
# src/communicator.py
import sys
from typing import Optional, TextIO

from src.game_state import GameState


class Communicator:
    def __init__(
        self,
        input_stream: Optional[TextIO] = None,
        output_stream: Optional[TextIO] = None,
    ):
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout

    def send_ready(self):
        self.output_stream.write("ready\n")
        self.output_stream.flush()

    def send_command(self, command: str):
        self.output_stream.write(command + "\n")
        self.output_stream.flush()

    def receive_state(self) -> Optional[GameState]:
        line = self.input_stream.readline()
        if not line:
            return None
        return GameState.from_json(line.strip())
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_communicator.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/communicator.py tests/test_communicator.py
git commit -m "feat: stdin/stdout communicator for CommunicationMod"
```

---

### Task 4: Agent Interface + Rule-Based Agent

**Files:**
- Create: `src/agent.py`
- Create: `tests/test_agent.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent.py -v`
Expected: FAIL with import errors

**Step 3: Implement SimpleAgent**

```python
# src/agent.py
import logging
from abc import ABC, abstractmethod

from src.game_state import GameState

logger = logging.getLogger(__name__)


class Agent(ABC):
    @abstractmethod
    def act(self, state: GameState) -> str:
        """Given a game state, return a command string."""
        pass


class SimpleAgent(Agent):
    """Rule-based agent that plays Ironclad with simple heuristics."""

    def act(self, state: GameState) -> str:
        if state.is_in_combat:
            return self._handle_combat(state)

        if state.screen_type == "CARD_REWARD":
            return self._handle_card_reward(state)

        if state.screen_type == "REST":
            return self._handle_rest(state)

        if state.screen_type == "MAP":
            return self._handle_map(state)

        if state.screen_type == "COMBAT_REWARD":
            return "PROCEED"

        if state.screen_type == "BOSS_REWARD":
            return "CHOOSE 0"

        if "PROCEED" in state.available_commands:
            return "PROCEED"

        if "CHOOSE" in state.available_commands:
            return "CHOOSE 0"

        return "STATE"

    def _handle_combat(self, state: GameState) -> str:
        playable = [
            (i, card) for i, card in enumerate(state.hand)
            if card.get("is_playable", False)
        ]

        if not playable:
            return "END"

        # Find first living monster for targeting
        target = 0
        for i, m in enumerate(state.monsters):
            if not m.get("is_gone", False):
                target = i
                break

        # Play first playable card (1-indexed)
        idx, card = playable[0]
        card_index = idx + 1  # CommunicationMod uses 1-indexed cards
        if card.get("has_target", False):
            return f"PLAY {card_index} {target}"
        return f"PLAY {card_index}"

    def _handle_card_reward(self, state: GameState) -> str:
        # Always pick first offered card for now
        if "CHOOSE" in state.available_commands:
            return "CHOOSE 0"
        return "PROCEED"

    def _handle_rest(self, state: GameState) -> str:
        # Rest if below 60% HP, otherwise smith
        hp_ratio = state.current_hp / max(state.max_hp, 1)
        if hp_ratio < 0.6:
            return "CHOOSE rest"
        return "CHOOSE smith"

    def _handle_map(self, state: GameState) -> str:
        return "CHOOSE 0"
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/agent.py tests/test_agent.py
git commit -m "feat: agent interface and rule-based SimpleAgent"
```

---

### Task 5: Main Game Loop

**Files:**
- Create: `src/game_loop.py`
- Create: `tests/test_game_loop.py`
- Create: `main.py`

**Step 1: Write failing tests**

```python
# tests/test_game_loop.py
import io
import json
from src.game_loop import GameLoop
from src.agent import SimpleAgent
from src.communicator import Communicator


def make_state(screen_type="NONE", commands=None, in_game=True, combat=True):
    """Helper to build a JSON game state string."""
    if commands is None:
        commands = ["PLAY", "END"] if combat else ["CHOOSE", "PROCEED"]
    data = {
        "available_commands": commands,
        "ready_for_command": True,
        "in_game": in_game,
        "game_state": {
            "screen_type": screen_type,
            "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": 80, "max_hp": 80, "gold": 99,
            "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
            "combat_state": {
                "hand": [
                    {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
                     "is_playable": True, "has_target": True, "uuid": "a1"},
                ],
                "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
                "monsters": [
                    {"name": "Jaw Worm", "current_hp": 42, "max_hp": 42,
                     "block": 0, "intent": "ATTACK", "is_gone": False},
                ],
                "player": {"current_hp": 80, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
                "turn": 1,
            } if combat else None,
        },
    }
    return json.dumps(data)


def test_game_loop_sends_ready():
    inp = io.StringIO(make_state() + "\n")
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    agent = SimpleAgent()
    loop = GameLoop(comm, agent)
    loop.step()  # sends ready + processes one state
    output = out.getvalue()
    assert output.startswith("ready\n")


def test_game_loop_processes_state_and_sends_action():
    inp = io.StringIO(make_state() + "\n")
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    agent = SimpleAgent()
    loop = GameLoop(comm, agent)
    loop.step()
    output_lines = out.getvalue().strip().split("\n")
    assert len(output_lines) == 2  # "ready" + one action
    assert output_lines[0] == "ready"
    assert output_lines[1].startswith("PLAY")


def test_game_loop_stops_on_eof():
    inp = io.StringIO("")
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    agent = SimpleAgent()
    loop = GameLoop(comm, agent)
    loop.step()
    output = out.getvalue()
    assert output == "ready\n"  # Only ready, no action (EOF)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_game_loop.py -v`
Expected: FAIL with import errors

**Step 3: Implement GameLoop**

```python
# src/game_loop.py
import logging

from src.communicator import Communicator
from src.agent import Agent

logger = logging.getLogger(__name__)


class GameLoop:
    def __init__(self, communicator: Communicator, agent: Agent):
        self.communicator = communicator
        self.agent = agent
        self._ready_sent = False

    def step(self):
        """Send ready (if needed) and process one state."""
        if not self._ready_sent:
            self.communicator.send_ready()
            self._ready_sent = True

        state = self.communicator.receive_state()
        if state is None:
            logger.info("No more input, stopping.")
            return False

        if state.error:
            logger.warning("Received error: %s", state.error)
            return True

        if not state.ready_for_command:
            return True

        action = self.agent.act(state)
        logger.info("Floor %d | HP %d/%d | Action: %s", state.floor, state.current_hp, state.max_hp, action)
        self.communicator.send_command(action)
        return True

    def run(self):
        """Run the game loop until the game ends or input is exhausted."""
        self.communicator.send_ready()
        self._ready_sent = True
        while True:
            state = self.communicator.receive_state()
            if state is None:
                logger.info("Connection closed.")
                break

            if state.error:
                logger.warning("Error: %s", state.error)
                continue

            if not state.ready_for_command:
                continue

            action = self.agent.act(state)
            logger.info("Floor %d | HP %d/%d | Action: %s", state.floor, state.current_hp, state.max_hp, action)
            self.communicator.send_command(action)
```

**Step 4: Implement main.py**

```python
# main.py
import logging

from src.communicator import Communicator
from src.agent import SimpleAgent
from src.game_loop import GameLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("game.log"), logging.StreamHandler()],
)


def main():
    communicator = Communicator()
    agent = SimpleAgent()
    loop = GameLoop(communicator, agent)
    loop.run()


if __name__ == "__main__":
    main()
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_game_loop.py -v`
Expected: All 3 tests PASS

**Step 6: Run full test suite**

Run: `python -m pytest -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/game_loop.py tests/test_game_loop.py main.py
git commit -m "feat: game loop and main entry point"
```

---

### Task 6: Integration Test with CommunicationMod

This task is manual — requires the game running.

**Step 1: Configure CommunicationMod**

In your Slay the Spire preferences directory, edit the CommunicationMod config to point to this script:

```
command=python C:\\Users\\Joe\\Documents\\code\\slay-the-spire-ai\\main.py
```

**Step 2: Launch Slay the Spire with mods enabled**

Make sure ModTheSpire, BaseMod, and CommunicationMod are installed and active.

**Step 3: Start a run**

The bot should connect, receive game states, and start playing. Watch `game.log` for output.

**Step 4: Observe and note any issues**

Common issues to watch for:
- Unhandled screen types (add logging for unknown screens)
- Card/event choices that need special handling
- Timeouts or communication errors

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration fixes from first live test"
```
