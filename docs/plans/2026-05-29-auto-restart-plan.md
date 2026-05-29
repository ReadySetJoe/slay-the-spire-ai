# Auto-Restart Loop with Stats Tracking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bot automatically restarts Ironclad runs after game over, logging results to JSONL for overnight unattended play.

**Architecture:** RunTracker collects stats during a run and writes a JSONL line on game over. The game loop handles GAME_OVER screen and not-in-game state to auto-start new runs via the START command.

**Tech Stack:** Python 3.11+, pytest

---

### Task 1: Run Tracker

**Files:**
- Create: `src/run_tracker.py`
- Create: `tests/test_run_tracker.py`

**Step 1: Write failing tests**

```python
# tests/test_run_tracker.py
import json
import os
import tempfile
from src.run_tracker import RunTracker
from src.game_state import GameState

SAMPLE_GAME_OVER_WIN = json.dumps({
    "available_commands": ["PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "GAME_OVER",
        "screen_state": {"victory": True},
        "seed": 12345,
        "floor": 55,
        "ascension_level": 0,
        "class": "IRONCLAD",
        "current_hp": 32,
        "max_hp": 80,
        "gold": 150,
        "deck": [{"id": "Strike_R"}, {"id": "Defend_R"}, {"id": "Offering"}],
        "relics": [{"id": "Burning Blood"}, {"id": "Vajra"}],
        "potions": [],
        "map": [],
        "act": 3,
        "combat_state": None,
    }
})

SAMPLE_GAME_OVER_LOSS = json.dumps({
    "available_commands": ["PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "GAME_OVER",
        "screen_state": {"victory": False},
        "seed": 99999,
        "floor": 12,
        "ascension_level": 0,
        "class": "IRONCLAD",
        "current_hp": 0,
        "max_hp": 75,
        "gold": 50,
        "deck": [{"id": "Strike_R"}, {"id": "Defend_R"}],
        "relics": [{"id": "Burning Blood"}],
        "potions": [],
        "map": [],
        "act": 1,
        "combat_state": None,
    }
})


def test_record_run_win():
    tracker = RunTracker()
    state = GameState.from_json(SAMPLE_GAME_OVER_WIN)
    record = tracker.record_run(state)
    assert record["result"] == "win"
    assert record["seed"] == 12345
    assert record["floor_reached"] == 55
    assert record["deck_size"] == 3
    assert record["relic_count"] == 2
    assert record["run_number"] == 1


def test_record_run_loss():
    tracker = RunTracker()
    state = GameState.from_json(SAMPLE_GAME_OVER_LOSS)
    record = tracker.record_run(state)
    assert record["result"] == "loss"
    assert record["floor_reached"] == 12
    assert record["current_hp"] == 0


def test_run_number_increments():
    tracker = RunTracker()
    state_win = GameState.from_json(SAMPLE_GAME_OVER_WIN)
    state_loss = GameState.from_json(SAMPLE_GAME_OVER_LOSS)
    r1 = tracker.record_run(state_win)
    r2 = tracker.record_run(state_loss)
    assert r1["run_number"] == 1
    assert r2["run_number"] == 2


def test_save_to_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_log.jsonl")
        tracker = RunTracker(log_path=path)
        state = GameState.from_json(SAMPLE_GAME_OVER_WIN)
        tracker.record_run(state)
        tracker.record_run(GameState.from_json(SAMPLE_GAME_OVER_LOSS))

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["result"] == "win"
        second = json.loads(lines[1])
        assert second["result"] == "loss"


def test_summary():
    tracker = RunTracker()
    tracker.record_run(GameState.from_json(SAMPLE_GAME_OVER_WIN))
    tracker.record_run(GameState.from_json(SAMPLE_GAME_OVER_LOSS))
    tracker.record_run(GameState.from_json(SAMPLE_GAME_OVER_LOSS))
    summary = tracker.summary()
    assert summary["total_runs"] == 3
    assert summary["wins"] == 1
    assert summary["losses"] == 2
    assert summary["win_rate"] == 1 / 3
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_run_tracker.py -v`

**Step 3: Implement RunTracker**

```python
# src/run_tracker.py
import json
import logging
import os
from datetime import datetime, timezone

from src.game_state import GameState

logger = logging.getLogger(__name__)


class RunTracker:
    def __init__(self, log_path: str = "data/run_log.jsonl"):
        self.log_path = log_path
        self.run_number = 0
        self.runs: list[dict] = []

    def record_run(self, state: GameState) -> dict:
        self.run_number += 1

        victory = False
        if state.screen_state:
            victory = state.screen_state.get("victory", False)

        record = {
            "run_number": self.run_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": "win" if victory else "loss",
            "seed": state.seed,
            "floor_reached": state.floor,
            "ascension_level": state.ascension_level,
            "player_class": state.player_class,
            "current_hp": state.current_hp,
            "max_hp": state.max_hp,
            "gold": state.gold,
            "deck_size": len(state.deck),
            "relic_count": len(state.relics),
            "act": state.act,
        }

        self.runs.append(record)
        self._write_record(record)

        logger.info(
            "Run #%d complete: %s | Floor %d | HP %d/%d | Deck %d | Relics %d",
            record["run_number"], record["result"], record["floor_reached"],
            record["current_hp"], record["max_hp"],
            record["deck_size"], record["relic_count"],
        )

        return record

    def _write_record(self, record: dict):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def summary(self) -> dict:
        wins = sum(1 for r in self.runs if r["result"] == "win")
        losses = sum(1 for r in self.runs if r["result"] == "loss")
        total = len(self.runs)
        return {
            "total_runs": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total > 0 else 0,
            "avg_floor": sum(r["floor_reached"] for r in self.runs) / total if total > 0 else 0,
        }
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_run_tracker.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/run_tracker.py tests/test_run_tracker.py
git commit -m "feat: run tracker with JSONL logging"
```

---

### Task 2: Game Over Handling + Auto-Start

The game loop needs to:
1. Detect GAME_OVER screen → record stats → PROCEED
2. When not in game (main menu) → send `START IRONCLAD 0`

**Files:**
- Modify: `src/game_loop.py`
- Modify: `tests/test_game_loop.py`

**Step 1: Write failing tests**

Add to `tests/test_game_loop.py`:

```python
from src.run_tracker import RunTracker
import tempfile
import os

def make_game_over(victory=False):
    data = {
        "available_commands": ["PROCEED"],
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "screen_type": "GAME_OVER",
            "screen_state": {"victory": victory},
            "seed": 1, "floor": 10, "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": 0, "max_hp": 80, "gold": 50,
            "deck": [{"id": "Strike_R"}], "relics": [{"id": "Burning Blood"}],
            "potions": [], "map": [], "act": 1,
            "combat_state": None,
        },
    }
    return json.dumps(data)


def make_menu_state():
    return json.dumps({
        "available_commands": ["START"],
        "ready_for_command": True,
        "in_game": False,
    })


def test_game_loop_handles_game_over():
    """Game over should record stats and PROCEED."""
    inp = io.StringIO(make_game_over() + "\n")
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    agent = SimpleAgent()
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = RunTracker(log_path=os.path.join(tmpdir, "log.jsonl"))
        loop = GameLoop(comm, agent, run_tracker=tracker)
        loop.step()
        output_lines = out.getvalue().strip().split("\n")
        assert output_lines[-1] == "PROCEED"
        assert tracker.run_number == 1


def test_game_loop_auto_starts_new_run():
    """When not in game, should send START command."""
    inp = io.StringIO(make_menu_state() + "\n")
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    agent = SimpleAgent()
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = RunTracker(log_path=os.path.join(tmpdir, "log.jsonl"))
        loop = GameLoop(comm, agent, run_tracker=tracker)
        loop.step()
        output_lines = out.getvalue().strip().split("\n")
        assert output_lines[-1] == "START IRONCLAD 0"
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_game_loop.py -v`

**Step 3: Update GameLoop**

```python
# src/game_loop.py
import logging

from src.communicator import Communicator
from src.agent import Agent
from src.run_tracker import RunTracker

logger = logging.getLogger(__name__)


class GameLoop:
    def __init__(self, communicator: Communicator, agent: Agent,
                 run_tracker: RunTracker | None = None):
        self.communicator = communicator
        self.agent = agent
        self.run_tracker = run_tracker or RunTracker()
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

        if not state.in_game:
            if "START" in state.available_commands:
                logger.info("Starting new run...")
                self.communicator.send_command("START IRONCLAD 0")
            return True

        if state.screen_type == "GAME_OVER":
            self.run_tracker.record_run(state)
            summary = self.run_tracker.summary()
            logger.info(
                "Stats: %d runs | %d wins | %d losses | %.1f%% win rate | avg floor %.1f",
                summary["total_runs"], summary["wins"], summary["losses"],
                summary["win_rate"] * 100, summary["avg_floor"],
            )
            self.communicator.send_command("PROCEED")
            return True

        action = self.agent.act(state)
        logger.info("Floor %d | HP %d/%d | Screen: %s | Action: %s",
                    state.floor, state.current_hp, state.max_hp, state.screen_type, action)
        self.communicator.send_command(action)
        return True

    def run(self):
        """Run the game loop until input is exhausted."""
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

            if not state.in_game:
                if "START" in state.available_commands:
                    logger.info("Starting new run...")
                    self.communicator.send_command("START IRONCLAD 0")
                continue

            if state.screen_type == "GAME_OVER":
                self.run_tracker.record_run(state)
                summary = self.run_tracker.summary()
                logger.info(
                    "Stats: %d runs | %d wins | %d losses | %.1f%% win rate | avg floor %.1f",
                    summary["total_runs"], summary["wins"], summary["losses"],
                    summary["win_rate"] * 100, summary["avg_floor"],
                )
                self.communicator.send_command("PROCEED")
                continue

            logger.debug("Screen: %s | Commands: %s", state.screen_type, state.available_commands)
            action = self.agent.act(state)
            logger.info("Floor %d | HP %d/%d | Screen: %s | Action: %s",
                        state.floor, state.current_hp, state.max_hp, state.screen_type, action)
            self.communicator.send_command(action)
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/Scripts/activate && python -m pytest -v`
Expected: All tests PASS (existing tests still work because run_tracker is optional)

**Step 5: Commit**

```bash
git add src/game_loop.py tests/test_game_loop.py
git commit -m "feat: game over handling and auto-start new runs"
```

---

### Task 3: Update main.py + Add data/ to gitignore

**Files:**
- Modify: `main.py`
- Modify: `.gitignore`

**Step 1: Update main.py**

```python
# main.py
import logging

from src.communicator import Communicator
from src.agent import SimpleAgent
from src.game_loop import GameLoop
from src.run_tracker import RunTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("game.log"), logging.StreamHandler()],
)


def main():
    communicator = Communicator()
    agent = SimpleAgent()
    tracker = RunTracker(log_path="data/run_log.jsonl")
    loop = GameLoop(communicator, agent, run_tracker=tracker)
    loop.run()


if __name__ == "__main__":
    main()
```

**Step 2: Add data/ to .gitignore**

Append to `.gitignore`:
```
data/
game.log
```

**Step 3: Create data directory**

```bash
mkdir -p data
```

**Step 4: Run full test suite**

Run: `source .venv/Scripts/activate && python -m pytest -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add main.py .gitignore
git commit -m "feat: wire up run tracker in main, gitignore data/"
```
