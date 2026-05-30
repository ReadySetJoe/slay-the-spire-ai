# Live Monitoring Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two OBS browser source pages (`/live` and `/stats`) served by a standalone Flask server that displays real-time Slay the Spire AI game state and run statistics.

**Architecture:** `LiveStateWriter` writes `data/live_state.json` after each agent action and each run end. A standalone `dashboard.py` Flask server reads that file and exposes it via `/api/state`; two HTML pages at `/live` and `/stats` poll that endpoint every 2.5 seconds and update the DOM in place.

**Tech Stack:** Python, Flask (new dep), pytest (existing)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/live_state.py` | Create | `LiveStateWriter` — writes `data/live_state.json` |
| `dashboard.py` | Create | Flask server — serves `/live`, `/stats`, `/api/state` |
| `tests/test_live_state.py` | Create | Tests for `LiveStateWriter` |
| `tests/test_dashboard.py` | Create | Tests for Flask routes |
| `src/game_loop.py` | Modify | Pass action + state to `LiveStateWriter.write()` after each `agent.act()` |
| `src/run_tracker.py` | Modify | Call `LiveStateWriter.write_run_summary()` in `record_run()` |
| `main.py` | Modify | Construct `LiveStateWriter`, pass to `GameLoop` and `RunTracker` |

---

## Task 1: Create `src/live_state.py`

**Files:**
- Create: `src/live_state.py`
- Create: `tests/test_live_state.py`

- [ ] **Step 1: Install Flask**

```bash
pip install flask
```

Expected: Flask installs without error.

- [ ] **Step 2: Write failing tests**

Create `tests/test_live_state.py`:

```python
import json
import os
import tempfile
import pytest
from src.live_state import LiveStateWriter
from src.game_state import GameState

COMBAT_STATE = json.dumps({
    "available_commands": ["PLAY", "END"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "NONE",
        "seed": 1, "floor": 5, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 47, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": {
            "hand": [
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
                 "is_playable": True, "has_target": True, "uuid": "a1"},
                {"id": "Bash", "name": "Bash", "cost": 2, "type": "ATTACK",
                 "is_playable": True, "has_target": True, "uuid": "a2"},
            ],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [
                {"name": "Jaw Worm", "current_hp": 34, "max_hp": 44,
                 "block": 0, "intent": "ATTACK", "is_gone": False},
                {"name": "Cultist", "current_hp": 10, "max_hp": 50,
                 "block": 0, "intent": "BUFF", "is_gone": False},
            ],
            "player": {"current_hp": 47, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "turn": 1,
        },
    },
})

NON_COMBAT_STATE = json.dumps({
    "available_commands": ["CHOOSE", "PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "MAP",
        "seed": 1, "floor": 5, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 60, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": None,
    },
})


def test_write_creates_live_section():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        state = GameState.from_json(COMBAT_STATE)
        writer.write(state, "END")
        with open(path) as f:
            data = json.load(f)
        assert "live" in data
        live = data["live"]
        assert live["screen_type"] == "NONE"
        assert live["current_hp"] == 47
        assert live["max_hp"] == 80
        assert live["last_action"] == "END"
        assert len(live["monsters"]) == 2
        assert live["monsters"][0]["name"] == "Jaw Worm"
        assert live["monsters"][0]["current_hp"] == 34


def test_write_enriches_play_with_target():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        state = GameState.from_json(COMBAT_STATE)
        writer.write(state, "PLAY 2 0")
        with open(path) as f:
            data = json.load(f)
        assert data["live"]["last_action"] == "PLAY 2 0 (Bash → Jaw Worm)"


def test_write_enriches_play_without_target():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        state = GameState.from_json(COMBAT_STATE)
        writer.write(state, "PLAY 1")
        with open(path) as f:
            data = json.load(f)
        assert data["live"]["last_action"] == "PLAY 1 (Strike)"


def test_write_preserves_stats_section():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        existing = {"stats": {"run_number": 5, "wins": 2, "losses": 3,
                              "win_rate": 0.4, "avg_floor": 12.0}}
        with open(path, "w") as f:
            json.dump(existing, f)
        state = GameState.from_json(COMBAT_STATE)
        writer.write(state, "END")
        with open(path) as f:
            data = json.load(f)
        assert data["stats"]["run_number"] == 5


def test_write_empty_monsters_when_not_in_combat():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        state = GameState.from_json(NON_COMBAT_STATE)
        writer.write(state, "CHOOSE 0")
        with open(path) as f:
            data = json.load(f)
        assert data["live"]["monsters"] == []
        assert data["live"]["screen_type"] == "MAP"


def test_write_run_summary_updates_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        summary = {
            "total_runs": 10,
            "wins": 3,
            "losses": 7,
            "win_rate": 0.3,
            "avg_floor": 18.5,
        }
        writer.write_run_summary(summary)
        with open(path) as f:
            data = json.load(f)
        assert data["stats"]["run_number"] == 10
        assert data["stats"]["wins"] == 3
        assert data["stats"]["losses"] == 7
        assert data["stats"]["win_rate"] == 0.3
        assert data["stats"]["avg_floor"] == 18.5


def test_write_run_summary_preserves_live_section():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        existing = {"live": {"screen_type": "COMBAT", "current_hp": 40,
                             "max_hp": 80, "last_action": "END",
                             "monsters": [], "updated_at": "2026-01-01T00:00:00Z"}}
        with open(path, "w") as f:
            json.dump(existing, f)
        writer.write_run_summary({"total_runs": 1, "wins": 1, "losses": 0,
                                   "win_rate": 1.0, "avg_floor": 55.0})
        with open(path) as f:
            data = json.load(f)
        assert data["live"]["screen_type"] == "COMBAT"
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
pytest tests/test_live_state.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.live_state'`

- [ ] **Step 4: Create `src/live_state.py`**

```python
import json
import logging
import os
import re
from datetime import datetime, timezone

from src.game_state import GameState

logger = logging.getLogger(__name__)


class LiveStateWriter:
    def __init__(self, path: str = "data/live_state.json"):
        self.path = path

    def write(self, state: GameState, action: str) -> None:
        live = {
            "screen_type": state.screen_type,
            "current_hp": state.current_hp,
            "max_hp": state.max_hp,
            "last_action": self._enrich_action(state, action),
            "monsters": [
                {
                    "name": m.get("name", ""),
                    "current_hp": m.get("current_hp", 0),
                    "max_hp": m.get("max_hp", 0),
                    "intent": m.get("intent", ""),
                }
                for m in state.monsters
                if not m.get("is_gone", False)
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._update("live", live)

    def write_run_summary(self, summary: dict) -> None:
        stats = {
            "run_number": summary.get("total_runs", 0),
            "wins": summary.get("wins", 0),
            "losses": summary.get("losses", 0),
            "win_rate": round(summary.get("win_rate", 0.0), 4),
            "avg_floor": round(summary.get("avg_floor", 0.0), 1),
        }
        self._update("stats", stats)

    def _enrich_action(self, state: GameState, action: str) -> str:
        m = re.match(r"^PLAY\s+(\d+)(?:\s+(\d+))?", action)
        if not m:
            return action
        card_idx = int(m.group(1)) - 1
        target_idx = int(m.group(2)) if m.group(2) is not None else None
        card_name = ""
        if 0 <= card_idx < len(state.hand):
            card_name = state.hand[card_idx].get("name") or state.hand[card_idx].get("id", "")
        if target_idx is not None and 0 <= target_idx < len(state.monsters):
            monster_name = state.monsters[target_idx].get("name", "Enemy")
            return f"{action} ({card_name} → {monster_name})"
        if card_name:
            return f"{action} ({card_name})"
        return action

    def _update(self, key: str, value: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        existing = {}
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        existing[key] = value
        try:
            with open(self.path, "w") as f:
                json.dump(existing, f, indent=2)
        except OSError as e:
            logger.error("Failed to write live state: %s", e)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/test_live_state.py -v
```

Expected: 8 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/live_state.py tests/test_live_state.py
git commit -m "feat: add LiveStateWriter for live dashboard state"
```

---

## Task 2: Wire `LiveStateWriter` into `GameLoop`

**Files:**
- Modify: `src/game_loop.py`
- Modify: `tests/test_game_loop.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_game_loop.py`:

```python
def test_game_loop_calls_live_state_writer():
    """GameLoop should call writer.write() with state and action after each agent act."""
    import tempfile, os, json
    from src.live_state import LiveStateWriter

    inp = io.StringIO(make_state() + "\n")
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    agent = SimpleAgent()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        loop = GameLoop(comm, agent, live_state_writer=writer)
        loop.step()
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "live" in data
        assert data["live"]["last_action"].startswith("PLAY")
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_game_loop.py::test_game_loop_calls_live_state_writer -v
```

Expected: `TypeError: GameLoop.__init__() got an unexpected keyword argument 'live_state_writer'`

- [ ] **Step 3: Modify `src/game_loop.py`**

Add `live_state_writer` parameter to `__init__` and call it in both `step()` and `run()` after each `agent.act()`:

```python
# src/game_loop.py
import logging

from src.communicator import Communicator
from src.agent import Agent
from src.run_tracker import RunTracker

logger = logging.getLogger(__name__)


class GameLoop:
    def __init__(self, communicator: Communicator, agent: Agent,
                 run_tracker: RunTracker | None = None,
                 live_state_writer=None):
        self.communicator = communicator
        self.agent = agent
        self.run_tracker = run_tracker or RunTracker()
        self.live_state_writer = live_state_writer
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
            if hasattr(self.agent, 'on_game_over'):
                self.agent.on_game_over(state)
            self.communicator.send_command("PROCEED")
            return True

        action = self.agent.act(state)
        if self.live_state_writer:
            self.live_state_writer.write(state, action)
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
                if hasattr(self.agent, 'on_game_over'):
                    self.agent.on_game_over(state)
                self.communicator.send_command("PROCEED")
                continue

            logger.debug("Screen: %s | Commands: %s", state.screen_type, state.available_commands)
            action = self.agent.act(state)
            if self.live_state_writer:
                self.live_state_writer.write(state, action)
            logger.info("Floor %d | HP %d/%d | Screen: %s | Action: %s",
                        state.floor, state.current_hp, state.max_hp, state.screen_type, action)
            self.communicator.send_command(action)
```

- [ ] **Step 4: Run all game_loop tests**

```bash
pytest tests/test_game_loop.py -v
```

Expected: all tests pass including the new one.

- [ ] **Step 5: Commit**

```bash
git add src/game_loop.py tests/test_game_loop.py
git commit -m "feat: wire LiveStateWriter into GameLoop"
```

---

## Task 3: Wire `LiveStateWriter` into `RunTracker`

**Files:**
- Modify: `src/run_tracker.py`
- Modify: `tests/test_run_tracker.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_run_tracker.py`:

```python
def test_run_tracker_calls_live_state_writer():
    """RunTracker should call writer.write_run_summary() after each run."""
    import tempfile, os, json
    from src.live_state import LiveStateWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "run_log.jsonl")
        live_path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=live_path)
        tracker = RunTracker(log_path=log_path, live_state_writer=writer)
        state = GameState.from_json(SAMPLE_GAME_OVER_WIN)
        tracker.record_run(state)
        assert os.path.exists(live_path)
        with open(live_path) as f:
            data = json.load(f)
        assert data["stats"]["run_number"] == 1
        assert data["stats"]["wins"] == 1
        assert data["stats"]["losses"] == 0
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_run_tracker.py::test_run_tracker_calls_live_state_writer -v
```

Expected: `TypeError: RunTracker.__init__() got an unexpected keyword argument 'live_state_writer'`

- [ ] **Step 3: Modify `src/run_tracker.py`**

```python
import json
import logging
import os
from datetime import datetime, timezone

from src.game_state import GameState

logger = logging.getLogger(__name__)


class RunTracker:
    def __init__(self, log_path: str = "data/run_log.jsonl", live_state_writer=None):
        self.log_path = log_path
        self.live_state_writer = live_state_writer
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
        self._update_graphs()
        if self.live_state_writer:
            self.live_state_writer.write_run_summary(self.summary())

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

    def _update_graphs(self):
        from src.grapher import generate_graphs
        data_dir = os.path.dirname(self.log_path) or "."
        generate_graphs(
            log_path=self.log_path,
            scores_path=os.path.join(data_dir, "card_scores.json"),
            output_dir=os.path.join(data_dir, "graphs"),
        )

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

- [ ] **Step 4: Run all run_tracker tests**

```bash
pytest tests/test_run_tracker.py -v
```

Expected: all tests pass including the new one.

- [ ] **Step 5: Commit**

```bash
git add src/run_tracker.py tests/test_run_tracker.py
git commit -m "feat: wire LiveStateWriter into RunTracker"
```

---

## Task 4: Wire up `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update `main.py`**

Add `LiveStateWriter` construction and pass it to both `GameLoop` and `RunTracker`. The full updated `main()` function:

```python
def main():
    use_rl = "--rl" in sys.argv

    from src.live_state import LiveStateWriter
    live_writer = LiveStateWriter(path="data/live_state.json")

    communicator = Communicator()
    tracker = RunTracker(log_path="data/run_log.jsonl", live_state_writer=live_writer)

    from src.card_scorer import CardScorer
    scorer = CardScorer(path="data/card_scores.json")

    if use_rl:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from src.combat_env import CombatEnv
        from src.callbacks import EpisodeLoggerCallback

        env = CombatEnv(communicator=communicator, run_tracker=tracker, scorer=scorer)
        model_path = "data/combat_model.zip"
        checkpoint_dir = "data/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        model = _load_model(model_path, checkpoint_dir, env)
        if model is None:
            model = MaskablePPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                verbose=1,
            )
            logger.info("Created new MaskablePPO model")
        callbacks = CallbackList([
            EpisodeLoggerCallback(summary_freq=100),
            CheckpointCallback(
                save_freq=1000,
                save_path="data/checkpoints/",
                name_prefix="combat",
                verbose=1,
            ),
        ])

        logger.info("Starting RL training (MaskablePPO)...")
        try:
            model.learn(total_timesteps=10_000_000, callback=callbacks)
            logger.info("Training complete.")
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        finally:
            model.save(model_path)
            logger.info("Model saved to %s", model_path)
    else:
        from src.agent import SimpleAgent
        from src.game_loop import GameLoop

        agent = SimpleAgent(scorer=scorer)
        loop = GameLoop(communicator, agent, run_tracker=tracker,
                        live_state_writer=live_writer)
        loop.run()
```

- [ ] **Step 2: Run full test suite to confirm nothing broke**

```bash
pytest tests/ -v --ignore=tests/test_combat_env.py
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: wire LiveStateWriter into main entry point"
```

---

## Task 5: Create `dashboard.py`

**Files:**
- Create: `dashboard.py`
- Create: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_dashboard.py`:

```python
import json
import os
import tempfile
import pytest


@pytest.fixture
def client(tmp_path):
    live_state_path = tmp_path / "live_state.json"
    import dashboard
    dashboard.LIVE_STATE_PATH = str(live_state_path)
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c, live_state_path


def test_api_state_returns_empty_when_no_file(client):
    c, _ = client
    resp = c.get("/api/state")
    assert resp.status_code == 200
    assert resp.get_json() == {}


def test_api_state_returns_file_contents(client):
    c, live_state_path = client
    data = {"live": {"screen_type": "COMBAT", "current_hp": 47, "max_hp": 80,
                     "last_action": "END", "monsters": [], "updated_at": "2026-01-01T00:00:00Z"},
            "stats": {"run_number": 5, "wins": 2, "losses": 3,
                      "win_rate": 0.4, "avg_floor": 12.0}}
    live_state_path.write_text(json.dumps(data))
    resp = c.get("/api/state")
    assert resp.status_code == 200
    assert resp.get_json()["live"]["current_hp"] == 47
    assert resp.get_json()["stats"]["run_number"] == 5


def test_live_route_returns_html(client):
    c, _ = client
    resp = c.get("/live")
    assert resp.status_code == 200
    assert b"text/html" in resp.content_type.encode()
    assert b"screen-type" in resp.data
    assert b"hp-bar" in resp.data
    assert b"last-action" in resp.data


def test_stats_route_returns_html(client):
    c, _ = client
    resp = c.get("/stats")
    assert resp.status_code == 200
    assert b"text/html" in resp.content_type.encode()
    assert b"run-number" in resp.data
    assert b"win-rate" in resp.data
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_dashboard.py -v
```

Expected: `ModuleNotFoundError: No module named 'dashboard'`

- [ ] **Step 3: Create `dashboard.py`**

```python
import json
import os
from flask import Flask, Response

app = Flask(__name__)
LIVE_STATE_PATH = "data/live_state.json"

_LIVE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: transparent;
    font-family: monospace;
    color: #e0e0e0;
  }
  .panel {
    background: rgba(20, 20, 40, 0.92);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 16px 18px;
    width: 300px;
  }
  .screen-type {
    color: #ff6b35;
    font-weight: bold;
    font-size: 14px;
    letter-spacing: 0.04em;
    margin-bottom: 12px;
  }
  .hp-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
  }
  .hp-label { font-size: 11px; color: #a8a8b3; min-width: 18px; }
  .hp-track {
    flex: 1;
    background: rgba(255,255,255,0.12);
    border-radius: 3px;
    height: 7px;
  }
  .hp-fill {
    background: #ff4757;
    height: 7px;
    border-radius: 3px;
    transition: width 0.4s ease;
  }
  .hp-text { color: #ff4757; font-size: 11px; min-width: 52px; text-align: right; }
  .section-label {
    color: #a8a8b3;
    font-size: 10px;
    margin-bottom: 4px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .last-action { color: #7bed9f; font-size: 12px; margin-bottom: 10px; }
  .enemy-name { color: #e0e0e0; }
  .enemy-hp { color: #ff4757; }
  .enemy-intent { color: #ffa07a; }
  #enemies { font-size: 11px; }
  .enemy-row { margin-bottom: 3px; }
</style>
</head>
<body>
<div class="panel">
  <div class="screen-type" id="screen-type">— —</div>
  <div class="hp-row">
    <span class="hp-label">HP</span>
    <div class="hp-track"><div class="hp-fill" id="hp-bar" style="width:100%"></div></div>
    <span class="hp-text" id="hp-text">— / —</span>
  </div>
  <div class="section-label">Last Action</div>
  <div class="last-action" id="last-action">—</div>
  <div class="section-label" id="enemies-label" style="display:none">Enemies</div>
  <div id="enemies"></div>
</div>
<script>
const ICONS = {
  NONE: "⚔", COMBAT: "⚔", MAP: "🗺", REST: "🔥",
  SHOP_ROOM: "🛒", SHOP_SCREEN: "🛒", EVENT: "❓",
  CARD_REWARD: "🃏", CHEST: "📦", BOSS_REWARD: "👑",
  COMBAT_REWARD: "💰", GAME_OVER: "💀", HAND_SELECT: "🤚",
  GRID: "🔲",
};

async function poll() {
  try {
    const resp = await fetch("/api/state");
    const data = await resp.json();
    const live = data.live || {};
    const st = live.screen_type || "—";
    document.getElementById("screen-type").textContent =
      (ICONS[st] || "") + " " + st.replace("_", " ");
    const hp = live.current_hp ?? 0;
    const maxHp = live.max_hp || 1;
    const pct = Math.round((hp / maxHp) * 100);
    document.getElementById("hp-bar").style.width = pct + "%";
    document.getElementById("hp-text").textContent = hp + " / " + maxHp;
    document.getElementById("last-action").textContent = live.last_action || "—";
    const monsters = live.monsters || [];
    const enemiesDiv = document.getElementById("enemies");
    const enemiesLabel = document.getElementById("enemies-label");
    if (monsters.length === 0) {
      enemiesDiv.innerHTML = "";
      enemiesLabel.style.display = "none";
    } else {
      enemiesLabel.style.display = "block";
      enemiesDiv.innerHTML = monsters.map(m =>
        `<div class="enemy-row">` +
        `<span class="enemy-name">${m.name}</span> ` +
        `<span class="enemy-hp">${m.current_hp}</span>/${m.max_hp}hp` +
        (m.intent ? ` &nbsp;·&nbsp; <span class="enemy-intent">${m.intent}</span>` : "") +
        `</div>`
      ).join("");
    }
  } catch (e) {}
}
setInterval(poll, 2500);
poll();
</script>
</body>
</html>"""

_STATS_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: transparent; font-family: monospace; color: #e0e0e0; }
  .panel {
    background: rgba(20, 20, 40, 0.92);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 16px 18px;
    width: 260px;
  }
  .title {
    color: #a29bfe;
    font-weight: bold;
    font-size: 13px;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
  }
  .grid { display: grid; grid-template-columns: 1fr 1fr; row-gap: 8px; column-gap: 12px; font-size: 12px; }
  .label { color: #a8a8b3; }
  .val-gold { color: #ffd700; }
  .val-green { color: #7bed9f; }
  .val-red { color: #ff4757; }
  .val-blue { color: #74b9ff; }
</style>
</head>
<body>
<div class="panel">
  <div class="title">RUN STATS</div>
  <div class="grid">
    <div class="label">Run <span class="val-gold" id="run-number">—</span></div>
    <div class="label">Win rate <span class="val-green" id="win-rate">—</span></div>
    <div class="label">Avg floor <span class="val-blue" id="avg-floor">—</span></div>
    <div class="label">
      W/L <span class="val-green" id="wins">—</span>
      / <span class="val-red" id="losses">—</span>
    </div>
  </div>
</div>
<script>
async function poll() {
  try {
    const resp = await fetch("/api/state");
    const data = await resp.json();
    const s = data.stats || {};
    document.getElementById("run-number").textContent = s.run_number ?? "—";
    document.getElementById("win-rate").textContent =
      s.win_rate != null ? Math.round(s.win_rate * 100) + "%" : "—";
    document.getElementById("avg-floor").textContent =
      s.avg_floor != null ? s.avg_floor.toFixed(1) : "—";
    document.getElementById("wins").textContent = s.wins ?? "—";
    document.getElementById("losses").textContent = s.losses ?? "—";
  } catch (e) {}
}
setInterval(poll, 2500);
poll();
</script>
</body>
</html>"""


@app.route("/api/state")
def api_state():
    if not os.path.exists(LIVE_STATE_PATH):
        return Response("{}", status=200, mimetype="application/json")
    try:
        with open(LIVE_STATE_PATH) as f:
            return Response(f.read(), status=200, mimetype="application/json")
    except OSError:
        return Response("{}", status=200, mimetype="application/json")


@app.route("/live")
def live():
    return Response(_LIVE_HTML, status=200, mimetype="text/html")


@app.route("/stats")
def stats():
    return Response(_STATS_HTML, status=200, mimetype="text/html")


if __name__ == "__main__":
    print("Live view:  http://localhost:5000/live")
    print("Stats view: http://localhost:5000/stats")
    app.run(host="0.0.0.0", port=5000, debug=False)
```

- [ ] **Step 4: Run dashboard tests**

```bash
pytest tests/test_dashboard.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/test_combat_env.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add dashboard.py tests/test_dashboard.py
git commit -m "feat: add Flask dashboard server with /live and /stats OBS sources"
```

---

## Usage

Start the game AI as normal:
```bash
python main.py
```

In a separate terminal, start the dashboard:
```bash
python dashboard.py
```

Add to OBS as two Browser Sources:
- **Live view:** `http://localhost:5000/live` — 300×170px
- **Stats panel:** `http://localhost:5000/stats` — 260×120px

Both have transparent backgrounds and will composite cleanly over the game capture.
