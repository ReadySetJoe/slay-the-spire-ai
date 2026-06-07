# Never-Stuck Game Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the game loop from getting stuck indefinitely by fixing the GRID/HAND_SELECT card-selection bug and adding a `StuckDetectorAgent` wrapper that escalates through fallback commands and logs a rich diagnostic block when oscillation is detected.

**Architecture:** `StuckDetectorAgent` wraps any `Agent`, tracks a `(screen_type, frozenset(available_commands))` fingerprint, and takes over `act()` once the same fingerprint has been seen `threshold` times in a row — cycling through `CONFIRM → CANCEL → PROCEED → SKIP → STATE` and logging CRITICAL. The GRID/HAND_SELECT fix corrects UUID-based selected-card tracking in `SimpleAgent`. Both `GameLoop` and `CombatEnv` wrap their agent/simple_agent at construction time.

**Tech Stack:** Python, pytest, unittest.mock

---

## File Map

| File | Change |
|------|--------|
| `src/agent.py` | Add `import json`; add `_FALLBACK_SEQUENCE` constant; extract `_handle_grid_hand_select` from inline block; fix UUID-based selected tracking; add `StuckDetectorAgent` class |
| `src/game_loop.py` | Import `StuckDetectorAgent`; wrap `agent` in `__init__` |
| `src/combat_env.py` | Import `StuckDetectorAgent`; wrap `simple_agent` in `__init__` |
| `tests/test_agent.py` | Add GRID/HAND_SELECT fixtures + 4 tests; add `StuckDetectorAgent` fixtures + 9 tests |
| `tests/test_game_loop.py` | Add 1 wiring test |
| `tests/test_combat_env.py` | Add 1 wiring test |

---

### Task 1: Fix GRID/HAND_SELECT selected-card tracking

**Files:**
- Modify: `src/agent.py`
- Modify: `tests/test_agent.py`

The existing inline GRID/HAND_SELECT handler in `SimpleAgent.act()` checks `c.get("selected", False)` on each card object, but CommunicationMod never puts a `selected` boolean on individual card objects — it uses a separate `selected` (HAND_SELECT) or `selected_cards` (GRID) array in `screen_state`. This causes oscillation on any multi-card selection (e.g., Neow's "Transform 2 cards"). Fix: extract the block into `_handle_grid_hand_select` and compare card UUIDs against the separate array.

- [ ] **Step 1: Add card fixtures and failing tests to `tests/test_agent.py`**

Append to `tests/test_agent.py`:

```python
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
```

- [ ] **Step 2: Run new tests to verify they fail**

```
python -m pytest tests/test_agent.py -k "grid or hand_select" -v
```

Expected: all 4 tests FAIL (wrong action returned by existing code)

- [ ] **Step 3: Extract `_handle_grid_hand_select` and fix UUID tracking in `src/agent.py`**

In `src/agent.py`, replace the inline GRID/HAND_SELECT block in `act()`:

```python
        if state.screen_type in ("GRID", "HAND_SELECT"):
            # CONFIRM takes priority: sent once enough cards are selected
            if "CONFIRM" in state.available_commands:
                return "CONFIRM"
            if "CHOOSE" in state.available_commands:
                cards = state.screen_state.get("cards", []) if state.screen_state else []
                # Only consider cards not already selected to avoid deselect oscillation
                unselected = [(i, c) for i, c in enumerate(cards)
                              if not c.get("selected", False)]
                if unselected:
                    best_local = pick_best_card([c for _, c in unselected])
                    best_idx = unselected[best_local if best_local is not None else 0][0]
                else:
                    best_idx = 0
                return f"CHOOSE {best_idx}"
            return "CANCEL"
```

with:

```python
        if state.screen_type in ("GRID", "HAND_SELECT"):
            return self._handle_grid_hand_select(state)
```

Then add this method to `SimpleAgent` (place it after `_handle_event`):

```python
    def _handle_grid_hand_select(self, state: GameState) -> str:
        if "CONFIRM" in state.available_commands:
            return "CONFIRM"
        if "CHOOSE" in state.available_commands:
            ss = state.screen_state or {}
            cards = ss.get("cards", [])
            # CommunicationMod tracks selected cards in a separate array, not as a
            # boolean on individual card objects. HAND_SELECT uses "selected";
            # GRID uses "selected_cards".
            already_selected = {
                c.get("uuid")
                for c in ss.get("selected", []) + ss.get("selected_cards", [])
            }
            unselected = [
                (i, c) for i, c in enumerate(cards)
                if c.get("uuid") not in already_selected
            ]
            if unselected:
                best_local = pick_best_card([c for _, c in unselected])
                best_idx = unselected[best_local if best_local is not None else 0][0]
                return f"CHOOSE {best_idx}"
        return "CANCEL"
```

- [ ] **Step 4: Run the new tests to verify they pass**

```
python -m pytest tests/test_agent.py -k "grid or hand_select" -v
```

Expected: all 4 PASS

- [ ] **Step 5: Run the full agent test suite to check for regressions**

```
python -m pytest tests/test_agent.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```
git add src/agent.py tests/test_agent.py
git commit -m "fix: use UUID comparison for GRID/HAND_SELECT selected-card tracking"
```

---

### Task 2: Add StuckDetectorAgent

**Files:**
- Modify: `src/agent.py`
- Modify: `tests/test_agent.py`

`StuckDetectorAgent` wraps any `Agent`. It tracks how many consecutive times the same `(screen_type, frozenset(available_commands))` fingerprint has been seen. Below `threshold`, it delegates to the wrapped agent. At `threshold` and above, it takes over — returning fallback commands in sequence (`CONFIRM → CANCEL → PROCEED → SKIP → STATE`) and logging CRITICAL on the first stuck step and every `log_interval` steps after.

- [ ] **Step 1: Add `StuckDetectorAgent` tests to `tests/test_agent.py`**

First add the missing `import json` at the top of `tests/test_agent.py` (it uses `json.dumps` in fixtures already, so this may already be there — confirm before adding):

Then append to `tests/test_agent.py`:

```python
# ── StuckDetectorAgent tests ──────────────────────────────────────────────────

import logging
from unittest.mock import MagicMock
from src.agent import StuckDetectorAgent


def _sda_state(screen_type="MAP", commands=("CHOOSE", "STATE"), floor=2, hp=60, max_hp=80):
    """Minimal non-combat state for StuckDetectorAgent tests."""
    return GameState.from_json(json.dumps({
        "available_commands": list(commands),
        "ready_for_command": True, "in_game": True,
        "game_state": {
            "screen_type": screen_type,
            "screen_state": {},
            "seed": 1, "floor": floor, "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": hp, "max_hp": max_hp, "gold": 50,
            "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
            "combat_state": None,
        }
    }))


def test_sda_delegates_normally_when_fingerprint_changes():
    """Passes every call through to wrapped agent while fingerprint keeps changing."""
    inner = MagicMock()
    inner.act.return_value = "CHOOSE 0"
    agent = StuckDetectorAgent(inner, threshold=3)

    agent.act(_sda_state("MAP",  ["CHOOSE"]))
    agent.act(_sda_state("REST", ["CHOOSE"]))
    agent.act(_sda_state("MAP",  ["CHOOSE", "CANCEL"]))

    assert inner.act.call_count == 3


def test_sda_delegates_below_threshold():
    """Still delegates when repeat count is below threshold."""
    inner = MagicMock()
    inner.act.return_value = "CHOOSE 0"
    agent = StuckDetectorAgent(inner, threshold=3)

    state = _sda_state("MAP", ["CHOOSE"])
    agent.act(state)  # seen_count = 1
    agent.act(state)  # seen_count = 2, still below threshold=3

    assert inner.act.call_count == 2


def test_sda_triggers_confirm_at_threshold():
    """Returns CONFIRM (first fallback) on the threshold-th identical state."""
    inner = MagicMock()
    inner.act.return_value = "CHOOSE 0"
    agent = StuckDetectorAgent(inner, threshold=3)

    state = _sda_state("MAP", ["CHOOSE", "CONFIRM"])
    agent.act(state)  # seen_count=1, delegate
    agent.act(state)  # seen_count=2, delegate
    action = agent.act(state)  # seen_count=3, stuck → CONFIRM

    assert action == "CONFIRM"
    assert inner.act.call_count == 2  # third call did NOT delegate


def test_sda_advances_fallback_sequence():
    """Returns successive fallback commands across consecutive stuck steps."""
    inner = MagicMock()
    inner.act.return_value = "CHOOSE 0"
    agent = StuckDetectorAgent(inner, threshold=3)

    state = _sda_state("MAP", ["CHOOSE", "CONFIRM", "CANCEL", "PROCEED"])
    agent.act(state)  # seen_count=1
    agent.act(state)  # seen_count=2

    a3 = agent.act(state)  # stuck_step=0 → CONFIRM
    a4 = agent.act(state)  # stuck_step=1 → CANCEL
    a5 = agent.act(state)  # stuck_step=2 → PROCEED

    assert a3 == "CONFIRM"
    assert a4 == "CANCEL"
    assert a5 == "PROCEED"


def test_sda_falls_back_to_state_when_cmd_unavailable():
    """Returns STATE when the scheduled fallback command is not available."""
    inner = MagicMock()
    inner.act.return_value = "CHOOSE 0"
    agent = StuckDetectorAgent(inner, threshold=3)

    # CONFIRM / CANCEL / PROCEED / SKIP not available
    state = _sda_state("MAP", ["CHOOSE"])
    agent.act(state)
    agent.act(state)
    action = agent.act(state)  # stuck_step=0, CONFIRM unavailable → STATE

    assert action == "STATE"


def test_sda_resets_on_fingerprint_change():
    """Counter resets when a different screen/commands is seen."""
    inner = MagicMock()
    inner.act.return_value = "CHOOSE 0"
    agent = StuckDetectorAgent(inner, threshold=3)

    map_state  = _sda_state("MAP",  ["CHOOSE", "CONFIRM"])
    rest_state = _sda_state("REST", ["CHOOSE"])

    agent.act(map_state)   # seen=1
    agent.act(map_state)   # seen=2
    agent.act(rest_state)  # fingerprint changes → reset, seen=1, delegate
    agent.act(map_state)   # different from REST → reset, seen=1, delegate
    agent.act(map_state)   # seen=2, below threshold
    action = agent.act(map_state)  # seen=3 → stuck again

    assert action == "CONFIRM"
    assert inner.act.call_count == 5  # all steps except the last delegated


def test_sda_logs_critical_when_stuck(caplog):
    """Logs CRITICAL with diagnostic block on the first stuck step."""
    inner = MagicMock()
    inner.act.return_value = "CHOOSE 0"
    agent = StuckDetectorAgent(inner, threshold=3)

    state = _sda_state("MAP", ["CHOOSE"], floor=3, hp=45, max_hp=80)
    agent.act(state)
    agent.act(state)

    with caplog.at_level(logging.CRITICAL, logger="src.agent"):
        agent.act(state)

    assert "STUCK DETECTED" in caplog.text
    assert "MAP" in caplog.text
    assert "Paste this block into Claude to diagnose" in caplog.text


def test_sda_resets_on_game_over():
    """on_game_over resets stuck state so next run starts clean."""
    inner = MagicMock()
    inner.act.return_value = "CHOOSE 0"
    agent = StuckDetectorAgent(inner, threshold=3)

    state = _sda_state("MAP", ["CHOOSE", "CONFIRM"])
    agent.act(state)
    agent.act(state)
    agent.act(state)  # now stuck

    game_over = _sda_state("GAME_OVER", ["PROCEED"])
    agent.on_game_over(game_over)

    # After reset, should delegate normally for threshold-1 more steps
    agent.act(state)   # seen=1, delegate
    action = agent.act(state)  # seen=2, still below threshold
    assert inner.act.call_count == 4  # 2 pre-stuck + 2 post-reset


def test_sda_delegates_on_game_over_to_wrapped_agent():
    """on_game_over forwards to the wrapped agent's on_game_over if it exists."""
    inner = MagicMock()
    agent = StuckDetectorAgent(inner, threshold=3)

    game_over = _sda_state("GAME_OVER", ["PROCEED"])
    agent.on_game_over(game_over)

    inner.on_game_over.assert_called_once_with(game_over)
```

- [ ] **Step 2: Run new tests to verify they fail**

```
python -m pytest tests/test_agent.py -k "sda" -v
```

Expected: all 9 tests FAIL (`StuckDetectorAgent` not yet defined)

- [ ] **Step 3: Add `import json` and `_FALLBACK_SEQUENCE` to `src/agent.py`**

At the top of `src/agent.py`, add `import json` alongside the existing imports:

```python
import json
import logging
import random
from abc import ABC, abstractmethod
from typing import Optional
```

After the imports (before the `logger = ...` line), add:

```python
_FALLBACK_SEQUENCE = ("CONFIRM", "CANCEL", "PROCEED", "SKIP", "STATE")
```

- [ ] **Step 4: Add `StuckDetectorAgent` class to `src/agent.py`**

Append the following class at the end of `src/agent.py` (after `SimpleAgent`):

```python
class StuckDetectorAgent(Agent):
    """Wraps any Agent and prevents infinite oscillation.

    Tracks a fingerprint of (screen_type, frozenset(available_commands)).
    Once the same fingerprint has been seen `threshold` times in a row, the
    wrapper takes over and cycles through _FALLBACK_SEQUENCE instead of
    delegating to the wrapped agent. A CRITICAL log with full screen_state
    is emitted so the stuck state can be diagnosed from the log alone.

    Attribute access for unknown names falls through to the wrapped agent,
    allowing callers to use agent-specific methods (e.g. _check_potions)
    transparently.
    """

    def __init__(self, agent: Agent, threshold: int = 3, log_interval: int = 10):
        self._agent = agent
        self._threshold = threshold
        self._log_interval = log_interval
        self._last_fp: Optional[tuple] = None
        self._seen_count: int = 0
        self._action_history: list = []

    def __getattr__(self, name: str):
        return getattr(self._agent, name)

    def act(self, state: GameState) -> str:
        fp = (state.screen_type, frozenset(state.available_commands))

        if fp != self._last_fp:
            self._last_fp = fp
            self._seen_count = 1
            self._action_history.clear()
            action = self._agent.act(state)
            self._action_history.append(action)
            return action

        self._seen_count += 1

        if self._seen_count < self._threshold:
            action = self._agent.act(state)
            self._action_history.append(action)
            return action

        # Stuck — take over with fallback sequence
        stuck_step = self._seen_count - self._threshold  # 0-based

        if stuck_step == 0 or stuck_step % self._log_interval == 0:
            logger.critical(
                "STUCK DETECTED — floor=%d hp=%d/%d screen=%s commands=%s\n"
                "Actions sent (last 5): %s\n"
                "screen_state: %s\n"
                "Paste this block into Claude to diagnose.",
                state.floor, state.current_hp, state.max_hp,
                state.screen_type, sorted(state.available_commands),
                list(self._action_history[-5:]),
                json.dumps(state.screen_state, default=str),
            )

        idx = min(stuck_step, len(_FALLBACK_SEQUENCE) - 1)
        cmd = _FALLBACK_SEQUENCE[idx]
        if cmd not in state.available_commands and cmd != "STATE":
            cmd = "STATE"

        self._action_history.append(cmd)
        return cmd

    def on_game_over(self, state: GameState) -> None:
        if hasattr(self._agent, "on_game_over"):
            self._agent.on_game_over(state)
        self._last_fp = None
        self._seen_count = 0
        self._action_history.clear()
```

- [ ] **Step 5: Run the stuck detector tests**

```
python -m pytest tests/test_agent.py -k "sda" -v
```

Expected: all 9 PASS

- [ ] **Step 6: Run full agent suite**

```
python -m pytest tests/test_agent.py -v
```

Expected: all tests PASS

- [ ] **Step 7: Commit**

```
git add src/agent.py tests/test_agent.py
git commit -m "feat: add StuckDetectorAgent with fallback sequence and CRITICAL diagnostic log"
```

---

### Task 3: Wire StuckDetectorAgent into GameLoop and CombatEnv

**Files:**
- Modify: `src/game_loop.py`
- Modify: `src/combat_env.py`
- Modify: `tests/test_game_loop.py`
- Modify: `tests/test_combat_env.py`

Wire `StuckDetectorAgent` at construction time in both places. `CombatEnv` calls `self.simple_agent._check_potions()` directly — this works transparently because `StuckDetectorAgent.__getattr__` delegates unknown attribute lookups to the wrapped agent.

- [ ] **Step 1: Add wiring tests**

Append to `tests/test_game_loop.py`:

```python
def test_game_loop_wraps_agent_with_stuck_detector():
    from src.agent import StuckDetectorAgent
    loop = GameLoop(communicator=MagicMock(), agent=SimpleAgent())
    assert isinstance(loop.agent, StuckDetectorAgent)
```

Append to `tests/test_combat_env.py`:

```python
def test_combat_env_wraps_simple_agent_with_stuck_detector():
    from src.agent import StuckDetectorAgent
    env = CombatEnv(communicator=MagicMock())
    assert isinstance(env.simple_agent, StuckDetectorAgent)
```

- [ ] **Step 2: Run wiring tests to confirm they fail**

```
python -m pytest tests/test_game_loop.py::test_game_loop_wraps_agent_with_stuck_detector tests/test_combat_env.py::test_combat_env_wraps_simple_agent_with_stuck_detector -v
```

Expected: both FAIL (`StuckDetectorAgent` not yet wired)

- [ ] **Step 3: Wire into `GameLoop`**

In `src/game_loop.py`, update the import:

```python
from src.agent import Agent, StuckDetectorAgent
```

In `GameLoop.__init__`, replace:

```python
        self.agent = agent
```

with:

```python
        self.agent = StuckDetectorAgent(agent)
```

- [ ] **Step 4: Wire into `CombatEnv`**

In `src/combat_env.py`, update the import:

```python
from src.agent import SimpleAgent, StuckDetectorAgent
```

In `CombatEnv.__init__`, replace:

```python
        self.simple_agent = SimpleAgent(scorer=scorer)
```

with:

```python
        self.simple_agent = StuckDetectorAgent(SimpleAgent(scorer=scorer))
```

- [ ] **Step 5: Run wiring tests**

```
python -m pytest tests/test_game_loop.py::test_game_loop_wraps_agent_with_stuck_detector tests/test_combat_env.py::test_combat_env_wraps_simple_agent_with_stuck_detector -v
```

Expected: both PASS

- [ ] **Step 6: Run full test suite**

```
python -m pytest tests/ -v
```

Expected: all tests PASS (no regressions in existing game_loop or combat_env tests)

- [ ] **Step 7: Commit**

```
git add src/game_loop.py src/combat_env.py tests/test_game_loop.py tests/test_combat_env.py
git commit -m "feat: wire StuckDetectorAgent into GameLoop and CombatEnv"
```
