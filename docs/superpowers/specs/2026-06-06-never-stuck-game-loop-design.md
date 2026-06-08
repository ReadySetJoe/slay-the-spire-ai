# Never-Stuck Game Loop Design

**Date:** 2026-06-06
**Status:** Approved

## Problem

The game loop can get stuck in an infinite oscillation when `SimpleAgent` sends a command that has no net effect — the most immediate example being the GRID/HAND_SELECT screen where selecting a card requires 2 picks but the agent toggles the same card repeatedly. The root cause of that specific bug is that `c.get("selected", False)` always returns `False` because CommunicationMod tracks selected cards in a separate `selected`/`selected_cards` array, not as a boolean field on individual card objects. But the same class of stuck state can arise from any unhandled or mishandled screen.

The goal: the loop must never stay stuck indefinitely. If stuck, log a rich diagnostic block so the issue can be diagnosed from the log alone.

## Design

### StuckDetectorAgent wrapper

A new `StuckDetectorAgent(Agent)` class in `src/agent.py` wraps any `Agent` and intercepts `act()`:

```python
class StuckDetectorAgent(Agent):
    def __init__(self, agent: Agent, threshold: int = 3, log_interval: int = 10):
        ...
    def act(self, state: GameState) -> str:
        ...
    def on_game_over(self, state: GameState):
        ...  # delegates to wrapped agent and resets stuck state
```

**Fingerprint:** `(state.screen_type, frozenset(state.available_commands))`

**Counter logic:**
- If fingerprint matches previous → increment counter, append last action to history
- If fingerprint changes → reset counter to 0, clear history, delegate to wrapped agent normally

**Fallback sequence (triggered when counter ≥ threshold):**

Maintain a fallback index into this ordered list: `[CONFIRM, CANCEL, PROCEED, SKIP, STATE]`

At threshold (counter == threshold): set fallback index to 0, return `fallback_sequence[0]` if present in `available_commands`, else advance index until one is found or fall through to `STATE`.

Each subsequent stuck step (counter > threshold): advance fallback index by 1 and return the next command if available, else advance again. Once the index reaches `STATE`, keep returning `STATE` on every subsequent stuck step.

This ensures the wrapper tries each escape command exactly once before giving up — CONFIRM at step 3, CANCEL at step 4, PROCEED at step 5, SKIP at step 6, STATE from step 7 onward. A command that is unavailable is skipped (index advances past it). A command that is available but has no effect (fingerprint unchanged) is abandoned in favour of the next one.

**Diagnostic log** — fires at the stuck threshold and every `log_interval` steps after:

```
STUCK DETECTED — floor=N hp=X/Y screen=SCREEN_TYPE commands=[...]
Actions sent (last 5): [...]
screen_state: {full JSON}
Paste this block into Claude to diagnose.
```

Logged at `CRITICAL` level so it appears regardless of log level configuration.

### GRID/HAND_SELECT selected-card fix

In `SimpleAgent._handle_grid_hand_select` (extracted from the inline block in `act()`), build a set of already-selected UUIDs from the separate arrays in `screen_state`:

```python
ss = state.screen_state or {}
# CommunicationMod uses "selected" for HAND_SELECT and "selected_cards" for GRID
already_selected = {
    c.get("uuid") for c in ss.get("selected", []) + ss.get("selected_cards", [])
}
unselected = [(i, c) for i, c in enumerate(cards)
              if c.get("uuid") not in already_selected]
```

This correctly excludes already-selected cards regardless of which key CommunicationMod uses.

### Wiring

Two wiring points — no other files change:

1. **`src/game_loop.py` `__init__`:** wrap `agent` with `StuckDetectorAgent(agent)`
2. **`src/combat_env.py` `__init__`:** wrap `simple_agent` with `StuckDetectorAgent(simple_agent)`

### Reset on game over

`StuckDetectorAgent.on_game_over()` delegates to the wrapped agent's `on_game_over()` and resets stuck state (counter, history, last fingerprint). This ensures a fresh run starts clean.

## Files Changed

| File | Change |
|------|--------|
| `src/agent.py` | Add `StuckDetectorAgent`; extract `_handle_grid_hand_select` from inline block; fix UUID-based selected tracking |
| `src/game_loop.py` | Wrap agent with `StuckDetectorAgent` in `__init__` |
| `src/combat_env.py` | Wrap `simple_agent` with `StuckDetectorAgent` in `__init__` |
| `tests/test_agent.py` | New tests for `StuckDetectorAgent` and fixed GRID selection |

## Thresholds

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `threshold` | 3 | React quickly; single-card GRID selections resolve in 1-2 steps so false positives are unlikely |
| `log_interval` | 10 | Log every 10 stuck steps to avoid log spam while keeping diagnosis data fresh |

## Success Criteria

- GRID/HAND_SELECT with `num_cards=2` no longer loops
- Any unhandled screen that repeats N times triggers escalation and logs a CRITICAL diagnostic
- All existing tests pass
- The diagnostic log block contains enough information to diagnose the stuck state from the log alone
