# Live Monitoring Dashboard — Design Spec

**Date:** 2026-05-30  
**Status:** Approved

## Overview

A lightweight web dashboard that gives stream viewers live visibility into the Slay the Spire AI while it runs. Two independent OBS browser sources: one showing current game state, one showing aggregate run statistics. Both are served by a standalone Flask server that reads a JSON file the game loop maintains.

## Architecture

The system has three layers:

1. **Data writer** — `LiveStateWriter` in `src/live_state.py` writes `data/live_state.json` after each agent action and after each run ends.
2. **Flask server** — `dashboard.py` at the project root serves two HTML pages and one JSON endpoint. Run independently with `python dashboard.py` (default port 5000).
3. **Browser panels** — Each HTML page polls `/api/state` every 2.5 seconds and updates the DOM in place. No page reloads.

The game process and dashboard process are fully decoupled: the dashboard reads only from the file and can be started/stopped without affecting the AI.

## OBS Browser Sources

Two separate URLs, each a self-contained overlay page:

- `http://localhost:5000/live` — Live game state panel (~300×170px)
- `http://localhost:5000/stats` — Run stats panel (~260×120px)

Both use a dark semi-transparent background (`rgba(20, 20, 40, 0.92)`) with a subtle border, monospace font, and color-coded values (green for wins/positive, red for HP/losses, gold for run number, blue for floor). Transparent outer background so they composite cleanly over the game capture in OBS.

## Data Model — `data/live_state.json`

```json
{
  "live": {
    "screen_type": "COMBAT",
    "current_hp": 47,
    "max_hp": 80,
    "last_action": "PLAY 2 0 (Bash → Jaw Worm)",
    "monsters": [
      { "name": "Jaw Worm", "current_hp": 34, "max_hp": 44, "intent": "Attack 11" }
    ],
    "updated_at": "2026-05-30T12:34:56Z"
  },
  "stats": {
    "run_number": 47,
    "wins": 16,
    "losses": 31,
    "win_rate": 0.34,
    "avg_floor": 18.2
  }
}
```

- `live` updates after every agent action
- `stats` updates only when a run ends
- `monsters` is always a list (empty when not in combat) — no null-checks needed in JS
- `last_action` is built in `LiveStateWriter.write()`: for `PLAY N T` commands, the card name is looked up from `state.hand[N-1]` and the target monster name from `state.monsters[T]` to produce e.g. `"PLAY 2 0 (Bash → Jaw Worm)"`; all other commands are stored as-is (e.g. `"END"`, `"CHOOSE rest"`)

## Live View Panel — displayed fields

Intentionally minimal — floor, gold, deck size, and relic count are visible in the game capture and omitted here.

- **Screen type** — colored header (e.g. "⚔ COMBAT", "🗺 MAP", "🔥 REST")
- **HP bar** — colored progress bar with numeric HP / max HP
- **Last action** — what the agent just did, human-readable
- **Enemies** — name, current/max HP, intent (hidden when not in combat)

## Stats Panel — displayed fields

- Run number
- Win rate (%)
- Average floor reached
- Total wins / losses

## New Files

### `src/live_state.py`

```python
class LiveStateWriter:
    def __init__(self, path: str = "data/live_state.json"): ...
    def write(self, state: GameState, last_action: str) -> None: ...
    def write_run_summary(self, summary: dict) -> None: ...
```

`write()` updates the `live` key. `write_run_summary()` updates the `stats` key. Both read the existing JSON, update their respective key, and write the whole file back — so neither method overwrites the other's key.

### `dashboard.py`

Flask app with three routes:

| Route | Response |
|-------|----------|
| `GET /live` | HTML — live game state OBS source |
| `GET /stats` | HTML — run stats OBS source |
| `GET /api/state` | JSON — contents of `data/live_state.json` |

HTML pages are embedded as strings in the file. Each page's JS polls `/api/state` every 2500ms and updates named DOM elements in place.

## Modified Files

| File | Change |
|------|--------|
| `src/game_loop.py` | Accept `LiveStateWriter` instance; call `writer.write(state, action)` after each `agent.act()` |
| `src/run_tracker.py` | Accept `LiveStateWriter` instance; call `writer.write_run_summary(self.summary())` inside `record_run()` |
| `main.py` | Construct `LiveStateWriter`, pass to both `GameLoop` and `RunTracker` |

## Files Not Modified

`agent.py`, `grapher.py`, `communicator.py`, `game_state.py`, `callbacks.py`, `combat_env.py` — the dashboard is purely observational.

## Error Handling

- If `data/live_state.json` doesn't exist yet, `/api/state` returns `{}` with a 200. The JS handles empty state gracefully (shows dashes).
- If the game process isn't running, the dashboard shows the last known state with no indication of staleness (acceptable for streaming use).
- `LiveStateWriter` catches and logs `OSError` on file write rather than crashing the game loop.
