# Training Progress Visualization — Design Spec

**Date:** 2026-06-06  
**Status:** Approved

## Goal

Add a live chart at `localhost:5000/training` that shows the RL reward signal as the agent trains. Lets the user answer "is the agent learning?" without leaving the browser.

## Scope

- New `/training` route in `dashboard.py` (scatter + rolling average reward chart)
- `EpisodeLoggerCallback` writes per-episode training data into `live_state.json`
- New `write_training_step` method on `LiveStateWriter`
- `main.py` passes `live_writer` to `EpisodeLoggerCallback`

Out of scope: win rate or floor-reached charts (already covered by `/stats`), loss/entropy from SB3 internals, historical playback across restarts.

## Data Flow

```
EpisodeLoggerCallback (training process)
  → LiveStateWriter.write_training_step()
    → live_state.json  { "training": { ... } }
      → /api/state (Flask, existing endpoint)
        → /training page (Chart.js, polls every 3s)
```

No new files. No new API endpoints.

## Data Structure

`live_state.json` gains a `training` key written after every episode:

```json
{
  "training": {
    "total_episodes": 1247,
    "total_timesteps": 384920,
    "episodes": [
      { "ep": 748, "reward": -3.2, "steps": 42 },
      { "ep": 749, "reward":  1.5, "steps": 61 }
    ]
  }
}
```

- `episodes`: rolling window of the last 500 episodes (constant `_TRAINING_WINDOW = 500` in `callbacks.py`). Each entry holds only the fields needed for the chart.
- `total_episodes`, `total_timesteps`: running counters, never truncated. Used to label the x-axis correctly after the window rolls.

## Component Changes

### `src/callbacks.py` — `EpisodeLoggerCallback`

- Add `live_state_writer=None` parameter to `__init__`.
- Add `_total_episodes: int = 0` and `_total_timesteps: int = 0` instance counters.
- Add `_training_episodes: deque` with `maxlen=_TRAINING_WINDOW`.
- In `_log_episode`, append `{"ep": self._total_episodes, "reward": ep["r"], "steps": ep["l"]}` to the deque, then call `self.live_state_writer.write_training_step(...)` if the writer is set.

### `src/live_state.py` — `LiveStateWriter`

Add one method:

```python
def write_training_step(
    self,
    episodes: list[dict],
    total_episodes: int,
    total_timesteps: int,
) -> None:
    self._update("training", {
        "total_episodes": total_episodes,
        "total_timesteps": total_timesteps,
        "episodes": episodes,
    })
```

No other changes. Reuses the existing atomic `_update` write pattern.

### `main.py`

Change:
```python
EpisodeLoggerCallback(summary_freq=100)
```
to:
```python
EpisodeLoggerCallback(summary_freq=100, live_state_writer=live_writer)
```

### `dashboard.py` — `/training` route

New route `@app.route("/training")` serving `_TRAINING_HTML`, an inline string styled with the existing dark theme (`#14142a` background, `#1a1a2e` panels, `#a29bfe` accent).

**Page layout:**
- Header: "TRAINING"
- Four stat chips: Total Episodes · Total Timesteps · Avg Reward (last 100) · Best Episode
- Chart.js canvas: scatter plot of raw episode rewards (blue, 35% opacity) overlaid with a rolling average line (orange, window=50)
- Polls `/api/state` every 3 seconds; gracefully shows "—" if `data.training` is absent (agent not running in RL mode)

**Chart details:**
- X axis: episode number (using `ep` field from each record)
- Y axis: episode reward (auto-scaled)
- Rolling average computed client-side in JavaScript before rendering
- Chart.js loaded from CDN: `https://cdn.jsdelivr.net/npm/chart.js`

## Constants

| Name | Location | Value | Purpose |
|---|---|---|---|
| `_TRAINING_WINDOW` | `callbacks.py` | `500` | Max episodes kept in `live_state.json` |
| Rolling avg window | `_TRAINING_HTML` JS | `50` | Smoothing window for the trend line |
| Poll interval | `_TRAINING_HTML` JS | `3000 ms` | How often the browser fetches `/api/state` |

## Error Handling

- If `live_state_writer` is `None` (e.g. running without the dashboard), `write_training_step` is never called — no change in behavior.
- If the training process hasn't written yet, `/api/state` returns no `training` key; the chart page shows "—" in all stat chips and an empty chart.
- `write_training_step` reuses `LiveStateWriter._update`, which already handles `OSError` gracefully (logs and returns).

## Testing

- Unit test: `EpisodeLoggerCallback` with a mock `live_state_writer` — verify `write_training_step` is called with correct episode data, correct rolling window truncation, and correct running totals.
- Unit test: `LiveStateWriter.write_training_step` — verify it merges correctly into `live_state.json` without clobbering the `live` or `stats` keys.
- Manual: run `python main.py --rl`, open `localhost:5000/training`, confirm chart updates live.
