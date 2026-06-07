# Training Progress Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live chart at `localhost:5000/training` showing RL episode reward over time, updating every 3 seconds while the agent trains.

**Architecture:** `EpisodeLoggerCallback` writes a rolling window of episode rewards into `live_state.json` under a `training` key via a new `LiveStateWriter.write_training_step` method. The existing `/api/state` endpoint already serves `live_state.json`, so `dashboard.py` only needs a new `/training` route that polls that endpoint and renders a Chart.js scatter+rolling-average chart.

**Tech Stack:** Python/Flask (existing), Chart.js CDN, stable-baselines3 `BaseCallback`

---

## File Map

| File | Change |
|---|---|
| `src/live_state.py` | Add `write_training_step` method |
| `src/callbacks.py` | Add `_TRAINING_WINDOW` constant, `live_state_writer` param, rolling deque, call writer |
| `main.py` | Pass `live_writer` to `EpisodeLoggerCallback` |
| `dashboard.py` | Add `_TRAINING_HTML` string and `/training` route |
| `tests/test_live_state.py` | Add two tests for `write_training_step` |
| `tests/test_callbacks.py` | New file — four tests for callback training data |
| `tests/test_dashboard.py` | Add one test for `/training` route |

---

## Task 1: `LiveStateWriter.write_training_step`

**Files:**
- Modify: `src/live_state.py`
- Modify: `tests/test_live_state.py`

- [ ] **Step 1: Write two failing tests**

Append to `tests/test_live_state.py`:

```python
def test_write_training_step_creates_training_section():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        episodes = [
            {"ep": 1, "reward": -2.0, "steps": 30},
            {"ep": 2, "reward": 1.5, "steps": 45},
        ]
        writer.write_training_step(episodes, total_episodes=2, total_timesteps=75)
        with open(path) as f:
            data = json.load(f)
        assert "training" in data
        assert data["training"]["total_episodes"] == 2
        assert data["training"]["total_timesteps"] == 75
        assert data["training"]["episodes"] == episodes


def test_write_training_step_preserves_other_sections():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "live_state.json")
        writer = LiveStateWriter(path=path)
        existing = {"stats": {"run_number": 5, "wins": 2, "losses": 3,
                              "win_rate": 0.4, "avg_floor": 12.0}}
        with open(path, "w") as f:
            json.dump(existing, f)
        writer.write_training_step([], total_episodes=0, total_timesteps=0)
        with open(path) as f:
            data = json.load(f)
        assert data["stats"]["run_number"] == 5
        assert "training" in data
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_live_state.py::test_write_training_step_creates_training_section tests/test_live_state.py::test_write_training_step_preserves_other_sections -v
```

Expected: FAIL with `AttributeError: 'LiveStateWriter' object has no attribute 'write_training_step'`

- [ ] **Step 3: Implement `write_training_step` in `src/live_state.py`**

Add this method to the `LiveStateWriter` class, after `write_run_summary`:

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

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_live_state.py::test_write_training_step_creates_training_section tests/test_live_state.py::test_write_training_step_preserves_other_sections -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/live_state.py tests/test_live_state.py
git commit -m "feat: add LiveStateWriter.write_training_step"
```

---

## Task 2: `EpisodeLoggerCallback` training data capture

**Files:**
- Modify: `src/callbacks.py`
- Create: `tests/test_callbacks.py`

- [ ] **Step 1: Create `tests/test_callbacks.py` with four failing tests**

```python
import pytest
from unittest.mock import MagicMock
from src.callbacks import EpisodeLoggerCallback


def test_callback_calls_write_training_step_after_episode():
    writer = MagicMock()
    cb = EpisodeLoggerCallback(summary_freq=100, live_state_writer=writer)
    cb.num_timesteps = 50
    cb._log_episode({"r": 2.5, "l": 30, "hp": 40, "max_hp": 80, "floor": 3})
    writer.write_training_step.assert_called_once()
    kwargs = writer.write_training_step.call_args.kwargs
    assert kwargs["total_episodes"] == 1
    assert kwargs["total_timesteps"] == 50
    assert kwargs["episodes"] == [{"ep": 1, "reward": 2.5, "steps": 30}]


def test_callback_without_writer_does_not_raise():
    cb = EpisodeLoggerCallback(summary_freq=100)
    cb.num_timesteps = 0
    cb._log_episode({"r": 1.0, "l": 10, "hp": 80, "max_hp": 80, "floor": 1})


def test_callback_rolling_window_caps_at_500():
    writer = MagicMock()
    cb = EpisodeLoggerCallback(summary_freq=100, live_state_writer=writer)
    cb.num_timesteps = 0
    for i in range(505):
        cb._log_episode({"r": float(i), "l": 10, "hp": 80, "max_hp": 80, "floor": 1})
    kwargs = writer.write_training_step.call_args.kwargs
    assert len(kwargs["episodes"]) == 500
    assert kwargs["total_episodes"] == 505
    assert kwargs["episodes"][-1]["reward"] == 504.0


def test_callback_tracks_total_episodes_across_calls():
    writer = MagicMock()
    cb = EpisodeLoggerCallback(summary_freq=100, live_state_writer=writer)
    cb.num_timesteps = 0
    for _ in range(3):
        cb._log_episode({"r": 1.0, "l": 10, "hp": 80, "max_hp": 80, "floor": 1})
    kwargs = writer.write_training_step.call_args.kwargs
    assert kwargs["total_episodes"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_callbacks.py -v
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'live_state_writer'`

- [ ] **Step 3: Update `src/callbacks.py`**

Replace the entire file with:

```python
# src/callbacks.py
import logging
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)

_TRAINING_WINDOW = 500


class EpisodeLoggerCallback(BaseCallback):
    """Logs per-episode stats and periodic summaries to the game log."""

    def __init__(self, summary_freq: int = 100, live_state_writer=None):
        super().__init__(verbose=0)
        self.summary_freq = summary_freq
        self.live_state_writer = live_state_writer
        self._episode_count = 0
        self._episode_stats: deque = deque(maxlen=self.summary_freq)
        self._training_episodes: deque = deque(maxlen=_TRAINING_WINDOW)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done and "episode" in info:
                self._log_episode(info["episode"])
        return True

    def _log_episode(self, ep: dict):
        self._episode_count += 1
        self._episode_stats.append(ep)
        self._training_episodes.append({
            "ep": self._episode_count,
            "reward": ep["r"],
            "steps": ep["l"],
        })
        if self.live_state_writer is not None:
            self.live_state_writer.write_training_step(
                episodes=list(self._training_episodes),
                total_episodes=self._episode_count,
                total_timesteps=self.num_timesteps,
            )
        logger.info(
            "[Episode %d] reward=%.2f | steps=%d | hp=%d/%d | floor=%d | total_steps=%d",
            self._episode_count, ep["r"], ep["l"],
            ep.get("hp", 0), ep.get("max_hp", 0), ep.get("floor", 0),
            self.num_timesteps,
        )
        if self._episode_count % self.summary_freq == 0:
            recent = list(self._episode_stats)
            avg_r = sum(e["r"] for e in recent) / len(recent)
            avg_l = sum(e["l"] for e in recent) / len(recent)
            win_rate = sum(1 for e in recent if e["r"] > 0) / len(recent)
            avg_floor = sum(e.get("floor", 0) for e in recent) / len(recent)
            logger.info(
                "[Summary ep %d-%d] avg_reward=%.2f | avg_steps=%.1f | win_rate=%.1f%% | avg_floor=%.1f",
                self._episode_count - self.summary_freq + 1, self._episode_count,
                avg_r, avg_l, win_rate * 100, avg_floor,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_callbacks.py -v
```

Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/callbacks.py tests/test_callbacks.py
git commit -m "feat: capture per-episode training data in EpisodeLoggerCallback"
```

---

## Task 3: Wire `live_writer` into `EpisodeLoggerCallback` in `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update the `EpisodeLoggerCallback` instantiation in `main.py`**

Find this line (around line 100):
```python
EpisodeLoggerCallback(summary_freq=100),
```

Replace it with:
```python
EpisodeLoggerCallback(summary_freq=100, live_state_writer=live_writer),
```

- [ ] **Step 2: Run the full test suite to confirm nothing regressed**

```
pytest tests/ -v
```

Expected: all existing tests PASS

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: wire live_writer into EpisodeLoggerCallback"
```

---

## Task 4: `/training` route in `dashboard.py`

**Files:**
- Modify: `dashboard.py`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Write a failing test**

Append to `tests/test_dashboard.py`:

```python
def test_training_route_returns_html(client):
    c, _ = client
    resp = c.get("/training")
    assert resp.status_code == 200
    assert b"text/html" in resp.content_type.encode()
    assert b"TRAINING" in resp.data
    assert b"chart.js" in resp.data.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_dashboard.py::test_training_route_returns_html -v
```

Expected: FAIL with `404`

- [ ] **Step 3: Add `_TRAINING_HTML` and the `/training` route to `dashboard.py`**

After the closing triple-quote of `_STATS_HTML`, add:

```python
_TRAINING_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Training Progress</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #14142a; font-family: monospace; color: #e0e0e0; padding: 16px; }
  .panel {
    background: #1a1a2e;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 16px 18px;
    max-width: 820px;
  }
  .title {
    color: #a29bfe;
    font-weight: bold;
    font-size: 13px;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
  }
  .stats-row { display: flex; gap: 10px; margin-bottom: 14px; }
  .stat {
    background: #14142a;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 6px;
    padding: 8px 12px;
    flex: 1;
  }
  .stat-label { color: #a8a8b3; font-size: 10px; letter-spacing: 0.05em; }
  .stat-value { font-size: 14px; font-weight: bold; margin-top: 2px; }
  .no-data { color: #555; font-size: 11px; text-align: center; padding: 40px 0; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<div class="panel">
  <div class="title">TRAINING</div>
  <div class="stats-row">
    <div class="stat">
      <div class="stat-label">EPISODES</div>
      <div class="stat-value" id="total-episodes" style="color:#ffd700">—</div>
    </div>
    <div class="stat">
      <div class="stat-label">TIMESTEPS</div>
      <div class="stat-value" id="total-timesteps" style="color:#74b9ff">—</div>
    </div>
    <div class="stat">
      <div class="stat-label">AVG REWARD (100)</div>
      <div class="stat-value" id="avg-reward" style="color:#7bed9f">—</div>
    </div>
    <div class="stat">
      <div class="stat-label">BEST EPISODE</div>
      <div class="stat-value" id="best-reward" style="color:#a29bfe">—</div>
    </div>
  </div>
  <canvas id="reward-chart" height="220" style="display:none"></canvas>
  <div class="no-data" id="no-data">No training data yet — start the agent with --rl</div>
</div>
<script>
const ROLL_WINDOW = 50;
let chart = null;

function rollingAvg(values) {
  return values.map((_, i) => {
    const slice = values.slice(Math.max(0, i - ROLL_WINDOW + 1), i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

function initChart(labels, rewards, avgs) {
  const ctx = document.getElementById("reward-chart").getContext("2d");
  chart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Episode reward",
          data: labels.map((x, i) => ({ x, y: rewards[i] })),
          backgroundColor: "rgba(74, 144, 217, 0.35)",
          pointRadius: 2,
          showLine: false,
        },
        {
          label: "Rolling avg (w=50)",
          data: labels.map((x, i) => ({ x, y: avgs[i] })),
          borderColor: "#f39c12",
          borderWidth: 2,
          pointRadius: 0,
          showLine: true,
          tension: 0.3,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      plugins: {
        legend: {
          labels: { color: "#a8a8b3", font: { family: "monospace", size: 11 } },
        },
      },
      scales: {
        x: {
          ticks: { color: "#555", font: { family: "monospace", size: 10 } },
          grid: { color: "rgba(255,255,255,0.06)" },
        },
        y: {
          ticks: { color: "#555", font: { family: "monospace", size: 10 } },
          grid: { color: "rgba(255,255,255,0.06)" },
        },
      },
    },
  });
}

function updateChart(labels, rewards, avgs) {
  chart.data.datasets[0].data = labels.map((x, i) => ({ x, y: rewards[i] }));
  chart.data.datasets[1].data = labels.map((x, i) => ({ x, y: avgs[i] }));
  chart.update("none");
}

async function poll() {
  try {
    const resp = await fetch("/api/state");
    const data = await resp.json();
    const t = data.training;
    const noData = document.getElementById("no-data");
    const canvas = document.getElementById("reward-chart");
    if (!t || !t.episodes || t.episodes.length === 0) {
      noData.style.display = "";
      canvas.style.display = "none";
      return;
    }
    noData.style.display = "none";
    canvas.style.display = "";
    document.getElementById("total-episodes").textContent =
      t.total_episodes.toLocaleString();
    document.getElementById("total-timesteps").textContent =
      t.total_timesteps.toLocaleString();
    const rewards = t.episodes.map(e => e.reward);
    const labels = t.episodes.map(e => e.ep);
    const recent = rewards.slice(-100);
    const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const best = Math.max(...rewards);
    document.getElementById("avg-reward").textContent =
      (avg >= 0 ? "+" : "") + avg.toFixed(2);
    document.getElementById("best-reward").textContent =
      (best >= 0 ? "+" : "") + best.toFixed(2);
    const avgs = rollingAvg(rewards);
    if (!chart) {
      initChart(labels, rewards, avgs);
    } else {
      updateChart(labels, rewards, avgs);
    }
  } catch (e) {
    console.error("poll error:", e);
  }
}

setInterval(poll, 3000);
poll();
</script>
</body>
</html>"""
```

Then add the route after the existing `@app.route("/stats")` handler:

```python
@app.route("/training")
def training():
    return Response(_TRAINING_HTML, status=200, mimetype="text/html")
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_dashboard.py::test_training_route_returns_html -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add dashboard.py tests/test_dashboard.py
git commit -m "feat: add /training route with live reward chart"
```
