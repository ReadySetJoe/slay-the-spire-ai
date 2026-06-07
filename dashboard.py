import json
import os
import sys
from flask import Flask, Response

app = Flask(__name__)
LIVE_STATE_PATH = "data/live_state.json"

_STATS_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #14142a; font-family: monospace; color: #e0e0e0; }
  .panel {
    background: #1a1a2e;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 16px 18px;
    width: 280px;
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
  .divider { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 12px 0; }
  .deck-title { color: #a29bfe; font-weight: bold; font-size: 11px; letter-spacing: 0.05em; margin-bottom: 8px; }
  .deck-list {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3px 8px;
    font-size: 11px;
    max-height: 160px;
    overflow-y: auto;
  }
  .deck-entry { color: #c8c8d8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .deck-count { color: #a8a8b3; }
  .debug { margin-top: 10px; font-size: 10px; color: #666; word-break: break-all; }
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
  <hr class="divider">
  <div class="deck-title">DECK (<span id="deck-size">—</span>)</div>
  <div class="deck-list" id="deck-list"></div>
  <div class="debug" id="debug-info"></div>
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

    const live = data.live || null;
    const deck = (live || {}).deck || [];
    const total = deck.reduce((sum, e) => sum + e.count, 0);
    document.getElementById("deck-size").textContent = total || "—";
    const list = document.getElementById("deck-list");
    list.innerHTML = deck.map(e =>
      `<div class="deck-entry">${e.count > 1 ? '<span class="deck-count">' + e.count + '×</span> ' : ''}${e.name}</div>`
    ).join("");

    const dbg = document.getElementById("debug-info");
    dbg.textContent = live
      ? `live.deck: ${deck.length} entries | screen: ${live.screen_type || "?"}`
      : `no 'live' key in JSON (keys: ${Object.keys(data).join(", ")})`;
  } catch (e) {
    document.getElementById("debug-info").textContent = "fetch error: " + e;
  }
}
setInterval(poll, 2500);
poll();
</script>
</body>
</html>"""


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
      document.getElementById("total-episodes").textContent = "—";
      document.getElementById("total-timesteps").textContent = "—";
      document.getElementById("avg-reward").textContent = "—";
      document.getElementById("best-reward").textContent = "—";
      if (chart) {
        chart.destroy();
        chart = null;
      }
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


@app.route("/api/state")
def api_state():
    abs_path = os.path.abspath(LIVE_STATE_PATH)
    if not os.path.exists(abs_path):
        return Response(
            json.dumps({"_path": abs_path, "_exists": False}),
            status=200, mimetype="application/json",
        )
    try:
        with open(abs_path) as f:
            data = json.load(f)
        data["_path"] = abs_path
        data["_exists"] = True
        return Response(json.dumps(data), status=200, mimetype="application/json")
    except (OSError, json.JSONDecodeError) as e:
        return Response(
            json.dumps({"_path": abs_path, "_exists": True, "_error": str(e)}),
            status=200, mimetype="application/json",
        )


@app.route("/api/debug")
def api_debug():
    abs_path = os.path.abspath(LIVE_STATE_PATH)
    exists = os.path.exists(abs_path)
    info = {
        "live_state_path": abs_path,
        "file_exists": exists,
        "cwd": os.getcwd(),
    }
    if exists:
        info["mtime"] = os.path.getmtime(abs_path)
        info["size_bytes"] = os.path.getsize(abs_path)
    return Response(json.dumps(info, indent=2), status=200, mimetype="application/json")


@app.route("/stats")
def stats():
    return Response(_STATS_HTML, status=200, mimetype="text/html")


@app.route("/training")
def training():
    return Response(_TRAINING_HTML, status=200, mimetype="text/html")


_GAME_DATA_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\SlayTheSpire\data"

if __name__ == "__main__":
    data_dir = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--data-dir" and i < len(sys.argv) - 1:
            data_dir = sys.argv[i + 1]
        elif arg.startswith("--data-dir="):
            data_dir = arg.split("=", 1)[1]

    if data_dir is None:
        if os.path.isdir(_GAME_DATA_DIR):
            data_dir = _GAME_DATA_DIR
        else:
            data_dir = "data"

    LIVE_STATE_PATH = os.path.join(data_dir, "live_state.json")
    print(f"Reading stats from: {os.path.abspath(LIVE_STATE_PATH)}")
    print("Stats panel: http://localhost:5000/stats")
    print("Debug info:  http://localhost:5000/api/debug")
    app.run(host="0.0.0.0", port=5000, debug=False)
