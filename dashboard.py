import json
import os
import sys
from flask import Flask, Response

app = Flask(__name__)
LIVE_STATE_PATH = "data/live_state.json"
RUN_LOG_PATH = "data/run_log.jsonl"

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
    padding: 22px 26px;
    width: 420px;
  }
  .title {
    color: #a29bfe;
    font-weight: bold;
    font-size: 20px;
    letter-spacing: 0.05em;
    margin-bottom: 16px;
  }
  .grid { display: grid; grid-template-columns: 1fr 1fr; row-gap: 12px; column-gap: 16px; font-size: 18px; }
  .label { color: #a8a8b3; }
  .val-gold { color: #ffd700; }
  .val-green { color: #7bed9f; }
  .val-red { color: #ff4757; }
  .val-blue { color: #74b9ff; }
  .divider { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 16px 0; }
  .deck-title { color: #a29bfe; font-weight: bold; font-size: 16px; letter-spacing: 0.05em; margin-bottom: 10px; }
  .deck-list {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px 10px;
    font-size: 15px;
    max-height: 240px;
    overflow-y: auto;
  }
  .deck-entry { color: #c8c8d8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .deck-count { color: #a8a8b3; }
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

  } catch (e) {
    console.error("poll error:", e);
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
<title>Floor Progress</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #14142a; font-family: monospace; color: #e0e0e0; padding: 20px; }
  .panel {
    background: #1a1a2e;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 22px 26px;
    max-width: 960px;
  }
  .title {
    color: #a29bfe;
    font-weight: bold;
    font-size: 48px;
    letter-spacing: 0.05em;
    margin-bottom: 16px;
  }
  .no-data { color: #555; font-size: 20px; text-align: center; padding: 60px 0; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<div class="panel">
  <div class="title">FLOOR PROGRESS</div>
  <canvas id="floor-chart" height="260" style="display:none"></canvas>
  <div class="no-data" id="no-data">No run data yet</div>
</div>
<script>
const ROLL = 20;
let chart = null;

function rollingAvg(values) {
  return values.map((_, i) => {
    const slice = values.slice(Math.max(0, i - ROLL + 1), i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

function initChart(runs, floors, avgs, colors) {
  const ctx = document.getElementById("floor-chart").getContext("2d");
  chart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "floor reached",
          data: runs.map((x, i) => ({ x, y: floors[i] })),
          backgroundColor: colors,
          pointRadius: 4,
          showLine: false,
        },
        {
          label: "rolling avg (w=20)",
          data: runs.map((x, i) => ({ x, y: avgs[i] })),
          borderColor: "#f39c12",
          borderWidth: 2.5,
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
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ctx.datasetIndex === 0
              ? `run ${ctx.parsed.x}  floor ${ctx.parsed.y}`
              : `avg ${ctx.parsed.y.toFixed(1)}`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "run", color: "#c0c0c0", font: { family: "monospace", size: 28 } },
          ticks: { color: "#c0c0c0", font: { family: "monospace", size: 24 } },
          grid: { color: "rgba(255,255,255,0.06)" },
        },
        y: {
          title: { display: true, text: "floor", color: "#c0c0c0", font: { family: "monospace", size: 28 } },
          min: 0,
          ticks: { color: "#c0c0c0", font: { family: "monospace", size: 24 }, stepSize: 5 },
          grid: { color: "rgba(255,255,255,0.06)" },
        },
      },
    },
  });
}

function updateChart(runs, floors, avgs, colors) {
  chart.data.datasets[0].data = runs.map((x, i) => ({ x, y: floors[i] }));
  chart.data.datasets[0].backgroundColor = colors;
  chart.data.datasets[1].data = runs.map((x, i) => ({ x, y: avgs[i] }));
  chart.update("none");
}

async function poll() {
  try {
    const resp = await fetch("/api/runs");
    const data = await resp.json();
    const noData = document.getElementById("no-data");
    const canvas = document.getElementById("floor-chart");

    if (!data.runs || data.runs.length === 0) {
      noData.style.display = "";
      canvas.style.display = "none";
      if (chart) { chart.destroy(); chart = null; }
      return;
    }

    noData.style.display = "none";
    canvas.style.display = "";

    const runs   = data.runs.map(r => r.run_number);
    const floors = data.runs.map(r => r.floor_reached);
    const colors = data.runs.map(r =>
      r.result === "win" ? "rgba(255,215,0,0.8)" : "rgba(74,144,217,0.5)"
    );
    const avgs = rollingAvg(floors);

    if (!chart) {
      initChart(runs, floors, avgs, colors);
    } else {
      updateChart(runs, floors, avgs, colors);
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


@app.route("/api/runs")
def api_runs():
    abs_path = os.path.abspath(RUN_LOG_PATH)
    if not os.path.exists(abs_path):
        return Response(json.dumps({"runs": []}), status=200, mimetype="application/json")
    runs = []
    try:
        with open(abs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                runs.append({
                    "run_number":   record.get("run_number"),
                    "floor_reached": record.get("floor_reached", 0),
                    "result":        record.get("result", "loss"),
                })
    except (OSError, json.JSONDecodeError):
        pass
    return Response(json.dumps({"runs": runs}), status=200, mimetype="application/json")


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
    RUN_LOG_PATH = os.path.join(data_dir, "run_log.jsonl")
    print(f"Reading stats from: {os.path.abspath(LIVE_STATE_PATH)}")
    print("Stats panel: http://localhost:5000/stats")
    print("Training:    http://localhost:5000/training")
    print("Debug info:  http://localhost:5000/api/debug")
    app.run(host="0.0.0.0", port=5000, debug=False)
