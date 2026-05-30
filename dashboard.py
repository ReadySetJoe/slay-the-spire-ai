import json
import os
import sys
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
    background: #14142a;
    font-family: monospace;
    color: #e0e0e0;
  }
  .panel {
    background: #1a1a2e;
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
  <div id="debug-line" style="margin-top:10px;font-size:9px;color:#555;"></div>
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
    const dbg = document.getElementById("debug-line");
    if (live.updated_at) {
      const secs = Math.round((Date.now() - new Date(live.updated_at)) / 1000);
      dbg.textContent = `updated ${secs}s ago · ${data._path || ""}`;
    } else {
      dbg.textContent = `no data · ${data._path || ""}`;
    }
  } catch (e) {
    document.getElementById("debug-line").textContent = "fetch error: " + e;
  }
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
  body { background: #14142a; font-family: monospace; color: #e0e0e0; }
  .panel {
    background: #1a1a2e;
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


@app.route("/live")
def live():
    return Response(_LIVE_HTML, status=200, mimetype="text/html")


@app.route("/stats")
def stats():
    return Response(_STATS_HTML, status=200, mimetype="text/html")


_GAME_DATA_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\SlayTheSpire\data"

if __name__ == "__main__":
    data_dir = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--data-dir" and i < len(sys.argv) - 1:
            data_dir = sys.argv[i + 1]
        elif arg.startswith("--data-dir="):
            data_dir = arg.split("=", 1)[1]

    if data_dir is None:
        # Auto-detect: prefer the game install directory, fall back to local
        if os.path.isdir(_GAME_DATA_DIR):
            data_dir = _GAME_DATA_DIR
        else:
            data_dir = "data"

    LIVE_STATE_PATH = os.path.join(data_dir, "live_state.json")
    print(f"Reading live state from: {os.path.abspath(LIVE_STATE_PATH)}")
    print("Live view:  http://localhost:5000/live")
    print("Stats view: http://localhost:5000/stats")
    print("Debug info: http://localhost:5000/api/debug")
    app.run(host="0.0.0.0", port=5000, debug=False)
