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
