import json
import os
import sys
from flask import Flask, Response

app = Flask(__name__)
LIVE_STATE_PATH = "data/live_state.json"
RUN_LOG_PATH = "data/run_log.jsonl"

# ── shared CSS injected into every page ──────────────────────────────────────

_CSS = """
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #14142a; font-family: monospace; color: #e0e0e0; }
  .panel {
    background: #1a1a2e;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 22px 26px;
  }
  .title {
    color: #a29bfe;
    font-weight: bold;
    font-size: 20px;
    letter-spacing: 0.05em;
    margin-bottom: 16px;
  }
  .chips { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
  .chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 7px 13px;
    min-width: 64px;
    text-align: center;
  }
  .chip-label { font-size: 10px; letter-spacing: 0.12em; color: #a8a8b3; margin-bottom: 3px; }
  .chip-value { font-size: 20px; font-weight: bold; color: #e0e0e0; }
  .divider { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 14px 0; }
  .muted { color: #a8a8b3; }
  .gold  { color: #ffd700; }
  .green { color: #7bed9f; }
  .red   { color: #ff4757; }
  .blue  { color: #74b9ff; }
  .orange{ color: #f39c12; }
  .no-data { color: #555; font-size: 18px; text-align: center; padding: 60px 0; }
"""

_CHARTJS = '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>'

# ── /stats ────────────────────────────────────────────────────────────────────

_STATS_HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
{_CSS}
.outer {{ width: 480px; }}
.versions {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 4px; }}
.v-block {{ background: rgba(255,255,255,0.03); border-radius: 6px; padding: 12px 14px; }}
.v-label {{ font-size: 11px; letter-spacing: 0.12em; color: #a29bfe; margin-bottom: 8px; font-weight: bold; }}
.stat-row {{ display: flex; justify-content: space-between; font-size: 16px; margin-bottom: 5px; }}
.deck-title {{ color: #a29bfe; font-weight: bold; font-size: 14px; letter-spacing: 0.05em; margin-bottom: 8px; }}
.deck-list {{
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 3px 10px; font-size: 14px;
  max-height: 220px; overflow-y: auto;
}}
.deck-entry {{ color: #c8c8d8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
</style></head>
<body><div class="panel outer">
  <div class="title">RUN STATS</div>
  <div class="chips">
    <div class="chip">
      <div class="chip-label">RUNS</div>
      <div class="chip-value gold" id="run-number">—</div>
    </div>
    <div class="chip">
      <div class="chip-label">WIN %</div>
      <div class="chip-value green" id="win-rate">—</div>
    </div>
    <div class="chip">
      <div class="chip-label">WINS</div>
      <div class="chip-value green" id="wins-count">—</div>
    </div>
    <div class="chip">
      <div class="chip-label">LOSSES</div>
      <div class="chip-value red" id="losses-count">—</div>
    </div>
    <div class="chip">
      <div class="chip-label">HUNG</div>
      <div class="chip-value orange" id="hung-count">—</div>
    </div>
  </div>
  <div class="versions">
    <div class="v-block">
      <div class="v-label">V1 · CLASSIC</div>
      <div class="stat-row"><span class="muted">Runs</span>  <span class="gold"  id="v1-runs">—</span></div>
      <div class="stat-row"><span class="muted">Win %</span> <span class="green" id="v1-wr">—</span></div>
      <div class="stat-row"><span class="muted">Avg fl</span><span class="blue"  id="v1-fl">—</span></div>
      <div class="stat-row"><span class="muted">W / L</span>
        <span><span class="green" id="v1-w">—</span> / <span class="red" id="v1-l">—</span></span></div>
    </div>
    <div class="v-block">
      <div class="v-label">V2 · FULL RUN</div>
      <div class="stat-row"><span class="muted">Runs</span>  <span class="gold"  id="v2-runs">—</span></div>
      <div class="stat-row"><span class="muted">Win %</span> <span class="green" id="v2-wr">—</span></div>
      <div class="stat-row"><span class="muted">Avg fl</span><span class="blue"  id="v2-fl">—</span></div>
      <div class="stat-row"><span class="muted">W / L</span>
        <span><span class="green" id="v2-w">—</span> / <span class="red" id="v2-l">—</span></span></div>
    </div>
  </div>
  <hr class="divider">
  <div class="deck-title">CURRENT DECK (<span id="deck-size">—</span>)</div>
  <div class="deck-list" id="deck-list"></div>
</div>
<script>
function stats(runs, ver) {{
  const v = runs.filter(r => (r.version || "v1") === ver);
  const wins = v.filter(r => r.result === "win").length;
  const losses = v.filter(r => r.result === "loss").length;
  const avgFl = v.length ? (v.reduce((s,r) => s + r.floor_reached, 0) / v.length) : 0;
  return {{ runs: v.length, wins, losses, wr: v.length ? wins/v.length : 0, avgFl }};
}}
async function poll() {{
  try {{
    const [rResp, sResp] = await Promise.all([fetch("/api/runs"), fetch("/api/state")]);
    const rData = await rResp.json();
    const sData = await sResp.json();
    const runs = rData.runs || [];
    const v1 = stats(runs, "v1"), v2 = stats(runs, "v2");
    // overall chips
    const allWins = v1.wins + v2.wins;
    const allLoss = v1.losses + v2.losses;
    const allRuns = v1.runs + v2.runs;
    const allWr   = allRuns ? allWins / allRuns : 0;
    document.getElementById("run-number").textContent   = allRuns || "—";
    document.getElementById("win-rate").textContent     = allRuns ? Math.round(allWr*100)+"%" : "—";
    document.getElementById("wins-count").textContent   = allWins;
    document.getElementById("losses-count").textContent = allLoss;
    document.getElementById("hung-count").textContent   = sData.stats?.hung ?? "—";
    // per-version rows
    document.getElementById("v1-runs").textContent = v1.runs || "—";
    document.getElementById("v1-wr").textContent   = v1.runs ? Math.round(v1.wr*100)+"%" : "—";
    document.getElementById("v1-fl").textContent   = v1.runs ? v1.avgFl.toFixed(1) : "—";
    document.getElementById("v1-w").textContent    = v1.wins;
    document.getElementById("v1-l").textContent    = v1.losses;
    document.getElementById("v2-runs").textContent = v2.runs || "—";
    document.getElementById("v2-wr").textContent   = v2.runs ? Math.round(v2.wr*100)+"%" : "—";
    document.getElementById("v2-fl").textContent   = v2.runs ? v2.avgFl.toFixed(1) : "—";
    document.getElementById("v2-w").textContent    = v2.wins;
    document.getElementById("v2-l").textContent    = v2.losses;
    const deck = ((sData.live || {{}}).deck || []);
    const total = deck.reduce((s,e) => s + e.count, 0);
    document.getElementById("deck-size").textContent = total || "—";
    document.getElementById("deck-list").innerHTML = deck.map(e =>
      `<div class="deck-entry">${{e.count > 1 ? '<span class="muted">'+e.count+'×</span> ' : ''}}${{e.name}}</div>`
    ).join("");
  }} catch(e) {{ console.error(e); }}
}}
setInterval(poll, 2500); poll();
</script></body></html>"""

# ── /training ─────────────────────────────────────────────────────────────────

_TRAINING_HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">{_CHARTJS}<style>
{_CSS}
body {{ padding: 18px; }}
.panel {{ max-width: 1000px; }}
.header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 14px; }}
.legend {{ display: flex; gap: 20px; font-size: 13px; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
</style></head>
<body><div class="panel">
  <div class="header">
    <div class="title">FLOOR PROGRESS</div>
    <div class="legend">
      <span><span class="dot" style="background:#888"></span><span class="muted">v1</span></span>
      <span><span class="dot" style="background:#f39c12"></span><span class="muted">v1 avg</span></span>
      <span><span class="dot" style="background:#74b9ff"></span><span class="muted">v2</span></span>
      <span><span class="dot" style="background:#7bed9f"></span><span class="muted">v2 avg</span></span>
    </div>
  </div>
  <canvas id="chart" height="260" style="display:none"></canvas>
  <div class="no-data" id="no-data">No run data yet</div>
</div>
<script>
const ROLL = 20;
let chart = null;
function rolling(arr) {{
  return arr.map((_,i) => {{
    const sl = arr.slice(Math.max(0,i-ROLL+1), i+1);
    return sl.reduce((a,b)=>a+b,0)/sl.length;
  }});
}}
async function poll() {{
  try {{
    const data = await (await fetch("/api/runs")).json();
    const runs = data.runs || [];
    const canvas = document.getElementById("chart");
    const noData = document.getElementById("no-data");
    if (!runs.length) {{ noData.style.display=""; canvas.style.display="none"; if(chart){{chart.destroy();chart=null;}} return; }}
    noData.style.display = "none"; canvas.style.display = "";
    const v1 = runs.filter(r=>(r.version||"v1")==="v1");
    const v2 = runs.filter(r=>(r.version||"v1")==="v2");
    const v2Max = v2.length ? Math.max(...v2.map(r=>r.run_number)) : 0;
    const xMax = v2.length + 20;
    const v1Dots = v1.map(r=>({{ x:r.run_number, y:r.floor_reached }}));
    const v2Dots = v2.map(r=>({{ x:r.run_number, y:r.floor_reached }}));
    const v1Colors = v1.map(r=>r.result==="win"?"rgba(255,215,0,0.85)":"rgba(130,130,160,0.55)");
    const v2Colors = v2.map(r=>r.result==="win"?"rgba(255,215,0,0.9)":"rgba(74,144,217,0.65)");
    const v1Avg = rolling(v1.map(r=>r.floor_reached));
    const v2Avg = rolling(v2.map(r=>r.floor_reached));
    const datasets = [
      {{ label:"v1", data:v1Dots, backgroundColor:v1Colors, pointRadius:4, showLine:false }},
      {{ label:"v1 avg", data:v1.map((r,i)=>({{x:r.run_number,y:v1Avg[i]}})),
         borderColor:"#f39c12", borderWidth:2, pointRadius:0, showLine:true, tension:0.3 }},
      {{ label:"v2", data:v2Dots, backgroundColor:v2Colors, pointRadius:5, showLine:false }},
      {{ label:"v2 avg", data:v2.map((r,i)=>({{x:r.run_number,y:v2Avg[i]}})),
         borderColor:"#7bed9f", borderWidth:2.5, pointRadius:0, showLine:true, tension:0.3 }},
    ];
    const opts = {{
      animation:false, responsive:true,
      plugins:{{ legend:{{display:false}}, tooltip:{{ callbacks:{{ label: c =>
        c.datasetIndex%2===0 ? `run ${{c.parsed.x}}  fl ${{c.parsed.y}}`:`avg ${{c.parsed.y.toFixed(1)}}`
      }}}}  }},
      scales:{{
        x:{{ min:1, max:xMax, title:{{display:true,text:"run",color:"#c0c0c0",font:{{family:"monospace",size:22}}}},
             ticks:{{color:"#c0c0c0",font:{{family:"monospace",size:18}}}}, grid:{{color:"rgba(255,255,255,0.06)"}} }},
        y:{{ min:0, max:55, title:{{display:true,text:"floor",color:"#c0c0c0",font:{{family:"monospace",size:22}}}},
             ticks:{{color:"#c0c0c0",font:{{family:"monospace",size:18}},stepSize:5}}, grid:{{color:"rgba(255,255,255,0.06)"}} }},
      }},
    }};
    if (!chart) {{
      chart = new Chart(document.getElementById("chart").getContext("2d"), {{type:"scatter",data:{{datasets}},options:opts}});
    }} else {{
      chart.data.datasets = datasets;
      chart.options.scales.x.max = xMax;
      chart.update("none");
    }}
  }} catch(e) {{ console.error(e); }}
}}
setInterval(poll, 3000); poll();
</script></body></html>"""

# ── /reward ───────────────────────────────────────────────────────────────────

_REWARD_HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">{_CHARTJS}<style>
{_CSS}
body {{ padding: 18px; }}
.panel {{ max-width: 700px; }}
.header {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:14px; }}
.sub {{ font-size: 15px; }}
</style></head>
<body><div class="panel">
  <div class="header">
    <div class="title">V2 EPISODE REWARD</div>
    <div class="sub muted">avg (last 20): <span class="green" id="avg-reward">—</span></div>
  </div>
  <canvas id="chart" height="220" style="display:none"></canvas>
  <div class="no-data" id="no-data">No v2 run data yet</div>
</div>
<script>
const ROLL = 20;
let chart = null;
function rolling(arr) {{
  return arr.map((_,i) => {{
    const sl = arr.slice(Math.max(0,i-ROLL+1), i+1);
    return sl.reduce((a,b)=>a+b,0)/sl.length;
  }});
}}
async function poll() {{
  try {{
    const data = await (await fetch("/api/state")).json();
    const rewards = (data.v2 || {{}}).episode_rewards || [];
    const canvas = document.getElementById("chart");
    const noData = document.getElementById("no-data");
    if (!rewards.length) {{ noData.style.display=""; canvas.style.display="none"; if(chart){{chart.destroy();chart=null;}} return; }}
    noData.style.display = "none"; canvas.style.display = "";
    const recent = rewards.slice(-20);
    const avg = recent.reduce((a,b)=>a+b,0)/recent.length;
    document.getElementById("avg-reward").textContent = avg.toFixed(3);
    const labels = rewards.map((_,i)=>i+1);
    const avgs = rolling(rewards);
    const datasets = [
      {{ label:"reward", data:rewards, borderColor:"rgba(74,144,217,0.6)", backgroundColor:"rgba(74,144,217,0.15)",
         pointRadius:2, borderWidth:1.5, fill:true, tension:0.2 }},
      {{ label:"rolling avg", data:avgs, borderColor:"#7bed9f", borderWidth:2.5,
         pointRadius:0, tension:0.3, fill:false }},
    ];
    const opts = {{
      animation:false, responsive:true,
      plugins:{{ legend:{{display:false}} }},
      scales:{{
        x:{{ labels, title:{{display:true,text:"run",color:"#c0c0c0",font:{{family:"monospace",size:18}}}},
             ticks:{{color:"#c0c0c0",font:{{family:"monospace",size:14}}}}, grid:{{color:"rgba(255,255,255,0.06)"}} }},
        y:{{ title:{{display:true,text:"reward",color:"#c0c0c0",font:{{family:"monospace",size:18}}}},
             ticks:{{color:"#c0c0c0",font:{{family:"monospace",size:14}}}}, grid:{{color:"rgba(255,255,255,0.06)"}} }},
      }},
    }};
    if (!chart) {{
      chart = new Chart(document.getElementById("chart").getContext("2d"), {{type:"line",data:{{labels,datasets}},options:opts}});
    }} else {{
      chart.data.labels = labels;
      chart.data.datasets = datasets;
      chart.update("none");
    }}
  }} catch(e) {{ console.error(e); }}
}}
setInterval(poll, 3000); poll();
</script></body></html>"""

# ── /energy ───────────────────────────────────────────────────────────────────

_ENERGY_HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">{_CHARTJS}<style>
{_CSS}
body {{ padding: 18px; }}
.panel {{ width: 460px; }}
.gauge-wrap {{ display:flex; flex-direction:column; align-items:center; margin: 8px 0 16px; }}
.gauge-pct {{ font-size: 56px; font-weight: bold; margin-top: -8px; }}
.gauge-label {{ font-size: 13px; letter-spacing: 0.12em; color: #a8a8b3; margin-top: 4px; }}
.trend-label {{ font-size: 13px; color: #a8a8b3; margin-bottom: 6px; }}
</style></head>
<body><div class="panel">
  <div class="title">ENERGY EFFICIENCY</div>
  <div class="gauge-wrap">
    <svg id="gauge-svg" viewBox="-10 -10 220 120" width="240" xmlns="http://www.w3.org/2000/svg">
      <path d="M 0 100 A 100 100 0 0 1 200 100" fill="none" stroke="#2a2a4a" stroke-width="22" stroke-linecap="round"/>
      <path id="gauge-arc" d="M 0 100 A 100 100 0 0 1 200 100" fill="none" stroke="#7bed9f" stroke-width="22"
            stroke-linecap="round" stroke-dasharray="314 314" stroke-dashoffset="0"/>
    </svg>
    <div class="gauge-pct green" id="gauge-pct">—</div>
    <div class="gauge-label">avg energy used per turn (v2)</div>
  </div>
  <hr class="divider">
  <div class="trend-label">per-run trend</div>
  <canvas id="trend-chart" height="100" style="display:none"></canvas>
  <div class="no-data" id="no-data" style="padding:20px 0">No v2 data yet</div>
</div>
<script>
const ARC = Math.PI * 100;
let trendChart = null;
function effColor(v) {{
  return v > 0.8 ? "#7bed9f" : v > 0.5 ? "#f39c12" : "#ff4757";
}}
async function poll() {{
  try {{
    const data = await (await fetch("/api/state")).json();
    const hist = (data.v2 || {{}}).energy_efficiency || [];
    const noData = document.getElementById("no-data");
    const canvas = document.getElementById("trend-chart");
    if (!hist.length) {{
      noData.style.display=""; canvas.style.display="none";
      if(trendChart){{trendChart.destroy();trendChart=null;}}
      return;
    }}
    noData.style.display = "none"; canvas.style.display = "";
    const latest = hist[hist.length-1];
    const col = effColor(latest);
    document.getElementById("gauge-pct").textContent = Math.round(latest*100)+"%";
    document.getElementById("gauge-pct").style.color = col;
    const arc = document.getElementById("gauge-arc");
    arc.setAttribute("stroke", col);
    arc.setAttribute("stroke-dasharray", `${{ARC*latest}} ${{ARC}}`);
    const labels = hist.map((_,i)=>i+1);
    const tOpts = {{
      animation:false, responsive:true,
      plugins:{{ legend:{{display:false}} }},
      scales:{{
        x:{{ display:false }},
        y:{{ min:0, max:1, ticks:{{ color:"#a8a8b3", font:{{family:"monospace",size:12}},
              callback: v=>Math.round(v*100)+"%" }},
             grid:{{color:"rgba(255,255,255,0.06)"}} }},
      }},
    }};
    const datasets = [{{
      data: hist, borderColor: col, backgroundColor:"rgba(120,120,120,0.1)",
      borderWidth:2, pointRadius:2, fill:true, tension:0.3,
    }}];
    if (!trendChart) {{
      trendChart = new Chart(canvas.getContext("2d"), {{type:"line",data:{{labels,datasets}},options:tOpts}});
    }} else {{
      trendChart.data.labels = labels;
      trendChart.data.datasets = datasets;
      trendChart.update("none");
    }}
  }} catch(e) {{ console.error(e); }}
}}
setInterval(poll, 3000); poll();
</script></body></html>"""

# ── /deaths ───────────────────────────────────────────────────────────────────

_DEATHS_HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">{_CHARTJS}<style>
{_CSS}
body {{ padding: 18px; }}
.panel {{ max-width: 760px; }}
.header {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:14px; }}
.legend {{ display:flex; gap:18px; font-size:13px; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; }}
</style></head>
<body><div class="panel">
  <div class="header">
    <div class="title">FLOOR DEATH MAP</div>
    <div class="legend">
      <span><span class="dot" style="background:rgba(130,130,160,0.65)"></span><span class="muted">v1</span></span>
      <span><span class="dot" style="background:rgba(74,144,217,0.75)"></span><span class="muted">v2</span></span>
    </div>
  </div>
  <canvas id="chart" height="300" style="display:none"></canvas>
  <div class="no-data" id="no-data">No run data yet</div>
</div>
<script>
let chart = null;
async function poll() {{
  try {{
    const data = await (await fetch("/api/runs")).json();
    const runs = data.runs || [];
    const canvas = document.getElementById("chart");
    const noData = document.getElementById("no-data");
    if (!runs.length) {{ noData.style.display=""; canvas.style.display="none"; if(chart){{chart.destroy();chart=null;}} return; }}
    noData.style.display="none"; canvas.style.display="";
    // bucket by floor
    const v1Counts = {{}}, v2Counts = {{}};
    runs.forEach(r => {{
      const fl = r.floor_reached;
      const ver = r.version || "v1";
      if (ver==="v1") v1Counts[fl] = (v1Counts[fl]||0)+1;
      else v2Counts[fl] = (v2Counts[fl]||0)+1;
    }});
    const maxFl = Math.max(...runs.map(r=>r.floor_reached), 1);
    const floors = Array.from({{length:maxFl}},(_,i)=>i+1);
    const v1Data = floors.map(f=>v1Counts[f]||0);
    const v2Data = floors.map(f=>v2Counts[f]||0);
    const datasets = [
      {{ label:"v1", data:v1Data, backgroundColor:"rgba(130,130,160,0.65)", borderRadius:2 }},
      {{ label:"v2", data:v2Data, backgroundColor:"rgba(74,144,217,0.75)", borderRadius:2 }},
    ];
    const opts = {{
      animation:false, responsive:true,
      plugins:{{ legend:{{display:false}}, tooltip:{{ callbacks:{{
        label: c => `${{c.dataset.label}}: ${{c.parsed.y}} run${{c.parsed.y!==1?"s":""}}`
      }}}} }},
      scales:{{
        x:{{ stacked:true, title:{{display:true,text:"floor",color:"#c0c0c0",font:{{family:"monospace",size:18}}}},
             ticks:{{color:"#c0c0c0",font:{{family:"monospace",size:13}}}}, grid:{{color:"rgba(255,255,255,0.06)"}} }},
        y:{{ stacked:true, title:{{display:true,text:"deaths",color:"#c0c0c0",font:{{family:"monospace",size:18}}}},
             ticks:{{color:"#c0c0c0",font:{{family:"monospace",size:13}},precision:0}}, grid:{{color:"rgba(255,255,255,0.06)"}} }},
      }},
    }};
    if (!chart) {{
      chart = new Chart(canvas.getContext("2d"), {{type:"bar",data:{{labels:floors,datasets}},options:opts}});
    }} else {{
      chart.data.labels = floors;
      chart.data.datasets = datasets;
      chart.update("none");
    }}
  }} catch(e) {{ console.error(e); }}
}}
setInterval(poll, 4000); poll();
</script></body></html>"""

# ── /actions ──────────────────────────────────────────────────────────────────

_ACTIONS_HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">{_CHARTJS}<style>
{_CSS}
body {{ padding: 18px; }}
.panel {{ width: 380px; }}
.legend-list {{ margin-top: 14px; display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:14px; }}
.leg-row {{ display:flex; align-items:center; gap:8px; }}
.leg-dot {{ width:12px; height:12px; border-radius:3px; flex-shrink:0; }}
.leg-label {{ color:#a8a8b3; }}
.leg-val {{ margin-left:auto; color:#e0e0e0; }}
</style></head>
<body><div class="panel">
  <div class="title">LAST RUN · ACTION MIX</div>
  <canvas id="chart" height="240" style="display:none"></canvas>
  <div class="no-data" id="no-data">No v2 data yet</div>
  <div class="legend-list" id="legend"></div>
</div>
<script>
const COLORS = ["#74b9ff","#f39c12","#7bed9f","#a29bfe"];
const LABELS = ["Play Card","End Turn","Non-combat","Potion"];
const KEYS   = ["play","end","noncombat","potion"];
let chart = null;
async function poll() {{
  try {{
    const data = await (await fetch("/api/state")).json();
    const ac = (data.v2 || {{}}).action_counts_last_run || {{}};
    const canvas = document.getElementById("chart");
    const noData = document.getElementById("no-data");
    const total = KEYS.reduce((s,k)=>s+(ac[k]||0),0);
    if (!total) {{ noData.style.display=""; canvas.style.display="none"; if(chart){{chart.destroy();chart=null;}} return; }}
    noData.style.display="none"; canvas.style.display="";
    const values = KEYS.map(k=>ac[k]||0);
    const datasets = [{{ data:values, backgroundColor:COLORS, borderWidth:0, hoverOffset:4 }}];
    const opts = {{
      animation:false, responsive:true,
      plugins:{{ legend:{{display:false}}, tooltip:{{ callbacks:{{
        label: c => ` ${{c.label}}: ${{c.parsed}} (${{Math.round(c.parsed/total*100)}}%)`
      }}}}  }},
      cutout: "65%",
    }};
    if (!chart) {{
      chart = new Chart(canvas.getContext("2d"), {{type:"doughnut",data:{{labels:LABELS,datasets}},options:opts}});
    }} else {{
      chart.data.datasets[0].data = values;
      chart.update("none");
    }}
    document.getElementById("legend").innerHTML = LABELS.map((l,i) =>
      `<div class="leg-row">
        <div class="leg-dot" style="background:${{COLORS[i]}}"></div>
        <span class="leg-label">${{l}}</span>
        <span class="leg-val">${{values[i]}} <span class="muted" style="font-size:12px">(${{Math.round(values[i]/total*100)}}%)</span></span>
      </div>`
    ).join("");
  }} catch(e) {{ console.error(e); }}
}}
setInterval(poll, 3000); poll();
</script></body></html>"""

# ── /ticker ───────────────────────────────────────────────────────────────────

_TICKER_HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
{_CSS}
body {{ padding: 0; }}
.ticker {{
  background: #1a1a2e;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px;
  padding: 14px 20px;
  width: 720px;
}}
.top {{ display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px; }}
.location {{ font-size:20px; font-weight:bold; color:#a29bfe; letter-spacing:0.04em; }}
.screen-badge {{
  font-size:12px; letter-spacing:0.1em; color:#14142a;
  background:#a29bfe; border-radius:4px; padding:2px 8px;
}}
.hp-row {{ display:flex; align-items:center; gap:10px; margin-bottom:10px; }}
.hp-bar-wrap {{ flex:1; background:#2a2a4a; border-radius:4px; height:10px; overflow:hidden; }}
.hp-bar {{ height:100%; border-radius:4px; transition: width 0.3s; }}
.hp-text {{ font-size:14px; color:#a8a8b3; white-space:nowrap; }}
.action-line {{ font-size:18px; color:#e0e0e0; margin-bottom:8px; min-height:22px; }}
.enemies {{ font-size:14px; color:#a8a8b3; min-height:18px; }}
.enemy-name {{ color:#e0e0e0; }}
.enemy-hp {{ color:#74b9ff; }}
.intent {{ font-size:11px; color:#f39c12; }}
</style></head>
<body><div class="ticker">
  <div class="top">
    <div class="location" id="location">—</div>
    <div class="screen-badge" id="screen">—</div>
  </div>
  <div class="hp-row">
    <div class="hp-bar-wrap"><div class="hp-bar" id="hp-bar" style="width:0%;background:#7bed9f"></div></div>
    <div class="hp-text"><span id="hp">—</span> / <span id="max-hp">—</span></div>
  </div>
  <div class="action-line" id="last-action">—</div>
  <div class="enemies" id="enemies"></div>
</div>
<script>
async function poll() {{
  try {{
    const data = await (await fetch("/api/state")).json();
    const live = data.live || {{}};
    const hp = live.current_hp ?? 0;
    const maxHp = live.max_hp ?? 1;
    const pct = Math.round(hp / maxHp * 100);
    const hpColor = pct > 60 ? "#7bed9f" : pct > 30 ? "#f39c12" : "#ff4757";
    document.getElementById("hp").textContent = hp;
    document.getElementById("max-hp").textContent = maxHp;
    document.getElementById("hp-bar").style.width = pct + "%";
    document.getElementById("hp-bar").style.background = hpColor;
    const screen = live.screen_type || "—";
    document.getElementById("screen").textContent = screen;
    document.getElementById("last-action").textContent = live.last_action ? "→ " + live.last_action : "—";
    const monsters = live.monsters || [];
    document.getElementById("enemies").innerHTML = monsters.map(m =>
      `<span class="enemy-name">${{m.name}}</span> `+
      `<span class="enemy-hp">${{m.current_hp}}/${{m.max_hp}}</span> `+
      `<span class="intent">${{m.intent||""}}</span>  `
    ).join("  ·  ") || "";
    // location: try to extract floor from run summary
    const stats = data.stats || {{}};
    const runNum = stats.run_number ?? "—";
    document.getElementById("location").textContent = `Run #${{runNum}} · ${{screen}}`;
  }} catch(e) {{ console.error(e); }}
}}
setInterval(poll, 1000); poll();
</script></body></html>"""

# ── /cards ────────────────────────────────────────────────────────────────────

_CARDS_HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
{_CSS}
body {{ padding: 0; }}
.panel {{ width: 400px; }}
.pick-list {{ display:flex; flex-direction:column; gap:6px; max-height:420px; overflow-y:auto; }}
.pick-row {{
  display:flex; align-items:center; gap:10px;
  background:rgba(255,255,255,0.03); border-radius:6px; padding:7px 10px;
  font-size:15px;
}}
.tier-badge {{
  width:22px; height:22px; border-radius:4px; display:flex; align-items:center;
  justify-content:center; font-weight:bold; font-size:13px; flex-shrink:0;
}}
.card-name {{ flex:1; color:#e0e0e0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.run-num {{ color:#555; font-size:12px; flex-shrink:0; }}
</style></head>
<body><div class="panel">
  <div class="title">RECENT CARD PICKS</div>
  <div class="pick-list" id="pick-list"></div>
  <div class="no-data" id="no-data">No picks yet</div>
</div>
<script>
const TIER_STYLE = {{
  "S": "background:#ffd700;color:#14142a",
  "A": "background:#7bed9f;color:#14142a",
  "B": "background:#74b9ff;color:#14142a",
  "C": "background:#555;color:#e0e0e0",
  "D": "background:#ff4757;color:#fff",
}};
async function poll() {{
  try {{
    const data = await (await fetch("/api/state")).json();
    const picks = (data.v2 || {{}}).recent_card_picks || [];
    const list = document.getElementById("pick-list");
    const noData = document.getElementById("no-data");
    if (!picks.length) {{ noData.style.display=""; list.innerHTML=""; return; }}
    noData.style.display = "none";
    list.innerHTML = picks.map(p => {{
      const style = TIER_STYLE[p.tier] || TIER_STYLE["C"];
      return `<div class="pick-row">
        <div class="tier-badge" style="${{style}}">${{p.tier}}</div>
        <div class="card-name">${{p.name}}</div>
        <div class="run-num">#${{p.run||"—"}}</div>
      </div>`;
    }}).join("");
  }} catch(e) {{ console.error(e); }}
}}
setInterval(poll, 3000); poll();
</script></body></html>"""

# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/state")
def api_state():
    abs_path = os.path.abspath(LIVE_STATE_PATH)
    if not os.path.exists(abs_path):
        return Response(json.dumps({"_path": abs_path, "_exists": False}),
                        status=200, mimetype="application/json")
    try:
        with open(abs_path) as f:
            data = json.load(f)
        data["_path"] = abs_path
        data["_exists"] = True
        return Response(json.dumps(data), status=200, mimetype="application/json")
    except (OSError, json.JSONDecodeError) as e:
        return Response(json.dumps({"_path": abs_path, "_exists": True, "_error": str(e)}),
                        status=200, mimetype="application/json")


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
                r = json.loads(line)
                runs.append({
                    "run_number":       r.get("run_number"),
                    "floor_reached":    r.get("floor_reached", 0),
                    "result":           r.get("result", "loss"),
                    "version":          r.get("version", "v1"),
                    "episode_reward":   r.get("episode_reward"),
                    "energy_efficiency": r.get("energy_efficiency"),
                })
    except (OSError, json.JSONDecodeError):
        pass
    return Response(json.dumps({"runs": runs}), status=200, mimetype="application/json")


@app.route("/api/debug")
def api_debug():
    abs_path = os.path.abspath(LIVE_STATE_PATH)
    exists = os.path.exists(abs_path)
    info = {"live_state_path": abs_path, "file_exists": exists, "cwd": os.getcwd()}
    if exists:
        info["mtime"] = os.path.getmtime(abs_path)
        info["size_bytes"] = os.path.getsize(abs_path)
    return Response(json.dumps(info, indent=2), status=200, mimetype="application/json")


# ── View routes ───────────────────────────────────────────────────────────────

@app.route("/stats")
def stats():
    return Response(_STATS_HTML, status=200, mimetype="text/html")

@app.route("/training")
def training():
    return Response(_TRAINING_HTML, status=200, mimetype="text/html")

@app.route("/reward")
def reward():
    return Response(_REWARD_HTML, status=200, mimetype="text/html")

@app.route("/energy")
def energy():
    return Response(_ENERGY_HTML, status=200, mimetype="text/html")

@app.route("/deaths")
def deaths():
    return Response(_DEATHS_HTML, status=200, mimetype="text/html")

@app.route("/actions")
def actions():
    return Response(_ACTIONS_HTML, status=200, mimetype="text/html")

@app.route("/ticker")
def ticker():
    return Response(_TICKER_HTML, status=200, mimetype="text/html")

@app.route("/cards")
def cards():
    return Response(_CARDS_HTML, status=200, mimetype="text/html")


# ── startup ───────────────────────────────────────────────────────────────────

_GAME_DATA_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\SlayTheSpire\data"

if __name__ == "__main__":
    data_dir = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--data-dir" and i < len(sys.argv) - 1:
            data_dir = sys.argv[i + 1]
        elif arg.startswith("--data-dir="):
            data_dir = arg.split("=", 1)[1]

    if data_dir is None:
        data_dir = _GAME_DATA_DIR if os.path.isdir(_GAME_DATA_DIR) else "data"

    LIVE_STATE_PATH = os.path.join(data_dir, "live_state.json")
    RUN_LOG_PATH    = os.path.join(data_dir, "run_log.jsonl")

    base = "http://localhost:5001"
    print(f"Reading from: {os.path.abspath(data_dir)}")
    print()
    print("── OBS overlay endpoints ──────────────────────────")
    print(f"  Stats (v1 vs v2)   {base}/stats")
    print(f"  Training chart     {base}/training")
    print(f"  Reward curve (v2)  {base}/reward")
    print(f"  Energy efficiency  {base}/energy")
    print(f"  Floor death map    {base}/deaths")
    print(f"  Action mix (v2)    {base}/actions")
    print(f"  Live ticker        {base}/ticker")
    print(f"  Card pick history  {base}/cards")
    print()
    print(f"  API                {base}/api/runs  {base}/api/state")
    print("────────────────────────────────────────────────────")
    app.run(host="0.0.0.0", port=5001, debug=False)
