"""The brick-sorting console page, shared by the loop on port 8020 and the demo board on 8090.

`page(proxy=False)` is the page loop_control.py serves at / (same-origin routes: /ctl/state, /sys/stats, /loop/*,
/pose/*, /collect/*, /live/<cam>). `page(proxy=True)` is the same page for dlsj.demo_board, which forwards /live/*
to 8020.

Write the 8020 module with:  python -m dlsj.live_console > loop_page.py
"""

from __future__ import annotations

import sys

ROUTES = {
    False: "{state:'/ctl/state',sys:'/sys/stats',cam:(n,w)=>`/live/${n}?w=${w}`,strip:'/loop/strip.jpg',cmd:p=>'/'+p,proxy:false}",
    True: "{state:'/live/state',sys:'/live/sys/stats',cam:(n,w)=>`/live/cam/${n}?w=${w}`,strip:'/live/strip.jpg',cmd:p=>'/live/'+p,proxy:true}",
}


def page(proxy: bool = False) -> str:
    return TEMPLATE.replace("__ROUTES__", ROUTES[proxy])


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>DLSJ · Live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#09090b;--panel:#111113;--panel2:#18181b;--hover:#202024;--line:rgba(255,255,255,.07);--line2:rgba(255,255,255,.12);
--fg:#fafafa;--fg2:#d4d4d8;--mute:#8b8b94;--faint:#52525b;--ok:#4ade80;--warn:#fbbf24;--red:#ef4444;
--yellow:#facc15;--orange:#fb923c;--purple:#a78bfa;--mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:var(--bg);color:var(--fg);font:400 13px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased;font-feature-settings:"tnum","cv11";display:grid;grid-template-rows:auto 1fr;overflow:hidden}
::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-thumb{background:var(--panel2);border-radius:8px}
.num{font-family:var(--mono);font-feature-settings:"tnum";letter-spacing:-.02em}
/* header */
header{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:28px;padding:12px 20px;border-bottom:1px solid var(--line)}
.id{display:flex;align-items:center;gap:14px;white-space:nowrap}
.brand{font-weight:600;font-size:12px;letter-spacing:.24em}
.model{font-size:12px;color:var(--faint)}
.pill{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--mute);white-space:nowrap}
.pill i{width:6px;height:6px;border-radius:50%;background:var(--faint)}
.pill b{font-weight:500;color:var(--fg)}
.pill.running i{background:var(--ok);animation:pulse 1.8s infinite}.pill.paused i{background:var(--warn)}.pill.done i{background:var(--fg)}
.pill.lost b,.pill.lost{color:var(--red)}.pill.lost i{background:var(--red)}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(74,222,128,.5)}70%{box-shadow:0 0 0 6px rgba(74,222,128,0)}100%{box-shadow:0 0 0 0 rgba(74,222,128,0)}}
.flow{display:grid;grid-template-columns:repeat(8,1fr);gap:3px;max-width:720px;width:100%;justify-self:center}
.st{display:flex;flex-direction:column;gap:6px;min-width:0}
.st u{display:block;height:2px;background:var(--panel2);overflow:hidden;position:relative}
.st span{display:flex;gap:6px;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);white-space:nowrap}
.st em{font-style:normal;font-family:var(--mono);letter-spacing:0;text-transform:none}
.st.done u{background:var(--faint)}.st.done span{color:var(--mute)}
.st.on u{background:rgba(255,255,255,.12)}.st.on u::after{content:"";position:absolute;inset:0;width:35%;background:var(--fg);animation:slide 1.4s ease-in-out infinite}
.st.on span{color:var(--fg)}
.paused .st.on u::after{animation:none;width:100%;background:var(--warn)}
@keyframes slide{0%{transform:translateX(-100%)}100%{transform:translateX(290%)}}
.ctl{display:flex;gap:6px;align-items:center}
button{font:500 12.5px Inter,system-ui,sans-serif;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:8px;height:32px;padding:0 13px;cursor:pointer;transition:background .15s,border-color .15s;white-space:nowrap}
button:hover{background:var(--hover);border-color:var(--line2)}button:focus-visible{outline:2px solid var(--fg2);outline-offset:2px}button:disabled{opacity:.3;cursor:default}
button.primary{background:var(--fg);color:var(--bg);border-color:var(--fg);min-width:92px}button.primary:hover{background:#fff}
button.stop{background:var(--red);border-color:var(--red);color:#fff;font-weight:600;letter-spacing:.12em;padding:0 18px}button.stop:hover{background:#f25c5c}
button.ghost{background:transparent;border-color:transparent;color:var(--mute)}button.ghost:hover{color:var(--fg);background:var(--panel)}
button.sel{border-color:var(--fg);color:var(--fg)}
kbd{font:500 10px var(--mono);color:var(--faint);margin-left:7px}
button.primary kbd{color:#71717a}button.stop kbd{color:rgba(255,255,255,.6)}
/* layout */
.app{overflow:auto;padding:16px 20px 20px;display:flex;flex-direction:column;gap:14px;min-width:0}
.kpis{display:grid;grid-template-columns:repeat(6,minmax(0,1fr)) minmax(220px,1.7fr);border:1px solid var(--line);border-radius:12px;background:var(--panel)}
.kpi{padding:12px 16px;border-left:1px solid var(--line);min-width:0}.kpi:first-child{border-left:0}
.kpi small,.lbl{display:block;font-size:10.5px;font-weight:500;letter-spacing:.12em;text-transform:uppercase;color:var(--faint)}
.kpi b{display:block;font:400 26px/1.15 var(--mono);letter-spacing:-.03em;margin-top:4px;white-space:nowrap}
.kpi b sub{font:400 11px Inter,system-ui;color:var(--mute);letter-spacing:0;vertical-align:baseline;margin-left:4px}
.plates{display:flex;flex-direction:column;justify-content:center;gap:6px}
.pl{display:grid;grid-template-columns:66px 1fr 24px;align-items:center;gap:8px;font-size:11.5px;color:var(--mute);text-transform:capitalize}
.pl u{display:block;height:4px;background:var(--panel2);border-radius:2px;overflow:hidden}.pl u i{display:block;height:100%;transition:width .4s}
.pl b{font:500 12px var(--mono);color:var(--fg);text-align:right}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:7px}
.grid{display:grid;grid-template-columns:minmax(0,.95fr) minmax(0,1.1fr) 320px;gap:14px;align-items:start}
.col{display:flex;flex-direction:column;gap:14px;min-width:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;min-width:0}
.lbl{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:12px}
.lbl em{font-style:normal;text-transform:none;letter-spacing:0;font-size:11.5px;color:var(--mute);font-weight:400;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cam{position:relative;border-radius:12px;overflow:hidden;background:#000;aspect-ratio:4/3;border:1px solid var(--line)}
.cam.wide{aspect-ratio:16/9}
.cam img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.cam canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.tag{position:absolute;left:10px;top:10px;display:flex;gap:6px;align-items:center;background:rgba(9,9,11,.7);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);padding:3px 9px;border-radius:6px;font-size:10.5px;font-weight:500;letter-spacing:.1em;text-transform:uppercase}
.tag small{font-size:11px;color:var(--mute);font-weight:400;letter-spacing:0;text-transform:none}.tag small:empty{display:none}
.tag.r{left:auto;right:10px}
/* inspection */
.insp{padding:0;overflow:hidden}
.insp .cam{border:0;border-radius:0}
.insp.spin .cam{box-shadow:inset 0 0 0 2px var(--fg)}
.verdict{display:grid;grid-template-columns:auto 1fr auto;gap:16px;align-items:center;padding:16px 18px;border-top:1px solid var(--line)}
.sw{width:46px;height:46px;border-radius:50%;background:var(--panel2);position:relative;transition:background .3s}
.sw.run::after{content:"";position:absolute;inset:-5px;border-radius:50%;border:2px solid transparent;border-top-color:var(--fg);animation:spin .9s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.out{font-size:24px;font-weight:500;text-transform:capitalize;line-height:1.1}
.why{font-size:12px;color:var(--mute);margin-top:3px}
.plate-to{text-align:right;font-size:11px;color:var(--faint);letter-spacing:.1em;text-transform:uppercase}
.plate-to b{display:block;font:500 15px Inter;letter-spacing:0;text-transform:capitalize;color:var(--fg);margin-top:2px}
.votes{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border-top:1px solid var(--line)}
.votes div{background:var(--panel);padding:10px 14px}
.votes small{display:block;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.votes b{font:400 18px var(--mono)}
.strip{display:block;width:100%;border-top:1px solid var(--line)}
/* policy */
.instr{font-size:13.5px;color:var(--fg2);margin-bottom:12px;min-height:19px;line-height:1.4}
.instr.idle{color:var(--faint)}
.joints{display:flex;flex-direction:column;gap:7px}
.j{display:grid;grid-template-columns:38px 1fr 52px;align-items:center;gap:10px;font-size:11px;color:var(--mute)}
.j u{position:relative;display:block;height:14px;background:var(--panel2);border-radius:3px}
.j u::before{content:"";position:absolute;left:50%;top:3px;bottom:3px;width:1px;background:var(--line2)}
.j u.g::before{display:none}
.j .m{position:absolute;top:3px;bottom:3px;width:3px;margin-left:-1.5px;border-radius:1px;background:var(--fg);transition:left .12s}
.j .c{position:absolute;top:0;bottom:0;width:1px;background:var(--ok);transition:left .12s}
.j .d{position:absolute;top:6px;height:2px;background:rgba(74,222,128,.35)}
.j b{font:400 11.5px var(--mono);color:var(--fg2);text-align:right}
.j .r{position:absolute;bottom:1px;height:3px;border-radius:1px;background:rgba(251,191,36,.55);transition:left .15s,width .15s,opacity .2s}
.j u.g::after{content:"";position:absolute;left:50%;bottom:1px;width:1px;height:3px;background:var(--line2)}
.j u.g.nr::after{display:none}
.legend{display:flex;gap:14px;font-size:10.5px;color:var(--faint);margin-top:10px}
.legend i{display:inline-block;width:10px;height:2px;vertical-align:middle;margin-right:5px}
.rsec{margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
.rhead{display:flex;align-items:center;gap:10px;min-width:0}
.rhead .lbl{display:inline;margin:0}
.chip{font:500 9.5px/1 Inter,system-ui,sans-serif;letter-spacing:.12em;text-transform:uppercase;padding:3px 7px;border-radius:999px;border:1px solid var(--line2);color:var(--faint);white-space:nowrap;transition:color .2s,background .2s,border-color .2s}
.chip.shadow{color:var(--mute)}.chip.on{color:var(--fg);border-color:rgba(255,255,255,.25)}
.chip.act{color:var(--ok);background:rgba(74,222,128,.1);border-color:rgba(74,222,128,.35)}
.rw{margin-left:auto;font:400 10.5px var(--mono);color:var(--faint);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.rdot{display:inline-block;width:5px;height:5px;border-radius:50%;background:var(--faint);margin-right:5px;vertical-align:1px;transition:background .2s}.rdot.ok{background:var(--ok)}
.rsec .stats{grid-template-columns:repeat(4,minmax(0,1fr))}.rsec .stats div{padding:7px 8px}.rsec .stats small{font-size:9.5px;letter-spacing:.04em;white-space:nowrap}.rsec .stats b{white-space:nowrap;font-size:12.5px}
.rsec .stats b sub{font:400 9.5px Inter,system-ui;color:var(--faint);margin-left:2px;vertical-align:baseline}
.stats b.warn{color:var(--warn)}
canvas.chart{display:block;width:100%;height:70px;margin-top:12px;background:linear-gradient(var(--panel2),var(--panel2)) bottom/100% 1px no-repeat}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden;margin-top:12px}
.stats div{background:var(--panel);padding:7px 10px}
.stats small{display:block;font-size:10px;color:var(--faint);letter-spacing:.06em;text-transform:uppercase}.stats b{font:400 13px var(--mono)}
/* system */
.dev{display:grid;grid-template-columns:78px 1fr 62px;align-items:center;gap:10px;padding:7px 0;border-top:1px solid var(--line)}.dev:first-of-type{border-top:0}
.dev span{font-size:11px;font-weight:500;letter-spacing:.08em}.dev span small{display:block;font-size:10px;color:var(--faint);font-weight:400;letter-spacing:0}
.dev canvas{width:100%;height:26px;display:block}
.dev b{font:400 16px var(--mono);text-align:right}.dev b sub{font-size:10px;color:var(--faint);vertical-align:baseline}
.sysfoot{display:flex;justify-content:space-between;font:400 11px var(--mono);color:var(--faint);margin-top:8px}
.gs{display:flex;justify-content:space-between;align-items:baseline;font-size:14px;font-weight:500;text-transform:capitalize}
.gs small{font:400 11px var(--mono);color:var(--faint);text-transform:none}
.gs.holding span{color:var(--ok)}.gs.empty span{color:var(--red)}
.track{position:relative;height:4px;background:var(--panel2);border-radius:2px;margin-top:12px}
.track .band{position:absolute;top:0;bottom:0;background:rgba(74,222,128,.3)}
.track .pos{position:absolute;top:-4px;width:3px;height:12px;margin-left:-1.5px;border-radius:1px;background:var(--fg);transition:left .15s}
.track .cmd{position:absolute;top:-2px;width:1px;height:8px;background:var(--ok);transition:left .15s}
.bottom{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr);gap:14px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{color:var(--faint);font-weight:500;text-align:left;padding:0 10px 8px;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase}
td{padding:8px 10px;border-top:1px solid var(--line);white-space:nowrap}
td.n{font-family:var(--mono);font-size:12px}td small{color:var(--faint);margin-left:6px;font-family:var(--mono);font-size:11px}
.ok{color:var(--ok)}.bad{color:var(--red)}.mute{color:var(--mute)}.faint{color:var(--faint)}
.ev{display:flex;flex-direction:column;font-size:12px;max-height:230px;overflow:auto}
.ev div{display:grid;grid-template-columns:58px 1fr;gap:8px;padding:6px 0;border-top:1px solid var(--line);color:var(--mute)}.ev div:first-child{border-top:0}
.ev time{color:var(--faint);font-family:var(--mono);font-size:11px}
/* tools drawer */
.scrim{position:fixed;inset:0;background:rgba(0,0,0,.45);opacity:0;pointer-events:none;transition:opacity .2s;z-index:9}
.drawer{position:fixed;top:0;right:0;bottom:0;width:min(420px,100vw);background:var(--bg);border-left:1px solid var(--line);transform:translateX(102%);transition:transform .25s ease;z-index:10;overflow:auto;padding:16px 18px;display:flex;flex-direction:column;gap:12px}
body.tools .drawer{transform:none}body.tools .scrim{opacity:1;pointer-events:auto}
.dh{display:flex;justify-content:space-between;align-items:center}
.row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.row button{height:30px;padding:0 11px;font-size:12px}
.drawer table td{padding:4px 2px;font-size:12.5px;border:0}.drawer table button{height:24px;padding:0 7px;font-size:11.5px;min-width:30px}
.note{color:var(--mute);font-size:12px;line-height:1.45}
a{color:var(--fg2);text-decoration:none;border-bottom:1px solid var(--line2)}a:hover{color:var(--fg)}
@media (max-width:1400px){.grid{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.grid>.col:last-child{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}.kpis{grid-template-columns:repeat(4,1fr)}.kpi{border-top:1px solid var(--line)}}
@media (max-width:900px){.grid,.bottom{grid-template-columns:1fr}header{grid-template-columns:1fr auto}.flow{grid-column:1/-1;grid-row:2;max-width:none}.kpis{grid-template-columns:repeat(2,1fr)}kbd{display:none}}
</style></head><body>
<header>
 <div class="id"><span class="brand">DLSJ</span><span class="model" id="mname"></span><span class="pill" id="pill"><i></i><b id="mode">Connecting</b><span id="cyc"></span></span></div>
 <nav class="flow" id="flow" aria-label="Loop stages"></nav>
 <div class="ctl">
  <a href="/brief" target="_blank" style="border:0;font-size:12.5px;color:var(--mute);margin-right:6px">Slide ↗</a>
  <button class="ghost" onclick="tools(true)">Tools<kbd>T</kbd></button>
  <button onclick="L('home')">Home<kbd>H</kbd></button>
  <button class="primary" id="go" data-c="start" onclick="L(this.dataset.c)">Start<kbd>S</kbd></button>
  <button class="stop" onclick="L('stop')">STOP<kbd>Q</kbd></button>
 </div>
</header>
<main class="app">
 <section class="kpis">
  <div class="kpi"><small>Sorted</small><b id="k_tot">0</b></div>
  <div class="kpi"><small>Rate</small><b id="k_rate">–<sub>/min</sub></b></div>
  <div class="kpi"><small>Avg cycle</small><b id="k_avg">–<sub>s</sub></b></div>
  <div class="kpi"><small>Success</small><b id="k_ok">–<sub>%</sub></b></div>
  <div class="kpi"><small>Inference</small><b id="k_lat">–<sub>ms</sub></b></div>
  <div class="kpi"><small>Run time</small><b id="k_up">–</b></div>
  <div class="kpi plates" id="plates"></div>
 </section>
 <section class="grid">
  <div class="col">
   <div class="cam"><img id="imtop" alt="Top camera"><canvas id="ovtop" width="640" height="480"></canvas><span class="tag">Top<small id="ptop"></small></span><span class="tag r" id="remain"></span></div>
   <div class="cam wide"><img id="imwrist" alt="Wrist camera"><span class="tag">Wrist</span></div>
  </div>
  <div class="col">
   <section class="card insp" id="insp">
    <div class="cam"><img id="iminsp" alt="Inspection camera"><canvas id="ovinsp" width="640" height="480"></canvas><span class="tag">Inspection<small id="pinsp"></small></span><span class="tag r" id="ifr"></span></div>
    <div class="verdict"><div class="sw" id="dsw"></div><div style="min-width:0"><div class="out" id="dout">Waiting</div><div class="why" id="dwhy">No brick inspected yet</div></div><div class="plate-to">Plate<b id="dplate">–</b></div></div>
    <div class="votes"><div><small>Frames</small><b id="v_fr">–</b></div><div><small>Spin time</small><b id="v_s">–</b></div><div><small>Attempts</small><b id="v_at">–</b></div><div><small>Last live</small><b id="v_live">–</b></div></div>
    <img class="strip" id="strip" alt="" hidden onerror="this.hidden=true">
   </section>
  </div>
  <div class="col">
   <section class="card"><div class="lbl">Policy output<em id="vph"></em></div>
    <div class="instr idle" id="prompt">Idle</div>
    <div class="joints" id="joints"></div>
    <div class="legend"><span><i style="background:var(--fg)"></i>measured</span><span><i style="background:var(--ok)"></i>action</span><span id="rleg" hidden><i style="background:rgba(251,191,36,.7);height:3px"></i>residual</span><span id="sync" style="margin-left:auto"></span></div>
    <div class="rsec" id="rsec" hidden>
     <div class="rhead"><span class="lbl">Residual</span><span class="chip" id="rchip">Off</span><span class="rw" id="rw"></span></div>
     <div class="stats"><div><small>Decisions</small><b id="r_dec">–</b></div><div><small>Rejected</small><b id="r_rej">–</b></div><div><small>Detector</small><b><i class="rdot" id="r_dot"></i><span id="r_det">–</span></b></div><div><small>Step max</small><b id="r_max">–</b></div></div>
    </div>
    <canvas class="chart" id="trail" width="600" height="140"></canvas>
    <div class="stats"><div><small>Median</small><b id="med">–</b></div><div><small>Chunk age</small><b id="age">–</b></div><div><small>Chunks</small><b id="chk">–</b></div><div><small>Queued</small><b id="q">–</b></div></div>
   </section>
   <section class="card"><div class="lbl">Laptop<em>Core Ultra X9 388H</em></div>
    <div class="dev"><span>GPU<small>policy</small></span><canvas id="s_gpu" width="300" height="52"></canvas><b id="u_gpu">–</b></div>
    <div class="dev"><span>NPU<small>vision</small></span><canvas id="s_npu" width="300" height="52"></canvas><b id="u_npu">–</b></div>
    <div class="dev"><span>CPU<small>control</small></span><canvas id="s_cpu" width="300" height="52"></canvas><b id="u_cpu">–</b></div>
    <div class="dev"><span>MEM<small id="u_memgb"></small></span><canvas id="s_mem" width="300" height="52"></canvas><b id="u_mem">–</b></div>
    <div class="sysfoot"><span id="u_freq"></span><span id="u_temp"></span><span id="u_rss"></span></div>
   </section>
   <section class="card"><div class="lbl">Gripper<em id="gnum"></em></div>
    <div class="gs" id="gs"><span id="gstate">–</span><small>hold <span id="hb"></span></small></div>
    <div class="track"><div class="band" id="band"></div><div class="cmd" id="gcmd"></div><div class="pos" id="gpos"></div></div></section>
  </div>
 </section>
 <section class="bottom">
  <section class="card"><div class="lbl">Cycles<em id="tot"></em></div>
   <table><thead><tr><th>#</th><th>Brick</th><th>Grasp</th><th>Inspection</th><th>Plate</th><th>Time</th><th>Result</th></tr></thead><tbody id="cycles"></tbody></table></section>
  <section class="card"><div class="lbl">Events</div><div class="ev" id="ev"></div></section>
 </section>
</main>
<div class="scrim" onclick="tools(false)"></div>
<aside class="drawer" aria-label="Tools">
 <div class="dh"><span class="brand">TOOLS</span><button class="ghost" onclick="tools(false)">Close<kbd>Esc</kbd></button></div>
 <section class="card"><div class="lbl">Inspection pose<em id="ipsrc"></em></div>
  <div id="offwrap"><div class="row">
   <button id="holdbtn" onclick="PZ(this.dataset.on?'hold_off':'hold_on')">Hold at camera</button>
   <button id="spinbtn" class="primary" onclick="PZ('spin')">Spin now</button><button onclick="PZ('off_save')">Save offset</button><button onclick="PZ('off_reset')">Zero offset</button></div>
   <div class="note" style="margin-bottom:6px">Offset from where the VLA stops. Pan negative moves the brick right on the inspection camera.</div>
   <table id="offtab"></table><div class="note" id="vlaend"></div></div>
  <div id="fixwrap"><div class="row">
   <button id="holdbtn2" onclick="PZ(this.dataset.on?'hold_off':'hold_on')">Hold at camera</button>
   <button id="spinbtn2" class="primary" onclick="PZ('spin')">Spin now</button>
   <button onclick="PZ('goto')">Go to pose</button><button onclick="PZ('save')">Save</button><button onclick="PZ('reset')">Demo pose</button></div>
   <table id="iptab"></table></div>
  <div class="row" style="margin:10px 0 0"><button onclick="PZ('grip_open')">Open gripper</button><button onclick="PZ('grip_close')">Close gripper</button></div>
  <div class="note" id="ipmsg" style="margin-top:8px"></div></section>
 <section class="card"><div class="lbl">Data collection<a id="review" target="_blank" style="text-transform:none;letter-spacing:0;font-size:12px">Review ↗</a></div>
  <div class="row" id="colbtns"></div>
  <div class="row"><button onclick="CL('bring')">Bring to camera</button><button class="primary" onclick="CL('spin')">Spin &amp; save</button></div>
  <div class="note" id="colmsg"></div><div class="note" id="colcounts" style="margin-top:8px;color:var(--fg)"></div></section>
 <div class="note" id="classic"></div>
</aside>
<script>
const R=__ROUTES__;
const $=id=>document.getElementById(id);
const COL={yellow:'#facc15',orange:'#fb923c',purple:'#a78bfa',blue:'#3b82f6',red:'#ef4444',white:'#f4f4f5',brown:'#a16207',green:'#22c55e'};
const STEPS=[['survey','Survey'],['pick','Pick'],['bring','Bring'],['tune','Align'],['inspect','Inspect'],['place','Place'],['home','Home'],['verify','Verify']];
const LBL={none:'plain',circle:'circle',cross:'cross',marked:'marked',unknown:'uncertain'};
const JL={shoulder_pan:'pan',shoulder_lift:'lift',elbow_flex:'elbow',wrist_flex:'wrist flex',wrist_roll:'roll'};
const JN=['pan','lift','elbow','flex','roll','grip'];
const setH=(id,h)=>{const e=$(id);if(e.innerHTML!==h)e.innerHTML=h};
const setT=(id,t)=>{const e=$(id);if(e.textContent!==t)e.textContent=t};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const post=p=>fetch(R.cmd(p),{method:'POST'}).catch(()=>{});
const L=c=>post('loop/'+c),PZ=c=>post('pose/'+c),CL=c=>post('collect/'+c);
function tools(o){document.body.classList.toggle('tools',o)}
$('imtop').src=R.cam('top',640);$('iminsp').src=R.cam('inspect',640);$('imwrist').src=R.cam('wrist',480);
const LOOP_HOST=R.proxy?location.protocol+'//'+location.hostname+':8020':'';
$('review').href=LOOP_HOST+'/collect';
$('classic').innerHTML=`<a href="${LOOP_HOST}/classic" target="_blank">Classic console ↗</a>`;
addEventListener('keydown',e=>{if(e.repeat||e.metaKey||e.ctrlKey||e.altKey||/INPUT|TEXTAREA/.test(e.target.tagName))return;const k=e.key.toLowerCase();
 if(k==='q'){L('stop');e.preventDefault()}else if(k==='s')L('start');else if(k==='p')L('pause');else if(k==='r')L('resume');else if(k==='h')L('home');
 else if(k==='t')tools(!document.body.classList.contains('tools'));else if(k==='escape')tools(false)});
$('flow').innerHTML=STEPS.map(s=>`<div class="st" id="c_${s[0]}"><u></u><span>${s[1]}<em></em></span></div>`).join('');
$('joints').innerHTML=JN.map((n,i)=>`<div class="j"><span>${n}</span><u class="${i===5?'g nr':''}" id="ju${i}"><i class="r" id="jr${i}" hidden></i><i class="d" id="jd${i}"></i><i class="c" id="jc${i}"></i><i class="m" id="jm${i}"></i></u><b id="jv${i}">–</b></div>`).join('');
// ---- helpers
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const jpos=(i,v)=>i===5?clamp(v,0,100):clamp((v+180)/360*100,0,100);
function spark(id,hist,col,max){const c=$(id),x=c.getContext('2d'),w=c.width,h=c.height;x.clearRect(0,0,w,h);
 const v=(hist||[]).filter(n=>n!=null);if(v.length<2)return;const n=hist.length,mx=max||100;
 x.beginPath();hist.forEach((p,i)=>{if(p==null)return;const X=i/(n-1)*w,Y=h-2-clamp(p/mx,0,1)*(h-4);i?x.lineTo(X,Y):x.moveTo(X,Y)});
 x.strokeStyle=col;x.lineWidth=2;x.stroke();x.lineTo(w,h);x.lineTo(0,h);x.closePath();
 const g=x.createLinearGradient(0,0,0,h);g.addColorStop(0,col+'33');g.addColorStop(1,col+'00');x.fillStyle=g;x.fill()}
const fmtT=s=>{s=Math.max(0,Math.floor(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60),ss=s%60;return (h?h+':'+String(m).padStart(2,'0'):m)+':'+String(ss).padStart(2,'0')};
// ---- state
let stripSeq=-1,curStep=null,curCycle=null,lastS=null,durs={},trail=[],latH=[];
function flow(lp){const step=lp.step==='release'?'place':lp.step,order=STEPS.map(s=>s[0]),cur=order.indexOf(step),act=lp.mode==='running'||lp.mode==='paused';
 if(lp.cycle!==curCycle){curCycle=lp.cycle;durs={}}
 if(step!==curStep){if(curStep&&lastS!=null)durs[curStep]=lastS;curStep=step}
 lastS=lp.step_s;$('flow').classList.toggle('paused',lp.mode==='paused');
 order.forEach((s,i)=>{const c=$('c_'+s),on=act&&s===step,done=act&&cur>=0&&i<cur;
  c.className='st'+(on?' on':done?' done':'');
  c.querySelector('em').textContent=on?`${(lp.step_s??0).toFixed(0)}s`:(done&&durs[s]!=null?`${durs[s].toFixed(0)}s`:'')})}
function offline(){$('pill').className='pill lost';setT('mode','Offline');setT('cyc',R.proxy?' · loop on 8020 not reachable':'')}
function kpis(r,lp){const tally=lp.tallies||{},tot=Object.values(tally).reduce((a,b)=>a+b,0),mx=Math.max(1,...Object.values(tally),1),cy=lp.cycles||[];
 setT('k_tot',String(tot));
 const run=lp.started?Date.now()/1000-lp.started:null;
 $('k_up').textContent=run!=null&&lp.mode!=='off'?fmtT(run):'–';
 const rate=run&&run>30?tot/(run/60):null;$('k_rate').innerHTML=(rate!=null?rate.toFixed(1):'–')+'<sub>/min</sub>';
 const fin=cy.filter(c=>c.result&&c.result!=='running'),ok=fin.filter(c=>c.result==='sorted'),secs=ok.map(c=>c.seconds).filter(v=>v!=null);
 $('k_avg').innerHTML=(secs.length?(secs.reduce((a,b)=>a+b,0)/secs.length).toFixed(1):'–')+'<sub>s</sub>';
 $('k_ok').innerHTML=(fin.length?Math.round(ok.length/fin.length*100):'–')+'<sub>%</sub>';
 $('k_lat').innerHTML=(r.latency_ms!=null?Math.round(r.latency_ms):'–')+'<sub>ms</sub>';
 setH('plates',['yellow','orange','purple'].map(c=>`<div class="pl"><span><i class="dot" style="background:${COL[c]}"></i>${c}</span><u><i style="width:${(tally[c]||0)/mx*100}%;background:${COL[c]}"></i></u><b>${tally[c]||0}</b></div>`).join(''))}
function policy(r){const ph=r.phase||'idle';setT('vph',ph);
 const pr=$('prompt');pr.className='instr'+(ph==='running'?'':' idle');setT('prompt',ph==='running'?`“${r.prompt}”`:(ph==='inspecting'?'Paused while inspecting':'Idle'));
 const m=r.measured,c=r.cmd;
 for(let i=0;i<6;i++){const mi=m?m[i]:null,ci=c?c[i]:null;
  $('jm'+i).style.display=mi==null?'none':'';$('jc'+i).style.display=ci==null?'none':'';
  if(mi!=null)$('jm'+i).style.left=jpos(i,mi)+'%';if(ci!=null)$('jc'+i).style.left=jpos(i,ci)+'%';
  const d=$('jd'+i);if(mi!=null&&ci!=null){const a=jpos(i,mi),b=jpos(i,ci);d.style.left=Math.min(a,b)+'%';d.style.width=Math.abs(a-b)+'%';d.style.display=''}else d.style.display='none';
  setT('jv'+i,ci!=null?(i<5&&ci>0?'+':'')+ci.toFixed(1):mi!=null?mi.toFixed(1):'–')}
 if(m&&c){const err=Math.max(...m.slice(0,5).map((v,i)=>Math.abs(v-c[i])));setT('sync',`max error ${err.toFixed(1)}°`)}else setT('sync','');
 setT('med',r.latency_median_ms!=null?Math.round(r.latency_median_ms)+' ms':'–');
 setT('age',r.chunk_age_s!=null?r.chunk_age_s.toFixed(2)+' s':'–');setT('chk',String(r.chunks??'–'));setT('q',String(r.queued??'–'));
 const sm=r.studio||{};setT('mname',sm.name||'');
 resid(r.residual);
 trail.push(c?c.slice():null);if(trail.length>200)trail.shift();drawTrail()}
function resid(x){const has=!!x&&typeof x==='object';$('rsec').hidden=!has;$('rleg').hidden=!has;
 for(let i=0;i<6;i++){$('ju'+i).classList.toggle('nr',!has||i<5)}
 if(!has){for(let i=0;i<6;i++)$('jr'+i).hidden=true;return}
 const mode=x.mode||'off',act=mode==='on'&&!!x.pick;
 const ch=$('rchip'),cls='chip'+(act?' act':mode==='on'?' on':mode==='shadow'?' shadow':'');if(ch.className!==cls)ch.className=cls;
 setT('rchip',act?'Acting':mode==='on'?'On · idle':mode==='shadow'?'Shadow':'Off');
 const wn=x.weights?String(x.weights).replace(/\/+$/,'').split('/').pop():'';setT('rw',wn);$('rw').title=x.weights||'';
 const o=Array.isArray(x.o)?x.o:[];
 for(let i=0;i<6;i++){const v=Number(o[i])||0,w=clamp(Math.abs(v)/15*50,0,50),e=$('jr'+i);
  e.hidden=Math.abs(v)<.05;e.style.left=(v>=0?50:50-w)+'%';e.style.width=w+'%';e.style.opacity=act?1:.5;
  $('ju'+i).title=`residual ${v>0?'+':''}${v.toFixed(1)}${i<5?'°':''}`}
 setT('r_dec',x.decisions!=null?String(x.decisions):'–');
 const rj=x.rejected;setT('r_rej',rj!=null?String(rj):'–');$('r_rej').classList.toggle('warn',rj>0);
 setH('r_det',x.det_ms_median!=null?Number(x.det_ms_median).toFixed(1)+'<sub>ms</sub>':'–');$('r_dot').classList.toggle('ok',!!x.det_valid);
 setH('r_max',x.max_ms!=null?Number(x.max_ms).toFixed(1)+'<sub>ms</sub>':'–')}
function drawTrail(){const cv=$('trail'),x=cv.getContext('2d'),w=cv.width,h=cv.height;x.clearRect(0,0,w,h);
 x.strokeStyle='rgba(255,255,255,.06)';x.lineWidth=1;[.25,.5,.75].forEach(f=>{x.beginPath();x.moveTo(0,h*f);x.lineTo(w,h*f);x.stroke()});
 if(!trail.some(Boolean)){x.fillStyle='#52525b';x.font='500 20px Inter,system-ui';x.fillText('action stream appears while the policy runs',12,h/2+7);return}
 const n=200,sh=['#fafafa','#d4d4d8','#a1a1aa','#71717a','#52525b','#4ade80'];
 for(let j=0;j<6;j++){x.beginPath();let pen=false;trail.forEach((p,i)=>{if(!p){pen=false;return}const X=(n-trail.length+i)/(n-1)*w,Y=h-jpos(j,p[j])/100*h;pen?x.lineTo(X,Y):x.moveTo(X,Y);pen=true});x.strokeStyle=sh[j];x.lineWidth=j===5?1.5:2;x.stroke()}}
function gripper(lp){const g=lp.grasp||{},cfg=lp.config||{hold:[4,9]},sc=v=>clamp(v/40*100,0,100);
 setT('gstate',g.state||'–');$('gs').className='gs '+(g.state||'');setT('hb',`${cfg.hold[0]}–${cfg.hold[1]}`);
 setT('gnum',g.pos!=null?`pos ${g.pos} · cmd ${g.cmd}`:'');
 $('band').style.left=sc(cfg.hold[0])+'%';$('band').style.width=(sc(cfg.hold[1])-sc(cfg.hold[0]))+'%';
 if(g.pos!=null){$('gpos').style.left=sc(g.pos)+'%';$('gcmd').style.left=sc(g.cmd)+'%'}}
function inspection(r,lp,pc){const ins=r.inspection||{},fr=lp.inspect_frames||[],ph=r.phase||'idle',spin=ph==='inspecting'||ph==='collecting';
 $('insp').classList.toggle('spin',spin);setT('ifr',fr.length?`${fr.length} frames`:'');
 const ins0=pc.inspect;setT('pinsp',spin?'spinning · NPU':(ins0&&ins0.held?'brick held':''));
 if(fr.length!==stripSeq){stripSeq=fr.length;const s=$('strip');if(fr.length){s.onload=()=>s.hidden=false;s.src=R.strip+'?'+Date.now()}else s.hidden=true}
 $('dsw').classList.toggle('run',ins.state==='running');
 const sl=pc.spin_last;setT('v_live',sl&&sl.age_s<2?`${sl.label} ${Number(sl.conf).toFixed(2)}`:'–');
 if(ins.state==='running'){setT('dout','Inspecting');setT('dwhy','Spinning the brick in front of the camera');$('dsw').style.background='var(--panel2)';setT('dplate','–')}
 else if(ins.outcome){setT('dout',LBL[ins.outcome]||ins.outcome);$('dsw').style.background=COL[ins.bowl]||'#333';setT('dplate',ins.bowl||'–');
  setT('dwhy',ins.reason||'');setT('v_fr',ins.frames!=null?String(ins.frames):'–');setT('v_s',ins.seconds!=null?ins.seconds+' s':'–');
  setT('v_at',ins.attempts!=null?String(ins.attempts)+(ins.early_stop?' · early':''):'–')}}
async function tick(){let r;try{r=await fetch(R.state).then(x=>{if(!x.ok)throw 0;return x.json()})}catch(e){return offline()}
 const lp=r.loop||{},pc=r.perception||{},mode=lp.mode||'off';
 $('pill').className='pill '+mode;
 setT('mode',{off:'Idle',running:'Running',paused:'Paused',done:'Done'}[mode]||mode);
 setT('cyc',(lp.cycle?` · cycle ${lp.cycle}`:'')+(lp.attempts?` · retry ${lp.attempts}`:''));
 const go=$('go'),gc=mode==='running'?['pause','Pause','P']:mode==='paused'?['resume','Resume','R']:['start','Start','S'];
 if(go.dataset.c!==gc[0]){go.dataset.c=gc[0];go.innerHTML=`${gc[1]}<kbd>${gc[2]}</kbd>`}
 flow(lp);kpis(r,lp);policy(r);gripper(lp);inspection(r,lp,pc);
 const tb=pc.table||[],vf=pc.verify;
 setT('remain',`${tb.length} on table`+(vf?(vf.state==='running'?` · verifying ${vf.frames||0}/${vf.need||''}`:vf.state==='clear'?' · verified clear':` · ${(vf.bricks||[]).length} left`):''));
 setT('ptop',pc.age_s!=null&&pc.age_s>1?`${pc.age_s}s ago`:'');
 const cy=lp.cycles||[],tot=Object.values(lp.tallies||{}).reduce((a,b)=>a+b,0);setT('tot',cy.length?`last ${cy.length}`:'');
 setH('cycles',cy.slice().reverse().map(c=>`<tr><td class="n faint">${c.n}</td><td><i class="dot" style="background:${COL[c.brick]||'#ccc'}"></i>${esc(c.brick)}</td><td class="n">${c.grip!=null?c.grip+(c.t_grasp!=null?`<small>${c.t_grasp}s</small>`:''):'–'}</td><td>${c.outcome?(LBL[c.outcome]||c.outcome)+(c.frames!=null?`<small>${c.frames}f · ${c.inspect_s}s</small>`:''):'–'}</td><td>${c.bowl?`<i class="dot" style="background:${COL[c.bowl]}"></i>${c.bowl}`:'–'}</td><td class="n">${c.seconds!=null?c.seconds+' s':'–'}</td><td class="${c.result==='sorted'?'ok':c.result==='running'?'mute':'bad'}">${esc(c.result||'')}</td></tr>`).join('')||'<tr><td colspan="7" class="faint">No cycles yet. Press Start.</td></tr>');
 setH('ev',(lp.events||[]).map(e=>`<div><time>${e.t?new Date(e.t*1000).toLocaleTimeString([],{hour12:false}):''}</time><span>${esc(e.message||[e.kind,e.step||e.bowl||e.outcome].filter(Boolean).join(' '))}</span></div>`).join('')||'<div class="faint" style="grid-template-columns:1fr">Nothing yet</div>');
 drawTop(pc,lp.step);drawInsp(r);drawPose(r.inspect_pose);drawCollect(r.collect)}
async function sys(){let s;try{s=await fetch(R.sys).then(x=>{if(!x.ok)throw 0;return x.json()})}catch(e){return}
 const h=s.history||{},pct=v=>v!=null?Math.round(v)+'<sub>%</sub>':'–';
 $('u_gpu').innerHTML=pct(s.gpu_pct);$('u_npu').innerHTML=pct(s.npu_pct);$('u_cpu').innerHTML=pct(s.cpu_pct);$('u_mem').innerHTML=pct(s.mem_pct);
 setT('u_memgb',s.mem_used_gb!=null?`${s.mem_used_gb.toFixed(1)}/${Math.round(s.mem_total_gb)} GB`:'');
 spark('s_gpu',h.gpu,'#fafafa');spark('s_npu',h.npu,'#fafafa');spark('s_cpu',h.cpu,'#a1a1aa');spark('s_mem',h.mem_pct,'#71717a');
 setT('u_freq',s.gpu_freq_mhz!=null?`GPU ${s.gpu_freq_mhz} MHz`:'');setT('u_temp',s.cpu_temp_c!=null?`${Math.round(s.cpu_temp_c)}°C`:'');setT('u_rss',s.proc_rss_gb!=null?`loop ${s.proc_rss_gb.toFixed(1)} GB`:'')}
function box(c,b,col,w,label,sq){c.strokeStyle=col;c.lineWidth=w;c.beginPath();if(c.roundRect)c.roundRect(b[0],b[1],b[2]-b[0],b[3]-b[1],3);else c.rect(b[0],b[1],b[2]-b[0],b[3]-b[1]);c.stroke();
 if(!label)return;c.font='500 12px Inter,system-ui';const pad=sq?16:6,tw=c.measureText(label).width,y=Math.max(0,b[1]-20);
 c.fillStyle='rgba(9,9,11,.8)';c.beginPath();if(c.roundRect)c.roundRect(b[0],y,tw+pad+6,17,4);else c.rect(b[0],y,tw+pad+6,17);c.fill();
 if(sq){c.fillStyle=sq;c.fillRect(b[0]+6,y+5,7,7)}c.fillStyle='#fafafa';c.fillText(label,b[0]+pad,y+13)}
function drawTop(pc,step){const c=$('ovtop').getContext('2d');c.clearRect(0,0,640,480);const ws=pc.workspace||[0,60,560,480];
 c.fillStyle='rgba(9,9,11,.6)';c.fillRect(0,0,640,ws[1]);c.fillRect(ws[2],ws[1],640-ws[2],480-ws[1]);
 (pc.plates||[]).forEach(p=>{c.strokeStyle=COL[p.colour]||'#ccc';c.globalAlpha=.55;c.lineWidth=1.5;c.setLineDash([4,6]);c.beginPath();c.arc(p.cx,p.cy,p.r,0,7);c.stroke();c.setLineDash([]);c.globalAlpha=1});
 (pc.sorted||[]).forEach(b=>box(c,b.box,'rgba(255,255,255,.2)',1));
 (pc.table||[]).forEach(b=>box(c,b.box,'rgba(250,250,250,.85)',1.5,Number(b.conf).toFixed(2),COL[b.colour]||'#ccc'));
 if(!['survey','verify','idle','done','paused'].includes(step)){c.font='500 11px Inter,system-ui';c.fillStyle='rgba(255,255,255,.45)';c.fillText('LAST SURVEY · DETECTOR PAUSED WHILE THE ARM MOVES',12,ws[1]+18)}}
function drawInsp(r){const pc=r.perception||{},c=$('ovinsp').getContext('2d');c.clearRect(0,0,640,480);
 const spin=r.phase==='inspecting'||r.phase==='collecting',sl=pc.spin_last;
 if(spin&&sl&&sl.age_s<1.0){const col=sl.confident==='cross'?COL.purple:sl.confident==='circle'?COL.orange:sl.confident==='plain'?COL.yellow:'#d4d4d8';
  box(c,sl.box,col,2,`${sl.label} ${Number(sl.conf).toFixed(2)} · ${sl.angle>0?'+':''}${sl.angle}°`);return}
 if(spin)return;const ins=pc.inspect;if(!ins||ins.age_s>1.5)return;
 c.strokeStyle='rgba(255,255,255,.1)';c.lineWidth=1;(ins.band||[]).forEach(x=>{c.beginPath();c.moveTo(x*640,0);c.lineTo(x*640,480);c.stroke()});
 (ins.others||[]).forEach(b=>box(c,b,'rgba(255,255,255,.25)',1));
 if(ins.held)box(c,ins.held,'#fafafa',1.5,null)}
function nudges(ip,vals,cmd){return ip.joints.map((j,i)=>`<tr><td class="mute">${JL[j]||j}</td><td style="text-align:right;width:60px" class="num">${cmd==='off_nudge'&&vals[i]>0?'+':''}${Number(vals[i]).toFixed(1)}</td><td style="text-align:right">${[-5,-1,1,5].map(d=>`<button onclick="PZ('${cmd}/${j}/${d}')">${d>0?'+':''}${d}</button>`).join(' ')}</td></tr>`).join('')}
function drawPose(ip){if(!ip)return;const off=ip.mode==='vla_offset';$('offwrap').hidden=!off;$('fixwrap').hidden=off;
 const hb=$(off?'holdbtn':'holdbtn2');hb.textContent=ip.hold?'Hold at camera: on':'Hold at camera';hb.dataset.on=ip.hold?'1':'';hb.classList.toggle('sel',!!ip.hold);
 $(off?'spinbtn':'spinbtn2').disabled=!ip.at_camera;
 setT('ipmsg',(ip.at_camera?'Holding at the camera. ':'')+(ip.message||''));
 if(off){setT('ipsrc','offset · '+ip.offset_source);setT('vlaend',ip.vla_end?'VLA stopped at '+ip.vla_end.join(', '):'');setH('offtab',nudges(ip,ip.offset,'off_nudge'))}
 else{setT('ipsrc',(ip.mode==='vla_pose'?'VLA bring, then ':'')+ip.source);setH('iptab',nudges(ip,ip.pose,'nudge'))}}
function drawCollect(c){if(!c)return;setH('colbtns',['off','plain','circle','cross'].map(l=>`<button class="${(c.label||'off')===l?'sel':''}" onclick="CL('set/${l}')">${l==='off'?'Off':l[0].toUpperCase()+l.slice(1)}</button>`).join(''));
 setT('colmsg',(c.busy?'Spinning… ':'')+(c.message||'Pick the true label of the brick; every spin is saved under it.'));
 setH('colcounts',Object.entries(c.counts||{}).map(([k,v])=>`${esc(k)} <span class="mute">${v.spins} spins · ${v.crops} crops</span>`).join('<br>'))}
setInterval(tick,150);tick();setInterval(sys,500);sys();
</script></body></html>
"""

if __name__ == "__main__":
    sys.stdout.write('"""Brick-sorting console page served by loop_control.py at /. Generated by dlsj.live_console; edit that."""\n\n')
    sys.stdout.write("PAGE = " + repr(page(proxy=False)) + "\n")
