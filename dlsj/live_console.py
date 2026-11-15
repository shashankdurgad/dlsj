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
:root{--bg:#060607;--panel:#0c0c0e;--panel2:#151518;--hover:#1c1c20;--line:rgba(255,255,255,.075);--line2:rgba(255,255,255,.14);
--fg:#f4f4f5;--fg2:#c9c9d0;--mute:#8a8a93;--faint:#4f4f57;--ok:#4ade80;--warn:#fbbf24;--red:#ef4444;
--yellow:#facc15;--orange:#fb923c;--purple:#a78bfa;--mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:var(--bg);color:var(--fg);font:400 12.5px/1.4 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased;font-feature-settings:"tnum","cv11";display:grid;grid-template-rows:auto minmax(0,1fr);overflow:hidden}
::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-thumb{background:#222227;border-radius:6px}::-webkit-scrollbar-track{background:transparent}
.num{font-family:var(--mono);font-feature-settings:"tnum";letter-spacing:-.02em}
/* header */
header{display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:24px;padding:0 14px;height:50px;border-bottom:1px solid var(--line);background:#09090b}
.id{display:flex;align-items:center;gap:12px;white-space:nowrap}
.brand{font:600 12px/1 Inter,system-ui;letter-spacing:.26em}
.model{font:400 11px var(--mono);color:var(--faint);max-width:160px;overflow:hidden;text-overflow:ellipsis}
.pill{display:flex;align-items:center;gap:7px;font:400 11px var(--mono);color:var(--mute);white-space:nowrap;padding:4px 9px;border:1px solid var(--line);border-radius:3px}
.pill i{width:6px;height:6px;border-radius:50%;background:var(--faint)}
.pill b{font:500 10.5px Inter,system-ui;letter-spacing:.14em;text-transform:uppercase;color:var(--fg)}
.pill.running i{background:var(--ok);animation:pulse 1.8s infinite}.pill.paused i{background:var(--warn)}.pill.done i{background:var(--fg)}
.pill.lost b,.pill.lost{color:var(--red);border-color:rgba(239,68,68,.4)}.pill.lost i{background:var(--red)}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(74,222,128,.5)}70%{box-shadow:0 0 0 6px rgba(74,222,128,0)}100%{box-shadow:0 0 0 0 rgba(74,222,128,0)}}
.flow{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:3px 12px;max-width:760px;width:100%;justify-self:center;counter-reset:st}
.st{display:flex;flex-direction:column;gap:5px;min-width:0;counter-increment:st}
.st u{display:block;height:3px;background:var(--panel2);overflow:hidden;position:relative}
.st span{display:flex;gap:5px;font:500 9.5px/1 Inter,system-ui;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);white-space:nowrap;overflow:hidden}
.st span::before{content:counter(st,decimal-leading-zero);font:400 9px var(--mono);letter-spacing:0;color:var(--faint);opacity:.8}
.st em{font-style:normal;font:400 9.5px var(--mono);letter-spacing:0;text-transform:none;margin-left:auto}
.st.done u{background:#3a3a41}.st.done span{color:var(--mute)}
.st.on u{background:rgba(255,255,255,.12)}.st.on u::after{content:"";position:absolute;inset:0;width:35%;background:var(--fg);animation:slide 1.4s ease-in-out infinite}
.st.on span{color:var(--fg)}
.paused .st.on u::after{animation:none;width:100%;background:var(--warn)}
@keyframes slide{0%{transform:translateX(-100%)}100%{transform:translateX(290%)}}
.ctl{display:flex;gap:5px;align-items:center}
.clk{font:400 11px var(--mono);color:var(--faint);margin-right:8px}
button{font:500 12px Inter,system-ui,sans-serif;color:var(--fg);background:var(--panel);border:1px solid var(--line2);border-radius:4px;height:30px;padding:0 12px;cursor:pointer;transition:background .15s,border-color .15s;white-space:nowrap}
button:hover{background:var(--hover);border-color:rgba(255,255,255,.22)}button:focus-visible{outline:2px solid var(--fg2);outline-offset:2px}button:disabled{opacity:.3;cursor:default}
button.primary{background:var(--fg);color:var(--bg);border-color:var(--fg);min-width:90px}button.primary:hover{background:#fff}
button.stop{background:var(--red);border-color:var(--red);color:#fff;font-weight:600;letter-spacing:.16em;padding:0 20px;height:34px;box-shadow:0 0 0 1px rgba(239,68,68,.25),0 0 18px rgba(239,68,68,.18)}button.stop:hover{background:#f25c5c}
button.ghost{background:transparent;border-color:transparent;color:var(--mute)}button.ghost:hover{color:var(--fg);background:var(--panel)}
button.sel{border-color:var(--fg);color:var(--fg)}
kbd{font:500 9.5px var(--mono);color:var(--faint);margin-left:7px}
button.primary kbd{color:#71717a}button.stop kbd{color:rgba(255,255,255,.65)}
.slide{border:0;font:400 11.5px Inter,system-ui;color:var(--mute);margin-right:4px}
/* layout */
.app{overflow:hidden;padding:8px 12px 12px;display:grid;grid-template-rows:auto minmax(0,1fr);gap:8px;min-width:0;min-height:0}
.grid{display:grid;grid-template-columns:minmax(0,1fr) clamp(360px,31vw,500px);gap:10px;min-height:0}
.cams{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);grid-template-rows:auto minmax(0,1fr);gap:10px;min-width:0;min-height:0}
.col{display:flex;flex-direction:column;gap:10px;min-width:0;min-height:0}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:4px;min-width:0;min-height:0;display:flex;flex-direction:column;position:relative;overflow:hidden}
.ph{display:flex;align-items:center;gap:8px;height:28px;flex:none;padding:0 10px;border-bottom:1px solid var(--line);font:500 9.5px/1 Inter,system-ui;letter-spacing:.16em;text-transform:uppercase;color:var(--mute);white-space:nowrap}
.ph s{text-decoration:none;font:400 9.5px var(--mono);letter-spacing:0;color:var(--faint)}
.ph b{font-weight:500;color:var(--fg)}
.ph em{margin-left:auto;font:400 10.5px var(--mono);letter-spacing:0;text-transform:none;color:var(--faint);overflow:hidden;text-overflow:ellipsis;min-width:0}
.pb{padding:10px 12px}
.grow{flex:1 1 0;min-height:0;overflow:auto}
.lbl{display:block;font:500 9.5px/1 Inter,system-ui;letter-spacing:.14em;text-transform:uppercase;color:var(--faint)}
/* kpis */
.kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));background:var(--panel);border:1px solid var(--line);border-radius:4px}
.kpi{display:flex;align-items:baseline;gap:12px;padding:8px 14px;border-left:1px solid var(--line);min-width:0;position:relative}.kpi:first-child{border-left:0}
.kpi small{font:500 9.5px/1 Inter,system-ui;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);white-space:nowrap}
.kpi small s{text-decoration:none;font:400 9.5px var(--mono);letter-spacing:0;text-transform:none;margin-left:4px}
.kpi b{font:400 20px/1.1 var(--mono);letter-spacing:-.03em;white-space:nowrap}
.kpi b sub{font:400 10.5px Inter,system-ui;color:var(--mute);letter-spacing:0;vertical-align:baseline;margin-left:3px}
.kpi canvas{flex:1 1 0;min-width:0;max-width:150px;height:22px;align-self:center;margin-left:auto}
/* cameras */
.cam{position:relative;overflow:hidden;background:#000;flex:none}
.cam4{aspect-ratio:4/3;max-height:calc(100vh - 400px)}
.cam img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.cam4 img,.cam4 canvas{object-fit:contain}
.cam canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.crn{position:absolute;inset:5px;pointer-events:none;--c:rgba(255,255,255,.45);background:
linear-gradient(var(--c),var(--c)) 0 0/9px 1px,linear-gradient(var(--c),var(--c)) 0 0/1px 9px,
linear-gradient(var(--c),var(--c)) 100% 0/9px 1px,linear-gradient(var(--c),var(--c)) 100% 0/1px 9px,
linear-gradient(var(--c),var(--c)) 0 100%/9px 1px,linear-gradient(var(--c),var(--c)) 0 100%/1px 9px,
linear-gradient(var(--c),var(--c)) 100% 100%/9px 1px,linear-gradient(var(--c),var(--c)) 100% 100%/1px 9px;background-repeat:no-repeat}
.tag{position:absolute;left:8px;bottom:8px;display:flex;gap:6px;align-items:center;background:rgba(6,6,7,.72);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);padding:2px 7px;border-radius:2px;font:400 10.5px/1.5 var(--mono);color:var(--fg2)}
.tag:empty{display:none}.tag.r{left:auto;right:8px}
.live{position:absolute;left:14px;top:11px;font:500 9px/1 var(--mono);letter-spacing:.14em;color:var(--fg2);display:flex;align-items:center;gap:5px}
.live::before{content:"";width:5px;height:5px;border-radius:50%;background:var(--red);animation:blink 2s infinite}
@keyframes blink{50%{opacity:.25}}
.insp.spin .cam{box-shadow:inset 0 0 0 2px var(--fg)}
/* wrist + gripper */
.wcam{flex:1 1 0;min-height:80px}
.grip{display:grid;grid-template-columns:112px minmax(0,1fr) minmax(0,34%);gap:14px;align-items:center;padding:8px 12px;border-top:1px solid var(--line);flex:none}
.gs{display:flex;flex-direction:column;gap:3px;font:500 15px/1.1 Inter,system-ui;text-transform:capitalize}
.gs small{font:400 10px var(--mono);color:var(--faint);text-transform:none}
.gs.holding span{color:var(--ok)}.gs.empty span{color:var(--red)}
.track{position:relative;height:6px;background:var(--panel2);border-radius:1px;margin:4px 0 5px}
.track .band{position:absolute;top:0;bottom:0;background:rgba(74,222,128,.25);border-left:1px solid rgba(74,222,128,.6);border-right:1px solid rgba(74,222,128,.6)}
.track .pos{position:absolute;top:-4px;width:3px;height:14px;margin-left:-1.5px;border-radius:1px;background:var(--fg);transition:left .15s}
.track .cmd{position:absolute;top:-2px;width:1px;height:10px;background:var(--ok);transition:left .15s}
.scale{display:flex;justify-content:space-between;font:400 9px var(--mono);color:var(--faint)}
canvas.gt{width:100%;height:34px;display:block}
/* verdict */
.verdict{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:14px;align-items:center;padding:14px 14px 12px}
.sw{width:46px;height:46px;border-radius:50%;background:var(--panel2);position:relative;transition:background .3s;box-shadow:inset 0 0 0 1px rgba(255,255,255,.08)}
.sw.run::after{content:"";position:absolute;inset:-5px;border-radius:50%;border:1.5px solid transparent;border-top-color:var(--fg);animation:spin .9s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.out{font:500 24px/1.1 Inter,system-ui;text-transform:capitalize}
.why{font-size:11.5px;color:var(--mute);margin-top:4px;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.plate-to{text-align:right;font:500 9.5px Inter;color:var(--faint);letter-spacing:.14em;text-transform:uppercase}
.plate-to b{display:block;font:500 17px Inter;letter-spacing:0;text-transform:capitalize;color:var(--fg);margin-top:3px}
.route{display:flex;flex-direction:column;justify-content:center;gap:5px;padding:0 14px 12px;flex:1 1 auto;min-height:0}
.rt{display:grid;grid-template-columns:minmax(0,1fr) 22px auto 58px;align-items:center;gap:8px;height:30px;padding:0 10px;border:1px solid transparent;border-radius:3px;font:400 11px var(--mono);color:var(--mute);transition:color .25s,border-color .25s,background .25s}
.rt>span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rt u{display:block;height:1px;background:currentColor;opacity:.5;position:relative}.rt u::after{content:"";position:absolute;right:-1px;top:-3px;border:3.5px solid transparent;border-left:5px solid currentColor;border-right:0}
.rt b{font:500 10px Inter,system-ui;letter-spacing:.12em;text-transform:uppercase;color:var(--c)}
.seg{display:inline-flex;gap:3px}
.seg button{width:28px;height:22px;display:grid;place-items:center;padding:0;border:1px solid var(--line);border-radius:4px;background:transparent;cursor:pointer;transition:background .15s,border-color .15s}
.seg button i{width:10px;height:10px;border-radius:50%;background:var(--c);opacity:.5;transition:opacity .15s,box-shadow .15s}
.seg button:hover:not(:disabled):not(.sel){border-color:var(--line2);background:var(--hover)}.seg button:hover:not(:disabled) i{opacity:.8}
.seg button.sel{border-color:var(--c);background:color-mix(in srgb,var(--c) 20%,transparent)}.seg button.sel i{opacity:1;box-shadow:0 0 0 3px color-mix(in srgb,var(--c) 25%,transparent)}
.seg button:disabled{cursor:default}
.seg button:focus-visible,.rtf button:focus-visible{outline:2px solid var(--fg2);outline-offset:1px}
.rt.on{color:var(--fg);border-color:var(--line2);background:rgba(255,255,255,.03)}
.rtf{display:flex;justify-content:space-between;align-items:center;padding:2px 10px 0;font:400 10px Inter,system-ui;color:var(--faint)}
.rtf button{background:none;border:0;padding:0;font:500 10px Inter,system-ui;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);cursor:pointer;text-decoration:underline;text-underline-offset:2px}
.rtf button.dirty{color:var(--warn)}.rtf button:hover{color:var(--fg)}
.votes{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-top:1px solid var(--line);margin-top:auto}
.votes div{padding:8px 12px;border-left:1px solid var(--line);min-width:0}.votes div:first-child{border-left:0}
.votes small{display:block;font:500 9px/1 Inter;letter-spacing:.14em;text-transform:uppercase;color:var(--faint)}
.votes b{display:block;font:400 13.5px/1.2 var(--mono);margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.strip{display:block;width:100%;height:52px;object-fit:cover;border-top:1px solid var(--line);flex:none}
/* policy */
.instr{font-size:12.5px;color:var(--fg2);min-height:34px;line-height:1.35;padding:0 0 8px}
.instr.idle{color:var(--faint)}
.joints{display:flex;flex-direction:column;gap:4px}
.j{display:grid;grid-template-columns:36px 1fr 48px;align-items:center;gap:8px;font:400 10.5px var(--mono);color:var(--mute)}
.j u{position:relative;display:block;height:12px;background:var(--panel2);border-radius:1px;background-image:repeating-linear-gradient(90deg,transparent 0 calc(12.5% - 1px),rgba(255,255,255,.035) calc(12.5% - 1px) 12.5%)}
.j u::before{content:"";position:absolute;left:50%;top:2px;bottom:2px;width:1px;background:var(--line2)}
.j u.g::before{display:none}
.j .m{position:absolute;top:2px;bottom:2px;width:3px;margin-left:-1.5px;border-radius:1px;background:var(--fg);transition:left .12s}
.j .c{position:absolute;top:0;bottom:0;width:1px;background:var(--ok);transition:left .12s}
.j .d{position:absolute;top:4.5px;height:3px;background:rgba(74,222,128,.3)}
.j b{font:400 11px var(--mono);color:var(--fg2);text-align:right}
.j .r{position:absolute;bottom:1px;height:3px;border-radius:1px;background:rgba(251,191,36,.6);transition:left .15s,width .15s,opacity .2s}
.j u.g::after{content:"";position:absolute;left:50%;bottom:1px;width:1px;height:3px;background:var(--line2)}
.j u.g.nr::after{display:none}
.legend{display:flex;gap:12px;font:400 10px var(--mono);color:var(--faint);margin-top:7px}
.legend i{display:inline-block;width:10px;height:2px;vertical-align:middle;margin-right:5px}
.line{display:flex;align-items:center;gap:10px;margin-top:8px;padding-top:8px;border-top:1px solid var(--line);font:400 10.5px var(--mono);color:var(--faint);white-space:nowrap;min-width:0;overflow:hidden}
.line b{font-weight:400;color:var(--fg2)}.line b sub{font:400 9px Inter,system-ui;color:var(--faint);margin-left:1px;vertical-align:baseline}
.line b.warn{color:var(--warn)}
.line .lbl{flex:none}
.chip{font:500 9px/1 Inter,system-ui,sans-serif;letter-spacing:.14em;text-transform:uppercase;padding:3px 7px;border-radius:2px;border:1px solid var(--line2);color:var(--faint);white-space:nowrap;transition:color .2s,background .2s,border-color .2s}
.chip.shadow{color:var(--mute)}.chip.on{color:var(--fg);border-color:rgba(255,255,255,.25)}
.chip.act{color:var(--ok);background:rgba(74,222,128,.1);border-color:rgba(74,222,128,.35)}
.rw{margin-left:auto;color:var(--faint);overflow:hidden;text-overflow:ellipsis;min-width:0;max-width:30%}
.rdot{display:inline-block;width:5px;height:5px;border-radius:50%;background:var(--faint);margin-right:4px;vertical-align:1px;transition:background .2s}.rdot.ok{background:var(--ok)}
/* compute */
.devs{display:grid;grid-template-columns:repeat(4,minmax(0,1fr))}
.dev{padding:9px 12px 8px;border-left:1px solid var(--line);min-width:0}.dev:first-child{border-left:0}
.dt{display:flex;align-items:baseline;justify-content:space-between;gap:6px}
.dev span{font:500 10.5px/1 Inter,system-ui;letter-spacing:.12em}
.dt b{font:400 15px/1 var(--mono);letter-spacing:-.03em}.dt b sub{font-size:9px;color:var(--faint);vertical-align:baseline}
.dev small{display:block;font:400 9.5px var(--mono);color:var(--faint);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar{display:block;height:3px;background:var(--panel2);margin-top:7px;border-radius:1px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:var(--fg);transition:width .4s}.bar i.hot{background:var(--red)}
.dev canvas.sp{width:100%;height:18px;display:block;margin-top:4px}
.sysfoot{display:flex;justify-content:space-between;gap:8px;font:400 10px var(--mono);color:var(--faint);padding:6px 12px;border-top:1px solid var(--line)}
/* throughput */
.leg{display:flex;gap:10px;font:400 10px var(--mono);color:var(--faint);white-space:nowrap;text-transform:none;letter-spacing:0;margin-left:auto}
.leg i{display:inline-block;width:8px;height:8px;margin-right:5px;vertical-align:-1px;border-radius:1px}
canvas.cc{display:block;width:100%;height:96px}
.plates{display:flex;gap:16px;margin-top:6px;font-size:11px;color:var(--mute);text-transform:capitalize;align-items:baseline}
.plates b{font:500 12px var(--mono);color:var(--fg);margin-left:6px}
.plates em{margin-left:auto;font:400 10px var(--mono);color:var(--faint);font-style:normal;text-transform:none}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:0}
/* log */
.tab{height:22px;padding:0 9px;font:500 9.5px Inter,system-ui;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);background:transparent;border-color:transparent}
.tab s{text-decoration:none;font:400 9.5px var(--mono);letter-spacing:0;margin-left:5px;color:var(--faint)}
.tab.sel{color:var(--fg);border-color:var(--line2);background:var(--panel2)}
.log .evp{display:none}.log.evs .evp{display:block}.log.evs .cyp{display:none}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{position:sticky;top:0;background:var(--panel);z-index:1}
th{color:var(--faint);font:500 9.5px Inter;text-align:left;padding:6px 10px;letter-spacing:.14em;text-transform:uppercase;border-bottom:1px solid var(--line)}
td{padding:4px 10px;border-top:1px solid var(--line);white-space:nowrap}
tbody tr:first-child td{border-top:0}
td.n{font-family:var(--mono);font-size:11.5px}td small{color:var(--faint);margin-left:6px;font-family:var(--mono);font-size:10.5px}
.ok{color:var(--ok)}.bad{color:var(--red)}.mute{color:var(--mute)}.faint{color:var(--faint)}
.ev{display:flex;flex-direction:column;font-size:11.5px}
.ev div{display:grid;grid-template-columns:8px 56px minmax(0,1fr);gap:6px;align-items:baseline;padding:4px 10px;border-top:1px solid var(--line);color:var(--fg2)}.ev div:first-child{border-top:0}
.ev div::before{content:"";width:5px;height:5px;border-radius:50%;background:var(--faint);align-self:center}
.ev div.k-cycle::before{background:var(--ok)}.ev div.k-error::before,.ev div.k-stop::before{background:var(--red)}.ev div.k-outcome::before{background:var(--fg2)}
.ev div:first-child{color:var(--fg)}
.ev time{color:var(--faint);font-family:var(--mono);font-size:10.5px}
.ev span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* tools drawer */
.scrim{position:fixed;inset:0;background:rgba(0,0,0,.5);opacity:0;pointer-events:none;transition:opacity .2s;z-index:9}
.drawer{position:fixed;top:0;right:0;bottom:0;width:min(420px,100vw);background:var(--bg);border-left:1px solid var(--line);transform:translateX(102%);transition:transform .25s ease;z-index:10;overflow:auto;padding:14px 16px;display:flex;flex-direction:column;gap:12px}
body.tools .drawer{transform:none}body.tools .scrim{opacity:1;pointer-events:auto}
.dh{display:flex;justify-content:space-between;align-items:center}
.card{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:12px 14px;min-width:0}
.card .lbl{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:12px}
.card .lbl em{font:400 11px var(--mono);text-transform:none;letter-spacing:0;color:var(--mute);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.row button{height:30px;padding:0 11px;font-size:12px}
.drawer table td{padding:4px 2px;font-size:12.5px;border:0}.drawer table button{height:24px;padding:0 7px;font-size:11.5px;min-width:30px}
.note{color:var(--mute);font-size:12px;line-height:1.45}
a{color:var(--fg2);text-decoration:none;border-bottom:1px solid var(--line2)}a:hover{color:var(--fg)}
@media (max-width:1400px){.flow .st span::before{display:none}.kpi b{font-size:18px}}
@media (max-width:1300px){header{gap:14px}.clk,.model{display:none}.flow .st em{display:none}.kpi{padding:7px 12px}.dev{padding:8px 9px 7px}.rt{grid-template-columns:minmax(0,1fr) 24px 84px}.line{gap:8px}th,td{padding-left:7px;padding-right:7px}td small{margin-left:4px}}
@media (max-height:880px){.instr{min-height:0}canvas.cc{height:72px}.sysfoot{display:none}.strip{display:none!important}.verdict{padding:10px 12px 8px}.route{gap:2px}.rt{height:24px}.dev canvas.sp{height:14px}}
@media (max-width:1100px){body{overflow:auto;display:block}.app{overflow:visible}.grid{grid-template-columns:1fr}.cam4{max-height:none}.wcam{flex:none;aspect-ratio:16/9}.grow{flex:none;max-height:300px}
 header{height:auto;padding:8px 14px;grid-template-columns:1fr auto}.flow{grid-column:1/-1;grid-row:2;max-width:none;padding-bottom:4px}}
@media (max-width:760px){.cams{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.devs{grid-template-columns:repeat(2,minmax(0,1fr))}kbd{display:none}}
</style></head><body>
<header>
 <div class="id"><span class="brand">DLSJ</span><span class="model" id="mname"></span><span class="pill" id="pill"><i></i><b id="mode">Connecting</b><span id="cyc"></span></span></div>
 <nav class="flow" id="flow" aria-label="Loop stages"></nav>
 <div class="ctl">
  <span class="clk" id="clk"></span>
  <a href="/brief" target="_blank" class="slide">Slide ↗</a>
  <button class="ghost" onclick="tools(true)">Tools<kbd>T</kbd></button>
  <button onclick="L('home')">Home<kbd>H</kbd></button>
  <button class="primary" id="go" data-c="start" onclick="L(this.dataset.c)">Start<kbd>S</kbd></button>
  <button class="stop" onclick="L('stop')">STOP<kbd>Q</kbd></button>
 </div>
</header>
<main class="app">
 <section class="kpis">
  <div class="kpi"><small>Sorted</small><b id="k_tot">0</b></div>
  <div class="kpi"><small>Rate<s id="k_up"></s></small><b id="k_rate">–<sub>/min</sub></b></div>
  <div class="kpi"><small>Avg cycle</small><b id="k_avg">–<sub>s</sub></b></div>
  <div class="kpi"><small>Success</small><b id="k_ok">–<sub>%</sub></b></div>
  <div class="kpi"><small>Inference</small><b id="k_lat">–<sub>ms</sub></b><canvas id="k_spark"></canvas></div>
 </section>
 <section class="grid">
  <div class="cams">
   <section class="panel"><div class="ph"><s>01</s><b>Perceive</b>top camera<em id="ptop"></em></div>
    <div class="cam cam4"><img id="imtop" alt="Top camera"><canvas id="ovtop" width="640" height="480"></canvas><i class="crn"></i><span class="live">LIVE</span><span class="tag r" id="remain"></span></div>
   </section>
   <section class="panel insp" id="insp"><div class="ph"><s>02</s><b>Detect</b>inspection camera<em id="pinsp"></em></div>
    <div class="cam cam4"><img id="iminsp" alt="Inspection camera"><canvas id="ovinsp" width="640" height="480"></canvas><i class="crn"></i><span class="live">LIVE</span><span class="tag r" id="ifr"></span></div>
   </section>
   <section class="panel"><div class="ph">Wrist<b>·</b>gripper<em id="gnum"></em></div>
    <div class="cam wcam"><img id="imwrist" alt="Wrist camera"><i class="crn"></i></div>
    <div class="grip"><div class="gs" id="gs"><span id="gstate">–</span><small>hold <span id="hb"></span></small></div>
     <div><div class="track"><div class="band" id="band"></div><div class="cmd" id="gcmd"></div><div class="pos" id="gpos"></div></div>
      <div class="scale"><span>0</span><span>closed · open</span><span>40</span></div></div>
     <canvas class="gt" id="gtrail"></canvas></div>
   </section>
   <section class="panel"><div class="ph"><s>03</s><b>Reason</b>verdict → plate</div>
    <div class="verdict"><div class="sw" id="dsw"></div><div style="min-width:0"><div class="out" id="dout">Waiting</div><div class="why" id="dwhy">No brick inspected yet</div></div><div class="plate-to">Plate<b id="dplate">–</b></div></div>
    <div class="route" id="route"></div>
    <div class="votes"><div><small>Frames</small><b id="v_fr">–</b></div><div><small>Spin time</small><b id="v_s">–</b></div><div><small>Attempts</small><b id="v_at">–</b></div><div><small>Last live</small><b id="v_live">–</b></div></div>
    <img class="strip" id="strip" alt="" hidden onerror="this.hidden=true">
   </section>
  </div>
  <div class="col">
   <section class="panel"><div class="ph"><s>04</s><b>Act</b>policy output<em id="vph"></em></div>
    <div class="pb">
     <div class="instr idle" id="prompt">Idle</div>
     <div class="joints" id="joints"></div>
     <div class="legend"><span><i style="background:var(--fg)"></i>measured</span><span><i style="background:var(--ok)"></i>action</span><span id="rleg" hidden><i style="background:rgba(251,191,36,.7);height:3px"></i>residual</span><span id="sync" style="margin-left:auto"></span></div>
     <div class="line" id="rsec" hidden><span class="lbl">Residual</span><span class="chip" id="rchip">Off</span>
      <span>dec <b id="r_dec">–</b></span><span>rej <b id="r_rej">–</b></span><span><i class="rdot" id="r_dot"></i>det <b id="r_det">–</b></span><span>max <b id="r_max">–</b></span><span class="rw" id="rw"></span></div>
     <div class="line"><span class="lbl">Latency</span><span>med <b id="med">–</b></span><span>p95 <b id="p95">–</b></span><span>age <b id="age">–</b></span><span>chunks <b id="chk">–</b></span><span>queue <b id="q">–</b></span></div>
    </div>
   </section>
   <section class="panel"><div class="ph"><s>05</s><b>Optimize</b>compute<em>Core Ultra X9 388H</em></div>
    <div class="devs">
     <div class="dev"><div class="dt"><span>GPU</span><b id="u_gpu">–</b></div><small>VLA policy</small><u class="bar"><i id="a_gpu"></i></u><canvas class="sp" id="s_gpu"></canvas></div>
     <div class="dev"><div class="dt"><span>NPU</span><b id="u_npu">–</b></div><small>YOLO detectors</small><u class="bar"><i id="a_npu"></i></u><canvas class="sp" id="s_npu"></canvas></div>
     <div class="dev"><div class="dt"><span>CPU</span><b id="u_cpu">–</b></div><small>control</small><u class="bar"><i id="a_cpu"></i></u><canvas class="sp" id="s_cpu"></canvas></div>
     <div class="dev"><div class="dt"><span>MEM</span><b id="u_mem">–</b></div><small id="u_memgb">&nbsp;</small><u class="bar"><i id="a_mem"></i></u><canvas class="sp" id="s_mem"></canvas></div>
    </div>
    <div class="sysfoot"><span id="u_freq"></span><span id="u_temp"></span><span id="u_rss"></span></div>
   </section>
   <section class="panel"><div class="ph">Cycle time<span class="leg"><span><i style="background:#e4e4e7"></i>grasp</span><span><i style="background:#8a8a93"></i>inspect</span><span><i style="background:#3f3f46"></i>place · home</span></span></div>
    <div class="pb"><canvas class="cc" id="cyc_chart"></canvas><div class="plates" id="plates"></div></div>
   </section>
   <section class="panel log" id="log" style="flex:1"><div class="ph"><button class="tab sel" id="tab_c" onclick="logTab(0)">Cycles</button><button class="tab" id="tab_e" onclick="logTab(1)">Events<s id="evn"></s></button><em id="tot"></em></div>
    <div class="grow cyp"><table><thead><tr><th>#</th><th>Brick</th><th>Inspection</th><th>Plate</th><th>Time</th><th>Result</th></tr></thead><tbody id="cycles"></tbody></table></div>
    <div class="grow evp"><div class="ev" id="ev"></div></div>
   </section>
  </div>
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
const RT_ALL=[['plain','plain','no mark'],['circle','circle','minor defect'],['cross','cross','defective'],['marked','marked','has a mark'],['uncertain','uncertain','after re-spin']];
const RT_DEF={plain:'yellow',circle:'orange',cross:'purple',marked:'purple',uncertain:'purple'};
const RT_OUT={none:'plain',unknown:'uncertain'};
const RT_PLATES=['yellow','orange','purple'];
const rtS={tab:null,bin:null,out:null,ro:false,hold:0};
const JL={shoulder_pan:'pan',shoulder_lift:'lift',elbow_flex:'elbow',wrist_flex:'wrist flex',wrist_roll:'roll'};
const JN=['pan','lift','elbow','flex','roll','grip'];
const MONO='JetBrains Mono,ui-monospace,Menlo,monospace';
const setH=(id,h)=>{const e=$(id);if(e.innerHTML!==h)e.innerHTML=h};
const setT=(id,t)=>{const e=$(id);if(e.textContent!==t)e.textContent=t};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const post=p=>fetch(R.cmd(p),{method:'POST'}).catch(()=>{});
const L=c=>post('loop/'+c),PZ=c=>post('pose/'+c),CL=c=>post('collect/'+c);
function tools(o){document.body.classList.toggle('tools',o)}
function logTab(e){$('log').classList.toggle('evs',!!e);$('tab_c').classList.toggle('sel',!e);$('tab_e').classList.toggle('sel',!!e)}
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
function fit(c){const r=window.devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;if(!w||!h)return null;const W=Math.round(w*r),H=Math.round(h*r);
 if(c.width!==W||c.height!==H){c.width=W;c.height=H}const x=c.getContext('2d');x.setTransform(r,0,0,r,0,0);x.clearRect(0,0,w,h);return [x,w,h]}
function hair(x,w,h,fr){x.strokeStyle='rgba(255,255,255,.055)';x.lineWidth=1;fr.forEach(f=>{const y=Math.round(h*f)+.5;x.beginPath();x.moveTo(0,y);x.lineTo(w,y);x.stroke()})}
function spark(id,hist,col,max){const f=fit($(id));if(!f)return;const [x,w,h]=f;hair(x,w,h,[.5]);
 const v=(hist||[]).filter(n=>n!=null);if(v.length<2)return;const n=hist.length,mx=max||100;
 x.beginPath();hist.forEach((p,i)=>{if(p==null)return;const X=i/(n-1)*w,Y=h-1-clamp(p/mx,0,1)*(h-3);i?x.lineTo(X,Y):x.moveTo(X,Y)});
 x.strokeStyle=col;x.lineWidth=1.25;x.lineJoin='round';x.stroke();x.lineTo(w,h);x.lineTo(0,h);x.closePath();
 const g=x.createLinearGradient(0,0,0,h);g.addColorStop(0,col+'2a');g.addColorStop(1,col+'00');x.fillStyle=g;x.fill()}
const fmtT=s=>{s=Math.max(0,Math.floor(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60),ss=s%60;return (h?h+':'+String(m).padStart(2,'0'):m)+':'+String(ss).padStart(2,'0')};
const pctl=(a,q)=>{if(!a.length)return null;const s=a.slice().sort((p,r)=>p-r);return s[Math.min(s.length-1,Math.floor(q*s.length))]};
// ---- state
let stripSeq=-1,curStep=null,curCycle=null,lastS=null,durs={},latH=[],gH=[],lastLat=null;
function flow(lp){const step=lp.step==='release'?'place':lp.step,order=STEPS.map(s=>s[0]),cur=order.indexOf(step),act=lp.mode==='running'||lp.mode==='paused';
 if(lp.cycle!==curCycle){curCycle=lp.cycle;durs={}}
 if(step!==curStep){if(curStep&&lastS!=null)durs[curStep]=lastS;curStep=step}
 lastS=lp.step_s;$('flow').classList.toggle('paused',lp.mode==='paused');
 order.forEach((s,i)=>{const c=$('c_'+s),on=act&&s===step,done=act&&cur>=0&&i<cur;
  c.className='st'+(on?' on':done?' done':'');
  c.querySelector('em').textContent=on?`${(lp.step_s??0).toFixed(0)}s`:(done&&durs[s]!=null?`${durs[s].toFixed(0)}s`:'')});
}
function offline(){$('pill').className='pill lost';setT('mode','Offline');setT('cyc',R.proxy?' · loop on 8020 not reachable':'')}
function kpis(r,lp){const tally=lp.tallies||{},tot=Object.values(tally).reduce((a,b)=>a+b,0),cy=lp.cycles||[];
 setT('k_tot',String(tot));
 const run=lp.started?Date.now()/1000-lp.started:null;
 setT('k_up',run!=null&&lp.mode!=='off'?fmtT(run):'');
 const rate=run&&run>30?tot/(run/60):null;setH('k_rate',(rate!=null?rate.toFixed(1):'–')+'<sub>/min</sub>');
 const fin=cy.filter(c=>c.result&&c.result!=='running'),ok=fin.filter(c=>c.result==='sorted'),secs=ok.map(c=>c.seconds).filter(v=>v!=null);
 setH('k_avg',(secs.length?(secs.reduce((a,b)=>a+b,0)/secs.length).toFixed(1):'–')+'<sub>s</sub>');
 setH('k_ok',(fin.length?Math.round(ok.length/fin.length*100):'–')+'<sub>%</sub>');
 setH('k_lat',(r.latency_ms!=null?Math.round(r.latency_ms):'–')+'<sub>ms</sub>');
 setH('plates',['yellow','orange','purple'].map(c=>`<span><i class="dot" style="background:${COL[c]}"></i>${c}<b>${tally[c]||0}</b></span>`).join('')+`<em>${tot?'plate mix · '+tot+' sorted':'plate mix'}</em>`);
 drawCycles(cy)}
function drawCycles(cy){const f=fit($('cyc_chart'));if(!f)return;const [x,w,h]=f,list=cy.slice(-14),padL=26,padB=16,ch=h-padB-4;
 const est=Object.values(durs).reduce((a,b)=>a+b,0)+(lastS||0);
 const tops=list.map(c=>c.result==='running'?est:(c.seconds||0)),mx=Math.max(30,...tops)*1.12,step=mx>90?30:mx>45?15:10;
 x.font=`400 9px ${MONO}`;x.textAlign='right';
 for(let v=0;v<=mx;v+=step){const y=Math.round(4+ch-v/mx*ch)+.5;x.strokeStyle=v?'rgba(255,255,255,.05)':'rgba(255,255,255,.14)';x.lineWidth=1;x.beginPath();x.moveTo(padL,y);x.lineTo(w,y);x.stroke();x.fillStyle='#4f4f57';x.fillText(v+'s',padL-5,y+3)}
 x.textAlign='left';
 if(!list.length){x.fillStyle='#4f4f57';x.font=`400 10px ${MONO}`;x.fillText('no cycles yet',padL+8,4+ch/2);return}
 const slots=Math.max(list.length,10),sw=(w-padL)/slots,bw=Math.min(26,sw*.58),Y=v=>4+ch-v/mx*ch;
 const okS=list.filter(c=>c.result==='sorted'&&c.seconds!=null).map(c=>c.seconds);
 list.forEach((c,i)=>{const X=padL+sw*i+(sw-bw)/2;
  if(c.result==='running'){if(est>0){x.strokeStyle='rgba(255,255,255,.4)';x.setLineDash([3,3]);x.strokeRect(X+.5,Y(est)+.5,bw-1,ch-(Y(est)-4)-1);x.setLineDash([])}}
  else{const tot=c.seconds||0,g=c.t_grasp,ins=c.inspect_s;
   if(g!=null&&ins!=null&&tot>0){const rest=Math.max(0,tot-g-ins);[[g,'#e4e4e7'],[ins,'#8a8a93'],[rest,'#3f3f46']].reduce((b,[v,col])=>{x.fillStyle=col;x.fillRect(X,Y(b+v),bw,Math.max(0,Y(b)-Y(b+v)-1));return b+v},0)}
   else{x.fillStyle='#5a5a62';x.fillRect(X,Y(tot),bw,Y(0)-Y(tot))}
   if(c.result!=='sorted'){x.fillStyle='#ef4444';x.fillRect(X,Y(tot)-3,bw,2)}}
  x.fillStyle=COL[c.bowl]||'#2f2f35';x.beginPath();x.arc(X+bw/2,h-padB+6,2.5,0,7);x.fill();
  x.fillStyle='#4f4f57';x.font=`400 9px ${MONO}`;x.textAlign='center';if(sw>16)x.fillText(String(c.n),X+bw/2,h-1);x.textAlign='left'});
 if(okS.length){const av=okS.reduce((a,b)=>a+b,0)/okS.length,y=Math.round(Y(av))+.5;x.strokeStyle='rgba(74,222,128,.55)';x.setLineDash([4,3]);x.beginPath();x.moveTo(padL,y);x.lineTo(w,y);x.stroke();x.setLineDash([]);
  const t=`avg ${av.toFixed(1)}s`;x.font=`500 9px ${MONO}`;const tw=x.measureText(t).width;x.fillStyle='rgba(12,12,14,.9)';x.fillRect(w-tw-8,y-12,tw+8,11);x.fillStyle='#4ade80';x.fillText(t,w-tw-4,y-3.5)}}
function policy(r){const ph=r.phase||'idle',sm=r.studio||{};setT('vph',[ph,sm.backend,sm.device].filter(Boolean).join(' · '));
 const pr=$('prompt');pr.className='instr'+(ph==='running'?'':' idle');setT('prompt',ph==='running'?`“${r.prompt}”`:(ph==='inspecting'?'Paused while inspecting':'Idle'));
 const m=r.measured,c=r.cmd;
 for(let i=0;i<6;i++){const mi=m?m[i]:null,ci=c?c[i]:null;
  $('jm'+i).style.display=mi==null?'none':'';$('jc'+i).style.display=ci==null?'none':'';
  if(mi!=null)$('jm'+i).style.left=jpos(i,mi)+'%';if(ci!=null)$('jc'+i).style.left=jpos(i,ci)+'%';
  const d=$('jd'+i);if(mi!=null&&ci!=null){const a=jpos(i,mi),b=jpos(i,ci);d.style.left=Math.min(a,b)+'%';d.style.width=Math.abs(a-b)+'%';d.style.display=''}else d.style.display='none';
  setT('jv'+i,ci!=null?(i<5&&ci>0?'+':'')+ci.toFixed(1):mi!=null?mi.toFixed(1):'–')}
 if(m&&c){const err=Math.max(...m.slice(0,5).map((v,i)=>Math.abs(v-c[i])));setT('sync',`max err ${err.toFixed(1)}°`)}else setT('sync','');
 setH('med',r.latency_median_ms!=null?Math.round(r.latency_median_ms)+'<sub>ms</sub>':'–');
 setH('age',r.chunk_age_s!=null?r.chunk_age_s.toFixed(2)+'<sub>s</sub>':'–');setT('chk',String(r.chunks??'–'));setT('q',String(r.queued??'–'));
 setT('mname',sm.name||'');
 resid(r.residual);
 if(r.latency_ms!=null&&(r.latency_ms!==lastLat||r.chunks!=null)){latH.push(r.latency_ms);lastLat=r.latency_ms;if(latH.length>240)latH.shift()}
 drawLat()}
function drawLat(){const p=pctl(latH,.95);setH('p95',p!=null?Math.round(p)+'<sub>ms</sub>':'–');
 spark('k_spark',latH,'#a1a1aa',Math.max(1,...latH)*1.15)}
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
function gripper(lp){const g=lp.grasp||{},cfg=lp.config||{hold:[4,9]},sc=v=>clamp(v/40*100,0,100);
 setT('gstate',g.state||'–');$('gs').className='gs '+(g.state||'');setT('hb',`${cfg.hold[0]}–${cfg.hold[1]}`);
 setT('gnum',g.pos!=null?`pos ${g.pos} · cmd ${g.cmd}`:'');
 $('band').style.left=sc(cfg.hold[0])+'%';$('band').style.width=(sc(cfg.hold[1])-sc(cfg.hold[0]))+'%';
 if(g.pos!=null){$('gpos').style.left=sc(g.pos)+'%';$('gcmd').style.left=sc(g.cmd)+'%'}
 gH.push(g.pos!=null?[g.pos,g.cmd]:null);if(gH.length>200)gH.shift();
 const f=fit($('gtrail'));if(!f)return;const [x,w,h]=f,Y=v=>h-1-clamp(v/40,0,1)*(h-2),n=200;
 x.fillStyle='rgba(74,222,128,.08)';x.fillRect(0,Y(cfg.hold[1]),w,Y(cfg.hold[0])-Y(cfg.hold[1]));
 [[1,'rgba(74,222,128,.75)',1],[0,'#f4f4f5',1.25]].forEach(([k,col,lw])=>{x.beginPath();let pen=false;gH.forEach((p,i)=>{if(!p){pen=false;return}const X=(n-gH.length+i)/(n-1)*w;pen?x.lineTo(X,Y(p[k])):x.moveTo(X,Y(p[k]));pen=true});x.strokeStyle=col;x.lineWidth=lw;x.stroke()})}
// Plate routing: each outcome row is a control. Rows follow the inspector (held_v5 is binary: plain / marked / uncertain).
function drawRoute(){const t=rtS.tab;if(!t)return;
 const vis=rtS.bin===true?['plain','marked','uncertain']:['plain','circle','cross','uncertain'];
 const dirty=Object.keys(RT_DEF).some(k=>t[k]!==RT_DEF[k]);
 setH('route',RT_ALL.filter(r=>vis.includes(r[0])).map(r=>{const o=r[0],c=t[o];
  return `<div class="rt${rtS.out===o?' on':''}" style="--c:${COL[c]}"><span>${r[1]} <span style="opacity:.6">· ${r[2]}</span></span><u></u><div class="seg" role="group" aria-label="Plate for ${o}">${RT_PLATES.map(k=>`<button style="--c:${COL[k]}" class="${k===c?'sel':''}" title="${o} → ${k} plate" aria-pressed="${k===c}" ${rtS.ro?'disabled':''} onclick="SR('${o}','${k}')"><i></i></button>`).join('')}</div><b>${c}</b></div>`}).join('')
  +`<div class="rtf"><span>${rtS.ro?'Routing not supported by this loop':'Click a colour to pick the plate'}</span><button class="${dirty?'dirty':''}" ${rtS.ro?'disabled':''} onclick="SR('reset')">Reset</button></div>`)}
function SR(o,c){if(rtS.ro)return;if(o==='reset')rtS.tab={...RT_DEF};else rtS.tab={...rtS.tab,[o]:c};rtS.hold=Date.now()+700;drawRoute();post(o==='reset'?'routing/reset':'routing/'+o+'/'+c)}
function inspection(r,lp,pc){const ins=r.inspection||{},fr=lp.inspect_frames||[],ph=r.phase||'idle',spin=ph==='inspecting'||ph==='collecting';
 $('insp').classList.toggle('spin',spin);setT('ifr',fr.length?`${fr.length} frames`:'');
 const ins0=pc.inspect;setT('pinsp',spin?'spinning · NPU':(ins0&&ins0.held?'brick held':''));
 if(fr.length!==stripSeq){stripSeq=fr.length;const s=$('strip');if(fr.length){s.onload=()=>s.hidden=false;s.src=R.strip+'?'+Date.now()}else s.hidden=true}
 $('dsw').classList.toggle('run',ins.state==='running');
 const sl=pc.spin_last;setT('v_live',sl&&sl.age_s<2?`${sl.label} ${Number(sl.conf).toFixed(2)}`:'–');
 if(ins.state==='running'){setT('dout','Inspecting');setT('dwhy','Spinning the brick in front of the camera');$('dsw').style.background='var(--panel2)';setT('dplate','–')}
 else if(ins.outcome){setT('dout',LBL[ins.outcome]||ins.outcome);$('dsw').style.background=COL[ins.bowl]||'#333';setT('dplate',ins.bowl||'–');
  setT('dwhy',ins.reason||'');setT('v_fr',ins.frames!=null?String(ins.frames):'–');setT('v_s',ins.seconds!=null?ins.seconds+' s':'–');
  setT('v_at',ins.attempts!=null?String(ins.attempts)+(ins.early_stop?' · early':''):'–')}
 rtS.ro=!r.routing;rtS.bin=r.routing_binary??null;rtS.out=ins.state==='running'||!ins.outcome?null:(RT_OUT[ins.outcome]||ins.outcome);
 if(Date.now()>=rtS.hold)rtS.tab=r.routing?{...RT_DEF,...r.routing}:{...RT_DEF};
 drawRoute()}
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
 setT('ptop',pc.age_s!=null&&pc.age_s>1?`survey ${pc.age_s}s ago`:'');
 const cy=lp.cycles||[];setT('tot',cy.length?`last ${cy.length} cycles`:'');
 setH('cycles',cy.slice().reverse().map(c=>`<tr><td class="n faint">${c.n}</td><td><i class="dot" style="background:${COL[c.brick]||'#ccc'}"></i>${esc(c.brick)}${c.grip!=null?`<small>grip ${c.grip}</small>`:''}</td><td>${c.outcome?(LBL[c.outcome]||c.outcome)+(c.frames!=null?`<small>${c.frames}f</small>`:''):'–'}</td><td>${c.bowl?`<i class="dot" style="background:${COL[c.bowl]}"></i>${c.bowl}`:'–'}</td><td class="n">${c.seconds!=null?c.seconds+' s':'–'}</td><td class="${c.result==='sorted'?'ok':c.result==='running'?'mute':'bad'}">${esc(c.result||'')}</td></tr>`).join('')||'<tr><td colspan="6" class="faint">No cycles yet. Press Start.</td></tr>');
 const evs=lp.events||[];setT('evn',evs.length?String(evs.length):'');
 setH('ev',evs.map(e=>`<div class="k-${esc(e.kind||'')}"><time>${e.t?new Date(e.t*1000).toLocaleTimeString([],{hour12:false}):''}</time><span>${esc(e.message||[e.kind,e.step||e.bowl||e.outcome].filter(Boolean).join(' '))}</span></div>`).join('')||'<div class="faint">Nothing yet</div>');
 drawTop(pc,lp.step);drawInsp(r);drawPose(r.inspect_pose);drawCollect(r.collect)}
const SYS=[['gpu','gpu','#f4f4f5'],['npu','npu','#f4f4f5'],['cpu','cpu','#c9c9d0'],['mem','mem_pct','#8a8a93']];
async function sys(){let s;try{s=await fetch(R.sys).then(x=>{if(!x.ok)throw 0;return x.json()})}catch(e){return}
 const h=s.history||{},pct=v=>v!=null?Math.round(v)+'<sub>%</sub>':'–';
 SYS.forEach(([k,hk,col])=>{const v=s[k+'_pct'],b=$('a_'+k);setH('u_'+k,pct(v));b.style.width=(v!=null?clamp(v,0,100):0)+'%';b.classList.toggle('hot',v>=95);spark('s_'+k,h[hk],col)});
 setT('u_memgb',s.mem_used_gb!=null?`${s.mem_used_gb.toFixed(1)}/${Math.round(s.mem_total_gb)} GB`:'memory');
 setT('u_freq',s.gpu_freq_mhz!=null?`GPU ${s.gpu_freq_mhz}${s.gpu_max_mhz?'/'+s.gpu_max_mhz:''} MHz`:'');setT('u_temp',s.cpu_temp_c!=null?`CPU ${Math.round(s.cpu_temp_c)}°C`:'');setT('u_rss',s.proc_rss_gb!=null?`loop ${s.proc_rss_gb.toFixed(1)} GB`:'')}
function box(c,b,col,w,label,sq){c.strokeStyle=col;c.lineWidth=w;c.beginPath();if(c.roundRect)c.roundRect(b[0],b[1],b[2]-b[0],b[3]-b[1],3);else c.rect(b[0],b[1],b[2]-b[0],b[3]-b[1]);c.stroke();
 if(!label)return;c.font='500 13px Inter,system-ui';const pad=sq?16:6,tw=c.measureText(label).width,y=Math.max(0,b[1]-20);
 c.fillStyle='rgba(6,6,7,.82)';c.beginPath();if(c.roundRect)c.roundRect(b[0],y,tw+pad+6,18,3);else c.rect(b[0],y,tw+pad+6,18);c.fill();
 if(sq){c.fillStyle=sq;c.fillRect(b[0]+6,y+5.5,7,7)}c.fillStyle='#f4f4f5';c.fillText(label,b[0]+pad,y+13.5)}
function drawTop(pc,step){const c=$('ovtop').getContext('2d');c.clearRect(0,0,640,480);const ws=pc.workspace||[0,60,560,480];
 c.fillStyle='rgba(6,6,7,.6)';c.fillRect(0,0,640,ws[1]);c.fillRect(ws[2],ws[1],640-ws[2],480-ws[1]);
 (pc.plates||[]).forEach(p=>{c.strokeStyle=COL[p.colour]||'#ccc';c.globalAlpha=.6;c.lineWidth=2;c.setLineDash([5,6]);c.beginPath();c.arc(p.cx,p.cy,p.r,0,7);c.stroke();c.setLineDash([]);c.globalAlpha=1});
 (pc.sorted||[]).forEach(b=>box(c,b.box,'rgba(255,255,255,.2)',1));
 (pc.table||[]).forEach(b=>box(c,b.box,'rgba(250,250,250,.85)',2,Number(b.conf).toFixed(2),COL[b.colour]||'#ccc'));
 if(!['survey','verify','idle','done','paused'].includes(step)){c.font='500 13px Inter,system-ui';c.fillStyle='rgba(255,255,255,.5)';c.fillText('LAST SURVEY · DETECTOR PAUSED WHILE THE ARM MOVES',12,ws[1]+20)}}
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
const clk=()=>setT('clk',new Date().toLocaleTimeString([],{hour12:false}));
setInterval(tick,150);tick();setInterval(sys,500);sys();setInterval(clk,1000);clk();
</script></body></html>
"""

if __name__ == "__main__":
    sys.stdout.write('"""Brick-sorting console page served by loop_control.py at /. Generated by dlsj.live_console; edit that."""\n\n')
    sys.stdout.write("PAGE = " + repr(page(proxy=False)) + "\n")
