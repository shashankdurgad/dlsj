"""The presentation slide for the brick-sorting demo, served at /brief next to the live console on port 8020.

A fixed 16:9 canvas (1920x1080) scaled to fit any screen. It polls the same-origin endpoints /ctl/state (loop state)
and /sys/stats (laptop utilisation), both optional: the slide stays complete and quiet when either is missing.
Stdlib only and independent of the rest of the package, so it can be shipped as a single file.

Write the 8020 module with:  python -m dlsj.brief_slide > brief_page.py
"""

from __future__ import annotations

import sys


def page() -> str:
    return TEMPLATE


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>DLSJ · Brief</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0b0c0f;--panel:#14161b;--panel2:#1c1f26;--hover:#252932;--line:rgba(255,255,255,.06);--fg:#f1f2f4;--mute:#959ba5;--faint:#5b616b;
--ok:#34d399;--warn:#fbbf24;--red:#ef4444;--blue:#60a5fa;--yellow:#facc15;--orange:#fb923c;--purple:#a78bfa;
--cpu:#959ba5;--gpu:#60a5fa;--npu:#34d399}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{background:var(--bg);color:var(--fg);font:400 22px/1.4 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased;font-feature-settings:"tnum","cv11"}
a{color:inherit;text-decoration:none}
#slide{position:absolute;left:50%;top:50%;width:1920px;height:1080px;transform:translate(-50%,-50%) scale(var(--s,1));transform-origin:center center;
 padding:44px 96px 40px;display:flex;flex-direction:column;gap:0}
#slide>*{flex:none}
/* header */
.top{display:flex;align-items:center;justify-content:space-between;height:40px}
.brand{font-weight:600;font-size:22px;letter-spacing:.26em}
.right{display:flex;align-items:center;gap:28px}
.pill{display:flex;align-items:center;gap:12px;padding:8px 20px 8px 16px;border-radius:99px;background:var(--panel);font-size:20px;color:var(--mute);white-space:nowrap}
.pill i{width:10px;height:10px;border-radius:50%;background:var(--faint)}
.pill b{font-weight:500;color:var(--fg)}
.pill.running i{background:var(--ok);animation:pulse 1.8s infinite}.pill.paused i{background:var(--warn)}.pill.done i{background:var(--blue)}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.5)}70%{box-shadow:0 0 0 10px rgba(52,211,153,0)}100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}}
.back{font-size:18px;color:var(--faint);transition:color .15s}.back:hover{color:var(--fg)}
/* hero */
.hero{display:flex;justify-content:space-between;align-items:stretch;gap:64px;margin-top:34px;height:216px}
.hero .t{display:flex;flex-direction:column;justify-content:space-between;padding:0;min-width:0}
h1{font-weight:300;font-size:90px;line-height:1.02;letter-spacing:-.04em}
h1 span{color:var(--faint)}
.sub{font-size:25px;color:var(--mute);font-weight:400;letter-spacing:-.005em}
.sub i{font-style:normal;color:var(--faint);margin:0 10px}
.armcard{position:relative;flex:none;width:720px;border-radius:22px;overflow:hidden;background:#1b1e25;border:1px solid var(--line)}
.armcard canvas{position:absolute;left:0;top:-53%;width:100%;height:190%;display:block}
.armcard span{position:absolute;left:24px;top:20px;font-size:17px;color:var(--faint);letter-spacing:.01em}
.armcard[hidden]{display:none}
/* loop */
.loop{position:relative;margin-top:30px}
.row{display:grid;grid-template-columns:repeat(5,1fr);gap:30px}
.st{position:relative;display:flex;flex-direction:column;min-width:0;height:352px;padding:24px 28px 18px;border-radius:24px;background:var(--panel);border:1px solid var(--line);
 transition:border-color .35s,background .35s,box-shadow .35s}
.st:not(:last-child)::after{content:"";position:absolute;right:-22px;top:50%;width:12px;height:12px;margin-top:-6px;border-top:2px solid var(--faint);border-right:2px solid var(--faint);transform:rotate(45deg)}
.st.soft{border-color:rgba(255,255,255,.12)}
.run .st:not(.on){opacity:.66}.run .st.soft{opacity:.88}
.st.on{background:var(--panel2);border-color:rgba(52,211,153,.7);box-shadow:0 0 0 1px rgba(52,211,153,.35),0 0 70px rgba(52,211,153,.12)}
.paused .st.on{border-color:rgba(251,191,36,.7);box-shadow:0 0 0 1px rgba(251,191,36,.3),0 0 70px rgba(251,191,36,.1)}
.st>*{flex:none}
.st u{position:absolute;left:28px;right:28px;top:0;height:3px;border-radius:0 0 3px 3px;overflow:hidden;opacity:0;transition:opacity .3s}
.st.on u{opacity:1}
.st u::after{content:"";position:absolute;inset:0;width:36%;background:var(--ok);animation:slide 1.7s ease-in-out infinite}
.paused .st.on u::after{animation:none;width:100%;background:var(--warn)}
@keyframes slide{0%{transform:translateX(-100%)}100%{transform:translateX(300%)}}
.nm{display:flex;align-items:baseline;font-size:18px;font-weight:500;letter-spacing:.16em;text-transform:uppercase;color:var(--faint);transition:color .3s}
.nm em{font-style:normal;letter-spacing:0;font-weight:400;font-size:18px}
.st.on .nm{color:var(--ok)}.paused .st.on .nm{color:var(--warn)}
.verb{margin-top:12px;font-size:42px;font-weight:300;letter-spacing:-.03em;line-height:1.1;white-space:nowrap}
.model{margin-top:auto}
.model.one{display:flex;align-items:center;justify-content:space-between;gap:12px}
.model.one .chips{margin-top:0}
.model b{display:block;font-size:26px;font-weight:500;letter-spacing:-.015em;line-height:1.15}
.chips{display:flex;gap:7px;margin-top:12px}
.chip{display:inline-flex;align-items:center;height:30px;padding:0 12px;border-radius:99px;font-size:15px;font-weight:600;letter-spacing:.1em;background:var(--panel2);border:1px solid var(--line)}
.on .chip{background:var(--bg)}
.chip.cpu{color:var(--cpu)}.chip.gpu{color:var(--gpu)}.chip.npu{color:var(--npu)}
.fact{margin-top:10px;font-size:20px;color:var(--mute);line-height:1.35;height:54px;overflow:hidden}
.live{margin-top:12px;padding-top:12px;border-top:1px solid var(--line);font-size:20px;color:var(--faint);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;height:44px;transition:color .3s}
.on .live{color:var(--fg)}
.rules{margin-top:12px;display:grid;gap:1px;font-size:20px;color:var(--mute);align-content:start}
.rules div{display:flex;align-items:center;gap:10px;transition:opacity .3s,color .3s}
.rules i{width:11px;height:11px;border-radius:50%;flex:none}
.rules em{font-style:normal;color:var(--faint);margin-left:auto;font-size:17px}
.rules i.ring{background:none!important;border:2px solid var(--purple)}
.rules.pick div{opacity:.35}.rules.pick div.hit{opacity:1;color:var(--fg)}
.back-path{position:absolute;left:0;right:0;bottom:-46px;height:46px;overflow:visible;pointer-events:none}
.back-path path{fill:none;stroke:var(--line);stroke-width:2;stroke-dasharray:3 9;stroke-linecap:round}
.back-path .head{stroke-dasharray:none;stroke:var(--faint)}
.running .back-path path{stroke:rgba(255,255,255,.18);animation:dash 1.6s linear infinite}
.running .back-path .head{animation:none;stroke:var(--mute)}
@keyframes dash{to{stroke-dashoffset:-24}}
.back-lbl{position:absolute;left:50%;bottom:-48px;transform:translateX(-50%);padding:0 18px;background:var(--bg);font-size:17px;letter-spacing:.14em;text-transform:uppercase;color:var(--faint)}
/* live tiles */
.tiles{display:grid;grid-template-columns:1.55fr 1fr 1fr 1fr 1fr;gap:30px;margin-top:60px;height:178px}
.tile{position:relative;border-radius:22px;background:var(--panel);border:1px solid var(--line);padding:20px 26px 0;overflow:hidden;display:flex;flex-direction:column}
.lbl{display:flex;justify-content:space-between;align-items:baseline;font-size:16px;font-weight:500;letter-spacing:.16em;text-transform:uppercase;color:var(--faint)}
.lbl em{font-style:normal;letter-spacing:0;text-transform:none;font-weight:400;font-size:17px;color:var(--mute)}
.lbl .d{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:9px;vertical-align:1px}
.big{margin-top:6px;font-size:64px;font-weight:300;letter-spacing:-.04em;line-height:1}
.big small{font-size:24px;color:var(--mute);letter-spacing:0;margin-left:6px;font-weight:400}
.big{display:flex;align-items:baseline}
.big em{font-style:normal;margin-left:auto;font-size:17px;font-weight:400;color:var(--faint);letter-spacing:0;white-space:nowrap}
.spark{display:block;margin:auto -26px 0;width:calc(100% + 52px);height:50px}
.sorted{flex-direction:row;align-items:center;gap:34px;padding-bottom:20px}
.sorted .tot{display:flex;flex-direction:column;justify-content:space-between;height:100%}
.sorted .big{font-size:84px;margin-top:6px}
.plates{display:flex;gap:30px;margin-left:auto;align-items:flex-end;height:100%}
.plates div{display:flex;flex-direction:column;gap:2px;align-items:flex-start;min-width:64px}
.plates b{font-size:50px;font-weight:300;letter-spacing:-.03em;line-height:1}
.plates span{font-size:17px;color:var(--mute);display:flex;align-items:center;gap:8px;text-transform:capitalize}
.plates i{width:11px;height:11px;border-radius:50%}
/* facts */
.facts{display:grid;grid-template-columns:repeat(4,1fr);gap:30px;margin-top:auto;height:50px}
.facts div{display:flex;align-items:center;gap:18px;min-width:0}
.facts b{font-size:42px;font-weight:300;letter-spacing:-.035em;color:var(--fg);white-space:nowrap}
.facts b small{font-size:22px;color:var(--mute);letter-spacing:0;margin-left:3px}
.facts span{font-size:18px;color:var(--faint);line-height:1.3}
</style></head><body>
<main id="slide">
 <header class="top">
  <span class="brand">DLSJ</span>
  <div class="right"><span class="pill" id="pill"><i></i><b id="mode">Standby</b><span id="cyc"></span></span><a class="back" href="/" title="Live console">Console ↗</a></div>
 </header>
 <section class="hero">
  <div class="t"><h1>Inspect every brick.<br><span>Sort it right.</span></h1>
   <p class="sub">SO-101 arm<i>/</i>Intel Physical AI Studio<i>/</i>OpenVINO<i>/</i>Core Ultra</p></div>
  <div class="armcard" id="armcard" hidden><canvas id="arm"></canvas><span>Learned from teleoperation</span></div>
 </section>
 <section class="loop" id="loop" aria-label="Closed loop">
  <div class="row">
   <div class="st" id="s_perceive"><u></u><div class="nm">Perceive</div><div class="verb">See</div>
    <div class="model"><b>YOLO26n<br>brick detector</b><span class="chips"><span class="chip npu">NPU</span></span></div>
    <div class="fact">Top camera: colour and position of each brick</div><div class="live" id="l_perceive">–</div></div>
   <div class="st" id="s_detect"><u></u><div class="nm">Detect</div><div class="verb">Inspect</div>
    <div class="model"><b>YOLO26n<br>held-brick v5</b><span class="chips"><span class="chip npu">NPU</span></span></div>
    <div class="fact">Plain or marked, while the wrist spins the brick</div><div class="live" id="l_detect">–</div></div>
   <div class="st" id="s_reason"><u></u><div class="nm">Reason</div><div class="verb">Decide</div>
    <div class="model one"><b>Rule table</b><span class="chips"><span class="chip cpu">CPU</span></span></div>
    <div class="rules" id="rules"><div data-b="yellow"><i style="background:var(--yellow)"></i>Plain<em>yellow plate</em></div><div data-b="orange"><i style="background:var(--orange)"></i>Circle<em>orange plate</em></div><div data-b="purple"><i style="background:var(--purple)"></i>Cross<em>purple plate</em></div><div data-b="unsure"><i class="ring"></i>Unsure<em>re-spin, then purple</em></div></div>
    <div class="live" id="l_reason">–</div></div>
   <div class="st" id="s_act"><u></u><div class="nm">Act</div><div class="verb">Pick &amp; place</div>
    <div class="model"><b>MolmoAct2<br>fine-tuned VLA</b><span class="chips"><span class="chip gpu">GPU</span></span></div>
    <div class="fact">One model, three prompts, real-time chunking</div><div class="live" id="l_act">–</div></div>
   <div class="st" id="s_optimize"><u></u><div class="nm">Optimize</div><div class="verb">Tune</div>
    <div class="model"><b>OpenVINO<br>INT8 + RTC</b><span class="chips"><span class="chip cpu">CPU</span><span class="chip gpu">GPU</span><span class="chip npu">NPU</span></span></div>
    <div class="fact">Chunk time: FP16 617 ms, INT8 348 ms</div><div class="live" id="l_optimize">–</div></div>
  </div>
  <svg class="back-path" id="bp" viewBox="0 0 1728 46" preserveAspectRatio="none"><path d="M1555 4 V26 Q1555 40 1541 40 H187 Q173 40 173 26 V10"/><path class="head" d="M165 17 L173 7 L181 17"/></svg>
  <span class="back-lbl">Next brick</span>
 </section>
 <section class="tiles">
  <div class="tile sorted"><div class="tot"><div class="lbl">Sorted</div><div class="big" id="total">0</div></div>
   <div class="plates" id="plates"></div></div>
  <div class="tile"><div class="lbl">Latency<em>per chunk</em></div><div class="big"><span id="lat_med">–</span><small>ms</small><em id="lat_last"></em></div>
   <svg class="spark" id="sp_lat" viewBox="0 0 100 40" preserveAspectRatio="none"></svg></div>
  <div class="tile"><div class="lbl"><span><i class="d" style="background:var(--gpu)"></i>GPU</span><em>Arc B390</em></div><div class="big"><span id="gpu_v">–</span><small>%</small><em id="gpu_s"></em></div>
   <svg class="spark" id="sp_gpu" viewBox="0 0 100 40" preserveAspectRatio="none"></svg></div>
  <div class="tile"><div class="lbl"><span><i class="d" style="background:var(--npu)"></i>NPU</span><em>2 models</em></div><div class="big"><span id="npu_v">–</span><small>%</small><em id="npu_s"></em></div>
   <svg class="spark" id="sp_npu" viewBox="0 0 100 40" preserveAspectRatio="none"></svg></div>
  <div class="tile"><div class="lbl"><span><i class="d" style="background:var(--cpu)"></i>CPU</span><em id="cpu_t">cameras, control</em></div><div class="big"><span id="cpu_v">–</span><small>%</small><em id="cpu_s"></em></div>
   <svg class="spark" id="sp_cpu" viewBox="0 0 100 40" preserveAspectRatio="none"></svg></div>
 </section>
 <footer class="facts">
  <div><b>456</b><span>teleoperated demos<br>fine-tune the VLA</span></div>
  <div><b>37<small>/38</small></b><span>rig spins judged right<br>plain vs marked</span></div>
  <div><b>92.9<small>%</small></b><span>Anomalib PatchCore, trained<br>on plain bricks only</span></div>
  <div><b>1.8<small>×</small></b><span>faster chunks, INT8<br>against FP16 on the Arc GPU</span></div>
 </footer>
</main>
<script type="importmap">{"imports":{"three":"/vendor/three.module.js"}}</script>
<script type="module" src="/demo_arm.js?v=4"></script>
<script>
const $=id=>document.getElementById(id);
const COL={yellow:'#facc15',orange:'#fb923c',purple:'#a78bfa'};
const LBL={none:'plain',circle:'circle',cross:'cross',marked:'marked',unknown:'uncertain'};
const STAGE={survey:'perceive',verify:'perceive',inspect:'detect',pick:'act',bring:'act',tune:'act',place:'act',release:'act',home:'act'};
const NAMES={survey:'Survey',verify:'Verify',inspect:'Inspect',pick:'Pick',bring:'Bring',tune:'Align',place:'Place',release:'Place',home:'Home'};
const STAGES=['perceive','detect','reason','act','optimize'];
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const setT=(id,t)=>{const e=$(id);if(e&&e.textContent!==t)e.textContent=t};
/* fit the 1920x1080 canvas to the window */
function fit(){const s=Math.min(innerWidth/1920,innerHeight/1080);document.documentElement.style.setProperty('--s',s)}
addEventListener('resize',fit);fit();
/* the arm card only shows when the arm assets are served here */
fetch('/so101/kin.json').then(r=>{if(r.ok)$('armcard').hidden=false}).catch(()=>{});
/* sparklines */
function spark(id,vals,col,max){const e=$(id);if(!e)return;const v=(vals||[]).filter(x=>x!=null&&isFinite(x)).slice(-60);
 if(v.length<2){e.innerHTML='';return}
 let lo=0,hi=max;
 if(!hi){const a=Math.min(...v),b=Math.max(...v),span=Math.max(b-a,b*.25,1),mid=(a+b)/2;lo=mid-span/2;hi=mid+span/2}
 const n=v.length-1,pts=v.map((x,i)=>`${(i/n*100).toFixed(1)},${(38-Math.max(0,Math.min(1,(x-lo)/(hi-lo)))*34).toFixed(1)}`);
 e.innerHTML=`<defs><linearGradient id="g_${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${col}" stop-opacity=".28"/><stop offset="1" stop-color="${col}" stop-opacity="0"/></linearGradient></defs>`+
  `<polygon points="0,40 ${pts.join(' ')} 100,40" fill="url(#g_${id})"/><polyline points="${pts.join(' ')}" fill="none" stroke="${col}" stroke-width="1.6" vector-effect="non-scaling-stroke" stroke-linejoin="round" stroke-linecap="round"/>`}
/* loop state */
let reasonUntil=0,lastKey=null,first=true,latHist=[];
function setStage(active,soft,paused){
 $('loop').classList.toggle('paused',!!paused);$('loop').classList.toggle('run',active!=null);
 STAGES.forEach(s=>{const e=$('s_'+s);e.classList.toggle('on',s===active);e.classList.toggle('soft',s===soft&&s!==active)})}
function offline(){$('pill').className='pill';setT('mode','Standby');setT('cyc','');$('slide').classList.remove('running');setStage(null,null,false);
 ['perceive','detect','reason','act','optimize'].forEach(s=>setT('l_'+s,'–'));
 $('rules').classList.remove('pick');[...$('rules').children].forEach(d=>d.classList.remove('hit'));
 setT('total','–');setT('lat_med','–');setT('lat_last','');latHist=[];spark('sp_lat',[],'#f1f2f4');
 $('plates').dataset.h='';$('plates').innerHTML=['yellow','orange','purple'].map(c=>`<div><b>–</b><span><i style="background:${COL[c]}"></i>${c}</span></div>`).join('')}
async function get(url){const c=new AbortController(),t=setTimeout(()=>c.abort(),1500);
 try{const r=await fetch(url,{signal:c.signal,cache:'no-store'});if(!r.ok)throw 0;return await r.json()}finally{clearTimeout(t)}}
async function state(){let r;try{r=await get('/ctl/state')}catch(e){return offline()}
 const lp=r.loop||{},pc=r.perception||{},ins=r.inspection||{},mode=lp.mode||'off',act=mode==='running'||mode==='paused';
 $('pill').className='pill '+mode;
 const step=lp.step==='release'?'place':lp.step;
 setT('mode',{off:'Idle',running:'Running',paused:'Paused',done:'Done'}[mode]||mode);
 setT('cyc',(act&&NAMES[step]?' · '+NAMES[step]:'')+(act&&lp.cycle?` · brick ${lp.cycle}`:''));
 $('slide').classList.toggle('running',mode==='running');
 /* which stage is lit */
 const key=ins.outcome?`${lp.cycle}:${ins.outcome}:${ins.bowl}`:null;
 if(key!==lastKey){if(key&&!first)reasonUntil=Date.now()+2600;lastKey=key}first=false;
 let cur=STAGE[lp.step]||null;
 if(lp.step==='inspect'&&ins.state!=='running'&&ins.outcome)cur='reason';
 if((lp.step==='place'||lp.step==='release')&&Date.now()<reasonUntil)cur='reason';
 setStage(act?cur:null,act?'optimize':null,mode==='paused');
 /* live lines */
 const tb=pc.table||[];
 setT('l_perceive',act||tb.length?`${tb.length} on the table`:'–');
 if(ins.state==='running')setT('l_detect','Spinning the brick');
 else if(ins.outcome)setT('l_detect',`${LBL[ins.outcome]||ins.outcome}`+(ins.frames!=null?` · ${ins.frames} frames`:'')+(ins.attempts>1?' · re-spun':''));
 else setT('l_detect',act?'Waiting for a brick':'–');
 $('rules').classList.toggle('pick',!!(ins.bowl&&ins.outcome&&ins.state!=='running'));
 const hit=ins.outcome==='unknown'?'unsure':ins.bowl;
 [...$('rules').children].forEach(d=>d.classList.toggle('hit',d.dataset.b===hit));
 if(ins.outcome&&ins.bowl&&ins.state!=='running')setT('l_reason',`→ ${ins.bowl} plate`);else setT('l_reason','–');
 const run=r.phase==='running'&&r.prompt;
 setT('l_act',run?`“${r.prompt}”`:(r.phase==='inspecting'?'Paused while inspecting':'–'));
 const med=r.latency_median_ms,lat=r.latency_ms;
 setT('l_optimize',med!=null?`${Math.round(med)} ms median chunk`:'–');
 setT('lat_med',med!=null?Math.round(med):'–');setT('lat_last',lat!=null?`last ${Math.round(lat)}`:'');
 if(lat!=null){latHist.push(lat);if(latHist.length>60)latHist.shift()}
 spark('sp_lat',latHist,'#f1f2f4');
 /* sorted */
 const tally=lp.tallies||{},tot=Object.values(tally).reduce((a,b)=>a+b,0);
 setT('total',tot);
 const ph=['yellow','orange','purple'].map(c=>`<div><b>${tally[c]||0}</b><span><i style="background:${COL[c]}"></i>${c}</span></div>`).join('');
 if($('plates').dataset.h!==ph){$('plates').dataset.h=ph;$('plates').innerHTML=ph}
}
/* laptop utilisation */
const hist={cpu:[],gpu:[],npu:[]};
const pct=v=>v==null||!isFinite(v)?'–':Math.round(v);
async function stats(){let s;try{s=await get('/sys/stats')}catch(e){s=null}
 if(!s)hist.cpu=[],hist.gpu=[],hist.npu=[];
 s=s||{};const h=s.history||{};
 ['cpu','gpu','npu'].forEach(k=>{
  const v=s[k+'_pct'];setT(k+'_v',pct(v));
  if(Array.isArray(h[k])&&h[k].length)hist[k]=h[k].slice(-60);
  else if(v!=null){hist[k].push(v);if(hist[k].length>60)hist[k].shift()}
  spark('sp_'+k,hist[k],k==='gpu'?'#60a5fa':k==='npu'?'#34d399':'#959ba5',100)});
 setT('gpu_s',s.gpu_freq_mhz!=null?`${(s.gpu_freq_mhz/1000).toFixed(1)} GHz`:(s.gpu_temp_c!=null?`${Math.round(s.gpu_temp_c)}°C`:''));
 setT('cpu_s',s.cpu_temp_c!=null?`${Math.round(s.cpu_temp_c)}°C`:'')}
setInterval(state,500);setInterval(stats,500);state();stats();
</script></body></html>
"""

if __name__ == "__main__":
    sys.stdout.write('"""Presentation slide served by loop_control.py at /brief. Generated by dlsj.brief_slide; edit that."""\n\n')
    sys.stdout.write("BRIEF = " + repr(page()) + "\n")
