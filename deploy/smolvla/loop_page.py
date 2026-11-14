"""The closed-loop dashboard page (served by loop_control.py at /). Polls /ctl/state at ~7 Hz."""

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Brick sorting loop</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#121417;--panel:#1b1e22;--panel2:#22262b;--line:#2c3137;--fg:#eef0f2;--mute:#8a919a;--ok:#3ecf8e;
--warn:#f2b33d;--red:#e5484d;--acc:#5aa9ff;--yellow:#f2c230;--orange:#f08a24;--purple:#9b7fe0}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font:400 14px/1.4 ui-sans-serif,system-ui,-apple-system,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:12px 18px;border-bottom:1px solid var(--line);flex-wrap:wrap}
h1{font-size:17px;font-weight:600;letter-spacing:.2px}
.pill{padding:3px 10px;border-radius:999px;background:var(--panel2);color:var(--mute);font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.6px}
.pill.running{background:#123b2a;color:var(--ok)}.pill.paused{background:#3d2e10;color:var(--warn)}.pill.done{background:#14304d;color:var(--acc)}
.sp{flex:1}
button{font:inherit;color:var(--fg);background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:8px 14px;cursor:pointer}
button:hover{border-color:#47505a}button:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
button.go{background:#17462f;border-color:#1f6b45}button.stop{background:var(--red);border-color:var(--red);font-weight:700;padding:8px 22px}
kbd{font:600 11px ui-monospace,monospace;color:var(--mute);margin-left:4px}
.pipe{display:flex;gap:6px;padding:12px 18px;overflow-x:auto;border-bottom:1px solid var(--line)}
.chip{flex:1;min-width:104px;padding:9px 10px;border-radius:10px;background:var(--panel);border:1px solid var(--line);color:var(--mute)}
.chip b{display:block;color:var(--fg);font-size:13px;font-weight:600}.chip span{font-size:11.5px}
.chip.on{border-color:var(--ok);background:#13261e}.chip.on b{color:var(--ok)}.chip.done{opacity:.75}.chip.done b::after{content:" \2713";color:var(--ok)}
.msg{padding:10px 18px;font-size:15px;min-height:42px;border-bottom:1px solid var(--line)}
.grid{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(0,1fr) minmax(300px,.85fr);gap:14px;padding:14px 18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;min-width:0}
.card h2{font-size:12px;font-weight:600;color:var(--mute);text-transform:uppercase;letter-spacing:.7px;margin-bottom:8px;display:flex;justify-content:space-between;gap:8px}
.card h2 em{font-style:normal;color:var(--fg);text-transform:none;letter-spacing:0;font-weight:500}
img.cam{width:100%;display:block;border-radius:8px;background:#000;aspect-ratio:4/3;object-fit:cover}
.camwrap{position:relative}.camwrap canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.tag{position:absolute;left:8px;bottom:8px;background:rgba(0,0,0,.65);padding:3px 8px;border-radius:6px;font-size:12px}
.col{display:flex;flex-direction:column;gap:14px;min-width:0}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:13px}.kv dt{color:var(--mute)}.kv dd{text-align:right;font-variant-numeric:tabular-nums}
.prompt{font-size:15px;padding:8px 10px;background:var(--panel2);border-radius:8px;margin-bottom:8px}
.gauge{position:relative;height:28px;border-radius:7px;background:var(--panel2);overflow:hidden;margin:6px 0 4px}
.gauge .band{position:absolute;top:0;bottom:0;background:#1d4a34}.gauge .pos{position:absolute;top:0;bottom:0;width:3px;background:var(--fg)}
.gauge .cmd{position:absolute;top:0;bottom:0;width:2px;background:var(--acc);opacity:.8}
.gstate{font-size:20px;font-weight:300}.gstate.holding{color:var(--ok)}.gstate.empty{color:var(--red)}
.strip{width:100%;border-radius:8px;display:block;background:#000;min-height:40px;image-rendering:auto}
.decision{display:flex;align-items:center;gap:12px;margin-top:10px}.sw{width:38px;height:38px;border-radius:50%;flex:none;background:var(--panel2)}
.decision b{font-size:20px;font-weight:500;display:block}.decision span{color:var(--mute);font-size:12.5px}
.plates{display:flex;gap:10px}.plate{flex:1;text-align:center;padding:8px 4px;border-radius:10px;background:var(--panel2)}
.plate i{display:block;width:26px;height:26px;border-radius:50%;margin:0 auto 4px}.plate b{font-size:22px;font-weight:300;display:block}
.bricks{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}.bk{padding:2px 8px;border-radius:6px;font-size:12px;background:var(--panel2)}
.bk.t{outline:2px solid var(--ok)}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th{color:var(--mute);font-weight:500;text-align:left;padding:4px 6px;border-bottom:1px solid var(--line)}td{padding:4px 6px;border-bottom:1px solid var(--line)}
.ok{color:var(--ok)}.bad{color:var(--red)}.mute{color:var(--mute)}
.ev{font-size:12px;color:var(--mute);max-height:150px;overflow:auto}.ev div{padding:2px 0}
@media (max-width:1200px){.grid{grid-template-columns:1fr 1fr}.grid>.col:last-child{grid-column:1/-1}}
@media (max-width:760px){.grid{grid-template-columns:1fr}header{gap:8px}}
</style></head><body>
<header>
 <h1>Brick sorting &middot; closed loop</h1><span class="pill" id="mode">off</span><span class="mute" id="cyc"></span>
 <span class="sp"></span>
 <button class="go" onclick="L('start')">Start loop <kbd>S</kbd></button>
 <button onclick="L('pause')">Pause <kbd>P</kbd></button><button onclick="L('resume')">Resume <kbd>R</kbd></button>
 <button onclick="L('home')">Home <kbd>H</kbd></button>
 <button class="stop" onclick="L('stop')">STOP <kbd style="color:#fff">Q</kbd></button>
</header>
<nav class="pipe" id="pipe" aria-label="Loop stages"></nav>
<div class="msg" id="msg">Connecting...</div>
<main class="grid">
 <div class="col">
  <section class="card"><h2>Perceive &middot; top camera <em id="ptop"></em></h2><div class="camwrap"><img class="cam" src="/stream/top" alt="Top camera with detected bricks and plates"><canvas id="ovtop" width="640" height="480"></canvas></div></section>
  <section class="card"><h2>Table &amp; plates <em id="remain"></em></h2>
   <div class="plates" id="plates"></div><div class="bricks" id="bricks"></div></section>
 </div>
 <div class="col">
  <section class="card"><h2>Detect &middot; inspection camera <em id="pinsp"></em></h2><div class="camwrap"><img class="cam" id="inspimg" src="/stream/inspect" alt="Inspection camera with YOLO result"><canvas id="ovinsp" width="640" height="480"></canvas><span class="tag" id="insptag">live</span></div></section>
  <section class="card"><h2>Wrist camera <em class="mute">policy input</em></h2><img class="cam" src="/stream/wrist" alt="Wrist camera"></section>
 </div>
 <div class="col">
  <section class="card"><h2>Act &middot; VLA <em id="vlaph"></em></h2><div class="prompt" id="prompt">&ndash;</div>
   <dl class="kv"><dt>Inference</dt><dd id="lat">&ndash;</dd><dt>Actions queued</dt><dd id="q">&ndash;</dd><dt>Chunking</dt><dd id="bl">&ndash;</dd><dt>Stage time</dt><dd id="st">&ndash;</dd></dl></section>
  <section class="card"><h2>Grasp check <em id="gnum"></em></h2><div class="gstate" id="gstate">&ndash;</div>
   <div class="gauge" id="gauge"><div class="band" id="band"></div><div class="cmd" id="gcmd"></div><div class="pos" id="gpos"></div></div>
   <div class="mute" style="font-size:12px">white = gripper position, blue = command, green band = holding a brick (4&ndash;9). Holding = squeezing an object at brick width.</div></section>
  <section class="card"><h2>Reason &middot; inspection <em id="ifr"></em></h2><img class="strip" id="strip" alt="YOLO crops from the spin">
   <div class="decision"><div class="sw" id="dsw"></div><div><b id="dout">&ndash;</b><span id="dwhy">waiting for a spin</span></div></div></section>
 </div>
 <section class="card" style="grid-column:1/-1"><h2>Cycles <em id="tot"></em></h2>
  <table><thead><tr><th>#</th><th>Brick</th><th>Grasp</th><th>Inspection</th><th>Plate</th><th>Time</th><th>Result</th></tr></thead><tbody id="cycles"></tbody></table>
  <div class="ev" id="ev" style="margin-top:8px"></div></section>
</main>
<script>
const $=id=>document.getElementById(id),COL={yellow:'#f2c230',orange:'#f08a24',purple:'#9b7fe0',blue:'#3d8bff',red:'#e5484d',white:'#f4f4f4',brown:'#8b5a2b'};
const STEPS=[['survey','Survey','bricks left?'],['pick','Pick','VLA'],['bring','Bring','grasp confirmed'],['inspect','Spin + YOLO','NPU'],['place','Place','VLA'],['home','Home','repeat']];
const LBL={none:'plain',circle:'circle',cross:'cross',unknown:'uncertain'};
function L(c){fetch('/loop/'+c,{method:'POST'})}
addEventListener('keydown',e=>{if(e.repeat||e.target.tagName==='INPUT')return;const k=e.key.toLowerCase();
 if(k==='q'){L('stop');e.preventDefault()}else if(k==='s')L('start');else if(k==='p')L('pause');else if(k==='r')L('resume');else if(k==='h')L('home')});
$('pipe').innerHTML=STEPS.map(s=>`<div class="chip" id="c_${s[0]}"><b>${s[1]}</b><span>${s[2]}</span></div>`).join('');
let stripSeq=-1;
async function tick(){let r;try{r=await fetch('/ctl/state').then(x=>x.json());window._last=r}catch(e){$('msg').textContent='Disconnected from the robot';return}
 const lp=r.loop||{},pc=r.perception||{};
 $('mode').textContent=lp.mode||'off';$('mode').className='pill '+(lp.mode||'');
 $('cyc').textContent=lp.cycle?`cycle ${lp.cycle}`+(lp.attempts?` · ${lp.attempts} retry`:''):'';
 $('msg').textContent=lp.message||r.message||'';
 const step=lp.step==='release'?'place':lp.step,order=STEPS.map(s=>s[0]),cur=order.indexOf(step);
 order.forEach((s,i)=>{const c=$('c_'+s);c.className='chip'+(s===step?' on':'')+(cur>=0&&i<cur&&lp.mode==='running'?' done':'')});
 $('prompt').textContent=r.phase==='running'?r.prompt:(r.phase==='inspecting'?'(policy paused: inspecting)':'(policy idle)');
 $('vlaph').textContent=r.phase;$('lat').textContent=r.latency_ms!=null?`${r.latency_ms} ms (median ${r.latency_median_ms})`:'–';
 $('q').textContent=r.queued;$('bl').textContent=r.blend||'–';$('st').textContent=(lp.step_s??'–')+' s';
 const g=lp.grasp||{},cfg=lp.config||{hold:[4,9]},sc=v=>Math.max(0,Math.min(100,v/40*100));
 $('gstate').textContent=g.state||'–';$('gstate').className='gstate '+(g.state||'');$('gnum').textContent=g.pos!=null?`pos ${g.pos} · cmd ${g.cmd}`:'';
 $('band').style.left=sc(cfg.hold[0])+'%';$('band').style.width=(sc(cfg.hold[1])-sc(cfg.hold[0]))+'%';
 if(g.pos!=null){$('gpos').style.left=sc(g.pos)+'%';$('gcmd').style.left=sc(g.cmd)+'%'}
 const tb=pc.table||[],so=pc.sorted||[],tgt=pc.target;
 $('remain').textContent=`${tb.length} on the table · ${so.length} in plates`;$('ptop').textContent=pc.age_s!=null?`${pc.age_s}s ago`:'';
 const tally=lp.tallies||{};$('plates').innerHTML=['yellow','orange','purple'].map(c=>`<div class="plate"><i style="background:${COL[c]}"></i><b>${tally[c]||0}</b><span class="mute">${c} · this run</span></div>`).join('');
 $('bricks').innerHTML=tb.map(b=>`<span class="bk${tgt&&b.box+''==tgt.box+''?' t':''}" style="color:${COL[b.colour]||'#ccc'}">${b.colour} ${b.conf.toFixed(2)}</span>`).join('')||'<span class="mute">no bricks on the table</span>';
 const ins0=pc.inspect;$('pinsp').textContent=r.phase==='inspecting'?'spinning':(ins0&&ins0.held?(ins0.close?'held brick close':'held brick far'):'');
 const fr=lp.inspect_frames||[];$('ifr').textContent=fr.length?`${fr.length} frames`:'';
 if(fr.length!==stripSeq){stripSeq=fr.length;$('strip').src='/loop/strip.jpg?'+Date.now()}
 const ins=r.inspection||{};
 if(ins.state==='running'){$('dout').textContent='Inspecting...';$('dwhy').textContent='spinning the wrist, YOLO on the NPU';$('dsw').style.background='var(--panel2)'}
 else if(ins.outcome){$('dout').textContent=`${LBL[ins.outcome]||ins.outcome} → ${ins.bowl} plate`;$('dsw').style.background=COL[ins.bowl]||'#333';
  $('dwhy').textContent=`${ins.reason||''} · ${ins.frames} frames, ${ins.seconds}s${ins.early_stop?', early stop':''}${ins.attempts>1?', re-spun':''}`}
 const cy=lp.cycles||[];$('tot').textContent=Object.values(tally).reduce((a,b)=>a+b,0)+' sorted';
 $('cycles').innerHTML=cy.slice().reverse().map(c=>`<tr><td>${c.n}</td><td style="color:${COL[c.brick]||'#ccc'}">${c.brick}</td><td>${c.grip!=null?c.grip+' @'+c.t_grasp+'s':'–'}</td><td>${c.outcome?(LBL[c.outcome]||c.outcome)+' ('+c.frames+'f, '+c.inspect_s+'s)':'–'}</td><td>${c.bowl?`<span style="color:${COL[c.bowl]}">${c.bowl}</span>`:'–'}</td><td>${c.seconds!=null?c.seconds+' s':'–'}</td><td class="${c.result==='sorted'?'ok':c.result==='running'?'mute':'bad'}">${c.result}</td></tr>`).join('')||'<tr><td colspan="7" class="mute">no cycles yet</td></tr>';
 $('ev').innerHTML=(lp.events||[]).map(e=>`<div>${new Date(e.t*1000).toLocaleTimeString()} · ${e.kind} ${e.step||e.bowl||e.outcome||''} ${e.message||''}</div>`).join('');
}
function box(c,b,col,w,label){c.strokeStyle=col;c.lineWidth=w;c.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);
 if(label){c.font='600 15px system-ui';const tw=c.measureText(label).width;c.fillStyle='rgba(0,0,0,.6)';c.fillRect(b[0],Math.max(0,b[1]-20),tw+8,19);c.fillStyle=col;c.fillText(label,b[0]+4,Math.max(15,b[1]-5))}}
function drawTop(pc,step){const c=$('ovtop').getContext('2d');c.clearRect(0,0,640,480);const ws=pc.workspace||[0,60,560,480];
 c.fillStyle='rgba(0,0,0,.4)';c.fillRect(0,0,640,ws[1]);c.fillRect(ws[2],ws[1],640-ws[2],480-ws[1]);
 const PC={yellow:'#f2c230',orange:'#f08a24',purple:'#b9a2ff'};
 (pc.plates||[]).forEach(p=>{c.strokeStyle=PC[p.colour]||'#ccc';c.lineWidth=3;c.beginPath();c.arc(p.cx,p.cy,p.r,0,7);c.stroke();
  c.font='600 14px system-ui';c.fillStyle=PC[p.colour]||'#ccc';c.fillText(p.colour+' plate',p.cx-p.r*.6,p.cy+p.r+16)});
 (pc.sorted||[]).forEach(b=>box(c,b.box,'#999',1));
 const t=pc.target;(pc.table||[]).forEach(b=>{const isT=t&&b.box+''==t.box+'';box(c,b.box,isT?'#3eff8e':(COL[b.colour]||'#ccc'),isT?4:2,(isT?'TARGET ':'')+b.colour+' '+b.conf.toFixed(2))});
 if(!['survey','idle','done','paused'].includes(step)){c.font='13px system-ui';c.fillStyle='rgba(255,255,255,.8)';c.fillText('last survey result (detector paused while the arm moves)',8,ws[1]+18)}}
let inspMode='live';
function drawInsp(r){const pc=r.perception||{},c=$('ovinsp').getContext('2d');c.clearRect(0,0,640,480);
 const spin=r.phase==='inspecting';const want=spin?'spin':'live';
 if(want!==inspMode){inspMode=want;$('inspimg').src=spin?'/stream/perc_inspect':'/stream/inspect'}
 $('insptag').textContent=spin?'spin: YOLO on every frame (NPU)':'live';if(spin)return;
 const ins=pc.inspect;if(!ins||ins.age_s>1.5)return;
 c.strokeStyle='rgba(255,255,255,.25)';c.lineWidth=1;[ins.band[0]*640,ins.band[1]*640].forEach(x=>{c.beginPath();c.moveTo(x,0);c.lineTo(x,480);c.stroke()});
 (ins.others||[]).forEach(b=>box(c,b,'rgba(170,170,170,.8)',1));
 if(ins.held)box(c,ins.held,ins.close?'#3eff8e':'#ffc93c',4,ins.close?'held brick: close, ready to spin':'held brick: bringing closer')}
const _tick=tick;tick=async function(){await _tick();try{const r=window._last;if(r){drawTop(r.perception||{},(r.loop||{}).step);drawInsp(r)}}catch(e){}};
setInterval(tick,150);tick();
</script></body></html>"""
