// Brick sorter: live mission control + presentation.
import { createTwin } from './twin.js';

const $ = (s, el = document) => el.querySelector(s);
const params = new URLSearchParams(location.search);
const DEMO = params.has('demo');           // ?demo : synthetic data, clearly labelled, for rehearsal
const POLL_MS = 170;                        // ~6 Hz
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const arrows = (s) => String(s ?? '').replace(/\s*->\s*/g, ' → ').replace(/\+-/g, '±').replace(/>=/g, '≥');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');

const BRICK_HEX = { blue: '#3d7be0', brown: '#8a5a3c', red: '#d0453a', white: '#f4f4f4', yellow: '#e9b923', green: '#3aa25a', orange: '#e8862a', purple: '#7646c6' };
const DEV_VAR = { GPU: 'var(--gpu)', NPU: 'var(--npu)', CPU: 'var(--cpu)' };

// ---------------------------------------------------------------- twin
const twinCanvas = $('#twin');
let twin = null;
try {
  twin = createTwin(twinCanvas, {
    onError: (e) => { console.warn('twin:', e); $('#twinFallback').hidden = false; $('#twinHint').hidden = true; },
  });
  if (!twin) { $('#twinFallback').hidden = false; $('#twinHint').hidden = true; }
} catch (e) { console.warn('twin:', e); $('#twinFallback').hidden = false; }

// ---------------------------------------------------------------- stage strip
const STAGES = [
  { key: 'survey', label: 'Survey', dev: 'NPU' },
  { key: 'pick', label: 'Pick', dev: 'GPU' },
  { key: 'grasp', label: 'Grasp ✓', dev: 'CPU' },
  { key: 'bring', label: 'Bring', dev: 'GPU' },
  { key: 'inspect', label: 'Spin + YOLO', dev: 'NPU' },
  { key: 'place', label: 'Place', dev: 'GPU' },
  { key: 'next', label: 'Next', dev: 'CPU' },
];
$('#strip').innerHTML = STAGES.map((s) =>
  `<li style="--c:${DEV_VAR[s.dev]}"><span>${s.label}</span><span class="dev-tag">${s.dev}</span></li>`).join('');
const stripItems = [...$('#strip').children];

function stageIndex(st) {
  const loop = st.loop || {};
  const step = loop.step;
  const running = loop.mode === 'running' || loop.mode === 'paused' || st.replay;
  if (!running && !['inspecting', 'moving', 'running'].includes(st.phase)) return -1;
  switch (step) {
    case 'survey': return 0;
    case 'pick': return (loop.grasp && loop.grasp.state === 'holding') ? 2 : 1;
    case 'bring': return 3;
    case 'inspect': return 4;
    case 'place': case 'release': return 5;
    case 'home': case 'verify': return 6;
    default: return st.phase === 'inspecting' ? 4 : -1;
  }
}

function stepTitle(st) {
  const loop = st.loop || {};
  const brick = st.replay ? null : (loop.target?.colour || st.brick);
  const bowl = st.inspection?.bowl || st.bowl;
  if (st.phase === 'stopped') return 'Stopped';
  if (loop.mode === 'paused') return 'Paused';
  switch (loop.step) {
    case 'survey': return 'Surveying the table';
    case 'pick': return loop.grasp?.state === 'holding' ? 'Grasp confirmed' : (brick ? `Picking the ${brick} brick` : 'Picking a brick');
    case 'bring': return 'Bringing it to the camera';
    case 'inspect': return 'Spinning it for YOLO';
    case 'place': return bowl ? `Placing it on the ${bowl} plate` : 'Placing the brick';
    case 'release': return 'Releasing';
    case 'home': return 'Going home';
    case 'verify': return 'Checking the table is clear';
    case 'done': return 'Table clear. Run complete';
  }
  if (st.phase === 'homing') return 'Going home';
  if (st.phase === 'inspecting') return 'Spinning it for YOLO';
  return 'Idle at home';
}

// ---------------------------------------------------------------- helpers
function verdictOf(o) {
  o = String(o || '').toLowerCase();
  if (['none', 'plain', 'good', 'ok'].includes(o)) return 'plain';
  if (['marked', 'circle', 'cross', 'defective', 'defect'].includes(o)) return 'marked';
  if (['unknown', 'uncertain'].includes(o)) return 'unknown';
  return null;
}
const VERDICT_TEXT = { plain: 'Plain: yellow plate', marked: 'Marked: purple plate', unknown: 'Uncertain: purple plate (safe side)' };
const fmt = (v, d = 0) => (typeof v === 'number' && isFinite(v)) ? v.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d }) : '–';
const hms = (t) => { try { return new Date(t * 1000).toLocaleTimeString('en-GB', { hour12: false }); } catch { return ''; } };
function pulse(el) { if (!el || reduced.matches) return; el.classList.remove('on'); void el.offsetWidth; el.classList.add('on'); }

// ---------------------------------------------------------------- render
const latHist = [];
let lastChunks = null, lastSent = null, lastEventsKey = '', lastCyclesKey = '', lastFramesKey = '';
const JOINT_LABELS = ['Pan', 'Lift', 'Elbow', 'Wrist flex', 'Roll', 'Gripper'];
$('#joints').innerHTML = JOINT_LABELS.map((n, i) => `<div class="joint"><span class="jn">${n}</span><span class="jv" id="j${i}">–</span></div>`).join('');

// ?pose=pan,lift,elbow,wflex,roll,grip freezes the twin at a pose (for checking the joint mapping)
const FIXED_POSE = params.get('pose') ? params.get('pose').split(',').map(Number) : null;
if (FIXED_POSE || DEMO) window.twin = twin;
function render(st) {
  const loop = st.loop || {};
  if (FIXED_POSE) st = { ...st, measured: FIXED_POSE };
  // twin
  if (twin) {
    twin.setMeasured(st.measured);
    twin.setPerception(st.replay ? { table: [], sorted: [] } : st.perception);
    const v = verdictOf(st.inspection?.outcome);
    const bowl = st.inspection?.bowl || (['place', 'release'].includes(loop.step) ? st.bowl : null);
    twin.setScene({
      spinning: st.phase === 'inspecting' || loop.step === 'inspect',
      brickColour: st.replay ? null : (loop.target?.colour || st.brick),
      holding: loop.grasp?.state === 'holding' || (st.replay && st.measured && st.measured[5] > 2.4 && st.measured[5] < 12 && ['bring', 'inspect', 'place'].includes(loop.step)),
      verdict: v,
      targetPlate: ['place', 'release'].includes(loop.step) ? (bowl === 'yellow' || bowl === 'purple' ? bowl : (v === 'plain' ? 'yellow' : v ? 'purple' : null)) : null,
    });
  }
  (st.measured || []).forEach((d, i) => { const el = $('#j' + i); if (el) el.textContent = i === 5 ? fmt(d, 0) : `${fmt(d, 0)}°`; });

  // HUD + strip
  $('#hudStep').textContent = stepTitle(st);
  $('#hudMsg').textContent = loop.message || st.message || '';
  const idx = stageIndex(st);
  stripItems.forEach((li, i) => { li.classList.toggle('done', idx >= 0 && i < idx); li.classList.toggle('on', i === idx); li.toggleAttribute('aria-current', i === idx); });

  // prompt
  if (st.prompt) $('#prompt').textContent = st.prompt;

  // latency
  if (typeof st.latency_ms === 'number') {
    if (st.chunks !== lastChunks || latHist.length === 0) {
      latHist.push(st.latency_ms); if (latHist.length > 60) latHist.shift();
      if (lastChunks !== null) pulse($('#pulseGpu'));
      lastChunks = st.chunks;
    }
    $('#latV').textContent = fmt(st.latency_ms);
    $('#latMed').textContent = `median ${fmt(st.latency_median_ms)} ms per chunk`;
    $('#devGpu').textContent = '~' + fmt(st.latency_median_ms || st.latency_ms);
    drawSpark();
  }
  if (st.sent !== lastSent) { if (lastSent !== null) pulse($('#pulseCpu')); lastSent = st.sent; }
  if (st.phase === 'inspecting' || loop.step === 'survey' || loop.step === 'verify') pulse($('#pulseNpu'));

  // gripper gauge
  const g = loop.grasp || {};
  const pos = typeof g.pos === 'number' ? g.pos : (st.measured || [])[5];
  const SCALE = 60;
  const hold = loop.config?.hold || [2.6, 9.0];
  const gHold = $('#gHold');
  gHold.setAttribute('pathLength', 100); gHold.style.strokeDasharray = `${(hold[1] - hold[0]) / SCALE * 100} 200`; gHold.style.strokeDashoffset = `${-hold[0] / SCALE * 100}`;
  const gVal = $('#gVal');
  gVal.setAttribute('pathLength', 100);
  const frac = Math.max(0.01, Math.min(1, (pos || 0) / SCALE)) * 100;
  gVal.style.strokeDasharray = `${frac} 200`;
  $('#gripV').textContent = fmt(pos, 1);
  const gs = g.state || (pos > 12 ? 'open' : 'closed');
  $('#gauge').dataset.s = gs;
  $('#gripState').textContent = { holding: 'Holding a brick', empty: 'Empty', open: 'Open', lost: 'Brick lost', closed: 'Closed' }[gs] || gs;

  renderVerdict(st);

  // tallies
  const t = loop.tallies || {};
  $('#tPlain').textContent = t.yellow || 0;
  $('#tMarked').textContent = (t.purple || 0) + (t.orange || 0);
  $('#cycleInfo').textContent = loop.cycle ? `Cycle ${loop.cycle}${loop.mode ? `, loop ${loop.mode}` : ''}` : (loop.mode ? `Loop ${loop.mode}` : '');

  renderCycles(loop.cycles || []);
  renderEvents(loop.events || []);

  // device footer
  const parts = [];
  if (st.studio?.name) parts.push(`Policy loaded from the Physical AI Studio registry: ${esc(st.studio.name)}.`);
  if (st.residual?.mode) parts.push(`Grasp residual: ${esc(st.residual.mode)}${st.residual.det_ms_median ? `, ${fmt(st.residual.det_ms_median, 1)} ms` : ''}.`);
  $('#devFoot').innerHTML = parts.join(' ');

  drawOverlays(st);
  $('#footState').textContent = st.phase ? `Phase: ${st.phase}` : '';
}

function renderVerdict(st) {
  const loop = st.loop || {};
  const insp = st.inspection;
  let v = verdictOf(insp?.outcome), why = insp?.reason || '';
  let meta = insp && insp.frames ? `${insp.frames} frames in ${fmt(insp.seconds, 1)} s` : '';
  if (!v && loop.cycles?.length) {
    const c = [...loop.cycles].reverse().find((x) => x.outcome || x.bowl);
    if (c) {
    v = verdictOf(c.outcome) || (c.bowl === 'yellow' ? 'plain' : c.bowl === 'purple' ? 'marked' : null);
    why = c.result || ''; meta = c.frames ? `${c.frames} frames in ${fmt(c.inspect_s, 1)} s` : '';
    if (why === 'sorted') why = `Cycle ${c.n}, ${c.brick || ''} brick`;
    }
  }
  const inspecting = st.phase === 'inspecting' || loop.step === 'inspect';
  const card = $('#verdictCard');
  card.dataset.verdict = v || 'none';
  $('#verdictV').textContent = v ? VERDICT_TEXT[v] : (inspecting ? 'Reading the spin…' : 'No brick inspected yet');
  $('#verdictWhy').textContent = [arrows(why), meta].filter(Boolean).join('. ');

  const fr = loop.inspect_frames || [];
  const key = fr.length + ':' + (fr.at(-1)?.i ?? '');
  if (key !== lastFramesKey) {
    lastFramesKey = key;
    $('#frames').innerHTML = fr.length
      ? fr.map((f) => {
          const lbl = String(f.top || '').toLowerCase();
          const cls = (f.confident ? ' c' : '') + (lbl === 'plain' ? ' p' : lbl ? ' m' : '');
          const h = Math.max(6, Math.min(100, (f.conf || 0) * 100));
          return `<span class="f${cls}" style="height:${h}%" title="frame ${f.i}: ${esc(f.top || 'nothing')} ${fmt((f.conf || 0) * 100)}%"></span>`;
        }).join('')
      : `<span class="frames-empty">Frames appear here while the wrist spins the brick.</span>`;
  }
}

function drawSpark() {
  const svg = $('#spark'); if (latHist.length < 2) return;
  const lo = Math.min(...latHist) * 0.9, hi = Math.max(...latHist) * 1.05 || 1;
  const pts = latHist.map((v, i) => [i / (latHist.length - 1) * 160, 34 - (v - lo) / (hi - lo || 1) * 30]);
  const d = 'M' + pts.map((p) => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' L');
  $('.spark-line', svg).setAttribute('d', d);
  $('.spark-area', svg).setAttribute('d', d + ' L160 36 L0 36 Z');
}

function renderCycles(cycles) {
  const key = JSON.stringify(cycles.map((c) => [c.n, c.outcome, c.seconds]));
  if (key === lastCyclesKey) return; lastCyclesKey = key;
  const tb = $('#cycles');
  if (!cycles.length) { tb.innerHTML = `<tr class="empty"><td colspan="7">No cycles yet. Press Start loop to sort the bricks on the table.</td></tr>`; return; }
  tb.innerHTML = cycles.slice().reverse().map((c) => {
    const v = verdictOf(c.outcome);
    return `<tr><td>${esc(c.n)}</td>
      <td><span class="brick-dot" style="background:${BRICK_HEX[c.brick] || '#ccc'}"></span>${esc(c.brick || '–')}</td>
      <td>${v ? (v === 'plain' ? 'plain' : v === 'marked' ? 'marked' : 'uncertain') : (c.result === 'running' ? '<i>in progress</i>' : esc(c.outcome || c.result || '–'))}</td>
      <td><span class="pill ${esc(c.bowl)}">${esc(c.bowl || '–')}</span></td>
      <td class="num">${fmt(c.frames)}</td><td class="num">${typeof c.inspect_s === 'number' ? fmt(c.inspect_s, 1) + ' s' : '–'}</td>
      <td class="num">${typeof c.seconds === 'number' ? fmt(c.seconds, 0) + ' s' : '–'}</td></tr>`;
  }).join('');
}

function renderEvents(events) {
  const key = events.length + ':' + (events[0]?.t ?? '');
  if (key === lastEventsKey) return; lastEventsKey = key;
  const items = events.slice(0, 40).map((e) => {
    let msg = e.message;
    if (!msg && e.kind === 'verify') msg = `YOLO check: ${e.bricks ?? 0} brick(s) on the table`;
    if (!msg) msg = [e.kind, e.step, e.outcome].filter(Boolean).join(' ');
    return `<li><time>${hms(e.t)}</time><span>${esc(arrows(msg))}</span></li>`;
  });
  $('#events').innerHTML = items.join('') || `<li class="empty">Events from the loop appear here.</li>`;
}

// ---------------------------------------------------------------- cameras
const cams = [...document.querySelectorAll('.cam')];
cams.forEach((f) => {
  f.classList.add('off');
  if (DEMO) $('.cam-off', f).textContent = 'Live camera feeds are hidden in demo mode.';
});
function startCam(fig) {
  const img = $('img', fig);
  $('.cam-off', fig).textContent = `${$('b', fig).textContent} feed unavailable. Retrying.`;
  img.onload = () => fig.classList.remove('off');
  img.onerror = () => { fig.classList.add('off'); setTimeout(() => startCam(fig), 5000); };
  img.src = img.dataset.src + '&t=' + Date.now();
}
function stopCam(fig) { const img = $('img', fig); img.onerror = null; img.removeAttribute('src'); fig.classList.add('off'); }

function drawOverlays(st) {
  const p = st.perception || {};
  for (const fig of cams) {
    const im = $('img', fig);
    if (im.getAttribute('src') && im.naturalWidth > 0) fig.classList.remove('off'); // MJPEG may not fire load
    const cv = $('canvas', fig);
    if (cv.width !== 640) { cv.width = 640; cv.height = 480; }
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, 640, 480);
    ctx.lineWidth = 3; ctx.font = '600 15px system-ui, sans-serif';
    const cam = fig.dataset.cam;
    if (cam === 'top') {
      for (const pl of p.plates || []) {
        ctx.strokeStyle = pl.colour === 'yellow' ? '#e9b923' : pl.colour === 'purple' ? '#9b72e6' : '#fff';
        ctx.setLineDash([8, 6]); ctx.beginPath(); ctx.arc(pl.cx, pl.cy, pl.r, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
        label(ctx, pl.cx - 30, pl.cy - pl.r - 4, `${pl.colour} plate`, ctx.strokeStyle);
      }
      for (const b of p.sorted || []) {
        const [x0, y0, x1, y1] = b.box;
        ctx.strokeStyle = 'rgba(255,255,255,.55)'; ctx.lineWidth = 1.5; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      }
      const tgt = p.target?.box || st.loop?.target?.box;
      for (const b of p.table || []) {
        const [x0, y0, x1, y1] = b.box;
        const isT = tgt && Math.abs(tgt[0] - x0) < 4 && Math.abs(tgt[1] - y0) < 4;
        ctx.strokeStyle = isT ? '#ffffff' : '#18c2a2'; ctx.lineWidth = isT ? 4 : 2.5;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        label(ctx, x0, y0 - 4, `${b.colour && b.colour !== '?' ? b.colour : 'brick'} ${fmt((b.conf || 0) * 100)}%`, isT ? '#ffffff' : '#18c2a2');
      }
    } else if (cam === 'inspect') {
      const s = p.spin_last;
      const spinning = st.phase === 'inspecting' && s && s.box && (s.age_s ?? 0) < 2;
      fig.classList.toggle('spinning', !!spinning);
      if (spinning) {
        const [x0, y0, x1, y1] = s.box;
        const lbl = String(s.label || '').toLowerCase();
        const c = lbl === 'plain' ? '#e9b923' : lbl ? '#9b72e6' : '#18c2a2';
        ctx.strokeStyle = c; ctx.lineWidth = s.confident ? 4 : 2; ctx.setLineDash(s.confident ? [] : [6, 5]);
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.setLineDash([]);
        label(ctx, x0, y0 - 4, `${s.label || '?'} ${fmt((s.conf || 0) * 100)}%${typeof s.angle === 'number' ? `, ${fmt(s.angle)}°` : ''}`, c);
      }
    }
  }
}
function label(ctx, x, y, text, color) {
  const w = ctx.measureText(text).width + 12;
  x = Math.max(0, Math.min(640 - w, x)); y = Math.max(22, y);
  ctx.fillStyle = 'rgba(15,27,36,.82)'; ctx.fillRect(x, y - 20, w, 22);
  ctx.fillStyle = color; ctx.fillText(text, x + 6, y - 4);
}

// ---------------------------------------------------------------- controls
async function sendCmd(cmd) {
  const msg = $('#ctlMsg');
  if (mode !== 'live') { msg.textContent = 'Robot offline: controls unavailable.'; return; }
  msg.textContent = cmd === 'stop' ? 'Stopping…' : 'Sending…';
  try {
    const r = await fetch('/api/loop/' + cmd, { method: 'POST' });
    msg.textContent = r.ok ? ({ start: 'Loop started', pause: 'Paused', resume: 'Resumed', stop: 'Stopped', home: 'Going home' }[cmd]) : `Robot refused ${cmd} (HTTP ${r.status})`;
  } catch { msg.textContent = `Could not reach the robot to ${cmd}.`; }
}
$('#btnStop').addEventListener('click', () => sendCmd('stop'));
document.querySelectorAll('.controls .btn').forEach((b) => b.addEventListener('click', () => sendCmd(b.dataset.cmd)));
function setControls(st) {
  const live = mode === 'live';
  const m = st?.loop?.mode;
  for (const b of document.querySelectorAll('.controls .btn')) {
    const c = b.dataset.cmd;
    b.disabled = !live || (c === 'start' && m === 'running') || (c === 'pause' && m !== 'running') || (c === 'resume' && m !== 'paused');
  }
}

// ---------------------------------------------------------------- connection / replay / demo
let mode = 'connecting';
let replay = null, replayT0 = 0, lastLive = null;
function setMode(m, detail = '') {
  if (m !== mode) {
    mode = m;
    if (m === 'live') cams.forEach((f) => { if (!$('img', f).getAttribute('src')) startCam(f); });
  }
  const conn = $('#conn');
  conn.dataset.s = m === 'live' ? 'live' : m === 'demo' ? 'demo' : m === 'connecting' ? '' : 'offline';
  $('#connText').textContent = { live: 'Live', demo: 'Demo data', connecting: 'Connecting', replay: 'Offline: replay', offline: 'Offline' }[m];
  const ban = $('#offlineBanner');
  ban.hidden = m === 'live' || m === 'connecting';
  ban.dataset.kind = m === 'demo' ? 'demo' : '';
  $('#offlineTitle').textContent = m === 'demo' ? 'Demo mode: synthetic data for rehearsal, not the robot'
    : m === 'replay' ? 'Robot offline: showing the last recorded run'
    : 'Robot offline: showing the last known state';
  $('#offlineDetail').textContent = detail;
  $('#btnStop').disabled = false;
}

async function getJSON(url, ms = 2500) {
  const ctl = new AbortController(); const to = setTimeout(() => ctl.abort(), ms);
  try { const r = await fetch(url, { signal: ctl.signal, cache: 'no-store' }); if (!r.ok) throw new Error('HTTP ' + r.status); return await r.json(); }
  finally { clearTimeout(to); }
}

let replayTried = false;
async function loadReplay() {
  if (replayTried) return replay; replayTried = true;
  try { const r = await getJSON('/api/replay', 6000); const arr = Array.isArray(r) ? r : r.frames || r.samples; if (arr?.length > 1) { replay = arr; replayT0 = performance.now(); } } catch { /* optional */ }
  return replay;
}
function replayState() {
  const dur = replay.at(-1).t - replay[0].t;
  const t = replay[0].t + ((performance.now() - replayT0) / 1000) % dur;
  let lo = 0, hi = replay.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (replay[mid].t <= t) lo = mid; else hi = mid; }
  const a = replay[lo], b = replay[hi], k = (t - a.t) / ((b.t - a.t) || 1);
  const measured = a.measured.map((v, i) => v + (b.measured[i] - v) * Math.max(0, Math.min(1, k)));
  const base = lastLive || {};
  return { ...base, replay: true, phase: a.phase || 'running', measured, loop: { ...(base.loop || {}), mode: 'running', step: a.step, message: `Recorded run, ${fmt(t - replay[0].t, 0)} of ${fmt(dur, 0)} s`, grasp: { state: measured[5] > 2.4 && measured[5] < 12 ? 'holding' : measured[5] >= 12 ? 'open' : 'empty', pos: measured[5] } } };
}

async function tick() {
  if (DEMO) { setMode('demo', 'Remove ?demo from the address to connect to the robot.'); const s = demoState(); render(s); setControls(s); return setTimeout(tick, POLL_MS); }
  let st = null, err = '';
  try { st = await getJSON('/api/state'); if (st.offline) { err = st.error || 'controller unreachable'; st = null; } }
  catch (e) { err = e.name === 'AbortError' ? 'the showcase server did not answer' : String(e.message || e); }
  if (st) {
    lastLive = st; setMode('live'); render(st); setControls(st);
  } else {
    await loadReplay();
    if (replay) { setMode('replay', `Reason: ${err}. Retrying the robot in the background.`); render(replayState()); }
    else { setMode('offline', `Reason: ${err}. Retrying.`); if (lastLive) render(lastLive); else render({ measured: [0, -100, 95, 5, 0, 1], phase: 'idle', loop: {} }); }
    setControls(null);
  }
  setTimeout(tick, st ? POLL_MS : (replay ? 100 : 1500));
}

// synthetic rehearsal data (?demo only)
let demoBase = null;
fetch('/api/state').then((r) => r.json()).then((j) => { if (!j.offline) demoBase = j; }).catch(() => {});
function demoState() {
  const T = 34, t = (performance.now() / 1000) % T, cyc = Math.floor(performance.now() / 1000 / T) + 1;
  const HOME = [15, -102, 97, 8, 4, 1], PICK = [2, -18, 42, 62, 0, 36], GRAB = [2, -14, 44, 64, 0, 5.5], INSP = [-24, 20, -24, 72, -17, 5.5], PLACE_Y = [38, -30, 52, 62, 0, 5.5], PLACE_P = [62, -24, 40, 66, 0, 5.5];
  const marked = cyc % 2 === 0;
  const place = marked ? PLACE_P : PLACE_Y;
  const segs = [[0, 3, HOME, HOME, 'survey'], [3, 8, HOME, PICK, 'pick'], [8, 10, PICK, GRAB, 'pick'], [10, 15, GRAB, INSP, 'bring'], [15, 23, INSP, INSP, 'inspect'], [23, 28, INSP, place, 'place'], [28, 29.5, place, [...place.slice(0, 5), 36], 'release'], [29.5, 34, [...place.slice(0, 5), 36], HOME, 'home']];
  const sg = segs.find((s) => t >= s[0] && t < s[1]) || segs[0];
  const k = (t - sg[0]) / (sg[1] - sg[0]); const e = k * k * (3 - 2 * k);
  const m = sg[2].map((v, i) => v + (sg[3][i] - v) * e);
  const inspecting = sg[4] === 'inspect';
  if (inspecting) m[4] = -17 + 180 * Math.min(1, k * 1.15);
  const nFr = inspecting ? Math.floor(k * 61) : (['place', 'release', 'home'].includes(sg[4]) ? 61 : 0);
  const frames = Array.from({ length: nFr }, (_, i) => { const c = 0.45 + 0.5 * Math.abs(Math.sin(i * 0.37 + cyc)); return { i, angle: i * 3, top: (marked && (i % 5 > 1)) ? 'marked' : 'plain', conf: c, confident: c > 0.6, ms: 4 }; });
  const decided = ['place', 'release', 'home'].includes(sg[4]);
  const holding = (sg[4] === 'pick' && t > 9.4) || ['bring', 'inspect', 'place'].includes(sg[4]);
  const cycles = Array.from({ length: cyc - 1 }, (_, i) => ({ n: i + 1, brick: ['blue', 'red', 'white', 'brown', 'yellow'][i % 5], grip: 5.4, outcome: (i + 1) % 2 === 0 ? 'marked' : 'none', bowl: (i + 1) % 2 === 0 ? 'purple' : 'yellow', frames: 61, inspect_s: 8.2, seconds: 34 }));
  const base = demoBase || {};
  return {
    ...base, phase: inspecting ? 'inspecting' : 'running', measured: m,
    prompt: { pick: 'pick up the blue brick', bring: 'bring the brick to the inspection camera', place: `place the brick in the ${marked ? 'purple' : 'yellow'} bowl` }[sg[4]] || base.prompt || 'pick up the blue brick',
    latency_ms: 300 + 25 * Math.sin(performance.now() / 900), latency_median_ms: 304, chunks: Math.floor(performance.now() / 330), sent: Math.floor(performance.now() / 33),
    bowl: marked ? 'purple' : 'yellow',
    inspection: decided ? { state: 'done', outcome: marked ? 'marked' : 'none', bowl: marked ? 'purple' : 'yellow', reason: marked ? '38 of 61 frames confidently marked' : '57 of 61 frames plain (94 %)', frames: 61, seconds: 8.2 } : null,
    perception: { ...(base.perception || {}), table: sg[4] === 'survey' || sg[4] === 'pick' ? [{ box: [300, 210, 336, 246], colour: 'blue', conf: 0.91 }, { box: [240, 250, 274, 284], colour: "red", conf: 0.88 }] : [], target: { box: [300, 210, 336, 246] },
      spin_last: inspecting ? { box: [250, 170, 390, 300], label: frames.at(-1)?.top || 'plain', conf: frames.at(-1)?.conf || 0.8, angle: m[4], confident: true, age_s: 0.1 } : null },
    loop: { ...(base.loop || {}), mode: 'running', step: sg[4], cycle: cyc, message: `Demo cycle ${cyc}`, tallies: { yellow: Math.ceil((cyc - 1) / 2), purple: Math.floor((cyc - 1) / 2) }, grasp: { state: holding ? 'holding' : 'empty', pos: m[5] }, inspect_frames: frames, cycles, target: { colour: 'blue', box: [300, 210, 336, 246] }, events: base.loop?.events || [] },
  };
}

// ---------------------------------------------------------------- story
const ACCENT = { runtime: 'gpu', inspection: 'npu' };
function parseNum(s) { const m = String(s).replace(/,/g, '').match(/-?\d+(\.\d+)?/); return m ? parseFloat(m[0]) : null; }
function unitOf(s) { const m = String(s).replace(/,/g, '').match(/-?\d+(?:\.\d+)?\s*([a-zA-Z%°]+)/); return m ? m[1] : ''; }

function barsChart(ch, accent) {
  const bars = ch.bars || []; const max = Math.max(...bars.map((b) => b[1]), 1);
  const rowH = 34, labW = 196, w = 560, valW = 92;
  const svgRows = bars.map(([l, v], i) => {
    const bw = Math.max(3, (v / max) * (w - labW - valW));
    const isBase = i === 0, isNow = /deployed/i.test(l) || i === bars.length - 1;
    const cls = isBase ? 'b-base' : 'b-now';
    const op = isBase || isNow ? 1 : 0.45;
    return `<g transform="translate(0,${i * rowH})"><text x="0" y="21">${esc(l)}</text>
      <rect class="${cls}" x="${labW}" y="6" width="${bw.toFixed(1)}" height="20" rx="4" opacity="${op}"/>
      <text class="v" x="${labW + bw + 8}" y="21">${fmt(v)} ${esc(ch.unit || '')}</text></g>`;
  }).join('');
  const gain = bars.length > 1 ? bars[0][1] / bars.at(-1)[1] : null;
  return `<figure class="chart" style="--c:var(--${accent})"><h4><span>${esc(ch.title)}</span>${gain && gain > 1.5 ? `<span class="gain">${fmt(gain, 0)}x lower</span>` : ''}</h4>
    <svg viewBox="0 0 ${w} ${bars.length * rowH}" role="img" aria-label="${esc(ch.title)}: ${bars.map((b) => `${b[0]} ${b[1]} ${ch.unit || ''}`).join(', ')}">${svgRows}</svg></figure>`;
}

function pairedCharts(table, accent, covered = []) {
  // "baseline vs now" tables: one compact two-bar chart per numeric row (skipping rows an explicit chart already shows)
  const rows = table.rows.filter((r) => parseNum(r[1]) !== null && parseNum(r[2]) !== null && !/^p99/i.test(r[0])
    && !covered.some((t) => t.toLowerCase().startsWith(String(r[0]).split(' ')[0].toLowerCase())));
  if (!rows.length) return '';
  return rows.map((r) => {
    const b = parseNum(r[1]), n = parseNum(r[2]), max = Math.max(b, n), u = unitOf(r[1]);
    const w = 560, labW = 90, valW = 110, bw = (v) => Math.max(3, v / max * (w - labW - valW));
    const ratio = b / n;
    return `<figure class="chart" style="--c:var(--${accent})"><h4><span>${esc(r[0][0].toUpperCase() + r[0].slice(1))}</span><span class="gain">${fmt(ratio, ratio < 10 ? 1 : 0)}x ${/deg|°/.test(u) ? 'smaller' : 'lower'}</span></h4>
      <svg viewBox="0 0 ${w} 64" role="img" aria-label="${esc(r[0])}: baseline ${esc(r[1])}, now ${esc(r[2])}">
        <text x="0" y="21">Baseline</text><rect class="b-base" x="${labW}" y="6" width="${bw(b).toFixed(1)}" height="20" rx="4"/><text class="v" x="${labW + bw(b) + 8}" y="21">${esc(r[1])}</text>
        <text x="0" y="53">Now</text><rect class="b-now" x="${labW}" y="38" width="${bw(n).toFixed(1)}" height="20" rx="4"/><text class="v" x="${labW + bw(n) + 8}" y="53">${esc(r[2])}</text>
      </svg></figure>`;
  }).join('');
}

function tableHTML(t) {
  return `<div class="tbl"><h3>${esc(t.title)}</h3><div class="table-scroll"><table>
    <thead><tr>${t.columns.map((c, i) => `<th${i && t.rows.every((r) => parseNum(r[i]) !== null && String(r[i]).length < 14) ? ' class="num"' : ''}>${esc(c)}</th>`).join('')}</tr></thead>
    <tbody>${t.rows.map((r) => `<tr>${r.map((c, i) => `<td${i && t.rows.every((rr) => parseNum(rr[i]) !== null && String(rr[i]).length < 14) ? ' class="num"' : ''}>${esc(arrows(c))}</td>`).join('')}</tr>`).join('')}</tbody>
  </table></div></div>`;
}

function renderStory(s) {
  if (s.title) {
    const [a, ...b] = s.title.split(':');
    $('#title').textContent = b.length ? a.trim() : s.title;
    if (b.length) $('#tagline').textContent = b.join(':').trim();
  }
  if (s.tagline) $('#how .lead').textContent = arrows(s.tagline);
  if (s.pipeline?.length) {
    $('#pipeline').innerHTML = s.pipeline.map((p) => {
      const d = String(p.device || '').toUpperCase();
      return `<li style="--c:${DEV_VAR[d] || 'var(--muted)'}"><div class="p-step">${esc(p.step)}</div><span class="chip ${d.toLowerCase()} p-dev">${esc(d)}</span><p class="p-what">${esc(arrows(p.what))}</p></li>`;
    }).join('');
  }
  $('#navSections').innerHTML = (s.sections || []).map((x) => `<a href="#s-${esc(x.id)}">${esc(x.title.split(' ')[0] === 'Defect' ? 'Detection' : x.title.replace(/ optimisation| collection/i, ''))}</a>`).join('');
  $('#story').innerHTML = (s.sections || []).map((x) => {
    const acc = ACCENT[x.id] || 'ink';
    const stats = (x.stats || []).map((st) => {
      const v = arrows(st.value);
      return `<div class="stat"><div class="stat-v" style="${v.length > 7 ? 'font-size:clamp(34px,3vw,52px)' : ''}">${esc(v)}</div><div class="stat-l">${esc(arrows(st.label))}</div>${st.note ? `<div class="stat-n">${esc(arrows(st.note))}</div>` : ''}</div>`;
    }).join('');
    let charts = (x.charts || []).map((c) => barsChart(c, acc === 'ink' ? 'gpu' : acc)).join('');
    const tables = (x.tables || []).map((t) => {
      if (t.columns?.length === 3 && /baseline/i.test(t.columns[1]) && /now/i.test(t.columns[2])) charts += pairedCharts(t, acc === 'ink' ? 'gpu' : acc, (x.charts || []).map((c) => c.title || ''));
      return tableHTML(t);
    }).join('');
    const bullets = x.bullets?.length ? `<ul class="bullets">${x.bullets.map((b) => `<li>${esc(arrows(b))}</li>`).join('')}</ul>` : '';
    const cav = x.caveats?.length ? `<div class="caveats">${x.caveats.map((c, i) => `<p>${i === 0 ? '<b>Caveats.</b> ' : ''}${esc(arrows(c))}</p>`).join('')}</div>` : '';
    const left = charts ? `<div class="charts">${charts}</div>` : tables;
    const right = (charts ? tables : '') + bullets + cav;
    return `<section class="story-sec" id="s-${esc(x.id)}" data-accent="${acc}" aria-labelledby="h-${esc(x.id)}">
      <div class="sec-head"><h2 id="h-${esc(x.id)}">${esc(x.title)}</h2>${x.lead ? `<p class="lead">${esc(arrows(x.lead))}</p>` : ''}</div>
      ${stats ? `<div class="stats">${stats}</div>` : ''}
      <div class="story-grid"><div>${left}</div><div>${right}</div></div></section>`;
  }).join('');
}

const FALLBACK_PIPELINE = [
  { step: 'Perceive', device: 'NPU', what: 'The top camera finds the bricks left on the table and the plates.' },
  { step: 'Pick', device: 'GPU', what: 'MolmoAct2 picks the brick; the gripper reading confirms the grasp.' },
  { step: 'Bring', device: 'GPU', what: 'The same VLA brings the brick to the inspection camera.' },
  { step: 'Detect', device: 'NPU', what: 'The wrist spins the brick while YOLO reads every frame.' },
  { step: 'Reason', device: 'CPU', what: 'An evidence rule over all frames decides plain or marked.' },
  { step: 'Place', device: 'GPU', what: 'Plain goes to the yellow plate, marked or uncertain to the purple plate.' },
];
async function loadStory() {
  try { renderStory(await getJSON('/api/story', 5000)); }
  catch {
    if (!$('#pipeline').children.length) renderStory({ pipeline: FALLBACK_PIPELINE, sections: [] });
    $('#story').innerHTML = `<p class="story-wait">The results sections load from the showcase server. Retrying.</p>`;
    setTimeout(loadStory, 8000);
  }
}

loadStory();
tick();
