// DLSJ research page: charts from story.json, the replaying twin, the data-driven gallery, BibTeX copy.
// Every path is relative so the page works from any sub-path (GitHub Pages: /dlsj/research/).
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (v, d = 0) => Number(v).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
const reduced = matchMedia('(prefers-reduced-motion: reduce)');

async function getJSON(url) { const r = await fetch(url, { cache: 'no-cache' }); if (!r.ok) throw new Error(url + ' ' + r.status); return r.json(); }

// Numbers as measured (copied from story.json, used only if ../story.json is not published next to this page).
const FALLBACK = {
  latency: [['stock PyTorch CPU', 10770], ['OpenVINO GPU FP16', 362], ['OpenVINO GPU INT8', 285], ['INT8 + RTC (deployed)', 313]],
  runtime: [['Latency per chunk', 10770, 313, 'ms'], ['Energy per chunk', 437, 12, 'J'], ['Peak memory', 26.8, 5.9, 'GB'], ['Error vs the demos', 8.34, 2.8, '°']],
  inspect: [['YOLO v2 classifier on a centre crop', 27, 3, 8], ['held-brick box → v2 classifier on the crop', 31, 2, 5], ['held-cube YOLO26n v4 (plain / circle / cross)', 37, 0, 1], ['held-brick YOLO26n v5 (plain / marked), live', 37, 0, 1]],
  policy: [['MolmoAct2-SO100_101, zero-shot', 8.34], ['LoRA fine-tune, 5.5k steps', 3.52], ['LoRA fine-tune, 8k steps (deployed)', 2.8]],
};
const num = (s) => { const m = String(s).replace(/,/g, '').match(/-?\d+(\.\d+)?/); return m ? parseFloat(m[0]) : null; };

function fromStory(story) {
  const d = structuredClone(FALLBACK);
  const sec = (id) => (story.sections || []).find((x) => x.id === id) || {};
  const rt = sec('runtime');
  const lat = (rt.charts || []).find((c) => /latency/i.test(c.title));
  if (lat?.bars?.length) d.latency = lat.bars;
  const bvn = (rt.tables || []).find((t) => /baseline/i.test(t.columns?.[1] || ''));
  if (bvn) {
    const pick = (re, label, unit) => { const r = bvn.rows.find((x) => re.test(x[0])); return r && num(r[1]) != null ? [label, num(r[1]), num(r[2]), unit] : null; };
    const rows = [pick(/^latency/i, 'Latency per chunk', 'ms'), pick(/energy/i, 'Energy per chunk', 'J'), pick(/ram|memory/i, 'Peak memory', 'GB'), pick(/error/i, 'Error vs the demos', '°')];
    if (rows.every(Boolean)) d.runtime = rows;
  }
  const insp = (sec('inspection').tables || []).find((t) => t.columns?.includes('uncertain'));
  if (insp) d.inspect = insp.rows.map((r) => [String(r[0]).replace(/\s*->\s*/g, ' → '), num(r[1]), num(r[2]), num(r[3])]);
  const pol = (sec('training').tables || []).find((t) => /accuracy/i.test(t.title));
  if (pol) d.policy = pol.rows.map((r) => [r[0], num(r[1])]);
  return d;
}

// ---------------------------------------------------------------- charts (inline SVG, 700 units wide = the text column)
function latencyChart(bars) {
  const W = 700, labW = 200, valW = 100, rowH = 40, max = Math.max(...bars.map((b) => b[1]));
  const rows = bars.map(([l, v], i) => {
    const w = Math.max(4, (v / max) * (W - labW - valW));
    const cls = i === 0 ? 'base' : /deployed/i.test(l) ? 'now' : 'dim';
    return `<g transform="translate(0 ${i * rowH})"><text x="0" y="25">${esc(l)}</text>
      <rect class="${cls}" x="${labW}" y="8" width="${w.toFixed(1)}" height="24" rx="3"/>
      <text class="v" x="${labW + w + 10}" y="25">${fmt(v)} ms</text></g>`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${bars.length * rowH}" role="img" aria-label="Latency per action chunk: ${bars.map((b) => `${b[0]} ${b[1]} ms`).join(', ')}">${rows}</svg>`;
}

function pairChart([label, base, now, unit]) {
  const W = 340, labW = 78, valW = 92, max = Math.max(base, now), bw = (v) => Math.max(4, (v / max) * (W - labW - valW));
  const ratio = base / now;
  const dec = (v) => (unit === '°' ? 2 : v % 1 ? 1 : 0);
  const u = unit === '°' ? '°' : ' ' + unit;
  return `<div class="mini-chart"><h4><span>${esc(label)}</span><b>${fmt(ratio, ratio < 10 ? 1 : 0)}x ${unit === '°' ? 'smaller' : 'lower'}</b></h4>
    <svg viewBox="0 0 ${W} 72" role="img" aria-label="${esc(label)}: baseline ${base}${u}, now ${now}${u}">
      <text x="0" y="23">Baseline</text><rect class="base" x="${labW}" y="7" width="${bw(base).toFixed(1)}" height="22" rx="3"/><text class="v" x="${labW + bw(base) + 8}" y="23">${fmt(base, dec(base))}${u}</text>
      <text x="0" y="59">Now</text><rect class="now" x="${labW}" y="43" width="${bw(now).toFixed(1)}" height="22" rx="3"/><text class="v" x="${labW + bw(now) + 8}" y="59">${fmt(now, dec(now))}${u}</text>
    </svg></div>`;
}

function inspectChart(rows) {
  const W = 700, rowH = 64, barH = 22;
  const C = { right: 'var(--npu)', uncertain: '#a79c86', wrong: 'var(--robot)' };
  const body = rows.map(([l, r, u, w], i) => {
    const tot = r + u + w || 1, sc = W / tot;
    let x = 0;
    const seg = (v, c, k) => { if (!v) return ''; const s = `<rect x="${x.toFixed(1)}" y="26" width="${(v * sc - 2).toFixed(1)}" height="${barH}" fill="${c}"${k === 'right' && i === rows.length - 1 ? '' : ''}/>${v * sc > 28 ? `<text class="v" x="${(x + 8).toFixed(1)}" y="42" style="fill:#fff">${v}</text>` : ''}`; x += v * sc; return s; };
    return `<g transform="translate(0 ${i * rowH})"><text x="0" y="17"${i === rows.length - 1 ? ' style="fill:var(--text);font-weight:800"' : ''}>${esc(l)}</text>
      ${seg(r, C.right, 'right')}${seg(u, C.uncertain)}${seg(w, C.wrong)}</g>`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${rows.length * rowH}" role="img" aria-label="Rig evaluation, 38 spins: ${rows.map((r) => `${r[0]}: ${r[1]} right, ${r[2]} uncertain, ${r[3]} wrong`).join('; ')}">${body}</svg>
    <div class="legend"><span><i style="background:${C.right}"></i>right</span><span><i style="background:${C.uncertain}"></i>uncertain</span><span><i style="background:${C.wrong}"></i>wrong</span></div>`;
}

function policyChart(rows) {
  const W = 700, labW = 270, valW = 80, rowH = 40, max = Math.max(...rows.map((r) => r[1]));
  const body = rows.map(([l, v], i) => {
    const w = Math.max(4, (v / max) * (W - labW - valW));
    const cls = i === 0 ? 'base' : /deployed/i.test(l) ? 'now' : 'dim';
    return `<g transform="translate(0 ${i * rowH})"><text x="0" y="25">${esc(l)}</text><rect class="${cls}" x="${labW}" y="8" width="${w.toFixed(1)}" height="24" rx="3"/><text class="v" x="${labW + w + 10}" y="25">${fmt(v, 2)}°</text></g>`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${rows.length * rowH}" role="img" aria-label="Policy error: ${rows.map((r) => `${r[0]} ${r[1]} degrees`).join(', ')}">${body}</svg>`;
}

function renderCharts(d) {
  $('#figLatency').innerHTML = latencyChart(d.latency);
  $('#figRuntime').innerHTML = d.runtime.map(pairChart).join('');
  $('#figInspect').innerHTML = inspectChart(d.inspect);
  $('#figPolicy').innerHTML = policyChart(d.policy);
}

// ---------------------------------------------------------------- teaser + gallery (media.json)
function mediaEl(m, cls = '') {
  if (m.type === 'video') {
    const auto = reduced.matches ? ' controls' : ' autoplay';
    return `<video class="${cls}" src="${esc(m.src)}"${m.poster ? ` poster="${esc(m.poster)}"` : ''} muted loop playsinline preload="metadata"${auto} aria-label="${esc(m.caption)}"></video>`;
  }
  return `<a href="${esc(m.src)}" target="_blank" rel="noopener"><img src="${esc(m.src)}" alt="${esc(m.caption)}" loading="lazy"></a>`;
}

function renderGallery(items) {
  $('#galleryGrid').innerHTML = items.map((m) =>
    `<figure class="g-item${m.layout === 'tall' ? ' tall' : ''}"><div class="g-media">${mediaEl(m)}</div><figcaption>${esc(m.caption)}</figcaption></figure>`).join('');
  // play gallery videos only while on screen
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    const v = e.target; if (reduced.matches) return;
    if (e.isIntersecting) v.play().catch(() => {}); else v.pause();
  }), { threshold: 0.2 });
  document.querySelectorAll('#galleryGrid video').forEach((v) => { v.removeAttribute('autoplay'); io.observe(v); });
}

function setupTeaser(t) {
  const v = $('#teaserVideo');
  if (t?.src) { v.src = t.src; if (t.poster) v.poster = t.poster; if (t.caption) $('#teaserCap').textContent = t.caption; }
  if (reduced.matches) { v.controls = true; return; }
  v.play().catch(() => { v.controls = true; });
}

// ---------------------------------------------------------------- the twin replaying the recorded run (Method c)
async function setupTwin(replay) {
  const canvas = $('#methodTwin');
  const frame = canvas.parentElement;
  const fallback = () => { frame.innerHTML = `<img src="../media/twin.jpg" alt="The 3D digital twin of the SO-101" style="width:100%;height:100%;object-fit:cover">`; };
  if (!replay?.length) { fallback(); return; }
  let twin;
  try {
    const { createTwin } = await import('../static/twin.js');
    twin = createTwin(canvas, { onError: fallback });
  } catch (e) { console.warn('twin unavailable:', e); fallback(); return; }
  if (!twin) { fallback(); return; }

  const t0 = replay[0].t, dur = replay.at(-1).t - t0;
  const scrub = $('#twinScrub'), playBtn = $('#twinPlay');
  let t = (scrub.value / 1000) * dur, playing = false, last = 0;
  function sample(time) {
    const tt = t0 + time;
    let lo = 0, hi = replay.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (replay[mid].t <= tt) lo = mid; else hi = mid; }
    const a = replay[lo], b = replay[hi], k = Math.max(0, Math.min(1, (tt - a.t) / ((b.t - a.t) || 1)));
    return { m: a.measured.map((x, i) => x + (b.measured[i] - x) * k), step: a.step };
  }
  const STEP = { pick: 'Picking', bring: 'Bringing to camera', place: 'Placing' };
  function apply() {
    const { m, step } = sample(t);
    twin.setMeasured(m);
    twin.setScene({ holding: m[5] > 2.4 && m[5] < 12 && (step === 'bring' || step === 'place'), spinning: false });
    $('#twinT').textContent = `${fmt(t)} s`;
    $('#twinStep').textContent = STEP[step] || step || '';
    scrub.value = Math.round((t / dur) * 1000);
  }
  function loop(now) {
    if (!playing) return;
    const dt = last ? (now - last) / 1000 : 0; last = now;
    t = (t + dt) % dur; apply();
    requestAnimationFrame(loop);
  }
  function setPlaying(p) {
    playing = p; last = 0;
    playBtn.setAttribute('aria-pressed', String(p));
    playBtn.setAttribute('aria-label', p ? 'Pause the recorded run' : 'Play the recorded run');
    if (p) requestAnimationFrame(loop);
  }
  playBtn.addEventListener('click', () => setPlaying(!playing));
  scrub.addEventListener('input', () => { t = (scrub.value / 1000) * dur; apply(); });
  // pause when scrolled away, so the page has only one moving thing on screen at a time
  new IntersectionObserver(([e]) => { if (!e.isIntersecting && playing) setPlaying(false); }).observe(frame);
  apply();
}

// ---------------------------------------------------------------- nav highlight + BibTeX
function setupNav() {
  const links = [...document.querySelectorAll('.nav-links a')];
  const map = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]));
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    links.forEach((a) => a.classList.remove('on'));
    map.get(e.target.id)?.classList.add('on');
  }), { rootMargin: '-45% 0px -50% 0px' });
  map.forEach((_, id) => { const el = document.getElementById(id); if (el) io.observe(el); });
}

$('#copyBib').addEventListener('click', async () => {
  const text = $('#bibCode').textContent;
  try { await navigator.clipboard.writeText(text); $('#copyMsg').textContent = 'BibTeX copied.'; }
  catch {
    const r = document.createRange(); r.selectNodeContents($('#bibCode'));
    const s = getSelection(); s.removeAllRanges(); s.addRange(r);
    $('#copyMsg').textContent = 'Selected: press Ctrl+C or Cmd+C to copy.';
  }
});

// ---------------------------------------------------------------- boot
setupNav();
const [story, replay, media] = await Promise.all([
  getJSON('../story.json').catch(() => null),
  getJSON('../replay.json').catch(() => null),
  getJSON('./media.json').catch(() => null),
]);
renderCharts(story ? fromStory(story) : FALLBACK);
setupTeaser(media?.teaser);
renderGallery(media?.gallery || [
  { type: 'image', src: '../media/twin.jpg', caption: 'The live 3D digital twin.' },
  { type: 'image', src: '../media/mission.jpg', caption: 'Mission control during a run.' },
]);
setupTwin(Array.isArray(replay) ? replay : replay?.frames);
