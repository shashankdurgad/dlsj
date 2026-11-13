"""Browser status page for the loop (same pattern as `dlsj.teleop_web`): http://localhost:8001

Shows every camera feed, the state machine state and current skill, the last InspectionResult drawn as a
heatmap overlay with the region box, the last decision, and live per-stage latency (median / p90 / p99).
Buttons: STOP (abort the loop) and NEXT BRICK (confirm when the loop waits for a brick).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from dlsj.contracts import InspectionResult

log = logging.getLogger(__name__)


def overlay(frame_rgb: np.ndarray, result: InspectionResult | None, alpha: float = 0.45) -> np.ndarray:
    """Heatmap (red = anomalous) blended over the inspect frame, plus the region box. Pure NumPy."""
    img = frame_rgb.astype(np.float32).copy()
    if result is None:
        return frame_rgb
    if result.heatmap is not None and result.heatmap.shape[:2] == img.shape[:2]:
        h = np.clip(result.heatmap.astype(np.float32), 0.0, 1.0)[..., None]
        heat_rgb = np.concatenate([255 * h, 255 * (1 - np.abs(2 * h - 1)), 255 * (1 - h)], axis=2)  # blue->green->red
        img = (1 - alpha * h) * img + alpha * h * heat_rgb
    if result.region is not None:
        x0, y0, x1, y1 = (int(v) for v in result.region)
        hh, ww = img.shape[:2]
        x0, x1 = max(0, min(x0, ww - 1)), max(0, min(x1, ww - 1))
        y0, y1 = max(0, min(y0, hh - 1)), max(0, min(y1, hh - 1))
        col = np.array([255, 40, 200], dtype=np.float32)
        for t in range(3):
            img[max(0, y0 - t), x0:x1 + 1] = col
            img[min(hh - 1, y1 + t), x0:x1 + 1] = col
            img[y0:y1 + 1, max(0, x0 - t)] = col
            img[y0:y1 + 1, min(ww - 1, x1 + t)] = col
    return np.clip(img, 0, 255).astype(np.uint8)


def _jpeg(rgb: np.ndarray) -> bytes | None:
    try:
        import cv2
    except ImportError:
        return None
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>DLSJ Loop</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e6e8ee;--mute:#8a90a0;--ok:#3ecf8e;--bad:#ff6b6b;--acc:#6aa9ff;--pink:#ff7ac8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 ui-sans-serif,system-ui,-apple-system}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid #262a33;position:sticky;top:0;background:var(--bg);z-index:2}
h1{font-size:15px;margin:0}.pill{padding:2px 8px;border-radius:99px;background:#262a33;color:var(--mute);font-variant-numeric:tabular-nums}
.pill.ok{color:var(--ok)}.pill.bad{color:var(--bad)}button{background:#262a33;color:var(--fg);border:1px solid #333845;border-radius:6px;padding:4px 10px;cursor:pointer}
button.stop{background:#5a1f24;border-color:#8a2c33}
main{display:grid;grid-template-columns:1fr 380px;gap:16px;padding:16px}@media(max-width:1000px){main{grid-template-columns:1fr}}
#cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.cam{background:var(--card);border-radius:10px;overflow:hidden}.cam img{display:block;width:100%;background:#000;aspect-ratio:4/3;object-fit:contain}
.cam .bar{padding:6px 10px;color:var(--mute)}.cam .bar b{color:var(--fg)}
.side{display:flex;flex-direction:column;gap:12px}.card{background:var(--card);border-radius:10px;padding:12px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);margin:0 0 8px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:2px 4px;text-align:right}td:first-child,th:first-child{text-align:left;color:var(--mute)}
th{color:var(--mute);font-weight:500}.big{font-size:20px;font-weight:600}.perfect{color:var(--ok)}.defective{color:var(--pink)}.uncertain{color:#ffc857}
code{color:var(--acc)}
</style></head><body>
<header><h1>DLSJ closed loop</h1><span id="state" class="pill">…</span><span id="skill" class="pill"></span>
<span id="msg" style="color:var(--mute)"></span><span style="flex:1"></span>
<button id="next">next brick</button> <button id="stop" class="stop">STOP</button></header>
<main><section><div id="cams"></div>
<div class="cam" style="margin-top:12px"><img id="insp" alt="last inspection"><div class="bar"><b>last inspection</b> heatmap + region</div></div></section>
<aside class="side">
<div class="card"><h2>Inspection</h2><div id="label" class="big">–</div><div id="insp_detail" style="color:var(--mute)"></div></div>
<div class="card"><h2>Decision</h2><div id="decision">–</div></div>
<div class="card"><h2>Counts</h2><div id="counts">–</div></div>
<div class="card"><h2>Latency (ms)</h2><table><thead><tr><th>stage</th><th>n</th><th>median</th><th>p90</th><th>p99</th></tr></thead><tbody id="lat"></tbody></table></div>
</aside></main>
<script>
const cams=document.getElementById('cams'),seen={};let lastInsp=null;
async function tick(){
  try{
    const s=await (await fetch('/state')).json(),c=s.loop;
    for(const n of s.cameras){if(!seen[n]){const d=document.createElement('div');d.className='cam';
      d.innerHTML=`<img src="/stream/${encodeURIComponent(n)}"><div class="bar"><b>${n}</b></div>`;cams.appendChild(d);seen[n]=d;}}
    const st=document.getElementById('state');st.textContent=c.state;st.className='pill '+(c.state==='error'||c.state==='aborted'?'bad':'ok');
    const sk=c.skill||{};document.getElementById('skill').textContent=sk.skill?`${sk.skill} · tick ${sk.tick||0} · queue ${sk.queue??'–'}`:'';
    document.getElementById('msg').textContent=(c.task?`brick #${c.task.id} · `:'')+(c.message||'');
    const r=c.last_inspection;if(r){const L=document.getElementById('label');L.textContent=r.label.toUpperCase();L.className='big '+r.label;
      document.getElementById('insp_detail').innerHTML=`score <b>${r.score.toFixed(3)}</b> / thr ${r.threshold.toFixed(3)} · ${r.latency_ms.toFixed(1)} ms on ${r.device||'?'}<br>frames ${r.frame_scores.map(x=>x.toFixed(2)).join(', ')}<br>region ${r.region?JSON.stringify(r.region):'none'}`;
      const key=JSON.stringify(r);if(key!==lastInsp){lastInsp=key;document.getElementById('insp').src='/inspection.jpg?'+Date.now();}}
    const d=c.last_decision;if(d)document.getElementById('decision').innerHTML=`<code>${d.skill||'re-grab'}</code>${d.forced?' (forced)':''}<br><span style="color:var(--mute)">${d.reason}</span>`;
    document.getElementById('counts').textContent=Object.entries(c.counts).map(([k,v])=>`${k}: ${v}`).join(' · ')||'–';
    document.getElementById('lat').innerHTML=Object.entries(c.latency).filter(([k,v])=>v.n).map(([k,v])=>`<tr><td>${k}</td><td>${v.n}</td><td>${v.median}</td><td>${v.p90}</td><td>${v.p99}</td></tr>`).join('');
  }catch(e){const st=document.getElementById('state');st.textContent='server gone';st.className='pill bad';}
}
document.getElementById('stop').onclick=()=>fetch('/cmd/stop',{method:'POST'});
document.getElementById('next').onclick=()=>fetch('/cmd/next',{method:'POST'});
setInterval(tick,300);tick();
</script></body></html>"""


class StatusServer:
    def __init__(self, controller, cameras, host: str = "127.0.0.1", port: int = 8001,
                 on_next=None):
        self.controller, self.cameras, self.on_next = controller, cameras, on_next
        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]

    def start(self) -> "StatusServer":
        threading.Thread(target=self.server.serve_forever, name="status-http", daemon=True).start()
        log.info("status page: http://localhost:%d", self.port)
        return self

    def stop(self) -> None:
        self.server.shutdown()

    def inspection_jpeg(self) -> bytes | None:
        c = self.controller
        if c.last_inspect_frame is None:
            return None
        return _jpeg(overlay(c.last_inspect_frame, c.last_result))

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path == "/cmd/stop":
                    outer.controller.abort()
                    if outer.on_next:
                        outer.on_next(False)
                    return self._send(200, b"{}", "application/json")
                if self.path == "/cmd/next" and outer.on_next:
                    outer.on_next(True)
                    return self._send(200, b"{}", "application/json")
                self._send(404, b"", "text/plain")

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if path == "/state":
                    body = {"cameras": list(outer.cameras.names), "loop": outer.controller.snapshot()}
                    return self._send(200, json.dumps(body, default=str).encode(), "application/json")
                if path == "/inspection.jpg":
                    j = outer.inspection_jpeg()
                    return self._send(200, j, "image/jpeg") if j else self._send(404, b"", "text/plain")
                if path.startswith("/stream/"):
                    from urllib.parse import unquote

                    name = unquote(path.removeprefix("/stream/"))
                    if name not in outer.cameras.names:
                        return self._send(404, b"", "text/plain")
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        while True:
                            jpeg = outer.cameras.jpeg(name)
                            if jpeg:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                                 + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                            time.sleep(0.1)  # 10 fps preview is plenty and keeps the CPU for control
                    except (BrokenPipeError, ConnectionResetError):
                        return
                self._send(404, b"", "text/plain")

        return Handler
