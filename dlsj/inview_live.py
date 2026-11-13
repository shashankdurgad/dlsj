"""Watch the brick-in-view classifier live on the inspection camera: http://localhost:8003.

    uv run python -m dlsj.inview_live --camera 0                 # add --device NPU on the XPS

Every new camera frame goes through the classifier, using the same detector the recorder and controller
use. The page shows:
* the annotated stream: the centre crop the model sees, a P(in view) bar and the threshold;
* a large in-view indicator, the probability, inference latency and the classifier rate;
* a chime each time the state flips to in view.
"""

import argparse
import json
import logging
import threading
import time
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

from dlsj import teleop_web
from dlsj.inview_classifier import DEFAULT_MODEL, InViewClassifier

log = logging.getLogger("inview_live")


class AnnotatedStream:
    """Runs the classifier on each new frame of `src` and exposes the annotated JPEG like a CameraStream."""

    def __init__(self, src, clf: InViewClassifier, quality: int = 75):
        self.src, self.clf, self.quality = src, clf, quality
        self.name = "annotated"
        self.jpeg: bytes | None = None
        self.seq = 0
        self.cond = threading.Condition()
        self.latency_ms = 0.0
        self.hz = 0.0
        self._stop = threading.Event()

    def start(self) -> "AnnotatedStream":
        threading.Thread(target=self._run, name="classify", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        last_seq, n, t_rate = -1, 0, time.perf_counter()
        lat = []
        while not self._stop.is_set():
            frame, seq = self.src.frame, self.src.seq
            if frame is None or seq == last_seq:
                time.sleep(0.003)
                continue
            last_seq = seq
            t0 = time.perf_counter()
            self.clf.update(frame)
            lat.append((time.perf_counter() - t0) * 1e3)
            ok, buf = cv2.imencode(".jpg", self.clf.debug_image(frame), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if ok:
                with self.cond:
                    self.jpeg = buf.tobytes()
                    self.seq += 1
                    self.cond.notify_all()
            n += 1
            now = time.perf_counter()
            if now - t_rate >= 1.0:
                self.hz, self.latency_ms = n / (now - t_rate), float(np.median(lat))
                n, t_rate, lat = 0, now, []

    def info(self) -> dict:
        return {"name": self.name, "index": -1, "fps": round(self.hz, 1), "size": None, "error": self.src.error}

    def state(self) -> dict:
        d = self.clf.last
        return {"in_view": d.in_view, "candidate": d.candidate, "prob": round(self.clf.prob, 3),
                "stable": d.stable, "need": self.clf.stable_frames, "threshold": self.clf.threshold,
                "latency_ms": round(self.latency_ms, 2), "hz": round(self.hz, 1), "device": self.clf.device,
                "camera_error": self.src.error}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>In view</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@200;300;400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#15171a;--panel:#1d2024;--line:#2b2f35;--fg:#f1f2f4;--mute:#80868f;--dim:#4a4f57;--ok:#3ecf8e;--red:#e31937}
*{box-sizing:border-box;margin:0}html,body{height:100%}
body{background:var(--bg);color:var(--fg);font:400 15px/1.35 Barlow,ui-sans-serif,system-ui,sans-serif;-webkit-font-smoothing:antialiased;overflow:hidden}
.app{display:grid;grid-template-columns:minmax(0,1fr) 340px;height:100vh}
.view{padding:20px;min-width:0;display:flex}
.cam{flex:1;border-radius:18px;overflow:hidden;background:#000;outline:3px solid transparent;outline-offset:-3px;transition:outline-color .15s}
.cam img{width:100%;height:100%;object-fit:contain;display:block}
.on .cam{outline-color:var(--ok)}
.side{padding:28px 24px;display:flex;flex-direction:column;gap:26px}
.big{display:flex;flex-direction:column;align-items:center;gap:10px;padding:26px 0}
.lamp{width:150px;height:150px;border-radius:50%;background:var(--panel);border:2px solid var(--line);transition:background .15s,box-shadow .15s,border-color .15s}
.on .lamp{background:var(--ok);border-color:var(--ok);box-shadow:0 0 50px rgba(62,207,142,.45)}
.near .lamp{border-color:var(--fg)}
.word{font-size:28px;font-weight:300}
.p{font-weight:200;font-size:64px;line-height:1;font-variant-numeric:tabular-nums;text-align:center}
.bar{height:6px;background:var(--line);border-radius:4px;position:relative;overflow:hidden}
.bar i{position:absolute;inset:0 auto 0 0;width:0;background:var(--fg);transition:width .08s}
.on .bar i{background:var(--ok)}
.bar b{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--mute)}
.stats{margin-top:auto;display:grid;grid-template-columns:1fr 1fr;gap:14px;color:var(--mute);font-size:13px}
.stats b{display:block;color:var(--fg);font-weight:300;font-size:26px;font-variant-numeric:tabular-nums}
@media (max-width:860px){body{overflow:auto}.app{grid-template-columns:1fr;height:auto}.cam{aspect-ratio:4/3}.lamp{width:110px;height:110px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body><div class="app" id="app">
<section class="view"><div class="cam"><img src="/stream/annotated" alt="Inspection camera with classifier overlay"></div></section>
<aside class="side">
 <div class="big"><div class="lamp"></div><div class="word" id="word">Waiting</div></div>
 <div><div class="p" id="p">0.00</div><div class="bar"><i id="bar"></i><b id="thr"></b></div></div>
 <div class="stats"><span><b id="lat">–</b>ms per frame</span><span><b id="hz">–</b>frames per second</span><span><b id="stable">–</b>steady frames</span><span><b id="dev">–</b>device</span></div>
</aside></div>
<script>
const $=id=>document.getElementById(id),app=$('app');let was=false,ac=null;
function chime(){try{ac=ac||new(window.AudioContext||window.webkitAudioContext)();const o=ac.createOscillator(),g=ac.createGain(),t=ac.currentTime;o.frequency.value=1568;o.connect(g);g.connect(ac.destination);g.gain.setValueAtTime(1e-4,t);g.gain.exponentialRampToValueAtTime(.25,t+.01);g.gain.exponentialRampToValueAtTime(1e-4,t+.18);o.start(t);o.stop(t+.2)}catch(e){}}
async function tick(){try{const r=await fetch('/live/state').then(x=>x.json());
 app.classList.toggle('on',r.in_view);app.classList.toggle('near',r.candidate&&!r.in_view);
 if(r.in_view&&!was)chime();was=r.in_view;
 $('word').textContent=r.camera_error?'Camera error':r.in_view?'In view':r.candidate?'Settling':'Not in view';
 $('p').textContent=r.prob.toFixed(2);$('bar').style.width=(r.prob*100)+'%';$('thr').style.left=(r.threshold*100)+'%';
 $('lat').textContent=r.latency_ms.toFixed(2);$('hz').textContent=r.hz.toFixed(0);$('stable').textContent=r.stable+'/'+r.need;$('dev').textContent=r.device;
}catch(e){$('word').textContent='Disconnected'}}
setInterval(tick,80);tick();
</script></body></html>"""


def make_handler(cam, ann: AnnotatedStream):
    base = teleop_web.make_handler({"inspect": cam, "annotated": ann}, None)

    class Handler(base):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/live/state":
                return self._send(200, json.dumps(ann.state()).encode(), "application/json")
            return super().do_GET()

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, required=True)
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--device", default="CPU")
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--stable-frames", type=int, default=3)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8003)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cam = teleop_web.CameraStream("inspect", args.camera, args.width, args.height, 30, 80).start()
    ann = AnnotatedStream(cam, InViewClassifier(args.model, args.device, args.threshold, args.stable_frames)).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(cam, ann))
    server.daemon_threads = True
    log.info("live view: http://localhost:%d  (model %s on %s)", args.port, args.model, args.device)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ann.stop()
        cam.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
