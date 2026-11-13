"""Record teleop demos into a LeRobot v3 dataset from a browser: http://localhost:8000.

Built on `dlsj.teleop_web`, which runs the leader-to-follower control loop, the home/engage logic and
the camera threads. This module adds a recorder that samples that loop at the dataset rate, plus a
recording page.

    uv run python -m dlsj.teleop_record --cameras wrist=0,top=2,inspect=4 \
        --repo-id local/duplo_teleop --task "Pick a brick and present it to the inspection camera" \
        --operator alice --target-s 6

Keys in the browser (they work on Wayland too, unlike global key capture):
  SPACE        start a take (rising beep) / stop it (falling beep) / accept it
  f            save the stopped take as a FAILURE
  h / Bksp     discard the take
  e            engage / hold the follower

Each frame stores:
* `observation.state`: the follower's measured joints.
* `action`: the command sent to the follower, i.e. the leader pose, as in `lerobot-record`.
* `observation.images.<cam>`: one image per camera.

Frames are only recorded while the follower is live, i.e. following the leader. Labels (success, duration,
operator, method) go to `meta/dlsj/labels.jsonl`, the same format as the kinesthetic recorder, so the two
datasets can be mixed.
"""

import argparse
import json
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from dlsj import arms, teleop_web
from dlsj.kinesthetic.session import EpisodeStats

log = logging.getLogger("teleop_record")


class EpisodeRecorder:
    """Samples `teleop.latest` and the camera streams at `fps` into a LeRobot dataset.

    Phases: idle -> recording -> review -> (saving) -> idle. A take is only written by accept/fail.
    """

    def __init__(self, teleop, streams: dict, repo_id: str, task: str, fps: int = 30, root: str | Path | None = None,
                 episode_time_s: float = 60.0, operator: str = "", vcodec: str = "h264", resume: bool = False):
        self.teleop, self.streams = teleop, streams
        self.repo_id, self.task, self.fps = repo_id, task, fps
        self.root = Path(root) if root else Path("data") / repo_id.split("/")[-1]
        self.episode_time_s, self.operator, self.vcodec, self.resume = episode_time_s, operator, vcodec, resume
        self.joints = list(teleop.joints)
        self.phase = "idle"
        self.message = "engage the follower, then press SPACE to record"
        self.dataset = None
        self.cam_shapes: dict[str, tuple[int, int, int]] = {}
        self.stats = EpisodeStats()
        self.commands: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._take_start = 0.0
        self._take_duration = 0.0
        self._frames = 0
        self._paused_ticks = 0
        self.saved = {"success": 0, "failure": 0}

    # ------------------------------------------------------------------ dataset
    def _wait_for_cameras(self, timeout_s: float = 20.0) -> None:
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            frames = {n: s.frame for n, s in self.streams.items()}
            if all(f is not None for f in frames.values()):
                self.cam_shapes = {n: f.shape for n, f in frames.items()}
                return
            time.sleep(0.1)
        missing = [n for n, s in self.streams.items() if s.frame is None]
        raise RuntimeError(f"no frames from cameras {missing} after {timeout_s}s")

    def features(self) -> dict:
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import hw_to_dataset_features

        joints = {j: float for j in self.joints}
        return {
            **hw_to_dataset_features(joints, ACTION, use_video=True),
            **hw_to_dataset_features({**joints, **self.cam_shapes}, OBS_STR, use_video=True),
        }

    def open_dataset(self) -> None:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets import LeRobotDataset

        self._wait_for_cameras()
        n = max(1, len(self.streams))
        kw = dict(root=self.root, image_writer_threads=4 * n, rgb_encoder=RGBEncoderConfig(vcodec=self.vcodec))
        if self.resume and (self.root / "meta" / "info.json").exists():
            self.dataset = LeRobotDataset.resume(self.repo_id, **kw)
            want = set(self.features())
            have = set(self.dataset.features) - {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            if want != have:
                raise ValueError(f"dataset features differ: {sorted(have ^ want)}")
        else:
            robot_type = getattr(getattr(self.teleop, "robot", None), "robot_type", "so_follower")
            self.dataset = LeRobotDataset.create(self.repo_id, self.fps, robot_type=robot_type,
                                                 features=self.features(), use_videos=True, **kw)
        log.info("recording to %s (%d episodes already)", self.root, self.dataset.num_episodes)

    @property
    def sidecar(self) -> Path:
        p = self.root / "meta" / "dlsj"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------ commands
    def command(self, name: str) -> None:
        self.commands.put(name)

    def stop(self) -> None:
        self._stop.set()

    def _teleop_live(self) -> bool:
        latest = getattr(self.teleop, "latest", None)
        return latest is not None and latest.get("phase") == "live"

    def _handle(self, c: str) -> None:
        if c == "space":
            c = {"idle": "start", "recording": "stop", "review": "accept"}.get(self.phase, "")
        if c == "start" and self.phase == "idle":
            if not self._teleop_live():
                self.message = "follower is not live: press ENGAGE (e) first"
                return
            self._frames, self._paused_ticks = 0, 0
            self._take_start = time.perf_counter()
            self.phase, self.message = "recording", "recording: SPACE to stop"
        elif c == "stop" and self.phase == "recording":
            self._to_review("stopped")
        elif c in ("accept", "fail") and self.phase in ("recording", "review"):
            if self.phase == "recording":
                self._to_review("stopped")
            self._save(success=c == "accept")
        elif c == "discard" and self.phase in ("recording", "review"):
            self.dataset.clear_episode_buffer()
            self.phase, self.message = "idle", f"discarded ({self._take_duration or time.perf_counter() - self._take_start:.1f}s)"
            self._take_duration = 0.0

    def _to_review(self, why: str) -> None:
        self._take_duration = time.perf_counter() - self._take_start
        self.phase = "review"
        self.message = (f"{why} after {self._take_duration:.1f}s, {self._frames} frames: "
                        "SPACE accept, f failure, h discard")

    def _save(self, success: bool) -> None:
        if self._frames == 0:
            self.dataset.clear_episode_buffer()
            self.phase, self.message = "idle", "nothing recorded"
            return
        self.phase, self.message = "saving", "saving episode…"
        ep = self.dataset.num_episodes
        self.dataset.save_episode()
        label = {
            "episode_index": ep, "success": success, "duration_s": round(self._take_duration, 3),
            "n_frames": self._frames, "paused_ticks": self._paused_ticks, "method": "teleop",
            "operator": self.operator, "task": self.task, "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.sidecar / "labels.jsonl", "a") as fh:
            fh.write(json.dumps(label) + "\n")
        self.saved["success" if success else "failure"] += 1
        if success:
            self.stats.add(self._take_duration)
        self.phase = "idle"
        self.message = f"saved episode {ep} as {'SUCCESS' if success else 'FAILURE'} ({self._take_duration:.1f}s)"
        self._take_duration = 0.0

    # ------------------------------------------------------------------ sampling
    def _frame(self) -> dict | None:
        latest = getattr(self.teleop, "latest", None)
        if latest is None or latest.get("phase") != "live":
            return None
        frame = {
            "observation.state": np.asarray(latest["follower"], dtype=np.float32),
            "action": np.asarray(latest["cmd"], dtype=np.float32),
            "task": self.task,
        }
        for name, stream in self.streams.items():
            img = stream.frame
            if img is None:
                return None
            h, w, _ = self.cam_shapes[name]
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            frame[f"observation.images.{name}"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return frame

    def tick(self) -> None:
        """One recorder step: apply queued commands, then record a frame if a take is running."""
        while not self.commands.empty():
            with self.lock:
                self._handle(self.commands.get_nowait())
        with self.lock:
            if self.phase != "recording":
                return
            frame = self._frame()
            if frame is None:
                self._paused_ticks += 1  # follower not live (held / re-engaging) or a camera dropped out
            else:
                self.dataset.add_frame(frame)
                self._frames += 1
            if time.perf_counter() - self._take_start >= self.episode_time_s:
                self._to_review(f"time limit {self.episode_time_s:.0f}s")

    def run(self) -> None:
        period = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001  keep the UI alive; the take is discarded
                log.exception("recorder error")
                with self.lock:
                    self.dataset.clear_episode_buffer()
                    self.phase, self.message = "idle", f"error, take discarded: {e}"
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)

    def close(self) -> None:
        with self.lock:
            if self.phase in ("recording", "review") and self.dataset is not None:
                self.dataset.clear_episode_buffer()
            if self.dataset is not None:
                self.dataset.finalize()

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = time.perf_counter() - self._take_start if self.phase == "recording" else self._take_duration
            return {
                "phase": self.phase, "message": self.message, "elapsed": round(elapsed, 2), "frames": self._frames,
                "episodes": self.dataset.num_episodes if self.dataset is not None else 0, "saved": dict(self.saved),
                "stats": self.stats.summary(), "task": self.task, "root": str(self.root),
                "teleop_live": self._teleop_live(),
            }


# ----------------------------------------------------------------------------- web
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SO-101 Record</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e6e8ee;--mute:#8a90a0;--ok:#3ecf8e;--bad:#ff6b6b;--rec:#ff4d4d;--acc:#6aa9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 ui-sans-serif,system-ui,-apple-system}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid #262a33;position:sticky;top:0;background:var(--bg);z-index:2}
h1{font-size:15px;margin:0}.pill{padding:2px 8px;border-radius:99px;background:#262a33;color:var(--mute);font-variant-numeric:tabular-nums}
.pill.ok{color:var(--ok)}.pill.bad{color:var(--bad)}
main{display:grid;grid-template-columns:1fr 340px;gap:16px;padding:16px}@media(max-width:960px){main{grid-template-columns:1fr}}
#cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;align-content:start}
.cam{background:var(--card);border-radius:10px;overflow:hidden;border:3px solid transparent}.cam img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#000}
.cam .bar{padding:6px 10px;color:var(--mute)}.cam .bar b{color:var(--fg)}
body.rec .cam{border-color:var(--rec)}
.side{background:var(--card);border-radius:10px;padding:14px;align-self:start;display:flex;flex-direction:column;gap:12px}
#big{font-size:44px;font-weight:700;font-variant-numeric:tabular-nums;text-align:center}
#phase{text-align:center;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
body.rec #phase{color:var(--rec)}body.review #phase{color:var(--acc)}
#msg{color:var(--mute);min-height:2.8em;text-align:center}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:8px}button{background:#262a33;color:var(--fg);border:1px solid #333845;border-radius:8px;padding:10px;cursor:pointer;font-size:14px}
button.primary{grid-column:1/-1;font-size:18px;padding:14px;background:#2b3240}body.rec button.primary{background:#5a1f1f}
kbd{background:#262a33;border-radius:4px;padding:0 5px;font-size:12px}.mute{color:var(--mute);font-size:12px}
.bar2{height:6px;background:#262a33;border-radius:4px;overflow:hidden}.bar2>div{height:100%;width:0;background:var(--ok)}
</style></head><body>
<header><h1>SO-101 record</h1><span id="tele" class="pill">teleop…</span><span id="eps" class="pill"></span><span style="flex:1"></span><span id="root" class="mute"></span></header>
<main><section id="cams"></section>
<aside class="side">
<div id="phase">idle</div><div id="big">0.0s</div><div class="bar2"><div id="prog"></div></div><div id="msg"></div>
<div class="btns"><button id="main" class="primary">Start <kbd>space</kbd></button>
<button id="fail">Failure <kbd>f</kbd></button><button id="disc">Discard <kbd>h</kbd></button>
<button id="eng">Engage <kbd>e</kbd></button><button id="home">Go home</button></div>
<div id="task" class="mute"></div><div id="stats" class="mute"></div>
<div class="mute">Beeps: rising = recording started, falling = stopped. Keys work while this tab is focused.</div>
</aside></main>
<script>
const TARGET=__TARGET__;let last=null,teleLive=false;const seen={},cams=document.getElementById('cams');
const AC=window.AudioContext||window.webkitAudioContext;let ac=null;
function tone(seq){try{ac=ac||new AC();let t=ac.currentTime;for(const[f,d]of seq){if(f>0){const o=ac.createOscillator(),g=ac.createGain();
o.frequency.value=f;o.connect(g);g.connect(ac.destination);g.gain.setValueAtTime(0.0001,t);g.gain.exponentialRampToValueAtTime(0.3,t+0.01);
g.gain.exponentialRampToValueAtTime(0.0001,t+d);o.start(t);o.stop(t+d+0.02);}t+=d;}}catch(e){}}
const BEEP={start:[[880,.08],[0,.04],[1320,.12]],stop:[[660,.10],[0,.03],[330,.22]]};
const post=p=>fetch(p,{method:'POST'});const rec=c=>post('/rec/'+c);const tele=c=>post('/cmd/'+c);
document.getElementById('main').onclick=()=>rec('space');document.getElementById('fail').onclick=()=>rec('fail');
document.getElementById('disc').onclick=()=>rec('discard');document.getElementById('home').onclick=()=>tele('home');
document.getElementById('eng').onclick=()=>tele(teleLive?'hold':'engage');
document.addEventListener('keydown',e=>{if(e.repeat||e.target.tagName==='INPUT')return;const k=e.key.toLowerCase();
 if(k===' '){e.preventDefault();rec('space');}else if(k==='f')rec('fail');else if(k==='h'||k==='backspace'){e.preventDefault();rec('discard');}
 else if(k==='e')tele(teleLive?'hold':'engage');});
async function tick(){try{
 const s=await (await fetch('/state')).json(),r=await (await fetch('/rec/state')).json();
 for(const c of s.cameras){if(!seen[c.name]){const d=document.createElement('div');d.className='cam';
  d.innerHTML=`<img src="/stream/${encodeURIComponent(c.name)}"><div class="bar"><b>${c.name}</b> <span></span></div>`;cams.appendChild(d);seen[c.name]=d;}
  seen[c.name].querySelector('.bar span').textContent=c.error?c.error:`${c.size||'…'} · ${c.fps} fps`;}
 const t=s.teleop;teleLive=!!(t&&(t.phase==='live'||t.phase==='engaging'));const te=document.getElementById('tele');
 te.textContent=t?`${t.phase} · ${t.hz} Hz`:'no robot';te.className='pill '+(t&&t.phase==='live'?'ok':'bad');
 document.getElementById('eng').innerHTML=(teleLive?'Hold':'Engage')+' <kbd>e</kbd>';
 if(last&&last!==r.phase){if(r.phase==='recording')tone(BEEP.start);else if(last==='recording')tone(BEEP.stop);}
 last=r.phase;document.body.className=r.phase==='recording'?'rec':(r.phase==='review'?'review':'');
 document.getElementById('phase').textContent=r.phase;document.getElementById('big').textContent=r.elapsed.toFixed(1)+'s';
 const pr=document.getElementById('prog');pr.style.width=TARGET?Math.min(100,100*r.elapsed/TARGET)+'%':'0';pr.style.background=TARGET&&r.elapsed>TARGET?'var(--bad)':'var(--ok)';
 document.getElementById('msg').textContent=r.message;document.getElementById('main').innerHTML=
  ({idle:'Start',recording:'Stop',review:'Accept',saving:'Saving…'}[r.phase]||r.phase)+' <kbd>space</kbd>';
 document.getElementById('eps').textContent=`${r.episodes} episodes · ✓${r.saved.success} ✗${r.saved.failure}`;
 document.getElementById('task').textContent='Task: '+r.task;document.getElementById('stats').textContent=r.stats+(TARGET?` · target ${TARGET}s`:'');
 document.getElementById('root').textContent=r.root;
}catch(e){document.getElementById('tele').textContent='server gone';}}
setInterval(tick,100);tick();
</script></body></html>"""


def make_handler(streams: dict, teleop, recorder: EpisodeRecorder, target_s: float | None):
    base = teleop_web.make_handler(streams, teleop)
    page = PAGE.replace("__TARGET__", json.dumps(target_s)).encode()

    class Handler(base):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/rec/state":
                return self._send(200, json.dumps(recorder.snapshot()).encode(), "application/json")
            return super().do_GET()

        def do_POST(self):
            name = self.path.removeprefix("/rec/")
            if self.path.startswith("/rec/") and name in ("space", "start", "stop", "accept", "fail", "discard"):
                recorder.command(name)
                return self._send(200, b"{}", "application/json")
            return super().do_POST()

    return Handler


# ----------------------------------------------------------------------------- main
def main() -> None:
    from http.server import ThreadingHTTPServer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="dataset name, e.g. local/duplo_teleop")
    p.add_argument("--task", required=True, help="language instruction stored with every frame")
    p.add_argument("--root", help="dataset directory (default data/<repo name>)")
    p.add_argument("--resume", action="store_true", help="append to an existing dataset")
    p.add_argument("--record-fps", type=int, default=30)
    p.add_argument("--episode-time", type=float, default=60.0, help="a take stops itself after this many seconds")
    p.add_argument("--target-s", type=float, help="tempo target shown as a progress bar")
    p.add_argument("--operator", default="")
    p.add_argument("--vcodec", default="h264", help="h264 is cheap to encode; 'auto' tries hardware encoders")
    p.add_argument("--follower-port", default="auto")
    p.add_argument("--leader-port", default="auto")
    p.add_argument("--follower-id", default="follower")
    p.add_argument("--leader-id", default="leader")
    p.add_argument("--calibrate", choices=["follower", "leader", "both"])
    p.add_argument("--ramp-speed", type=float, default=30.0)
    p.add_argument("--invert", default="none",
                   help="leader joints mirrored onto the follower, comma-separated, or 'none' (as teleop_web)")
    p.add_argument("--match-deg", type=float, default=5.0,
                   help="auto-engage once the leader is within this many degrees of the follower (as teleop_web)")
    p.add_argument("--fps", type=int, default=60, help="teleop control loop rate")
    p.add_argument("--max-relative-target", type=float, default=None)
    p.add_argument("--auto-engage", action="store_true")
    p.add_argument("--cameras", required=True, help="name=index,... e.g. wrist=0,top=2,inspect=4")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--jpeg-quality", type=int, default=75)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    args.no_robot = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if "auto" in (args.follower_port, args.leader_port):
        follower, leader, volts = arms.find_ports()
        log.info("arms found by supply voltage: %s", {k: v for k, v in volts.items() if v is not None})
        args.follower_port = follower if args.follower_port == "auto" else args.follower_port
        args.leader_port = leader if args.leader_port == "auto" else args.leader_port
        if not (args.follower_port and args.leader_port):
            p.error("could not find both arms; check USB cables and the follower's 12 V supply")

    cams = teleop_web.parse_cameras(args.cameras, 8)
    streams = {n: teleop_web.CameraStream(n, i, args.width, args.height, args.cam_fps, args.jpeg_quality).start()
               for n, i in cams.items()}
    teleop = teleop_web.Teleop(args)
    recorder = EpisodeRecorder(teleop, streams, args.repo_id, args.task, args.record_fps, args.root,
                               args.episode_time, args.operator, args.vcodec, args.resume)
    server = None
    worker = rec_thread = None
    try:
        teleop.connect()  # interactive calibration happens here when needed
        recorder.open_dataset()
        worker = threading.Thread(target=teleop.run, name="control", daemon=True)
        worker.start()
        rec_thread = threading.Thread(target=recorder.run, name="recorder", daemon=True)
        rec_thread.start()
        server = ThreadingHTTPServer((args.host, args.port), make_handler(streams, teleop, recorder, args.target_s))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
        log.info("record UI: http://localhost:%d  (cameras: %s)", args.port, ", ".join(cams))
        while worker.is_alive():
            worker.join(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        if rec_thread is not None:
            rec_thread.join(5)
        recorder.close()
        teleop.stop()
        if worker is not None:
            try:
                worker.join(20)
            except KeyboardInterrupt:
                log.warning("second Ctrl+C: releasing torque now")
        teleop.disconnect()
        for s in streams.values():
            s.stop()
        if server is not None:
            server.shutdown()


if __name__ == "__main__":
    main()
