"""SO-101 leader→follower teleop with every camera in a browser at http://localhost:8000.

    uv run python -m dlsj.teleop_web --cameras wrist=0,d415=1

* Ports are found automatically by servo supply voltage (follower ~12 V, leader ~5 V).
* Calibration is stored in the repo under `calibration/` and checked on every start; a missing or bad
  file (unmoved joint, range wrapped around the encoder) starts a fresh calibration in the terminal.
  `--calibrate follower|leader|both` forces one.
* Start-up: the follower ramps slowly to its saved home pose and holds. ENGAGE (web button, or Enter in
  the terminal) ramps it onto the leader's pose and then it follows. "save home" stores the follower's
  current pose as home. Ctrl+C ramps back home, then releases torque.
* Cameras are read with plain OpenCV in their own threads, so a slow or missing camera never stalls teleop.
  `--no-robot --cameras auto` shows every camera with its index.
"""

import argparse
import json
import logging
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from dlsj import arms

log = logging.getLogger("teleop_web")


# ----------------------------------------------------------------------------- cameras
class CameraStream:
    """Grabs frames from one OpenCV camera in a thread and keeps the latest one as JPEG."""

    def __init__(self, name: str, index: int, width: int, height: int, fps: int, quality: int):
        self.name, self.index = name, index
        self.width, self.height, self.fps = width, height, fps
        self.quality = quality
        self.jpeg: bytes | None = None
        self.frame: np.ndarray | None = None  # latest raw BGR frame, for recording
        self.seq = 0
        self.measured_fps = 0.0
        self.shape: tuple[int, int] | None = None
        self.error: str | None = None
        self.cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cam-{name}", daemon=True)

    def start(self) -> "CameraStream":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index)
        if self.width and self.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def _run(self) -> None:
        params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.error = f"cannot open camera index {self.index}"
                time.sleep(2.0)
                continue
            self.error = None
            t_last, n = time.perf_counter(), 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.error = "read failed, reopening"
                    break
                ok, buf = cv2.imencode(".jpg", frame, params)
                if not ok:
                    continue
                n += 1
                now = time.perf_counter()
                if now - t_last >= 1.0:
                    self.measured_fps, t_last, n = n / (now - t_last), now, 0
                with self.cond:
                    self.jpeg, self.shape, self.frame = buf.tobytes(), frame.shape[:2], frame
                    self.seq += 1
                    self.cond.notify_all()
            cap.release()
            time.sleep(0.5)

    def info(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "fps": round(self.measured_fps, 1),
            "size": f"{self.shape[1]}x{self.shape[0]}" if self.shape else None,
            "error": self.error,
        }


def probe_camera_indices(max_index: int) -> list[int]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            found.append(i)
    return found


def parse_cameras(spec: str, max_index: int) -> dict[str, int]:
    if spec == "auto":
        idx = probe_camera_indices(max_index)
        log.info("auto-detected camera indices: %s", idx)
        return {f"cam{i}": i for i in idx}
    out = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, index = part.partition("=")
        out[name.strip()] = int(index)
    return out


# ----------------------------------------------------------------------------- teleop
def step_toward(cmd: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    """Move `cmd` toward `target`, no joint by more than `max_step`."""
    return cmd + np.clip(target - cmd, -max_step, max_step)


class Teleop:
    """Follower control loop with a home pose and a smooth engage.

    Phases: connecting → homing (ramp to the saved home pose) → ready (hold, wait for engage) →
    engaging (ramp onto the leader's pose, chasing it if it moves) → live (follow the leader).
    Stopping ramps back home before torque is released.
    """

    def __init__(self, args):
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.teleoperators import make_teleoperator_from_config
        from lerobot.teleoperators.so_leader import SO101LeaderConfig

        self.args = args
        self.fps = args.fps
        self.max_step = args.ramp_speed / args.fps  # units per tick while homing / engaging
        self.robot = make_robot_from_config(
            SO101FollowerConfig(
                port=args.follower_port,
                id=args.follower_id,
                calibration_dir=arms.FOLLOWER_CAL_DIR,
                max_relative_target=args.max_relative_target,
            )
        )
        self.leader = make_teleoperator_from_config(
            SO101LeaderConfig(port=args.leader_port, id=args.leader_id, calibration_dir=arms.LEADER_CAL_DIR)
        )
        self.joints = list(self.robot.action_features)  # "shoulder_pan.pos", ...
        self.home = arms.load_home(args.follower_id)
        inverted = {j.strip() for j in getattr(args, "invert", "none").split(",") if j.strip() and j.strip() != "none"}
        if unknown := inverted - {j.removesuffix(".pos") for j in self.joints}:
            raise ValueError(f"--invert: unknown joints {sorted(unknown)}")
        # Leader joints whose direction is mirrored relative to the follower (flipped around 0 = mid travel).
        self.sign = np.array([-1.0 if j.removesuffix(".pos") in inverted else 1.0 for j in self.joints])
        align = arms.load_align(args.leader_id)
        self.align = self._vec(align) if align else np.zeros(len(self.joints))
        self.phase = "connecting"
        self.message = ""
        self.cmd: np.ndarray | None = None
        self.commands: queue.Queue[str] = queue.Queue()
        self.state: dict = {"phase": self.phase}
        self.lock = threading.Lock()
        self._stop = threading.Event()

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        if arms.ensure_calibrated(self.robot, "follower", force=self.args.calibrate in ("follower", "both")):
            stale = arms.retire_home(self.args.follower_id)
            if stale:
                log.warning("follower recalibrated: old home pose moved to %s; save a new one", stale.name)
            self.home = None
        if arms.ensure_calibrated(self.leader, "leader", force=self.args.calibrate in ("leader", "both")):
            p = arms.align_path(self.args.leader_id)
            if p.exists():
                p.replace(p.with_suffix(".json.stale"))
                log.warning("leader recalibrated: old align offsets set aside; align again")
            self.align = np.zeros(len(self.joints))

    def disconnect(self) -> None:
        for dev in (self.leader, self.robot):
            try:
                dev.disconnect()
            except Exception as e:  # noqa: BLE001
                log.warning("disconnect %s: %s", dev, e)

    def stop(self) -> None:
        self._stop.set()

    def command(self, name: str) -> None:
        self.commands.put(name)

    # ------------------------------------------------------------------ helpers
    def _vec(self, d: dict) -> np.ndarray:
        return np.array([float(d[j]) for j in self.joints])

    def _dict(self, v: np.ndarray) -> dict:
        return {j: float(x) for j, x in zip(self.joints, v, strict=True)}

    def _set_phase(self, phase: str, message: str = "") -> None:
        if phase != self.phase:
            log.info("phase: %s%s", phase, f" ({message})" if message else "")
        self.phase, self.message = phase, message

    def _ready(self) -> None:
        if self.home is None:
            self._set_phase("ready", "no home pose saved: pose the follower and press 'save home'")
        else:
            self._set_phase("ready", "bring the leader to the follower's pose (or press ENGAGE)")

    def _handle(self, c: str, follower: np.ndarray) -> None:
        if c == "engage" and self.phase in ("ready", "homing"):
            self._set_phase("engaging", "ramping onto the leader")
        elif c == "hold" and self.phase in ("live", "engaging"):
            self.cmd = follower.copy()
            self.latch_ok = False  # don't re-latch at once: the leader is still on the follower's pose
            self._set_phase("ready", "holding; move the leader away and back, or press ENGAGE")
        elif c == "home":
            if self.home is None:
                self._notify("no home pose saved yet")
            else:
                self._set_phase("homing", "moving to home")
        elif c == "align":
            if self.phase not in ("ready", "homing"):
                self._notify("align only while holding (press hold first)")
                return
            self.align = follower - self._raw_lead
            p = arms.save_align(self.args.leader_id, self._dict(self.align))
            self._notify("aligned: " + ", ".join(
                f"{j.removesuffix('.pos')} {d:+.1f}" for j, d in zip(self.joints, self.align, strict=True) if abs(d) >= 0.5
            ) + f" (saved {p.name})")
        elif c == "save_home":
            pose = self._dict(follower)
            p = arms.save_home(self.args.follower_id, pose)
            self.home = json.loads(p.read_text())
            self._notify(f"home saved to {p}")

    def _notify(self, msg: str, seconds: float = 6.0) -> None:
        log.info(msg)
        self.notice, self.notice_until = msg, time.perf_counter() + seconds

    def _match_latch(self, lead: np.ndarray) -> None:
        """In `ready`, engage on its own once the leader is within `match_deg` of the held pose on every joint."""
        tol = getattr(self.args, "match_deg", 5.0)
        if tol <= 0:
            return
        diff = np.abs(lead - self.cmd)
        worst = int(np.argmax(diff))
        if not self.latch_ok and diff[worst] > 2 * tol:
            self.latch_ok = True
        if self.latch_ok and diff[worst] < tol:
            self._set_phase("engaging", "leader matched, engaging")
        elif self.latch_ok and self.home is not None:
            self.message = (f"bring the leader to the follower's pose to engage: "
                            f"{self.joints[worst].removesuffix('.pos')} is {diff[worst]:.0f} off")

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        period = 1.0 / self.fps
        self.latch_ok = True
        self.notice, self.notice_until = "", 0.0
        obs = self.robot.get_observation()
        self.cmd = self._vec(obs)
        if self.home is not None:
            self._set_phase("homing", "moving to home")
        else:
            self._ready()
        if self.args.auto_engage:
            self.command("engage")

        t_rate, n, hz = time.perf_counter(), 0, 0.0
        stop_deadline = None
        while True:
            t0 = time.perf_counter()
            obs = self.robot.get_observation()
            follower = self._vec(obs)
            raw_lead = self.sign * self._vec(self.leader.get_action())
            self._raw_lead = raw_lead

            while not self.commands.empty():
                self._handle(self.commands.get_nowait(), follower)
            lead = raw_lead + self.align  # after commands, so a fresh align applies this tick

            if self._stop.is_set() and self.phase != "stopping":
                stop_deadline = t0 + 15.0
                self._set_phase("stopping", "returning home before releasing torque" if self.home else "")
            if self.phase == "stopping" and (
                self.home is None or t0 > stop_deadline or np.max(np.abs(self.cmd - self._vec(self.home))) < 0.5
            ):
                break

            if self.phase in ("homing", "stopping") and self.home is not None:
                self.cmd = step_toward(self.cmd, self._vec(self.home), self.max_step)
                if self.phase == "homing" and np.max(np.abs(self.cmd - self._vec(self.home))) < 0.5:
                    self._ready()
            elif self.phase == "engaging":
                self.cmd = step_toward(self.cmd, lead, self.max_step)
                if np.max(np.abs(self.cmd - lead)) < 1.0:
                    self._set_phase("live", "following the leader")
            elif self.phase == "live":
                self.cmd = lead
            elif self.phase == "ready":
                self._match_latch(lead)
            self.robot.send_action(self._dict(self.cmd))
            with self.lock:
                self.latest = {"t": t0, "follower": follower.copy(), "cmd": self.cmd.copy(), "phase": self.phase}

            n += 1
            if t0 - t_rate >= 1.0:
                hz, t_rate, n = n / (t0 - t_rate), t0, 0
            names = [j.removesuffix(".pos") for j in self.joints]
            with self.lock:
                self.state = {
                    "phase": self.phase,
                    "message": self.notice if t0 < self.notice_until else self.message,
                    "has_home": self.home is not None,
                    "hz": round(hz, 1),
                    "leader": dict(zip(names, np.round(lead, 1).tolist(), strict=True)),
                    "follower": dict(zip(names, np.round(follower, 1).tolist(), strict=True)),
                }
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)


def stdin_commands(teleop: Teleop) -> None:
    """Enter: engage / hold. h: go home. s: save home."""
    for line in sys.stdin:
        c = line.strip().lower()
        if c == "":
            teleop.command("hold" if teleop.phase in ("live", "engaging") else "engage")
        elif c == "h":
            teleop.command("home")
        elif c == "s":
            teleop.command("save_home")
        elif c == "a":
            teleop.command("align")


# ----------------------------------------------------------------------------- web
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SO-101 Teleop</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e6e8ee;--mute:#8a90a0;--ok:#3ecf8e;--bad:#ff6b6b;--acc:#6aa9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 ui-sans-serif,system-ui,-apple-system}
header{display:flex;gap:16px;align-items:center;padding:10px 16px;border-bottom:1px solid #262a33;position:sticky;top:0;background:var(--bg);z-index:2}
h1{font-size:15px;margin:0}.pill{padding:2px 8px;border-radius:99px;background:#262a33;color:var(--mute);font-variant-numeric:tabular-nums}
.pill.ok{color:var(--ok)}.pill.bad{color:var(--bad)}button{background:#262a33;color:var(--fg);border:1px solid #333845;border-radius:6px;padding:4px 10px;cursor:pointer}
main{display:grid;grid-template-columns:1fr 300px;gap:16px;padding:16px}@media(max-width:900px){main{grid-template-columns:1fr}}
#cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px}
.cam{background:var(--card);border-radius:10px;overflow:hidden}.cam img{display:block;width:100%;background:#000;aspect-ratio:4/3;object-fit:contain;cursor:zoom-in}
.cam .bar{display:flex;justify-content:space-between;padding:6px 10px;color:var(--mute);font-variant-numeric:tabular-nums}.cam .bar b{color:var(--fg)}
.cam.big{grid-column:1/-1}.side{background:var(--card);border-radius:10px;padding:12px;align-self:start}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:3px 4px;text-align:right}td:first-child,th:first-child{text-align:left;color:var(--mute)}
th{color:var(--mute);font-weight:500}.err{color:var(--bad)}
</style></head><body>
<header><h1>SO-101 teleop</h1><span id="status" class="pill">connecting…</span><span id="hz" class="pill"></span>
<span id="msg" style="color:var(--mute)"></span><span style="flex:1"></span>
<span id="ctl" hidden><button id="engage">engage</button> <button id="home">go home</button> <button id="save">save home</button> <button id="align">align</button></span></header>
<main><section id="cams"></section>
<aside class="side"><table><thead><tr><th>joint</th><th>leader</th><th>follower</th><th>Δ</th></tr></thead><tbody id="joints"></tbody></table>
<p style="color:var(--mute);font-size:12px">Click a camera to enlarge.<br>Align: put both arms in the same pose while holding, then press align.<br>Terminal: Enter = engage/hold, h = home, s = save home, a = align, Ctrl+C = return home and stop.</p></aside></main>
<script>
const cams=document.getElementById('cams'),seen={};
async function tick(){
  try{
    const s=await (await fetch('/state')).json();
    for(const c of s.cameras){
      if(!seen[c.name]){
        const d=document.createElement('div');d.className='cam';
        d.innerHTML=`<img src="/stream/${encodeURIComponent(c.name)}"><div class="bar"><b>${c.name}</b><span></span></div>`;
        d.querySelector('img').onclick=()=>d.classList.toggle('big');
        cams.appendChild(d);seen[c.name]=d;
      }
      seen[c.name].querySelector('.bar span').innerHTML=c.error?`<span class="err">${c.error}</span>`:`idx ${c.index} · ${c.size||'…'} · ${c.fps} fps`;
    }
    const t=s.teleop,st=document.getElementById('status');
    if(!t){st.textContent='cameras only';st.className='pill';}
    else if(!t.leader){st.textContent=t.phase+'… (see terminal)';st.className='pill';}
    else{st.textContent=t.phase;st.className='pill '+(t.phase==='live'?'ok':(t.phase==='ready'?'':'bad'));
      document.getElementById('msg').textContent=t.message||'';
      document.getElementById('hz').textContent=t.hz+' Hz';document.getElementById('ctl').hidden=false;
      const live=t.phase==='live'||t.phase==='engaging';eng.textContent=live?'hold':'engage';eng.dataset.cmd=live?'hold':'engage';
      document.getElementById('joints').innerHTML=Object.keys(t.leader).map(k=>{const l=t.leader[k],f=t.follower[k];
        return `<tr><td>${k}</td><td>${l.toFixed(1)}</td><td>${f!==undefined?f.toFixed(1):'–'}</td><td>${f!==undefined?(l-f).toFixed(1):''}</td></tr>`}).join('');}
  }catch(e){document.getElementById('status').textContent='server gone';document.getElementById('status').className='pill bad';}
}
const eng=document.getElementById('engage');eng.dataset.cmd='engage';
const cmd=c=>fetch('/cmd/'+c,{method:'POST'});
eng.onclick=()=>cmd(eng.dataset.cmd);
document.getElementById('home').onclick=()=>cmd('home');
document.getElementById('align').onclick=()=>{if(confirm("Both arms in the SAME physical pose? The leader will be offset to match the follower."))cmd('align')};
document.getElementById('save').onclick=()=>{if(confirm("Save the follower current pose as home?"))cmd('save_home')};
setInterval(tick,250);tick();
</script></body></html>"""


def make_handler(streams: dict[str, CameraStream], teleop: Teleop | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            name = self.path.removeprefix("/cmd/")
            if teleop is not None and self.path.startswith("/cmd/") and name in ("engage", "hold", "home", "save_home", "align"):
                teleop.command(name)
                return self._send(200, b"{}", "application/json")
            self._send(404, b"", "text/plain")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/state":
                body = {
                    "cameras": [s.info() for s in streams.values()],
                    "teleop": teleop.snapshot() if teleop else None,
                }
                return self._send(200, json.dumps(body).encode(), "application/json")
            if self.path.startswith("/snapshot/"):
                cam = streams.get(self.path.removeprefix("/snapshot/").removesuffix(".jpg"))
                if cam and cam.jpeg:
                    return self._send(200, cam.jpeg, "image/jpeg")
                return self._send(404, b"", "text/plain")
            if self.path.startswith("/stream/"):
                from urllib.parse import unquote

                cam = streams.get(unquote(self.path.removeprefix("/stream/")))
                if cam is None:
                    return self._send(404, b"", "text/plain")
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with cam.cond:
                            cam.cond.wait_for(lambda: cam.seq != last, timeout=2.0)
                            jpeg, last = cam.jpeg, cam.seq
                        if jpeg is None:
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpeg)).encode()
                            + b"\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError):
                    return
            self._send(404, b"", "text/plain")

    return Handler


# ----------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--follower-port", default="auto", help="'auto' = the arm with a ~12 V servo supply")
    p.add_argument("--leader-port", default="auto", help="'auto' = the arm with a ~5 V servo supply")
    p.add_argument("--calibrate", choices=["follower", "leader", "both"], help="force a fresh calibration")
    p.add_argument("--ramp-speed", type=float, default=30.0, help="deg/s when homing and engaging")
    p.add_argument("--invert", default="none",
                   help="comma list of leader joints to mirror (e.g. wrist_roll,wrist_flex), or 'none'")
    p.add_argument("--match-deg", type=float, default=5.0,
                   help="engage on its own when the leader is within this many degrees of the follower (0 = off)")
    p.add_argument("--auto-engage", action="store_true", help="follow the leader as soon as homing ends")
    p.add_argument("--follower-id", default="follower")
    p.add_argument("--leader-id", default="leader")
    p.add_argument("--no-robot", action="store_true", help="cameras only, no arms")
    p.add_argument("--fps", type=int, default=60, help="teleop loop rate")
    p.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="cap per-step follower motion (degrees) for safety; unset = no cap",
    )
    p.add_argument("--cameras", default="auto", help='"auto" or name=index,... e.g. wrist=1,top=2')
    p.add_argument("--max-camera-index", type=int, default=8)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.no_robot and "auto" in (args.follower_port, args.leader_port):
        follower, leader, volts = arms.find_ports()
        log.info("arms found by supply voltage: %s", {k: v for k, v in volts.items() if v is not None})
        args.follower_port = follower if args.follower_port == "auto" else args.follower_port
        args.leader_port = leader if args.leader_port == "auto" else args.leader_port
        if not (args.follower_port and args.leader_port):
            p.error(f"could not find both arms (follower={args.follower_port}, leader={args.leader_port}); "
                    "check USB cables and the follower's 12 V supply")
        log.info("follower=%s leader=%s", args.follower_port, args.leader_port)

    cams = parse_cameras(args.cameras, args.max_camera_index)
    streams = {
        name: CameraStream(name, idx, args.width, args.height, args.cam_fps, args.jpeg_quality).start()
        for name, idx in cams.items()
    }

    teleop = None if args.no_robot else Teleop(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(streams, teleop))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    log.info("web UI: http://localhost:%d  (cameras: %s)", args.port, ", ".join(cams) or "none")

    worker = None
    try:
        if teleop is None:
            while True:
                time.sleep(1)
        else:
            teleop.connect()  # interactive calibration happens here when needed
            worker = threading.Thread(target=teleop.run, name="control", daemon=True)
            worker.start()
            threading.Thread(target=stdin_commands, args=(teleop,), daemon=True).start()
            log.info("control loop at %d Hz. Enter = engage/hold, h = home, s = save home, Ctrl+C = home + stop",
                     args.fps)
            while worker.is_alive():
                worker.join(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if teleop is not None:
            teleop.stop()
            if worker is not None:
                try:
                    worker.join(20)
                except KeyboardInterrupt:
                    log.warning("second Ctrl+C: releasing torque now")
            teleop.disconnect()
        for s in streams.values():
            s.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
