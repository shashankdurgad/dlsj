"""Run the fine-tuned π0.5 on the SO-101 follower (no leader), with a web console: http://<host>:8010

Keys (also buttons on the page):
* Space  start a take: the policy drives the arm with the pick prompt.
* M      mark the inspection moment: switch to "place the brick in the <plate> bowl" (as the data was labelled).
* 1/2/3  choose the plate (the place prompt's colour). Allowed any time; used from the next mark.
* + / -  raise / lower the per-tick speed limit (x1.25) while running.
* Q      stop now: discard queued and in-flight actions and hold the arm where it is (torque stays on).

Why the prompt switch is manual: the training data switches the instruction at the operator's mark, and the
in-view classifier fires much earlier than that mark on the RealSense inspection camera, so an automatic switch
would condition the policy differently from training.

Control: a 30 Hz loop sends one joint target per tick. A worker thread runs the policy on the newest frames and
state, and asks again as soon as fewer than `--replan` future actions remain (default: always, i.e. back to back,
~5 chunks/s on the XPS GPU). Each chunk is stamped with the tick its observation was taken at, so action j of a
chunk is meant for tick t0 + j: actions whose time passed during inference are skipped automatically.
Consecutive chunks are combined with `--blend`:
* ensemble   (default) ACT temporal ensembling: every chunk still covering this tick votes, weighted
             w_i = exp(-k * i) with i = 0 the OLDEST chunk (ACT's convention; k = `--ensemble-k`, small k ≈ mean).
* crossfade  the new chunk replaces the plan, fading in linearly from the old plan over `--blend-steps` ticks.
* none       the new chunk replaces the plan outright (the first version; jumps at chunk switches).
Every target is then rate-limited (`--max-step` units per tick, the gripper `--max-step-gripper`), and LeRobot's
`--max-relative-target` stays as the hard per-command clamp. On exit the arm ramps to its saved home pose before
torque is released; it is never dropped mid-air.

    python -m dlsj.deploy.pi05_control --export ~/inference/exports/pi05_junction_int8 --device GPU \
        --cameras wrist=/dev/v4l/by-id/usb-Sonix_..-video-index0 --realsense top=346422061366,inspect=243722062690

The teleop recorder must be stopped first: it owns the follower and the cameras.
"""

import argparse
import collections
import json
import logging
import queue
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np

from dlsj.deploy.pi05_runtime import PICK, PLACE

log = logging.getLogger("pi05_control")
CAMS = ("wrist", "top", "inspect")
BLENDS = ("ensemble", "crossfade", "none")
# Recorded teleop actions change by p95 3-5 and p99 5-8 units per 30 Hz tick (max ~13), per joint incl. gripper.
DEFAULT_MAX_STEP = 8.0


class Controller:
    """Policy → follower loop. `arm` needs read() -> (6,) and send((6,)); cameras need `.frame` (BGR)."""

    def __init__(self, arm, cameras: dict, policy, bowls=("yellow", "purple", "orange"), fps: int = 30,
                 max_step: float = DEFAULT_MAX_STEP, max_step_gripper: float = DEFAULT_MAX_STEP, replan: int = 50,
                 blend: str = "ensemble", ensemble_k: float = 0.01, blend_steps: int = 10, sync: bool = False):
        if blend not in BLENDS:
            raise ValueError(f"blend must be one of {BLENDS}")
        self.arm, self.cameras, self.policy = arm, cameras, policy
        self.bowls, self.fps, self.replan, self.sync = list(bowls), fps, replan, sync
        self.blend, self.ensemble_k, self.blend_steps = blend, ensemble_k, blend_steps
        self.max_step = np.array([max_step] * 5 + [max_step_gripper], dtype=float)
        self.phase = "idle"  # idle -> running -> stopped (Q) / idle
        self.stage = "pick"
        self.bowl: str | None = None
        self.message = "Press Space to start"
        self.t = 0  # control tick counter: chunks are aligned on it
        self.chunks: list[tuple[int, np.ndarray]] = []  # (tick of action 0, (n, 6) actions), oldest first
        self.cmd: np.ndarray | None = None
        self.generation = 0  # bumped by start/stop/mark: results from older requests are dropped
        self.inflight = False
        self.lat_ms: collections.deque = collections.deque(maxlen=30)
        self.last_chunk_t = 0.0
        self.sent = 0
        self.lock = threading.RLock()  # re-entrant: Q and sync-mode inference take it from inside tick()
        self.commands: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()

    # ------------------------------------------------------------------ inputs
    def command(self, name: str) -> None:
        if name in ("q", "stop"):  # act now, not at the next tick
            with self.lock:
                self._handle(name)
            return
        self.commands.put(name)

    def prompt(self) -> str:
        return PICK if self.stage == "pick" else PLACE.format(colour=self.bowl)

    def _obs(self) -> tuple[np.ndarray, dict]:
        import cv2

        imgs = {}
        for name in CAMS:
            frame = self.cameras[name].frame
            if frame is None:
                raise RuntimeError(f"no frame from camera {name}")
            imgs[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.asarray(self.arm.read(), dtype=float), imgs

    def _replan_now(self) -> None:
        """Forget the current plan (new prompt or restart); in-flight results become stale."""
        self.generation += 1
        self.chunks.clear()

    def _handle(self, c: str) -> None:
        if c in ("q", "stop"):
            self._hold("Stopped: holding position. Space to start again")
            self.phase = "stopped"
        elif c.startswith("bowl") and c[4:].isdigit():
            i = int(c[4:]) - 1
            if 0 <= i < len(self.bowls):
                self.bowl = self.bowls[i]
                if self.phase == "running" and self.stage == "place":
                    self._replan_now()  # replan with the new colour
        elif c in ("start", "space") and self.phase != "running":
            self._replan_now()
            self.phase, self.stage, self.message = "running", "pick", ""
            self.cmd = np.asarray(self.arm.read(), dtype=float)
        elif c in ("mark", "m") and self.phase == "running" and self.stage == "pick":
            if self.bowl is None:
                self.message = "Choose a plate (1–3) before marking"
                return
            self.stage = "place"
            self._replan_now()
        elif c in ("faster", "slower"):
            self.max_step = self.max_step * (1.25 if c == "faster" else 0.8)

    def _hold(self, message: str) -> None:
        """Drop all pending motion and hold the present position."""
        self._replan_now()
        present = np.asarray(self.arm.read(), dtype=float)
        self.arm.send(present)
        self.cmd = present
        self.message = message

    # ------------------------------------------------------------------ plan
    def _add_chunk(self, t0: int, chunk: np.ndarray) -> None:
        """Merge a chunk whose action j is meant for tick t0 + j (called at tick self.t)."""
        now = self.t
        if self.blend == "ensemble":
            self.chunks.append((t0, chunk))
        else:
            fresh = chunk[max(0, now - t0):]  # from the current tick on
            if self.blend == "crossfade" and self.chunks:
                n = min(self.blend_steps, len(fresh))
                old = [self._planned(now + j) for j in range(n)]
                for j in range(n):
                    if old[j] is not None:
                        a = (j + 1) / (n + 1)  # 0 -> 1: old plan fades out, new one in
                        fresh[j] = (1 - a) * old[j] + a * fresh[j]
            self.chunks = [(max(now, t0), fresh)]
        self.chunks = [(s, c) for s, c in self.chunks if s + len(c) > now]  # drop expired

    def _planned(self, t: int) -> np.ndarray | None:
        """The plan's target for tick t: weighted over every chunk covering t (ensemble), else the one plan."""
        votes = [c[t - s] for s, c in self.chunks if s <= t < s + len(c)]
        if not votes:
            return None
        if len(votes) == 1:
            return np.array(votes[0], dtype=float)
        w = np.exp(-self.ensemble_k * np.arange(len(votes)))  # i = 0 is the oldest chunk, as in ACT
        return (w[:, None] * np.asarray(votes)).sum(0) / w.sum()

    def remaining(self) -> int:
        """Future ticks covered by the newest chunk."""
        if not self.chunks:
            return 0
        s, c = self.chunks[-1]
        return max(0, s + len(c) - self.t)

    # ------------------------------------------------------------------ inference
    def _request(self) -> None:
        gen, task, t0 = self.generation, self.prompt(), self.t
        state, imgs = self._obs()
        self.inflight = True

        def work():
            t_start = time.perf_counter()
            try:
                chunk = np.asarray(self.policy(state, imgs, task), dtype=float)
            except Exception as e:  # noqa: BLE001
                log.exception("inference failed")
                with self.lock:
                    self.inflight = False
                    self.command("q")  # hold first, then say why
                    self.message = f"Inference failed: {e}. Holding position"
                return
            dt = time.perf_counter() - t_start
            with self.lock:
                self.inflight = False
                self.lat_ms.append(dt * 1e3)
                if gen != self.generation or self.phase != "running":
                    return  # stopped, restarted or re-prompted meanwhile
                self._add_chunk(t0, chunk)
                self.last_chunk_t = time.perf_counter()

        if self.sync:
            work()
        else:
            threading.Thread(target=work, daemon=True, name="pi05-infer").start()

    # ------------------------------------------------------------------ loop
    def tick(self) -> None:
        while True:
            try:
                c = self.commands.get_nowait()
            except queue.Empty:
                break
            with self.lock:
                self._handle(c)
        with self.lock:
            try:
                if self.phase != "running":
                    return
                if not self.inflight and self.remaining() <= self.replan:
                    self._request()
                target = self._planned(self.t)
                if target is not None:
                    self.cmd = self.cmd + np.clip(target - self.cmd, -self.max_step, self.max_step)
                    self.arm.send(self.cmd)
                    self.sent += 1
            finally:
                self.t += 1
                self.chunks = [(s, c) for s, c in self.chunks if s + len(c) > self.t]

    def run(self) -> None:
        period = 1.0 / self.fps
        nxt = time.perf_counter()
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001  never leave the arm chasing an old target
                log.exception("control tick failed")
                with self.lock:
                    try:
                        self._hold(f"Error: {e}")
                    except Exception:  # noqa: BLE001
                        log.exception("hold failed")
                    self.phase = "stopped"
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))
            nxt = max(nxt, time.perf_counter() - period)

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict:
        with self.lock:
            lat = list(self.lat_ms)
            return {"phase": self.phase, "stage": self.stage, "bowl": self.bowls and self.bowl, "bowls": self.bowls,
                    "prompt": self.prompt() if self.stage == "pick" or self.bowl else PICK,
                    "message": self.message, "queued": self.remaining(), "chunks": len(self.chunks),
                    "inflight": self.inflight, "blend": self.blend,
                    "latency_ms": round(lat[-1], 1) if lat else None,
                    "latency_median_ms": round(float(np.median(lat)), 1) if lat else None,
                    "chunk_age_s": round(time.perf_counter() - self.last_chunk_t, 2) if self.last_chunk_t else None,
                    "sent": self.sent, "max_step": np.round(self.max_step, 2).tolist(),
                    "cmd": None if self.cmd is None else np.round(self.cmd, 1).tolist()}


# ----------------------------------------------------------------------------- real hardware
class FollowerArm:
    """SO-101 follower via LeRobot, with the recorder's calibration, stale-goal reset and home pose."""

    def __init__(self, port: str, follower_id: str = "follower", max_relative_target: float | None = 10.0):
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower import SO101FollowerConfig

        from dlsj import arms

        self.arms = arms
        self.id = follower_id
        self.robot = make_robot_from_config(SO101FollowerConfig(
            port=port, id=follower_id, calibration_dir=arms.FOLLOWER_CAL_DIR,
            max_relative_target=max_relative_target, cameras={}))
        self.joints = list(self.robot.action_features)

    def connect(self) -> None:
        # same sequence as the recorder: bus connect, stored calibration, goal := present, configure (torque on)
        self.arms.ensure_calibrated(self.robot, "follower")

    def read(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([float(obs[j]) for j in self.joints])

    def send(self, target: np.ndarray) -> None:
        self.robot.send_action({j: float(v) for j, v in zip(self.joints, target, strict=True)})

    def go_home(self, fps: int = 30, speed: float = 30.0, timeout_s: float = 8.0) -> None:
        home = self.arms.load_home(self.id)
        if not home:
            return
        target = np.array([float(home[j]) for j in self.joints])
        cmd = self.read()
        for _ in range(int(timeout_s * fps)):
            cmd = cmd + np.clip(target - cmd, -speed / fps, speed / fps)
            self.send(cmd)
            if np.abs(target - cmd).max() < 0.5:
                break
            time.sleep(1 / fps)

    def disconnect(self) -> None:
        self.robot.disconnect()


# ----------------------------------------------------------------------------- web console
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Policy run</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#15171a;--panel:#1d2024;--line:#2b2f35;--fg:#f1f2f4;--mute:#80868f;--ok:#3ecf8e;--red:#e31937}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--fg);font:400 15px/1.4 ui-sans-serif,system-ui,sans-serif}
.app{display:grid;grid-template-columns:minmax(0,1fr) 340px;min-height:100vh}
.cams{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:16px;align-content:start}
.cams img{width:100%;border-radius:12px;background:#000;display:block}
.side{padding:24px;display:flex;flex-direction:column;gap:20px;border-left:1px solid var(--line)}
.state{font-size:34px;font-weight:300}.run .state{color:var(--ok)}.stopped .state{color:var(--red)}
.prompt{font-size:17px;min-height:48px}.msg{color:var(--mute);min-height:20px}
.plates{display:flex;gap:14px}.plate{width:54px;height:54px;border-radius:50%;border:0;opacity:.35;cursor:pointer;position:relative}
.plate.sel{opacity:1;box-shadow:0 0 0 3px var(--bg),0 0 0 5px var(--fg)}.plate kbd{position:absolute;right:-6px;bottom:-6px;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:0 5px;font-size:12px}
.btns{display:flex;flex-wrap:wrap;gap:10px}button.k{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:10px;padding:10px 14px;font:inherit;cursor:pointer}
button.stop{background:var(--red);border-color:var(--red);font-weight:600;width:100%;padding:18px;font-size:20px}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:12px;color:var(--mute);font-size:13px;margin-top:auto}.stats b{display:block;color:var(--fg);font-size:24px;font-weight:300}
button:focus-visible{outline:2px solid var(--fg);outline-offset:2px}
@media (max-width:860px){.app{grid-template-columns:1fr}.side{border-left:0}}
</style></head><body><div class="app" id="app">
<section class="cams"><img src="/stream/wrist" alt="Wrist camera"><img src="/stream/top" alt="Top camera"><img src="/stream/inspect" alt="Inspection camera"></section>
<aside class="side">
 <button class="k stop" onclick="cmd('q')">Stop now <kbd>Q</kbd></button>
 <div class="state" id="state">Idle</div>
 <div class="prompt" id="prompt"></div><div class="msg" id="msg"></div>
 <div class="plates" id="plates"></div>
 <div class="btns"><button class="k" onclick="cmd('start')">Start <kbd>Space</kbd></button><button class="k" onclick="cmd('mark')">Mark inspection <kbd>M</kbd></button></div>
 <div class="stats"><span><b id="lat">–</b>ms last inference</span><span><b id="med">–</b>ms median</span><span><b id="q">–</b>actions queued</span><span><b id="age">–</b>s since last chunk</span><span><b id="spd">–</b>max step per tick <kbd>+</kbd><kbd>-</kbd></span><span><b id="bl">–</b>chunk blending</span></div>
</aside></div>
<script>
const $=id=>document.getElementById(id),SW={yellow:'#f2c230',purple:'#9b7fe0',orange:'#f08a24'};
function cmd(c){fetch('/ctl/'+c,{method:'POST'})}
addEventListener('keydown',e=>{if(e.repeat)return;const k=e.key.toLowerCase();
 if(k==='q'){cmd('q');e.preventDefault()}else if(k===' '){cmd('start');e.preventDefault()}else if(k==='m')cmd('mark');
 else if(k==='+'||k==='=')cmd('faster');else if(k==='-'||k==='_')cmd('slower');else if('123'.includes(k)&&k)cmd('bowl'+k)});
async function tick(){try{const r=await fetch('/ctl/state').then(x=>x.json());
 $('app').className='app '+(r.phase==='running'?'run':r.phase);
 $('state').textContent=r.phase==='running'?(r.stage==='pick'?'Running · pick':'Running · place'):r.phase==='stopped'?'Stopped':'Idle';
 $('prompt').textContent=r.prompt;$('msg').textContent=r.message||'';
 const p=$('plates');if(!p.children.length)r.bowls.forEach((c,i)=>{const b=document.createElement('button');b.className='plate';b.style.background=SW[c]||c;b.setAttribute('aria-label',c+' plate');b.innerHTML='<kbd>'+(i+1)+'</kbd>';b.onclick=()=>cmd('bowl'+(i+1));p.appendChild(b)});
 [...p.children].forEach((b,i)=>b.classList.toggle('sel',r.bowl===r.bowls[i]));
 $('lat').textContent=r.latency_ms??'–';$('med').textContent=r.latency_median_ms??'–';$('q').textContent=r.queued;$('age').textContent=r.chunk_age_s??'–';$('spd').textContent=r.max_step[0];$('bl').textContent=r.blend+' ('+r.chunks+')';
}catch(e){$('state').textContent='Disconnected'}}
setInterval(tick,150);tick();
</script></body></html>"""


def make_handler(streams: dict, ctl: Controller):
    from dlsj import teleop_web

    base = teleop_web.make_handler(streams, None)
    allowed = {"q", "stop", "start", "mark", "faster", "slower", *(f"bowl{i}" for i in range(1, 10))}

    class Handler(base):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/ctl/state":
                return self._send(200, json.dumps(ctl.snapshot()).encode(), "application/json")
            return super().do_GET()

        def do_POST(self):
            name = self.path.removeprefix("/ctl/")
            if self.path.startswith("/ctl/") and name in allowed:
                ctl.command(name)
                return self._send(200, b"{}", "application/json")
            return self._send(404, b"", "text/plain")

    return Handler


def warmup(policy, n: int = 2) -> None:
    """The first calls after compiling are slow (0.4-2.6 s); spend them before the arm is live."""
    blank = {c: np.zeros((480, 640, 3), np.uint8) for c in CAMS}
    for _ in range(n):
        t = time.perf_counter()
        policy(np.zeros(6, np.float32), blank, PICK)
        log.info("warm-up inference: %.0f ms", (time.perf_counter() - t) * 1e3)


def make_policy(args):
    if args.backend == "openvino":
        from dlsj.deploy.pi05_runtime import Pi05Policy

        return Pi05Policy(args.export, args.device, cache_dir=args.cache_dir)
    from dlsj.deploy.pi05_torch import TorchPi05Policy

    return TorchPi05Policy(args.ckpt, device=args.device)


def main() -> None:
    from dlsj import arms, teleop_web
    from dlsj.realsense_stream import RealSenseStream

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["openvino", "torch"], default="openvino")
    p.add_argument("--export", help="OpenVINO package (openvino backend)")
    p.add_argument("--ckpt", help="Studio .ckpt (torch backend, reference only)")
    p.add_argument("--device", default="GPU", help="OpenVINO device, or cpu/xpu for torch")
    p.add_argument("--cache-dir", default="~/desmond/ov_cache")
    p.add_argument("--follower-port", default="auto")
    p.add_argument("--follower-id", default="follower")
    p.add_argument("--cameras", default="", help="name=device for V4L cameras, e.g. wrist=/dev/v4l/by-id/...")
    p.add_argument("--realsense", default="", help="name=serial,..., e.g. top=346422061366,inspect=243722062690")
    p.add_argument("--bowls", default="yellow,purple,orange")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-step", type=float, default=DEFAULT_MAX_STEP,
                   help="joint target change per tick (calibrated units); demos move p99 5-8/tick. +/- keys tune it")
    p.add_argument("--max-step-gripper", type=float, default=DEFAULT_MAX_STEP)
    p.add_argument("--blend", choices=BLENDS, default="ensemble", help="how consecutive chunks are combined")
    p.add_argument("--ensemble-k", type=float, default=0.01, help="ensemble weights exp(-k*i), i=0 the oldest chunk")
    p.add_argument("--blend-steps", type=int, default=10, help="crossfade length in ticks")
    p.add_argument("--max-relative-target", type=float, default=10.0, help="LeRobot's per-command safety clamp")
    p.add_argument("--replan", type=int, default=50,
                   help="request a new chunk when this few future actions remain (50 = back to back)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8010)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    policy = make_policy(args)  # compile first: slow, and the arm should not wait torqued meanwhile
    warmup(policy)
    streams: dict = {}
    for name, dev in teleop_web.parse_cameras(args.cameras, 8).items() if args.cameras else []:
        streams[name] = teleop_web.CameraStream(name, dev, 640, 480, 30, 80).start()
    for item in filter(None, args.realsense.split(",")):
        name, serial = item.split("=")
        streams[name] = RealSenseStream(name, serial, depth=False).start()
    if missing := set(CAMS) - set(streams):
        raise SystemExit(f"missing cameras: {sorted(missing)}")
    port = args.follower_port
    if port == "auto":
        port, _, _ = arms.find_ports()
    arm = FollowerArm(port, args.follower_id, args.max_relative_target)
    arm.connect()
    ctl = Controller(arm, streams, policy, args.bowls.split(","), args.fps, args.max_step, args.max_step_gripper,
                     args.replan, args.blend, args.ensemble_k, args.blend_steps)
    loop = threading.Thread(target=ctl.run, name="control", daemon=True)
    loop.start()
    teleop_web.install_stop_signals()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(streams, ctl))
    server.daemon_threads = True
    log.info("policy console: http://localhost:%d  (%s on %s)", args.port, args.backend, args.device)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ctl.command("q")
        time.sleep(0.2)
        ctl.stop()
        loop.join(2)
        try:
            arm.go_home(args.fps)
        finally:
            arm.disconnect()
            for s in streams.values():
                s.stop()


if __name__ == "__main__":
    main()
