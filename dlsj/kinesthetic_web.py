"""Kinesthetic-teaching recorder in the browser: guide the follower by hand, record a LeRobot v3 dataset.

This is the same Flash ALOHA-style workflow as `dlsj.kinesthetic.record`. It uses the same follower
calibration (the repo's `calibration/`), port auto-detection and camera streams as `dlsj.teleop_web` /
`dlsj.teleop_record`, so kinesthetic and teleop datasets have identical features and units.

    uv run python -m dlsj.kinesthetic_web --cameras wrist=0,top=2,inspect=4 \
        --repo-id local/duplo_kinesthetic --task "Pick a brick and present it to the inspection camera" \
        --operator alice --target-s 6

Open http://localhost:8001. Keys while the page is focused (a foot pedal that types space/h works too):
  SPACE   start a take (torque released, rising beep) / stop (arm held, falling beep) / accept
  h       reject the take, or cancel a replay. Between takes: hold or release the arm
  f       save as FAILURE           n: save as SUCCESS now
  r       replay the stopped take on the robot (the overlay shows the first frame over the live camera)
  g       toggle the gripper        up / down: nudge the gripper
  p       save the current pose as the start pose
Quit with Ctrl+C in the terminal: an unsaved take is discarded, and the arm is released after 3 seconds.
"""

import argparse
import json
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from dlsj import arms, teleop_web

log = logging.getLogger("kinesthetic_web")

KEY_NAMES = {"space", "enter", "up", "down", "left", "right", "backspace", "h", "x", "n", "f", "r", "g", "tab", "p", "1",
             "s"}


class _StreamCamera:
    """`read_latest()` in RGB at a fixed size, like a LeRobot camera, backed by a teleop_web CameraStream."""

    def __init__(self, stream, shape: tuple[int, int, int]):
        self.stream, self.shape = stream, shape

    def read_latest(self) -> np.ndarray | None:
        img = self.stream.frame
        if img is None:
            return None
        h, w, _ = self.shape
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class FollowerWithStreams:
    """The follower arm plus teleop_web camera streams, presented as one LeRobot-style robot.

    `get_observation()` returns the joint positions plus one RGB frame per camera, under the camera's name.
    """

    def __init__(self, robot, streams: dict, calibrate: bool = False, wait_s: float = 20.0):
        self._robot, self.streams, self._calibrate = robot, streams, calibrate
        self.bus = robot.bus
        self.robot_type = getattr(robot, "robot_type", "so_follower")
        shapes = _wait_for_frames(streams, wait_s)
        self.cameras = {n: _StreamCamera(s, shapes[n]) for n, s in streams.items()}
        self._shapes = shapes

    @property
    def action_features(self) -> dict:
        return self._robot.action_features

    @property
    def observation_features(self) -> dict:
        return {**self._robot.action_features, **self._shapes}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self) -> None:
        arms.ensure_calibrated(self._robot, "follower", force=self._calibrate)

    def disconnect(self) -> None:
        self._robot.disconnect()

    def get_observation(self) -> dict:
        obs = self._robot.get_observation()  # joints only: the LeRobot robot has no cameras configured
        for name, cam in self.cameras.items():
            frame = cam.read_latest()
            if frame is None:
                raise RuntimeError(f"camera {name} has no frame")
            obs[name] = frame
        return obs


def _wait_for_frames(streams: dict, timeout_s: float) -> dict[str, tuple[int, int, int]]:
    import time

    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        if all(s.frame is not None for s in streams.values()):
            return {n: s.frame.shape for n, s in streams.items()}
        time.sleep(0.1)
    missing = [n for n, s in streams.items() if s.frame is None]
    raise RuntimeError(f"no frames from cameras {missing} after {timeout_s}s")


# ----------------------------------------------------------------------------- web
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SO-101 Kinesthetic</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e6e8ee;--mute:#8a90a0;--ok:#3ecf8e;--bad:#ff6b6b;--rec:#ff4d4d;--acc:#6aa9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 ui-sans-serif,system-ui,-apple-system}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid #262a33;position:sticky;top:0;background:var(--bg);z-index:2}
h1{font-size:15px;margin:0}.pill{padding:2px 8px;border-radius:99px;background:#262a33;color:var(--mute);font-variant-numeric:tabular-nums}
.pill.ok{color:var(--ok)}.pill.bad{color:var(--bad)}
main{display:grid;grid-template-columns:1fr 360px;gap:16px;padding:16px}@media(max-width:960px){main{grid-template-columns:1fr}}
#cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;align-content:start}
.cam{background:var(--card);border-radius:10px;overflow:hidden;border:3px solid transparent}.cam img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#000}
.cam .bar{padding:6px 10px;color:var(--mute)}.cam .bar b{color:var(--fg)}
body.recording .cam{border-color:var(--rec)}
#ov{display:none;background:var(--card);border-radius:10px;overflow:hidden;border:3px solid var(--acc)}#ov img{width:100%;display:block}
body.waiting #ov{display:block}
.side{background:var(--card);border-radius:10px;padding:14px;align-self:start;display:flex;flex-direction:column;gap:10px}
#big{font-size:44px;font-weight:700;font-variant-numeric:tabular-nums;text-align:center}
#phase{text-align:center;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
body.recording #phase{color:var(--rec)}body.review #phase{color:var(--acc)}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:8px}button{background:#262a33;color:var(--fg);border:1px solid #333845;border-radius:8px;padding:9px;cursor:pointer;font-size:13px}
button.primary{grid-column:1/-1;font-size:18px;padding:14px;background:#2b3240}body.recording button.primary{background:#5a1f1f}
kbd{background:#262a33;border-radius:4px;padding:0 5px;font-size:12px}.mute{color:var(--mute);font-size:12px}
#log{font-family:ui-monospace,monospace;font-size:12px;color:var(--mute);max-height:220px;overflow:auto;white-space:pre-wrap}
.bar2{height:6px;background:#262a33;border-radius:4px;overflow:hidden}.bar2>div{height:100%;width:0;background:var(--ok)}
</style></head><body>
<header><h1>SO-101 kinesthetic</h1><span id="arm" class="pill">arm…</span><span id="grip" class="pill"></span><span id="eps" class="pill"></span>
<span style="flex:1"></span><span id="root" class="mute"></span></header>
<main><section id="cams"><div id="ov" class="cam"><img id="ovimg"><div class="bar"><b>replay alignment</b>: reset the scene to match, then <kbd>space</kbd> (or <kbd>h</kbd> to cancel)</div></div></section>
<aside class="side">
<div id="phase">idle</div><div id="big">0.0s</div><div class="bar2"><div id="prog"></div></div>
<div class="btns"><button id="go" class="primary">Start <kbd>space</kbd></button>
<button data-k="f">Failure <kbd>f</kbd></button><button data-k="h">Reject / hold <kbd>h</kbd></button>
<button data-k="r">Replay <kbd>r</kbd></button><button data-k="g">Gripper <kbd>g</kbd></button>
<button data-k="up">Gripper open <kbd>↑</kbd></button><button data-k="down">Gripper close <kbd>↓</kbd></button>
<button data-k="p">Save start pose <kbd>p</kbd></button><button data-k="n">Save success <kbd>n</kbd></button></div>
<div id="task" class="mute"></div><div id="stats" class="mute"></div><div id="track" class="mute"></div>
<div id="log"></div>
<div class="mute">Hold the arm before pressing space: torque is released when recording starts. Beeps: rising = started, falling = stopped.</div>
</aside></main>
<script>
const TARGET=__TARGET__;let last=null,ovTimer=null;const seen={},cams=document.getElementById('cams');
const AC=window.AudioContext||window.webkitAudioContext;let ac=null;
function tone(seq){try{ac=ac||new AC();let t=ac.currentTime;for(const[f,d]of seq){if(f>0){const o=ac.createOscillator(),g=ac.createGain();
o.frequency.value=f;o.connect(g);g.connect(ac.destination);g.gain.setValueAtTime(0.0001,t);g.gain.exponentialRampToValueAtTime(0.3,t+0.01);
g.gain.exponentialRampToValueAtTime(0.0001,t+d);o.start(t);o.stop(t+d+0.02);}t+=d;}}catch(e){}}
const BEEP={start:[[880,.08],[0,.04],[1320,.12]],stop:[[660,.10],[0,.03],[330,.22]]};
const key=k=>fetch('/kin/key/'+k,{method:'POST'});
document.getElementById('go').onclick=()=>key('space');
document.querySelectorAll('button[data-k]').forEach(b=>b.onclick=()=>key(b.dataset.k));
const MAP={' ':'space','enter':'enter','arrowup':'up','arrowdown':'down','arrowleft':'left','arrowright':'right','backspace':'backspace','tab':'tab'};
document.addEventListener('keydown',e=>{if(e.repeat)return;const k=e.key.toLowerCase();const name=MAP[k]||(k.length===1?k:null);
 if(!name)return;if([' ','tab','backspace','arrowup','arrowdown'].includes(k))e.preventDefault();key(name);});
async function tick(){try{
 const s=await (await fetch('/state')).json(),r=await (await fetch('/kin/state')).json();
 for(const c of s.cameras){if(!seen[c.name]){const d=document.createElement('div');d.className='cam';
  d.innerHTML=`<img src="/stream/${encodeURIComponent(c.name)}"><div class="bar"><b>${c.name}</b> <span></span></div>`;cams.appendChild(d);seen[c.name]=d;}
  seen[c.name].querySelector('.bar span').textContent=c.error?c.error:`${c.size||'…'} · ${c.fps} fps`;}
 if(last&&last!==r.phase){if(r.phase==='recording')tone(BEEP.start);else if(last==='recording')tone(BEEP.stop);}
 last=r.phase;const waiting=r.waiting_for==='replay_go';
 document.body.className=r.phase+(waiting?' waiting':'');
 if(waiting&&!ovTimer){ovTimer=setInterval(()=>{document.getElementById('ovimg').src='/kin/overlay.jpg?t='+Date.now()},150);}
 if(!waiting&&ovTimer){clearInterval(ovTimer);ovTimer=null;}
 document.getElementById('phase').textContent=r.replaying?'replaying':(waiting?'reset scene for replay':r.phase);
 document.getElementById('big').textContent=r.elapsed.toFixed(1)+'s';
 const pr=document.getElementById('prog');pr.style.width=TARGET?Math.min(100,100*r.elapsed/TARGET)+'%':'0';pr.style.background=TARGET&&r.elapsed>TARGET?'var(--bad)':'var(--ok)';
 document.getElementById('go').innerHTML=({idle:'Start',recording:'Stop',review:'Accept'}[r.phase]||r.phase)+' <kbd>space</kbd>';
 const a=document.getElementById('arm');a.textContent=r.arm_held?'arm HELD':'arm FREE';a.className='pill '+(r.arm_held?'ok':'bad');
 document.getElementById('grip').textContent=`gripper ${r.gripper.mode==='keys'?(r.gripper.open?'open':'closed')+' ('+r.gripper.target+')':'by hand'}`;
 document.getElementById('eps').textContent=`${r.episodes} episodes`;
 document.getElementById('task').textContent='Task: '+r.task;document.getElementById('stats').textContent=r.stats+(TARGET?` · target ${TARGET}s`:'');
 document.getElementById('track').textContent=r.tracking?('last replay worst error: '+Object.entries(r.tracking.max_abs).filter(([k])=>k!=='gripper').map(([k,v])=>k.replace('_flex','').replace('shoulder_','sh_')+' '+v.toFixed(1)+'°').join(', ')):'';
 document.getElementById('log').textContent=r.messages.slice().reverse().join('\\n');
 document.getElementById('root').textContent=r.root;
}catch(e){document.getElementById('arm').textContent='server gone';}}
setInterval(tick,100);tick();
</script></body></html>"""


def make_handler(streams: dict, recorder, target_s: float | None):
    base = teleop_web.make_handler(streams, None)
    page = PAGE.replace("__TARGET__", json.dumps(target_s)).encode()

    class Handler(base):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/kin/state":
                return self._send(200, json.dumps(recorder.snapshot()).encode(), "application/json")
            if self.path.startswith("/kin/overlay.jpg"):
                img = recorder.overlay_rgb()
                if img is None:
                    return self._send(404, b"", "text/plain")
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
                return self._send(200, buf.tobytes(), "image/jpeg")
            return super().do_GET()

        def do_POST(self):
            name = self.path.removeprefix("/kin/key/")
            if self.path.startswith("/kin/key/") and name in KEY_NAMES:
                recorder.keys.put(name)
                return self._send(200, b"{}", "application/json")
            return self._send(404, b"", "text/plain")

    return Handler


# ----------------------------------------------------------------------------- main
def build_recorder(args, streams: dict):
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    from dlsj.kinesthetic.record import KinestheticRecordConfig, KinestheticRecorder

    robot_cfg = SO101FollowerConfig(port=args.follower_port, id=args.follower_id, calibration_dir=arms.FOLLOWER_CAL_DIR)
    root = Path(args.root) if args.root else Path("data") / args.repo_id.split("/")[-1]
    cfg = KinestheticRecordConfig(
        robot=robot_cfg,
        dataset=DatasetRecordConfig(
            repo_id=args.repo_id, single_task=args.task, root=root, fps=args.record_fps,
            episode_time_s=args.episode_time, video=True, push_to_hub=False, no_stamp=True,
            rgb_encoder=RGBEncoderConfig(vcodec=args.vcodec),
        ),
        gripper_mode=args.gripper_mode, gripper_open=args.gripper_open, gripper_closed=args.gripper_closed,
        action_lookahead=args.action_lookahead, target_s=args.target_s, go_to_start=args.go_to_start,
        overlay=False,  # the browser shows the alignment overlay instead of an OpenCV window
        overlay_camera=args.overlay_camera, operator=args.operator, terminal_keys=False, sounds=False,
        resume=args.resume or (root / "meta" / "info.json").exists(),
    )
    robot = FollowerWithStreams(make_robot_from_config(robot_cfg), streams, calibrate=args.calibrate)
    return KinestheticRecorder(cfg, robot=robot)


def main() -> None:
    from http.server import ThreadingHTTPServer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="dataset name, e.g. local/duplo_kinesthetic")
    p.add_argument("--task", required=True)
    p.add_argument("--root", help="dataset directory (default data/<repo name>); appended to if it exists")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--record-fps", type=int, default=30)
    p.add_argument("--episode-time", type=float, default=60.0)
    p.add_argument("--target-s", type=float)
    p.add_argument("--operator", default="")
    p.add_argument("--vcodec", default="h264")
    p.add_argument("--gripper-mode", choices=["keys", "free"], default="keys")
    p.add_argument("--gripper-open", type=float, default=50.0, help="0-100; read the gripper value in the UI")
    p.add_argument("--gripper-closed", type=float, default=0.0)
    p.add_argument("--action-lookahead", type=int, default=1)
    p.add_argument("--go-to-start", action="store_true")
    p.add_argument("--overlay-camera")
    p.add_argument("--follower-port", default="auto")
    p.add_argument("--follower-id", default="follower")
    p.add_argument("--calibrate", action="store_true", help="force a fresh follower calibration")
    p.add_argument("--cameras", required=True, help="name=index,... e.g. wrist=0,top=2,inspect=4")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--jpeg-quality", type=int, default=75)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.follower_port == "auto":
        follower, _leader, volts = arms.find_ports()
        log.info("arms found by supply voltage: %s", {k: v for k, v in volts.items() if v is not None})
        if not follower:
            p.error("no follower found (12 V supply); pass --follower-port")
        args.follower_port = follower

    cams = teleop_web.parse_cameras(args.cameras, 8)
    streams = {n: teleop_web.CameraStream(n, i, args.width, args.height, args.cam_fps, args.jpeg_quality).start()
               for n, i in cams.items()}
    server = None
    try:
        recorder = build_recorder(args, streams)
        server = ThreadingHTTPServer((args.host, args.port), make_handler(streams, recorder, args.target_s))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
        log.info("kinesthetic UI: http://localhost:%d  (cameras: %s)", args.port, ", ".join(cams))
        recorder.run()  # connects, opens the dataset, runs until Ctrl+C
    except KeyboardInterrupt:
        pass
    finally:
        for s in streams.values():
            s.stop()
        if server is not None:
            server.shutdown()


if __name__ == "__main__":
    main()
