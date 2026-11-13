"""Record teleop demos of pick → inspect → place into a LeRobot v3 dataset, from a browser.

    uv run python -m dlsj.teleop_record --cameras top=2,wrist=0,inspect=4 \
        --repo-id local/duplo_teleop --bowls yellow,purple,orange --operator alice --target-s 8

Open http://localhost:8000.

One take is one continuous demo:
1. Pick the brick and hold it in front of the inspection camera.
2. When it is fully in view (detected automatically, with a chime), pause for about half a second, as
   the deployed system will while it inspects.
3. Place the brick in the bowl of your choice. Choose the bowl with 1/2/3 at any point before saving.

Every frame stores a task string:
* frames before the in-view moment get `--pick-task`;
* frames after it get `--place-task` with the chosen colour.

LeRobot keeps a task per frame, so one dataset trains a language-conditioned policy for both stages.
`meta/dlsj/labels.jsonl` records the split frame and bowl for each episode.

The policy never sees good or defective. At deployment the controller turns the inspection verdict into a
bowl colour and prompts "place the brick in the <colour> bowl".

Keys while the page is focused:
  space     start, stop, save        1 2 3   choose bowl         i   mark in view by hand
  f         save as failed           h       discard             e   engage or hold the arm
  d         show what the in-view detector sees
"""

import argparse
import contextlib
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
from dlsj.inspect_detect import BrickInViewDetector, DetectorConfig
from dlsj.kinesthetic.session import EpisodeStats

log = logging.getLogger("teleop_record")

PICK_TASK = "pick up the brick and hold it in front of the inspection camera"
PLACE_TASK = "place the brick in the {colour} bowl"


class EpisodeRecorder:
    """Samples `teleop.latest` and the camera streams at `fps` into a LeRobot dataset.

    Phases: idle → recording → review → (saving) → idle. Within a take, the stage goes pick → place
    when the inspection camera sees the held brick (or the operator presses i).
    """

    def __init__(self, teleop, streams: dict, repo_id: str, fps: int = 30, root: str | Path | None = None,
                 episode_time_s: float = 60.0, operator: str = "", vcodec: str = "h264", resume: bool = False,
                 bowls: list[str] | None = None, pick_task: str = PICK_TASK, place_task: str = PLACE_TASK,
                 inspect_camera: str | None = "inspect", detector: BrickInViewDetector | None = None,
                 record_depth: bool = True):
        self.teleop, self.streams = teleop, streams
        self.repo_id, self.fps = repo_id, fps
        self.root = Path(root) if root else Path("data") / repo_id.split("/")[-1]
        self.episode_time_s, self.operator, self.vcodec, self.resume = episode_time_s, operator, vcodec, resume
        self.bowls = bowls or ["yellow", "purple", "orange"]
        self.pick_task, self.place_task = pick_task, place_task
        if "{colour}" not in place_task:
            raise ValueError("--place-task must contain {colour}")
        self.inspect_camera = inspect_camera if inspect_camera in streams else None
        self.detector = detector or BrickInViewDetector()
        # RealSense streams carry aligned depth; it is recorded as observation.images.<cam>_depth (uint16 mm)
        self.depth_cams = [n for n, s in streams.items() if record_depth and getattr(s, "use_depth", False)]
        self.depth_shapes: dict[str, tuple[int, int, int]] = {}
        self.joints = list(teleop.joints)
        self.phase = "idle"
        self.message = ""
        self.dataset = None
        self.cam_shapes: dict[str, tuple[int, int, int]] = {}
        self.stats = EpisodeStats()
        self.commands: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.saved = {"success": 0, "failure": 0}
        self.bowl: str | None = None
        self._reset_take()

    def _reset_take(self) -> None:
        self._take_start = 0.0
        self._take_duration = 0.0
        self._frames = 0
        self._paused_ticks = 0
        self.stage = "pick"
        self.split_frame: int | None = None
        self.split_auto = False

    # ------------------------------------------------------------------ dataset
    def _wait_for_cameras(self, timeout_s: float = 20.0) -> None:
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            frames = {n: s.frame for n, s in self.streams.items()}
            depths = {n: self.streams[n].depth for n in self.depth_cams}
            if all(f is not None for f in frames.values()) and all(d is not None for d in depths.values()):
                self.cam_shapes = {n: f.shape for n, f in frames.items()}
                self.depth_shapes = {f"{n}_depth": (d.shape[0], d.shape[1], 1) for n, d in depths.items()}
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
            **hw_to_dataset_features({**joints, **self.cam_shapes, **self.depth_shapes}, OBS_STR, use_video=True),
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

    def _current_task(self) -> str:
        if self.stage == "pick":
            return self.pick_task
        return self.place_task.format(colour=self.bowl or "?")

    def _split(self, auto: bool) -> None:
        self.split_frame, self.split_auto, self.stage = self._frames, auto, "place"
        self.message = ""  # the stage track and the chime show it

    def _handle(self, c: str) -> None:
        if c == "space":
            c = {"idle": "start", "recording": "stop", "review": "accept"}.get(self.phase, "")
        if c.startswith("bowl") and c[4:].isdigit():
            i = int(c[4:]) - 1
            if 0 <= i < len(self.bowls) and self.phase != "saving":
                self.bowl = self.bowls[i]
            return
        if c == "inview" and self.phase == "recording" and self.stage == "pick":
            self._split(auto=False)
        elif c == "start" and self.phase == "idle":
            if not self._teleop_live():
                self.message = "Engage the arm first"
                return
            self._reset_take()
            self.detector.reset()
            self._take_start = time.perf_counter()
            self.phase, self.message = "recording", ""
        elif c == "stop" and self.phase == "recording":
            self._to_review()
        elif c in ("accept", "fail") and self.phase in ("recording", "review"):
            if self.phase == "recording":
                self._to_review()
            if c == "accept" and self.split_frame is None:
                self.message = "Mark the in-view moment first (i), or save as failed (f)"
                return
            if c == "accept" and self.bowl is None:
                self.message = "Choose a bowl (1–3)"
                return
            self._save(success=c == "accept")
        elif c == "discard" and self.phase in ("recording", "review"):
            self.dataset.clear_episode_buffer()
            self.phase, self.message = "idle", "Discarded"
            self._reset_take()

    def _to_review(self) -> None:
        self._take_duration = time.perf_counter() - self._take_start
        self.phase = "review"

    def _save(self, success: bool) -> None:
        if self._frames == 0:
            self.dataset.clear_episode_buffer()
            self.phase, self.message = "idle", "Nothing recorded"
            self._reset_take()
            return
        self.phase, self.message = "saving", "Saving"
        place = self.place_task.format(colour=self.bowl) if self.bowl else None
        if self.split_frame is not None and place:
            tasks = self.dataset.writer.episode_buffer["task"]
            for i in range(self.split_frame, len(tasks)):
                tasks[i] = place  # frames recorded before the bowl was chosen get the final instruction
        ep = self.dataset.num_episodes
        self.dataset.save_episode()
        label = {
            "episode_index": ep, "success": success, "duration_s": round(self._take_duration, 3),
            "n_frames": self._frames, "paused_ticks": self._paused_ticks, "method": "teleop",
            "inspection_frame": self.split_frame, "inspection_auto": self.split_auto, "bowl": self.bowl,
            "pick_task": self.pick_task, "place_task": place, "operator": self.operator,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.sidecar / "labels.jsonl", "a") as fh:
            fh.write(json.dumps(label) + "\n")
        self.saved["success" if success else "failure"] += 1
        if success:
            self.stats.add(self._take_duration)
        self.phase = "idle"
        self.message = f"Saved episode {ep}" if success else f"Saved episode {ep} as failed"
        self.bowl = None  # choose again for the next take, so a stale choice can't mislabel it
        self._reset_take()

    # ------------------------------------------------------------------ sampling
    def _frame(self) -> dict | None:
        latest = getattr(self.teleop, "latest", None)
        if latest is None or latest.get("phase") != "live":
            return None
        frame = {
            "observation.state": np.asarray(latest["follower"], dtype=np.float32),
            "action": np.asarray(latest["cmd"], dtype=np.float32),
            "task": self._current_task(),
        }
        for name, stream in self.streams.items():
            lock = getattr(stream, "cond", None) or contextlib.nullcontext()
            with lock:  # colour and depth from the same RealSense frameset
                img, depth = stream.frame, (stream.depth if name in self.depth_cams else None)
            if img is None or (name in self.depth_cams and depth is None):
                return None
            h, w, _ = self.cam_shapes[name]
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            frame[f"observation.images.{name}"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if depth is not None:
                dh, dw, _ = self.depth_shapes[f"{name}_depth"]
                if depth.shape[:2] != (dh, dw):
                    depth = cv2.resize(depth, (dw, dh), interpolation=cv2.INTER_NEAREST)
                frame[f"observation.images.{name}_depth"] = np.ascontiguousarray(depth, dtype=np.uint16).reshape(dh, dw, 1)
        return frame

    def _detect(self) -> None:
        if self.inspect_camera is None:
            return
        img = self.streams[self.inspect_camera].frame
        if img is not None:
            self.detector.update(img)

    def tick(self) -> None:
        """One recorder step: apply queued commands, run the in-view detector, record a frame if a take is running."""
        while not self.commands.empty():
            with self.lock:
                self._handle(self.commands.get_nowait())
        with self.lock:
            self._detect()
            if self.phase != "recording":
                return
            if self.stage == "pick" and self.detector.last.in_view:
                self._split(auto=True)
            frame = self._frame()
            if frame is None:
                self._paused_ticks += 1  # follower not live (held / re-engaging) or a camera dropped out
            else:
                self.dataset.add_frame(frame)
                self._frames += 1
            if time.perf_counter() - self._take_start >= self.episode_time_s:
                self._to_review()
                self.message = "Time limit"

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
                    self.phase, self.message = "idle", f"Take discarded after an error: {e}"
                    self._reset_take()
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)

    def close(self) -> None:
        with self.lock:
            if self.phase in ("recording", "review") and self.dataset is not None:
                self.dataset.clear_episode_buffer()
            if self.dataset is not None:
                self.dataset.finalize()

    def inspect_debug_jpeg(self) -> bytes | None:
        if self.inspect_camera is None:
            return None
        img = self.streams[self.inspect_camera].frame
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", self.detector.debug_image(img), [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = time.perf_counter() - self._take_start if self.phase == "recording" else self._take_duration
            d = self.detector.last
            return {
                "phase": self.phase, "stage": self.stage, "message": self.message, "elapsed": round(elapsed, 2),
                "frames": self._frames, "episodes": self.dataset.num_episodes if self.dataset is not None else 0,
                "saved": dict(self.saved), "median_s": _median(self.stats.durations_s),
                "bowls": self.bowls, "bowl": self.bowl, "split_frame": self.split_frame,
                "inspect_camera": self.inspect_camera,
                "in_view": d.in_view, "candidate": d.candidate, "detect_reason": d.reason,
                "prob": round(float(getattr(self.detector, "prob", d.area)), 3),
                "threshold": getattr(self.detector, "threshold", None),
                "guide": "square" if hasattr(self.detector, "prob") else list(self.detector.cfg.roi),
                "depth_cams": self.depth_cams,
                "teleop_live": self._teleop_live(), "root": str(self.root),
                "task": self._current_task() if self.phase == "recording" else None,
            }


def make_detector(args):
    """The trained classifier when available (or asked for), else the colour-blob detector."""
    from dlsj.inview_classifier import DEFAULT_MODEL, InViewClassifier

    model = None if args.inspect_model == "colour" else (
        DEFAULT_MODEL if args.inspect_model == "auto" else Path(args.inspect_model))
    if model is not None and Path(model).exists():
        log.info("in-view: classifier %s on %s", model, args.inspect_device)
        return InViewClassifier(model, args.inspect_device, args.inspect_threshold, args.inspect_stable_frames or 3)
    if args.inspect_model not in ("auto", "colour"):
        raise SystemExit(f"--inspect-model {args.inspect_model} not found")
    log.info("in-view: colour-blob detector")
    return BrickInViewDetector(DetectorConfig(
        roi=DetectorConfig.parse_roi(args.inspect_roi), min_area=args.inspect_min_area,
        min_sat=args.inspect_min_sat, stable_frames=args.inspect_stable_frames or 5))


def _median(xs: list[float]) -> float | None:
    return round(float(np.median(xs)), 1) if xs else None


# ----------------------------------------------------------------------------- web
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Record</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@200;300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#15171a;--panel:#1d2024;--line:#2b2f35;--fg:#f1f2f4;--mute:#80868f;--dim:#4a4f57;--red:#e31937;--glow:#e9f2ff}
*{box-sizing:border-box;margin:0}
html,body{height:100%}
body{background:var(--bg);color:var(--fg);font:400 15px/1.35 Barlow,ui-sans-serif,system-ui,sans-serif;-webkit-font-smoothing:antialiased;overflow:hidden}
.app{display:grid;grid-template-columns:minmax(0,1fr) 380px;grid-template-rows:minmax(0,1fr) auto;height:100vh}
.view{padding:20px 0 12px 20px;display:flex;flex-direction:column;gap:12px;min-height:0}
.main{position:relative;flex:1;min-height:0;border-radius:18px;overflow:hidden;background:#000}
.main img{width:100%;height:100%;object-fit:contain;display:block}
.main .name{position:absolute;left:16px;bottom:12px;color:var(--mute);font-size:13px}
.thumbs{display:flex;gap:12px;height:112px}
.thumb{position:relative;aspect-ratio:4/3;height:100%;border-radius:12px;overflow:hidden;background:#000;cursor:pointer;border:0;padding:0;outline:2px solid transparent;transition:outline-color .25s,box-shadow .25s}
.thumb img{width:100%;height:100%;object-fit:cover;display:block;opacity:.85}
.thumb span{position:absolute;left:8px;bottom:5px;font-size:12px;color:var(--fg);text-shadow:0 1px 2px #000}
.thumb.active{outline-color:var(--dim)}
.thumb.seen{outline-color:var(--glow);box-shadow:0 0 22px rgba(233,242,255,.35)}
.thumb:focus-visible{outline-color:var(--fg)}
.cluster{padding:28px 28px 12px;display:flex;flex-direction:column;gap:30px;min-height:0}
.state{display:flex;justify-content:space-between;align-items:baseline;color:var(--mute);font-size:15px}
.state b{font-weight:500;color:var(--fg)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--dim);margin-right:8px;vertical-align:1px;transition:background .3s}
.live .dot{background:var(--fg)}
.gauge{position:relative;width:300px;height:300px;align-self:center}
.gauge svg{position:absolute;inset:0;transform:rotate(-90deg)}
.gauge circle{fill:none;stroke-width:3}
.gauge .ring{stroke:var(--line)}
.gauge .arc{stroke:var(--fg);stroke-linecap:round;transition:stroke .3s}
.rec .gauge .arc{stroke:var(--red)}
.rec.over .time .n{color:var(--red)}
.time{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.time .n{font-weight:200;font-size:104px;line-height:1;letter-spacing:-.04em;font-variant-numeric:tabular-nums}
.time .u{color:var(--mute);font-size:15px;margin-top:6px;transition:color .3s}
.rec .time .u{color:var(--red)}
.stages{display:grid;grid-template-columns:auto 1fr auto 1fr auto;align-items:center;gap:10px;font-size:14px;color:var(--dim)}
.node{display:flex;align-items:center;gap:8px;transition:color .3s}
.node i{width:10px;height:10px;border-radius:50%;border:1.5px solid var(--dim);transition:background .3s,border-color .3s,box-shadow .3s}
.node.on{color:var(--fg)}.node.on i{background:var(--fg);border-color:var(--fg)}
.node.now i{box-shadow:0 0 0 5px rgba(241,242,244,.12)}
.rec .node.now i{background:var(--red);border-color:var(--red);box-shadow:0 0 0 5px rgba(227,25,55,.18)}
.seg{height:1.5px;background:var(--line);transition:background .3s}.seg.on{background:var(--fg)}
.bowls{display:flex;gap:18px;justify-content:center}
.bowl{width:64px;height:64px;border-radius:50%;border:0;cursor:pointer;position:relative;background:var(--c);opacity:.35;transform:scale(.88);transition:opacity .2s,transform .2s,box-shadow .2s}
.bowl.sel{opacity:1;transform:scale(1);box-shadow:0 0 0 3px var(--bg),0 0 0 5px var(--fg)}
.bowl kbd{position:absolute;right:-6px;bottom:-6px;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:1px 6px;font:500 12px Barlow,sans-serif}
.bowl:focus-visible{box-shadow:0 0 0 3px var(--bg),0 0 0 5px var(--glow)}
.msg{min-height:22px;text-align:center;color:var(--mute);font-size:15px;transition:color .2s}
.cue .msg{color:var(--fg);font-size:20px;font-weight:500}
.guide{position:absolute;border:1.5px dashed rgba(241,242,244,.55);border-radius:10px;pointer-events:none;display:none;transition:border-color .15s,box-shadow .15s}
.guide.show{display:block}.guide.near{border-color:var(--fg);border-style:solid}
.guide.hit{border:2.5px solid var(--glow);box-shadow:0 0 24px rgba(233,242,255,.45),inset 0 0 24px rgba(233,242,255,.25)}
.meter{display:grid;grid-template-columns:auto 1fr;align-items:center;gap:12px;color:var(--mute);font-size:13px}
.mbar{position:relative;height:6px;border-radius:4px;background:var(--line)}
.mbar i{position:absolute;inset:0 auto 0 0;width:0;border-radius:4px;background:var(--mute);transition:width .1s}
.mbar b{position:absolute;top:-5px;bottom:-5px;width:1.5px;background:var(--fg);opacity:.6}
.inview .mbar i{background:var(--glow)}
.tally{margin-top:auto;display:flex;justify-content:space-between;color:var(--mute);font-size:14px}
.tally b{display:block;color:var(--fg);font-weight:300;font-size:30px;font-variant-numeric:tabular-nums}
.keys{grid-column:1/-1;display:flex;gap:26px;padding:12px 20px 16px;color:var(--mute);font-size:13px;border-top:1px solid var(--line)}
.keys kbd{color:var(--fg);font:500 12px Barlow,sans-serif;border:1px solid var(--line);border-radius:6px;padding:1px 7px;margin-right:6px}
.keys .grow{flex:1}
@media (max-width:900px){body{overflow:auto}.app{grid-template-columns:minmax(0,1fr);height:auto}.view{padding:16px;min-width:0}.main{flex:none;height:62vw}.thumbs{height:84px;overflow-x:auto}.thumb{flex:none}.cluster{padding:8px 16px 16px;gap:24px}.gauge{width:240px;height:240px}.time .n{font-size:80px}.keys{flex-wrap:wrap;gap:12px 18px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head>
<body><div class="app" id="app">
<section class="view"><div class="main" id="mainbox"><img id="mainimg" alt=""><div class="guide" id="mguide"></div><div class="name" id="mainname"></div></div><div class="thumbs" id="thumbs"></div></section>
<aside class="cluster">
 <div class="state" id="state"><span><span class="dot"></span><b id="arm">Connecting</b></span><span id="eps"></span></div>
 <div class="gauge"><svg viewBox="0 0 300 300"><circle class="ring" cx="150" cy="150" r="140"/><circle class="arc" id="arc" cx="150" cy="150" r="140" stroke-dasharray="0 880"/></svg>
  <div class="time"><span class="n" id="t">0.0</span><span class="u" id="unit">Ready</span></div></div>
 <div class="stages"><span class="node"><i></i>Pick</span><span class="seg" id="s1"></span><span class="node"><i></i>In view</span><span class="seg" id="s2"></span><span class="node"><i></i>Place</span></div>
 <div class="bowls" id="bowls"></div>
 <div class="meter" id="meter"><span>Inspection</span><div class="mbar"><i id="mfill"></i><b id="mthr"></b></div></div>
 <div class="msg" id="msg"></div>
 <div class="tally"><span><b id="ok">0</b>Saved</span><span><b id="bad">0</b>Failed</span><span><b id="med">–</b>Median time</span></div>
</aside>
<footer class="keys"><span><kbd>Space</kbd><span id="kspace">Start</span></span><span><kbd>1–3</kbd>Bowl</span><span><kbd>I</kbd>In view</span><span><kbd>F</kbd>Failed</span><span><kbd>H</kbd>Discard</span><span class="grow"></span><span><kbd>E</kbd><span id="keng">Engage</span></span><span><kbd>D</kbd>Detector</span></footer>
</div>
<script>
const TARGET=__TARGET__,C=2*Math.PI*140,SW={red:'#e0393e',blue:'#2f6fe0',green:'#2fb36b',yellow:'#f5c518',orange:'#f2711c',white:'#e8e8e8',black:'#2a2a2a',purple:'#8e44d9',pink:'#e86aa6',grey:'#8d939b',gray:'#8d939b'};
let main=null,last={},debug=false,teleLive=false,ac=null;const thumbs={};
const $=id=>document.getElementById(id),app=$('app');
function tone(seq){try{ac=ac||new(window.AudioContext||window.webkitAudioContext)();let t=ac.currentTime;for(const[f,d]of seq){if(f){const o=ac.createOscillator(),g=ac.createGain();o.type='sine';o.frequency.value=f;o.connect(g);g.connect(ac.destination);g.gain.setValueAtTime(1e-4,t);g.gain.exponentialRampToValueAtTime(.25,t+.012);g.gain.exponentialRampToValueAtTime(1e-4,t+d);o.start(t);o.stop(t+d+.02)}t+=d}}catch(e){}}
const BEEP={start:[[880,.08],[0,.04],[1320,.12]],stop:[[660,.1],[0,.03],[330,.22]],view:[[1568,.18]]};
const rec=c=>fetch('/rec/'+c,{method:'POST'}),tele=c=>fetch('/cmd/'+c,{method:'POST'});
function setMain(n){main=n;if(!(debug&&n===last.inspect_camera))$('mainimg').src='/stream/'+encodeURIComponent(n);$('mainname').textContent=n;for(const k in thumbs)thumbs[k].classList.toggle('active',k===n)}
setInterval(()=>{if(debug&&main&&main===last.inspect_camera)$('mainimg').src='/rec/inspect.jpg?'+Date.now()},120);
document.addEventListener('keydown',e=>{if(e.repeat||e.metaKey||e.ctrlKey)return;const k=e.key.toLowerCase();
 if(k===' '){e.preventDefault();rec('space')}else if(k.length===1&&'123'.includes(k))rec('bowl'+k);else if(k==='i')rec('inview');
 else if(k==='f')rec('fail');else if(k==='h'||k==='backspace'){e.preventDefault();rec('discard')}else if(k==='e')tele(teleLive?'hold':'engage');
 else if(k==='d'){debug=!debug;if(last.inspect_camera)setMain(last.inspect_camera)}});
function bowls(r){const b=$('bowls');if(b.dataset.k!==r.bowls.join()){b.dataset.k=r.bowls.join();b.innerHTML='';
 r.bowls.forEach((c,i)=>{const x=document.createElement('button');x.className='bowl';x.style.setProperty('--c',SW[c]||c);x.setAttribute('aria-label',c+' bowl');x.innerHTML='<kbd>'+(i+1)+'</kbd>';x.onclick=()=>rec('bowl'+(i+1));b.appendChild(x)})}
 [...b.children].forEach((x,i)=>x.classList.toggle('sel',r.bowl===r.bowls[i]))}
function fracs(r,img){if(Array.isArray(r.guide))return r.guide;const w=img.naturalWidth||4,h=img.naturalHeight||3,s=Math.min(w,h);return[(w-s)/2/w,(h-s)/2/h,(w+s)/2/w,(h+s)/2/h]}
function place(g,box,img,f){const cw=box.clientWidth,ch=box.clientHeight,nw=img.naturalWidth||4,nh=img.naturalHeight||3;
 const k=g.dataset.cover?Math.max(cw/nw,ch/nh):Math.min(cw/nw,ch/nh),rw=nw*k,rh=nh*k,ox=(cw-rw)/2,oy=(ch-rh)/2;
 g.style.left=(ox+f[0]*rw)+'px';g.style.top=(oy+f[1]*rh)+'px';g.style.width=((f[2]-f[0])*rw)+'px';g.style.height=((f[3]-f[1])*rh)+'px'}
function guides(r){const ic=r.inspect_camera;const state=g=>{g.classList.toggle('hit',r.in_view);g.classList.toggle('near',r.candidate&&!r.in_view)};
 const mg=$('mguide'),mi=$('mainimg');mg.classList.toggle('show',!!ic&&main===ic&&!debug);
 if(ic&&main===ic){place(mg,$('mainbox'),mi,fracs(r,mi));state(mg)}
 const t=ic&&thumbs[ic];if(t){let g=t.querySelector('.guide');if(!g){g=document.createElement('div');g.className='guide show';g.dataset.cover=1;t.appendChild(g)}
  const im=t.querySelector('img');place(g,t,im,fracs(r,im));state(g)}}
async function tick(){try{
 const [s,r]=await Promise.all([fetch('/state').then(x=>x.json()),fetch('/rec/state').then(x=>x.json())]);
 for(const c of s.cameras){if(!thumbs[c.name]){const t=document.createElement('button');t.className='thumb';t.innerHTML='<img alt="" src="/stream/'+encodeURIComponent(c.name)+'"><span>'+c.name+'</span>';t.onclick=()=>setMain(c.name);$('thumbs').appendChild(t);thumbs[c.name]=t;if(!main)setMain(c.name)}}
 const t=s.teleop;teleLive=!!(t&&(t.phase==='live'||t.phase==='engaging'));
 $('arm').textContent=!t?'No arm':t.phase==='live'?'Live':t.phase==='engaging'?'Engaging':t.phase==='ready'?'Holding':t.phase[0].toUpperCase()+t.phase.slice(1);
 $('state').classList.toggle('live',!!t&&t.phase==='live');$('keng').textContent=teleLive?'Hold':'Engage';
 $('eps').textContent=r.episodes+' episodes';
 if(last.phase&&last.phase!==r.phase){if(r.phase==='recording')tone(BEEP.start);else if(last.phase==='recording')tone(BEEP.stop)}
 if(r.phase==='recording'&&last.stage==='pick'&&r.stage==='place')tone(BEEP.view);
 const recording=r.phase==='recording',over=TARGET&&r.elapsed>TARGET;
 app.classList.toggle('rec',recording);app.classList.toggle('over',!!over);
 $('t').textContent=r.elapsed.toFixed(1);
 $('unit').textContent=recording?'Recording':r.phase==='review'?'Review':r.phase==='saving'?'Saving':(r.teleop_live?'Ready':'Engage the arm');
 const frac=TARGET?Math.min(1,r.elapsed/TARGET):(recording?(r.elapsed%1):0);$('arc').setAttribute('stroke-dasharray',(frac*C)+' '+C);
 const active=recording||r.phase==='review',now=!active?-1:(r.stage==='place'?2:(r.in_view?1:0));
 document.querySelectorAll('.node').forEach((n,i)=>{n.classList.toggle('on',active&&i<=now);n.classList.toggle('now',active&&i===now)});
 $('s1').classList.toggle('on',active&&now>=1);$('s2').classList.toggle('on',active&&now>=2);
 if(r.inspect_camera&&thumbs[r.inspect_camera])thumbs[r.inspect_camera].classList.toggle('seen',r.in_view);
 bowls(r);guides(r);
 $('meter').style.display=(r.inspect_camera&&r.threshold)?'':'none';$('meter').classList.toggle('inview',r.in_view);
 $('mfill').style.width=Math.min(100,100*(r.prob||0))+'%';$('mthr').style.left=r.threshold?(100*r.threshold)+'%':'-9px';
 const cue=recording&&r.stage==='place';app.classList.toggle('cue',cue);
 $('msg').textContent=cue?(r.bowl?'Place in the '+r.bowl+' bowl':'In view. Choose a bowl (1\u20133)'):(r.message||(r.phase==='review'?'Space to save':(recording?'Centre the brick in the dashed square':'')));
 $('kspace').textContent={idle:'Start',recording:'Stop',review:'Save',saving:'Saving'}[r.phase]||'';
 $('ok').textContent=r.saved.success;$('bad').textContent=r.saved.failure;$('med').textContent=r.median_s===null?'–':r.median_s+'s';
 last=r;
}catch(e){$('arm').textContent='Disconnected'}}
setInterval(tick,100);tick();
</script></body></html>"""


def make_handler(streams: dict, teleop, recorder: EpisodeRecorder, target_s: float | None):
    base = teleop_web.make_handler(streams, teleop)
    page = PAGE.replace("__TARGET__", json.dumps(target_s)).encode()
    commands = {"space", "start", "stop", "accept", "fail", "discard", "inview"} | {f"bowl{i}" for i in range(1, 10)}

    class Handler(base):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/rec/state":
                return self._send(200, json.dumps(recorder.snapshot()).encode(), "application/json")
            if self.path.startswith("/rec/inspect.jpg"):
                jpeg = recorder.inspect_debug_jpeg()
                return self._send(200, jpeg, "image/jpeg") if jpeg else self._send(404, b"", "text/plain")
            return super().do_GET()

        def do_POST(self):
            name = self.path.removeprefix("/rec/")
            if self.path.startswith("/rec/") and name in commands:
                recorder.command(name)
                return self._send(200, b"{}", "application/json")
            return super().do_POST()

    return Handler


# ----------------------------------------------------------------------------- main
def main() -> None:
    from http.server import ThreadingHTTPServer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="dataset name, e.g. local/duplo_teleop")
    p.add_argument("--bowls", default="yellow,purple,orange", help="bowl colours in key order (1, 2, 3)")
    p.add_argument("--pick-task", default=PICK_TASK)
    p.add_argument("--place-task", default=PLACE_TASK, help="must contain {colour}")
    p.add_argument("--inspect-camera", default="inspect", help="camera name used for the in-view detector")
    p.add_argument("--inspect-model", default="auto",
                   help="OpenVINO brick-in-view classifier (.xml); 'auto' = models/brick_in_view_v1 if present, "
                        "'colour' = the colour-blob detector")
    p.add_argument("--inspect-device", default="CPU", help="OpenVINO device for the classifier: CPU, GPU or NPU")
    p.add_argument("--inspect-threshold", type=float, default=0.6, help="P(in view) needed per frame")
    p.add_argument("--inspect-roi", default="0.1,0.1,0.9,0.9", help="x0,y0,x1,y1 fractions of the inspect image")
    p.add_argument("--inspect-min-area", type=float, default=0.03, help="brick blob area as a fraction of the ROI")
    p.add_argument("--inspect-min-sat", type=int, default=100)
    p.add_argument("--inspect-stable-frames", type=int, default=None,
                   help="consecutive in-view frames needed (default 3 for the classifier, 5 for colour)")
    p.add_argument("--root", help="dataset directory (default data/<repo name>)")
    p.add_argument("--resume", action="store_true", help="append to an existing dataset")
    p.add_argument("--record-fps", type=int, default=30)
    p.add_argument("--episode-time", type=float, default=60.0, help="a take stops itself after this many seconds")
    p.add_argument("--target-s", type=float, help="target time per take, shown as the ring around the timer")
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
    p.add_argument("--cameras", default="", help="OpenCV cameras, name=index_or_path,... e.g. wrist=/dev/video38")
    p.add_argument("--realsense", default="",
                   help="RealSense cameras by serial, name=serial,... e.g. top=346422061366,inspect=243722062690")
    p.add_argument("--no-depth", action="store_true", help="don't record RealSense depth")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--jpeg-quality", type=int, default=75)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    teleop_web.install_stop_signals()
    args.no_robot = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if "auto" in (args.follower_port, args.leader_port):
        follower, leader, volts = arms.find_ports()
        log.info("arms found by supply voltage: %s", {k: v for k, v in volts.items() if v is not None})
        args.follower_port = follower if args.follower_port == "auto" else args.follower_port
        args.leader_port = leader if args.leader_port == "auto" else args.leader_port
        if not (args.follower_port and args.leader_port):
            p.error("could not find both arms; check USB cables and the follower's 12 V supply")

    from dlsj.realsense_stream import RealSenseStream

    cams = teleop_web.parse_cameras(args.cameras, 8) if args.cameras else {}
    rs_cams = dict(part.split("=", 1) for part in filter(None, (x.strip() for x in args.realsense.split(","))))
    if not cams and not rs_cams:
        p.error("give at least one camera with --cameras and/or --realsense")
    streams = {n: teleop_web.CameraStream(n, i, args.width, args.height, args.cam_fps, args.jpeg_quality).start()
               for n, i in cams.items()}
    streams |= {n: RealSenseStream(n, sn, args.width, args.height, args.cam_fps, depth=not args.no_depth,
                                   quality=args.jpeg_quality).start() for n, sn in rs_cams.items()}
    if args.inspect_camera not in streams:
        log.warning("no camera named %r: the in-view detector is off, mark the moment with i", args.inspect_camera)
    detector = make_detector(args)
    teleop = teleop_web.Teleop(args)
    recorder = EpisodeRecorder(teleop, streams, args.repo_id, args.record_fps, args.root, args.episode_time,
                               args.operator, args.vcodec, args.resume, [b.strip() for b in args.bowls.split(",")],
                               args.pick_task, args.place_task, args.inspect_camera, detector,
                               record_depth=not args.no_depth)
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
        log.info("record UI: http://localhost:%d  (cameras: %s; depth: %s)", args.port, ", ".join(streams),
                 ", ".join(recorder.depth_cams) or "none")
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
