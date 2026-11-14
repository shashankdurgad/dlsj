"""Live spin-and-inspect test page: YOLO-only plain / circle / cross, streamed to a browser.

    python -m dlsj.inspect.sweep_live --model-dir ~/inference/models/brick-yolo-classifier-v1 \\
        --realsense inspect=243722062690,top=346422061366 \\
        --cameras wrist=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0 \\
        --device NPU --port 8012       # optional --roi x0,y0,x1,y1; --no-arm = camera only; --arm auto = connect at start

Open http://<host>:8012. The page shows the `inspect` stream (crop box + the latest per-frame label and
probabilities drawn on it) with small `wrist` and `top` views, and:

* Classify live: continuous per-frame YOLO on the inspect stream, no motion (plus the release's 3-frame mean).
* Spin & inspect: `dlsj.loop.sweep.Sweeper` rolls the held brick on wrist_roll (N stops over a range, within the
  calibrated limits, settle, fresh frames), each frame classified on a background worker while the wrist moves on;
  then `dlsj.inspect.aggregate` (`yolo_topk` by default) decides plain / circle / cross / uncertain -> bowl.
  Saved to `outputs/sweeps/<time>_live/` (frame_XX.png, crop_XX.jpg, sweep.json, contact.jpg, scores.png).
* Dry run (checkbox): the same sweep against a simulated wrist (no command reaches the arm); real camera frames.
* Stop: aborts; the wrist then returns to its start angle (non-abortable, at the speed cap).
* Go to inspect pose: slow ramp (`--ramp-speed`, default 20 deg/s) to the saved `inspect` pose
  (`calibration/robots/so_follower/<id>.inspect.json`); the gripper command is kept.
* Connect arm: connects the follower with the repo calibration (goal := present position, torque on, holding).

Safety: nothing moves on page load or on connect; the arm only moves on Spin & inspect (not dry run) or Go to
inspect pose. The sweep starts from the servos' Goal_Position (not the measured pose, so a gripper squeezing a brick
keeps squeezing); speed is capped per tick (hard limit 120 deg/s); Stop works mid-sweep; the wrist is always sent
back to its start angle; exiting keeps torque on (holding).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import signal
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np

from dlsj.contracts import MarkType
from dlsj.inspect import aggregate as agg_mod
from dlsj.inspect.aggregate import ALL_STRATEGIES, AggregateConfig, YOLO_CLASSES
from dlsj.loop.poses import GRIPPER, JOINT_KEYS
from dlsj.loop.sweep import ROLL, SweepConfig, Sweeper

log = logging.getLogger("sweep_live")

BOWL = {MarkType.NONE: "yellow", MarkType.CIRCLE: "orange", MarkType.CROSS: "purple"}
CLASS_BGR = {"plain": (40, 210, 240), "circle": (40, 140, 255), "cross": (200, 80, 160), None: (200, 200, 200)}
MAX_SPEED_HARD = 120.0  # deg/s, whatever the page asks
DEFAULTS = {"frames": 12, "range_deg": 180.0, "settle_ms": 150.0, "speed": 60.0, "device": "NPU",
            "strategy": "yolo_topk", "dry_run": True, "grasp_check": True}


def bowl_for(outcome: MarkType) -> str:
    return BOWL.get(outcome, "re-inspect, then purple")


# ----------------------------------------------------------------------------- cameras
class Cams:
    """Camera streams (`dlsj.teleop_web.CameraStream` / `dlsj.realsense_stream.RealSenseStream`, BGR) with the
    `Cameras` protocol the sweep needs (`fresh` returns RGB) plus what the page needs."""

    def __init__(self, streams: dict):
        self.streams = streams
        self.names = list(streams)

    @classmethod
    def open(cls, cameras: str = "", realsense: str = "", width=640, height=480, fps=30, depth=False) -> "Cams":
        streams = {}
        if cameras:
            from dlsj.teleop_web import CameraStream, parse_cameras

            streams |= {n: CameraStream(n, i, width, height, fps, 80).start()
                        for n, i in parse_cameras(cameras, 8).items()}
        rs = dict(p.split("=", 1) for p in filter(None, (x.strip() for x in realsense.split(","))))
        if rs:
            from dlsj.realsense_stream import RealSenseStream

            streams |= {n: RealSenseStream(n, sn, width, height, fps, depth=depth, quality=80).start()
                        for n, sn in rs.items()}
        if "inspect" not in streams:
            raise SystemExit("need an `inspect` camera (--realsense inspect=<serial> or --cameras inspect=...)")
        return cls(streams)

    def get(self, name: str) -> tuple[int, np.ndarray | None]:
        s = self.streams[name]
        with s.cond:
            return s.seq, s.frame

    def wait_new(self, name: str, last: int, timeout: float = 2.0) -> tuple[int, np.ndarray | None]:
        s = self.streams[name]
        with s.cond:
            s.cond.wait_for(lambda: s.seq != last, timeout=timeout)
            return s.seq, s.frame

    def jpeg(self, name: str) -> bytes | None:
        s = self.streams.get(name)
        return s.jpeg if s is not None else None

    def latest(self) -> dict[str, np.ndarray]:
        out = {}
        for n in self.names:
            _, f = self.get(n)
            if f is not None:
                out[n] = np.ascontiguousarray(f[..., ::-1])
        return out

    def fresh(self, name: str, n: int, timeout_s: float = 2.0) -> list[np.ndarray]:
        seq, _ = self.get(name)
        out, t_end = [], time.monotonic() + timeout_s
        while len(out) < n:
            new, f = self.wait_new(name, seq, max(0.0, t_end - time.monotonic()))
            if new == seq or f is None:
                raise TimeoutError(f"camera {name}: only {len(out)}/{n} new frames in {timeout_s}s")
            seq = new
            out.append(np.ascontiguousarray(f[..., ::-1]))
        return out

    def info(self) -> list[dict]:
        return [s.info() for s in self.streams.values()]

    def stop(self) -> None:
        for s in self.streams.values():
            s.stop()


# ----------------------------------------------------------------------------- arm adapters
class SimArm:
    """Dry run: a simulated follower (first-order lag) starting at `start`. Nothing reaches the real arm."""

    def __init__(self, start: np.ndarray, lag: float = 0.5):
        self.state = np.asarray(start, np.float32).copy()
        self.cmd = self.state.copy()
        self.lag = lag
        self.sent = 0

    def read_state(self) -> np.ndarray:
        return self.state.copy()

    def read_gripper_current(self) -> float | None:
        return None

    def send(self, target: np.ndarray) -> None:
        self.cmd = np.asarray(target, np.float32).copy()
        self.state = self.state + self.lag * (self.cmd - self.state)
        self.sent += 1

    def goal(self) -> np.ndarray:
        return self.cmd.copy()


def serial_ports_in_use() -> dict[str, str]:
    """{port: pids} for /dev/ttyACM* / ttyUSB* held by another process (`fuser`; empty when fuser is missing)."""
    import glob
    import os
    import subprocess

    busy = {}
    for port in sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")):
        try:
            r = subprocess.run(["fuser", port], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return {}
        pids = [x for x in r.stdout.split() if x.strip().isdigit() and int(x) != os.getpid()]
        if pids:
            busy[port] = " ".join(pids)
    return busy


class RealArm:
    """`dlsj.loop.hardware.LeRobotArm` + `goal()` (Goal_Position in observation units) and torque kept on exit."""

    def __init__(self, port: str = "auto", follower_id: str = "follower"):
        from dlsj.loop.hardware import LeRobotArm

        busy = serial_ports_in_use()
        if busy:
            raise RuntimeError(f"serial port(s) held by another process, not connecting: {busy}")

        self.arm = LeRobotArm(port, follower_id)
        try:  # never drop the arm: LeRobot disables torque on disconnect by default
            self.arm.robot.config.disable_torque_on_disconnect = False
        except Exception as e:  # noqa: BLE001
            log.warning("could not keep torque on at disconnect: %s", e)
        self.arm.connect()  # repo calibration; goal := present (no lunge), torque on, holding
        self.follower_id = follower_id
        self.port = self.arm.robot.config.port

    def read_state(self) -> np.ndarray:
        return self.arm.read_state()

    def send(self, target: np.ndarray) -> None:
        self.arm.send(target)

    def read_gripper_current(self) -> float | None:
        return self.arm.read_gripper_current()

    def goal(self) -> np.ndarray:
        """The commanded pose (servo Goal_Position), so a gripper squeezing a brick keeps squeezing."""
        try:
            g = self.arm.robot.bus.sync_read("Goal_Position", num_retry=3)
            return np.array([float(g[k.removesuffix(".pos")]) for k in JOINT_KEYS], np.float32)
        except Exception as e:  # noqa: BLE001
            log.warning("Goal_Position read failed (%s); using the measured pose", e)
            return self.read_state()

    def disconnect(self) -> None:
        self.arm.disconnect()


# ----------------------------------------------------------------------------- overlay
def draw_overlay(bgr: np.ndarray, roi, live: dict | None, label_note: str = "") -> np.ndarray:
    img = bgr.copy()
    if roi is None:
        from dlsj.inspect.yolo_inspector import default_roi

        roi = default_roi(img.shape[1], img.shape[0])
    if live is not None and live.get("box"):
        box, conf_cls = live["box"], live.get("confident")
    else:
        box, conf_cls = roi, None
    col = CLASS_BGR.get(conf_cls, CLASS_BGR[None])
    if box:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), col, 2)
    lines = []
    if live is not None:
        p = live["probs"]
        head = live["confident"] or f"unsure ({live['top']})"
        lines.append(f"{head.upper()}  {live['conf']:.2f}  m{live['margin']:.2f}  {live['ms']:.0f}ms {live['device']}")
        lines.append("  ".join(f"{c} {p[c]:.2f}" for c in YOLO_CLASSES))
    if label_note:
        lines.append(label_note)
    if lines:
        cv2.rectangle(img, (0, 0), (img.shape[1], 8 + 22 * len(lines)), (0, 0, 0), -1)
        for k, t in enumerate(lines):
            cv2.putText(img, t, (8, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col if k == 0 else (235, 235, 235),
                        1, cv2.LINE_AA)
    return img


def _jpeg(bgr: np.ndarray, q: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes()


# ----------------------------------------------------------------------------- app
class App:
    def __init__(self, cams, inspector, out_root: str | Path = "outputs/sweeps", arm=None, follower_id="follower",
                 follower_port="auto", allow_connect: bool = True, ramp_speed: float = 20.0, fps: float = 30.0,
                 live_hz: float = 8.0, arm_factory=None, pose_loader=None, limits_loader=None):
        self.cams, self.insp = cams, inspector
        self.out_root = Path(out_root).expanduser()
        self.arm, self.follower_id, self.follower_port = arm, follower_id, follower_port
        self.allow_connect, self.ramp_speed, self.fps, self.live_hz = allow_connect, ramp_speed, fps, live_hz
        self.arm_factory = arm_factory or (lambda: RealArm(self.follower_port, self.follower_id))
        self.pose_loader = pose_loader or self._load_pose
        self.limits_loader = limits_loader or self._load_limits
        self.settings = dict(DEFAULTS, device=inspector.device,
                             roi=list(inspector.roi) if inspector.roi else None)
        self.lock = threading.RLock()
        self.motion_lock = threading.Lock()  # one motion job at a time
        self.abort = threading.Event()
        self.live_on = threading.Event()
        self.closing = threading.Event()
        self.live: dict | None = None
        self.live_window: deque = deque(maxlen=3)
        self.lat: dict[str, deque] = {}
        self.job: dict | None = None
        self.thumbs: dict[str, dict[int, bytes]] = {}
        self.message = ""
        self.arm_error = ""
        self._live_thread = threading.Thread(target=self._live_loop, name="live-classify", daemon=True)
        self._live_thread.start()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load_pose(follower_id):
        from dlsj.loop.poses import load_pose

        return load_pose(follower_id, "inspect")

    @staticmethod
    def _load_limits(follower_id):
        from dlsj import arms

        return arms.joint_limits_deg(follower_id, "wrist_roll")

    def _record_latency(self, c: dict) -> None:
        d = self.lat.setdefault(c["device"], deque(maxlen=500))
        d.append((c["model_ms"], c["ms"]))

    def latency(self) -> dict:
        out = {}
        for dev, d in list(self.lat.items()):
            if d:
                a = np.array(d)
                out[dev] = {"n": len(a), "model_median_ms": round(float(np.median(a[:, 0])), 2),
                            "model_p90_ms": round(float(np.percentile(a[:, 0], 90)), 2),
                            "e2e_median_ms": round(float(np.median(a[:, 1])), 2),
                            "e2e_p90_ms": round(float(np.percentile(a[:, 1], 90)), 2)}
        return out

    def busy(self) -> bool:
        return self.job is not None and self.job.get("state") in ("running", "returning", "moving")

    @staticmethod
    def _public(c: dict) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k != "crop"} | {
            "probs": {k: round(v, 4) for k, v in c["probs"].items()}}

    # ------------------------------------------------------------------ live classify
    def _live_loop(self) -> None:
        seq = -1
        while not self.closing.is_set():
            if not self.live_on.is_set() or self.busy():
                time.sleep(0.05)
                continue
            t0 = time.perf_counter()
            try:
                seq, f = self.cams.wait_new("inspect", seq, 1.0)
                if f is None:
                    continue
                c = self.insp.classify(np.ascontiguousarray(f[..., ::-1]))
                self._record_latency(c)
                with self.lock:
                    self.live_window.append(c["probs"])
                    self.live = self._public(c) | {"t": time.time()}
                    self.live["crop_jpeg"] = base64.b64encode(_jpeg(c["crop"], 85)).decode()
                    self.live["mean3"] = self._mean3()
            except Exception as e:  # noqa: BLE001
                log.warning("live classify: %s", e)
                self.message = f"live classify error: {e}"
                time.sleep(0.5)
            dt = time.perf_counter() - t0
            if self.live_hz > 0 and dt < 1.0 / self.live_hz:
                time.sleep(1.0 / self.live_hz - dt)

    def _mean3(self) -> dict:
        """The release's rule on the last 3 live frames: mean probability vector, conf >= .8, margin >= .2."""
        if len(self.live_window) < 3:
            return {"ready": False, "n": len(self.live_window)}
        mean = {c: float(np.mean([p[c] for p in self.live_window])) for c in YOLO_CLASSES}
        conf_c, top, conf, margin = agg_mod.gate(mean, self.insp.min_conf, self.insp.min_margin)
        return {"ready": True, "probs": {k: round(v, 4) for k, v in mean.items()}, "decision": conf_c or "uncertain",
                "top": top, "conf": round(conf, 4), "margin": round(margin, 4)}

    def set_live(self, on: bool) -> dict:
        if on:
            self.live_window.clear()
            self.live_on.set()
        else:
            self.live_on.clear()
        return {"live": self.live_on.is_set()}

    # ------------------------------------------------------------------ settings
    def apply_settings(self, s: dict) -> dict:
        out = dict(self.settings)
        if "frames" in s:
            out["frames"] = int(np.clip(int(s["frames"]), 1, 60))
        if "range_deg" in s:
            out["range_deg"] = float(np.clip(float(s["range_deg"]), 0.0, 270.0))
        if "settle_ms" in s:
            out["settle_ms"] = float(np.clip(float(s["settle_ms"]), 0.0, 3000.0))
        if "speed" in s:
            out["speed"] = float(np.clip(float(s["speed"]), 5.0, MAX_SPEED_HARD))
        if "strategy" in s:
            if s["strategy"] not in ALL_STRATEGIES:
                raise ValueError(f"unknown strategy {s['strategy']!r}")
            out["strategy"] = s["strategy"]
        if "roi" in s and s["roi"] is not None:
            from dlsj.inspect.core import parse_box

            roi = s["roi"] if isinstance(s["roi"], (list, tuple)) else (parse_box(str(s["roi"])) if str(s["roi"]).strip()
                                                                       else None)
            if roi is not None:
                roi = tuple(int(v) for v in roi)
                if len(roi) != 4 or roi[2] <= roi[0] or roi[3] <= roi[1] or min(roi) < 0:
                    raise ValueError(f"bad ROI {roi}")
            if self.busy():
                raise RuntimeError("can't change the ROI during a sweep")
            self.insp.roi = roi
            self.live_window.clear()
        for b in ("dry_run", "grasp_check"):
            if b in s:
                out[b] = bool(s[b])
        if "device" in s and str(s["device"]).upper() != self.insp.device:
            if self.busy():
                raise RuntimeError("can't switch device during a sweep")
            out["device"] = self.insp.set_device(str(s["device"]))
            self.live_window.clear()
        out["device"] = self.insp.device
        out["roi"] = list(self.insp.roi) if self.insp.roi else None
        self.settings = out
        return out

    # ------------------------------------------------------------------ arm
    def connect_arm(self) -> dict:
        if self.arm is not None:
            return {"ok": True, "message": "already connected"}
        if not self.allow_connect:
            raise RuntimeError("arm connection disabled (--no-arm)")
        with self.motion_lock:
            try:
                self.arm = self.arm_factory()
                self.arm_error = ""
            except Exception as e:  # noqa: BLE001
                self.arm_error = f"{type(e).__name__}: {e}"
                raise
        return {"ok": True, "message": "connected; torque on, holding the current pose"}

    def arm_state(self) -> dict:
        d = {"connected": self.arm is not None, "error": self.arm_error, "allow_connect": self.allow_connect,
             "has_inspect_pose": self.pose_loader(self.follower_id) is not None}
        if self.arm is not None and self.motion_lock.acquire(blocking=False):  # never block on a running job
            try:
                d["state"] = [round(float(v), 2) for v in self.arm.read_state()]
            except Exception as e:  # noqa: BLE001
                d["error"] = str(e)
            finally:
                self.motion_lock.release()
        lim = self.limits_loader(self.follower_id)
        d["wrist_roll_limits"] = None if lim is None else [round(v, 1) for v in lim]
        return d

    # ------------------------------------------------------------------ jobs
    def start_sweep(self, settings: dict | None = None) -> dict:
        if settings:
            self.apply_settings(settings)
        s = dict(self.settings)
        if not s["dry_run"] and self.arm is None:
            raise RuntimeError("no arm connected: tick Dry run, or Connect arm first")
        if not self.motion_lock.acquire(blocking=False):
            raise RuntimeError("busy: another sweep or move is running")
        try:
            jid = datetime.now().strftime("%Y%m%d-%H%M%S") + ("_dry" if s["dry_run"] else "_live")
            self.abort.clear()
            self.job = {"id": jid, "kind": "sweep", "state": "running", "settings": s, "frames": [], "plan": [],
                        "started": time.time(), "decision": None, "message": "starting"}
            self.thumbs[jid] = {}
            threading.Thread(target=self._sweep_job, args=(self.job,), name="sweep", daemon=True).start()
        except Exception:
            self.motion_lock.release()
            raise
        return {"id": jid}

    def stop(self) -> dict:
        self.abort.set()
        return {"ok": True, "state": None if self.job is None else self.job["state"]}

    def _sweep_job(self, job: dict) -> None:
        from concurrent.futures import ThreadPoolExecutor

        s = job["settings"]
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sweep-infer")
        recs: list = []
        crops: dict[int, np.ndarray] = {}
        sres = None
        try:
            if s["dry_run"]:
                start = self.arm.goal() if self.arm is not None else np.array([0, -40, 50, 0, 0, 30], np.float32)
                robot = SimArm(start)
                limits = self.limits_loader(self.follower_id)
            else:
                robot = self.arm
                limits = self.limits_loader(self.follower_id)
            start_cmd = robot.goal()
            cfg = SweepConfig(frames=s["frames"], range_deg=s["range_deg"], settle_ms=s["settle_ms"],
                              max_speed_deg_s=min(float(s["speed"]), MAX_SPEED_HARD), limits=limits,
                              grasp_empty_pos=4.0 if s["grasp_check"] else -1e9, camera="inspect", fps=self.fps,
                              return_to_start=True)
            job["plan"] = [round(a, 2) for a in agg_plan(start_cmd, cfg)]
            job["start_roll"] = round(float(start_cmd[ROLL]), 2)
            job["message"] = f"sweeping {len(job['plan'])} stops"
            futures = []

            def infer(frame, i, angle):
                c = self.insp.classify(frame)
                self._record_latency(c)
                rec = self.insp.to_frame(i, angle, c)
                crops[i] = c["crop"]
                self.thumbs[job["id"]][i] = _jpeg(cv2.resize(c["crop"], (160, 160), interpolation=cv2.INTER_AREA), 85)
                pub = self._public(c)
                with self.lock:
                    job["frames"].append({"i": i, "angle": None if angle is None else round(float(angle), 2), **pub})
                    job["frames"].sort(key=lambda d: d["i"])
                return rec

            def on_frame(frame, angle):
                futures.append(pool.submit(infer, frame, len(futures), angle))

            sweeper = Sweeper(robot, self.cams, cfg, self.abort)
            sres = sweeper.run(start_cmd, on_frame=on_frame)
            job["sweep"] = sres.record()
            if not sres.returned:  # aborted (Sweeper skips the return on abort) or the return timed out
                job["state"] = "returning"
                job["message"] = "returning the wrist to its start angle"
                back = Sweeper(robot, self.cams, cfg, threading.Event())  # fresh event: the return is not abortable
                cmd = (sres.final_cmd if sres.final_cmd is not None else start_cmd).copy()
                st, state = back._move_roll(cmd, float(start_cmd[ROLL]), check_grasp=False)
                cmd[:] = start_cmd
                robot.send(cmd.copy())
                job["sweep"]["returned"] = st == "ok"
                job["sweep"]["final_roll"] = round(float(robot.read_state()[ROLL]), 2)
            recs = [f.result() for f in futures]
            job["status"] = sres.status
            if recs and sres.status == "ok":
                res = agg_mod.aggregate_all(recs, 0.5, 1 - self.insp.min_conf, self.insp.min_conf,
                                            self.insp.agg, agg_mod.ALL_STRATEGIES)
                chosen = res[s["strategy"]]
                out = chosen.outcome()
                job["decision"] = {"strategy": s["strategy"], "outcome": out.value, "bowl": bowl_for(out),
                                   "reason": chosen.summary.get("reason", ""),
                                   "mark_reason": chosen.summary.get("mark_reason", ""),
                                   "naive": {"outcome": res["yolo_mean"].outcome().value,
                                             "bowl": bowl_for(res["yolo_mean"].outcome()),
                                             "reason": res["yolo_mean"].summary.get("reason", "")},
                                   "alternatives": {k: a.outcome().value for k, a in res.items()},
                                   "summary": chosen.summary}
            else:
                res = None
            job["saved"] = str(self._save(job, sres, recs, crops, res))
            job["state"] = "done" if sres.status == "ok" else sres.status
            job["message"] = (f"done: {job['decision']['outcome']} -> {job['decision']['bowl']}" if job["decision"]
                              else f"sweep {sres.status}; no decision")
        except Exception as e:  # noqa: BLE001
            log.exception("sweep failed")
            job["state"], job["message"] = "error", f"{type(e).__name__}: {e}"
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            job["finished"] = time.time()
            self.motion_lock.release()

    def _save(self, job, sres, recs, crops, results) -> Path | None:
        if sres is None or not sres.frames:
            return None
        from dlsj.loop.sweep_test import save_sweep

        out = self.out_root / job["id"]
        info = {**self.insp.sweep_meta(), "model": self.insp.health().get("model_dir"), "stage2": "yolo",
                "classifier": "brick-yolo-classifier-v1"}
        save_sweep(out, sres.frames, sres.angles, None, sres, info, recs or None, None, results,
                   {"source": "sweep_live" + (" (dry run)" if job["settings"]["dry_run"] else ""),
                    "settings": job["settings"], "decision": job.get("decision"), "yolo": self.insp.health()})
        for i, c in crops.items():
            cv2.imwrite(str(out / f"crop_{i:02d}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return out

    def goto_inspect(self) -> dict:
        if self.arm is None:
            raise RuntimeError("no arm connected")
        pose = self.pose_loader(self.follower_id)
        if pose is None:
            raise RuntimeError("no saved inspect pose (python -m dlsj.loop.run --save-pose inspect)")
        if not self.motion_lock.acquire(blocking=False):
            raise RuntimeError("busy")
        self.abort.clear()
        self.job = {"id": datetime.now().strftime("%Y%m%d-%H%M%S") + "_pose", "kind": "pose", "state": "moving",
                    "message": f"ramping to the inspect pose at {self.ramp_speed:.0f} deg/s", "frames": [],
                    "started": time.time()}
        threading.Thread(target=self._pose_job, args=(self.job, np.asarray(pose, np.float32)), daemon=True).start()
        return {"ok": True}

    def _pose_job(self, job, pose) -> None:
        from dlsj.teleop_web import step_toward

        try:
            cmd = self.arm.goal()
            target = pose.copy()
            target[GRIPPER] = cmd[GRIPPER]  # keep the gripper as it is (it may hold a brick)
            step = self.ramp_speed / self.fps
            t_end = time.perf_counter() + 30.0
            while not self.abort.is_set() and time.perf_counter() < t_end:
                t0 = time.perf_counter()
                cmd = step_toward(cmd, target, step)
                self.arm.send(cmd.copy())
                if float(np.max(np.abs(cmd - target))) < 0.5:
                    break
                time.sleep(max(0.0, 1.0 / self.fps - (time.perf_counter() - t0)))
            job["state"] = "aborted" if self.abort.is_set() else "done"
            job["message"] = "stopped (holding where it is)" if self.abort.is_set() else "at the inspect pose"
        except Exception as e:  # noqa: BLE001
            job["state"], job["message"] = "error", f"{type(e).__name__}: {e}"
        finally:
            job["finished"] = time.time()
            self.motion_lock.release()

    # ------------------------------------------------------------------ history
    def history(self, limit: int = 30) -> list[dict]:
        out = []
        if not self.out_root.exists():
            return out
        for p in sorted(self.out_root.glob("*/sweep.json"), reverse=True):
            try:
                d = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if not str(d.get("source", "")).startswith("sweep_live"):
                continue
            dec = d.get("decision") or {}
            out.append({"id": p.parent.name, "created": d.get("created"), "source": d.get("source"),
                        "n": len(d.get("frames", [])), "status": (d.get("sweep") or {}).get("status"),
                        "outcome": dec.get("outcome"), "bowl": dec.get("bowl"),
                        "naive": (dec.get("naive") or {}).get("outcome")})
            if len(out) >= limit:
                break
        return out

    def state(self) -> dict:
        with self.lock:
            job = None
            if self.job is not None:
                job = {k: v for k, v in self.job.items()}
                job["frames"] = list(self.job.get("frames", []))
            return {"settings": self.settings, "live_on": self.live_on.is_set(),
                    "live": self.live, "job": job, "model": self.insp.health(), "latency": self.latency(),
                    "cameras": self.cams.info(), "message": self.message, "strategies": list(ALL_STRATEGIES)}

    def close(self) -> None:
        self.closing.set()
        self.abort.set()
        if self.motion_lock.acquire(timeout=15.0):  # wait for the wrist to return
            self.motion_lock.release()
        if self.arm is not None and hasattr(self.arm, "disconnect"):
            try:
                self.arm.disconnect()  # torque stays on (disable_torque_on_disconnect = False)
            except Exception as e:  # noqa: BLE001
                log.warning("disconnect: %s", e)


def agg_plan(start_cmd, cfg: SweepConfig) -> list[float]:
    from dlsj.loop.sweep import plan_angles

    return plan_angles(float(start_cmd[ROLL]), cfg)


# ----------------------------------------------------------------------------- HTTP
def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 100_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_POST(self):
            try:
                body = self._body()
                p = self.path
                if p == "/api/live":
                    return self._json(app.set_live(bool(body.get("on", True))))
                if p == "/api/settings":
                    return self._json(app.apply_settings(body))
                if p == "/api/sweep":
                    return self._json(app.start_sweep(body))
                if p == "/api/stop":
                    return self._json(app.stop())
                if p == "/api/pose/inspect":
                    return self._json(app.goto_inspect())
                if p == "/api/arm/connect":
                    return self._json(app.connect_arm())
                return self._json({"error": "not found"}, 404)
            except Exception as e:  # noqa: BLE001
                return self._json({"error": f"{e}"}, 409 if isinstance(e, RuntimeError) else 400)

        def do_GET(self):
            p = self.path.split("?", 1)[0]
            try:
                if p in ("/", "/index.html"):
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if p == "/api/state":
                    return self._json(app.state())
                if p == "/api/arm":
                    return self._json(app.arm_state())
                if p == "/api/history":
                    return self._json(app.history())
                if p.startswith("/api/history/"):
                    sid = unquote(p.removeprefix("/api/history/"))
                    f = app.out_root / Path(sid).name / "sweep.json"
                    if not f.exists():
                        return self._json({"error": "no such sweep"}, 404)
                    return self._send(200, f.read_bytes(), "application/json")
                if p.startswith("/thumb/"):
                    sid, _, i = p.removeprefix("/thumb/").removesuffix(".jpg").rpartition("/")
                    sid = Path(unquote(sid)).name
                    jpg = app.thumbs.get(sid, {}).get(int(i))
                    if jpg is None:
                        f = app.out_root / sid / f"crop_{int(i):02d}.jpg"
                        jpg = f.read_bytes() if f.exists() else None
                    return self._send(200, jpg, "image/jpeg") if jpg else self._send(404, b"", "text/plain")
                if p.startswith("/snapshot/"):
                    name = unquote(p.removeprefix("/snapshot/").removesuffix(".jpg"))
                    if name not in app.cams.names:
                        return self._send(404, b"", "text/plain")
                    if name == "inspect":
                        _, f = app.cams.get(name)
                        if f is None:
                            return self._send(503, b"", "text/plain")
                        return self._send(200, _jpeg(draw_overlay(f, app.insp.roi, app.live if app.live_on.is_set()
                                                                  else None)), "image/jpeg")
                    jpg = app.cams.jpeg(name)
                    return self._send(200, jpg, "image/jpeg") if jpg else self._send(503, b"", "text/plain")
                if p.startswith("/stream/"):
                    return self._stream(unquote(p.removeprefix("/stream/")))
                return self._send(404, b"", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 500)

        def _stream(self, name: str):
            if name not in app.cams.names:
                return self._send(404, b"", "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last, period = -1, 1.0 / (15.0 if name == "inspect" else 10.0)
            try:
                while not app.closing.is_set():
                    t0 = time.perf_counter()
                    last, f = app.cams.wait_new(name, last, 2.0)
                    if f is None:
                        continue
                    if name == "inspect":
                        job = app.job if app.busy() else None
                        note = f"SWEEP {len(job['frames'])}/{len(job.get('plan') or [])}" if job else ""
                        live = app.live if app.live_on.is_set() and not job else None
                        if job and job["frames"]:
                            live = dict(job["frames"][-1], box=app.insp.roi)
                            live.setdefault("confident", None)
                        jpg = _jpeg(draw_overlay(f, app.insp.roi, live, note), 75)
                    else:
                        jpg = app.cams.jpeg(name)
                        if jpg is None:
                            continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    dt = time.perf_counter() - t0
                    if dt < period:
                        time.sleep(period - dt)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Spin and Inspect</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1b2029;--muted:#667085;--line:#e3e6ec;--accent:#2f6fde;--danger:#d92d20;
--plain:#e8b400;--circle:#f07a1a;--cross:#8b4fc6;--ok:#1f9d55}
@media (prefers-color-scheme:dark){:root{--bg:#12151a;--panel:#1b1f26;--ink:#e7e9ee;--muted:#98a2b3;--line:#2b313b;
--accent:#5b8ff0;--danger:#f04438}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
header{display:flex;gap:12px;align-items:baseline;padding:10px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:17px;margin:0}header small{color:var(--muted)}
main{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:14px;padding:14px 16px;max-width:1500px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.cams img{width:100%;display:block;border-radius:6px;background:#000}
.small{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}.small figure{margin:0}
figcaption{font-size:12px;color:var(--muted)}
.btns{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button.stop{background:var(--danger);border-color:var(--danger);color:#fff}
button.on{outline:2px solid var(--accent)}button:disabled{opacity:.45;cursor:not-allowed}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px 12px}
label{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:3px}
input,select{font:inherit;padding:5px 7px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
label.chk{flex-direction:row;align-items:center;gap:6px;font-size:13px;color:var(--ink)}
.decision{font-size:22px;font-weight:650;display:flex;align-items:center;gap:10px}
.dot{width:22px;height:22px;border-radius:50%;display:inline-block;border:1px solid var(--line)}
.muted{color:var(--muted)}.mono{font-family:ui-monospace,monospace;font-size:12px}
.strip{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px}.thumb{flex:0 0 auto;width:92px;text-align:center;font-size:11px}
.thumb img{width:92px;height:92px;border-radius:6px;border:3px solid var(--line);display:block}
.bar{height:8px;border-radius:4px;background:var(--line);overflow:hidden;display:flex}.bar span{display:block;height:100%}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}
h2{font-size:14px;margin:4px 0 8px}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
#chart{width:100%;height:190px}.status{min-height:20px}
</style></head><body>
<header><h1>Spin &amp; inspect</h1><small id="model">loading…</small><small id="armline"></small></header>
<main>
<section>
 <div class="panel cams"><img id="inspect" alt="inspect camera with the crop box and the latest prediction">
  <div class="small"><figure><img id="wrist" alt="wrist camera"><figcaption>wrist</figcaption></figure>
  <figure><img id="top" alt="top camera"><figcaption>top</figcaption></figure></div></div>
 <div class="panel" style="margin-top:14px"><h2>Live (no motion)</h2>
  <div class="row"><img id="livecrop" width="120" height="120" alt="last model crop" style="border-radius:6px;background:#000">
  <div style="flex:1;min-width:200px"><div id="livetxt" class="muted">Classify live is off.</div>
  <div id="livebars"></div><div id="mean3" class="muted"></div></div></div></div>
</section>
<section>
 <div class="panel">
  <div class="btns">
   <button id="bLive">Classify live</button>
   <button id="bSweep" class="primary">Spin &amp; inspect</button>
   <button id="bStop" class="stop">Stop</button>
   <button id="bPose">Go to inspect pose</button>
   <button id="bArm">Connect arm</button>
  </div>
  <div class="grid">
   <label>Frames<input id="frames" type="number" min="1" max="60"></label>
   <label>Sweep range (deg)<input id="range_deg" type="number" min="0" max="270" step="5"></label>
   <label>Settle (ms)<input id="settle_ms" type="number" min="0" max="3000" step="10"></label>
   <label>Speed cap (deg/s)<input id="speed" type="number" min="5" max="120" step="5"></label>
   <label>Device<select id="device"><option>NPU</option><option>GPU</option><option>CPU</option></select></label>
   <label>Strategy<select id="strategy"></select></label>
   <label>Crop ROI x0,y0,x1,y1<input id="roi" placeholder="auto (centre)" title="fixed inspect crop in 640x480 pixels; empty = centred square"></label>
  </div>
  <div class="row" style="margin-top:8px"><label class="chk"><input id="dry_run" type="checkbox"> Dry run (simulated wrist, no arm motion)</label>
  <label class="chk"><input id="grasp_check" type="checkbox"> Grasp check</label></div>
  <div id="status" class="status muted" role="status" style="margin-top:8px"></div>
 </div>
 <div class="panel" style="margin-top:14px"><h2>Sweep</h2>
  <div id="decision" class="decision muted">No sweep yet</div><div id="reason" class="muted"></div><div id="naive" class="muted"></div>
  <svg id="chart" viewBox="0 0 600 190" role="img" aria-label="P(plain), P(circle), P(cross) per wrist angle"></svg>
  <div class="row muted" style="font-size:12px"><span><i class="dot" style="background:var(--plain);width:10px;height:10px"></i> plain</span>
  <span><i class="dot" style="background:var(--circle);width:10px;height:10px"></i> circle</span>
  <span><i class="dot" style="background:var(--cross);width:10px;height:10px"></i> cross</span><span>filled = confident (≥0.8, margin ≥0.2)</span></div>
  <div id="strip" class="strip" style="margin-top:8px"></div>
  <div id="alts" class="mono muted" style="margin-top:6px"></div>
 </div>
 <div class="panel" style="margin-top:14px"><h2>Latency per frame</h2><div id="lat" class="mono muted"></div></div>
 <div class="panel" style="margin-top:14px"><h2>History</h2><table id="hist"></table></div>
</section></main>
<script>
const $=id=>document.getElementById(id);const COL={plain:'var(--plain)',circle:'var(--circle)',cross:'var(--cross)'};
const BOWL={yellow:'#e8b400',orange:'#f07a1a',purple:'#8b4fc6'};
let S=null,settingsLoaded=false,lastJob=null;
async function api(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
 const j=await r.json();if(!r.ok)throw Error(j.error||r.status);return j;}
function settings(){return{frames:+$('frames').value,range_deg:+$('range_deg').value,settle_ms:+$('settle_ms').value,speed:+$('speed').value,
 device:$('device').value,strategy:$('strategy').value,roi:$('roi').value.trim(),dry_run:$('dry_run').checked,grasp_check:$('grasp_check').checked};}
function msg(t,err){$('status').textContent=t;$('status').style.color=err?'var(--danger)':'';}
async function act(fn){try{const r=await fn();if(r&&r.message)msg(r.message);}catch(e){msg(e.message,true);}refresh();}
$('bLive').onclick=()=>act(()=>api('/api/live',{on:!S.live_on}));
$('bSweep').onclick=()=>{const s=settings();if(!s.dry_run&&!confirm('The arm will roll the wrist through '+s.range_deg+'° ('+s.frames+' stops, ≤'+s.speed+'°/s). Brick held, area clear?'))return;act(()=>api('/api/sweep',s));};
$('bStop').onclick=()=>act(()=>api('/api/stop'));
$('bPose').onclick=()=>{if(confirm('Ramp the arm slowly to the saved inspect pose?'))act(()=>api('/api/pose/inspect'));};
$('bArm').onclick=()=>{if(confirm('Connect the follower? Torque turns on and it holds its current pose (no motion).'))act(()=>api('/api/arm/connect'));};
for(const id of ['device','strategy','roi'])$(id).onchange=()=>act(()=>api('/api/settings',settings()));
function bars(p){return '<div class="bar">'+['plain','circle','cross'].map(c=>`<span style="width:${(p[c]*100).toFixed(1)}%;background:${COL[c]}" title="${c} ${p[c].toFixed(3)}"></span>`).join('')+'</div>';}
function fmtP(p){return ['plain','circle','cross'].map(c=>c+' '+p[c].toFixed(2)).join(' · ');}
function chart(frames){const W=600,H=190,L=34,R=8,T=8,B=24,svg=$('chart');if(!frames.length){svg.innerHTML='';return;}
 const ang=frames.map((f,k)=>f.angle==null?k:f.angle),lo=Math.min(...ang),hi=Math.max(...ang),span=Math.max(hi-lo,1e-6);
 const x=a=>L+(W-L-R)*(frames.length==1?.5:(a-lo)/span),y=v=>T+(H-T-B)*(1-v);
 let g=`<line x1="${L}" x2="${W-R}" y1="${y(.8)}" y2="${y(.8)}" stroke="currentColor" opacity=".25" stroke-dasharray="4 4"/>`;
 for(const v of [0,.5,1])g+=`<text x="${L-6}" y="${y(v)+4}" font-size="10" text-anchor="end" fill="currentColor" opacity=".6">${v}</text>`;
 const order=frames.map((f,k)=>[ang[k],f]).sort((a,b)=>a[0]-b[0]);
 for(const c of ['plain','circle','cross']){g+=`<polyline fill="none" stroke="${COL[c]}" stroke-width="2" points="${order.map(([a,f])=>x(a)+','+y(f.probs[c])).join(' ')}"/>`;
  for(const [a,f] of order){const conf=f.confident===c;g+=`<circle cx="${x(a)}" cy="${y(f.probs[c])}" r="${conf?5:3}" fill="${conf?COL[c]:'var(--panel)'}" stroke="${COL[c]}" stroke-width="1.5"><title>${a.toFixed(1)}°: ${c} ${f.probs[c].toFixed(3)}</title></circle>`;}}
 for(const [a] of order)g+=`<text x="${x(a)}" y="${H-6}" font-size="10" text-anchor="middle" fill="currentColor" opacity=".6">${Math.round(a)}</text>`;
 svg.innerHTML=g;}
function strip(job){$('strip').innerHTML=(job.frames||[]).map(f=>{const c=f.confident;const col=c?COL[c]:'var(--line)';
 return `<div class="thumb"><img src="/thumb/${job.id}/${f.i}.jpg" style="border-color:${col}" alt="frame ${f.i}"><b>${c||'unsure'}</b> ${f.conf.toFixed(2)}<br>${f.angle==null?'':f.angle.toFixed(0)+'°'}${bars(f.probs)}</div>`;}).join('');}
function showJob(job){if(!job||job.kind!=='sweep'){if(job)$('status').textContent=job.message;return;}
 const n=(job.plan||[]).length;msg(`${job.id}: ${job.state} · ${job.frames.length}/${n} frames · ${job.message}`,job.state==='error');
 chart(job.frames);strip(job);const d=job.decision;
 if(d){$('decision').className='decision';$('decision').innerHTML=`<span class="dot" style="background:${BOWL[d.bowl]||'transparent'}"></span>${d.outcome.toUpperCase()} → ${d.bowl}`;
  $('reason').textContent=`${d.strategy}: ${d.reason}${d.mark_reason?' · '+d.mark_reason:''}`;
  $('naive').textContent=`Release rule (mean of all frames): ${d.naive.outcome} → ${d.naive.bowl} · ${d.naive.reason}`;
  $('alts').textContent='all strategies: '+Object.entries(d.alternatives).map(([k,v])=>k+'='+v).join('  ');}
 else if(['running','returning'].includes(job.state)){$('decision').className='decision muted';$('decision').textContent=job.state==='returning'?'Returning wrist…':'Sweeping…';$('reason').textContent='';$('naive').textContent='';$('alts').textContent='';}
 else{$('decision').className='decision muted';$('decision').textContent='No decision ('+job.state+')';$('reason').textContent=job.message;}}
async function refresh(){try{S=await (await fetch('/api/state')).json();}catch(e){msg('server unreachable',true);return;}
 const m=S.model;$('model').textContent=`YOLO26n · ${m.device} (${m.precision})${m.requested_device!==m.device?' · fell back from '+m.requested_device:''} · gate ≥${m.min_confidence} / margin ≥${m.min_margin} · ROI ${m.roi?m.roi.join(','):'auto'}`;
 if(!settingsLoaded){const st=S.settings;$('strategy').innerHTML=S.strategies.map(s=>`<option>${s}</option>`).join('');
  for(const k of ['frames','range_deg','settle_ms','speed','device','strategy'])$(k).value=st[k];$('dry_run').checked=st.dry_run;$('grasp_check').checked=st.grasp_check;$('roi').value=st.roi?st.roi.join(','):'';settingsLoaded=true;}
 $('device').value=S.settings.device;
 $('bLive').classList.toggle('on',S.live_on);$('bLive').textContent=S.live_on?'Classify live: on':'Classify live';
 const L=S.live;if(S.live_on&&L){$('livecrop').src='data:image/jpeg;base64,'+L.crop_jpeg;$('livetxt').innerHTML=`<b style="color:${L.confident?COL[L.confident]:'inherit'}">${L.confident||'unsure ('+L.top+')'}</b> conf ${L.conf.toFixed(2)} margin ${L.margin.toFixed(2)} · ${L.ms.toFixed(1)} ms on ${L.device} (${L.source})<br><span class="mono">${fmtP(L.probs)}</span>`;$('livebars').innerHTML=bars(L.probs);
  const M=L.mean3;$('mean3').textContent=M&&M.ready?`Release 3-frame mean: ${M.decision} (${M.top} ${M.conf.toFixed(2)}, margin ${M.margin.toFixed(2)})`:'';}
 else if(!S.live_on){$('livetxt').textContent='Classify live is off.';$('livebars').innerHTML='';$('mean3').textContent='';}
 $('lat').textContent=Object.entries(S.latency).map(([d,v])=>`${d}: model ${v.model_median_ms} ms (p90 ${v.model_p90_ms}) · crop+model ${v.e2e_median_ms} ms (p90 ${v.e2e_p90_ms}) · n=${v.n}`).join('\n')||'no frames yet';
 const busy=S.job&&['running','returning','moving'].includes(S.job.state);$('bSweep').disabled=busy;$('bPose').disabled=busy;
 if(S.job)showJob(S.job);if(S.message&&!S.job)msg(S.message,true);
 if(S.job&&S.job.state!==lastJob){lastJob=S.job.state;history();arm();}}
async function arm(){try{const a=await (await fetch('/api/arm')).json();$('bArm').disabled=a.connected||!a.allow_connect;$('bPose').disabled=!a.connected||!a.has_inspect_pose;
 $('armline').textContent=a.connected?`arm connected · wrist_roll ${a.state?a.state[4].toFixed(1)+'°':'?'} · limits ${a.wrist_roll_limits||'?'}`:`arm not connected${a.error?' · '+a.error:''} (dry run only)`;}catch(e){}}
async function history(){try{const h=await (await fetch('/api/history')).json();$('hist').innerHTML='<tr><th>sweep</th><th>frames</th><th>status</th><th>decision</th><th>release rule</th></tr>'+
 h.map(r=>`<tr><td class="mono">${r.id}</td><td>${r.n}</td><td>${r.status||''}</td><td>${r.outcome?r.outcome+' → '+r.bowl:'—'}</td><td>${r.naive||'—'}</td></tr>`).join('');}catch(e){}}
for(const n of ['inspect','wrist','top'])$(n).src='/stream/'+n;
setInterval(refresh,400);setInterval(arm,3000);refresh();arm();history();
</script></body></html>"""


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="unpacked brick-yolo-classifier-v1 (classifier.json, best.xml)")
    ap.add_argument("--device", default="NPU", help="NPU (default), GPU or CPU; falls back NPU -> GPU -> CPU")
    ap.add_argument("--roi", default=None, help="x0,y0,x1,y1 fixed inspect crop (default: centred square, 60%% of height)")
    ap.add_argument("--padding", type=float, default=0.0, help="crop padding (fraction of the box; detector crops used 0.05)")
    ap.add_argument("--jpeg-quality", type=int, default=95, help="JPEG round trip of the crop like the live tester (0 = off)")
    ap.add_argument("--detector-weights", default=None, help="one-class brick detector (if released); crops inside --roi")
    ap.add_argument("--realsense", default="inspect=243722062690,top=346422061366", help="name=serial,...")
    ap.add_argument("--realsense-depth", action="store_true", help="also stream depth (heavier; off by default)")
    ap.add_argument("--cameras", default="wrist=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0",
                    help="OpenCV cameras name=index_or_path,...")
    ap.add_argument("--arm", default="none", help="'none' (connect later from the page), 'auto' or a port: connect at start")
    ap.add_argument("--no-arm", action="store_true", help="never connect the arm (dry runs only)")
    ap.add_argument("--follower-id", default="follower")
    ap.add_argument("--out", default="outputs/sweeps")
    ap.add_argument("--ramp-speed", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=30.0, help="sweep control rate")
    ap.add_argument("--live-hz", type=float, default=8.0, help="max live classifications per second")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8012)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from dlsj.inspect.core import parse_box
    from dlsj.inspect.yolo_inspector import YoloInspector

    insp = YoloInspector(args.model_dir, args.device, roi=parse_box(args.roi), padding=args.padding,
                         jpeg_quality=args.jpeg_quality, detector_weights=args.detector_weights)
    log.info("YOLO classifier on %s (%s)%s", insp.device, getattr(insp.classifier, "precision", "?"),
             f"; notes: {insp.notes}" if insp.notes else "")
    cams = Cams.open(args.cameras, args.realsense, depth=args.realsense_depth)
    arm = None
    if args.arm != "none" and not args.no_arm:
        arm = RealArm(args.arm, args.follower_id)
        log.info("follower connected on %s; torque on, holding", arm.port)
    app = App(cams, insp, args.out, arm, args.follower_id, args.arm if args.arm != "none" else "auto",
              allow_connect=not args.no_arm, ramp_speed=args.ramp_speed, fps=args.fps, live_hz=args.live_hz)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.daemon_threads = True

    def _stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:  # keep nohup's "ignore SIGHUP"
        signal.signal(signal.SIGHUP, _stop)
    log.info("serving http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        app.close()
        server.server_close()
        cams.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
