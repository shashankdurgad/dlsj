"""Closed-loop brick sorting with a perception dashboard: the run-time loop for demos and judging.

PERCEIVE -> DETECT -> REASON -> ACT, repeated until the table is empty:

    home --> survey (top camera: bricks left? which colour next?) --> pick (VLA "pick up the <colour> brick")
         --> grasp confirmed (gripper squeezing an object at brick width) --> bring (VLA "bring the brick to the
         inspection camera") --> arrived (arm settled, inspection camera sees the brick) --> spin + YOLO on the NPU
         (`auto_inspect.py`) --> plate decided --> place (VLA "place the brick in the <plate> bowl")
         --> released (gripper open) --> home --> survey ... --> no bricks left --> verify (YOLO brick detector,
         zoomed tiles of the top view, a brick must persist across 8 frames) --> clear: stay home, done; bricks
         left: survey again.

Recovery: empty grasp / pick timeout -> home and retry (max --max-attempts in a row, then pause); brick dropped
while bringing or spinning -> home, it is picked again from the table; uncertain inspection -> re-spin -> purple;
Stop (Q) or any fault -> hold and pause the loop; Resume restarts from home.

Built on `auto_control.py` (spin-and-inspect) on top of `smolvla_control.py` (VLA, RTC, speed limits, Home):
policy behaviour lives there. Dashboard at http://<xps>:<port>/ (the old console stays at /classic).

    cd ~/inference/dlsj-smolvla && PYTHONPATH=$PWD ~/desmond/venvs/pi05rt/bin/python \
        ~/inference/deploy/smolvla/loop_control.py <smolvla_control.py args> [loop args, see --loop-help]
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_control as A  # noqa: E402  (also puts the ultralytics overlay on sys.path)
import collect as C  # noqa: E402
import inspect_pose as IP  # noqa: E402
import perception as P  # noqa: E402
import smolvla_control as S  # noqa: E402

log = logging.getLogger("loop_control")
LCFG = SimpleNamespace(bring_mode="vla", use_inspect_pose=False, pose_speed=30.0, nudge_speed=15.0, hold_min=4.0, hold_max=9.0, lost_below=3.0, open_above=12.0, pick_timeout=45.0,
                       bring_timeout=25.0, place_timeout=40.0, max_cycles=12, max_attempts=3,
                       workspace=(0, 60, 560, 480), empty_seconds=2.0, perception_hz=4.0, held_min_px=150,
                       held_band=(0.25, 0.75), fg_frac=0.35,
                       top_detector="~/inference/models/brick-detector-v1", top_device="NPU", top_conf=0.3,
                       verify_conf=0.25, verify_frames=8, verify_timeout=10.0, max_verify_fails=2,
                       log_dir="~/inference/logs/loop")
STEPS = ["survey", "pick", "bring", "tune", "inspect", "place", "home", "verify"]
STUDIO: dict = {}  # Physical AI Studio model record of the loaded policy (dashboard), see studio_info()


def studio_info(rest: list[str]) -> dict:
    """The Physical AI Studio model behind `--export` (run_loop.sh resolves STUDIO_MODEL to the export dir).

    Read from the `studio_model.json` that `dlsj.deploy.studio_registry` keeps next to the export; {} for an export
    that did not come from Studio. Never raises: the dashboard line is informational.
    """
    export = S._arg(rest, "--export")
    if not export:
        return {}
    try:
        from dlsj.deploy.studio_registry import record_for_export

        rec = record_for_export(export)
    except Exception:  # noqa: BLE001
        return {}
    if not rec:
        return {}
    keep = ("model_id", "name", "policy", "version", "project_name", "dataset_name", "dataset_id", "backend")
    return {**{k: rec.get(k) for k in keep}, "device": S._arg(rest, "--device") or ""}


class FrameStream:
    """Looks like a camera stream to the base console (cond/seq/jpeg), so /stream/<name> serves our overlays."""

    def __init__(self, name: str):
        self.name, self.cond, self.seq, self.jpeg, self.frame, self.shape = name, threading.Condition(), 0, None, None, None

    def publish(self, bgr: np.ndarray, quality: int = 80) -> None:
        import cv2

        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with self.cond:
                self.jpeg, self.shape, self.seq = buf.tobytes(), bgr.shape[:2], self.seq + 1
                self.cond.notify_all()

    def info(self) -> dict:
        return {"name": self.name, "virtual": True}

    def start(self):
        return self

    def stop(self) -> None:
        pass


def make_loop_controller_class(base):
    auto = A.make_auto_controller_class(base)

    class LoopController(auto):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            real = {k: v for k, v in self.cameras.items()}
            self.perc_streams = {"perc_top": FrameStream("perc_top"), "perc_inspect": FrameStream("perc_inspect")}
            self.cameras.update(self.perc_streams)  # same dict the console serves /stream/<name> from
            self.real_cameras = real
            try:  # the sweep only reads the real cameras
                self.auto_inspector.cameras = type(self.auto_inspector.cameras)(real)
            except Exception:  # noqa: BLE001
                pass
            self.detector = getattr(getattr(self.auto_inspector, "insp", None), "detector", None)
            self.top_detector = _make_top_detector()  # brick-detector-v1 again, own low threshold, zoomed tiles
            self.auto_inspector.listener = self._on_inspect_frame
            self._verify_frames: list = []  # tables seen during the final "is it really empty" check
            self._insp_ref = None  # blurred grey inspection frame with the arm away (taken while surveying)
            # fixed inspection pose (same for every brick), tunable from the dashboard
            self.follower_id = getattr(self.arm, "id", "follower")
            self.ip_pose, self.ip_src = IP.load(self.follower_id)
            self.ip_msg = f"inspect pose: {self.ip_src}"
            self._mover = None
            self._grip_target = None
            # "vla_offset": the VLA's own bring, then this offset from where it stopped (tunable, saved per follower)
            self.ip_offset, self.ip_off_src = IP.load_offset(self.follower_id)
            self.tune_hold = False  # dashboard switch: pause at the camera so the offset can be nudged live
            self.collect = C.Store()  # rig data for retraining the YOLO classifier
            self.collect_label = None  # plain / circle / cross: every spin is saved under this TRUE label
            self.collect_msg = ""
            self._bring_base = None
            insp = getattr(self.auto_inspector, "insp", None)
            if insp is not None and hasattr(insp, "crop_box"):
                self._roi_crop_box = insp.crop_box
                insp.crop_box = self._crop_box_held  # YOLO crops the HELD brick, not bricks on the table behind it
            self.L = {"mode": "off", "step": "idle", "since": time.monotonic(), "cycle": 0, "attempts": 0,
                      "target": None, "message": "Press Start loop", "cycles": [], "tallies": {},
                      "grasp": {"state": "open"}, "started": None, "events": collections.deque(maxlen=40)}
            self.perc = {"t": 0.0, "table": [], "sorted": [], "plates": [], "target": None, "inspect_seen": None,
                         "inspect": None, "workspace": list(LCFG.workspace), "verify": None}
            self._plates: dict[str, P.Plate] = {}
            self._count_hist: collections.deque = collections.deque(maxlen=64)  # (t, n bricks on the table)
            self._pose_hist: collections.deque = collections.deque(maxlen=60)  # 2 s at 30 Hz
            self._grip_hist: collections.deque = collections.deque(maxlen=15)
            self._insp_frames: list[dict] = []
            self._insp_crops: list = []
            self._marks: dict[str, float] = {}
            self._last_top = None
            self._logf = None
            try:
                d = Path(LCFG.log_dir).expanduser()
                d.mkdir(parents=True, exist_ok=True)
                self._logf = open(d / time.strftime("loop_%Y%m%d_%H%M%S.jsonl"), "a", buffering=1)
            except OSError:
                log.exception("no loop log file")
            self._perc_stop = threading.Event()
            threading.Thread(target=self._perception_loop, daemon=True, name="perception").start()

        # ------------------------------------------------------------------ bookkeeping
        def _event(self, kind: str, **kw) -> None:
            rec = {"t": round(time.time(), 3), "kind": kind, "cycle": self.L["cycle"], **kw}
            self.L["events"].appendleft(rec)
            log.info("loop %s %s", kind, kw)
            if self._logf:
                try:
                    self._logf.write(json.dumps(rec, default=str) + "\n")
                except Exception:  # noqa: BLE001
                    pass

        def _goto(self, step: str, message: str = "") -> None:
            self.L["step"], self.L["since"] = step, time.monotonic()
            if message:
                self.L["message"] = message
            self._marks.clear()
            self._event("step", step=step, message=message)

        def _elapsed(self) -> float:
            return time.monotonic() - self.L["since"]

        def _held_for(self, key: str, cond: bool, seconds: float) -> bool:
            """True once `cond` has been true continuously for `seconds`."""
            now = time.monotonic()
            if not cond:
                self._marks.pop(key, None)
                return False
            return now - self._marks.setdefault(key, now) >= seconds

        def _go_home(self, message: str) -> None:
            self._handle("home")  # SmolController: ease to the saved home pose, then phase "idle"
            if self.phase != "homing":
                self._pause(f"Cannot go home: {self.message}")
                return
            self._goto("home", message)

        def _pause(self, message: str) -> None:
            if self.phase not in ("idle", "stopped"):
                super()._handle("q")  # hold where we are
            self.L["mode"] = "paused"
            self._goto("paused", message)

        # ------------------------------------------------------------------ operator commands
        def loop_command(self, name: str) -> None:
            with self.lock:
                if name == "start":
                    if self.L["mode"] == "running":
                        return
                    self.L.update(mode="running", cycle=0, attempts=0, cycles=[], tallies={}, started=time.time(),
                                  verify_fails=0)
                    self._go_home("Loop started: going home")
                elif name == "resume":
                    if self.L["mode"] in ("paused", "done"):
                        self.L["mode"], self.L["attempts"] = "running", 0
                        self._go_home("Resuming: going home first")
                elif name == "pause":
                    if self.L["mode"] == "running":
                        self._pause("Paused by operator: holding")
                elif name == "stop":
                    if self.phase == "inspecting":
                        self.auto_inspector.stop()
                    super()._handle("q")
                    self.L["mode"] = "off"
                    self._goto("idle", "Stopped: holding position")
                elif name == "home":
                    self.L["mode"] = "off"
                    self._handle("home")
                    self._goto("idle", "Going home (loop off)")

        def _handle(self, c: str) -> None:
            if c in ("q", "stop") and self.L["mode"] == "running":
                super()._handle(c)
                if self.phase != "inspecting":  # an inspection reports "aborted" itself
                    self.L["mode"] = "paused"
                    self._goto("paused", "Stopped (Q): holding. Resume restarts from home")
                return
            super()._handle(c)

        # ------------------------------------------------------------------ the loop (called at 30 Hz)
        def tick(self) -> None:
            super().tick()
            with self.lock:
                if self.measured is not None:
                    self._pose_hist.append(np.asarray(self.measured, float).copy())
                    self._grip_hist.append(float(self.measured[5]))
                self._update_grasp()
                if self.L["mode"] != "running":
                    self._tuning_tick()
                    return
                try:
                    getattr(self, f"_step_{self.L['step']}", lambda: None)()
                except Exception as e:  # noqa: BLE001  never leave the loop running blind
                    log.exception("loop step failed")
                    self._pause(f"Loop error: {e}")

        # ------------------------------------------------------------------ inspection pose tuning (loop off)
        def _tuning_tick(self) -> None:
            if self.phase == "posing" and self._mover is not None:
                if self._mover.step(self):
                    self._mover, self.phase = None, "posed"
                    self.message = "At the inspection pose: nudge joints, then Save"
            if self._grip_target is not None and self.phase in ("posed", "posing", "idle", "holding") \
                    and self.cmd is not None:
                base = np.asarray(self.cmd, float)
                t = base.copy()
                t[5] = self._grip_target
                m = self._step(base + np.clip(t - base, -1.0, 1.0))  # 30 deg/s on the gripper
                if abs(float(m[5]) - self._grip_target) < 1.0 or (self._grip_target < 5 and abs(t[5] - base[5]) < 1e-6):
                    self._grip_target = None

        def pose_command(self, name: str, *args) -> None:
            with self.lock:
                if name in ("off_nudge", "off_save", "off_reset", "spin", "hold_on", "hold_off"):
                    return self._offset_command(name, *args)
                if self.L["step"] == "tune" and name in ("nudge", "save", "reset"):  # tuning the pose at the camera
                    if name == "nudge":
                        self.ip_pose = IP.nudge(self.ip_pose, args[0], float(args[1]))
                        self.ip_src, self.ip_msg = "edited (not saved)", f"{args[0]} {float(args[1]):+g} deg"
                    elif name == "reset":
                        self.ip_pose, self.ip_src = np.asarray(IP.DEMO_POSE, float), "demo (not saved)"
                    else:
                        path = IP.save(self.ip_pose, self.follower_id)
                        self.ip_src, self.ip_msg = "saved", f"Saved to {path}"
                        return self._event("inspect_pose_saved", pose=np.round(self.ip_pose, 1).tolist())
                    self._mover, self.phase = IP.Mover([self.ip_pose], LCFG.nudge_speed), "posing"
                    return
                if self.L["mode"] == "running" or self.phase in ("running", "readying", "inspecting", "homing", "moving"):
                    self.ip_msg = "Stop the loop / wait for the arm before tuning the pose"
                    return
                if self.cmd is None:
                    self.cmd = np.asarray(self.measured if self.measured is not None else self.arm.read(), float)
                if name == "goto":
                    cur = np.asarray(self.cmd, float)[:5]
                    far = float(np.abs(cur - self.ip_pose).max()) > 20
                    ready = None if getattr(self, "ready", None) is None else np.asarray(self.ready, float)[:5]
                    self._mover = IP.Mover([ready if far else None, self.ip_pose], LCFG.pose_speed)
                    self.phase, self.ip_msg = "posing", "Moving to the inspection pose"
                elif name == "nudge":
                    joint, delta = args[0], float(args[1])
                    self.ip_pose = IP.nudge(self.ip_pose, joint, delta)
                    self.ip_src = "edited (not saved)"
                    self.ip_msg = f"{joint} {delta:+g} deg"
                    if self.phase in ("posed", "posing"):
                        self._mover = IP.Mover([self.ip_pose], LCFG.nudge_speed)
                        self.phase = "posing"
                elif name == "save":
                    path = IP.save(self.ip_pose, self.follower_id)
                    self.ip_src, self.ip_msg = "saved", f"Saved to {path}"
                    self._event("inspect_pose_saved", pose=np.round(self.ip_pose, 1).tolist())
                elif name == "reset":
                    self.ip_pose, self.ip_src = np.asarray(IP.DEMO_POSE, float), "demo (not saved)"
                    self.ip_msg = "Back to the demo pose"
                    if self.phase in ("posed", "posing"):
                        self._mover, self.phase = IP.Mover([self.ip_pose], LCFG.nudge_speed), "posing"
                elif name in ("grip_close", "grip_open"):
                    self._grip_target = 0.0 if name == "grip_close" else 30.0
                    if self.phase not in ("posed", "posing"):
                        self.phase = "holding"
                    self.ip_msg = "Gripper " + ("closing" if name == "grip_close" else "opening")

        def _offset_command(self, name: str, *args) -> None:
            at_cam = self.L["step"] == "tune"
            if name == "hold_on":
                self.tune_hold, self.ip_msg = True, "Will hold at the camera after the next bring"
            elif name == "hold_off":
                self.tune_hold, self.ip_msg = False, "No hold: spins straight after the offset move"
            elif name == "off_nudge":
                joint, delta = args[0], float(args[1])
                self.ip_offset = IP.nudge(self.ip_offset, joint, delta)
                self.ip_off_src = "edited (not saved)"
                self.ip_msg = f"offset {joint} {delta:+g} deg"
                if at_cam and self._bring_base is not None:
                    self._mover, self.phase = IP.Mover([self._bring_base + self.ip_offset], LCFG.nudge_speed), "posing"
            elif name == "off_reset":
                self.ip_offset, self.ip_off_src = np.zeros(5), "zero (not saved)"
                if at_cam and self._bring_base is not None:
                    self._mover, self.phase = IP.Mover([self._bring_base], LCFG.nudge_speed), "posing"
            elif name == "off_save":
                path = IP.save_offset(self.ip_offset, self.follower_id)
                self.ip_off_src, self.ip_msg = "saved", f"Offset saved to {path}"
                self._event("inspect_offset_saved", offset=np.round(self.ip_offset, 1).tolist())
            elif name == "spin":
                if at_cam:
                    self._spin_now()
                else:
                    self.ip_msg = "Spin now only works while holding at the camera"

        # ------------------------------------------------------------------ rig data collection
        def _record_spin(self, res: dict, source: str) -> None:
            if self.collect_label and res.get("saved") and res.get("state") == "ok":
                e = self.collect.add(self.collect_label, res["saved"], pred=res.get("outcome"), source=source)
                n = self.collect.counts()[self.collect_label]
                self.collect_msg = f"Saved spin {e['id']} as {self.collect_label} ({n['spins']} spins, {n['crops']} crops)"
                self._event("collect", label=self.collect_label, entry=e["id"])

        def collect_command(self, name: str, *args) -> None:
            with self.lock:
                if name == "label":
                    lab = args[0] if args else "off"
                    self.collect_label = lab if lab in C.LABELS else None
                    self.collect_msg = f"Collecting as {self.collect_label}" if self.collect_label else "Collection off"
                elif name == "spin":
                    if self.L["mode"] == "running" or self.phase in ("running", "readying", "inspecting", "homing",
                                                                       "moving", "posing", "collecting"):
                        self.collect_msg = "Spin & save needs the loop off and the arm still"
                        return
                    if not self.collect_label:
                        self.collect_msg = "Pick the brick's label first"
                        return
                    if self.L["grasp"].get("pos", 0.0) < LCFG.lost_below:
                        self.collect_msg = "No brick in the gripper (close it on a brick first)"
                        return
                    if self.cmd is None:
                        self.cmd = np.asarray(self.measured if self.measured is not None else self.arm.read(), float)
                    self._insp_frames, self._insp_crops = [], []
                    self.phase, self.collect_msg = "collecting", f"Spinning a {self.collect_label} brick..."
                    threading.Thread(target=self._collect_job, args=(np.asarray(self.cmd, float).copy(),),
                                     daemon=True, name="collect-spin").start()

        def _collect_job(self, start_cmd) -> None:
            try:
                r = self.auto_inspector.inspect(self.arm, start_cmd)
                res = {"state": r.status, "outcome": r.outcome, "saved": r.saved}
            except Exception as e:  # noqa: BLE001
                log.exception("collection spin failed")
                res = {"state": "error", "saved": [], "outcome": None, "reason": str(e)}
            with self.lock:
                self.inspection = {**res, "bowl": None, "reason": res.get("reason", ""), "frames": len(self._insp_frames),
                                   "seconds": None, "early_stop": False, "attempts": 1}
                self._record_spin(res, "manual")
                if res["state"] != "ok":
                    self.collect_msg = f"Spin {res['state']}: not saved"
                self.cmd = np.asarray(start_cmd, float)
                self.phase = "holding"

        def _update_grasp(self) -> None:
            if self.measured is None or self.cmd is None:
                return
            g, c = float(self.measured[5]), float(self.cmd[5])
            still = len(self._grip_hist) >= 10 and (max(self._grip_hist) - min(self._grip_hist)) < 0.6
            squeezing = c < g - 1.0  # commanded further closed than the fingers can go: something is between them
            if LCFG.hold_min <= g <= LCFG.hold_max and squeezing and still:
                state = "holding"
            elif g < LCFG.lost_below and c < LCFG.lost_below + 2:
                state = "empty"
            elif g >= LCFG.open_above:
                state = "open"
            else:
                state = "closing" if c < g else "moving"
            self.L["grasp"] = {"state": state, "pos": round(g, 1), "cmd": round(c, 1)}

        def _step_home(self) -> None:
            if self.phase == "idle":
                self.L["survey_from"] = "home"
                self._goto("survey", "Home. Looking at the table")
            elif self.phase == "stopped":
                self._pause("Homing interrupted")

        # ------------------------------------------------------------------ physical verification (end of the loop)
        def _start_verify(self) -> None:
            """The survey saw an empty table from home: before stopping, check it physically with the YOLO brick
            detector (brick-detector-v1, zoomed tiles, lower threshold) over several frames."""
            with self.lock:
                self._verify_frames = []
            self.perc["verify"] = {"state": "running", "frames": 0, "need": LCFG.verify_frames, "bricks": []}
            self._goto("verify", f"Survey says the table is clear: verifying with YOLO over {LCFG.verify_frames} frames")

        def _step_verify(self) -> None:
            if self.phase == "stopped":
                return self._pause("Stopped while verifying")
            frames = list(self._verify_frames)
            self.perc["verify"] = {**(self.perc.get("verify") or {}), "frames": len(frames)}
            if len(frames) < LCFG.verify_frames:
                if self._elapsed() > LCFG.verify_timeout:
                    self._pause(f"Verification got {len(frames)}/{LCFG.verify_frames} top frames in "
                                f"{LCFG.verify_timeout:.0f} s: check the top camera / detector")
                return
            left = P.persistent_bricks(frames)
            self.perc["verify"] = {"state": "clear" if not left else "bricks", "frames": len(frames),
                                   "need": LCFG.verify_frames, "bricks": left}
            self._event("verify", bricks=len(left), boxes=[b["box"] for b in left],
                        conf=[round(b["conf"], 2) for b in left])
            if not left:
                self.L["mode"] = "done"
                total = sum(self.L["tallies"].values())
                return self._goto("done", f"Verified clear: YOLO sees no bricks on the table in {len(frames)} "
                                          f"frames. {total} brick(s) sorted. Loop stopped")
            self.L["verify_fails"] = self.L.get("verify_fails", 0) + 1
            if self.L["verify_fails"] > LCFG.max_verify_fails:
                return self._pause(f"Verification still sees {len(left)} brick(s) after {LCFG.max_verify_fails} "
                                   "re-surveys but the survey cannot target them: check the table, then Resume")
            self.L["survey_from"] = "home"
            self._goto("survey", f"Verification found {len(left)} brick(s) still on the table: back to work")

        def _step_survey(self) -> None:
            if self._elapsed() < 1.0:  # let the arm settle
                return
            from_home = self.L.get("survey_from", "home") == "home"
            since = self.L["since"] + 0.8
            recent = [n for t, n in self._count_hist if t >= since]
            if len(recent) >= max(3, int(LCFG.empty_seconds * LCFG.perception_hz * 0.6)) and max(recent) == 0:
                if not from_home:  # the arm may hide bricks from up here: only the view from home decides
                    return self._go_home("No bricks seen from here: going home to check the whole table")
                return self._start_verify()
            if self.L["cycle"] >= LCFG.max_cycles:
                self._pause(f"Reached --max-cycles {LCFG.max_cycles}")
                return
            target = self.perc.get("target")
            if not recent or target is None or self.perc["t"] < since:
                if self._elapsed() > 8.0:
                    if not from_home:
                        return self._go_home("Nothing clear to pick from here: going home to look again")
                    self._pause("Bricks seen but none with a clear colour: check the top camera")
                return
            if self.L["attempts"] >= LCFG.max_attempts:
                self._pause(f"{self.L['attempts']} failed picks in a row: check the scene, then Resume")
                return
            if from_home:  # arm out of the inspection camera's view: the background for picking out the held brick
                self._capture_inspect_reference()
            colour = target["colour"]
            self.L["cycle"] += 1
            self.L["target"] = target
            self.brick, self.start_stage = colour, "pick"
            if from_home:
                self._handle("start")  # AutoController/SmolController: bowl reset, ready pose, then the VLA runs
            else:  # straight on from above the plate: no trip home
                self.bowl, self.inspection, self.stage = None, None, "pick"
                self._replan_now()
                self.phase, self.message = "running", ""
            self.L["survey_from"] = "home"
            self.L["cycles"].append({"n": self.L["cycle"], "brick": colour, "t0": time.time(), "result": "running"})
            self._goto("pick", f"Cycle {self.L['cycle']}: pick up the {colour} brick")

        def _cycle(self, **kw) -> None:
            if self.L["cycles"]:
                self.L["cycles"][-1].update(kw)

        def _step_pick(self) -> None:
            if self.phase == "stopped":
                return self._pause("Stopped during pick")
            if self.phase != "running" or self.stage != "pick":
                return
            gs = self.L["grasp"]["state"]
            if self._held_for("grasp", gs == "holding", 0.4):
                self.stage = "bring"
                self._replan_now()
                self._cycle(grip=self.L["grasp"]["pos"], t_grasp=round(time.time() - self.L["cycles"][-1]["t0"], 1))
                self._event("grasp", pos=self.L["grasp"]["pos"])
                if LCFG.bring_mode == "fixed":  # the same fixed pose for every brick: raised ready pose, then the camera
                    self._hold_here()
                    self.phase = "moving"
                    ready = None if getattr(self, "ready", None) is None else np.asarray(self.ready, float)[:5]
                    self._mover = IP.Mover([ready, self.ip_pose], LCFG.pose_speed)
                    return self._goto("bring", f"Grasp confirmed: moving to the inspection pose ({self.ip_src})")
                self._goto("bring", "Grasp confirmed: bringing the brick to the inspection camera")
                return
            if self._elapsed() > LCFG.pick_timeout:
                self.L["attempts"] += 1
                self._cycle(result="pick timeout")
                self._go_home(f"No grasp after {LCFG.pick_timeout:.0f} s: going home to retry")

        def _step_bring(self) -> None:
            if self.phase == "stopped":
                self._mover = None
                return self._pause("Stopped while bringing")
            if self._mover is not None:  # scripted move to the fixed inspection pose
                if self._held_for("lost", self.L["grasp"].get("pos", 99.0) < LCFG.lost_below, 0.5):
                    self._mover = None
                    self.L["attempts"] += 1
                    self._cycle(result="dropped while bringing")
                    return self._go_home("Brick dropped: going home, it will be picked again")
                done = self._mover.step(self)
                if self._held_for("posed", done, 0.5) and self.tune_hold and LCFG.bring_mode in ("vla_offset", "vla_pose"):
                    self._mover, self.phase = None, "posed"
                    return self._goto("tune", "Holding at the camera: nudge the offset, then Spin now (Save to keep)")
                if self._held_for("posed", done, 0.5):  # arrived and settled
                    self._mover = None
                    self.phase = "running"
                    self._cycle(t_inspect=round(time.time() - self.L["cycles"][-1]["t0"], 1))
                    self._insp_frames, self._insp_crops = [], []
                    self._start_inspection()
                    self._goto("inspect", "At the inspection pose: spinning the brick, YOLO on the NPU")
                return
            if self.phase != "running":
                return
            g = self.L["grasp"].get("pos", 99.0)
            if self._held_for("lost", g < LCFG.lost_below, 0.5):
                self.L["attempts"] += 1
                self._cycle(result="dropped while bringing")
                return self._go_home("Brick dropped: going home, it will be picked again")
            if self._elapsed() < 2.5:
                return
            # the VLA brings the brick to the camera by itself (the same bring prompt as the manual console): spin
            # once it has finished, i.e. the arm has been still for a while
            # "finished" = < 2.5 deg over the last 2 s on every joint. The VLA's last stretch towards the camera is a slow
            # 2-4 deg/s creep; a 0.67 s window called that "still" and spun ~10-15 deg short of where the manual runs
            # pressed M (traces, 15 Nov). Manual M presses came after 4.5-8 s of bring at <= ~1 deg/s.
            settled = (len(self._pose_hist) >= 60
                       and float(np.ptp(np.stack(self._pose_hist)[:, :5], axis=0).max()) < 2.5)
            arrived = (self._elapsed() > 4.0 and self._held_for("settled", settled, 0.5)) or self._elapsed() > LCFG.bring_timeout
            if arrived and LCFG.bring_mode == "vla_pose":  # straight from the VLA's endpoint to the one saved pose
                cur = np.asarray(self.measured if self.measured is not None else self.cmd, float)
                self._bring_base = cur[:5].copy()
                self._hold_here()
                self.phase = "moving"
                self._mover = IP.Mover([self.ip_pose], LCFG.pose_speed)
                self._cycle(vla_end=np.round(self._bring_base, 1).tolist(), pose=np.round(self.ip_pose, 1).tolist())
                return self._goto("bring", f"VLA at the camera: moving to the inspection pose ({self.ip_src})")
            if arrived and LCFG.bring_mode == "vla_offset" and (np.any(self.ip_offset) or self.tune_hold):
                cur = np.asarray(self.measured if self.measured is not None else self.cmd, float)
                self._bring_base = cur[:5].copy()  # where the arm actually stopped
                self._hold_here()
                self.phase = "moving"
                self._mover = IP.Mover([self._bring_base + self.ip_offset], LCFG.nudge_speed)
                self._cycle(vla_end=np.round(self._bring_base, 1).tolist(), offset=np.round(self.ip_offset, 1).tolist())
                return self._goto("bring", "VLA at the camera: applying the inspection offset")
            if arrived:
                self._cycle(t_inspect=round(time.time() - self.L["cycles"][-1]["t0"], 1))
                self._insp_frames, self._insp_crops = [], []
                self._start_inspection()  # AutoController: policy paused, wrist spin + YOLO in a thread
                self._goto("inspect", "At the camera: spinning the brick, YOLO on the NPU")

        def _step_tune(self) -> None:
            if self.phase == "stopped":
                self._mover = None
                return self._pause("Stopped while tuning")
            if self._held_for("lost", self.L["grasp"].get("pos", 99.0) < LCFG.lost_below, 0.5):
                self._mover = None
                self.L["attempts"] += 1
                self._cycle(result="dropped at the camera")
                return self._go_home("Brick dropped: going home, it will be picked again")
            if self._mover is not None and self._mover.step(self):
                self._mover, self.phase = None, "posed"

        def _spin_now(self) -> None:
            self._mover = None
            self.phase = "running"
            self._cycle(t_inspect=round(time.time() - self.L["cycles"][-1]["t0"], 1),
                        offset=np.round(self.ip_offset, 1).tolist())
            self._insp_frames, self._insp_crops = [], []
            self._start_inspection()
            self._goto("inspect", "At the camera: spinning the brick, YOLO on the NPU")

        def _step_inspect(self) -> None:
            if self.phase == "inspecting":
                return
            res = self.inspection or {}
            self._record_spin(res, "loop")
            if res.get("state") == "ok" and self.stage == "place" and self.phase == "running":
                self._cycle(outcome=res.get("outcome"), bowl=res.get("bowl"), frames=res.get("frames"),
                            inspect_s=res.get("seconds"), early=res.get("early_stop"))
                self._event("decision", outcome=res.get("outcome"), bowl=res.get("bowl"), reason=res.get("reason"))
                return self._goto("place", f"{_label(res.get('outcome'))} -> {res.get('bowl')} plate: placing")
            st = res.get("state")
            if st == "aborted":
                self._cycle(result="inspection stopped")
                return self._pause("Inspection stopped by operator")
            self.L["attempts"] += 1
            self._cycle(result=f"inspection {st}")
            self._go_home("Brick dropped during the spin: going home" if st == "grasp_lost"
                          else f"Inspection failed ({res.get('reason')}): going home")

        def _step_place(self) -> None:
            if self.phase == "stopped":
                return self._pause("Stopped while placing")
            if self.phase != "running":
                return
            if self._elapsed() > 1.0 and self._held_for("open", self.L["grasp"].get("pos", 0.0) >= LCFG.open_above, 0.3):
                return self._goto("release", "Released: clearing the plate")
            if self._elapsed() > LCFG.place_timeout:
                self._cycle(result="place timeout")
                self._pause(f"Not released after {LCFG.place_timeout:.0f} s: holding, check the gripper")

        def _step_release(self) -> None:
            if self._elapsed() < 0.8:  # let the VLA lift away from the plate
                return
            bowl = self.bowl or "?"
            self.L["tallies"][bowl] = self.L["tallies"].get(bowl, 0) + 1
            self.L["attempts"], self.L["verify_fails"] = 0, 0
            c = self.L["cycles"][-1] if self.L["cycles"] else {}
            self._cycle(result="sorted", seconds=round(time.time() - c.get("t0", time.time()), 1))
            self._event("sorted", bowl=bowl, seconds=c.get("seconds"))
            self._hold_here()  # no trip home: look for the next brick from above the plate
            self.L["survey_from"] = "plate"
            self._goto("survey", f"Placed in the {bowl} plate: looking for the next brick from here")

        def _hold_here(self) -> None:
            """Policy off, arm held at its current command (the base tick only moves the arm while 'running')."""
            self._replan_now()
            if self.cmd is None and self.measured is not None:
                self.cmd = np.asarray(self.measured, float).copy()
            self.phase = "holding"

        # ------------------------------------------------------------------ perception
        def _perception_loop(self) -> None:
            # The detector (ultralytics, Python-heavy) holds the GIL long enough to starve the 30 Hz servo reads
            # (timeouts / checksum / parsing errors on the Feetech bus), so it only runs when its answer is needed:
            # the top camera while surveying (arm home, nothing moving), the inspection camera while bringing.
            while not self._perc_stop.is_set():
                t0 = time.monotonic()
                step = self.L["step"]
                try:
                    self._perceive_once(detect_top=step in ("survey", "verify", "idle", "done", "paused"),
                                        detect_inspect=step in ("bring", "idle", "paused"))
                except Exception:  # noqa: BLE001
                    log.exception("perception failed")
                hz = LCFG.perception_hz if step in ("survey", "verify", "bring") else 1.5
                time.sleep(max(0.05, 1.0 / hz - (time.monotonic() - t0)))

        def _perceive_once(self, detect_top: bool = True, detect_inspect: bool = True) -> None:
            top = getattr(self.real_cameras.get("top"), "frame", None)
            if not detect_top:  # arm moving: keep the last result (the page overlays it on the live video)
                top = None
            if top is not None:
                top = top.copy()
                for p in P.find_plates(top):
                    self._plates[p.colour] = p  # plates don't move: keep the last good circle per colour
                plates = list(self._plates.values())
                if self.top_detector is not None:  # zoomed tiles: the ~40 px top-view bricks score 0.5+, not 0.1-0.45
                    dets = P.detect_tiled(self.top_detector, top, LCFG.workspace)
                else:
                    dets = self.detector.detect(top) if self.detector is not None else []
                table, srt = P.split_bricks(dets, plates, top, LCFG.workspace, min_conf=LCFG.top_conf)
                target = P.choose_target(table)
                if self.L["step"] == "verify":  # lower bar, but it has to persist across frames (persistent_bricks)
                    vt, _ = P.split_bricks(dets, plates, top, LCFG.workspace, min_conf=LCFG.verify_conf)
                    with self.lock:
                        self._verify_frames.append(vt)
                self._last_top = (table, srt, plates, target)
                now = time.monotonic()
                with self.lock:
                    arm_busy = self.L["step"] not in ("survey", "verify", "idle", "done", "paused")
                    self.perc.update(t=now, table=[b.to_dict() for b in table], sorted=[b.to_dict() for b in srt],
                                     plates=[vars(p) for p in plates],
                                     target=None if target is None else target.to_dict())
                    if not arm_busy:  # the arm itself can hide bricks: only count while it is home
                        self._count_hist.append((now, len(table)))
                    step = self.L["step"]
                self.perc["workspace"] = list(LCFG.workspace)
            if self.phase != "inspecting" and detect_inspect:
                insp = getattr(self.real_cameras.get("inspect"), "frame", None)
                if insp is not None:
                    self._perceive_inspect(insp.copy())

        def _perceive_inspect(self, frame) -> None:
            import cv2

            dets = self.detector.detect(frame) if self.detector is not None else []
            h, w = frame.shape[:2]
            best, frac = self._held_brick(frame, dets)
            close = False
            if best is not None:
                x0, y0, x1, y1 = best.box
                cx = (x0 + x1) / 2
                close = (max(x1 - x0, y1 - y0) >= LCFG.held_min_px
                         and LCFG.held_band[0] * w <= cx <= LCFG.held_band[1] * w)
            with self.lock:
                if best is not None:
                    self.perc["inspect_seen"] = {"t": time.monotonic(), "conf": round(float(best.confidence), 2),
                                                 "close": close, "box": [int(v) for v in best.box],
                                                 "fg": round(frac, 2)}
                self.perc["inspect"] = {"held": None if best is None else [int(v) for v in best.box],
                                        "close": close, "band": list(LCFG.held_band),
                                        "others": [[int(v) for v in d.box] for d in dets if d is not best],
                                        "t": time.monotonic()}

        # ------------------------------------------------------------------ held-brick selection (no training)
        def _capture_inspect_reference(self) -> None:
            f = getattr(self.real_cameras.get("inspect"), "frame", None)
            if f is not None:
                self._insp_ref = P.inspect_reference(f)

        def _held_brick(self, bgr, dets):
            return P.select_held(bgr, dets, self._insp_ref, LCFG.fg_frac)

        def _crop_box_held(self, bgr):
            """Replaces YoloInspector.crop_box during the spin: the held brick anywhere in the frame, else the ROI."""
            det = getattr(self.auto_inspector.insp, "detector", None)
            if det is not None:
                best, _ = self._held_brick(bgr, det.detect(bgr))
                if best is not None:
                    return tuple(int(v) for v in best.box), "held"
            return self._roi_crop_box(bgr)

        def _on_inspect_frame(self, ev: dict) -> None:
            import cv2

            img = ev["frame"].copy()
            label = ev.get("confident") or f"{ev.get('top')}?"
            col = {"none": (80, 220, 80), "circle": (36, 160, 240), "cross": (200, 100, 220)}.get(ev.get("confident"),
                                                                                                (180, 180, 180))
            if ev.get("box"):
                x0, y0, x1, y1 = (int(v) for v in ev["box"])
                cv2.rectangle(img, (x0, y0), (x1, y1), col, 3)
            _banner(img, f"SPIN  frame {ev['i'] + 1}  roll {ev['angle']:+.0f} deg   YOLO: {_label(label)} "
                         f"{ev.get('conf', 0):.2f}   {ev.get('device', '')} {ev.get('model_ms', 0):.1f} ms")
            self.perc_streams["perc_inspect"].publish(img)
            rec = {"i": ev["i"], "angle": round(ev["angle"]), "top": ev.get("top"), "conf": round(float(ev.get("conf", 0)), 2),
                   "confident": ev.get("confident"), "ms": round(float(ev.get("model_ms", 0) or 0), 1)}
            with self.lock:
                self._insp_frames.append(rec)
                if ev.get("crop") is not None:
                    self._insp_crops.append((ev["crop"], rec))

        def strip_jpeg(self) -> bytes | None:
            import cv2

            with self.lock:
                crops = list(self._insp_crops)[-12:]
            if not crops:
                return None
            tiles = []
            for crop, rec in crops:
                t = cv2.resize(crop, (112, 112))
                t = cv2.copyMakeBorder(t, 0, 22, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
                cv2.putText(t, f"{_label(rec['confident'] or rec['top'] + '?')[:7]} {rec['conf']:.2f}", (3, 128),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
                tiles.append(t)
            ok, buf = cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buf.tobytes() if ok else None

        def snapshot(self) -> dict:
            d = super().snapshot()
            with self.lock:
                L = self.L
                d["loop"] = {k: L[k] for k in ("mode", "step", "cycle", "attempts", "target", "message", "tallies",
                                                "grasp", "started")}
                d["loop"]["step_s"] = round(self._elapsed(), 1)
                d["loop"]["cycles"] = L["cycles"][-12:]
                d["loop"]["events"] = list(L["events"])[:12]
                d["loop"]["inspect_frames"] = self._insp_frames[-12:]
                d["perception"] = {k: v for k, v in self.perc.items() if k != "t"}
                d["collect"] = {"label": self.collect_label, "counts": self.collect.counts(), "message": self.collect_msg,
                                "busy": self.phase == "collecting"}
                d["inspect_pose"] = {"pose": np.round(self.ip_pose, 1).tolist(), "source": self.ip_src,
                                     "joints": IP.JOINTS, "message": self.ip_msg, "mode": LCFG.bring_mode,
                                     "enabled": LCFG.bring_mode in ("fixed", "vla_pose"),
                                     "offset": np.round(self.ip_offset, 1).tolist(), "offset_source": self.ip_off_src,
                                     "hold": self.tune_hold, "at_camera": self.L["step"] == "tune",
                                     "vla_end": None if self._bring_base is None else np.round(self._bring_base, 1).tolist()}
                d["perception"]["age_s"] = round(time.monotonic() - self.perc["t"], 1) if self.perc["t"] else None
                ins = self.perc.get("inspect")
                if ins:
                    d["perception"]["inspect"] = {**{k: v for k, v in ins.items() if k != "t"},
                                                  "age_s": round(time.monotonic() - ins["t"], 1)}
                d["perception"].pop("inspect_seen", None)
                d["loop"]["config"] = {"hold": [LCFG.hold_min, LCFG.hold_max], "lost_below": LCFG.lost_below,
                                       "open_above": LCFG.open_above}
            d["studio"] = STUDIO
            return d

        def stop(self) -> None:
            self._perc_stop.set()
            if hasattr(super(), "stop"):
                super().stop()

    return LoopController


def _make_top_detector():
    """brick-detector-v1 (GitHub release brick-auto-ensemble-v1, detector_openvino_model) for the top camera, at
    LCFG.verify_conf so the survey (top_conf) and the final check (verify_conf) filter the same detections.
    Plain OpenVINO on the NPU when this dlsj checkout has it (releases the GIL; ultralytics in-process starves the
    servo reads), else ultralytics on the CPU. None -> the loop falls back to the inspection detector, full frame."""
    path = Path(LCFG.top_detector).expanduser()
    try:
        try:
            from dlsj.inspect.brick_detector import make_brick_detector
        except ImportError:
            from dlsj.inspect.brick_detector import BrickDetector

            det = BrickDetector(path / "best.pt" if path.is_dir() else path, "cpu", LCFG.verify_conf)
        else:
            det = make_brick_detector(path, LCFG.top_device, LCFG.verify_conf)
        log.info("top detector: %s on %s, conf %.2f, tiles %s", path, getattr(det, "device", "?"), LCFG.verify_conf,
                 P.tile_grid(LCFG.workspace))
        return det
    except Exception:  # noqa: BLE001
        log.exception("no top detector; using the inspection detector on the full top frame")
        return None


def _label(outcome) -> str:
    return {"none": "plain", "circle": "circle", "cross": "cross", "unknown": "uncertain"}.get(outcome, str(outcome))


def _banner(img, text: str) -> None:
    import cv2

    cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, text, (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def wrap_handler(make_handler):
    """The console handler + the loop dashboard at / (old console at /classic)."""
    from loop_page import PAGE

    def handler(streams, ctl):
        base = make_handler(streams, ctl)

        class LoopHandler(base):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if self.path == "/classic":
                    self.path = "/"
                    return super().do_GET()
                if self.path == "/collect":
                    return self._send(200, C.PAGE.encode(), "text/html; charset=utf-8")
                if self.path == "/collect/state":
                    body = {"counts": ctl.collect.counts(), "entries": ctl.collect.view()}
                    return self._send(200, json.dumps(body).encode(), "application/json")
                if self.path.startswith("/collect/img?"):
                    import re
                    from urllib.parse import parse_qs, urlsplit

                    q = parse_qs(urlsplit(self.path).query)
                    d, f = Path(q.get("d", [""])[0]).resolve(), q.get("f", [""])[0]
                    allowed = Path("outputs/auto_inspect").resolve()
                    if re.fullmatch(r"crop_\d{2}\.jpg", f) and allowed in d.parents and (d / f).is_file():
                        return self._send(200, (d / f).read_bytes(), "image/jpeg")
                    return self._send(404, b"", "text/plain")
                if self.path.startswith("/loop/strip.jpg"):
                    body = ctl.strip_jpeg()
                    return self._send(200, body, "image/jpeg") if body else self._send(204, b"", "text/plain")
                return super().do_GET()

            def do_POST(self):
                if self.path.startswith("/collect/"):
                    parts = self.path.removeprefix("/collect/").split("/")
                    try:
                        if parts[0] == "set" and len(parts) == 2:
                            ctl.collect_command("label", parts[1])
                        elif parts[0] == "spin":
                            ctl.collect_command("spin")
                        elif parts[0] == "toggle" and len(parts) == 3:
                            ctl.collect.update(parts[1], toggle=int(parts[2]))
                        elif parts[0] == "label" and len(parts) == 3:
                            ctl.collect.update(parts[1], label=parts[2])
                        elif parts[0] == "delete" and len(parts) == 2:
                            ctl.collect.update(parts[1], delete=True)
                        else:
                            return self._send(404, b"", "text/plain")
                    except (KeyError, ValueError):
                        return self._send(400, b"", "text/plain")
                    return self._send(200, b"{}", "application/json")
                if self.path.startswith("/pose/"):
                    parts = self.path.removeprefix("/pose/").split("/")
                    if parts[0] in ("goto", "save", "reset", "grip_close", "grip_open", "off_save", "off_reset", "spin",
                                    "hold_on", "hold_off") or \
                            (parts[0] in ("nudge", "off_nudge") and len(parts) == 3 and parts[1] in IP.JOINTS):
                        try:
                            ctl.pose_command(parts[0], *parts[1:])
                        except ValueError:
                            return self._send(400, b"", "text/plain")
                        return self._send(200, b"{}", "application/json")
                    return self._send(404, b"", "text/plain")
                if self.path.startswith("/loop/"):
                    name = self.path.removeprefix("/loop/")
                    if name in ("start", "pause", "resume", "stop", "home"):
                        ctl.loop_command(name)
                        return self._send(200, b"{}", "application/json")
                    return self._send(404, b"", "text/plain")
                return super().do_POST()

        return LoopHandler

    return handler


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--loop-help", action="store_true")
    ap.add_argument("--hold-min", type=float, default=LCFG.hold_min, help="gripper reading when holding a brick: min")
    ap.add_argument("--hold-max", type=float, default=LCFG.hold_max)
    ap.add_argument("--lost-below", type=float, default=LCFG.lost_below, help="gripper below this = empty/dropped")
    ap.add_argument("--open-above", type=float, default=LCFG.open_above, help="gripper above this = released")
    ap.add_argument("--pick-timeout", type=float, default=LCFG.pick_timeout)
    ap.add_argument("--bring-timeout", type=float, default=LCFG.bring_timeout)
    ap.add_argument("--place-timeout", type=float, default=LCFG.place_timeout)
    ap.add_argument("--max-cycles", type=int, default=LCFG.max_cycles)
    ap.add_argument("--max-attempts", type=int, default=LCFG.max_attempts)
    ap.add_argument("--workspace", default="0,60,560,480", help="x0,y0,x1,y1 of the top image the loop looks at")
    ap.add_argument("--empty-seconds", type=float, default=LCFG.empty_seconds)
    ap.add_argument("--held-min-px", type=int, default=LCFG.held_min_px,
                    help="held brick at least this big (px) on the inspection camera = close enough to spin")
    ap.add_argument("--top-detector", default=LCFG.top_detector,
                    help="brick detector for the top camera (release brick-auto-ensemble-v1 detector)")
    ap.add_argument("--top-device", default=LCFG.top_device)
    ap.add_argument("--top-conf", type=float, default=LCFG.top_conf, help="survey: brick confidence threshold")
    ap.add_argument("--verify-conf", type=float, default=LCFG.verify_conf,
                    help="final empty-table check: lower threshold, the brick must persist in >= half the frames")
    ap.add_argument("--verify-frames", type=int, default=LCFG.verify_frames)
    ap.add_argument("--inspect-device", default="NPU")
    ap.add_argument("--inspect-roi", default="")
    ap.add_argument("--inspect-frames", type=int, default=12)
    ap.add_argument("--inspect-range", type=float, default=180.0)
    ap.add_argument("--inspect-speed", type=float, default=60.0)
    ap.add_argument("--bring-mode", choices=("vla", "vla_pose", "vla_offset", "fixed"), default="vla",
                    help="vla (default): the VLA brings the brick to the camera, spin when it settles; vla_pose: then straight to the saved inspection pose; vla_offset: VLA "
                         "bring + saved offset; fixed: scripted move via the ready pose")
    ours, rest = ap.parse_known_args()
    LCFG.bring_mode = ours.bring_mode
    if ours.loop_help:
        ap.print_help()
        return
    for k in ("hold_min", "hold_max", "lost_below", "open_above", "pick_timeout", "bring_timeout", "place_timeout",
              "max_cycles", "max_attempts", "empty_seconds", "held_min_px", "top_detector", "top_device", "top_conf",
              "verify_conf", "verify_frames"):
        setattr(LCFG, k, getattr(ours, k))
    LCFG.workspace = tuple(int(v) for v in ours.workspace.split(","))
    A.CFG.__dict__.update(inspect_device=ours.inspect_device, inspect_frames=ours.inspect_frames,
                          inspect_range=ours.inspect_range, inspect_speed=ours.inspect_speed,
                          roi=tuple(int(v) for v in ours.inspect_roi.split(",")) if ours.inspect_roi else None)
    S._manual_make_controller_class = S.make_controller_class
    S.make_controller_class = make_loop_controller_class
    orig_home = S.add_home_button

    def add_home_and_loop(pc):
        orig_home(pc)
        pc.make_handler = wrap_handler(pc.make_handler)

    S.add_home_button = add_home_and_loop
    STUDIO.update(studio_info(rest))
    if STUDIO:
        print(f"policy: Physical AI Studio model '{STUDIO['name']}' ({STUDIO['policy']}, id {STUDIO['model_id']}, "
              f"trained on '{STUDIO.get('dataset_name')}')", flush=True)  # logging is configured later, in S.main
    sys.argv = [sys.argv[0], *rest]
    S.main()


if __name__ == "__main__":
    main()
