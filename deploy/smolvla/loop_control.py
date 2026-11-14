"""Closed-loop brick sorting with a perception dashboard: the run-time loop for demos and judging.

PERCEIVE -> DETECT -> REASON -> ACT, repeated until the table is empty:

    home --> survey (top camera: bricks left? which colour next?) --> pick (VLA "pick up the <colour> brick")
         --> grasp confirmed (gripper squeezing an object at brick width) --> bring (VLA "bring the brick to the
         inspection camera") --> arrived (arm settled, inspection camera sees the brick) --> spin + YOLO on the NPU
         (`auto_inspect.py`) --> plate decided --> place (VLA "place the brick in the <plate> bowl")
         --> released (gripper open) --> home --> survey ... --> no bricks left: stay home, done.

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
import perception as P  # noqa: E402
import smolvla_control as S  # noqa: E402

log = logging.getLogger("loop_control")
LCFG = SimpleNamespace(hold_min=4.0, hold_max=9.0, lost_below=3.0, open_above=12.0, pick_timeout=45.0,
                       bring_timeout=25.0, place_timeout=40.0, max_cycles=12, max_attempts=3,
                       workspace=(0, 60, 560, 480), empty_seconds=2.0, perception_hz=4.0, held_min_px=150,
                       held_band=(0.25, 0.75), fg_frac=0.35,
                       log_dir="~/inference/logs/loop")
STEPS = ["survey", "pick", "bring", "inspect", "place", "home"]


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
            self.auto_inspector.listener = self._on_inspect_frame
            self._insp_ref = None  # blurred grey inspection frame with the arm away (taken while surveying)
            insp = getattr(self.auto_inspector, "insp", None)
            if insp is not None and hasattr(insp, "crop_box"):
                self._roi_crop_box = insp.crop_box
                insp.crop_box = self._crop_box_held  # YOLO crops the HELD brick, not bricks on the table behind it
            self.L = {"mode": "off", "step": "idle", "since": time.monotonic(), "cycle": 0, "attempts": 0,
                      "target": None, "message": "Press Start loop", "cycles": [], "tallies": {},
                      "grasp": {"state": "open"}, "started": None, "events": collections.deque(maxlen=40)}
            self.perc = {"t": 0.0, "table": [], "sorted": [], "plates": [], "target": None, "inspect_seen": None,
                         "inspect": None, "workspace": list(LCFG.workspace)}
            self._plates: dict[str, P.Plate] = {}
            self._count_hist: collections.deque = collections.deque(maxlen=64)  # (t, n bricks on the table)
            self._pose_hist: collections.deque = collections.deque(maxlen=20)
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
                    self.L.update(mode="running", cycle=0, attempts=0, cycles=[], tallies={}, started=time.time())
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
                    return
                try:
                    getattr(self, f"_step_{self.L['step']}", lambda: None)()
                except Exception as e:  # noqa: BLE001  never leave the loop running blind
                    log.exception("loop step failed")
                    self._pause(f"Loop error: {e}")

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

        def _step_survey(self) -> None:
            if self._elapsed() < 1.0:  # let the arm settle
                return
            from_home = self.L.get("survey_from", "home") == "home"
            since = self.L["since"] + 0.8
            recent = [n for t, n in self._count_hist if t >= since]
            if len(recent) >= max(3, int(LCFG.empty_seconds * LCFG.perception_hz * 0.6)) and max(recent) == 0:
                if not from_home:  # the arm may hide bricks from up here: only the view from home decides
                    return self._go_home("No bricks seen from here: going home to check the whole table")
                self.L["mode"] = "done"
                total = sum(self.L["tallies"].values())
                self._goto("done", f"Table clear: {total} brick(s) sorted. Staying home")
                return
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
                self._goto("bring", "Grasp confirmed: bringing the brick to the inspection camera")
                return
            if self._elapsed() > LCFG.pick_timeout:
                self.L["attempts"] += 1
                self._cycle(result="pick timeout")
                self._go_home(f"No grasp after {LCFG.pick_timeout:.0f} s: going home to retry")

        def _step_bring(self) -> None:
            if self.phase == "stopped":
                return self._pause("Stopped while bringing")
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
            settled = len(self._pose_hist) >= 15 and float(np.ptp(np.stack(self._pose_hist)[:, :5], axis=0).max()) < 1.5
            arrived = (self._elapsed() > 3.0 and self._held_for("settled", settled, 1.5)) or self._elapsed() > LCFG.bring_timeout
            if arrived:
                self._cycle(t_inspect=round(time.time() - self.L["cycles"][-1]["t0"], 1), camera_saw=bool(sees))
                self._insp_frames, self._insp_crops = [], []
                self._start_inspection()  # AutoController: policy paused, wrist spin + YOLO in a thread
                self._goto("inspect", "At the camera: spinning the brick, YOLO on the NPU")

        def _step_inspect(self) -> None:
            if self.phase == "inspecting":
                return
            res = self.inspection or {}
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
            self.L["attempts"] = 0
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
                    self._perceive_once(detect_top=step in ("survey", "idle", "done", "paused"),
                                        detect_inspect=step in ("bring", "idle", "paused"))
                except Exception:  # noqa: BLE001
                    log.exception("perception failed")
                hz = LCFG.perception_hz if step in ("survey", "bring") else 1.5
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
                dets = self.detector.detect(top) if self.detector is not None else []
                table, srt = P.split_bricks(dets, plates, top, LCFG.workspace)
                target = P.choose_target(table)
                self._last_top = (table, srt, plates, target)
                now = time.monotonic()
                with self.lock:
                    arm_busy = self.L["step"] not in ("survey", "idle", "done", "paused")
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
                d["perception"]["age_s"] = round(time.monotonic() - self.perc["t"], 1) if self.perc["t"] else None
                ins = self.perc.get("inspect")
                if ins:
                    d["perception"]["inspect"] = {**{k: v for k, v in ins.items() if k != "t"},
                                                  "age_s": round(time.monotonic() - ins["t"], 1)}
                d["perception"].pop("inspect_seen", None)
                d["loop"]["config"] = {"hold": [LCFG.hold_min, LCFG.hold_max], "lost_below": LCFG.lost_below,
                                       "open_above": LCFG.open_above}
            return d

        def stop(self) -> None:
            self._perc_stop.set()
            if hasattr(super(), "stop"):
                super().stop()

    return LoopController


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
                if self.path.startswith("/loop/strip.jpg"):
                    body = ctl.strip_jpeg()
                    return self._send(200, body, "image/jpeg") if body else self._send(204, b"", "text/plain")
                return super().do_GET()

            def do_POST(self):
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
    ap.add_argument("--inspect-device", default="NPU")
    ap.add_argument("--inspect-roi", default="")
    ap.add_argument("--inspect-frames", type=int, default=12)
    ap.add_argument("--inspect-range", type=float, default=180.0)
    ap.add_argument("--inspect-speed", type=float, default=60.0)
    ours, rest = ap.parse_known_args()
    if ours.loop_help:
        ap.print_help()
        return
    for k in ("hold_min", "hold_max", "lost_below", "open_above", "pick_timeout", "bring_timeout", "place_timeout",
              "max_cycles", "max_attempts", "empty_seconds", "held_min_px"):
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
    sys.argv = [sys.argv[0], *rest]
    S.main()


if __name__ == "__main__":
    main()
