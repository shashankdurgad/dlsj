"""Spin-and-inspect inside the VLA controller: YOLO v2 + brick detector on the NPU, the same sweep the sweep page runs.

Used by `smolvla_control.py` (SmolVLA / MolmoAct2 on Desmond's console): when the operator presses M with the brick
held at the inspection camera, the controller pauses the policy and calls `AutoInspector.inspect(cmd)`, which

1. rolls the wrist through the stops of `dlsj.loop.sweep.Sweeper` (calibrated wrist_roll limits, speed cap, settle,
   grasp check every tick, wrist returned to its start angle), classifying each `inspect` frame while the wrist moves on;
2. ends early once the evidence is decisive (`dlsj.inspect.aggregate.early_stop`), else aggregates all frames with
   `yolo_topk` (marked: >= 3 confident mark frames; plain: >= 50% confident plain frames);
3. maps the outcome to a bowl: plain -> yellow, circle -> orange, cross -> purple; uncertain -> one re-spin, then purple;
4. saves the sweep like the sweep page (frames, crops, decision) under `out_root`.

It never touches the policy; the caller holds the arm still (policy paused) for the duration. Everything heavy is
injected so it can be tested with the sweep page's fakes.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger("auto_inspect")

BOWL = {"none": "yellow", "circle": "orange", "cross": "purple"}
FALLBACK_BOWL = "purple"  # uncertain after the re-spin: treat as defective (PLAYBOOK REASON rule)


@dataclass
class InspectResult:
    outcome: str  # "none" (plain) | "circle" | "cross" | "unknown"
    bowl: str
    reason: str
    status: str  # "ok" | "grasp_lost" | "aborted" | "camera_error" | "error"
    attempts: int
    frames: int
    early_stop: bool
    seconds: float
    saved: list[str] = field(default_factory=list)


class ArmAdapter:
    """`pi05_control.FollowerArm` (read()/send()) -> the robot interface `Sweeper` uses."""

    def __init__(self, arm):
        self.arm = arm

    def read_state(self) -> np.ndarray:
        return np.asarray(self.arm.read(), np.float32)

    def send(self, target) -> None:
        self.arm.send(np.asarray(target, dtype=float))

    def read_gripper_current(self):
        return None


class AutoInspector:
    def __init__(self, cameras, inspector, limits=None, *, frames: int = 12, range_deg: float = 180.0,
                 speed_deg_s: float = 60.0, settle_ms: float = 150.0, fps: float = 30.0, early_stop: bool = True,
                 max_attempts: int = 2, grasp_empty_pos: float = 4.0, out_root: str | Path = "outputs/auto_inspect"):
        self.cameras, self.insp, self.limits = cameras, inspector, limits
        self.frames, self.range_deg, self.speed, self.settle_ms, self.fps = frames, range_deg, speed_deg_s, settle_ms, fps
        self.early_stop, self.max_attempts, self.grasp_empty_pos = early_stop, max_attempts, grasp_empty_pos
        self.out_root = Path(out_root).expanduser()
        self.abort = threading.Event()
        self.listener = None  # optional callable(event: dict): live per-frame results for a dashboard
        self.continuous = False  # roll without stopping and classify every camera frame (~1 per degree)
        self.continuous_speed = 30.0  # deg/s; at 30 fps that is one frame per degree

    # ------------------------------------------------------------------ one sweep
    def _sweep_once(self, robot, start_cmd: np.ndarray):
        from dlsj.inspect import aggregate as agg
        from dlsj.loop.sweep import SweepConfig, Sweeper

        frames = self.frames
        extra = {}
        if self.continuous:  # a planned angle per camera frame: range / (speed / fps)
            frames = int(self.range_deg / max(1e-3, self.continuous_speed / self.fps)) + 1
            extra = {"continuous": True, "continuous_speed_deg_s": self.continuous_speed}
        cfg = SweepConfig(frames=frames, range_deg=self.range_deg, settle_ms=self.settle_ms,
                          max_speed_deg_s=min(self.speed, 120.0), limits=self.limits,
                          grasp_empty_pos=self.grasp_empty_pos, camera="inspect", fps=self.fps, return_to_start=True,
                          **extra)
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspect-infer")
        futures, crops = [], {}

        self._crop_info = {}

        def infer(frame, i, angle):
            c = self.insp.classify(frame)
            crops[i] = c["crop"]
            self._crop_info[i] = {"angle": round(float(angle), 1), "top": c.get("top"),
                                  "conf": round(float(c.get("conf") or 0), 3), "confident": c.get("confident")}
            if self.listener is not None:
                try:
                    self.listener({"kind": "frame", "i": i, "angle": float(angle), "frame": frame, **c})
                except Exception:  # noqa: BLE001  a dashboard must never break the sweep
                    log.exception("inspection listener failed")
            return self.insp.to_frame(i, angle, c)

        def on_frame(frame, angle):
            futures.append(pool.submit(infer, frame, len(futures), angle))

        def should_stop() -> bool:
            done = [f.result(timeout=5.0) for f in futures]
            return agg.early_stop(done, self.insp.agg) is not None

        import inspect as _inspect

        # older dlsj checkouts (e.g. ~/inference/dlsj-smolvla) have no early stop: full sweep, same decision rule
        can_stop = (self.early_stop and hasattr(agg, "early_stop")
                    and "should_stop" in _inspect.signature(Sweeper.run).parameters)
        kw = {"should_stop": should_stop} if can_stop else {}
        cams = _LatestFrame(self.cameras) if self.continuous else self.cameras
        try:
            sres = Sweeper(robot, cams, cfg, self.abort).run(start_cmd.copy(), on_frame=on_frame, **kw)
            recs = [f.result(timeout=10.0) for f in futures]
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        res = None
        if recs and sres.status == "ok":
            res = agg.aggregate(recs, 0.5, 1 - self.insp.min_conf, self.insp.min_conf, scaled_agg(self.insp.agg, len(recs)),
                                "yolo_topk")
        return sres, recs, crops, res

    def _save(self, attempt: int, sres, recs, crops, res) -> str | None:
        try:
            import cv2

            from dlsj.loop.sweep_test import save_sweep

            out = self.out_root / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"_a{attempt}")
            n = len(sres.frames)
            keep = sorted({round(k * (n - 1) / 11) for k in range(12)}) if n > 12 else list(range(n))
            sub = (lambda xs: [xs[k] for k in keep if k < len(xs)]) if n > 12 else (lambda xs: xs)
            save_sweep(out, sub(sres.frames), sub(sres.angles), None, sres if n <= 12 else None, None,
                       (sub(recs) if recs else None), None, None,
                       {"source": "auto_inspect", "outcome": None if res is None else res.outcome().value,
                        "reason": None if res is None else res.summary.get("reason"), "frames_total": n,
                        "continuous": self.continuous})
            meta = []
            for i, c in sorted(crops.items()):
                cv2.imwrite(str(out / f"crop_{i:03d}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
                r = self._crop_info.get(i, {})
                meta.append({"i": i, **r})
            (out / "crops.json").write_text(__import__("json").dumps(meta))
            return str(out)
        except Exception as e:  # noqa: BLE001  saving must never break the run
            log.warning("could not save the sweep: %s", e)
            return None

    # ------------------------------------------------------------------ public
    def inspect(self, arm, start_cmd) -> InspectResult:
        """Blocking. `arm`: read()/send() (FollowerArm) or read_state()/send(). Returns the bowl to place in."""
        robot = arm if hasattr(arm, "read_state") else ArmAdapter(arm)
        t0 = time.perf_counter()
        self.abort.clear()
        saved, frames, early = [], 0, False
        cmd = np.asarray(start_cmd, np.float32)
        for attempt in range(1, self.max_attempts + 1):
            try:
                sres, recs, crops, res = self._sweep_once(robot, cmd)
            except Exception as e:  # noqa: BLE001
                log.exception("inspection sweep failed")
                return InspectResult("unknown", FALLBACK_BOWL, f"error: {e}", "error", attempt, frames, early,
                                     time.perf_counter() - t0, saved)
            frames += len(recs)
            early = early or bool(getattr(sres, "stopped_early", False))
            if (p := self._save(attempt, sres, recs, crops, res)):
                saved.append(p)
            if sres.status != "ok":
                return InspectResult("unknown", FALLBACK_BOWL, f"sweep {sres.status}", sres.status, attempt, frames,
                                     early, time.perf_counter() - t0, saved)
            outcome = res.outcome().value if res is not None else "unknown"
            if outcome in BOWL:
                return InspectResult(outcome, BOWL[outcome], res.summary.get("reason", ""), "ok", attempt, frames,
                                     early, time.perf_counter() - t0, saved)
            log.info("inspection uncertain (attempt %d): %s", attempt, res.summary.get("reason") if res else "")
        return InspectResult("unknown", FALLBACK_BOWL, "uncertain after re-inspection -> defective (purple)", "ok",
                             self.max_attempts, frames, early, time.perf_counter() - t0, saved)

    def stop(self) -> None:
        self.abort.set()


def _ov_dir(d: Path) -> Path:
    """Ultralytics picks its OpenVINO backend (where "npu" = the Intel NPU, not Ascend) only for *_openvino_model."""
    link = d.with_name(d.name + "_openvino_model")
    if not link.exists():
        link.symlink_to(d.name)
    return link


class HeldCubeInspector:
    """Joe's held-cube YOLO26n v2 detector (release held-cube-yolo26n-v2-20261115) as the spin classifier.

    One detector finds the cube held in the gripper and its visible symbol (held_cross / held_circle / held_plain) on
    the full inspection frame, so no crop selection is needed and cubes lying on the table are ignored by design.
    Each frame becomes the same dict YoloInspector.classify returns (probs over plain/circle/cross, confident class,
    box, 320 px crop for the dashboard / data collection), so the decision rule, the loop and collection are unchanged.
    Rig check, 38 labelled spins: v4 epoch 4 37 right / 0 uncertain / 1 wrong, v2 36 / 1 / 1 (YOLO v2 classifier live:
    29 / 5 / 4), 4.1 ms per frame on the NPU.
    """

    NAMES = {0: "cross", 1: "circle", 2: "plain"}

    def __init__(self, weights, device: str = "NPU", conf: float = 0.5, detector=None):
        from dlsj.inspect.aggregate import AggregateConfig
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="detect")
        self.device, self.min_conf, self.min_margin = device, conf, 0.0
        # the decision rule re-gates every frame with agg.yolo_min_conf: v2's 0.8 suits a classifier's softmax, but the
        # detector's confidences on held bricks sit at ~0.5-0.8, so with 0.8 most confident frames counted as unsure and
        # every brick came out uncertain -> purple (live, 15 Nov). Use the detector's own bar (0.5, Joe's demo setting).
        self.agg = AggregateConfig(strategy="yolo_topk", yolo_min_conf=conf, yolo_min_margin=0.2)
        self.detector = detector  # the top-camera brick detector the loop's perception uses
        self.model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=640, conf=0.05, device=device.lower(),
                           verbose=False)  # compile once, on the chosen device

    def classify(self, frame_rgb: np.ndarray) -> dict:
        from dlsj.inspect.brick_detector import inspection_crop
        from dlsj.inspect.yolo_inspector import default_roi

        t0 = time.perf_counter()
        bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
        t1 = time.perf_counter()
        r = self.model.predict(bgr, imgsz=640, conf=0.05, device=self.device.lower(), verbose=False)[0]
        t2 = time.perf_counter()
        h, w = bgr.shape[:2]
        if len(r.boxes):
            i = int(r.boxes.conf.argmax())
            top, conf = self.NAMES[int(r.boxes.cls[i])], float(r.boxes.conf[i])
            box = tuple(int(v) for v in r.boxes.xyxy[i].tolist())
            probs = {c: (1.0 - conf) / 2 for c in ("plain", "circle", "cross")}
            probs[top] = conf
            source = "held_cube"
        else:  # no held cube found: an unsure frame
            top, conf, box, source = "plain", 0.0, default_roi(w, h), "none"
            probs = {c: 1 / 3 for c in ("plain", "circle", "cross")}
        try:
            crop, _ = inspection_crop(bgr, box, padding=0.05, size=320)
        except ValueError:
            crop = np.zeros((320, 320, 3), np.uint8)
        ranked = sorted(probs.values(), reverse=True)
        return {"probs": probs, "confident": top if conf >= self.min_conf else None, "top": top, "conf": conf,
                "margin": ranked[0] - ranked[1], "box": box, "source": source, "crop": crop,
                "ms": (time.perf_counter() - t0) * 1e3, "model_ms": (t2 - t1) * 1e3, "device": self.device}

    @staticmethod
    def to_frame(index, angle, c):
        from dlsj.inspect.yolo_inspector import YoloInspector

        return YoloInspector.to_frame(index, angle, c)


class HeldCropInspector:
    """Two stages: the held-cube detector draws the box around the cube held in the gripper (it ignores cubes on the
    table), and the YOLO v2 classifier classifies ONLY that crop (same padding / 320 px / JPEG round trip / confidence
    gate as v2 was trained and tuned with). A frame with no held cube is an unsure frame: nothing in the background
    is ever classified. Same per-frame dict as YoloInspector.classify, so the loop and decision rule are unchanged."""

    def __init__(self, held_weights, classifier_dir, device: str = "NPU", held_conf: float = 0.25, padding: float = 0.05,
                 detector=None):
        from dlsj.inspect.yolo_inspector import YoloInspector
        from ultralytics import YOLO

        self.held = YOLO(str(held_weights), task="detect")
        self.cls = YoloInspector(classifier_dir, device, padding=padding)  # classifier only, no crop selection
        self.device, self.held_conf = device, held_conf
        self.min_conf, self.min_margin, self.agg = self.cls.min_conf, self.cls.min_margin, self.cls.agg
        self.detector = detector  # the top-camera brick detector the loop's perception uses
        self.held.predict(np.zeros((480, 640, 3), np.uint8), imgsz=640, conf=held_conf, device=device.lower(),
                          verbose=False)

    def held_box(self, bgr):
        r = self.held.predict(bgr, imgsz=640, conf=self.held_conf, device=self.device.lower(), verbose=False)[0]
        if not len(r.boxes):
            return None, 0.0
        i = int(r.boxes.conf.argmax())
        return tuple(int(v) for v in r.boxes.xyxy[i].tolist()), float(r.boxes.conf[i])

    def classify(self, frame_rgb: np.ndarray) -> dict:
        from dlsj.inspect.aggregate import YOLO_CLASSES, gate
        from dlsj.inspect.brick_detector import inspection_crop
        from dlsj.inspect.yolo_inspector import default_roi, jpeg_roundtrip

        t0 = time.perf_counter()
        bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
        box, hconf = self.held_box(bgr)
        t1 = time.perf_counter()
        if box is None:  # no held cube: unsure, and never a background crop
            h, w = bgr.shape[:2]
            roi = default_roi(w, h)
            crop, _ = inspection_crop(bgr, roi, padding=0.0, size=320)
            probs = {c: 1 / 3 for c in YOLO_CLASSES}
            return {"probs": probs, "confident": None, "top": "plain", "conf": 0.0, "margin": 0.0, "box": roi,
                    "source": "no_held_cube", "crop": crop, "ms": (time.perf_counter() - t0) * 1e3,
                    "model_ms": (t1 - t0) * 1e3, "device": self.device, "held_conf": 0.0}
        crop, bounds = inspection_crop(bgr, box, padding=self.cls.padding, size=320)
        crop = jpeg_roundtrip(crop, self.cls.jpeg_quality)
        t2 = time.perf_counter()
        p = self.cls.classifier.predict(crop)
        t3 = time.perf_counter()
        probs = {c: float(p[c]) for c in YOLO_CLASSES}
        confident, top, conf, margin = gate(probs, self.min_conf, self.min_margin)
        return {"probs": probs, "confident": confident, "top": top, "conf": conf, "margin": margin,
                "box": tuple(int(v) for v in bounds), "source": "held_crop", "crop": crop,
                "ms": (time.perf_counter() - t0) * 1e3, "model_ms": (t1 - t0 + t3 - t2) * 1e3, "device": self.device,
                "held_conf": hconf}

    @staticmethod
    def to_frame(index, angle, c):
        from dlsj.inspect.yolo_inspector import YoloInspector

        return YoloInspector.to_frame(index, angle, c)


class _LatestFrame:
    """Continuous spins: `fresh(name, 1)` returns the newest camera frame at once instead of waiting ~33 ms for the
    next one, so the roll never pauses between frames (the sweep only ticks the wrist when it isn't grabbing)."""

    def __init__(self, cameras):
        self.cameras = cameras

    def fresh(self, name, n, timeout_s: float = 2.0):
        s = getattr(self.cameras, "streams", {}).get(name)
        if n == 1 and s is not None and getattr(s, "frame", None) is not None:
            with s.cond:
                f = s.frame
            return [np.ascontiguousarray(f[..., ::-1])]  # BGR -> RGB, like StreamCameras.fresh
        return self.cameras.fresh(name, n, timeout_s)

    def __getattr__(self, k):
        return getattr(self.cameras, k)


def scaled_agg(cfg, n: int, base: int = 12):
    """The yolo_topk counts were tuned on 12 frames: scale "marked needs >= 2" and "plain allows <= 1 outlier" with
    the frame count (a continuous spin has ~180), so a dense spin is judged like a 12-stop one."""
    import dataclasses
    import math

    if n <= base or not dataclasses.is_dataclass(cfg):
        return cfg
    k = n / base
    return dataclasses.replace(cfg, yolo_min_marked_frames=math.ceil(cfg.yolo_min_marked_frames * k),
                               yolo_max_outlier_frames=math.floor(cfg.yolo_max_outlier_frames * k))


# held-cube detectors from Joe's GitHub releases, exported to OpenVINO under ~/inference/models
HELD_CUBE_WEIGHTS = {
    "held_cube": "held-cube-yolo26n-v4/held_cube_symbols_cross_v4_epoch4_openvino_model",  # v4 epoch 4 (default)
    "held_cube_v2": "held-cube-yolo26n-v2/held_cube_symbols_held_only_openvino_model",
}
# Joe's held-brick detector v1 (release held-brick-detector-v1-20261115): a later v4 checkpoint used detector-only
# (any symbol channel = held brick, confidence 0.2 chosen on validation; its wrapper merges overlapping boxes, so the
# best box is simply the highest-confidence one). Stage 1 of the held-crop pipeline.
HELD_BRICK_WEIGHTS, HELD_BRICK_CONF = "held-brick-detector-v1/held_brick_backbone_openvino_model", 0.2


def make_auto_inspector(cameras, follower_id: str = "follower", device: str = "NPU", roi=None,
                        kind: str = "held_cube", **kw) -> AutoInspector:
    """The production inspector on `device`, calibrated roll limits. kind: "held_cube" (Joe's held-cube YOLO26n v4
    detector, default), "held_cube_v2", or "yolo_v2" (YOLO v2 classifier on a detector/ROI crop)."""
    from dlsj import arms
    from dlsj.inspect.yolo_inspector import YoloInspector

    models = Path("~/inference/models").expanduser()
    if kind == "held_crop":  # held-cube v4 box -> YOLO v2 classifier on that crop only
        from dlsj.inspect.brick_detector import BrickDetector

        top_det = BrickDetector(str(_ov_dir(models / "brick-detector-v1")), device.lower())
        insp = HeldCropInspector(models / HELD_BRICK_WEIGHTS, models / "brick-yolo-classifier-v2", device,
                                 held_conf=HELD_BRICK_CONF, detector=top_det)
        try:
            limits = arms.joint_limits_deg(follower_id, "wrist_roll")
        except Exception as e:  # noqa: BLE001
            log.warning("no calibrated wrist_roll limits (%s); sweeping without clipping", e)
            limits = None
        log.info("auto-inspect: held_crop (held-brick detector v1 box -> YOLO v2 classifier) on %s, wrist_roll limits %s",
                 device, limits)
        return AutoInspector(cameras, insp, limits, **kw)
    if kind in HELD_CUBE_WEIGHTS:
        from dlsj.inspect.brick_detector import BrickDetector

        top_det = BrickDetector(str(_ov_dir(models / "brick-detector-v1")), device.lower())
        insp = HeldCubeInspector(models / HELD_CUBE_WEIGHTS[kind], device, detector=top_det)
        try:
            limits = arms.joint_limits_deg(follower_id, "wrist_roll")
        except Exception as e:  # noqa: BLE001
            log.warning("no calibrated wrist_roll limits (%s); sweeping without clipping", e)
            limits = None
        log.info("auto-inspect: %s (%s) on %s, wrist_roll limits %s", kind, HELD_CUBE_WEIGHTS[kind], device, limits)
        return AutoInspector(cameras, insp, limits, **kw)
    insp = YoloInspector(models / "brick-yolo-classifier-v2", device, roi=roi, padding=0.05,
                         detector_weights=str(_ov_dir(models / "brick-detector-v1")), detector_device=device.lower())
    try:
        limits = arms.joint_limits_deg(follower_id, "wrist_roll")
    except Exception as e:  # noqa: BLE001
        log.warning("no calibrated wrist_roll limits (%s); sweeping without clipping", e)
        limits = None
    log.info("auto-inspect: YOLO on %s, detector %s, wrist_roll limits %s", insp.device,
             getattr(insp.detector, "device", None), limits)
    return AutoInspector(cameras, insp, limits, **kw)
