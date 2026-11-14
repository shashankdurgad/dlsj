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

    # ------------------------------------------------------------------ one sweep
    def _sweep_once(self, robot, start_cmd: np.ndarray):
        from dlsj.inspect import aggregate as agg
        from dlsj.loop.sweep import SweepConfig, Sweeper

        cfg = SweepConfig(frames=self.frames, range_deg=self.range_deg, settle_ms=self.settle_ms,
                          max_speed_deg_s=min(self.speed, 120.0), limits=self.limits,
                          grasp_empty_pos=self.grasp_empty_pos, camera="inspect", fps=self.fps, return_to_start=True)
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspect-infer")
        futures, crops = [], {}

        def infer(frame, i, angle):
            c = self.insp.classify(frame)
            crops[i] = c["crop"]
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
        try:
            sres = Sweeper(robot, self.cameras, cfg, self.abort).run(start_cmd.copy(), on_frame=on_frame, **kw)
            recs = [f.result(timeout=10.0) for f in futures]
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        res = None
        if recs and sres.status == "ok":
            res = agg.aggregate(recs, 0.5, 1 - self.insp.min_conf, self.insp.min_conf, self.insp.agg, "yolo_topk")
        return sres, recs, crops, res

    def _save(self, attempt: int, sres, recs, crops, res) -> str | None:
        try:
            import cv2

            from dlsj.loop.sweep_test import save_sweep

            out = self.out_root / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"_a{attempt}")
            save_sweep(out, sres.frames, sres.angles, None, sres, None, recs or None, None, None,
                       {"source": "auto_inspect", "outcome": None if res is None else res.outcome().value,
                        "reason": None if res is None else res.summary.get("reason")})
            for i, c in crops.items():
                cv2.imwrite(str(out / f"crop_{i:02d}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
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


def make_auto_inspector(cameras, follower_id: str = "follower", device: str = "NPU", roi=None, **kw) -> AutoInspector:
    """The production inspector: YOLO v2 classifier + one-class brick detector on `device`, calibrated roll limits."""
    from dlsj import arms
    from dlsj.inspect.yolo_inspector import YoloInspector

    models = Path("~/inference/models").expanduser()
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
