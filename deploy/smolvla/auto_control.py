"""THE RUN-TIME LOOP (demo / judging / end-to-end runs). `smolvla_control.py` is the manual testing path only.

Automatic 3-stage controller: same as `smolvla_control.py` (SmolVLA / MolmoAct2, console :8010, RTC) plus
automatic spin-and-inspect. `smolvla_control.py` stays the manual path (M + 1/2/3 to choose the bowl).

Flow: Space = pick; M (brick grasped) = bring to the inspection camera; M again (brick at the camera) = pause the
policy, spin the wrist in front of `inspect`, classify with YOLO v2 + brick detector on the NPU (`auto_inspect.py`),
then place in the decided bowl automatically (plain -> yellow, circle -> orange, cross -> purple; uncertain -> one
re-spin -> purple). Q during the spin aborts and holds; a dropped brick holds. 1/2/3 still override the bowl.

    cd ~/inference/dlsj-smolvla && PYTHONPATH=$PWD ~/desmond/venvs/pi05rt/bin/python \
        ~/inference/deploy/smolvla/auto_control.py <all smolvla_control.py args> \
        [--inspect-device NPU] [--inspect-roi x0,y0,x1,y1] [--inspect-frames 12] [--inspect-range 180]

Needs the inspect camera (`--realsense top=...,inspect=<serial>`).
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# ultralytics (brick detector) lives in an overlay so Desmond's pi05rt venv stays untouched:
#   uv pip install --python <pi05rt python> --no-deps --target ~/inference/pydeps/ultra \
#       ultralytics==8.4.212 ultralytics-thop polars
_ULTRA = Path("~/inference/pydeps/ultra").expanduser()
if _ULTRA.is_dir():
    sys.path.append(str(_ULTRA))
import smolvla_control as S  # noqa: E402

log = logging.getLogger("auto_control")
CFG = SimpleNamespace()


def make_auto_controller_class(base):
    manual = S._manual_make_controller_class(base)

    class AutoController(manual):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            from auto_inspect import make_auto_inspector
            from dlsj.loop.hardware import StreamCameras

            if "inspect" not in self.cameras:
                raise SystemExit("auto_control needs the inspect camera (--realsense ...,inspect=<serial>)")
            self.inspection = None
            self.auto_inspector = make_auto_inspector(
                StreamCameras(self.cameras), device=CFG.inspect_device, roi=CFG.roi, frames=CFG.inspect_frames,
                range_deg=CFG.inspect_range, speed_deg_s=CFG.inspect_speed, fps=self.fps)

        def _at_camera(self) -> bool:
            return self.phase == "running" and self.stage == ("bring" if S.BRING else "pick")

        def _handle(self, c: str) -> None:
            if self.phase == "inspecting":
                if c in ("q", "stop"):
                    self.auto_inspector.stop()  # the sweep thread holds the arm and reports "aborted"
                    self.message = "Stopping the inspection..."
                return  # nothing else may start the policy mid-spin
            if c in ("start", "space") and self.phase not in ("running", "readying"):
                self.bowl, self.inspection = None, None  # new brick: inspect again (1/2/3 after Space overrides)
            if c in ("mark", "m") and self._at_camera() and self.bowl is None:
                self._start_inspection()
                return
            super()._handle(c)  # pick -> bring, manual bowl override (1/2/3 then M), space, q, +/-

        def _start_inspection(self) -> None:
            self._replan_now()  # in-flight chunks become stale
            self.phase, self.stage = "inspecting", "inspecting"
            self.message = "Inspecting: spinning the brick in front of the inspection camera"
            start_cmd = np.asarray(self.cmd if self.cmd is not None else self.arm.read(), dtype=float).copy()
            self.inspection = {"state": "running", "started": time.time()}
            threading.Thread(target=self._inspection_job, args=(start_cmd,), daemon=True, name="auto-inspect").start()

        def _inspection_job(self, start_cmd) -> None:
            try:
                res = self.auto_inspector.inspect(self.arm, start_cmd)
            except Exception as e:  # noqa: BLE001  never leave the arm without a decision
                log.exception("inspection failed")
                res = SimpleNamespace(outcome="unknown", bowl="purple", reason=f"error: {e}", status="error",
                                      attempts=0, frames=0, early_stop=False, seconds=0.0, saved=[])
            with self.lock:
                self.inspection = {"state": res.status, "outcome": res.outcome, "bowl": res.bowl,
                                   "reason": res.reason, "attempts": res.attempts, "frames": res.frames,
                                   "early_stop": res.early_stop, "seconds": round(res.seconds, 1), "saved": res.saved}
                log.info("inspection: %s", self.inspection)
                if res.status == "ok":
                    self.bowl, self.stage, self.phase = res.bowl, "place", "running"
                    self.cmd = np.asarray(start_cmd, dtype=float)  # the sweep returned the wrist to this pose
                    self._replan_now()
                    self.message = (f"Inspected: {res.outcome} -> {res.bowl} bowl ({res.frames} frames, "
                                    f"{res.seconds:.1f} s). Placing.")
                    return
                try:
                    present = np.asarray(self.arm.read(), dtype=float)
                    self.arm.send(present)
                    self.cmd = present
                except Exception:  # noqa: BLE001
                    log.exception("hold after inspection failed")
                self.phase, self.stage = "stopped", "pick"
                self.message = {"grasp_lost": "Brick dropped during the inspection: holding. Space to restart",
                                "aborted": "Inspection stopped: holding. Space to restart"}.get(
                    res.status, f"Inspection failed ({res.reason}): holding. Space to restart")

        def snapshot(self) -> dict:
            d = super().snapshot()
            d["inspection"] = self.inspection
            return d

    return AutoController


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--inspect-device", default="NPU", help="YOLO + detector device (default NPU)")
    ap.add_argument("--inspect-roi", default="", help="x0,y0,x1,y1 crop of the inspect image (default: centred)")
    ap.add_argument("--inspect-frames", type=int, default=12)
    ap.add_argument("--inspect-range", type=float, default=180.0)
    ap.add_argument("--inspect-speed", type=float, default=60.0)
    ours, rest = ap.parse_known_args()
    CFG.__dict__.update(vars(ours))
    CFG.roi = tuple(int(v) for v in ours.inspect_roi.split(",")) if ours.inspect_roi else None
    S._manual_make_controller_class = S.make_controller_class
    S.make_controller_class = make_auto_controller_class  # looked up by name inside S.main()
    sys.argv = [sys.argv[0], *rest]
    log.info("auto-inspect on %s", ours.inspect_device)
    S.main()


if __name__ == "__main__":
    main()
