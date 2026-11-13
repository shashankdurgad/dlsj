"""Collect `inspect`-camera frames for the defect detector (no robot needed: hold the brick at the present spot).

    python -m dlsj.inspect.capture --camera 2 --out ~/inference/inspect_data/real --roi 160,80,480,400

Keys in the preview window:
  g  save a PLAIN brick ("good"): grabs `--views` consecutive frames as <brick>_f0..f2.png in good/
  b  save a CROSS-MARKED brick ("bad"): same, in bad/
  r  draw a new ROI with the mouse (Enter/Space to accept); it is stored in dataset.json
  i  run the detector (needs --manifest) on the next `--views` frames and show the result
  q  quit

Full frames are saved (RGB PNG, the camera's resolution), so the ROI can still be changed before training;
`train.py` and the detector crop with the ROI from dataset.json / the manifest. Lock auto-exposure and
auto-white-balance first (PLAYBOOK §1.5, e.g. `v4l2-ctl -d /dev/videoN -c auto_exposure=1 -c white_balance_automatic=0`),
and capture under the demo lighting, with the gripper holding the brick as the policy will.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from dlsj.inspect import core
from dlsj.teleop_web import CameraStream

log = logging.getLogger("inspect.capture")


def grab(cam: CameraStream, n: int, timeout: float = 3.0) -> list[np.ndarray]:
    """`n` consecutive new frames from the camera thread, as RGB."""
    frames, seq = [], cam.seq
    deadline = time.monotonic() + timeout
    while len(frames) < n:
        with cam.cond:
            cam.cond.wait_for(lambda: cam.seq != seq, timeout=max(0.0, deadline - time.monotonic()))
            if cam.seq == seq:
                raise TimeoutError(f"camera {cam.index}: no new frame ({cam.error})")
            seq, frame = cam.seq, cam.frame
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="0", help="OpenCV index or device path (e.g. /dev/v4l/by-id/...)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--roi", help="x0,y0,x1,y1 (default: from dataset.json, else draw it with 'r')")
    ap.add_argument("--views", type=int, default=3, help="frames saved per brick")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--manifest", type=Path, help="trained detector, for live checks with 'i'")
    ap.add_argument("--device", default="NPU")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out = args.out.expanduser()
    for kind in ("good", "bad"):
        (out / kind).mkdir(parents=True, exist_ok=True)
    meta_path = out / "dataset.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"source": "capture"}
    roi = core.parse_box(args.roi) or (tuple(meta["crop"]) if meta.get("crop") else None)

    def save_meta(shape):
        meta.update(crop=list(roi) if roi else None, frame_size=[shape[1], shape[0]], camera=str(args.camera))
        meta_path.write_text(json.dumps(meta, indent=1) + "\n")

    inspector = None
    if args.manifest:
        from dlsj.inspect.detector import AnomalibInspector

        inspector = AnomalibInspector.from_manifest(args.manifest, device=args.device, batch=args.views)
        if roi is not None and inspector.manifest.crop != roi:
            log.warning("ROI %s differs from the manifest's %s; the detector uses the manifest's", roi,
                        inspector.manifest.crop)

    index = int(args.camera) if str(args.camera).isdigit() else args.camera
    cam = CameraStream("inspect", index, args.width, args.height, args.fps, quality=80).start()
    counts = {k: len(core.group_by_brick(core.list_images(out / k))) for k in ("good", "bad")}
    last_result, win = None, "inspect capture"
    try:
        while True:
            with cam.cond:
                cam.cond.wait(timeout=0.5)
                frame = None if cam.frame is None else cam.frame.copy()
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            view = frame.copy()
            if roi is not None:
                cv2.rectangle(view, roi[:2], roi[2:], (0, 255, 255), 2)
            cv2.putText(view, f"good {counts['good']}  bad {counts['bad']}   g/b save  r roi  i inspect  q quit",
                        (8, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, view)
            if last_result is not None:
                cv2.imshow("inspect result", last_result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                x, y, w, h = cv2.selectROI(win, frame, showCrosshair=True)
                if w and h:
                    roi = (int(x), int(y), int(x + w), int(y + h))
                    save_meta(frame.shape)
                    log.info("ROI %s", roi)
            elif key in (ord("g"), ord("b")):
                kind = "good" if key == ord("g") else "bad"
                frames = grab(cam, args.views)
                brick = time.strftime("cap%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
                for k, f in enumerate(frames):
                    core.write_rgb(out / kind / f"{brick}_f{k}.png", f)
                counts[kind] += 1
                save_meta(frame.shape)
                log.info("%s: %s (%d views)", kind, brick, len(frames))
            elif key == ord("i") and inspector is not None:
                frames = grab(cam, args.views)
                r = inspector.inspect(frames)
                log.info("%s score %.4g (frames %s) region %s %.1f ms on %s", r.label.value, r.score,
                         ", ".join(f"{s:.3g}" for s in r.frame_scores), r.region, r.latency_ms, r.device)
                last_result = cv2.cvtColor(core.overlay(frames[0], r), cv2.COLOR_RGB2BGR)
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
