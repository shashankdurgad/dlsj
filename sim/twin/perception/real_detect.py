"""Run the team's real brick detector (brick-detector-v1, one-class YOLO26n, OpenVINO CPU FP32, the deployed
`dlsj.inspect.brick_detector.OVBrickDetector`: letterbox 640, decode + NMS) on recorded REAL wrist frames of the
pick phase. Plain python with PyAV (never in an Isaac process).

    python sim/twin/perception/real_detect.py [--episodes 0 18] [--pre 75 --post 20] [--save-frames 24]
    -> sim/twin/out/perception/real_dets/ep<NNN>.json
         {"episode", "cf", "frames": [f...], "dets": [[[x0, y0, x1, y1, conf], ...] per frame], "ms": [...]}
       sim/twin/out/perception/real_frames/ep<NNN>/f<NNNN>.jpg   (the first --save-frames episodes, for figures)

Detections are kept down to conf 0.05 so any operating threshold (deployed: 0.5 inspection, 0.3 top survey) can be
applied afterwards. Weights: sim/twin/data/models/brick-detector-v1 (copied from the XPS ~/inference/models).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
WEIGHTS = REPO / "sim/twin/data/models/brick-detector-v1"
OUT = REPO / "sim/twin/out/perception"


def close_frame(grip, insp, thr=12.0, stay=15.0, open_min=15.0):
    """grasp_gate.common.close_frame (validate.py --grasp_rule gripper), without its residual-env import."""
    g = grip[:insp] if insp else grip
    if len(g) == 0 or g.max() < open_min:
        return None
    for f in range(1, len(g)):
        if g[f - 1] > thr >= g[f] and (g[f:] <= stay).all():
            return f
    return None


def keep_episodes():
    """{ep: cf} for the cal2 episodes GraspEnv / the grasp gate use (close frame found, not excluded)."""
    from sim.twin.align import frames as F

    lab = F.labels()
    out = {}
    for ep, lb in sorted(lab.items()):
        if lb.get("excluded") or not lb.get("inspection_frame"):
            continue
        try:
            st = F.states(ep)
        except Exception:  # noqa: BLE001
            continue
        cf = close_frame(st[:, 5], int(lb["inspection_frame"]))
        if cf is not None and cf >= 37:
            out[ep] = cf
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--pre", type=int, default=75)
    ap.add_argument("--post", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--save-frames", type=int, default=24)
    ap.add_argument("--shard", default="0,1")
    a = ap.parse_args()
    from dlsj.inspect.brick_detector import OVBrickDetector
    from sim.twin.align import frames as F

    det = OVBrickDetector(WEIGHTS, a.device, confidence=a.conf)
    eps = keep_episodes()
    todo = [e for e in sorted(eps) if not a.episodes or e in a.episodes]
    save = set(todo[: a.save_frames]) if not a.episodes else set(todo)
    i, n = (int(v) for v in a.shard.split(","))
    todo = todo[i::n]
    (OUT / "real_dets").mkdir(parents=True, exist_ok=True)
    t_all, n_all = time.time(), 0
    for ep in todo:
        path = OUT / "real_dets" / f"ep{ep:03d}.json"
        if path.exists() and not a.episodes:
            continue
        cf = eps[ep]
        nfr = len(F.states(ep))
        idx = list(range(max(0, cf - a.pre), min(nfr, cf + a.post + 1)))
        fr = F.video_frames(ep, "wrist", idx)
        rec = {"episode": ep, "cf": cf, "frames": [], "dets": [], "ms": []}
        for f in idx:
            if f not in fr:
                continue
            bgr = cv2.cvtColor(fr[f], cv2.COLOR_RGB2BGR)
            t0 = time.perf_counter()
            ds = det.detect(bgr)
            rec["ms"].append(round((time.perf_counter() - t0) * 1e3, 1))
            rec["frames"].append(f)
            rec["dets"].append([[*d.box, round(d.confidence, 4)] for d in ds])
            if ep in save:
                d = OUT / "real_frames" / f"ep{ep:03d}"
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"f{f:04d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            n_all += 1
        path.write_text(json.dumps(rec))
        print(ep, cf, len(rec["frames"]), f"{np.median(rec['ms']):.1f} ms", flush=True)
    print("DONE", n_all, f"{(time.time() - t_all) / max(n_all, 1) * 1e3:.1f} ms/frame incl. decode", flush=True)


if __name__ == "__main__":
    main()
