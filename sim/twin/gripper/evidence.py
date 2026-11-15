"""Real finger evidence: wrist / top frames at a spread of gripper readings (plain python; PyAV).

    docker run ... dlsj/twin:latest sim/twin/gripper/evidence.py [--eps 17 76]
Writes sim/twin/out/gripper/evidence_ep*.jpg (wrist row, top row; columns = gripper reading) and
evidence_frames.npz (full-res wrist frames + joint states, used by fit / compare scripts).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.align import frames as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--eps", type=int, nargs="*", default=[17, 76, 224])
ap.add_argument("--targets", type=float, nargs="*", default=[60, 45, 30, 20, 12, 6])
args = ap.parse_args()
out = REPO / "sim/twin/out/gripper"
out.mkdir(parents=True, exist_ok=True)
keep = {}
for ep in args.eps:
    s = F.states(ep)
    g = s[:, 5]
    print(ep, "frames", len(g), "grip min/max", g.min().round(2), g.max().round(2))
    idx = []
    for t in args.targets:
        i = int(np.argmin(np.abs(g - t)))
        if i not in idx:
            idx.append(i)
    idx.append(int(np.argmin(g)))
    rows = []
    for cam in ("wrist", "top"):
        fr = F.video_frames(ep, cam, idx)
        tiles = []
        for i in idx:
            im = fr[i].copy()
            cv2.putText(im, f"ep{ep} f{i} grip {g[i]:.1f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(im)
            if cam == "wrist":
                keep[f"ep{ep}_f{i}"] = fr[i]
                keep[f"ep{ep}_f{i}_state"] = s[i]
        rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(str(out / f"evidence_ep{ep}.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
np.savez_compressed(out / "evidence_frames.npz", **keep)
print("EVIDENCE_OK", sorted(k for k in keep if not k.endswith("state")))
