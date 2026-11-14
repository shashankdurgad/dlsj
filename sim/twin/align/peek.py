"""Dump real top / inspect / wrist frames at start, grasp mark and inspect mark for a few episodes."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frames import labels, states, video_frames  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out/align"
OUT.mkdir(parents=True, exist_ok=True)
L = labels()
print("label keys:", sorted(next(iter(L.values())).keys()))
rows = []
for ep in (440, 445, 450):
    lab, st = L[ep], states(ep)
    marks = [
        0,
        lab.get("grasp_frame") or len(st) // 3,
        lab.get("mark_frame") or lab.get("inspect_frame") or len(st) // 2,
    ]
    marks = [int(min(m, len(st) - 1)) for m in marks]
    print(ep, "frames", len(st), "marks", marks, "state@marks", np.round(st[marks], 1).tolist())
    row = []
    for cam in ("top", "inspect", "wrist"):
        fr = video_frames(ep, cam, marks)
        row.append(np.concatenate([cv2.resize(fr[m], (320, 240)) for m in marks], 1))
    rows.append(np.concatenate(row, 0))
cv2.imwrite(str(OUT / "peek.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
print("wrote", OUT / "peek.jpg")
