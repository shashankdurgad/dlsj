"""Contact sheets of the real cameras around the gripper-close frame (plain python, PyAV; never in an Isaac process).

    python sim/twin/grasp_gate/peek.py --n 8        -> sim/twin/out/grasp_gate/peek/ep<NNN>.jpg
Rows: top at frame 0, close - 10, close; wrist at close - 20, close - 5, close, close + 6.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.align import frames as F
from sim.twin.grasp_gate.common import OUT, load_demos, pick_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--episodes", type=int, nargs="*")
    args = ap.parse_args()
    demos = load_demos()
    out = OUT / "peek"
    out.mkdir(parents=True, exist_ok=True)
    for ep in args.episodes or pick_test(demos, args.n):
        cf = demos["eps"][ep]["cf"]
        ti, wi = [0, cf - 10, cf], [cf - 20, cf - 5, cf, cf + 6]
        top, wr = F.video_frames(ep, "top", ti), F.video_frames(ep, "wrist", wi)
        a = np.concatenate([top[i] for i in ti] + [np.zeros_like(top[0])], 1)
        b = np.concatenate([wr[i] for i in wi], 1)
        cv2.imwrite(str(out / f"ep{ep:03d}.jpg"), np.concatenate([a, b], 0)[..., ::-1])
        print(ep, "close", cf, np.round(demos["eps"][ep]["state"][cf], 1).tolist(), flush=True)


if __name__ == "__main__":
    main()
