"""Contact sheets of the wrist camera before / after the gripper-close frame, for labelling the picked brick's kind and
resting pose by eye (plain python, PyAV; never in an Isaac process).

    sim/twin/run.sh sim/twin/grasp_gate/wristsheet.py [--n 60] [--per 16]
    -> sim/twin/out/grasp_gate/wristsheet/sheet_<k>.jpg (per demo: wrist at close - 10 and close + 12)
Labels go to sim/twin/grasp_gate/brick_labels.json (gate.py --place labels).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.align import frames as F  # noqa: E402
from sim.twin.grasp_gate.common import OUT, load_demos, pick_test  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--per", type=int, default=16)
    ap.add_argument("--episodes", type=int, nargs="*")
    args = ap.parse_args()
    demos = load_demos()
    eps = args.episodes or pick_test(demos, args.n)
    out = OUT / "wristsheet"
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for ep in eps:
        cf = int(demos["eps"][ep]["cf"])
        idx = [max(cf - 10, 0), cf + 12]
        fr = F.video_frames(ep, "wrist", idx)
        pair = np.concatenate([cv2.resize(fr[i], (256, 192)) for i in idx], 1)
        cv2.putText(pair, f"ep {ep}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        tiles.append(pair)
    for k in range(0, len(tiles), args.per):
        chunk = tiles[k:k + args.per]
        chunk += [np.zeros_like(tiles[0])] * ((-len(chunk)) % 2)
        sheet = np.concatenate([np.concatenate(chunk[j:j + 2], 1) for j in range(0, len(chunk), 2)], 0)
        cv2.imwrite(str(out / f"sheet_{k // args.per}.jpg"), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
        print("wrote", out / f"sheet_{k // args.per}.jpg", flush=True)


if __name__ == "__main__":
    main()
