"""Real wrist frames around the sim's first fin-brick contact (forensics.py) of gate demos (plain python, PyAV).

    sim/twin/run.sh sim/twin/contact/realframes.py --tag base_fx
-> sim/twin/out/contact/<tag>/real_wrist_first_contact.jpg: per demo, the real wrist camera at the sim's first-contact
   tick - 6, at it, + 10 and at the close frame (label: sim outcome, first shape, ticks to the close).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.align import frames as F  # noqa: E402
from sim.twin.grasp_gate.common import load_demos  # noqa: E402

OUT = REPO / "sim/twin/out/contact"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", type=int, nargs="*")
    a = ap.parse_args()
    fx = json.loads((OUT / a.tag / "forensics.json").read_text())
    demos = load_demos()
    rows = []
    for r in fx:
        if "first_shape" not in r or (a.episodes and r["episode"] not in a.episodes) or \
                (not a.episodes and r["outcome"] == "success"):
            continue
        ep = r["episode"]
        cf = int(demos["eps"][ep]["cf"])
        f0 = cf + int(r["first_tick_rel_close"])
        idx = [max(f0 - 6, 0), f0, f0 + 10, cf]
        fr = F.video_frames(ep, "wrist", idx)
        tiles = []
        for i, k in zip(idx, ("first-6", "first", "first+10", "close"), strict=True):
            t = cv2.resize(fr[i], (288, 216))
            cv2.putText(t, k, (6, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            tiles.append(t)
        row = np.concatenate(tiles, 1)
        cv2.putText(row, f"ep {ep} sim {r['outcome']} | {r['first_shape']} on {r['face']} at close"
                         f"{r['first_tick_rel_close']}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        rows.append(row)
    sheet = np.concatenate(rows, 0)
    p = OUT / a.tag / "real_wrist_first_contact.jpg"
    cv2.imwrite(str(p), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
