"""Real wrist frames + joint states at a spread of recorded states (plain python, PyAV; never inside Isaac).

    python sim/twin/robot/real_wrist.py [--episodes 7 18 76 122 304] [--n 6]
writes sim/twin/out/robot/real_wrist.npz (frames uint8 (N,480,640,3), states (N,6), ep, idx) + real_wrist.jpg.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "sim/twin/align"))
from frames import states, video_frames  # noqa: E402

OUT = REPO / "sim/twin/out/robot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[7, 18, 76, 122, 190, 304, 418])
    ap.add_argument("--n", type=int, default=6, help="frames per episode, evenly spread")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fr_all, st_all, ep_all, ix_all = [], [], [], []
    for ep in a.episodes:
        st = states(ep)
        idx = np.linspace(0, len(st) - 1, a.n).round().astype(int).tolist()
        fr = video_frames(ep, "wrist", idx)
        for i in idx:
            if i in fr:
                fr_all.append(fr[i])
                st_all.append(st[i])
                ep_all.append(ep)
                ix_all.append(i)
    F = np.stack(fr_all)
    np.savez_compressed(OUT / "real_wrist.npz", frames=F, states=np.stack(st_all), ep=np.array(ep_all),
                        idx=np.array(ix_all))
    thumbs = [cv2.putText(cv2.resize(f, (256, 192)), f"{e}/{i} g{s[5]:.0f}", (4, 16), 0, 0.45, (255, 255, 0), 1)
              for f, e, i, s in zip(F, ep_all, ix_all, st_all, strict=True)]
    while len(thumbs) % a.n:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.concatenate(thumbs[r:r + a.n], 1) for r in range(0, len(thumbs), a.n)]
    cv2.imwrite(str(OUT / "real_wrist.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print("frames", F.shape, "->", OUT / "real_wrist.npz")


if __name__ == "__main__":
    main()
