"""Decode the frames used by the joint fit once and cache their real-robot masks.

Episodes: every 7th labelled episode (as make_samples.py, 66 episodes); split 2:1, held-out = episodes[2::3].
Frames (per episode): the slowest frame (min joint speed, so state / video sync errors are small) in each of six
equal segments -> top + wrist samples; for the inspect camera additionally the operator's inspection mark +-8 frames.
Masks at half resolution (320x240):
  top / inspect: differs from the episode median background AND dark (arm); red (the TPU fin-ray fingers, not
                 modelled by the URDF jaws) -> separate don't-care mask `dc`;
  wrist:         red jaws (the gripper is static in this view, so no background model): hue red, saturated.
Writes sim/twin/out/align/joint_fit/data.npz.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TWIN = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
import fit_cameras as fc  # noqa: E402
from frames import labels, states, video_frames  # noqa: E402

OUT = TWIN / "out/align/joint_fit"


def wrist_mask(fr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(fr, cv2.COLOR_RGB2HSV)
    red = ((hsv[..., 0] < 7) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 60)
    m = red.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.resize(m, None, fx=fc.SCALE, fy=fc.SCALE, interpolation=cv2.INTER_NEAREST)


def arm_mask(fr, bg):
    """fit_cameras.real_mask without the red jaws: the real fingers are TPU fin-ray fingers (not the URDF jaws), so
    red pixels are returned separately as a don't-care mask (dilated) instead of being matched."""
    hsv = cv2.cvtColor(fr, cv2.COLOR_RGB2HSV)
    diff = np.abs(fr.astype(np.int16) - bg.astype(np.int16)).max(-1) > 40
    dark = hsv[..., 2] < 70
    red = ((hsv[..., 0] < 8) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 70)
    m = cv2.morphologyEx((diff & dark).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    r = cv2.dilate(red.astype(np.uint8), np.ones((9, 9), np.uint8))
    sc = dict(fx=fc.SCALE, fy=fc.SCALE, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(m, None, **sc), cv2.resize(r, None, **sc)


def frames_for(ep: int, lab: dict):
    s = states(ep)
    n = len(s)
    sp = np.r_[np.inf, np.linalg.norm(np.diff(s[:, :5], axis=0), axis=1)]
    sp = np.convolve(sp, np.ones(3) / 3, "same")
    sp[:3] = sp[-3:] = np.inf
    edges = np.linspace(0, n, 7).astype(int)
    tw = [int(a + np.argmin(sp[a:b])) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    m = lab.get("inspection_frame")
    ins = [int(min(max(m + d, 0), n - 1)) for d in (-8, 0, 8)] if m is not None else []
    return s, tw, sorted(set(ins) | set(tw))


def one(ep: int):
    lab = labels()[ep]
    s, tw, ins = frames_for(ep, lab)
    n = len(s)
    bgi = sorted(set(np.linspace(0, n - 1, 25).astype(int).tolist()))
    items = []
    for cam, idx in (("top", tw), ("inspect", ins), ("wrist", tw)):
        vf = video_frames(ep, cam, sorted(set(idx) | (set(bgi) if cam != "wrist" else set())))
        if cam != "wrist":
            bg = np.median(np.stack([vf[i] for i in bgi if i in vf]), 0).astype(np.uint8)
        for i in idx:
            if i not in vf:
                continue
            fr = vf[i]
            m, dc = (wrist_mask(fr), np.zeros((240, 320), np.uint8)) if cam == "wrist" else arm_mask(fr, bg)
            speed = float(np.linalg.norm(s[min(i + 1, n - 1), :5] - s[i, :5]))
            items.append(dict(cam=cam, ep=ep, fr=i, dc=dc, state=s[i], speed=speed,
                              mask=m, rgb=cv2.resize(fr, (320, 240), interpolation=cv2.INTER_AREA)))
    return items


def main():
    eps = sorted(labels())[::7]
    with Pool(10) as p:
        res = p.map(one, eps)
    items = [it for r in res for it in r]
    OUT.mkdir(parents=True, exist_ok=True)
    def col(k):
        return np.array([it[k] for it in items])

    np.savez_compressed(OUT / "data.npz", cam=col("cam"), ep=col("ep"), fr=col("fr"), state=np.stack(col("state")),
                        speed=col("speed"), mask=col("mask"), dc=col("dc"), rgb=col("rgb"), episodes=np.array(eps))
    for c in ("top", "inspect", "wrist"):
        print(c, sum(i["cam"] == c for i in items), "items")
    # mask preview: 4 random items per camera (rgb | mask tinted)
    rng = np.random.default_rng(1)
    rows = []
    for c in ("top", "inspect", "wrist"):
        sel = [i for i in items if i["cam"] == c]
        tiles = []
        for it in [sel[k] for k in rng.choice(len(sel), 4, replace=False)]:
            im = it["rgb"].copy()
            im[it["mask"] > 0] = (0.4 * im[it["mask"] > 0] + [0, 150, 0]).astype(np.uint8)
            tiles.append(im)
        rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(str(OUT / "masks_preview.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print("PREP_OK", len(items))


if __name__ == "__main__":
    main()
