"""Table-level correspondences for the top camera: picked brick (top image, before the grasp) <-> FK fin-ray tip at
the gripper-close frame (W9 joint offsets).

Per labelled cal2 episode:
* close frame = align/joint_fit/tip_height.close_frame (first frame after the widest opening that reaches the held
  gripper reading, before the operator's inspection mark);
* bricks are detected (scene/layout.detect_bricks, arm and props masked) in frame 0 and in the inspection frame (brick
  lifted); the target is the frame-0 brick that has vanished from the table by the inspection frame, nearest to the
  FK tip (unambiguous only);
* its position is then refreshed from the latest pre-close frame (close-25 ... close-90) in which a same-colour brick
  is still detected within 2.5 cm of it (operator nudges);
* the brick centre is stored as a PIXEL (the detection's world centre projected back with the camera it was detected
  with), so the refit can back-project it to the brick-top plane z = 0.0172 m with any candidate camera.

    sim/twin/run.sh sim/twin/align/top_refit/corr.py [--workers 12]  -> sim/twin/out/top_refit/corr.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TWIN = HERE.parents[1]
REPO = TWIN.parents[1]
for p in (REPO, HERE.parent, HERE.parent / "joint_fit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from fk import SO101, sim_q  # noqa: E402

from sim.twin.scene import layout as LY  # noqa: E402

OUT = TWIN / "out/top_refit"
FIN_TIP = np.array([0.0063, 0.001, -0.0902])  # fin-ray fingertip in gripper_link (gripper agent)
BRICK_TOP_Z = LY.TABLE_Z + LY.BRICK_TOP  # 0.0172 m
PRE = (25, 40, 60, 90)  # frames before the close frame to refresh the brick position from (latest first)

_ARM = None


def w9_offsets() -> np.ndarray:
    res = json.loads((TWIN / "align/joint_fit/result.json").read_text())
    return np.array(res["grasp_depth_candidates"]["W9"]["joint_offsets_rad_for_sim"][:5], float)


def fin_tip(state, offsets) -> np.ndarray:
    global _ARM
    if _ARM is None:
        _ARM = SO101()
    G = _ARM.fk(sim_q(state, offsets, 0.0, 0.0))["gripper_link"]
    return G[:3, :3] @ FIN_TIP + G[:3, 3]


def close_frame(s, insp):
    g = s[:, 5]
    held = np.median(g[max(insp - 5, 0) : insp + 6])
    i_open = int(np.argmax(g[:insp]))
    after = np.where(g[i_open:insp] <= held + 1.0)[0]
    return (i_open + int(after[0]) if len(after) else None), float(held)


def bricks_at(img, state, cam):
    ign = LY.props_mask(cam, img.shape[:2]) | LY.arm_mask(state, cam, img.shape[:2])
    plates = LY.detect_plates(img, cam, ignore=ign)
    return LY.detect_bricks(img, cam, plates, ignore=ign), plates


def one(ep: int):
    from sim.twin.visuals.realdata import Dataset

    ds = Dataset()
    lab = LY._labels(LY.DATA).get(ep, {})
    insp = lab.get("inspection_frame")
    if insp is None:
        return dict(ep=ep, skip="no inspection frame")
    s = ds.states(ep)
    ci, held = close_frame(s, int(insp))
    if ci is None or held >= 15:
        return dict(ep=ep, skip=f"no close frame (held {held:.1f})")
    cam = LY.top_camera()
    pre = [ci - k for k in PRE if ci - k > 0]
    idx = sorted({0, int(insp), *pre})
    imgs = dict(zip(idx, ds.frames(ep, "top", idx), strict=True))
    b0, plates = bricks_at(imgs[0], s[0], cam)
    bpost, _ = bricks_at(imgs[int(insp)], s[int(insp)], cam)
    tip = fin_tip(s[ci], w9_offsets())
    vanished = [
        b
        for b in b0
        if not b["on_plate"]
        and all(not (p["color"] == b["color"] and math.hypot(p["x"] - b["x"], p["y"] - b["y"]) < 0.02) for p in bpost)
    ]
    if not vanished:
        return dict(ep=ep, skip="no vanished brick", n0=len(b0))
    d = np.array([math.hypot(b["x"] - tip[0], b["y"] - tip[1]) for b in vanished])
    order = np.argsort(d)
    k = int(order[0])
    if d[k] > 0.12:
        return dict(ep=ep, skip=f"nearest vanished brick {d[k]:.3f} m from the tip")
    if len(order) > 1 and d[order[1]] < d[k] + 0.04:
        return dict(ep=ep, skip="ambiguous: two vanished bricks near the tip")
    tgt, src = vanished[k], 0
    for f in sorted(pre, reverse=True):  # latest pre-close frame where the target is still seen
        bf, _ = bricks_at(imgs[f], s[f], cam)
        m = [b for b in bf if b["color"] == tgt["color"] and math.hypot(b["x"] - tgt["x"], b["y"] - tgt["y"]) < 0.025]
        if m:
            nudge = math.hypot(m[0]["x"] - tgt["x"], m[0]["y"] - tgt["y"])
            tgt, src = min(m, key=lambda b: math.hypot(b["x"] - tgt["x"], b["y"] - tgt["y"])), f
            break
    else:
        nudge = None
    uv, _ = LY.project(np.array([[tgt["x"], tgt["y"], BRICK_TOP_Z]]), cam)
    return dict(
        ep=ep,
        close_frame=int(ci),
        inspection_frame=int(insp),
        brick_frame=int(src),
        nudge_m=None if nudge is None else round(nudge, 4),
        color=tgt["color"],
        kind=tgt["kind"],
        yaw=tgt["yaw"],
        uv=uv[0].tolist(),
        xy_old_cam=[tgt["x"], tgt["y"]],
        tip_w9=tip.tolist(),
        state_close=s[ci].tolist(),
        dist_old_cam=round(float(d[k]), 4),
        n_vanished=len(vanished),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--episodes", type=int, nargs="*")
    a = ap.parse_args()
    eps = a.episodes or sorted(LY._labels(LY.DATA))
    with Pool(a.workers) as pool:
        rows = pool.map(one, eps, chunksize=4)
    ok = [r for r in rows if "skip" not in r]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "corr.json").write_text(json.dumps(dict(fin_tip=FIN_TIP.tolist(), brick_top_z=BRICK_TOP_Z, rows=rows)))
    from collections import Counter

    print(f"{len(ok)} / {len(rows)} episodes with a correspondence", flush=True)
    print("skips:", Counter(r["skip"].split(" (")[0].split(":")[0] for r in rows if "skip" in r), flush=True)
    if ok:
        d = np.array([r["dist_old_cam"] for r in ok]) * 100
        print(f"old top cam: brick-to-tip {np.median(d):.2f} cm median, p90 {np.percentile(d, 90):.2f}", flush=True)
    print("CORR_OK", flush=True)


if __name__ == "__main__":
    main()
