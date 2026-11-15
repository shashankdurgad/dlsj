"""The robot-side target selection (wrist_target.TargetTracker) on the real detector's boxes of recorded picks.

    python sim/twin/perception/eval_selector.py          (plain python; after real_detect.py, gt_real.py)
    -> sim/twin/out/perception/selector.json

Per frame (wrist-fitted episodes, all frames; fingertip-fallback episodes within 15 frames of the close):
  valid      the tracker reported a target;
  correct    ... and its box is the target brick (IoU >= 0.3 with the brick's geometric visible box, or the box the
             ground-truth matching assigned to the target);
  wrong      valid but another object (another brick, a fin, a plate);
  lost_det   the real detector had the target at conf >= 0.5 but the tracker did not report it.
Also checks that wrist_target.box_to_rel == geometry.Geometry.box_to_rel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.perception import geometry as G  # noqa: E402
from sim.twin.perception.gt_real import iou  # noqa: E402
from sim.twin.perception.wrist_target import BRICK_CENTRE_Z, TargetTracker, WristCamera  # noqa: E402

OUT = REPO / "sim/twin/out/perception"
BINS = [-20, -10, 0, 3, 6, 10, 15, 20, 30, 45, 76]


def main():
    from sim.twin.align import frames as F

    d = dict(np.load(OUT / "real_table.npz"))
    params = json.loads((Path(__file__).resolve().parent / "detector_model.json").read_text())
    cam = WristCamera(plane_dz=params["plane_dz"])
    # parity with the batched geometry
    geo = G.Geometry()
    jj = F.states(0)[40:50]
    bx = np.array([[200, 150, 320, 260]] * 10, float)
    r1, _, _ = cam.box_to_rel(bx, jj)
    r2, _ = geo.box_to_rel(bx, jj, BRICK_CENTRE_Z + params["plane_dz"], centre_z=BRICK_CENTRE_Z)
    parity = float(np.abs(r1 - r2).max())
    ok = d["fit_ok"].astype(bool)
    rows_by_ep = {}
    for i in np.where(d["area_px"] > 0)[0]:
        rows_by_ep.setdefault(int(d["ep"][i]), {})[int(d["frame"][i])] = i
    res = {k: [] for k in ("tt", "valid", "correct", "wrong", "det", "use", "err", "ep")}
    fits = json.loads((OUT / "real_fit.json").read_text())["episodes"]
    trk = TargetTracker(cam)
    for e, rmap in sorted(rows_by_ep.items()):
        rec = json.loads((OUT / f"real_dets/ep{e:03d}.json").read_text())
        st = F.states(e)
        trk.reset()
        for f, dets in zip(rec["frames"], rec["dets"], strict=True):
            t = trk.update(dets, st[f])
            i = rmap.get(f)
            if i is None:
                continue
            tt = int(d["tt"][i])
            use = (ok[i] and -20 <= tt <= 75) or (not ok[i] and -20 <= tt <= 15)
            gt_box = d["box_vis"][i]
            gt_pos = np.array([fits[str(e)]["x"], fits[str(e)]["y"]])
            det = d["conf"][i] >= 0.5
            corr = False
            if t.valid:
                b = np.array(t.box)
                corr = bool(iou(b, gt_box) >= 0.3) or (
                    np.isfinite(d["det_box"][i]).all() and bool(iou(b, d["det_box"][i]) >= 0.7)
                )
            res["tt"].append(tt)
            res["valid"].append(t.valid)
            res["correct"].append(corr)
            res["wrong"].append(t.valid and not corr)
            res["det"].append(det)
            res["use"].append(use)
            res["err"].append(np.linalg.norm(t.pos[:2] - gt_pos[:2]) if t.valid and not corr else np.nan)
            res["ep"].append(e)
    a = {k: np.array(v) for k, v in res.items()}
    u = a["use"]
    table = []
    for lo, hi in zip(BINS[:-1], BINS[1:], strict=True):
        k = u & (a["tt"] >= lo) & (a["tt"] < hi)
        if k.sum() < 20:
            continue
        table.append(
            {
                "tt": [lo, hi],
                "n": int(k.sum()),
                "target_detected@0.5": round(float(a["det"][k].mean()), 3),
                "valid": round(float(a["valid"][k].mean()), 3),
                "correct": round(float(a["correct"][k].mean()), 3),
                "wrong": round(float(a["wrong"][k].mean()), 3),
                "lost_det": round(float((a["det"][k] & ~a["correct"][k]).mean()), 3),
            }
        )
    wr, c = [], 0
    for k in range(len(a["wrong"])):
        if k and a["ep"][k] != a["ep"][k - 1] and c:
            wr.append(c)
            c = 0
        if a["wrong"][k] and a["use"][k]:
            c += 1
        elif c:
            wr.append(c)
            c = 0
    werr = a["err"][u & a["wrong"] & (a["tt"] >= 0)]
    out = {
        "parity_box_to_rel_m": parity,
        "wrong_runs_mean": round(float(np.mean(wr)), 2) if wr else 0.0,
        "wrong_offset_mm_p10_50_90": (np.nanpercentile(werr, [10, 50, 90]) * 1e3).round(1).tolist(),
        "overall": {
            "valid": round(float(a["valid"][u].mean()), 3),
            "correct": round(float(a["correct"][u].mean()), 3),
            "wrong": round(float(a["wrong"][u].mean()), 3),
            "target_detected@0.5": round(float(a["det"][u].mean()), 3),
            "wrong_given_valid": round(float(a["wrong"][u].sum() / max(a["valid"][u].sum(), 1)), 3),
        },
        "timeline": table,
    }
    (OUT / "selector.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
