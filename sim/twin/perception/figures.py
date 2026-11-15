"""Side-by-side figures: real wrist frame + the real detector / tracker vs the twin's render + the sim detector.

    python sim/twin/perception/figures.py [--n 8]        (plain python, PyAV; after render_vis.py + proxy_eval.py)
    -> sim/twin/out/perception/side_by_side.jpg

Rows: rendered jobs that replay a recorded state (not perturbed), most occluded first (rendered visible fraction).
Left: the REAL wrist frame at that state with every real box >= 0.5 (thin orange; the fins' false positives included)
and the tracker's target (thick green, wrist_target.TargetTracker run over the episode up to this frame).
Right: the twin's wrist render at the same joints (TwinEnv, raw pinhole), the target's visible pixels outlined
(cyan), the occlusion proxy's visible box (yellow) and one draw of the sim detector (green box, or MISS), with the
rendered / proxy visible fractions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.perception import geometry as G  # noqa: E402
from sim.twin.perception import proxy_eval as P  # noqa: E402
from sim.twin.perception.detector_model import DetectorModel  # noqa: E402
from sim.twin.perception.wrist_target import TargetTracker, WristCamera  # noqa: E402

OUT = REPO / "sim/twin/out/perception"


def label(img, text, y=24, col=(255, 255, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], y + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    from sim.twin.align import frames as F

    d = P.load()
    sys.path.insert(0, str(REPO / "sim/residual_env"))
    from calib import load_profile

    geo_r = G.Geometry(offsets_rad=list(load_profile("joe_cal2").joint_offsets), camera="sim")
    v = P.proxy(d, geo_r)
    inside = v["vis"] + v["occ_grip"] + v["occ_other"]
    vp = v["vis"] / np.maximum(inside, 1e-9)
    vr = d["n_vis"] / np.maximum(d["n_amodal"], 1)
    img = OUT / "render/img"
    cand = [
        i
        for i in range(len(d["job"]))
        if not d["perturbed"][i]
        and d["n_amodal"][i] > 200
        and d["tt"][i] >= 0
        and (img / f"rgb_{int(d['job'][i]):04d}.jpg").exists()
    ]
    cand.sort(key=lambda i: vr[i])
    rng = np.random.default_rng(3)
    model = DetectorModel(1, geo=geo_r)
    cam = WristCamera()
    rows = []
    for i in cand[: a.n]:
        ep, fr, j = int(d["ep"][i]), int(d["frame"][i]), int(d["job"][i])
        real = F.video_frames(ep, "wrist", [fr])[fr][..., ::-1].copy()
        rec = json.loads((OUT / f"real_dets/ep{ep:03d}.json").read_text())
        st = F.states(ep)
        trk = TargetTracker(cam)
        t = None
        for f, dets in zip(rec["frames"], rec["dets"], strict=True):
            if f > fr:
                break
            t = trk.update(dets, st[f])
            if f == fr:
                for b in dets:
                    if b[4] >= 0.5:
                        cv2.rectangle(real, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 165, 255), 1)
        if t is not None and t.valid:
            b = t.box
            cv2.rectangle(real, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 3)
        label(
            real,
            f"REAL ep{ep} f{fr}, {int(d['tt'][i])} fr to close, tracker "
            f"{'%.2f' % t.conf if t is not None and t.valid else 'none'}",
        )
        sim = cv2.imread(str(img / f"rgb_{j:04d}.jpg"))
        m = cv2.imread(str(img / f"mask_{j:04d}.png"), cv2.IMREAD_GRAYSCALE)
        cs, _ = cv2.findContours((m == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(sim, cs, -1, (255, 255, 0), 1)
        bv = v["box_vis"][i]
        if np.isfinite(bv).all():
            cv2.rectangle(sim, (int(bv[0]), int(bv[1])), (int(bv[2]), int(bv[3])), (0, 255, 255), 1)
        model.reset(None, rng)
        one = {k: (val[i : i + 1] if hasattr(val, "shape") and val.ndim else val) for k, val in v.items()}
        o = model.observe(d["joints"][i : i + 1].astype(float), one, rng, speed_px=np.array([5.0]))
        if o["detected"][0] and not o["wrong"][0]:
            b = o["box"][0]
            cv2.rectangle(sim, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 3)
        label(
            sim,
            f"TWIN vis render {vr[i]:.2f} proxy {vp[i]:.2f}, sim det "
            f"{'%.2f' % o['conf'][0] if o['detected'][0] else 'MISS'}",
        )
        rows.append(np.concatenate([real, sim], 1))
    sheet = np.concatenate(rows, 0)
    sheet = cv2.resize(sheet, (sheet.shape[1] * 3 // 4, sheet.shape[0] * 3 // 4), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "side_by_side.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", OUT / "side_by_side.jpg", len(rows), "rows")


if __name__ == "__main__":
    main()
