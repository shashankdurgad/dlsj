"""Accuracy of the camera-free occlusion proxy (geometry.py) against the twin's rendered instance segmentation.

    python sim/twin/perception/proxy_eval.py        (plain python; after render_vis.py)
    -> sim/twin/out/perception/proxy_eval.json, proxy_scatter.png, proxy_examples.jpg

For every rendered job (render/vis_*.npz) the proxy is evaluated on exactly the rendered state (measured joints after
the settle, brick pose / size, other bricks as 2x2-footprint boxes of their recorded positions) with the twin's own
camera model (Geometry(camera="sim"): centred pinhole f 341.8, local_to_cv_sim). Compared, in frame:
  rendered visible fraction  = target pixels with everything / target pixels with the robot + other bricks hidden
  proxy visible fraction     = vis / (vis + occ_grip + occ_other)   (the proxy's out-of-frame part removed)
and the visible boxes (IoU), plus the out-of-frame / occluded classification at a few thresholds.
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

OUT = REPO / "sim/twin/out/perception"


def load():
    parts = [dict(np.load(p, allow_pickle=True)) for p in sorted((OUT / "render").glob("vis_*.npz"))]
    # a shard whose target is never in frame rendered black (seen once on jarvis): drop it
    parts = [p for p in parts if (p["n_amodal"] > 0).any()]
    keys = [k for k in parts[0] if k != "occ_names"]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    d["occ_names"] = parts[0]["occ_names"]
    return d


def proxy(d, geo):
    n = len(d["job"])
    J = d["joints"].astype(float)
    R = G.quat_xyzw_to_R(d["bquat"].astype(float))
    size = d["bsize"].astype(float)
    m = max([len(json.loads(str(o))) for o in d["others"]] + [1])
    op, oR, osz, ov = (
        np.zeros((n, m, 3)),
        np.tile(np.eye(3), (n, m, 1, 1)),
        np.tile([0.0318, 0.0318, 0.0192], (n, m, 1)),
        np.zeros((n, m), bool),
    )
    for i, o in enumerate(d["others"]):
        for k, row in enumerate(json.loads(str(o))):
            op[i, k] = row[:3]
            oR[i, k] = G.quat_xyzw_to_R(np.array([row[3:7]]))[0]
            ov[i, k] = True
    return geo.visibility(J, d["bpos"].astype(float), R, size, (op, oR, osz, ov))


def main():
    d = load()
    # TwinEnv's ResidualEnv converts joints with joe_cal2's offsets (its `off`; TwinEnv does not switch it to the twin
    # profile's W offsets), so the rendered arm is FK(joe_cal2) of the recorded readings
    sys.path.insert(0, str(REPO / "sim/residual_env"))
    from calib import load_profile

    geo = G.Geometry(offsets_rad=list(load_profile("joe_cal2").joint_offsets), camera="sim", samples=7)
    v = proxy(d, geo)
    ok = d["n_amodal"] > 50  # target at least partly in frame in the render
    vr = d["n_vis"] / np.maximum(d["n_amodal"], 1)
    inside = v["vis"] + v["occ_grip"] + v["occ_other"]
    vp = np.where(inside > 1e-6, v["vis"] / np.maximum(inside, 1e-9), 0.0)
    err = vp - vr
    occn = list(d["occ_names"])
    fin_r = d["occ_by"][:, occn.index("fin")] / np.maximum(d["n_amodal"], 1)
    base_r = d["occ_by"][:, occn.index("base")] / np.maximum(d["n_amodal"], 1)
    other_r = (d["occ_by"][:, occn.index("brick")] + d["occ_by"][:, occn.index("plate")]) / np.maximum(d["n_amodal"], 1)
    res = {"n_jobs": int(len(vr)), "n_in_frame": int(ok.sum())}
    k = ok
    res["visible_fraction"] = {
        "mae": round(float(np.abs(err[k]).mean()), 4),
        "bias_proxy_minus_render": round(float(err[k].mean()), 4),
        "p90_abs": round(float(np.percentile(np.abs(err[k]), 90)), 4),
        "corr": round(float(np.corrcoef(vp[k], vr[k])[0, 1]), 3),
    }
    occl = k & (vr < 0.98)
    res["occluded_jobs"] = {
        "n": int(occl.sum()),
        "mae": round(float(np.abs(err[occl]).mean()), 4) if occl.any() else None,
        "bias": round(float(err[occl].mean()), 4) if occl.any() else None,
    }
    res["classification"] = {}
    for thr in (0.95, 0.8, 0.5):
        a, b = vr[k] < thr, vp[k] < thr
        res["classification"][f"vis<{thr}"] = {
            "render_rate": round(float(a.mean()), 3),
            "proxy_rate": round(float(b.mean()), 3),
            "agreement": round(float((a == b).mean()), 3),
        }
    res["by_occluder_fraction_of_amodal"] = {
        "fin_render_mean": round(float(fin_r[k].mean()), 4),
        "fin_proxy_mean": round(
            float(((v["occ_part"][:, 0] + v["occ_part"][:, 1]) / np.maximum(inside, 1e-9))[k].mean()), 4
        ),
        "base_render_mean": round(float(base_r[k].mean()), 4),
        "base_proxy_mean": round(
            float(((v["occ_part"][:, 2] + v["occ_part"][:, 3]) / np.maximum(inside, 1e-9))[k].mean()), 4
        ),
        "other_render_mean": round(float(other_r[k].mean()), 4),
        "other_proxy_mean": round(float((v["occ_other"] / np.maximum(inside, 1e-9))[k].mean()), 4),
    }
    bp = v["box_vis"]
    bp = np.stack(
        [np.clip(bp[:, 0], 0, 640), np.clip(bp[:, 1], 0, 480), np.clip(bp[:, 2], 0, 640), np.clip(bp[:, 3], 0, 480)], 1
    )
    br = d["box_vis_r"].astype(float)
    good = k & np.isfinite(bp).all(1) & np.isfinite(br).all(1)
    ious = iou(bp[good], br[good])
    res["visible_box_iou"] = {
        "median": round(float(np.median(ious)), 3),
        "p10": round(float(np.percentile(ious, 10)), 3),
    }
    for name, sel in (("replayed", ~d["perturbed"].astype(bool)), ("perturbed", d["perturbed"].astype(bool))):
        kk = k & sel
        if kk.any():
            res[f"vis_mae_{name}"] = round(float(np.abs(err[kk]).mean()), 4)
    by_tt = []
    for t in sorted(set(d["tt"].tolist()), reverse=True):
        kk = k & (d["tt"] == t)
        if kk.sum() >= 5:
            by_tt.append(
                {
                    "tt": int(t),
                    "n": int(kk.sum()),
                    "render_vis": round(float(vr[kk].mean()), 3),
                    "proxy_vis": round(float(vp[kk].mean()), 3),
                    "mae": round(float(np.abs(err[kk]).mean()), 3),
                }
            )
    res["by_time_to_close"] = by_tt
    (OUT / "proxy_eval.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    np.savez_compressed(OUT / "proxy_eval_rows.npz", job=d["job"], vr=vr, vp=vp, ok=ok, tt=d["tt"])
    plot(vr[k], vp[k], d["tt"][k], OUT / "proxy_scatter.png")
    examples(d, v, vr, vp)


def plot(vr, vp, tt, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(vr, vp, c=np.clip(tt, -10, 45), s=10, cmap="viridis", alpha=0.7)
    ax.plot([0, 1], [0, 1], color="#888", lw=1)
    ax.set_xlabel("rendered visible fraction (instance segmentation)")
    ax.set_ylabel("proxy visible fraction (ray / hull test)")
    ax.set_title("Occlusion proxy vs twin render (in frame)")
    fig.colorbar(sc, label="frames before the gripper close")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def examples(d, v, vr, vp):
    """Rendered wrist frame | rendered target mask (grey = hidden part) with the proxy's visible box (green) and
    visible samples' area, for the most occluded saved jobs."""
    import cv2

    img = OUT / "render/img"
    jobs = [int(j) for j in d["job"]]
    have = [i for i, j in enumerate(jobs) if (img / f"rgb_{j:04d}.jpg").exists() and d["n_amodal"][i] > 300]
    have.sort(key=lambda i: vr[i])
    tiles = []
    for i in have[:8]:
        j = jobs[i]
        rgb = cv2.imread(str(img / f"rgb_{j:04d}.jpg"))
        m = cv2.cvtColor(cv2.imread(str(img / f"mask_{j:04d}.png"), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        b = v["box_vis"][i]
        for im in (rgb, m):
            if np.isfinite(b).all():
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        cv2.putText(
            m,
            f"render vis {vr[i]:.2f}  proxy {vp[i]:.2f}  tt {int(d['tt'][i])}",
            (6, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )
        tiles.append(cv2.resize(np.concatenate([rgb, m], 1), (640, 240)))
    if tiles:
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate([np.concatenate(tiles[k : k + 2], 1) for k in range(0, len(tiles), 2)], 0)
        cv2.imwrite(str(OUT / "proxy_examples.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
