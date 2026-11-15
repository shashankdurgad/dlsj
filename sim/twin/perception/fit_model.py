"""Fit the simulated wrist detector (detector_model.py) to the real detector's behaviour on recorded pick phases.

    python sim/twin/perception/fit_model.py        (plain python + torch CPU; after gt_real.py)
    -> sim/twin/perception/detector_model.json  (model parameters, versioned with the code)
       sim/twin/out/perception/fit_report.json  (tables behind the report: rates vs time-to-close / visibility / ...)

Detection model (a hurdle: does the target get a box at the operating threshold, and with what confidence):
    s_t = x_t . beta_thr + b_ep + e_t,   a box with conf >= thr is output iff s_t > 0
    e_t = rho e_{t-1} + sqrt(1 - rho^2) eps_t   (unit variance; frame-to-frame correlation of the misses)
    b_ep: per-brick difficulty (colour on wood, glare, kind); the sim draws it from the fitted episodes' empirical
          distribution (41 quantiles; skewed: a tail of bricks the detector hardly ever sees)
    conf | box = clip(x_t . gamma + sigma_c n, thr, 0.79)   (the detector saturates at ~0.78)
x = [1, vis, occ_grip, oof, log(area/1e4), log(area/1e4)^2, blur, blur^2, closing (grip < 12), dist, dist^2, gripper
closing speed / 3 units per frame]
(dist = (camera-brick distance - 12 cm) / 5 cm), blur = image speed of
the brick centre / 10 px per frame. Probit with per-episode intercepts (MAP, Gaussian prior with fitted scale), torch;
rho by matching the real mean miss-run length in simulation; one fit per threshold (0.5 deployed, 0.3, 0.15).
Rows: wrist-fitted episodes (all frames) + fingertip-fallback episodes only within 15 frames of the close (where
that ground truth is good); 20 frames after the close are included (the residual can still act then).

Box model (matched boxes of wrist-fitted episodes): centre offset / size of the real box relative to the GEOMETRIC
VISIBLE box (in-frame, unoccluded part), normalised by its size; mean and std as linear functions of (1 - vis).
Pose: `Geometry.box_to_rel` with the ray-plane height (above the brick-centre height) that minimises the real
centre error; reported against the wrist-fitted brick centre.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.perception import geometry as G  # noqa: E402

OUT = REPO / "sim/twin/out/perception"
MODEL_JSON = Path(__file__).resolve().parent / "detector_model.json"
CENS = 0.05
FEATS = [
    "one",
    "vis",
    "occ_grip",
    "oof",
    "log_area",
    "log_area2",
    "blur",
    "blur2",
    "closing",
    "dist",
    "dist2",
    "grip_closing",
]
THRESHOLDS = (0.5, 0.3, 0.15)


def grip_speed(grip, ep):
    """Gripper closing speed (units per frame, >= 0) along each episode's rows (sorted by frame)."""
    v = np.r_[0.0, -np.diff(grip)]
    v[np.r_[True, ep[1:] != ep[:-1]]] = 0.0
    return np.clip(v, 0, None)


def features(vis, occ, oof, area_px, speed, grip, dist, gclose=None):
    """Design matrix (n, len(FEATS)) from proxy / rendered visibility + kinematics (numpy)."""
    sp = np.nan_to_num(np.clip(speed, 0, 60), nan=5.0) / 10.0
    la = np.log(np.clip(area_px, 300, 3e5) / 1e4)
    dd = (np.clip(dist, 0.03, 0.4) - 0.12) / 0.05
    gc = np.clip(np.zeros_like(vis) if gclose is None else gclose, 0, 6) / 3.0
    return np.stack(
        [np.ones_like(vis), vis, occ, oof, la, la * la, sp, sp**2, (grip < 12).astype(float), dd, dd * dd, gc], -1
    )


def fit_probit(X, y, ep, iters=800):
    """MAP probit with per-episode intercepts b ~ N(0, s_b): returns beta, b per episode, s_b, episode ids."""
    import torch

    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y.astype(float), dtype=torch.float64)
    ue, ei = np.unique(ep, return_inverse=True)
    eit = torch.tensor(ei)
    beta = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(len(ue), dtype=torch.float64, requires_grad=True)
    ls_b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([beta, b, ls_b], lr=0.05)
    nd = torch.distributions.Normal(0.0, 1.0)
    for _ in range(iters):
        opt.zero_grad()
        mu = Xt @ beta + b[eit]
        ll = (yt * torch.log(nd.cdf(mu).clamp(min=1e-12)) + (1 - yt) * torch.log(nd.cdf(-mu).clamp(min=1e-12))).sum()
        lp = (nd.log_prob(b / ls_b.exp()) - ls_b).sum()
        loss = -(ll + lp) / len(y)
        loss.backward()
        opt.step()
    return beta.detach().numpy(), b.detach().numpy(), float(ls_b.detach().exp()), ue


def sim_runs(mu, ep, frame, rho, rng):
    """Mean miss-run length of the latent AR(1) probit with means mu (rows sorted by episode, frame)."""
    e = np.zeros(len(mu))
    prev_ep, prev_f, x = None, None, 0.0
    eps = rng.standard_normal(len(mu))
    for i in range(len(mu)):
        if ep[i] != prev_ep:
            x = eps[i]
        else:
            r = rho ** max(1, frame[i] - prev_f)
            x = r * x + np.sqrt(1 - r * r) * eps[i]
        e[i] = x
        prev_ep, prev_f = ep[i], frame[i]
    det = mu + e > 0
    return miss_runs(det, ep)


def miss_runs(det, ep):
    out, c = [], 0
    for i in range(len(det)):
        if i and ep[i] != ep[i - 1] and c:
            out.append(c)
            c = 0
        if not det[i]:
            c += 1
        elif c:
            out.append(c)
            c = 0
    if c:
        out.append(c)
    return float(np.mean(out)) if out else 0.0


def fit_scores(d, use, rng=None):
    """{"det": {thr: {beta, sigma_b, b_quantiles, rho, b_by_episode}}, "conf": {gamma, sigma}} on rows `use`."""
    rng = rng or np.random.default_rng(0)
    X = features(
        d["vis"],
        d["occ_grip"],
        d["oof"],
        d["area_px"],
        d["speed"],
        d["grip"],
        d["dist"],
        grip_speed(d["grip"], d["ep"]),
    )
    order = np.lexsort((d["frame"], d["ep"]))
    use_o = order[use[order]]
    out = {"det": {}}
    for thr in THRESHOLDS:
        y = d["conf"] >= thr
        beta, b_ep, s_b, ue = fit_probit(X[use_o], y[use_o], d["ep"][use_o])
        bmap = dict(zip(ue.tolist(), b_ep.tolist(), strict=True))
        mu = X[use_o] @ beta + np.array([bmap[e] for e in d["ep"][use_o]])
        real = miss_runs(y[use_o], d["ep"][use_o])
        best = min(
            (
                (abs(sim_runs(mu, d["ep"][use_o], d["frame"][use_o], r, rng) - real), r)
                for r in (0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97)
            )
        )
        out["det"][str(thr)] = {
            "beta": beta.round(4).tolist(),
            "sigma_b": round(s_b, 4),
            "rho": best[1],
            "b_quantiles": np.percentile(b_ep, np.linspace(0, 100, 41)).round(4).tolist(),
            "miss_run_real": round(real, 2),
            "b_by_episode": {str(int(e)): round(float(v), 4) for e, v in bmap.items()},
        }
    k = use & (d["conf"] >= 0.15)
    A = X[k]
    g = np.linalg.lstsq(A, d["conf"][k], rcond=None)[0]
    out["conf"] = {"gamma": g.round(4).tolist(), "sigma": round(float(np.std(d["conf"][k] - A @ g)), 4), "max": 0.79}
    return out


def wrong_target() -> dict | None:
    """Markov wrong-lock parameters from eval_selector.py's run on the real boxes (selector.json), if present."""
    p = OUT / "selector.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    rate, run = s["overall"]["wrong"], max(s["wrong_runs_mean"], 1.0)
    q = np.array(s["wrong_offset_mm_p10_50_90"]) / 1e3
    p_exit = 1.0 / run
    return {
        "p_enter": round(rate * p_exit / max(1 - rate, 1e-6), 5),
        "p_exit": round(p_exit, 4),
        "offset_median_m": round(float(q[1]), 4),
        "offset_log_sd": round(float(np.log(q[2] / q[0]) / 2.563), 3),
        "conf": 0.6,
        "source": "selector.json (TargetTracker on the real boxes)",
    }


def main():
    d = dict(np.load(OUT / "real_table.npz"))
    fits = json.loads((OUT / "real_fit.json").read_text())["episodes"]
    ok = d["fit_ok"].astype(bool)
    tt = d["tt"]
    use = ((ok & (tt >= -20) & (tt <= 75)) | (~ok & (tt >= -20) & (tt <= 15))) & (d["area_px"] > 0)
    conf = d["conf"]
    sc = fit_scores(d, use)
    # the same correlation for the binary outcome (detected at 0.5), reported for the validation
    det = conf >= 0.5
    pairs_all = use[1:] & use[:-1] & (d["ep"][1:] == d["ep"][:-1]) & (np.diff(d["frame"]) == 1)
    p_stay_miss = float(np.mean(~det[1:][pairs_all & ~det[:-1]]))
    p_stay_det = float(np.mean(det[1:][pairs_all & det[:-1]]))

    # ---------------------------------------------------------------- box model
    m = (
        ok
        & (tt >= -20)
        & (tt <= 75)
        & d["matched"].astype(bool)
        & np.isfinite(d["det_box"]).all(1)
        & np.isfinite(d["box_vis"]).all(1)
    )
    gb, db = d["box_vis"][m], d["det_box"][m]
    gw, gh = np.maximum(gb[:, 2] - gb[:, 0], 4), np.maximum(gb[:, 3] - gb[:, 1], 4)
    dx = ((db[:, 0] + db[:, 2]) - (gb[:, 0] + gb[:, 2])) / 2 / gw
    dy = ((db[:, 1] + db[:, 3]) - (gb[:, 1] + gb[:, 3])) / 2 / gh
    lw = np.log(np.maximum(db[:, 2] - db[:, 0], 4) / gw)
    lh = np.log(np.maximum(db[:, 3] - db[:, 1], 4) / gh)
    occ1 = 1 - d["vis"][m]
    box = {}
    for name, v in (("dx", dx), ("dy", dy), ("lw", lw), ("lh", lh)):
        v = np.clip(v, -1.5, 1.5)
        A = np.stack([np.ones_like(occ1), occ1], 1)
        coef = np.linalg.lstsq(A, v, rcond=None)[0]
        r = v - A @ coef
        # robust scale (MAD) and its dependence on occlusion
        a2 = np.linalg.lstsq(A, np.abs(r), rcond=None)[0] * 1.2533
        box[name] = {"mean": coef.round(4).tolist(), "std": a2.round(4).tolist()}
    mm = np.where(m)[0]
    pr = (d["ep"][mm][1:] == d["ep"][mm][:-1]) & (np.diff(d["frame"][mm]) == 1)
    box_rho = float(np.corrcoef(dx[:-1][pr], dx[1:][pr])[0, 1])
    # amodal vs visible: which box does the real detector report when the brick is partly hidden?
    part = m & (d["vis"] < 0.85)
    ga = d["box_amodal"][part]
    ga = np.stack(
        [np.clip(ga[:, 0], 0, 640), np.clip(ga[:, 1], 0, 480), np.clip(ga[:, 2], 0, 640), np.clip(ga[:, 3], 0, 480)], 1
    )
    from sim.twin.perception.gt_real import iou

    iou_vis = float(np.median(iou(d["det_box"][part], d["box_vis"][part])))
    iou_amo = float(np.median(iou(d["det_box"][part], ga)))

    # ---------------------------------------------------------------- pose from the box (ray / plane)
    geo = G.Geometry()
    from sim.twin.align import frames as F

    states = {}
    rows = np.where(m)[0]
    jj = np.zeros((len(rows), 6))
    for k, i in enumerate(rows):
        e = int(d["ep"][i])
        if e not in states:
            states[e] = F.states(e)
        jj[k] = states[e][int(d["frame"][i])]
    from sim.twin.perception.gt_real import brick_dims

    gt = np.array(
        [
            [
                fits[str(int(e))]["x"],
                fits[str(int(e))]["y"],
                G.TABLE_Z + brick_dims(fits[str(int(e))]["kind"], fits[str(int(e))]["pose"])[1],
            ]
            for e in d["ep"][rows]
        ]
    )
    p_tip, Rg = geo.tip(jj)
    rel_gt = ((gt - p_tip)[:, None, :] @ Rg)[:, 0, :]
    best = None
    for dz in np.arange(-0.004, 0.0161, 0.002):
        rel, P = geo.box_to_rel(d["det_box"][rows], jj, gt[:, 2] + dz, centre_z=gt[:, 2])
        err = np.linalg.norm(P[:, :2] - gt[:, :2], axis=1)
        med = float(np.median(err))
        if best is None or med < best[0]:
            best = (med, dz, rel)
    plane_dz = float(best[1])
    rel, P = geo.box_to_rel(d["det_box"][rows], jj, gt[:, 2] + plane_dz, centre_z=gt[:, 2])
    # the same estimator on the GEOMETRIC visible box: the part of the error that is geometry, not detector
    relg, Pg = geo.box_to_rel(d["box_vis"][rows], jj, gt[:, 2] + plane_dz, centre_z=gt[:, 2])
    e_det = np.linalg.norm(rel - rel_gt, axis=1)
    e_geo = np.linalg.norm(relg - rel_gt, axis=1)
    pose_tt = []
    for lo, hi in ((-20, -1), (0, 4), (5, 9), (10, 19), (20, 39), (40, 75)):
        k = (d["tt"][rows] >= lo) & (d["tt"][rows] <= hi)
        if k.sum() < 10:
            continue
        pose_tt.append(
            {
                "tt": [lo, hi],
                "n": int(k.sum()),
                "err_mm_median": round(float(np.median(e_det[k])) * 1e3, 1),
                "err_mm_p90": round(float(np.percentile(e_det[k], 90)) * 1e3, 1),
                "bias_mm": (np.median(rel[k] - rel_gt[k], 0) * 1e3).round(1).tolist(),
                "geo_box_err_mm_median": round(float(np.median(e_geo[k])) * 1e3, 1),
            }
        )

    # ---------------------------------------------------------------- tables
    def rate_table(key, edges, mask):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            k = mask & (d[key] >= lo) & (d[key] < hi)
            if k.sum() < 20:
                continue
            c = conf[k]
            out.append(
                {
                    key: [lo, hi],
                    "n": int(k.sum()),
                    "det@0.5": round(float(np.mean(c >= 0.5)), 3),
                    "det@0.3": round(float(np.mean(c >= 0.3)), 3),
                    "conf_med_if_box": round(float(np.median(c[c >= CENS])), 3) if (c >= CENS).any() else None,
                    "vis": round(float(d["vis"][k].mean()), 3),
                    "occ_grip": round(float(d["occ_grip"][k].mean()), 3),
                    "oof": round(float(d["oof"][k].mean()), 3),
                }
            )
        return out

    tables = {
        "tt_wristfit": rate_table("tt", [-20, 0, 3, 6, 10, 15, 20, 30, 45, 76], ok & use),
        "tt_tipfallback": rate_table("tt", [-20, 0, 3, 6, 10, 16], ~ok & use),
        "vis": rate_table("vis", [0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.01], use),
        "occ_grip": rate_table("occ_grip", [0, 0.005, 0.05, 0.1, 0.2, 0.4, 1.01], use),
        "oof": rate_table("oof", [0, 0.005, 0.05, 0.2, 0.5, 1.01], use),
        "speed": rate_table("speed", [0, 3, 6, 10, 15, 20, 30, 1e9], use & (d["vis"] > 0.9)),
        "area_px": rate_table("area_px", [0, 3000, 6000, 12000, 25000, 50000, 1e9], use & (d["vis"] > 0.9)),
        "dist": rate_table("dist", [0, 0.07, 0.09, 0.12, 0.16, 0.22, 1], use & (d["vis"] > 0.9)),
        "ang": rate_table("ang", [0, 10, 20, 30, 40, 90], use & (d["vis"] > 0.9)),
    }
    # false positives on the fins: real boxes (conf >= 0.5) not on the target that overlap a fin zone
    model = {
        "doc": "sim/twin/perception/detector_model.py; fitted by fit_model.py on the real brick-detector-v1 "
        "(OpenVINO CPU FP32) over recorded cal2 wrist frames, pick phase",
        "features": FEATS,
        "threshold": 0.5,
        "det": {k: {kk: vv for kk, vv in v.items() if kk != "b_by_episode"} for k, v in sc["det"].items()},
        "conf": sc["conf"],
        "box": box,
        "box_rho": round(box_rho, 3),
        "plane_dz": round(plane_dz, 4),
        "camera": "physical",
        "min_side_px": 12,
        "latency_frames": 1,
        "period_frames": 1,
        "wrong_target": wrong_target(),
        "n_rows": int(use.sum()),
        "n_episodes": int(len(set(d["ep"][use].tolist()))),
    }
    MODEL_JSON.write_text(json.dumps(model, indent=1))
    rep = {
        "model": model,
        "tables": tables,
        "pose_vs_tt": pose_tt,
        "markov_det@0.5": {"p_miss_after_miss": round(p_stay_miss, 3), "p_det_after_det": round(p_stay_det, 3)},
        "box_partial": {
            "n": int(part.sum()),
            "median_iou_with_visible_box": round(iou_vis, 3),
            "median_iou_with_amodal_box": round(iou_amo, 3),
        },
        "box_stats_all": {
            k: [round(float(np.median(v)), 3), round(float(np.percentile(np.abs(v - np.median(v)), 68)), 3)]
            for k, v in (("dx", dx), ("dy", dy), ("lw", lw), ("lh", lh))
        },
        "episodes": {
            "wrist_fit": int(sum(1 for f in fits.values() if f.get("src") == "wrist")),
            "tip_fallback": int(sum(1 for f in fits.values() if f.get("src") == "tip")),
        },
        "per_episode_b": {t: v["b_by_episode"] for t, v in sc["det"].items()},
    }
    (OUT / "fit_report.json").write_text(json.dumps(rep, indent=1))
    print(
        json.dumps(
            {k: rep[k] for k in ("model", "pose_vs_tt", "markov_det@0.5", "box_partial", "box_stats_all")}, indent=1
        )
    )


if __name__ == "__main__":
    main()


def refit(ep_filter) -> dict:
    """detector_model.json with the detection / confidence model refitted on the episodes for which ep_filter(ep)
    is True (box / plane parameters kept): for held-out validation."""
    d = dict(np.load(OUT / "real_table.npz"))
    ok = d["fit_ok"].astype(bool)
    tt = d["tt"]
    sel = np.array([bool(ep_filter(int(e))) for e in d["ep"]])
    use = ((ok & (tt >= -20) & (tt <= 75)) | (~ok & (tt >= -20) & (tt <= 15))) & (d["area_px"] > 0) & sel
    sc = fit_scores(d, use)
    p = json.loads(MODEL_JSON.read_text())
    p["det"] = {k: {kk: vv for kk, vv in v.items() if kk != "b_by_episode"} for k, v in sc["det"].items()}
    p["conf"] = sc["conf"]
    return p
