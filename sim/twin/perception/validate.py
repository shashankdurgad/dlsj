"""Validate the simulated detector against the real one on the same recorded pick phases.

    python sim/twin/perception/validate.py [--seeds 40] [--holdout]      (plain python)
    -> sim/twin/out/perception/validate.json, validate_timeline.png, validate_examples.jpg

For every recorded episode (the rows of real_table.npz: the recorded joints, the wrist-fitted brick pose, the proxy's
visibility of that brick in every frame) the sim detector is run frame by frame (`DetectorModel.observe`, numpy, many
seeds) and compared with what the real detector did on the real frames: detection rate (conf >= 0.5) vs time to the
gripper close, miss run lengths, confidence, and the box -> pose error. --holdout: refit on the even episodes and
validate on the odd ones. Side-by-side examples (real frame with the real box | the proxy's fin / brick geometry with
the sim box) for frames where the fins hide the brick.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.perception import geometry as G  # noqa: E402
from sim.twin.perception.detector_model import BRICK_CENTRE_Z, DetectorModel, load_params  # noqa: E402

OUT = REPO / "sim/twin/out/perception"
TT_BINS = [-20, -10, 0, 3, 6, 10, 15, 20, 30, 45, 76]


def episode_vis(d, rows):
    return {
        "vis": d["vis"][rows],
        "occ_grip": d["occ_grip"][rows],
        "oof": d["oof"][rows],
        "dist": d["dist"][rows],
        "area_px": d["area_px"][rows],
        "uv": np.stack([d["u"][rows], d["v"][rows]], -1),
        "box_vis": d["box_vis"][rows],
        "front": np.ones(len(rows), bool),
    }


def runs(x):
    """Lengths of runs of True in a 1-D bool array."""
    out, c = [], 0
    for v in x:
        if v:
            c += 1
        elif c:
            out.append(c)
            c = 0
    if c:
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--own-b", action="store_true", help="use each episode's fitted difficulty (conditional check)")
    a = ap.parse_args()
    from sim.twin.align import frames as F

    d = dict(np.load(OUT / "real_table.npz"))
    fits = json.loads((OUT / "real_fit.json").read_text())["episodes"]
    params = load_params()
    if a.holdout:
        from sim.twin.perception import fit_model

        params = fit_model.refit(lambda e: e % 2 == 0)
    import os

    if os.environ.get("DLSJ_B_SCALE"):
        for v in params["det"].values():
            v["b_scale"] = float(os.environ["DLSJ_B_SCALE"])
            v["b_shift"] = float(os.environ.get("DLSJ_B_SHIFT", "0"))
    ok = d["fit_ok"].astype(bool)
    tt = d["tt"]
    use = ((ok & (tt >= -20) & (tt <= 75)) | (~ok & (tt >= -20) & (tt <= 15))) & (d["area_px"] > 0)
    eps = sorted(set(d["ep"][use].tolist()))
    if a.holdout:
        eps = [e for e in eps if e % 2 == 1]
    geo = G.Geometry()
    own_b = json.loads((OUT / "fit_report.json").read_text())["per_episode_b"]["0.5"]
    S = a.seeds
    rng = np.random.default_rng(0)
    real_det, sim_det, tts, src_ok = [], [], [], []
    real_err, sim_err, err_tt = [], [], []
    miss_real, miss_sim = [], []
    real_conf, sim_conf = [], []
    for e in eps:
        rows = np.where(use & (d["ep"] == e))[0]
        rows = rows[np.argsort(d["frame"][rows])]
        st = F.states(e)
        J = st[d["frame"][rows]]
        vis = episode_vis(d, rows)
        m = DetectorModel(S, params, geo)
        m.reset(None, rng)
        if a.own_b:
            m.b[:] = own_b.get(str(e), 0.0)
        sd = np.zeros((len(rows), S), bool)
        sc = np.zeros((len(rows), S))
        srel = np.full((len(rows), S, 3), np.nan)
        prev_f = None
        for i in range(len(rows)):
            v1 = {k: np.repeat(v[i : i + 1], S, 0) for k, v in vis.items()}
            f = d["frame"][rows[i]]
            dt = 1.0 if prev_f is None else float(f - prev_f)
            prev_f = f
            sp = np.full(S, d["speed"][rows[i]] if np.isfinite(d["speed"][rows[i]]) else 5.0)
            o = m.observe(np.repeat(J[i : i + 1], S, 0), v1, rng, dt_frames=dt, speed_px=sp)
            tgt = o["target_detected"]  # the detector's target box (before the tracker's wrong-lock model)
            sd[i], sc[i] = tgt, o["target_conf"]
            srel[i] = o["target_rel"]
        rd = d["conf"][rows] >= 0.5
        real_det.append(rd)
        sim_det.append(sd)
        real_conf.append(d["conf"][rows])
        sim_conf.append(sc[:, 0])
        tts.append(tt[rows])
        src_ok.append(ok[rows])
        miss_real += runs(~rd)
        for s in range(min(S, 10)):
            miss_sim += runs(~sd[:, s])
        if fits[str(e)].get("src") == "wrist":
            from sim.twin.perception.gt_real import brick_dims

            f_ = fits[str(e)]
            gt = np.array([f_["x"], f_["y"], G.TABLE_Z + brick_dims(f_["kind"], f_["pose"])[1]])
            p_tip, Rg = geo.tip(J)
            rel_gt = ((gt - p_tip)[:, None, :] @ Rg)[:, 0, :]
            db = d["det_box"][rows]
            have = rd & np.isfinite(db).all(1)
            if have.any():
                rr, _ = geo.box_to_rel(db[have], J[have], BRICK_CENTRE_Z + params["plane_dz"], centre_z=BRICK_CENTRE_Z)
                real_err += np.linalg.norm(rr - rel_gt[have], axis=1).tolist()
                err_tt += [("real", t) for t in tt[rows][have]]
            se = np.linalg.norm(srel - rel_gt[:, None, :], axis=-1)
            for i in range(len(rows)):
                v = se[i][np.isfinite(se[i])]
                if len(v):
                    sim_err.append((tt[rows][i], float(np.median(v)), len(v)))
    tt_all = np.concatenate(tts)
    ok_all = np.concatenate(src_ok)
    rd_all = np.concatenate(real_det)
    sd_all = np.concatenate(sim_det)
    rc_all, sc_all = np.concatenate(real_conf), np.concatenate(sim_conf)
    table = []
    for lo, hi in zip(TT_BINS[:-1], TT_BINS[1:], strict=True):
        k = (tt_all >= lo) & (tt_all < hi)
        if k.sum() < 20:
            continue
        kw_, kt_ = k & ok_all, k & ~ok_all
        row = {
            "tt": [lo, hi],
            "n": int(k.sum()),
            "real": round(float(rd_all[k].mean()), 3),
            "sim": round(float(sd_all[k].mean()), 3),
            "sim_p05_p95_over_seeds": np.percentile(sd_all[k].mean(0), [5, 95]).round(3).tolist(),
            "real_wristfit": round(float(rd_all[kw_].mean()), 3) if kw_.sum() else None,
            "sim_wristfit": round(float(sd_all[kw_].mean()), 3) if kw_.sum() else None,
            "real_tipfallback": round(float(rd_all[kt_].mean()), 3) if kt_.sum() > 20 else None,
            "sim_tipfallback": round(float(sd_all[kt_].mean()), 3) if kt_.sum() > 20 else None,
            "real_conf_med_if_box": round(float(np.median(rc_all[k][rc_all[k] >= 0.05])), 3)
            if (rc_all[k] >= 0.05).any()
            else None,
            "sim_conf_med_if_box": round(float(np.median(sc_all[k][sc_all[k] >= 0.05])), 3)
            if (sc_all[k] >= 0.05).any()
            else None,
        }
        er = np.array([v for (s, t), v in zip(err_tt, real_err, strict=True) if lo <= t < hi])
        es = np.array([v for t, v, _ in sim_err if lo <= t < hi])
        if len(er) > 10 and len(es) > 10:
            row["pose_err_mm_real_med_p90"] = [
                round(float(np.median(er)) * 1e3, 1),
                round(float(np.percentile(er, 90)) * 1e3, 1),
            ]
            row["pose_err_mm_sim_med_p90"] = [
                round(float(np.median(es)) * 1e3, 1),
                round(float(np.percentile(es, 90)) * 1e3, 1),
            ]
        table.append(row)
    mr, ms = np.array(miss_real), np.array(miss_sim)
    res = {
        "episodes": len(eps),
        "seeds": S,
        "holdout": a.holdout,
        "own_b": a.own_b,
        "timeline": table,
        "miss_runs": {
            "real_mean": round(float(mr.mean()), 2),
            "sim_mean": round(float(ms.mean()), 2),
            "real_p90": float(np.percentile(mr, 90)),
            "sim_p90": float(np.percentile(ms, 90)),
        },
        "overall": {"real": round(float(rd_all.mean()), 3), "sim": round(float(sd_all.mean()), 3)},
    }
    tag = ("_holdout" if a.holdout else "") + ("_ownb" if a.own_b else "")
    (OUT / f"validate{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    plot(table, OUT / f"validate_timeline{tag}.png")


def plot(table, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [np.mean(r["tt"]) / 30 for r in table]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, [r["real"] for r in table], "o-", color="#1f6feb", label="real detector (real frames)")
    ax[0].plot(x, [r["sim"] for r in table], "s--", color="#d9480f", label="sim detector (proxy visibility)")
    lo = [r["sim_p05_p95_over_seeds"][0] for r in table]
    hi = [r["sim_p05_p95_over_seeds"][1] for r in table]
    ax[0].fill_between(x, lo, hi, color="#d9480f", alpha=0.15)
    ax[0].set_xlabel("time to gripper close (s)")
    ax[0].set_ylabel("target detected (conf >= 0.5)")
    ax[0].invert_xaxis()
    ax[0].set_ylim(0, 1)
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.3)
    xr = [xx for xx, r in zip(x, table, strict=True) if "pose_err_mm_real_med_p90" in r]
    ax[1].plot(
        xr,
        [r["pose_err_mm_real_med_p90"][0] for r in table if "pose_err_mm_real_med_p90" in r],
        "o-",
        color="#1f6feb",
        label="real, median",
    )
    ax[1].plot(
        xr,
        [r["pose_err_mm_sim_med_p90"][0] for r in table if "pose_err_mm_sim_med_p90" in r],
        "s--",
        color="#d9480f",
        label="sim, median",
    )
    ax[1].plot(
        xr,
        [r["pose_err_mm_real_med_p90"][1] for r in table if "pose_err_mm_real_med_p90" in r],
        "o:",
        color="#1f6feb",
        alpha=0.6,
        label="real, p90",
    )
    ax[1].plot(
        xr,
        [r["pose_err_mm_sim_med_p90"][1] for r in table if "pose_err_mm_sim_med_p90" in r],
        "s:",
        color="#d9480f",
        alpha=0.6,
        label="sim, p90",
    )
    ax[1].set_xlabel("time to gripper close (s)")
    ax[1].set_ylabel("brick-centre error from the box (mm)")
    ax[1].invert_xaxis()
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
