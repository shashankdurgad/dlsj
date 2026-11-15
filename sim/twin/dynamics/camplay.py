"""Link motion the encoders do not see, from the wrist camera (plain python + PyAV; never in an Isaac process).

    sim/twin/dynamics/py.sh sim/twin/dynamics/camplay.py [--n 80]   -> out/dynamics/camplay.json, camplay.npz, plots

The wrist camera rides on gripper_link, so its view of the table moves with every link, horn and joint between the
base and the gripper. Per consecutive frame pair: measured global image shift of the upper band of the image (table,
plates; the red fingers and a held brick are below it) by phase correlation, vs the shift predicted from the
encoders: FK of the recorded states (twin offsets = joint fit W, wrist mount + intrinsics from joint_fit/result.json),
table-plane points re-projected into the next frame. The residual (measured - predicted) is motion the encoder
readings do not explain (play between servo output and link, link / horn bending, vibration) plus measurement error.
Tests:
* scale: slope of measured vs predicted on moving frames (should be ~1; validates the geometry);
* encoders static (no count change on any arm joint for 2 ticks): residual shift = unseen motion (or noise);
* reversals of pan / lift / wrist_flex: cumulative measured vs predicted shift along the motion over the 10 ticks
  after the command reverses, vs the same ratio in steady motion. Backlash between horn and link makes the camera
  lag the encoder after a reversal (ratio < 1, lag = (1 - ratio) * predicted).
* gripper close (grasp_gate.common.close_frame +-): unseen motion while the jaws squeeze the brick.
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
from sim.twin.align import frames as F  # noqa: E402
from sim.twin.dynamics import data as D  # noqa: E402
from sim.twin.dynamics.gravity import Arm, sim_q  # noqa: E402

OUT = REPO / "sim/twin/out/dynamics"
TABLE_Z = -0.002
ROI = (0, 168)  # rows of the 640x480 image (upper 35 %)


def camera():
    w = json.loads(D.JOINT_FIT.read_text())["wrist_camera_mount"]
    M = np.array(w["local_to_cv"])  # v_cv = M v_local
    return M.T, np.array(w["position_local"]), (w["fx"], w["fy"], w["cx"], w["cy"])


def predicted_shift(arm, off, states, R_gc, p_gc, K):
    """(T-1, 2) mean image shift (640x480 px) of table-plane points of the ROI between consecutive frames."""
    fx, fy, cx, cy = K
    us, vs = np.meshgrid(np.linspace(40, 600, 12), np.linspace(ROI[0] + 10, ROI[1] - 10, 5))
    rays = np.stack([(us.ravel() - cx) / fx, (vs.ravel() - cy) / fy, np.ones(us.size)], 1)
    T = arm.fk(sim_q(states, off))["gripper_link"]
    Rb, pb = T[0], T[1]
    Rc = Rb @ R_gc
    pc = pb + (Rb @ p_gc[:, None])[..., 0]
    out = np.full((len(states) - 1, 2), np.nan)
    for t in range(len(states) - 1):
        d = rays @ Rc[t].T
        lam = (TABLE_Z - pc[t, 2]) / d[:, 2]
        ok = lam > 0
        if ok.sum() < 5:
            continue
        X = pc[t] + lam[ok, None] * d[ok]
        x = (X - pc[t + 1]) @ Rc[t + 1]
        u2 = fx * x[:, 0] / x[:, 2] + cx
        v2 = fy * x[:, 1] / x[:, 2] + cy
        out[t] = [np.mean(u2 - us.ravel()[ok]), np.mean(v2 - vs.ravel()[ok])]
    return out


def measured_shift(frames):
    win = None
    prev = None
    out = []
    for f in frames:
        g = cv2.cvtColor(f[ROI[0]:ROI[1]], cv2.COLOR_RGB2GRAY).astype(np.float32)
        g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2))
        if win is None:
            win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        if prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev, g, win)
            out.append((2 * dx, 2 * dy, resp))
        prev = g
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from sim.twin.grasp_gate.common import close_frame, load_demos

    demos = load_demos()
    eps = sorted(np.random.default_rng(args.seed).choice(demos["keep"], args.n, replace=False).tolist())
    arm, off = Arm(), D.w_offsets()
    R_gc, p_gc, K = camera()
    rec = []
    for i, e in enumerate(eps):
        d = demos["eps"][e]
        S, A = d["state"].astype(float), d["action"].astype(float)
        fr = F.video_frames(e, "wrist", list(range(len(S))))
        if len(fr) < len(S) - 2:
            continue
        L = min(len(S), max(fr) + 1)
        meas = measured_shift([fr[k] for k in range(L)])
        pred = predicted_shift(arm, off, S[:L], R_gc, p_gc, K)
        cf = close_frame(S[:, 5], d["inspection_frame"])
        rec.append({"ep": e, "meas": meas, "pred": pred, "S": S[:L], "A": A[:L], "cf": cf})
        if i % 10 == 0:
            print(i, e, L, flush=True)
    np.savez_compressed(OUT / "camplay.npz",
                        **{f"{k}_{r['ep']}": r[k] for r in rec for k in ("meas", "pred", "S", "A")},
                        eps=np.array([r["ep"] for r in rec]), cf=np.array([r["cf"] or -1 for r in rec]))
    analyse(rec)


def best_lag(rec, lags=(-3, -2, -1, 0, 1, 2, 3)):
    """Camera frame k ticks after the state read (video latency): maximise the measured / predicted correlation."""
    sc = {}
    for k in lags:
        a, b = [], []
        for r in rec:
            m, p = r["meas"][:, :2], r["pred"]
            T = len(p)
            idx = np.arange(max(0, -k), min(T, T - k))
            ok = np.isfinite(p[idx]).all(1)
            a.append(m[idx[ok] + k])
            b.append(p[idx[ok]])
        a, b = np.concatenate(a).ravel(), np.concatenate(b).ravel()
        sc[k] = float(np.corrcoef(a, b)[0, 1])
    return max(sc, key=sc.get), sc


def analyse(rec):
    f = 343.0  # px / rad
    k, sc = best_lag(rec)
    print("video lag (ticks):", k, {kk: round(v, 3) for kk, v in sc.items()}, flush=True)
    for r in rec:  # align: meas[t] = shift of the frames that belong to states t -> t+1
        m = np.full_like(r["meas"], np.nan)
        m[:, 2] = 0.0
        T = len(m)
        if k >= 0:
            m[:T - k] = r["meas"][k:]
        else:
            m[-k:] = r["meas"][:T + k]
        r["meas"] = np.nan_to_num(m, nan=0.0)
    M, P, static, after_stop = [], [], [], []
    for r in rec:
        m, p, S = r["meas"], r["pred"], r["S"]
        dS = np.abs(np.diff(S[:, :5], axis=0))  # (T-1, 5): tick t -> t+1
        ok = np.isfinite(p).all(1) & (m[:, 2] > 0.3) & (np.hypot(*np.nan_to_num(p).T) < 25)
        M.append(m[ok, :2])
        P.append(p[ok])
        still = (dS < 1e-3).all(1)
        grip_still = np.abs(np.diff(S[:, 5])) < 0.2
        for t in range(2, len(m)):
            if ok[t] and still[t] and still[t - 1] and grip_still[t]:
                run = 0
                while t - run - 1 >= 0 and still[t - run - 1]:
                    run += 1
                (static if run >= 10 else after_stop).append((np.hypot(*(m[t, :2] - p[t])), run))
    M, P = np.concatenate(M), np.concatenate(P)
    mv = np.hypot(*P.T) > 3.0
    slope = [float(np.sum(M[mv, k] * P[mv, k]) / np.sum(P[mv, k] ** 2)) for k in range(2)]
    resid = M[mv] - P[mv] * np.array(slope)
    st = np.array(static)
    af = np.array(after_stop)
    res = {"n_eps": len(rec), "scale_slope_xy": np.round(slope, 3).tolist(),
           "moving_resid_px_sd": np.round(resid.std(0), 2).tolist(),
           "static_resid_px": {"n": int(len(st)), "median": round(float(np.median(st[:, 0])), 3),
                               "p90": round(float(np.percentile(st[:, 0], 90)), 3)} if len(st) else None,
           "just_stopped_resid_px": {"n": int(len(af)), "median": round(float(np.median(af[:, 0])), 3),
                                     "p90": round(float(np.percentile(af[:, 0], 90)), 3)} if len(af) else None,
           "px_per_deg": round(f * np.pi / 180, 2), "video_lag_ticks": k,
           "lag_corr": {str(kk): round(v, 3) for kk, v in sc.items()}}
    # reversals: cumulative shift over 10 ticks after the command reverses, along the predicted motion
    rev = {}
    for j, jn in ((0, "shoulder_pan"), (1, "shoulder_lift"), (3, "wrist_flex"), (4, "wrist_roll")):
        ratios, lags, steady = [], [], []
        from sim.twin.dynamics.fit import reversals

        for r in rec:
            m, p, A = r["meas"], r["pred"], r["A"]
            ok = np.isfinite(p).all(1) & (m[:, 2] > 0.3) & (np.hypot(*np.nan_to_num(p).T) < 25)
            revs = reversals(A[:, j], prom=3.0)
            for t, _ in revs:
                w = slice(t, t + 10)
                if t + 10 > len(p) or not ok[w].all():
                    continue
                pc, mc = p[w].sum(0), m[w, :2].sum(0)
                n = np.linalg.norm(pc)
                if n < 6:
                    continue
                u = pc / n
                ratios.append(float(mc @ u / n))
                lags.append(float(n - mc @ u))
            # steady motion control windows: 10 ticks, no reversal within +-15
            rt = [t for t, _ in revs]
            for t in range(15, len(p) - 25, 10):
                if any(abs(t - q) < 25 for q in rt) or not ok[t:t + 10].all():
                    continue
                pc, mc = p[t:t + 10].sum(0), m[t:t + 10, :2].sum(0)
                n = np.linalg.norm(pc)
                if n >= 6:
                    steady.append(float(mc @ (pc / n) / n))
        if ratios:
            rev[jn] = {"n": len(ratios), "ratio_median": round(float(np.median(ratios)), 3),
                       "lag_px_median": round(float(np.median(lags)), 2),
                       "steady_ratio_median": round(float(np.median(steady)), 3) if steady else None}
    res["reversal"] = rev
    # gripper close window: arm encoders static, jaws closing / squeezing
    gc = []
    for r in rec:
        cf = r["cf"]
        if cf is None or cf < 3:
            continue
        m, p, S = r["meas"], r["pred"], r["S"]
        dS = np.abs(np.diff(S[:, :5], axis=0))
        for t in range(max(cf - 5, 1), min(cf + 15, len(m))):
            if (dS[t] < 1e-3).all() and np.isfinite(p[t]).all() and m[t, 2] > 0.2:
                gc.append(np.hypot(*(m[t, :2] - p[t])))
    if gc:
        res["close_window_static_resid_px"] = {"n": len(gc), "median": round(float(np.median(gc)), 3),
                                               "p90": round(float(np.percentile(gc, 90)), 3)}
    (OUT / "camplay.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
