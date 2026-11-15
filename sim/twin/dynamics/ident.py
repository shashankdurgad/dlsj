"""Identify the SO-101 follower's play from the recorded encoders: steady-state error vs gravity torque and approach
direction (sag / compliance seen by the encoder, deadband / backlash), quantisation and rest jitter, small-command
response (stiction), command-to-motion delay, and the gripper's free / holding behaviour.

    sim/twin/dynamics/py.sh sim/twin/dynamics/ident.py [--plots DIR]
    -> sim/twin/out/dynamics/ident.json (+ plots)

Steady state (per joint): the command has been within +-0.15 deg of its current value for 10 ticks (0.33 s, ~3 servo
time constants) and the reading moved <= 1 count over the last 3 ticks; one sample per hold (its last tick).
Approach direction = sign of the command's last move of > 1 deg into the hold. Model per joint, least squares:
    e = meas - cmd = c0 + c_tau * tau_j + h * dir   (+ c_q * q_j for teleop: leader / follower scale mismatch)
tau_j = static gravity torque about joint j (N m, along the axis = the direction gravity drives the angle) at the
measured pose with the twin profile's joint offsets (joint fit W) and the URDF masses + the 20 g wrist camera.
c_tau (deg / N m) is the encoder-visible compliance (servo P error under load); h is half the direction-dependent
band (deadband / backlash the servo loop leaves). Episode-block bootstrap for the CIs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.dynamics import data as D  # noqa: E402
from sim.twin.dynamics.gravity import Arm, sim_q  # noqa: E402

OUT = REPO / "sim/twin/out/dynamics"
COUNT = 360.0 / 4096.0
HOLD, EPS, APPROACH = 10, 0.15, 1.0


def add_tau(seqs, arm, off):
    for d in seqs.values():
        d["tau"] = np.zeros((len(d["meas"]), 6))
        d["tau"][:, :5] = arm.gravity_torque(sim_q(d["meas"], off))


def holds(d, j, hold=HOLD, eps=EPS, approach=APPROACH):
    """[(t, dir, n_hold)]: last tick of every steady hold of joint j."""
    c, m = d["cmd"][:, j], d["meas"][:, j]
    T = len(c)
    ok = np.zeros(T, bool)
    for t in range(hold, T):
        w = c[t - hold:t + 1]
        ok[t] = np.abs(w - c[t]).max() < eps and abs(m[t] - m[t - 3]) <= COUNT * 1.01
    out = []
    t = hold
    while t < T:
        if not ok[t]:
            t += 1
            continue
        e = t
        while e + 1 < T and ok[e + 1] and abs(c[e + 1] - c[t]) < eps:
            e += 1
        start = t - hold
        k = start
        while k >= 0 and abs(c[k] - c[e]) <= approach:
            k -= 1
        if k >= 0:
            out.append((e, float(np.sign(c[e] - c[k])), e - start + 1))
        t = e + 1
    return out


def steady_table(seqs, j):
    rows = []
    for key, d in seqs.items():
        for t, s, n in holds(d, j):
            rows.append((key, d["meas"][t, j] - d["cmd"][t, j], d["tau"][t, j], s, d["meas"][t, j], n, t))
    return rows


def fit(rows, with_q, nboot=200, seed=0):
    if len(rows) < 20:
        return None
    keys = np.array([r[0] for r in rows])
    e = np.array([r[1] for r in rows])
    tau = np.array([r[2] for r in rows])
    s = np.array([r[3] for r in rows])
    q = np.array([r[4] for r in rows])

    def solve(idx):
        X = [np.ones(len(idx)), tau[idx], s[idx]] + ([q[idx] / 100.0] if with_q else [])
        X = np.stack(X, 1)
        b, *_ = np.linalg.lstsq(X, e[idx], rcond=None)
        return b, e[idx] - X @ b

    b, r = solve(np.arange(len(e)))
    uk = np.unique(keys)
    groups = {k: np.where(keys == k)[0] for k in uk}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot):
        idx = np.concatenate([groups[k] for k in rng.choice(uk, len(uk))])
        bs.append(solve(idx)[0])
    bs = np.array(bs)
    names = ["c0_deg", "c_tau_deg_per_Nm", "h_deg"] + (["c_q_per_100deg"] if with_q else [])
    return {"n": int(len(e)), "n_seq": int(len(uk)),
            **{k: round(float(v), 4) for k, v in zip(names, b, strict=True)},
            "ci95": {k: [round(float(np.percentile(bs[:, i], 2.5)), 4), round(float(np.percentile(bs[:, i], 97.5)), 4)]
                     for i, k in enumerate(names)},
            "resid_sd_deg": round(float(r.std()), 3),
            "tau_range_Nm": [round(float(np.percentile(tau, 2)), 3), round(float(np.percentile(tau, 98)), 3)],
            "e_by_dir": {"+": round(float(np.median(e[s > 0])), 3) if (s > 0).any() else None,
                         "-": round(float(np.median(e[s < 0])), 3) if (s < 0).any() else None}}


def quantisation(seqs):
    out = {}
    for j, jn in enumerate(D.JOINTS):
        v = np.concatenate([d["meas"][:, j] for d in seqs.values()])
        dv = np.abs(np.diff(np.concatenate([d["meas"][:, j] for d in seqs.values()])))
        dv = dv[(dv > 1e-6) & (dv < 1.0)]
        step = float(np.median(dv[dv < np.percentile(dv, 20)])) if len(dv) else None
        u = np.unique(np.round(v, 4))
        gaps = np.diff(u)
        out[jn] = {"min_step": round(float(dv.min()), 5) if len(dv) else None, "small_step_median": step,
                   "unique_values": int(len(u)),
                   "median_gap_unique": round(float(np.median(gaps)), 5) if len(gaps) else None}
    return out


def rest_jitter(seqs, hold=30):
    """Command constant (all 6 joints) for >= hold ticks: reading spread over the last hold//2 ticks."""
    out = {jn: [] for jn in D.JOINTS}
    for d in seqs.values():
        c, m = d["cmd"], d["meas"]
        T = len(c)
        t = hold
        while t < T:
            w = c[t - hold:t + 1]
            if (np.abs(w - c[t]).max(0) < 0.05).all():
                seg = m[t - hold // 2:t + 1]
                for j, jn in enumerate(D.JOINTS):
                    out[jn].append((seg[:, j].std(), float(np.mean(np.diff(seg[:, j]) != 0)),
                                    seg[:, j].max() - seg[:, j].min()))
                t += hold
            else:
                t += 1
    res = {}
    for jn, v in out.items():
        if not v:
            continue
        a = np.array(v)
        res[jn] = {"n_windows": len(a), "std_deg_median": round(float(np.median(a[:, 0])), 4),
                   "std_deg_p90": round(float(np.percentile(a[:, 0], 90)), 4),
                   "frac_ticks_changing": round(float(np.mean(a[:, 1])), 3),
                   "range_counts_median": round(float(np.median(a[:, 2]) / COUNT), 2)}
    return res


def delay_fit(seqs, j, delays=(0, 1, 2, 3)):
    """One-step first-order fit ds[t+1] = K (c[t-d] - s[t]) for each pure delay d (arm joints, moving ticks)."""
    res = {}
    for dl in delays:
        X, Y = [], []
        for d in seqs.values():
            c, m = d["cmd"][:, j], d["meas"][:, j]
            if len(c) < dl + 3:
                continue
            x = c[:len(c) - 1 - dl] - m[dl:-1]
            y = m[dl + 1:] - m[dl:-1]
            X.append(x)
            Y.append(y)
        X, Y = np.concatenate(X), np.concatenate(Y)
        sel = np.abs(X) < 20
        X, Y = X[sel], Y[sel]
        K = float(X @ Y / (X @ X))
        res[f"d{dl}"] = {"K": round(K, 4), "rmse": round(float(np.sqrt(np.mean((Y - K * X) ** 2))), 4)}
    return res


def small_steps(seqs, j, lo=0.05, hi=3.0, rest=8, after=12):
    """From rest (cmd and meas constant >= rest ticks), the command moves by D in (lo, hi] and then holds for >= after
    ticks: response ratio (meas change) / D after the hold, binned by |D|."""
    ev = []
    for d in seqs.values():
        c, m = d["cmd"][:, j], d["meas"][:, j]
        T = len(c)
        for t in range(rest, T - after - 1):
            if np.ptp(c[t - rest:t + 1]) > 0.02 or np.ptp(m[t - 3:t + 1]) > COUNT * 1.01:
                continue
            D_ = c[t + 1] - c[t]
            if not (lo < abs(D_) <= hi):
                continue
            if np.ptp(c[t + 1:t + 1 + after]) > 0.02:
                continue
            ev.append((abs(D_), (m[t + after] - m[t]) * np.sign(D_)))
    if not ev:
        return None
    ev = np.array(ev)
    bins = [0.05, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0]
    out = []
    for a, b in zip(bins[:-1], bins[1:], strict=True):
        s = ev[(ev[:, 0] > a) & (ev[:, 0] <= b)]
        if len(s) >= 3:
            out.append({"cmd_step": [a, b], "n": int(len(s)),
                        "moved_frac": round(float(np.mean(s[:, 1] > COUNT * 0.5)), 3),
                        "ratio_median": round(float(np.median(s[:, 1] / s[:, 0])), 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", default=str(OUT))
    ap.add_argument("--nboot", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    arm, off = Arm(), D.w_offsets()
    tele = {k: v for k, v in D.teleop().items() if not v["excluded"]}
    tr = D.traces()
    add_tau(tele, arm, off)
    add_tau(tr, arm, off)
    print(f"teleop {len(tele)} eps {sum(len(d['meas']) for d in tele.values())} frames; "
          f"traces {len(tr)} runs {sum(len(d['meas']) for d in tr.values())} ticks", flush=True)
    res = {"_doc": __doc__, "offsets": off.tolist(), "steady": {}, "delay": {}, "small_steps": {}}
    rows_all = {}
    for j, jn in enumerate(D.JOINTS[:5]):
        rows_all[jn] = {"teleop": steady_table(tele, j), "trace": steady_table(tr, j)}
        res["steady"][jn] = {"teleop": fit(rows_all[jn]["teleop"], True, args.nboot),
                             "trace": fit(rows_all[jn]["trace"], False, args.nboot)}
        res["delay"][jn] = {"teleop": delay_fit(tele, j), "trace": delay_fit(tr, j)}
        res["small_steps"][jn] = {"teleop": small_steps(tele, j), "trace": small_steps(tr, j)}
        print(jn, json.dumps(res["steady"][jn]), flush=True)
    res["quantisation"] = {"teleop": quantisation(tele), "trace": quantisation(tr)}
    res["rest_jitter"] = {"teleop": rest_jitter(tele), "trace": rest_jitter(tr)}
    (OUT / "ident.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT / "steady_rows.npz", **{f"{jn}_{src}": np.array([r[1:] for r in rr], float)
                                                     for jn, v in rows_all.items() for src, rr in v.items() if rr})
    print(json.dumps({k: res[k] for k in ("delay", "small_steps", "quantisation", "rest_jitter")}, indent=1))
    plot(rows_all, Path(args.plots))


def plot(rows_all, out: Path):
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    for col, jn in enumerate(("shoulder_lift", "elbow_flex", "wrist_flex")):
        for row, src in enumerate(("trace", "teleop")):
            rr = rows_all[jn][src]
            a = ax[row, col]
            if not rr:
                continue
            e = np.array([r[1] for r in rr])
            tau = np.array([r[2] for r in rr])
            s = np.array([r[3] for r in rr])
            for sv, colr, lab in ((1, "#c2410c", "approach from below (cmd rising)"),
                                  (-1, "#1d4ed8", "approach from above")):
                k = s == sv
                a.scatter(tau[k], e[k], s=5, alpha=0.35, color=colr, label=lab)
            a.axhline(0, color="k", lw=0.5)
            a.set_title(f"{jn} — {src} holds (n={len(e)})")
            a.set_xlabel("gravity torque about the joint (N m)")
            a.set_ylabel("measured − commanded (deg)")
            a.set_ylim(np.percentile(e, 1) - 1, np.percentile(e, 99) + 1)
            a.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out / "steady_vs_tau.png", dpi=110)
    print("wrote", out / "steady_vs_tau.png")


if __name__ == "__main__":
    main()
