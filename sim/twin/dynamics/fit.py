"""Fit the play model's per-joint parameters on the recorded holds and check it by open-loop replay (numpy, no Isaac).

    sim/twin/dynamics/py.sh sim/twin/dynamics/fit.py            -> out/dynamics/fit.json, replay.json, plots
                                                                   (and a params.json proposal: --write-params)

Physical model of an STS3215 joint at rest (P-only position loop, LeRobot P=16 / I=0, geared output with load-
dependent friction): the joint settles where Kp (cmd - meas) + tau_g + friction = 0 with |friction| <= F0 + mu |tau_g|,
i.e. inside a band around the load-shifted command,
    meas - cmd = c0 + c * tau  -  dir * (d0 + m * |tau|)          dir = +1 when the command came from below
c = 1/Kp (deg per N m: the encoder-visible sag), d0 = load-free half band (firmware deadzone + gear friction),
m = c * mu (load-dependent half band), c0 = residual constant offset. Fitted by least squares on the hold samples of
ident.py (teleop: every kept cal2 episode; traces: the deployed controllers), episode-block bootstrap.

Replay (the check): every kept cal2 episode's recorded actions through
  A  the twin's current servo model (servo_params.json: first-order K, velocity limit, constant bias; clamp off),
  B  A + the play model (bias replaced by c0 + c tau, band memory d0 + m |tau|),
compared with the recorded encoders: open-loop RMSE, hold error by approach direction, lost motion at reversals, and
the grasp point (TIP_LOCAL) at each episode's gripper-close frame (grasp_gate.common.close_frame) vs FK(recorded state).
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
from sim.twin.dynamics import ident as I  # noqa: E402
from sim.twin.dynamics.gravity import TIP_LOCAL, Arm, sim_q  # noqa: E402

OUT = REPO / "sim/twin/out/dynamics"
SERVO = REPO / "sim/realism/servo_params.json"


def fit_band(rows, use_tau=True, nboot=200, seed=0):
    keys = np.array([r[0] for r in rows])
    e = np.array([r[1] for r in rows])
    tau = np.array([r[2] for r in rows])
    s = np.array([r[3] for r in rows])

    def X(idx):
        cols = [np.ones(len(idx)), -s[idx]]
        if use_tau:
            cols += [tau[idx], -s[idx] * np.abs(tau[idx])]
        return np.stack(cols, 1)

    def solve(idx):
        b, *_ = np.linalg.lstsq(X(idx), e[idx], rcond=None)
        return b

    allidx = np.arange(len(e))
    b = solve(allidx)
    r = e - X(allidx) @ b
    uk = np.unique(keys)
    groups = {k: np.where(keys == k)[0] for k in uk}
    rng = np.random.default_rng(seed)
    bs = np.array([solve(np.concatenate([groups[k] for k in rng.choice(uk, len(uk))])) for _ in range(nboot)])
    names = ["c0", "d0"] + (["c", "m"] if use_tau else [])
    return {"n": int(len(e)), "n_seq": int(len(uk)), **{k: round(float(v), 4) for k, v in zip(names, b, strict=True)},
            "ci95": {k: [round(float(np.percentile(bs[:, i], 2.5)), 4), round(float(np.percentile(bs[:, i], 97.5)), 4)]
                     for i, k in enumerate(names)},
            "resid_sd": round(float(r.std()), 3), "resid_mad": round(float(np.median(np.abs(r))), 3)}


def fit_pooled(rows_by_joint, nboot=200, seed=0):
    """Shared c, m (one servo P gain / gear friction law) over the gravity-loaded joints, per-joint c0 and d0."""
    js = list(rows_by_joint)

    def build(sel):
        Xs, ys = [], []
        for i, j in enumerate(js):
            rr = [rows_by_joint[j][k] for k in sel[j]]
            if not rr:
                continue
            e = np.array([r[1] for r in rr])
            tau = np.array([r[2] for r in rr])
            s = np.array([r[3] for r in rr])
            X = np.zeros((len(e), 2 + 2 * len(js)))
            X[:, 0], X[:, 1] = tau, -s * np.abs(tau)
            X[:, 2 + 2 * i], X[:, 3 + 2 * i] = 1.0, -s
            Xs.append(X)
            ys.append(e)
        return np.concatenate(Xs), np.concatenate(ys)

    full = {j: np.arange(len(rows_by_joint[j])) for j in js}
    X, y = build(full)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    rng = np.random.default_rng(seed)
    keys = {j: np.array([r[0] for r in rows_by_joint[j]]) for j in js}
    uk = np.unique(np.concatenate(list(keys.values())))
    bs = []
    for _ in range(nboot):
        pick = rng.choice(uk, len(uk))
        cnt = {k: c for k, c in zip(*np.unique(pick, return_counts=True), strict=True)}
        sel = {j: np.concatenate([np.repeat(np.where(keys[j] == k)[0], c) for k, c in cnt.items()]
                                 + [np.zeros(0, int)]).astype(int) for j in js}
        Xb, yb = build(sel)
        bs.append(np.linalg.lstsq(Xb, yb, rcond=None)[0])
    bs = np.array(bs)
    ci = np.percentile(bs, [2.5, 97.5], 0)
    out = {"c": round(float(b[0]), 4), "m": round(float(b[1]), 4),
           "ci95": {"c": np.round(ci[:, 0], 4).tolist(), "m": np.round(ci[:, 1], 4).tolist()},
           "resid_sd": round(float((y - X @ b).std()), 3)}
    for i, j in enumerate(js):
        out[j] = {"c0": round(float(b[2 + 2 * i]), 4), "d0": round(float(b[3 + 2 * i]), 4),
                  "ci95": {"c0": np.round(ci[:, 2 + 2 * i], 4).tolist(), "d0": np.round(ci[:, 3 + 2 * i], 4).tolist()}}
    return out


# ---------------------------------------------------------------------------------------------- open-loop replay
def load_servo():
    p = json.loads(SERVO.read_text())["joints"]
    J = D.JOINTS
    return (np.array([p[j]["first_order_K_per_tick"] for j in J]),
            np.array([p[j]["velocity_limit_deg_s"] / 30 for j in J]),
            np.array([p[j]["steady_state_bias_deg"] for j in J]))


def replay(cmd, meas0, arm, off, mode, P=None):
    """cmd (E, T, 6) padded with the last action, meas0 (E, 6) -> simulated joints (E, T, 6), ours units.
    mode "A": servo model as the twin; "B": with the play model P = {c0, c, d0, m} (6,) arrays."""
    K, vmax, bias = load_servo()
    E, T, _ = cmd.shape
    out = np.zeros_like(cmd)
    if mode == "A":
        r = meas0 - bias
        for t in range(T):
            out[:, t] = r + bias
            r = r + np.clip(K * (cmd[:, t] - r), -vmax, vmax)
        return out
    import torch

    from sim.twin.dynamics.play import Play, PlayCfg

    cfg = PlayCfg(drive_stiffness=float("inf"), encoder=False,
                  overrides={k: np.asarray(v).tolist() for k, v in P.items()})
    play = Play(cfg, E, off, device="cpu", arm=arm)
    K, vmax = torch.tensor(K, dtype=torch.float32), torch.tensor(vmax, dtype=torch.float32)
    c = torch.tensor(cmd, dtype=torch.float32)
    r = torch.tensor(meas0, dtype=torch.float32)
    play.reset(torch.arange(E), r.clone())
    for t in range(T):
        out[:, t] = r.numpy()
        tgt = play.target(c[:, t], r)
        r = r + torch.clamp(K * (tgt - r), -vmax, vmax)
    return out


def lost_motion_all(cmd, sm, L):
    """Median lost motion (deg) per arm joint over every reversal of every episode."""
    out = []
    for j in range(5):
        v = []
        for i in range(len(L)):
            c, m = cmd[i, :L[i], j], sm[i, :L[i], j]
            for t, s in reversals(c):
                x = lost_motion(c, m, t, s)
                if x is not None:
                    v.append(x)
        out.append(np.median(v))
    return np.array(out)


def write_params(P, fits):
    path = Path(__file__).resolve().parent / "params.json"
    p = json.loads(path.read_text())
    for k in ("c0", "c", "d0", "m", "s"):
        p[k] = np.round(P[k], 4).tolist()
    p["_fit"] = {"pooled_teleop": fits["pooled_teleop"], "pooled_trace": fits["pooled_trace"]}
    path.write_text(json.dumps(p, indent=1))
    print("wrote", path)


def reversals(c, prom=1.0, win=45):
    """Indices of command extrema with >= prom deg of travel before and after (within win ticks)."""
    out = []
    for t in range(1, len(c) - 1):
        a, b = c[max(0, t - win):t], c[t + 1:t + 1 + win]
        if len(a) == 0 or len(b) == 0:
            continue
        if c[t] >= a.max() and c[t] >= b.max() and c[t] - a.min() >= prom and c[t] - b.min() >= prom:
            out.append((t, 1))
        elif c[t] <= a.min() and c[t] <= b.min() and a.max() - c[t] >= prom and b.max() - c[t] >= prom:
            out.append((t, -1))
    dedup = []
    for t, s in out:
        if not dedup or t - dedup[-1][0] > 3:
            dedup.append((t, s))
    return dedup


def lost_motion(c, m, t, s, k=2, win=45):
    """After the command reverses at t (s=+1: a maximum), how far the command has come back when the joint has come
    back k counts from its own extremum (deg); None if it never does within win ticks."""
    seg_m = m[t:t + win]
    if len(seg_m) < 3:
        return None
    ext = seg_m.max() if s > 0 else seg_m.min()
    i_ext = t + int(np.argmax(seg_m) if s > 0 else np.argmin(seg_m))
    for u in range(i_ext, min(len(m), t + win)):
        if (ext - m[u]) * s >= k * I.COUNT * 0.99:
            return float((c[t] - c[u]) * s)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--write-params", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    arm, off = Arm(), D.w_offsets()
    tele = {k: v for k, v in D.teleop().items() if not v["excluded"]}
    tr = D.traces()
    I.add_tau(tele, arm, off)
    I.add_tau(tr, arm, off)
    fits = {}
    for j, jn in enumerate(D.JOINTS):
        use_tau = jn in ("shoulder_lift", "elbow_flex", "wrist_flex")
        fits[jn] = {}
        for src, seqs in (("teleop", tele), ("trace", tr)):
            rows = I.steady_table(seqs, j)
            if jn == "gripper":  # free space only (no brick between the jaws)
                rows = [r for r in rows if r[4] > 15]
            fits[jn][src] = fit_band(rows, use_tau, args.nboot) if len(rows) > 20 else None
            # long holds only (>= 1 s): is the band a settling artefact?
            long_rows = [r for r in rows if r[5] >= 30]
            fits[jn][src + "_hold>=1s"] = fit_band(long_rows, use_tau, 50) if len(long_rows) > 20 else None
        print(jn, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "ci95"} if v else None
                              for k, v in fits[jn].items()}), flush=True)
    loaded = ("shoulder_lift", "elbow_flex", "wrist_flex")
    for src, seqs in (("teleop", tele), ("trace", tr)):
        fits[f"pooled_{src}"] = fit_pooled({jn: I.steady_table(seqs, D.JOINTS.index(jn)) for jn in loaded}, args.nboot)
        print(f"pooled_{src}", json.dumps(fits[f"pooled_{src}"]), flush=True)
    (OUT / "fit.json").write_text(json.dumps(fits, indent=1))

    # play parameters: pooled c, m on the loaded joints (teleop), per-joint c0 / d0 (pooled fit there, else per joint)
    P = {k: np.zeros(6) for k in ("c0", "c", "d0", "m", "s")}
    pool = fits["pooled_teleop"]
    for j, jn in enumerate(D.JOINTS):
        if jn in loaded:
            P["c"][j], P["m"][j] = pool["c"], pool["m"]
            P["c0"][j], P["d0"][j] = pool[jn]["c0"], pool[jn]["d0"]
        else:
            f = fits[jn]["teleop"]
            P["c0"][j], P["d0"][j] = f["c0"], f["d0"]
    print("P", {k: np.round(v, 3).tolist() for k, v in P.items()}, flush=True)

    # ------------------------------------------------------------------ replay all kept episodes
    sys.path.insert(0, str(REPO / "sim/twin/grasp_gate"))
    from sim.twin.grasp_gate.common import close_frame

    keys = sorted(tele)
    T = max(len(tele[k]["cmd"]) for k in keys)
    cmd = np.stack([np.pad(tele[k]["cmd"], ((0, T - len(tele[k]["cmd"])), (0, 0)), mode="edge") for k in keys])
    meas = np.stack([np.pad(tele[k]["meas"], ((0, T - len(tele[k]["meas"])), (0, 0)), mode="edge") for k in keys])
    L = np.array([len(tele[k]["cmd"]) for k in keys])
    mask = np.arange(T)[None] < L[:, None]
    # breakaway margin s per joint: match the real lost motion at reversals (grid, all joints at once)
    real_lm = lost_motion_all(cmd, meas, L)
    grid, lms = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], []
    for sv in grid:
        Ps = dict(P, s=np.full(6, sv))
        lms.append(lost_motion_all(cmd, replay(cmd, meas[:, 0], arm, off, "B", Ps), L))
        print("s", sv, np.round(lms[-1], 2).tolist(), "real", np.round(real_lm, 2).tolist(), flush=True)
    lms = np.array(lms)
    best = np.argmin(np.abs(lms - real_lm[None]), 0)
    P["s"] = np.r_[[grid[b] for b in best], 0.0]
    rep_s = {"grid": grid, "lost_motion_median": lms.round(3).tolist(), "real": real_lm.round(3).tolist()}
    print("P", {k: np.round(v, 3).tolist() for k, v in P.items()}, flush=True)
    sims = {"A": replay(cmd, meas[:, 0], arm, off, "A"), "B": replay(cmd, meas[:, 0], arm, off, "B", P),
            "B_no_s": replay(cmd, meas[:, 0], arm, off, "B", dict(P, s=np.zeros(6)))}
    # A as Isaac runs it: the arm PD drive (60 N m/rad, residual_env/scene.py) sags by tau / k on top of r + bias
    tauA = np.zeros_like(sims["A"])
    tauA[..., :5] = arm.gravity_torque(sim_q(sims["A"], off))
    sims["A_isaac"] = sims["A"] + np.rad2deg(tauA / 60.0)
    rep = {"rmse_deg": {}, "hold_err_by_dir": {}, "lost_motion_deg": {}, "close_frame": {}, "s_grid": rep_s}
    for name, sm in [("real", meas)] + list(sims.items()):
        if name != "real":
            err = (sm - meas)[mask]
            rep["rmse_deg"][name] = dict(zip(D.JOINTS, np.round(np.sqrt((err ** 2).mean(0)), 3).tolist(), strict=True))
        hd, lm = {}, {}
        for j, jn in enumerate(D.JOINTS[:5]):
            ev, lms = [], []
            for i in range(len(keys)):
                d = {"cmd": cmd[i, :L[i]], "meas": sm[i, :L[i]]}
                for t, s, _ in I.holds(d, j):
                    ev.append((s, d["meas"][t, j] - d["cmd"][t, j]))
                for t, s in reversals(d["cmd"][:, j]):
                    v = lost_motion(d["cmd"][:, j], d["meas"][:, j], t, s)
                    if v is not None:
                        lms.append(v)
            ev = np.array(ev)
            hd[jn] = {"from_below_median": round(float(np.median(ev[ev[:, 0] > 0, 1])), 3),
                      "from_above_median": round(float(np.median(ev[ev[:, 0] < 0, 1])), 3), "n": int(len(ev))}
            lm[jn] = {"median": round(float(np.median(lms)), 3), "p90": round(float(np.percentile(lms, 90)), 3),
                      "n": len(lms)}
        rep["hold_err_by_dir"][name] = hd
        rep["lost_motion_deg"][name] = lm
    # grasp point at the gripper-close frames
    cf = []
    for k in keys:
        f = close_frame(tele[k]["meas"][:, 5], tele[k]["inspection_frame"])
        cf.append(f if f is not None else -1)
    cf = np.array(cf)
    ok = cf > 0
    idx = np.where(ok)[0]
    tip = {}
    for name, sm in [("real", meas)] + list(sims.items()):
        q = sim_q(sm[idx, cf[idx]], off)
        tip[name] = arm.point(q, "gripper_link", TIP_LOCAL)
    for name in sims:
        dlt = (tip[name] - tip["real"]) * 1000
        rep["close_frame"][name] = {"n": int(len(idx)), "dz_mm_median": round(float(np.median(dlt[:, 2])), 2),
                                    "dz_mm_p10_p90": np.round(np.percentile(dlt[:, 2], [10, 90]), 2).tolist(),
                                    "dxy_mm_median": np.round(np.median(dlt[:, :2], 0), 2).tolist(),
                                    "joint_err_deg_median": dict(zip(D.JOINTS[:5], np.round(np.median(
                                        sims[name][idx, cf[idx], :5] - meas[idx, cf[idx], :5], 0), 3).tolist(),
                                        strict=True))}
    rep["close_frame"]["real_cmd_minus_meas_deg_median"] = dict(zip(D.JOINTS[:5], np.round(np.median(
        cmd[idx, cf[idx], :5] - meas[idx, cf[idx], :5], 0), 3).tolist(), strict=True))
    tau_cf = arm.gravity_torque(sim_q(meas[idx, cf[idx]], off))
    rep["close_frame"]["tau_real_median"] = np.round(np.median(tau_cf, 0), 3).tolist()
    rep["params_B"] = {k: np.round(v, 4).tolist() for k, v in P.items()}
    if args.write_params:
        write_params(P, fits)
    (OUT / "replay.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    np.savez_compressed(OUT / "replay_tips.npz", **{f"tip_{k}": v for k, v in tip.items()}, cf=cf, keys=np.array(keys))
    plot(keys, cmd, meas, sims, L, P, OUT)


def plot(keys, cmd, meas, sims, L, P, out):
    import matplotlib.pyplot as plt

    i = int(np.argmax(L > 300)) if (L > 300).any() else 0
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    tt = np.arange(L[i]) / 30
    for a, j in zip(ax, (1, 2, 3), strict=True):
        a.plot(tt, cmd[i, :L[i], j], color="#9ca3af", lw=1, label="command (leader)")
        a.plot(tt, meas[i, :L[i], j], color="k", lw=1.6, label="real encoder")
        a.plot(tt, sims["A"][i, :L[i], j], color="#2563eb", lw=1, label="twin servo model (A)")
        a.plot(tt, sims["B"][i, :L[i], j], color="#dc2626", lw=1, label="+ play model (B)")
        a.set_ylabel(f"{D.JOINTS[j]} (deg)")
    ax[0].legend(fontsize=8, ncol=4)
    ax[-1].set_xlabel(f"time (s), cal2 episode {keys[i]}")
    fig.tight_layout()
    fig.savefig(out / "replay_episode.png", dpi=110)


if __name__ == "__main__":
    main()
