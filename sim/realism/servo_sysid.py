"""System identification of the SO-101 follower from recorded teleop data (action = commanded target, state = measured).

Per joint, per dataset (calibration set):
  * delay d (ticks) + first-order gain K:  s[t+1] = s[t] + K * (a[t-d] - s[t])         (least squares per d)
  * velocity-limited first order:          ... clipped to +-vmax per tick
  * second order (PD-like):                v[t+1] = v[t] + kp*(a[t-d]-s[t]) - kd*v[t];  s[t+1] = s[t] + v[t+1]
  validated by multi-step open-loop rollout on held-out episodes vs baselines (s = a[t-1]; hold s0).
  * velocity / acceleration envelope, steady-state tracking error, measurement noise.
Gripper: closing speed, holding plateau and squeeze (command - measured) between grasp_frame and inspection, release.
Teleop command smoothness (per-tick |da|).  Controller logs: safety-clamp counts.

    python servo_sysid.py --out ~/inference/sim_realism   (on the XPS; read-only on datasets)
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30
SETS = {
    "old_cal": ("/home/intel/desmond/dlsj/data/duplo_teleop", "/home/intel/inference/data/duplo_teleop_3phase"),
    "new_cal2": ("/home/intel/desmond/build/dlsj-main/data/duplo_teleop_cal2",
                 "/home/intel/inference/data/duplo_teleop_cal2_3phase"),
}
LOGS = ["/home/intel/inference/logs/molmo/control.log", "/home/intel/inference/logs/smolvla/control.log",
        *sorted(glob.glob("/home/intel/desmond/pi05_control*.log"))]


def load(root: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    fs = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))
    df = pd.concat(pd.read_parquet(f, columns=["action", "observation.state", "episode_index", "frame_index"]) for f in fs)
    out = {}
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        out[int(ep)] = (np.stack(g["action"].values).astype(np.float64), np.stack(g["observation.state"].values).astype(np.float64))
    return out


def labels(root3: str) -> dict[int, dict]:
    p = Path(root3) / "meta/dlsj/labels.jsonl"
    return {int(l["episode_index"]): l for l in map(json.loads, p.read_text().splitlines()) if l} if p.exists() else {}


# ----------------------------------------------------------------------------- models
def pairs(eps, j, d):
    """x = a[t-d] - s[t], y = s[t+1] - s[t] (t >= d), stacked over episodes."""
    xs, ys, vs = [], [], []
    for a, s in eps:
        if len(s) <= d + 2:
            continue
        t = np.arange(d, len(s) - 1)
        xs.append(a[t - d, j] - s[t, j]); ys.append(s[t + 1, j] - s[t, j])
        v = np.zeros(len(s)); v[1:] = np.diff(s[:, j]); vs.append(v[t])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(vs)


def fit_first(eps, j, dmax=6):
    best = None
    for d in range(dmax + 1):
        x, y, _ = pairs(eps, j, d)
        k = float((x * y).sum() / max((x * x).sum(), 1e-9))
        rmse = float(np.sqrt(np.mean((y - k * x) ** 2)))
        if best is None or rmse < best["one_step_rmse"]:
            best = {"d": d, "K": k, "one_step_rmse": rmse}
    return best


def fit_second(eps, j, d):
    """dv = kp*e - kd*v  (v = s[t]-s[t-1]); y here is v[t+1] - v[t]."""
    xs_e, xs_v, ys = [], [], []
    for a, s in eps:
        if len(s) <= d + 3:
            continue
        t = np.arange(max(d, 1), len(s) - 1)
        v = np.zeros(len(s)); v[1:] = np.diff(s[:, j])
        xs_e.append(a[t - d, j] - s[t, j]); xs_v.append(v[t]); ys.append(v[t + 1] - v[t])
    X = np.stack([np.concatenate(xs_e), -np.concatenate(xs_v)], 1); y = np.concatenate(ys)
    (kp, kd), *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(kp), float(kd)


def rollout(a, s0, j, model):
    """Open-loop multi-step prediction of joint j from s0 given the whole command sequence."""
    T = len(a); p = np.empty(T); p[0] = s0; v = 0.0
    d = model["d"]
    for t in range(T - 1):
        e = a[max(t - d, 0), j] - p[t]
        if model["kind"] == "first":
            step = model["K"] * e
        elif model["kind"] == "first_vlim":
            step = np.clip(model["K"] * e, -model["vmax"], model["vmax"])
        else:
            v = v + model["kp"] * e - model["kd"] * v
            step = np.clip(v, -model["vmax"], model["vmax"]); v = step
        p[t + 1] = p[t] + step
    return p


def evaluate(eps, j, models):
    err = {k: [] for k in [*models, "baseline_cmd_prev", "baseline_hold"]}
    for a, s in eps:
        if len(s) < 10:
            continue
        for name, m in models.items():
            err[name].append(rollout(a, s[0, j], j, m) - s[:, j])
        base = np.r_[s[0, j], a[:-1, j]]
        err["baseline_cmd_prev"].append(base - s[:, j])
        err["baseline_hold"].append(np.full(len(s), s[0, j]) - s[:, j])
    return {k: round(float(np.sqrt(np.mean(np.concatenate(v) ** 2))), 3) for k, v in err.items()}


def stationary_mask(a, j, win=10, tol=0.3):
    """Ticks where the command has been ~constant for `win` ticks (steady state)."""
    da = np.abs(np.diff(a[:, j], prepend=a[0, j]))
    c = np.convolve(da > tol, np.ones(win), "full")[: len(da)]
    m = c == 0
    m[:win] = False
    return m


# ----------------------------------------------------------------------------- per set
def analyse(name, root, root3, rng):
    data = load(root)
    lab = labels(root3)
    eps_ids = sorted(data)
    rng.shuffle(eps_ids)
    n_val = max(10, len(eps_ids) // 10)
    val_ids, train_ids = eps_ids[:n_val], eps_ids[n_val:]
    train = [data[e] for e in train_ids]; val = [data[e] for e in val_ids]
    res = {"episodes": len(data), "train": len(train), "val": len(val), "joints": {}}
    for j, jn in enumerate(J):
        f1 = fit_first(train, j)
        x, y, _ = pairs(train, j, f1["d"])
        vel = np.abs(np.concatenate([np.diff(s[:, j]) for _, s in train]))
        acc = np.abs(np.concatenate([np.diff(s[:, j], 2) for _, s in train if len(s) > 3]))
        vmax = float(np.percentile(vel, 99.5))
        kp, kd = fit_second(train, j, f1["d"])
        models = {"first": {"kind": "first", "d": f1["d"], "K": f1["K"]},
                  "first_vlim": {"kind": "first_vlim", "d": f1["d"], "K": f1["K"], "vmax": vmax},
                  "second": {"kind": "second", "d": f1["d"], "kp": kp, "kd": kd, "vmax": vmax}}
        rm = evaluate(val, j, models)
        best = min(models, key=lambda k: rm[k])
        sse, noise = [], []
        for a, s in train:
            m = stationary_mask(a, j)
            if not m.any():
                continue
            sse.append(s[m, j] - a[m, j])
            # measurement noise: spread around the median within each CONTIGUOUS steady run of >= 8 ticks
            idx = np.flatnonzero(m)
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for r in runs:
                if len(r) >= 8:
                    seg = s[r, j]
                    noise.append(seg - np.median(seg))
        sse = np.concatenate(sse) if sse else np.zeros(1)
        noise_std = float(np.std(np.concatenate(noise))) if noise else 0.0
        tau_ticks = float(-1.0 / np.log(max(1e-6, 1 - f1["K"]))) if 0 < f1["K"] < 1 else None
        res["joints"][jn] = {
            "delay_ticks": f1["d"], "K_first_order": round(f1["K"], 4),
            "time_constant_s": round(tau_ticks / FPS, 4) if tau_ticks else None,
            "second_order_kp_kd": [round(kp, 4), round(kd, 4)],
            "vmax_deg_per_tick_p99_5": round(vmax, 3), "vmax_deg_s_p99_5": round(vmax * FPS, 1),
            "vel_max_deg_s": round(float(vel.max()) * FPS, 1),
            "acc_p99_deg_s2": round(float(np.percentile(acc, 99)) * FPS * FPS, 1),
            "steady_err_median": round(float(np.median(sse)), 3),
            "steady_abs_err_p50_p95": [round(float(np.percentile(np.abs(sse), 50)), 3),
                                       round(float(np.percentile(np.abs(sse), 95)), 3)],
            "meas_noise_std": round(noise_std, 4),
            "val_rollout_rmse": rm, "best_model": best,
            "cmd_delta_p50_p95_p99_max": [round(float(v), 3) for v in np.percentile(
                np.abs(np.concatenate([np.diff(a[:, j]) for a, _ in train])), [50, 95, 99, 100])],
        }
    # gripper events
    g = J.index("gripper"); hold, squeeze, close_v, open_v, open_after = [], [], [], [], []
    for ep, (a, s) in data.items():
        l = lab.get(ep, {})
        gf, ins = l.get("grasp_frame"), l.get("inspection_frame")
        if not gf or not ins or ins - gf < 15 or l.get("excluded"):
            continue
        w = slice(gf + 8, ins)
        hold.append(float(np.median(s[w, g]))); squeeze.append(float(np.median(a[w, g] - s[w, g])))
        lo = max(0, gf - 30)
        close_v.append(float(np.diff(s[lo:gf + 3, g]).min()) if gf + 3 - lo > 2 else np.nan)
        tail = s[ins:, g]
        if len(tail) > 5:
            open_v.append(float(np.diff(tail).max())); open_after.append(float(tail.max()))
    pct = lambda v: [round(float(x), 2) for x in np.nanpercentile(v, [5, 25, 50, 75, 95])] if len(v) else None  # noqa: E731
    res["gripper"] = {"episodes": len(hold), "hold_measured_p5_25_50_75_95": pct(hold),
                      "squeeze_cmd_minus_meas_p5_25_50_75_95": pct(squeeze),
                      "close_speed_min_dstate_per_tick_p5_50_95": pct(close_v),
                      "open_speed_max_dstate_per_tick_p5_50_95": pct(open_v),
                      "open_level_after_release_p5_50_95": pct(open_after)}
    res["_val_ids"] = val_ids[:20]
    return res, data, val


def clamp_counts():
    out = {}
    for p in LOGS:
        try:
            txt = Path(p).read_text(errors="ignore")
        except FileNotFoundError:
            continue
        n = txt.count("had to be clamped")
        per = Counter(re.findall(r"'(shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll|gripper)': \{\s*'original goal_pos'", txt))
        lat = [float(x) for x in re.findall(r"inference[^\n]*?([0-9]+(?:\.[0-9]+)?) ms", txt)]
        out[Path(p).name if "desmond" in p else p.split("/logs/")[-1]] = {
            "clamp_warnings": n, "clamped_joints": dict(per),
            "latency_ms_median": round(float(np.median(lat)), 1) if lat else None, "latency_n": len(lat)}
    return out


def plots(outdir: Path, name, val, res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    a, s = val[0]
    fig, ax = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for j, jn in enumerate(J):
        r = res["joints"][jn]
        m = {"kind": "first_vlim", "d": r["delay_ticks"], "K": r["K_first_order"], "vmax": r["vmax_deg_per_tick_p99_5"]}
        t = np.arange(len(s)) / FPS
        ax[j].plot(t, a[:, j], "k:", lw=1, label="command")
        ax[j].plot(t, s[:, j], "b", lw=1.2, label="measured")
        ax[j].plot(t, rollout(a, s[0, j], j, m), "r--", lw=1, label="model (open loop)")
        ax[j].set_ylabel(jn, fontsize=8)
    ax[0].legend(fontsize=7, loc="upper right"); ax[-1].set_xlabel("s")
    fig.suptitle(f"{name}: held-out episode, command vs measured vs fitted model")
    f = outdir / f"sysid_{name}_rollout.png"; fig.tight_layout(); fig.savefig(f, dpi=90); plt.close(fig); files.append(f.name)
    return files


def to_params(results):
    """Simulator-facing parameters (from the larger, old-calibration set; cal2 kept for reference)."""
    r = results["old_cal"]
    joints = {}
    for jn in J:
        x = r["joints"][jn]
        K = x["K_first_order"]
        joints[jn] = {
            "delay_ticks": x["delay_ticks"], "first_order_K_per_tick": K, "time_constant_s": x["time_constant_s"],
            "velocity_limit_deg_s": x["vmax_deg_s_p99_5"], "accel_p99_deg_s2": x["acc_p99_deg_s2"],
            "steady_state_bias_deg": x["steady_err_median"], "steady_state_abs_err_p95_deg": x["steady_abs_err_p50_p95"][1],
            "meas_noise_std": x["meas_noise_std"], "second_order_kp_kd_per_tick": x["second_order_kp_kd"],
            "units": "gripper 0-100 range units; others LeRobot degrees",
            "isaac_hint": "PD position drive at sim dt=1/30 s: emulate first-order K by stiffness/damping; add delay_ticks "
                          "command FIFO; clip joint velocity to velocity_limit_deg_s; add Gaussian meas_noise_std to "
                          "observed state",
        }
    g = r["gripper"]
    return {"source": "recorded teleop (duplo_teleop 517 eps old calibration; duplo_teleop_cal2 for comparison)",
            "control_rate_hz": FPS, "joints": joints,
            "gripper": {"hold_position_median": g["hold_measured_p5_25_50_75_95"][2],
                        "hold_position_p5_p95": [g["hold_measured_p5_25_50_75_95"][0], g["hold_measured_p5_25_50_75_95"][4]],
                        "squeeze_cmd_minus_meas_median": g["squeeze_cmd_minus_meas_p5_25_50_75_95"][2],
                        "close_speed_units_per_tick_median": g["close_speed_min_dstate_per_tick_p5_50_95"][2],
                        "open_speed_units_per_tick_median": g["open_speed_max_dstate_per_tick_p5_50_95"][2]},
            "teleop_action_rate_p99_deg_per_tick": {jn: r["joints"][jn]["cmd_delta_p50_p95_p99_max"][2] for jn in J}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path("~/inference/sim_realism").expanduser()))
    args = ap.parse_args()
    out = Path(args.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results, figs = {}, []
    for name, (root, root3) in SETS.items():
        res, data, val = analyse(name, root, root3, rng)
        results[name] = res
        figs += plots(out, name, val, res)
    results["controller_logs"] = clamp_counts()
    (out / "servo_sysid_raw.json").write_text(json.dumps(results, indent=1))
    (out / "servo_params.json").write_text(json.dumps(to_params(results), indent=1))
    print(json.dumps({"figs": figs}, indent=1))


if __name__ == "__main__":
    main()
