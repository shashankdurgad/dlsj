"""Per-joint real-vs-twin response table from twin_replay.py outputs (plain numpy; no Isaac).

    sim/twin/dynamics/py.sh sim/twin/servo_fit/compare.py g_grip1:0 g_arm1:3 ... [--fit]
    -> sim/twin/out/servo_fit/compare.json (+ a printed table)
<tag>:<set> picks one parameter set of a replay. Trace windows (deployed controller, LeRobot clamp 10) and teleop
windows (no clamp) are scored separately with metrics.py; --fit adds the first-order + delay fit (slow).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.servo_fit import metrics as M  # noqa: E402

OUT = REPO / "sim/twin/out/servo_fit"


def load(tag_set: str):
    tag, _, s = tag_set.partition(":")
    s = int(s or 0)
    Z = np.load(OUT / tag / "replay.npz")
    R = json.loads((OUT / tag / "result.json").read_text())
    W = R["W"]
    e = np.arange(s * W, (s + 1) * W)
    names = Z["names"]
    return names, Z["srcs"], Z["meas"][e].astype(np.float64), R["sets"][s], R["args"], R["W"]


def score(cmd, meas, clamp, fit):
    out = {}
    for j in range(5):
        r = {"lag": M.lag_xcorr(cmd, meas, j, clamp)[0], "vmax": M.vmax(cmd, meas, j, clamp),
             "stops": M.stops(cmd, meas, j), "small": M.small_steps(cmd, meas, j)}
        if fit:
            ks = np.arange(0.1, 0.91, 0.05)
            r["fo"] = M.fo_fit(cmd, meas, j, clamp, Ks=ks, vs=(3, 4, 5, 6, 8, 1e9))
            r["fo_small"] = M.fo_fit(cmd, meas, j, clamp, H=15, Ks=ks, vs=(1e9,), move=1.0, move_hi=5.0)
        out[M.JOINTS[j]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--out", default="compare.json")
    a = ap.parse_args()
    Wz = np.load(OUT / "windows.npz")
    res = {}
    real_done = {}
    for run in a.runs:
        names, srcs, Y, s, args, W = load(run)
        nw = len(Wz["name"])  # the same window subset as twin_replay.py --n_windows
        idx = np.linspace(0, nw - 1, args["n_windows"]).round().astype(int) if args.get("n_windows") else np.arange(nw)
        assert len(idx) == W and (Wz["name"][idx] == names).all()
        T = Y.shape[1]
        C, R = Wz["cmd"][idx, :T].astype(np.float64), Wz["meas"][idx, :T].astype(np.float64)
        tele = np.array([str(x) == "teleop" for x in srcs])
        rr = {"set": s, "args": {k: args[k] for k in ("env", "play")}}
        for part, k, clamp in (("trace", ~tele, 10.0), ("teleop", tele, 1e9)):
            if not k.any():
                continue
            key = (part, tuple(idx[k]))
            if key not in real_done:
                real_done[key] = score(list(C[k]), list(R[k]), clamp, a.fit)
            rr[part] = {"real": real_done[key], "twin": score(list(C[k]), list(Y[k]), clamp, a.fit),
                        "rmse": np.sqrt(((Y[k] - R[k]) ** 2).mean((0, 1))).round(3).tolist(),
                        "rmse_vel": np.sqrt(((np.diff(Y[k], axis=1) - np.diff(R[k], axis=1)) ** 2).mean((0, 1)))
                        .round(3).tolist(),
                        "bias": (Y[k] - R[k]).mean((0, 1)).round(3).tolist()}
        res[run] = rr
        for part in ("trace", "teleop"):
            if part not in rr:
                continue
            print(f"== {run} {json.dumps(s)} [{part}] rmse {rr[part]['rmse']} vel {rr[part]['rmse_vel']}")
            for j in M.JOINTS[:5]:
                re, tw = rr[part]["real"][j], rr[part]["twin"][j]

                def f(d):
                    st, sm = d["stops"] or {}, (d["small"] or {}).get("1-2") or {}
                    sm2 = (d["small"] or {}).get("2-3.5") or {}
                    fo, fs = d.get("fo") or {}, d.get("fo_small") or {}
                    return (f"lag {d['lag']} fo d{fo.get('delay_ticks')} K{fo.get('K')} v{fo.get('v_per_tick')} "
                            f"small d{fs.get('delay_ticks')} K{fs.get('K')} rmse{fs.get('rmse')} "
                            f"n{fs.get('n_windows')} | "
                            f"vmax99 {(d['vmax'] or {}).get('p99')} | ov50 {st.get('overshoot_p50')} "
                            f"ov90 {st.get('overshoot_p90')} end {st.get('end_err_p50')} "
                            f"rise {st.get('rise_10_90_ticks_p50')}"
                            f" | small1-2 frac {sm.get('achieved_frac_p50')} t1 {sm.get('first_move_ticks_p50')} "
                            f"t63 {sm.get('t63_ticks_p50')} | 2-3.5 frac {sm2.get('achieved_frac_p50')} "
                            f"t63 {sm2.get('t63_ticks_p50')}")
                print(f"  {j:14s} REAL {f(re)}\n  {'':14s} TWIN {f(tw)}")
    (OUT / a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
