"""Compare Isaac replays (gate_play.py traces) with the recorded encoders: tracking, hold error by approach direction,
lost motion at reversals, and the grasp point at the gripper-close frame.

    sim/twin/dynamics/py.sh sim/twin/dynamics/val_isaac.py p_none p_compensate  -> out/dynamics/val_isaac.json
Trace q = the sim's true joints (ours units), s_real = the recorded states, goal = the replayed actions.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.dynamics import data as D  # noqa: E402
from sim.twin.dynamics import ident as I  # noqa: E402
from sim.twin.dynamics.fit import lost_motion_all, replay  # noqa: E402
from sim.twin.dynamics.gravity import TIP_LOCAL, Arm, sim_q  # noqa: E402

GATE = REPO / "sim/twin/out/dynamics/gate"


def holds_by_dir(cmd, sm, L):
    out = {}
    for j, jn in enumerate(D.JOINTS[:5]):
        ev = []
        for i in range(len(L)):
            d = {"cmd": cmd[i, :L[i]], "meas": sm[i, :L[i]]}
            ev += [(s, d["meas"][t, j] - d["cmd"][t, j]) for t, s, _ in I.holds(d, j)]
        ev = np.array(ev) if ev else np.zeros((0, 2))
        out[jn] = [round(float(np.median(ev[ev[:, 0] > 0, 1])), 3) if (ev[:, 0] > 0).any() else None,
                   round(float(np.median(ev[ev[:, 0] < 0, 1])), 3) if (ev[:, 0] < 0).any() else None]
    return out


def main(tags):
    arm, off = Arm(), D.w_offsets()
    res = {}
    real_done = False
    for tag in tags:
        z = np.load(GATE / tag / "trace.npz")
        q, s, g = z["q"].astype(float), z["s_real"].astype(float), z["goal"].astype(float)
        L = (z["end"] - z["start"] + 1).astype(int)
        L = np.minimum(L, q.shape[1])
        c = (z["cf"] - z["start"]).astype(int)
        mask = np.arange(q.shape[1])[None] < L[:, None]
        err = (q - s)[mask]
        idx = np.arange(len(c))
        tl = z["fin_tip_local"] if "fin_tip_local" in z else TIP_LOCAL  # the gate's fin tip (fixed finger)
        tq = arm.point(sim_q(q[idx, c], off), "gripper_link", tl)
        ts = arm.point(sim_q(s[idx, c], off), "gripper_link", tl)
        dz = (tq - ts)[:, 2] * 1000
        r = {"n": int(len(c)), "rmse_deg": dict(zip(D.JOINTS, np.round(np.sqrt((err ** 2).mean(0)), 3).tolist(),
                                                     strict=True)),
             "close_joint_err_deg_median": dict(zip(D.JOINTS, np.round(np.median(q[idx, c] - s[idx, c], 0), 3)
                                                    .tolist(), strict=True)),
             "close_tip_dz_mm": {"median": round(float(np.median(dz)), 2),
                                 "p10_p90": np.round(np.percentile(dz, [10, 90]), 2).tolist()},
             "close_tip_z_above_table_mm": {"sim": round(float(np.median(tq[:, 2] - z["table_z"])) * 1000, 2),
                                            "fk_recorded": round(float(np.median(ts[:, 2] - z["table_z"])) * 1000, 2)},
             "fin_tip_local": z["fin_tip_local"].round(4).tolist(),
             "hold_by_dir_sim": holds_by_dir(g, q, L),
             "lost_motion_sim": np.round(lost_motion_all(g, q, L), 3).tolist()}
        # Isaac vs the same model run without physics (numpy / torch replay of the same goals from the same start):
        # what PhysX adds (drive lag, contacts), time-locked to the close frame
        P = json.loads((REPO / "sim/twin/dynamics/params.json").read_text())
        if tag == "p_none":
            off_m = replay(g, s[:, 0], arm, off, "A")
            tau = np.zeros_like(off_m)
            tau[..., :5] = arm.gravity_torque(sim_q(off_m, off))
            off_m = off_m + np.rad2deg(tau / 60.0)
        else:
            off_m = replay(g, s[:, 0], arm, off, "B", {k: np.array(P[k]) for k in ("c0", "c", "d0", "m", "s")})
        win = np.arange(-45, 31)
        dl, dzt = [], []
        for i in range(len(c)):
            t = c[i] + win
            okw = (t >= 0) & (t < L[i])
            row = np.full((len(win), 5), np.nan)
            row[okw] = q[i, t[okw], :5] - off_m[i, t[okw], :5]
            dl.append(row)
            zz = np.full(len(win), np.nan)
            zz[okw] = (arm.point(sim_q(q[i, t[okw]], off), "gripper_link", tl)[:, 2]
                       - arm.point(sim_q(off_m[i, t[okw]], off), "gripper_link", tl)[:, 2]) * 1000
            dzt.append(zz)
        dl, dzt = np.array(dl), np.array(dzt)
        r["isaac_minus_nophysics"] = {"ticks_from_close": win[::5].tolist(),
                                      "lift_deg_median": np.round(np.nanmedian(dl[:, ::5, 1], 0), 3).tolist(),
                                      "elbow_deg_median": np.round(np.nanmedian(dl[:, ::5, 2], 0), 3).tolist(),
                                      "fin_tip_dz_mm_median": np.round(np.nanmedian(dzt[:, ::5], 0), 2).tolist()}
        if not real_done:
            res["real"] = {"hold_by_dir": holds_by_dir(g, s, L),
                           "lost_motion": np.round(lost_motion_all(g, s, L), 3).tolist()}
            real_done = True
        res[tag] = r
        print(tag, json.dumps(r), flush=True)
    print("real", json.dumps(res["real"]))
    (REPO / "sim/twin/out/dynamics/val_isaac.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
