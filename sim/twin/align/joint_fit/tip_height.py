"""Fingertip height at the gripper-close frame of every cal2 episode, for the current and refitted joint models.

Close frame: after the widest opening before the operator's inspection mark, the first frame whose gripper reading is
within 1 unit of the held value (median reading around the inspection mark). At that instant the real jaws are around
the brick (2x2 / 2x4 Duplo: top 19.2 mm, centre 9.6 mm above the table), so the real fingertips are ~5-10 mm above
the table and the env's grasp point TIP_LOCAL (gripper_link (8, 0, -90) mm) ~10 mm above the brick centre
(physics agent: held reading 5.8 <-> brick centre ~10 mm below TIP_LOCAL), i.e. TIP_LOCAL z ~ 20 mm.

    python sim/twin/align/joint_fit/tip_height.py   -> prints a table, writes out/align/joint_fit/tip_height.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import model as M  # noqa: E402
from frames import labels, states  # noqa: E402

TIP_LOCAL = np.array([0.008, 0.0, -0.090])
OUT = M.TWIN / "out/align/joint_fit"


def close_frame(s, insp):
    g = s[:, 5]
    insp = int(insp) if insp is not None else len(g) - 1
    held = np.median(g[max(insp - 5, 0):insp + 6])
    i_open = int(np.argmax(g[:insp]))
    after = np.where(g[i_open:insp] <= held + 1.0)[0]
    return i_open + int(after[0]) if len(after) else None, held


def thetas():
    out = {"joe_cal2 (current)": M.zero_theta()}
    for f, keys in (("fit.json", ("B1", "B3", "C")), ("stage4.json", ("W", "W9", "WS9"))):
        p = OUT / f
        if p.exists():
            th = json.loads(p.read_text())["theta"]
            for k in keys:
                if k in th:
                    t = M.zero_theta()
                    t.update({kk: np.array(v, float) for kk, v in th[k].items()})
                    out[k] = t
    return out


def main():
    rob = M.Robot()
    arm = rob.arm
    jaw_pts = [("gripper_link", V) for n, V, F in arm.parts_fine["gripper_link"] if "follower" in n] + \
              [("moving_jaw_so101_v1_link", V) for _, V, F in arm.parts_fine["moving_jaw_so101_v1_link"]]
    L = labels()
    grasps = []
    for ep in sorted(L):
        s = states(ep)
        i, held = close_frame(s, L[ep].get("inspection_frame"))
        if i is not None and held < 15:
            grasps.append(s[i])
    print(f"{len(grasps)} episodes with a close frame", flush=True)
    res = {}
    lim = arm.limits
    variants = []
    for name, th in thetas().items():
        variants.append((name, th, False))
    variants.insert(1, ("joe_cal2 (current) [clamped to URDF limits, as Isaac]", M.zero_theta(), True))
    for name, th, clamp in variants:
        tip, low, tau, beyond, tilt = [], [], [], [], []
        for st in grasps:
            q = M.q_of(st, th)
            beyond.append((q < lim[:, 0]) | (q > lim[:, 1]))
            if clamp:
                q = np.clip(q, lim[:, 0], lim[:, 1])
            T = arm.fk(q)
            G = T["gripper_link"]
            tip.append((G[:3, :3] @ TIP_LOCAL + G[:3, 3])[2])
            tilt.append(np.degrees(np.arccos(np.clip(G[2, 2], -1, 1))))  # fingers (gripper -z) vs straight down
            low.append(min((V @ T[ln][:3, :3].T + T[ln][:3, 3])[:, 2].min() for ln, V in jaw_pts))
            tau.append(arm.gravity_torque(q, {"gripper_link": M.CAM_MASS}))
        tip, low, tau = 1e3 * np.array(tip), 1e3 * np.array(low), np.array(tau)
        beyond = np.array(beyond)
        tilt = np.array(tilt)
        def iqr(x):
            return f"median {np.median(x):6.1f} (IQR {np.percentile(x, 25):.1f} to {np.percentile(x, 75):.1f})"

        res[name] = dict(tip_mm=np.percentile(tip, [25, 50, 75]).tolist(),
                         jaw_low_mm=np.percentile(low, [25, 50, 75]).tolist(),
                         tau_median=np.median(tau, 0).tolist(), tilt_deg=np.percentile(tilt, [25, 50, 75]).tolist())
        tilt_m = np.median(tilt)
        print(f"{name:30s} TIP_LOCAL z {iqr(tip)} mm | lowest stock-jaw point z {iqr(low)} mm | gravity torque "
              f"lift/elbow/wflex {np.round(np.median(tau, 0)[1:4], 2).tolist()} N m | beyond limit (% of close frames) "
              f"lift {100 * beyond[:, 1].mean():.0f} wflex {100 * beyond[:, 3].mean():.0f} | gripper axis tilt from "
              f"vertical {tilt_m:.0f} deg (a fingertip dz needs {1 / np.cos(np.radians(tilt_m)):.2f}x that length "
              "along the fingers)", flush=True)
    # sensitivity of the fingertip height at the close frames to each offset (+1 deg) and to sag (+1 deg / N m)
    base = np.array([M.tip_z(rob, st, M.zero_theta()) for st in grasps])
    sens = {}
    for k, jn in enumerate(("pan", "lift", "elbow", "wflex", "wroll")):
        th = M.zero_theta()
        th["joints"][k] = np.radians(1.0)
        sens[jn] = 1e3 * np.median([M.tip_z(rob, st, th) for st in grasps] - base)
    for k, jn in enumerate(("sag_lift", "sag_elbow", "sag_wflex")):
        th = M.zero_theta()
        th["sag"][k] = np.radians(1.0)
        sens[jn] = 1e3 * np.median([M.tip_z(rob, st, th) for st in grasps] - base)
    res["dz_mm_per_deg"] = sens
    print("fingertip dz at close frames (median, mm) per +1 deg offset / per +1 deg/(N m) sag:",
          {k: round(v, 2) for k, v in sens.items()}, flush=True)
    res["n"] = len(grasps)
    (OUT / "tip_height.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
