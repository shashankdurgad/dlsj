"""Fit SO-101 joint offsets (+ inspect camera pose, wrist-camera mount, table-plane height) by silhouette matching on
the recorded cal2 frames (top + inspect; finger regions don't-care) and the wrist camera's table view. README.md.

    python sim/twin/align/joint_fit/fit.py [--workers 16] [--maxfev 2000]
Fits B0, B1, B2, B3, C0, C and scores them per camera on held-out episodes -> sim/twin/out/align/joint_fit/fit.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # align/ (frames.py)
import model as M  # noqa: E402

G = {}


def _init():
    import table as TB
    items, D = M.load_data()
    for t in TB.build(items, D):  # wrist table-view items (table.py), appended with cam = "table"
        items.append({**t, "cam": "table", "src": t["i"]})
    # gripper-close frames of every labelled episode (tip_height.close_frame): grasp-depth term / check
    from frames import labels, states
    from tip_height import close_frame
    eps = sorted(D["episodes"].tolist())
    held_eps = set(eps[2::3])
    for k, ep in enumerate(sorted(labels())):
        s = states(ep)
        i, hv = close_frame(s, labels()[ep].get("inspection_frame"))
        if i is not None and hv < 15:
            items.append(dict(cam="grasp", ep=ep, fr=i, state=s[i], held=(ep in held_eps) if ep in eps else k % 3 == 2))
    G["items"] = items
    G["robot"] = M.Robot()
    th0 = M.zero_theta()
    for it in items:
        if it["cam"] == "grasp":
            it["z0"] = M.tip_z(G["robot"], it["state"], th0)


def _losses(args):
    x, free, base, idx = args
    th = M.unpack(x, free, base)
    cams = M.cams_of(th)
    import table as TB
    out = []
    for i in idx:
        it = G["items"][i]
        if it["cam"] == "grasp":
            out.append(M.grasp_loss(G["robot"], it["state"], th, it["z0"]))
        elif it["cam"] == "table":
            out.append(TB.loss(G["robot"], it, th, 0.01 + th["plane"][0], cams))
        else:
            out.append(M.frame_loss(G["robot"], it["cam"], it["state"], it["real"], it["dt"], th, cams,
                                    dc=it.get("dc"))[:2])
    return out


class Evaluator:
    def __init__(self, pool, nw):
        self.pool, self.nw = pool, nw

    def per_item(self, x, free, base, idx):
        chunks = [idx[k::self.nw] for k in range(self.nw)]
        res = self.pool.map(_losses, [(x, free, base, c) for c in chunks])
        out = np.full((len(idx), 2), np.nan)
        for k, r in enumerate(res):
            out[k::self.nw] = np.array(r, float).reshape(-1, 2)
        return out

    def objective(self, free, base, idx, w=None, cam_weight=None):
        cams = np.array([G["items"][i]["cam"] for i in idx])
        w = np.ones(len(idx)) if w is None else np.asarray(w, float)

        def f(x):
            L = self.per_item(x, free, base, idx)[:, 0]
            tot = 0.0
            for c in np.unique(cams):  # mean per camera, cameras weighted equally (or cam_weight)
                m = (cams == c) & np.isfinite(L) & (w > 0)
                cw = 1.0 if cam_weight is None else cam_weight.get(c, 1.0)
                tot += cw * (np.sum(w[m] * L[m]) / np.sum(w[m]) if m.any() else M.CAP * 2)
            return tot
        return f


SCALES = dict(joints=np.radians(2.0) * np.ones(5), grip=np.array([0.05, 0.002]),
              inspect=np.r_[np.radians(1.5) * np.ones(3), 0.015 * np.ones(3)],
              wrist=np.r_[np.radians(2.0) * np.ones(3), 0.004 * np.ones(3)],
              top=np.r_[np.radians(1.0) * np.ones(3), 0.01 * np.ones(3)], plane=np.array([0.005]),
              sag=np.radians(2.0) / 0.5 * np.ones(3), jpr=np.radians(2.0) * np.ones(2))


def fit(ev, free, base, idx, w=None, x0=None, maxfev=1500, label=""):
    """Powell in scaled coordinates."""
    sc = np.concatenate([SCALES[k] for k in free])
    f = ev.objective(free, base, idx, w)
    x0 = M.pack(base, free) if x0 is None else x0
    t0 = time.time()
    f0 = f(x0)
    r = minimize(lambda z: f(z * sc), x0 / sc, method="Powell", options=dict(xtol=1e-3, ftol=1e-4, maxfev=maxfev))
    x = r.x * sc
    print(f"  [{label}] loss {f0:.3f} -> {r.fun:.3f} ({r.nfev} evals, {time.time() - t0:.0f}s)", flush=True)
    return M.unpack(x, free, base), float(r.fun)


def _pose(d):
    return f"{np.round(np.degrees(d[:3]), 2).tolist()} {np.round(1e3 * d[3:], 1).tolist()}"


def report(th, label):
    """Print a theta: offset deltas / totals (deg), gripper, inspect and wrist pose deltas, sag."""
    d = np.degrees(th["joints"])
    tot = np.degrees(M.JOE_OFFSETS + th["joints"])
    if "sag" in th and np.any(th["sag"]):
        print(f"  {label}: sag (deg per N m) lift/elbow/wflex {np.round(np.degrees(th['sag']), 2).tolist()} | "
              f"jpr (deg) {np.round(np.degrees(th.get('jpr', np.zeros(2))), 2).tolist()}", flush=True)
    print(f"  {label}: delta vs joe_cal2 (deg) pan {d[0]:+.2f} lift {d[1]:+.2f} elbow {d[2]:+.2f} "
          f"wflex {d[3]:+.2f} wroll {d[4]:+.2f} | total offsets (deg) {np.round(tot, 2).tolist()} | "
          f"grip bias {th['grip'][0]:+.3f} rad scale {th['grip'][1]:+.4f} | "
          f"inspect d(rot deg, mm) {_pose(th['inspect'])} | wrist d(rot deg, mm) {_pose(th['wrist'])}",
          flush=True)


def evaluate(ev, th, idx):
    """per-camera mean loss / IoU over idx (items invalid under either theta are dropped for both)."""
    items = G["items"]
    A = ev.per_item(M.pack(th, M.NAMES), list(M.NAMES), th, idx)
    out = {}
    for c in ("top", "inspect", "wrist", "table", "grasp"):
        m = np.array([items[i]["cam"] == c for i in idx]) & np.isfinite(A[:, 0])
        if not m.any():
            continue
        out[c] = dict(loss=float(np.nanmean(A[m, 0])), iou=float(np.nanmean(A[m, 1])), n=int(m.sum()))
    return out, A


CAMS = ("top", "inspect", "table", "grasp")  # wrist jaws are not used (the real fingers are not the URDF jaws)
OBJ = ("top", "inspect", "table")           # cameras in the objective; grasp = fingertip-height check only


def make_sets(ev, th0):
    items = G["items"]
    all_idx = list(range(len(items)))
    A0 = ev.per_item(M.pack(th0, M.NAMES), list(M.NAMES), th0, all_idx)
    ok = np.isfinite(A0[:, 0])  # items valid under the current profile (arm in view)
    sets = {(c, h): [i for i in all_idx if ok[i] and items[i]["cam"] == c and items[i]["held"] == h]
            for c in CAMS for h in (False, True)}
    print({f"{c}/{'held' if h else 'fit'}": len(v) for (c, h), v in sets.items()}, flush=True)
    return sets


def scoreboard(ev, cands, sets, res, title=""):
    fit_all = [i for c in CAMS for i in sets[(c, False)]]
    held_all = [i for c in CAMS for i in sets[(c, True)]]
    for k in ("heldout", "fit", "theta"):
        res.setdefault(k, {})
    print(f"\n{title} held-out (fit) per camera: loss / IoU   "
          "[table: loss / n pts; grasp: |dz + 9 mm| / mean fingertip dz vs joe_cal2 (mm)]", flush=True)
    for k, th in cands.items():
        h, _ = evaluate(ev, th, held_all)
        f, _ = evaluate(ev, th, fit_all)
        res["heldout"][k], res["fit"][k] = h, f
        res["theta"][k] = {kk: np.asarray(vv).tolist() for kk, vv in th.items()}
        print(f"  {k:30s} " + " | ".join(f"{c} {h[c]['loss']:6.2f}/{h[c]['iou']:.3f} "
                                         f"({f[c]['loss']:.2f}/{f[c]['iou']:.3f})" for c in CAMS if c in h), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--maxfev", type=int, default=2000)
    args = ap.parse_args()
    _init()
    pool = Pool(args.workers, initializer=_init)
    ev = Evaluator(pool, args.workers)
    th0 = M.zero_theta()
    sets = make_sets(ev, th0)
    top, ins, tab = sets[("top", False)], sets[("inspect", False)], sets[("table", False)]
    out_json = M.TWIN / "out/align/joint_fit/fit.json"
    res = {"sets": {f"{c}/{'held' if h else 'fit'}": v for (c, h), v in sets.items()}}
    th = {}

    def step(name, free, start, idx, note):
        print(f"{name}: {note}", flush=True)
        th[name], _ = fit(ev, free, start, idx, maxfev=args.maxfev, label=name)
        report(th[name], name)
        res.setdefault("theta", {})[name] = {k: v.tolist() for k, v in th[name].items()}
        out_json.write_text(json.dumps(res, indent=1))

    step("B0", ["inspect"], th0, ins, "inspect pose only, joints joe_cal2")
    step("B1", ["joints"], th0, top, "offsets from the top camera alone")
    step("B2", ["joints"], th["B1"], top + ins, "offsets, top + inspect (inspect FIXED at ChArUco)")
    step("B3", ["joints", "inspect"], {**th["B2"], "inspect": th["B0"]["inspect"]}, top + ins, "offsets + inspect pose")
    step("C0", ["joints", "inspect", "plane"], th["B3"], top + ins + tab, "B3 + wrist table view (mount joe_cal2)")
    step("C", ["joints", "inspect", "wrist", "plane"], th["C0"], top + ins + tab, "C0 + wrist mount free")
    cands = {"joe_cal2 (current twin)": th0, **{k: th[k] for k in ("B0", "B1", "B2", "B3", "C0", "C")}}
    scoreboard(ev, cands, sets, res)
    out_json.write_text(json.dumps(res, indent=1))
    pool.close()
    pool.join()
    print("FIT_OK", flush=True)


if __name__ == "__main__":
    main()
