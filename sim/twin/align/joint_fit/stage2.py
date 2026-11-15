"""After fit.py: 1-D loss profiles per joint around C, inspect-camera drift over the recording, bootstrap of C.

    python sim/twin/align/joint_fit/stage2.py [--boot 6] [--workers 16]   -> out/align/joint_fit/stage2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402
import model as M  # noqa: E402

FREE_C = ["joints", "inspect", "wrist", "plane"]


def th_from(d):
    th = M.zero_theta()
    th.update({k: np.array(v, float) for k, v in d.items()})
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--boot", type=int, default=6)
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()
    F._init()
    items = F.G["items"]
    pool = Pool(args.workers, initializer=F._init)
    ev = F.Evaluator(pool, args.workers)
    out = M.TWIN / "out/align/joint_fit"
    out_json = out / "stage2.json"
    r1 = json.loads((out / "fit.json").read_text())
    thC = th_from(r1["theta"]["C"])
    sets = F.make_sets(ev, M.zero_theta())
    fit_all = [i for c in F.OBJ for i in sets[(c, False)]]
    every = [i for c in F.CAMS for h in (False, True) for i in sets[(c, h)]]
    res = json.loads(out_json.read_text()) if out_json.exists() else {}

    if "profiles" not in args.skip:
        print("profiles: loss per camera vs joint delta around C (fit + held frames)", flush=True)
        grid = np.array([-8, -4, -2, -1, 0, 1, 2, 4, 8], float)
        prof = {}
        for j, jn in enumerate(("pan", "lift", "elbow", "wflex", "wroll")):
            rows = []
            for d in grid:
                th = {k: v.copy() for k, v in thC.items()}
                th["joints"][j] += np.radians(d)
                e, _ = F.evaluate(ev, th, every)
                rows.append([e[c]["loss"] for c in F.CAMS])
            rows = np.array(rows)
            prof[jn] = rows.tolist()
            print(f"  {jn:6s} " + " | ".join(f"{c}: " + " ".join(f"{v:5.2f}" for v in rows[:, k])
                                             for k, c in enumerate(F.CAMS)), flush=True)
        res["profiles"] = {"grid_deg": grid.tolist(), "cams": list(F.CAMS), "loss": prof}
        out_json.write_text(json.dumps(res, indent=1))

    if "drift" not in args.skip:
        ins = sets[("inspect", False)] + sets[("inspect", True)]
        eps = sorted({items[i]["ep"] for i in ins})
        drift = []
        print("inspect drift: pose per third of the recording (joints = C)", flush=True)
        for k, part in enumerate(np.array_split(np.array(eps), 3)):
            idx = [i for i in ins if items[i]["ep"] in set(part.tolist())]
            thk, lk = F.fit(ev, ["inspect"], thC, idx, maxfev=500, label=f"inspect third {k} eps {part[0]}-{part[-1]}")
            dR = np.degrees(thk["inspect"][:3] - thC["inspect"][:3])
            dt = 1e3 * (thk["inspect"][3:] - thC["inspect"][3:])
            print(f"  third {k}: d rot (deg) vs C {np.round(dR, 2).tolist()} d pos (mm) {np.round(dt, 1).tolist()}",
                  flush=True)
            drift.append(dict(episodes=[int(part[0]), int(part[-1])], inspect=thk["inspect"].tolist(), loss=lk))
        res["inspect_drift"] = drift
        out_json.write_text(json.dumps(res, indent=1))

    fit_ep = sorted({items[i]["ep"] for i in fit_all})
    rng = np.random.default_rng(0)
    boots = []
    for b in range(args.boot):
        draw = rng.choice(fit_ep, len(fit_ep), replace=True)
        cnt = {e: int((draw == e).sum()) for e in fit_ep}
        w = [cnt.get(items[i]["ep"], 0) for i in fit_all]
        thb, _ = F.fit(ev, FREE_C, thC, fit_all, w=w, maxfev=600, label=f"boot {b}")
        boots.append(M.pack(thb, FREE_C).tolist())
        B = np.array(boots)
        print(f"  boot {b}: joints (deg) {np.round(np.degrees(thb['joints']), 2).tolist()} | sd so far "
              f"{np.round(np.degrees(B[:, :5].std(0)), 2).tolist()}", flush=True)
        res["bootstrap"] = {"free": FREE_C, "x": boots}
        out_json.write_text(json.dumps(res, indent=1))
    pool.close()
    pool.join()
    print("STAGE2_OK", flush=True)


if __name__ == "__main__":
    main()
