"""Close forensics of twin_eval.py rollouts: the same per-close / per-episode statistics as real_forensics.py, plus the
twin's ground truth at each close (target brick in the gripper frame, horizontal tip-to-brick offset).

    python sim/twin/failure_match/twin_forensics.py <run dir under sim/twin/out/failure_match> [...] [--offsets W9]
        -> <run>/twin_closes.jsonl, <run>/twin_summary.json (runs given together are pooled into the first's summary
           as well as printed)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from closes import TIP_LOCAL, TWIN, Geometry, describe_close, segment_closes, t_reach

OUT = TWIN / "out/failure_match"


def quat_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def analyse(run: Path, geo: Geometry):
    eps, closes = [], []
    for f in sorted(run.glob("ep*.npz")):
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        M, G = d["meas"].astype(float), d["goal"].astype(float)
        if len(M) < 2:
            continue
        succ_tick = meta["ticks"] - 1 if meta["outcome"] == "success" else None
        cl = []
        for i0, i1 in segment_closes(M[:, 5]):
            c = describe_close(M, G, i0, i1, geo)
            k = c["shut"]
            Rg = quat_R(d["grip_q"][k])
            tip = d["grip_p"][k] + Rg @ TIP_LOCAL
            b = d["brick_p"][k]
            rel = Rg.T @ (b - d["grip_p"][k]) - TIP_LOCAL
            # nearest distractor at the shut instant (wrong-brick / cluster closes)
            dis = d["dis_p"][k] if d["dis_p"].ndim == 3 and d["dis_p"].shape[1] else np.zeros((0, 3))
            dd = [float(np.linalg.norm(p[:2] - tip[:2])) for p in dis]
            c.update(ep=f"{run.name}#{meta['n']}", t_s=round(i0 / 30.0, 2), attempt=len(cl) + 1,
                     tip_to_brick_xy_mm=round(float(np.linalg.norm(tip[:2] - b[:2])) * 1e3, 1),
                     brick_rel_grip_mm=[round(float(v) * 1e3, 1) for v in rel],
                     tip_to_distractor_xy_mm=round(min(dd) * 1e3, 1) if dd else None,
                     brick_disp_mm=round(float(np.linalg.norm(b[:2] - d["brick_p"][0][:2])) * 1e3, 1),
                     grasped_flag=bool(d["grasped"][k:i1].any()))
            if c["onset"] is not None:
                c["descent_during_close_mm"] = round(geo.at(M[c["onset"]])["tip_z_mm"] - c["tip_z_mm"], 1)
                zs = [geo.at(M[j])["tip_z_mm"] for j in range(max(0, c["onset"] - 90), c["onset"] + 1)]
                dw = 0
                for z in reversed(zs):
                    if abs(z - c["tip_z_mm"]) > 10:
                        break
                    dw += 1
                c["dwell_before_close_s"] = round(dw / 30.0, 2)
            lifted_here = succ_tick is not None and i0 <= succ_tick <= i1
            on_other = c["tip_to_distractor_xy_mm"] is not None and \
                c["tip_to_distractor_xy_mm"] < c["tip_to_brick_xy_mm"]
            c["outcome"] = "lifted" if lifted_here else ("empty" if c["empty"] else
                                                         "wrong_brick" if on_other else "brick_not_lifted")
            cl.append(c)
        closes += cl
        first = cl[0] if cl else None
        eps.append(dict(meta, n_closes=len(cl), n_empty=sum(c["empty"] for c in cl),
                        first=first["outcome"] if first else "no_close",
                        success_attempt=next((c["attempt"] for c in cl if c["outcome"] == "lifted"), None),
                        t_first_close_s=first["t_s"] if first else None, t_reach30_s=t_reach(M, geo)))
    return eps, closes


def stats(rows, key):
    v = np.array([r[key] for r in rows if r.get(key) is not None], float)
    return {"n": len(v), "median": round(float(np.median(v)), 1), "p10": round(float(np.percentile(v, 10)), 1),
            "p90": round(float(np.percentile(v, 90)), 1)} if len(v) else None


def summarise(eps, closes) -> dict:
    F = [c for c in closes if c["attempt"] == 1]
    keys = ("t_s", "low_mm", "tip_z_mm", "tip_to_brick_xy_mm", "dwell_before_close_s", "descent_during_close_mm",
            "tilt_deg")
    return {
        "episodes": len(eps), "outcomes": dict(Counter(e["outcome"] for e in eps)),
        "episodes_with_a_close": sum(e["n_closes"] > 0 for e in eps),
        "first_attempt_success": sum(e["first"] == "lifted" for e in eps),
        "episode_success": sum(e["outcome"] == "success" for e in eps),
        "first_close": dict(Counter(e["first"] for e in eps)),
        "closes": len(closes), "closes_empty": sum(c["empty"] for c in closes),
        "close_outcomes": dict(Counter(c["outcome"] for c in closes)),
        "closes_per_episode": stats([{"v": e["n_closes"]} for e in eps], "v"),
        "episodes_ge3_closes": sum(e["n_closes"] >= 3 for e in eps),
        "t_reach30_s": stats(eps, "t_reach30_s"), "t_first_close_s": stats(eps, "t_first_close_s"),
        "pushed_before_first_close": sum(1 for e in eps if e.get("pushed_t") is not None and
                                         (e["t_first_close_s"] is None or e["pushed_t"] / 30 < e["t_first_close_s"])),
        "first_close_empty": {k: stats([c for c in F if c["empty"]], k) for k in keys},
        "first_close_brick": {k: stats([c for c in F if not c["empty"]], k) for k in keys},
        "upside_down": {"episodes": sum(e.get("upside_down", False) for e in eps),
                        "first_attempt_success": sum(e["first"] == "lifted" for e in eps if e.get("upside_down"))},
        "by_distractors": {lab: {"n": len(E), "first_ok": sum(e["first"] == "lifted" for e in E),
                                 "ok": sum(e["outcome"] == "success" for e in E)}
                           for lab, E in (("0-1", [e for e in eps if e.get("n_distractors", 0) <= 1]),
                                          ("2-3", [e for e in eps if 2 <= e.get("n_distractors", 0) <= 3]),
                                          ("4+", [e for e in eps if e.get("n_distractors", 0) >= 4]))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--offsets", default="W9")
    ap.add_argument("--name", default=None, help="pooled summary name (default: first run)")
    a = ap.parse_args()
    geo = Geometry(a.offsets)
    E, C = [], []
    for r in a.runs:
        run = Path(r) if Path(r).is_absolute() else OUT / r
        eps, closes = analyse(run, geo)
        with open(run / "twin_closes.jsonl", "w") as fh:
            for c in closes:
                fh.write(json.dumps(c, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)) + "\n")
        (run / "twin_summary.json").write_text(json.dumps(summarise(eps, closes), indent=1))
        E += eps
        C += closes
    s = summarise(E, C)
    (OUT / f"pooled_{a.name or Path(a.runs[0]).name}.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
