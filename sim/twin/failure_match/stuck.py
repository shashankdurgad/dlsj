"""'Stuck on top / pushed before the close' on the first approach, real vs twin.

Twin (ground truth from twin_eval.py logs): before the first close's command onset,
  on_top    : a fin tip (fixed or moving, CAD tips through the gripper pose) lies inside the target's footprint
              and within +-4 mm of its top face for >= 10 ticks;
  pushed    : the target moved >= 10 mm in the table plane before the onset (any cause);
  stuck_push: on_top and pushed.
Both (real and twin, from the joints alone): a *high stall* before the onset (high_stall below: the descent stops
with the lowest fin tip at brick-top height). The twin labels calibrate how well this joint-only signal finds the
on-top cases, which is then counted in the real traces.

    python sim/twin/failure_match/stuck.py <twin run dirs...>   -> out/failure_match/stuck_summary.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from closes import GRIP_BIAS, GRIP_SCALE, TWIN, Geometry, segment_closes
from twin_forensics import quat_R

OUT = TWIN / "out/failure_match"
HALF = {"2x2": (0.0159, 0.0159), "2x4": (0.0318, 0.0159), "curved_2x3": (0.0239, 0.0159)}
BRICK_H = 0.0192


def onset_of_first_close(M, C):
    cl = segment_closes(M[:, 5])
    if not cl:
        return None
    k = cl[0][0]
    while k > 0 and C[k, 5] < 25:
        k -= 1
    return k + 1


def high_stall(M, geo: Geometry, end: int) -> dict:
    """Longest run of ticks before `end` with the lowest fin tip (W9 FK) 8-26 mm above the table (a 19.2 mm Duplo top
    +- the fin give / FK error) and the arm nearly still (body joints < 0.6 deg / tick): the descent stalled at
    brick-top height. Flagged at >= 9 ticks (0.3 s)."""
    best = run = 0
    at = None
    for k in range(3, max(3, end)):
        still = np.abs(M[k, :4] - M[k - 3, :4]).max() / 3 < 0.6
        low = geo.at(M[k])["low_mm"]
        if still and 8.0 <= low <= 26.0:
            run += 1
            if run > best:
                best, at = run, (round((k - run + 1) / 30, 2), round(low, 1))
        else:
            run = 0
    return {"high_stall": best >= 9, "high_stall_ticks": best,
            **({"stall_t_s": at[0], "stall_low_mm": at[1]} if at else {})}


def real_rows(geo):
    from real_forensics import DATA, TRAINED, episodes

    rows = []
    for f in sorted(glob.glob(str(DATA / "desmond/pi05_traces/*.jsonl"))):
        model, eps = episodes(f)
        if not model:
            continue
        for i, e in enumerate(eps):
            run = [t for t in e["ticks"] if t["phase"] == "running"]
            prompts = [x.get("prompt") for x in e["infer"] if x.get("prompt")]
            if len(run) < 90 or not all(p in TRAINED for p in prompts):
                continue
            M = np.array([t["measured"] for t in run], float)
            C = np.array([t["cmd"] for t in run], float)
            on = onset_of_first_close(M, C)
            end = on if on is not None else len(M)
            rows.append({"ep": f"{Path(f).name[5:-6]}#{i}", "model": model, **high_stall(M, geo, end)})
    return rows


def twin_rows(run_dir: Path, geo: Geometry):
    from sim.twin.gripper import model

    g = model.geometry()
    fix, jaw = g["fixed_tip"].astype(float), g["jaw_tip"].astype(float)
    rows = []
    for f in sorted(run_dir.glob("ep*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        M, G = d["meas"].astype(float), d["goal"].astype(float)
        on = onset_of_first_close(M, G)
        end = on if on is not None else len(M)
        kind = (meta.get("target") or {}).get("kind") or "2x2"
        hx, hy = HALF.get(kind, HALF["2x2"])
        top_ticks = 0
        for k in range(end):
            Rg = quat_R(d["grip_q"][k])
            qg = M[k, 5] * GRIP_SCALE + GRIP_BIAS
            tips = [d["grip_p"][k] + Rg @ fix, d["grip_p"][k] + Rg @ model.jaw_to_grip(jaw, qg)]
            Rb = quat_R(d["brick_q"][k])
            top = d["brick_p"][k][2] + BRICK_H / 2
            hit = False
            for p in tips:
                rel = Rb.T @ (p - d["brick_p"][k])
                if abs(rel[0]) < hx and abs(rel[1]) < hy and -0.004 < p[2] - top < 0.004:
                    hit = True
            top_ticks += hit
        disp = float(np.linalg.norm(d["brick_p"][end - 1][:2] - d["brick_p"][0][:2])) if end > 0 else 0.0
        on_top, pushed = top_ticks >= 10, disp >= 0.010
        rows.append({"ep": f"{run_dir.name}#{meta['n']}", "on_top": on_top, "on_top_ticks": top_ticks,
                     "pushed_before_close": pushed, "disp_before_close_mm": round(disp * 1e3, 1),
                     "stuck_push": on_top and pushed, "outcome": meta["outcome"],
                     **high_stall(M, geo, end)})
    return rows


def main():
    geo = Geometry("W9")
    real = real_rows(geo)
    twin = []
    for r in sys.argv[1:]:
        twin += twin_rows(Path(r) if Path(r).is_absolute() else OUT / r, geo)
    tp = sum(r["high_stall"] and r["on_top"] for r in twin)
    summ = {
        "real": {m: {"n": len(R), "high_stall": sum(r["high_stall"] for r in R),
                     "episodes": [r["ep"] for r in R if r["high_stall"]],
                     "stall_low_mm": [r.get("stall_low_mm") for r in R if r["high_stall"]]}
                 for m in ("ft8k", "ft5.5k") for R in [[r for r in real if r["model"] == m]]},
        "twin": {"n": len(twin), "on_top": sum(r["on_top"] for r in twin),
                 "pushed_before_close": sum(r["pushed_before_close"] for r in twin),
                 "stuck_push": sum(r["stuck_push"] for r in twin),
                 "high_stall": sum(r["high_stall"] for r in twin),
                 "high_stall_given_on_top": f"{tp}/{sum(r['on_top'] for r in twin)}",
                 "high_stall_given_not_on_top": f"{sum(r['high_stall'] and not r['on_top'] for r in twin)}/"
                                                f"{sum(not r['on_top'] for r in twin)}",
                 "stuck_push_given_high_stall": f"{sum(r['high_stall'] and r['stuck_push'] for r in twin)}/"
                                                f"{sum(r['high_stall'] for r in twin)}",
                 "disp_before_close_mm_median": float(np.median([r["disp_before_close_mm"] for r in twin]))
                 if twin else None}}
    (OUT / "stuck_real.jsonl").write_text("".join(json.dumps(r) + "\n" for r in real))
    (OUT / "stuck_twin.jsonl").write_text("".join(json.dumps(r) + "\n" for r in twin))
    (OUT / "stuck_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
