"""Real vs twin failure table (run real_forensics.py, twin_forensics.py and stuck.py first, or let this call them).

    python sim/twin/failure_match/compare.py <name> <twin run dir> [...]
        -> sim/twin/out/failure_match/compare_<name>.json and a printed table

Rows (first close per episode unless noted): first-close class (lifted / empty / jaws-on-brick-no-lift / no close,
with "pushed >= 10 mm before the close" and "fin on the brick top" carved out for the twin), lateral miss of the empty
first closes, clock (first close, approach arrival), shut height (grasp point at the shut instant; lowest fin during
the close), jaw speed (command onset -> reading < 15), regrasp closes with the fins >= 5 mm above the table.
Real: 8k + 5.5k with the trained prompts (n per row given). The real lateral miss is a proxy (the next successful
close's fingertip, see real_forensics / README); the twin's is the simulated truth.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np
from closes import TWIN, Geometry
from stuck import twin_rows
from twin_forensics import analyse

OUT = TWIN / "out/failure_match"


def q(v):
    v = np.asarray([x for x in v if x is not None], float)
    return {"n": len(v), "median": round(float(np.median(v)), 2), "p10": round(float(np.percentile(v, 10)), 2),
            "p90": round(float(np.percentile(v, 90)), 2)} if len(v) else None


def table(first: list, closes: list, eps: list, extra: dict | None = None) -> dict:
    F = first
    E = [c for c in F if c["outcome"] == "empty"]
    n = len(eps)
    cls = Counter(c["outcome"] for c in F)
    cls["no_close"] = n - len(F)
    regr = [c for c in closes if c["attempt"] > 1]
    out = {
        "episodes": n,
        "first_close_class": dict(cls),
        "first_attempt_success_frac": round(cls["lifted"] / max(1, n), 3),
        "episodes_ge3_closes": sum(e >= 3 for e in Counter(c["ep"] for c in closes).values()),
        "empty_first_close_lateral_miss_mm": q([c.get("tip_to_brick_xy_mm", c.get("proxy_miss_mm")) for c in E]),
        "t_first_close_s": q([c["t_s"] for c in F]),
        "t_reach30_s": q([e.get("t_reach30_s") for e in eps]),
        "grasp_point_z_at_shut_mm": q([c["tip_z_mm"] for c in F]),
        "lowest_fin_during_close_mm": q([c["low_min_mm"] for c in F]),
        "onset_to_below15_s": q([(c["i0"] - c["onset"]) / 30 for c in closes if c.get("onset") is not None]),
        "regrasp_closes_fins_ge5mm_above_table": f"{sum(c['low_min_mm'] >= 5 for c in regr)}/{len(regr)}",
    }
    out.update(extra or {})
    return out


def real_table():
    C = [json.loads(line) for line in open(OUT / "real_closes.jsonl")]
    E = [json.loads(line) for line in open(OUT / "real_episodes.jsonl")]
    C = [c for c in C if c["stage"] == "pick" and c["group"] != "ft5.5k_untrained"]
    E = [e for e in E if e["group"] != "ft5.5k_untrained"]
    # lateral-miss proxy: the episode's lifting close's fingertip as the brick position
    for ep in {c["ep"] for c in C}:
        cs = sorted([c for c in C if c["ep"] == ep], key=lambda c: c["attempt"])
        lift = next((c for c in cs if c["outcome"] == "lifted"), None)
        for c in cs:
            if lift is not None and c["attempt"] < lift["attempt"] and lift["attempt"] - c["attempt"] <= 2:
                c["proxy_miss_mm"] = round(float(np.hypot(c["tip_xy_mm"][0] - lift["tip_xy_mm"][0],
                                                          c["tip_xy_mm"][1] - lift["tip_xy_mm"][1])), 1)
    stuck = {}
    p = OUT / "stuck_real.jsonl"
    if p.exists():
        S = [json.loads(line) for line in open(p)]
        stuck = {"stall_at_brick_top_height_before_first_close (proxy, upper bound)":
                 f"{sum(r['high_stall'] for r in S)}/{len(S)}"}
    return table([c for c in C if c["attempt"] == 1], C, E, stuck)


def main():
    name, runs = sys.argv[1], sys.argv[2:]
    geo = Geometry("W9")
    eps, closes, stuck = [], [], []
    for r in runs:
        e, c = analyse(OUT / r, geo)
        eps += e
        closes += c
        stuck += twin_rows(OUT / r, geo)
    first = [c for c in closes if c["attempt"] == 1]
    st = {r["ep"]: r for r in stuck}
    extra = {"stall_at_brick_top_height_before_first_close (proxy, upper bound)":
             f"{sum(r['high_stall'] for r in stuck)}/{len(stuck)}",
             "pushed_ge10mm_before_first_close": sum(r["pushed_before_close"] for r in stuck),
             "fin_on_brick_top_ge10_ticks_before_first_close": sum(r["on_top"] for r in stuck),
             "first_close_class_with_push": dict(Counter(
                 ("pushed_then_" if st.get(c["ep"], {}).get("pushed_before_close") else "") + c["outcome"]
                 for c in first))}
    res = {"real": real_table(), "twin": table(first, closes, eps, extra), "twin_runs": runs}
    (OUT / f"compare_{name}.json").write_text(json.dumps(res, indent=1))
    for k in res["twin"]:
        print(f"{k:55s} real {json.dumps(res['real'].get(k))}\n{'':55s} twin {json.dumps(res['twin'][k])}")


if __name__ == "__main__":
    main()
