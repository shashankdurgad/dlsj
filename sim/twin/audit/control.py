"""Positive control: closed-loop behaviour of the fine-tune in the twin against the known REAL outcomes.

    python sim/twin/audit/control.py --run sim/twin/out/audit/runs/<stamp>        # plain python, CPU, no Isaac

A teammate's sim passed every replay while every policy scored 0 in it, so open-loop equivalence alone is not enough:
the twin must also reproduce what the policy DOES on the robot.
Real side (sim/twin/data/real_rollouts, the XPS controller traces): per operator episode that starts in the pick stage
with the jaws open, the FIRST close of the pick stage is classified as on a brick (min measured gripper >= 4.5 while
shut) or empty (< 4.5: the jaws shut on nothing, the dominant real failure); plus every close of the run, and the run
outcome references (5.5k: 20 of 45 pick starts reached bring; 8k loop: 9 starts, 6 grasps confirmed, 1 clear fail).
Twin side: the 8k fine-tune's closed-loop pick episodes in TwinEnv by the RL agent's harness (sim/twin/rl/molmo_eval.py:
start 0.7-1.2 s before the recorded close, brick at the close-frame fingertip +- 2.5 cm, 3 s, one close attempt,
success = lifted >= 3 cm for 0.5 s): MOLMO lines of sim/twin/out/rl/molmo8k_base*.log and of copies from the other host
in sim/twin/out/audit/control/*.log (read-only harvest; deduplicated by host file + episode index). The twin's single
attempt is compared with the real FIRST close. Writes <run>/control.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.audit.common import OUT, TWIN, jdump

TRACES = TWIN / "data/real_rollouts/home/desmond/pi05_traces"
EXPORTS = {"molmoact2_so101_rtc_int8": "zeroshot", "ft_cal2_456/rtc_int8": "ft5.5k", "ft_cal2_456_8k/rtc_int8": "ft8k"}
REAL_REF = {"ft5.5k": "45 runs started in pick, 20 reached bring (grasp confirmed / marked), 17 reached place",
            "ft8k_loop": "9 starts, 6 grasps confirmed, 1 clear fail (loop_control.py, 15:14-15:29)"}


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def real_side() -> dict:
    out: dict = {}
    for f in sorted(glob.glob(str(TRACES / "*.jsonl"))):
        recs = []
        for ln in open(f):
            try:
                recs.append(json.loads(ln))
            except json.JSONDecodeError:  # the last trace was still being written when copied
                pass
        if not recs or recs[0].get("kind") != "run":
            continue
        exp = recs[0].get("args", {}).get("export") or ""
        tg = next((v for k, v in EXPORTS.items() if exp.endswith(k)), None)
        if tg is None:
            continue
        eps, cur = [], None
        for r in recs[1:]:
            if r.get("kind") == "key" and r.get("key") in ("start", "q"):
                if cur:
                    eps.append(cur)
                cur = [] if r["key"] == "start" else None
            elif cur is not None and r.get("kind") == "tick" and r.get("phase") == "running":
                cur.append(r)
        if cur:
            eps.append(cur)
        d = out.setdefault(tg, {"episodes": 0, "first_close": {"brick": 0, "empty": 0}, "closes": 0, "empty": 0,
                                "no_close": 0, "files": set()})
        for e in eps:
            pick = [t for t in e if t.get("stage") == "pick"]
            if len(pick) < 30 or pick[0]["measured"][5] < 25:
                continue
            d["episodes"] += 1
            d["files"].add(os.path.basename(f))
            first, inside, mn = None, False, 99.0
            for t in pick:
                v = t["measured"][5]
                if v < 15:
                    inside, mn = True, min(mn, v)
                elif v >= 25 and inside:
                    kind = "empty" if mn < 4.5 else "brick"
                    d["closes"] += 1
                    d["empty"] += kind == "empty"
                    first = first or kind
                    inside, mn = False, 99.0
            if inside:
                kind = "empty" if mn < 4.5 else "brick"
                d["closes"] += 1
                d["empty"] += kind == "empty"
                first = first or kind
            if first is None:
                d["no_close"] += 1
            else:
                d["first_close"][first] += 1
    for d in out.values():
        n = d["first_close"]["brick"] + d["first_close"]["empty"]
        d["first_close_brick_rate"] = round(d["first_close"]["brick"] / n, 3) if n else None
        d["first_close_ci95"] = wilson(d["first_close"]["brick"], n)
        d["empty_close_frac"] = round(d["empty"] / d["closes"], 3) if d["closes"] else None
        d["files"] = sorted(d["files"])
    return out


def twin_side() -> dict:
    rows, seen = [], set()
    for f in sorted(glob.glob(str(TWIN / "out/rl/molmo8k_base*.log")) + glob.glob(str(OUT / "control/*.log"))):
        name = Path(f).name
        for ln in open(f, errors="replace"):
            if not ln.startswith("MOLMO {"):
                continue
            try:
                r = json.loads(ln[6:])
            except json.JSONDecodeError:
                continue
            key = (name, r.get("n"))
            if key in seen or r.get("black_frames", 0):
                continue
            seen.add(key)
            r["log"] = name
            rows.append(r)
    oc = [r["outcome"] for r in rows]
    n, ns = len(oc), oc.count("success")
    return {"episodes": n, "logs": sorted({r["log"] for r in rows}), "success": ns,
            "success_rate": round(ns / n, 3) if n else None, "success_ci95": wilson(ns, n),
            "outcomes": {k: oc.count(k) for k in sorted(set(oc))},
            "centred_given_success": round(sum(r.get("centred", False) for r in rows if r["outcome"] == "success")
                                           / max(ns, 1), 3),
            "by_prompt": {t: {"n": sum(r["task"] == t for r in rows),
                              "success": sum(r["task"] == t and r["outcome"] == "success" for r in rows)}
                          for t in sorted({r["task"] for r in rows})}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = Path(ap.parse_args().run)
    real, twin = real_side(), twin_side()
    res = {"real": real, "real_reference": REAL_REF, "twin_ft8k": twin}
    r8 = real.get("ft8k", {})
    if twin["episodes"] and r8.get("first_close_brick_rate") is not None:
        lo, hi = twin["success_ci95"]
        rl, rh = r8["first_close_ci95"]
        res["verdict"] = ("twin first-attempt success {:.0%} [{}-{}] vs real 8k first close on a brick {:.0%} [{}-{}]; "
                          "real 5.5k {}").format(twin["success_rate"], lo, hi, r8["first_close_brick_rate"], rl, rh,
                                                 real.get("ft5.5k", {}).get("first_close_brick_rate"))
    jdump(run / "control.json", res)
    print("CONTROL_OK", json.dumps({"twin": {k: twin[k] for k in ("episodes", "success_rate", "outcomes")},
                                    "real": {k: {x: v.get(x) for x in ("episodes", "first_close", "empty_close_frac")}
                                             for k, v in real.items()}}), flush=True)


if __name__ == "__main__":
    main()
