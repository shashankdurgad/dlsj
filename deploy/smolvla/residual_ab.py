"""A/B of the grasp residual on the robot: first-attempt pick success, residual off vs on, from the controller's
JSON-lines traces (`--trace <dir>`, one pi05-*.jsonl per run; the residual's records are in the same file).

A pick = one stretch of control ticks with phase "running" and stage "pick" (the loop starts a new one after every
home-and-retry). Per pick:
* mode: "on" if the residual changed the commands in it, "shadow" if it ran without applying, else "off";
* closes: close attempts = the commanded jaw falls below 15 after being above 30 (the twin's rule, image_task.py);
* grasped: the pick ended by handing over to "bring" (the loop does that only after its grasp check);
* first_attempt: grasped with exactly one close (no re-open / regrasp), the twin's first-attempt success;
  a pick still running when the trace ends is listed but left out of the counts;
* seconds, the residual's mean |applied offset| (xyz mm, wrist deg, gripper units) and its decisions.

    python deploy/smolvla/residual_ab.py ~/desmond/residual_test/traces/pi05-*.jsonl [--csv picks.csv]

Prints one line per pick and, per mode, first-attempt and eventual success with Wilson 95 % intervals, plus a
two-sided Fisher exact test off vs on. Label runs by launching them with the mode you mean (off / shadow / on).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

CLOSE_BELOW, OPEN_ABOVE = 15.0, 30.0


def picks(path: str | Path) -> list[dict]:
    out, cur = [], None

    def new(r):
        return {"file": Path(path).name, "t0": r["wall"], "t1": r["wall"], "closes": 0, "open": None, "res": 0,
                "applied": False, "decisions": 0, "o_sum": [0.0] * 6, "o_n": 0, "end": None}

    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = r.get("kind")
        live = r.get("phase") == "running" and r.get("stage") == "pick"
        if k == "residual":  # logged just before the tick record of the same control tick
            if cur is None and live:
                cur = new(r)
            if cur is not None:
                cur["res"] += 1
                cur["applied"] |= bool(r.get("applied")) and any(abs(v) > 1e-9 for v in r.get("o") or [])
                cur["decisions"] += int(bool(r.get("decision")))
                if r.get("o"):
                    cur["o_sum"] = [a + abs(b) for a, b in zip(cur["o_sum"], r["o"], strict=True)]
                    cur["o_n"] += 1
            continue
        if k != "tick":
            continue
        if live and cur is None:
            cur = new(r)
        if cur is None:
            continue
        if not live:  # the pick ended: handed to bring (grasp confirmed) or stopped / homed
            cur["end"] = r.get("stage") if r.get("phase") == "running" else r.get("phase")
            out.append(cur)
            cur = None
            continue
        g = float(r["cmd"][5])
        if g > OPEN_ABOVE or cur["open"] is None:
            cur["open"] = g > OPEN_ABOVE
        elif g < CLOSE_BELOW and cur["open"]:
            cur["closes"] += 1
            cur["open"] = False
        cur["t1"] = r["wall"]
    if cur is not None:
        cur["end"] = "trace_end"
        out.append(cur)
    for p in out:
        p["mode"] = "on" if p["applied"] else ("shadow" if p["res"] else "off")
        p["grasped"] = p["end"] == "bring"
        p["first_attempt"] = p["grasped"] and p["closes"] == 1
        p["seconds"] = round(p["t1"] - p["t0"], 1)
        n = max(p.pop("o_n"), 1)
        s = [v / n for v in p.pop("o_sum")]
        p["mean_abs_offset"] = [round(v * 1e3, 1) for v in s[:3]] + [round(v, 2) for v in s[3:]]
        p.pop("open")
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] (successes / failures of two groups)."""
    n, r1, c1 = a + b + c + d, a + b, a + c

    def pr(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)

    p0 = pr(a)
    return round(min(1.0, sum(pr(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if pr(x) <= p0 + 1e-12)), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    rows = [p for f in a.traces for p in picks(f)]
    for p in rows:
        print(json.dumps({k: p[k] for k in ("file", "mode", "closes", "grasped", "first_attempt", "seconds", "end",
                                            "decisions", "mean_abs_offset")}))
    summ = {}
    cut = [p for p in rows if p["end"] == "trace_end"]  # the run ended mid-pick: neither success nor failure
    if cut:
        print(f"{len(cut)} pick(s) cut off by the end of a trace: left out of the counts")
    for m in ("off", "shadow", "on"):
        g = [p for p in rows if p["mode"] == m and p["end"] != "trace_end"]
        if g:
            k1, k2 = sum(p["first_attempt"] for p in g), sum(p["grasped"] for p in g)
            summ[m] = (k1, len(g))
            print(f"{m:6s} picks {len(g):3d}  first-attempt {k1}/{len(g)} = {k1 / len(g):.2f} {wilson(k1, len(g))}  "
                  f"grasped {k2}/{len(g)} = {k2 / len(g):.2f} {wilson(k2, len(g))}  "
                  f"median s {sorted(p['seconds'] for p in g)[len(g) // 2]}")
    base = summ.get("off") or summ.get("shadow")  # shadow commands = off commands
    if base and "on" in summ:
        (k0, n0), (k1, n1) = base, summ["on"]
        print(f"first-attempt on - off: {k1 / n1 - k0 / n0:+.2f}, Fisher exact p = {fisher(k1, n1 - k1, k0, n0 - k0)}")
    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
            w.writeheader()
            w.writerows(rows)
    if not rows:
        sys.exit("no pick ticks in these traces (was --trace '' set?)")


if __name__ == "__main__":
    main()
