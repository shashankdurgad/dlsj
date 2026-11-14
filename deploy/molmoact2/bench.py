"""Clean latency re-bench of MolmoAct2 exports on the XPS GPU: interleaved rounds, load recorded per round.

Each round loads every export in turn (one model at a time) and times N calls on the same real frame; the median over
rounds is reported together with the 1-min load average seen before each measurement, so contention is visible
instead of silently inflating one variant.

    python bench.py EXPORT_DIR [EXPORT_DIR ...] --rounds 2 --iters 15 [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--frames-cache", default="~/inference/exports/molmo_opt/zeroshot/frames_zeroshot_60.npz")
    ap.add_argument("--cache-dir", default="~/inference/ov_cache")
    ap.add_argument("--json")
    a = ap.parse_args()
    z = np.load(Path(a.frames_cache).expanduser(), allow_pickle=True)
    frame = [f for f in z["frames"] if f["phase"] == "grasp"][0]
    res = {e: {"median_ms": [], "load1": []} for e in a.exports}
    for r in range(a.rounds):
        for e in a.exports:
            load1 = os.getloadavg()[0]
            _, lat = H.predict(e, [frame] * a.iters, "GPU", a.cache_dir)
            res[e]["median_ms"].append(lat["median_ms"])
            res[e]["load1"].append(round(load1, 2))
            print(f"round {r} {Path(e).name}: {lat['median_ms']} ms (load {load1:.1f})", flush=True)
    out = {Path(e).name: {"median_ms": round(float(np.median(v["median_ms"])), 1), "rounds": v["median_ms"],
                          "load1": v["load1"]} for e, v in res.items()}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
