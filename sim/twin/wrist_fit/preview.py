"""Contact sheet of wrist edits (CPU): python sim/twin/wrist_fit/preview.py --sim <npz> --edits a b c --out x.jpg"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.wrist_fit.edits import apply  # noqa: E402

SET = REPO / "sim/twin/out/audit/set"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--edits", nargs="+", default=["sim:"])
    ap.add_argument("--samples", default="2,9,37,58")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = json.loads((SET / "set.json").read_text())["samples"]
    lays = json.loads((SET / "layouts_first.json").read_text())
    real = np.load(SET / "real.npz")["wrist"]
    sz = np.load(a.sim)
    full = sz["wrist"].shape[0] == len(S)
    idx = [i for i, s in enumerate(S) if s["phase"] in ("reach", "pregrasp")]
    rows = []
    for i in [int(x) for x in a.samples.split(",")]:
        j = i if full else idx.index(i)
        ctx = {"real": real[i], "sim": sz["wrist"][j], "mask": sz["mask_wrist"][j], "s": S[i],
               "lay": lays.get(str(S[i]["ep"]))}
        row = [real[i]]
        for e in a.edits:
            base, _, ed = e.partition(":")
            row.append(apply(ed, ctx["real"] if base == "real" else ctx["sim"], ctx))
        rows.append(cv2.resize(np.concatenate(row, 1), None, fx=0.5, fy=0.5))
    cv2.imwrite(a.out, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
