"""Summary figure of the Isaac validation (val_isaac.json): lost motion at reversals, hold error by approach direction,
and what PhysX adds around the gripper-close frame.  sim/twin/dynamics/py.sh sim/twin/dynamics/plots.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "sim/twin/out/dynamics"
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3de"
COL = {"real": INK, "p_none": "#2a78d6", "p_compensate": "#eb6834"}
LAB = {"real": "real robot (encoders)", "p_none": "twin today", "p_compensate": "twin + play model"}
SHORT = ["pan", "lift", "elbow", "w.flex", "w.roll"]


def style(a):
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(MUTED)
    a.tick_params(colors=MUTED)


def main(path=OUT / "val_isaac.json"):
    v = json.loads(Path(path).read_text())
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1.3]})
    # (a) lost motion
    a = ax[0]
    x = np.arange(5)
    vals = {"real": v["real"]["lost_motion"], "p_none": v["p_none"]["lost_motion_sim"],
            "p_compensate": v["p_compensate"]["lost_motion_sim"]}
    for i, (k, y) in enumerate(vals.items()):
        a.bar(x + (i - 1) * 0.26, y, 0.24, color=COL[k], label=LAB[k], edgecolor="white", linewidth=2)
    a.set_xticks(x, SHORT)
    a.set_ylabel("command travel before the joint follows (deg)", color=INK)
    a.set_title("Lost motion at command reversals", color=INK, loc="left")
    a.legend(frameon=False, fontsize=9)
    style(a)
    # (b) hold error by approach direction
    a = ax[1]
    js = ["shoulder_lift", "elbow_flex", "wrist_flex", "shoulder_pan"]
    for i, (k, src) in enumerate((("real", v["real"]["hold_by_dir"]), ("p_none", v["p_none"]["hold_by_dir_sim"]),
                                  ("p_compensate", v["p_compensate"]["hold_by_dir_sim"]))):
        for jj, jn in enumerate(js):
            lo, hi = src[jn]
            y = jj * 1.0 + (i - 1) * 0.25
            a.plot([lo, hi], [y, y], color=COL[k], lw=2)
            a.plot([lo], [y], "o", ms=8, color=COL[k], mec="white", mew=1.5)
            a.plot([hi], [y], "s", ms=8, color=COL[k], mec="white", mew=1.5, label=LAB[k] if jj == 0 else None)
    a.set_yticks(range(len(js)), ["lift", "elbow", "w.flex", "pan"])
    a.axvline(0, color=MUTED, lw=0.8)
    a.set_xlabel("measured − commanded at rest (deg)   ● came from below   ■ came from above", color=INK)
    a.set_title("Where the joint stops, by approach direction", color=INK, loc="left")
    a.legend(frameon=False, fontsize=9, loc="center right")
    style(a)
    a.grid(axis="x", color=GRID, lw=0.8)
    # (c) PhysX around the close frame
    a = ax[2]
    for k in ("p_none", "p_compensate"):
        d = v[k]["isaac_minus_nophysics"]
        a.plot(d["ticks_from_close"], d["fin_tip_dz_mm_median"], color=COL[k], lw=2, marker="o", ms=5,
               label=LAB[k])
    a.axvline(0, color=MUTED, lw=0.8, ls="--")
    a.axhline(0, color=MUTED, lw=0.8)
    a.text(0.8, -1.2, "gripper-close\nframe", color=MUTED, fontsize=9)
    a.set_xlabel("ticks from the gripper-close frame (30 Hz)", color=INK)
    a.set_ylabel("fin-tip height: Isaac − same servo model without physics (mm)", color=INK)
    a.set_title("Contact holds the twin's fingers up at the grasp", color=INK, loc="left")
    a.legend(frameon=False, fontsize=9)
    style(a)
    fig.suptitle("SO-101 twin vs real: 60 held-out cal2 demos replayed in Isaac (rung 3, W offsets)", color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "validation_isaac.png", dpi=120)
    print("wrote", OUT / "validation_isaac.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
