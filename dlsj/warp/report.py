"""Plots for checking WARP scores before training a policy.

    python -m dlsj.warp.report --weights ~/desmond/warp/weights --out ~/desmond/warp/weights/report.png

Top: v_t traces for the fastest, median and slowest episodes. The dashed line marks the pick -> place split;
shading shows kept chunks (green = pace, blue = protected, orange = coverage). Bottom left: episode length
vs mean v (should slope down). Bottom right: kept fraction per stage x phase slice.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    d = Path(args.weights).expanduser()
    df = pd.read_parquet(d / "warp_weights.parquet")
    eps = pd.read_csv(d / "episodes.csv").sort_values("length_s")
    picks = list(eps.episode_index.iloc[[0, 1, len(eps) // 2, len(eps) // 2 + 1, -2, -1]])
    colours = {"pace": "#3a9d5d", "protected": "#3b6fd1", "coverage": "#e08a1e"}

    fig = plt.figure(figsize=(16, 13))
    for k, ep in enumerate(picks):
        ax = fig.add_subplot(4, 2, k + 1)
        e = df[df.episode_index == ep].sort_values("frame_index")
        t = e.frame_index.to_numpy() / 30.0
        ax.plot(t, e.v, color="black", lw=1.2, label="v (progress velocity)")
        ax.axhline(1.0, color="grey", lw=0.6, ls=":")
        ax.axhline(0.0, color="grey", lw=0.6)
        for reason, c in colours.items():
            m = (e.reason == reason).to_numpy()
            ax.fill_between(t, -1.5, 3, where=m, color=c, alpha=0.18, step="mid", label=reason)
        split = e.loc[e.stage == 1, "frame_index"]
        if len(split):
            ax.axvline(split.iloc[0] / 30.0, color="red", ls="--", lw=1, label="inspect split")
        ln = eps.loc[eps.episode_index == ep, "length_s"].iloc[0]
        ax.set_title(f"episode {ep}: {ln:.1f}s, kept {(e.weight > 0).mean():.0%}")
        ax.set_ylim(-1.5, 3)
        ax.set_xlabel("s")
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
    ax = fig.add_subplot(4, 2, 7)
    ax.scatter(eps.length_s, eps.mean_v, s=10)
    r = np.corrcoef(eps.length_s, eps.mean_v)[0, 1]
    ax.set_xlabel("episode length (s)")
    ax.set_ylabel("mean v")
    ax.set_title(f"longer episodes should score slower (r = {r:.2f})")
    ax = fig.add_subplot(4, 2, 8)
    cov = pd.read_csv(d / "coverage.csv")
    for st, g in cov.groupby("stage"):
        ax.plot(g.phase, g.kept, marker="o", label=st)
    ax.set_ylim(0, 1)
    ax.set_xlabel("phase slice within stage")
    ax.set_ylabel("kept fraction")
    ax.set_title("coverage: no slice of the task may be emptied")
    ax.legend()
    fig.tight_layout()
    out = Path(args.out).expanduser() if args.out else d / "report.png"
    fig.savefig(out, dpi=90)
    print(out)


if __name__ == "__main__":
    main()
