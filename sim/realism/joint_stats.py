"""Stage A2: per-episode key joint states and phase timing from the 3-phase labels (all episodes, parquet only).

Key frames per episode: start (0), grasp (grasp_frame), inspection (inspection_frame), release (first frame after the
inspection mark where the gripper opens past `open_thr` after having been closed), end. States are converted to the
OLD calibration frame (cal2 minus the recal_2026-11-14 offset), which is the frame of Joe's sim conversion.

    ~/inference/.venv/bin/python joint_stats.py --out ~/inference/tmp/realism/joints.json
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

SETS = {
    "old": ("/home/intel/desmond/dlsj/data/duplo_teleop", "/home/intel/inference/data/duplo_teleop_3phase", np.zeros(6)),
    "cal2": ("/home/intel/desmond/build/dlsj-main/data/duplo_teleop_cal2", "/home/intel/inference/data/duplo_teleop_cal2_3phase",
             np.array([20.35, -7.96, 0.18, 0.04, 0.0, 0.0])),
}


def release_frame(grip: np.ndarray, start: int, open_thr: float = 15.0) -> int | None:
    for f in range(max(start, 1), len(grip)):
        if grip[f - 1] <= open_thr < grip[f]:
            return f
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    eps_out = []
    for name, (root, lab_root, off) in SETS.items():
        fps = json.loads((Path(root) / "meta/info.json").read_text())["fps"]
        fs = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))
        df = pd.concat(pd.read_parquet(f, columns=["observation.state", "episode_index", "frame_index"]) for f in fs)
        labels = {l["episode_index"]: l for l in map(json.loads, open(lab_root + "/meta/dlsj/labels.jsonl"))}
        for e, g in df.groupby("episode_index"):
            l = labels.get(int(e))
            if not l or l.get("excluded"):
                continue
            S = np.stack(g.sort_values("frame_index")["observation.state"].values) - off  # -> old calibration
            gf, insp = int(l["grasp_frame"]), int(l["inspection_frame"])
            rel = release_frame(S[:, 5], insp)
            eps_out.append({
                "set": name, "episode": int(e), "bowl": l["bowl_final"], "fps": fps, "n": int(len(S)),
                "grasp_frame": gf, "inspection_frame": insp, "release_frame": rel,
                "start": S[0].round(2).tolist(), "grasp": S[gf].round(2).tolist(),
                "inspect": S[min(insp, len(S) - 1)].round(2).tolist(),
                "release": None if rel is None else S[rel].round(2).tolist(),
                "end": S[-1].round(2).tolist(),
                "gripper_at_grasp_plus10": float(S[min(gf + 10, len(S) - 1), 5]),
            })
    a.out.expanduser().write_text(json.dumps(eps_out))
    print("episodes", len(eps_out), "with release", sum(e["release"] is not None for e in eps_out))


if __name__ == "__main__":
    main()
