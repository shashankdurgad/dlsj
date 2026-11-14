"""Per-episode facts WARP needs, read from a LeRobot v3 dataset and the dlsj sidecar.

* length, global start index
* success (meta/dlsj/labels.jsonl; missing -> True)
* inspection_frame: the pick -> place split (labels.jsonl, else the first frame whose task changes)
* gripper command per frame (last action dimension named like "gripper")
"""

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Episode:
    index: int
    length: int
    start: int  # global frame index of frame 0
    success: bool
    split: int | None
    gripper: np.ndarray | None


def load_episodes(root: str | Path) -> tuple[list[Episode], float]:
    root = Path(root).expanduser()
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    names = info["features"]["action"].get("names") or []
    names = names if isinstance(names, list) else list(names.values())[0] if isinstance(names, dict) else []
    g_col = next((i for i, n in enumerate(names) if "gripper" in str(n)), None)

    cols = ["episode_index", "frame_index", "index", "action", "task_index"]
    frames = pd.concat(pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"),
                                                                                       recursive=True)))
    frames = frames.sort_values("index")

    labels = {}
    lab = root / "meta" / "dlsj" / "labels.jsonl"
    if lab.exists():
        for line in lab.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                labels[int(r["episode_index"])] = r

    eps = []
    for ep, g in frames.groupby("episode_index", sort=True):
        r = labels.get(int(ep), {})
        split = r.get("inspection_frame")
        if split is None:
            change = np.flatnonzero(np.diff(g["task_index"].to_numpy()) != 0)
            split = int(change[0] + 1) if len(change) else None
        grip = None
        if g_col is not None:
            grip = np.stack(g["action"].to_numpy())[:, g_col].astype(np.float32)
        eps.append(Episode(int(ep), len(g), int(g["index"].iloc[0]), bool(r.get("success", True)),
                           None if split is None else int(split), grip))
    return eps, fps


def load_features(feat_dir: str | Path, cameras: list[str], ep: int) -> np.ndarray:
    """Concatenate cached per-camera features for one episode: float32 [T, sum(D)]."""
    feat_dir = Path(feat_dir).expanduser()
    parts = [np.load(feat_dir / c / f"episode_{ep:06d}.npy").astype(np.float32) for c in cameras]
    t = min(len(p) for p in parts)
    return np.concatenate([p[:t] for p in parts], axis=1)
