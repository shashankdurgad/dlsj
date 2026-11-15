"""Recorded frames + joint states from a LeRobot v3 dataset (PyAV, no LeRobot import)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
CAMS = ("top", "wrist", "inspect")


@lru_cache(maxsize=4)
def episodes(root: str = str(DEFAULT_ROOT)):
    files = sorted(Path(root, "meta/episodes").glob("*/*.parquet"))
    return (
        pq.read_table(files[0]).to_pandas()
        if len(files) == 1
        else __import__("pandas").concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    )


@lru_cache(maxsize=4)
def frames_table(root: str = str(DEFAULT_ROOT)):
    import pandas as pd

    t = pd.concat(
        [
            pq.read_table(
                f, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"]
            ).to_pandas()
            for f in sorted(Path(root, "data").glob("*/*.parquet"))
        ],
        ignore_index=True,
    )
    return t


def labels(root: str = str(DEFAULT_ROOT)) -> dict[int, dict]:
    p = Path(root, "meta/dlsj/labels.jsonl")
    return {r["episode_index"]: r for r in map(json.loads, p.read_text().splitlines()) if r} if p.exists() else {}


def states(ep: int, root: str = str(DEFAULT_ROOT)) -> np.ndarray:
    t = frames_table(root)
    s = t[t.episode_index == ep].sort_values("frame_index")
    return np.stack(s["observation.state"].to_numpy()).astype(np.float64)


def fps(root: str = str(DEFAULT_ROOT)) -> int:
    return int(json.loads(Path(root, "meta/info.json").read_text())["fps"])


def video_frames(ep: int, cam: str, idx: list[int], root: str = str(DEFAULT_ROOT)) -> dict[int, np.ndarray]:
    """RGB uint8 frames (H, W, 3) of episode `ep`, camera `cam`, at frame indices `idx`."""
    e = episodes(root).set_index("episode_index").loc[ep]
    k = f"videos/observation.images.{cam}"
    path = Path(
        root,
        "videos",
        f"observation.images.{cam}",
        f"chunk-{int(e[k + '/chunk_index']):03d}",
        f"file-{int(e[k + '/file_index']):03d}.mp4",
    )
    t0, f = float(e[k + "/from_timestamp"]), fps(root)
    want = {round((t0 + i / f) * f): i for i in idx}
    out = {}
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(0.0, t0 + min(idx) / f - 1.0) / s.time_base), stream=s)
        for fr in c.decode(s):
            n = round(float(fr.pts * s.time_base) * f)
            if n in want and want[n] not in out:
                out[want[n]] = fr.to_ndarray(format="rgb24")
            if len(out) == len(want) or n > max(want):
                break
    return out
