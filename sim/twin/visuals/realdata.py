"""Read recorded LeRobot v3 frames / joint states (no Isaac): episode index -> camera frame + observation.state.

    from sim.twin.visuals.realdata import Dataset
    ds = Dataset()                         # default: sim/twin/data/canonical/duplo_teleop_cal2_rgb_456
    rgb = ds.frame(12, "top", 0)          # HxWx3 uint8, frame 0 of episode 12
    s = ds.state(12, 0)                   # 6 floats: 5 joints (deg) + gripper (0-100)
"""
from __future__ import annotations

import glob
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEFAULT_DS = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"


class Dataset:
    def __init__(self, root: str | Path = DEFAULT_DS):
        import pandas as pd
        import pyarrow.parquet as pq

        self.root = Path(root)
        fs = sorted(glob.glob(str(self.root / "meta/episodes/*/*.parquet")))
        self.eps = pd.concat([pq.read_table(f).to_pandas() for f in fs]).set_index("episode_index", drop=False)
        self.fps = 30

    def __len__(self):
        return len(self.eps)

    def task(self, ep: int) -> list:
        return list(self.eps.loc[ep, "tasks"])

    @lru_cache(maxsize=8)  # noqa: B019 (one dataset object per process)
    def _data(self, chunk: int, file: int):
        import pyarrow.parquet as pq

        t = pq.read_table(self.root / f"data/chunk-{chunk:03d}/file-{file:03d}.parquet",
                          columns=["episode_index", "frame_index", "observation.state", "action"]).to_pandas()
        return t

    def states(self, ep: int) -> np.ndarray:
        r = self.eps.loc[ep]
        t = self._data(int(r["data/chunk_index"]), int(r["data/file_index"]))
        t = t[t.episode_index == ep].sort_values("frame_index")
        return np.stack(t["observation.state"].to_numpy()).astype(np.float64)

    def state(self, ep: int, idx: int) -> np.ndarray:
        return self.states(ep)[idx]

    def video_path(self, ep: int, cam: str) -> tuple[Path, float]:
        r = self.eps.loc[ep]
        k = f"videos/observation.images.{cam}"
        p = self.root / k / f"chunk-{int(r[k + '/chunk_index']):03d}/file-{int(r[k + '/file_index']):03d}.mp4"
        return p, float(r[k + "/from_timestamp"])

    def frames(self, ep: int, cam: str, idxs) -> list[np.ndarray]:
        """Decode several frames (episode frame indices, any order) of one episode/camera."""
        import av

        path, t0 = self.video_path(ep, cam)
        idxs = list(idxs)
        want = {i: t0 + i / self.fps for i in idxs}
        out = {}
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            for i in sorted(idxs):
                ts = want[i]
                c.seek(int(max(0.0, ts - 0.5) / s.time_base), stream=s, backward=True)
                for fr in c.decode(s):
                    t = float(fr.pts * s.time_base)
                    if t >= ts - 0.5 / self.fps:
                        out[i] = fr.to_ndarray(format="rgb24")
                        break
        return [out[i] for i in idxs]

    def frame(self, ep: int, cam: str, idx: int = 0) -> np.ndarray:
        return self.frames(ep, cam, [idx])[0]
