"""Frozen base policies for the residual env. All return chunks (n, H, 6) in our units (old-calibration degrees,
gripper 0-100), aligned so chunk[:, j] is the target for tick (request tick + j)."""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_episodes(root: str | Path, episodes: list[int] | None = None) -> dict[int, dict]:
    """{ep: {"action": (T,6), "state": (T,6), "grasp_frame", "inspection_frame", "bowl"}} from a LeRobot dataset."""
    root = Path(root)
    lf = root / "meta/dlsj/labels.jsonl"
    labels = {}
    if lf.exists():
        for line in lf.read_text().splitlines():
            if line.strip():
                lab = json.loads(line)
                labels[int(lab["episode_index"])] = lab
    df = pd.concat(pd.read_parquet(f, columns=["action", "observation.state", "episode_index", "frame_index"])
                   for f in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True)))
    out = {}
    for ep, g in df.groupby("episode_index"):
        ep = int(ep)
        if episodes is not None and ep not in episodes:
            continue
        g = g.sort_values("frame_index")
        lab = labels.get(ep, {})
        out[ep] = {"action": np.stack(g["action"].values).astype(np.float32),
                   "state": np.stack(g["observation.state"].values).astype(np.float32),
                   "grasp_frame": lab.get("grasp_frame"), "inspection_frame": lab.get("inspection_frame"),
                   "bowl": lab.get("bowl_final", lab.get("bowl")), "excluded": lab.get("excluded")}
    return out


class HoldBase:
    """Keeps the commanded pose (no base motion): pure-residual sanity tests."""

    needs_images = False

    def __init__(self, horizon: int = 50):
        self.H = horizon

    def reset(self, ids):
        pass

    def request(self, ids, obs):
        s = obs["state"]
        return s.unsqueeze(1).expand(len(ids), self.H, 6).clone()


class ReplayBase:
    """Replays recorded teleop actions (one episode per env) as if they were the base policy's chunks: realistic,
    human-like base motion for fast residual-RL plumbing tests without a VLA in the loop."""

    needs_images = False

    def __init__(self, episodes: dict[int, dict], horizon: int = 50, device: str = "cuda:0", seed: int = 0):
        self.eps = list(episodes.values())
        self.H, self.device = horizon, device
        self.rng = np.random.default_rng(seed)
        self.assign: dict[int, int] = {}

    def reset(self, ids):
        for i in ids.tolist():
            self.assign[i] = int(self.rng.integers(len(self.eps)))

    def episode(self, i: int) -> dict:
        return self.eps[self.assign[i]]

    def request(self, ids, obs):
        out = []
        for k, i in enumerate(ids.tolist()):
            a = self.eps[self.assign[i]]["action"]
            t = int(obs["tick"][k])
            idx = np.clip(np.arange(t, t + self.H), 0, len(a) - 1)
            out.append(torch.from_numpy(a[idx]))
        return torch.stack(out).to(self.device)


class IPCVLABase:
    """Frozen VLA in a separate process (its own Python env), batched over envs via npz file IPC.

    Request file obs-XXXXX.npz: state (n,6), wrist/top (n,480,640,3) uint8, instruction (n,) str.
    Reply actions-XXXXX.npz: actions (n,H,6), latency_s. Worker: sim/residual_env/vla_batch_worker.py.
    The env applies the latency model in SIM time (cfg.base_latency_s), independent of the worker's wall time.
    """

    needs_images = True

    def __init__(self, ipc: str | Path, device: str = "cuda:0", timeout_s: float = 600.0):
        self.ipc, self.device, self.timeout = Path(ipc), device, timeout_s
        self.k = 0
        self.lat: list[float] = []
        t_end = time.time() + timeout_s
        while not (self.ipc / "ready").exists():
            if time.time() > t_end:
                raise TimeoutError("VLA worker not ready")
            time.sleep(1)

    def reset(self, ids):
        pass

    def request(self, ids, obs):
        imgs = obs["images"]
        tmp = self.ipc / f"obs-{self.k:05d}.tmp.npz"
        np.savez(tmp, state=obs["state"].cpu().numpy(), wrist=imgs["wrist"].cpu().numpy().astype(np.uint8),
                 top=imgs["top"].cpu().numpy().astype(np.uint8), instruction=np.asarray(obs["instruction"]))
        os.replace(tmp, self.ipc / f"obs-{self.k:05d}.npz")
        f = self.ipc / f"actions-{self.k:05d}.npz"
        t_end = time.time() + self.timeout
        while not f.exists():
            if time.time() > t_end:
                raise TimeoutError("VLA worker timed out")
            time.sleep(0.005)
        time.sleep(0.01)
        r = np.load(f)
        self.lat.append(float(r["latency_s"]))
        self.k += 1
        return torch.from_numpy(r["actions"].astype(np.float32)).to(self.device)
