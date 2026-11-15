"""Shared MolmoAct2 prior server for several env processes on one machine (AF_UNIX, like the pi0.5 template's
script/dicerl_libero_pi05_server.py): one model on one GPU; every request returns K candidate chunks from one
prefix forward (MolmoAct2 worker `act_k`) plus the pooled prefix latent.

Requests that arrive together are served in one round (the prefix forwards run back to back; batching B
observations through one prefix forward is the next step and needs a batch-B preprocessor call).

    python sim/twin/rl/prior_server.py --socket /tmp/molmo.sock [--snap PATH] [--steps 10]      # server
    from sim.twin.rl.prior_server import RemotePrior; p = RemotePrior("/tmp/molmo.sock")        # client
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing.connection import Client, Listener, wait
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from sim.twin.policies.molmoact2 import CHUNK, DEFAULT_SNAP, DEPLOY  # noqa: E402
from sim.twin.rl.priors import Prior  # noqa: E402


def serve(socket_path: str, snap: str | None, steps: int, cams=("wrist", "top"), max_round: int = 32):
    """Load the model in this process and answer {"op": "act_k", ...} / {"op": "ping"} requests."""
    from sim.twin.policies.molmoact2 import _Worker

    w = _Worker({"snap": snap or DEFAULT_SNAP, "cams": list(cams), "steps": steps, "device": "cuda",
                 "noise": "random", "seed": 0, "dtype": "bfloat16", "rtc_horizon": DEPLOY["rtc_horizon"],
                 "rtc_guidance": DEPLOY["rtc_guidance"], "text_len": None})
    listener = Listener(socket_path + ".tmp", family="AF_UNIX", backlog=64)
    listener._listener._unlink.cancel()
    os.replace(socket_path + ".tmp", socket_path)
    sock = listener._listener._socket
    conns, served, t0 = [], 0, time.perf_counter()
    print("PRIOR_SERVER_READY", w.ready, flush=True)
    try:
        while True:
            ready = wait([sock] + conns, timeout=5.0)
            batch = []
            for c in ready:
                if c is sock:
                    conns.append(listener.accept())
                    continue
                try:
                    batch.append((c, c.recv()))
                except (EOFError, OSError):
                    conns.remove(c)
            for c, req in batch[:max_round]:
                try:
                    if req.get("op") == "ping":
                        c.send({"ok": True, "served": served, "uptime_s": time.perf_counter() - t0})
                        continue
                    req = dict(req, op="act_k")
                    req.setdefault("cams", list(cams))
                    c.send(w.act_k(req))
                    served += 1
                except Exception as e:  # noqa: BLE001 (reply with the error, keep serving)
                    c.send({"error": f"{type(e).__name__}: {e}"})
    finally:
        listener.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


class RemotePrior(Prior):
    """Prior interface on top of a running prior server (obs_batch as in MolmoAct2Prior)."""

    needs_images = True

    def __init__(self, socket_path: str, exec_len: int = 15):
        self.conn = Client(socket_path, family="AF_UNIX")
        self.chunk_len, self.exec_len = CHUNK, exec_len
        self.last_info: list = []

    def sample(self, obs_batch, k: int = 4, seeds=None) -> np.ndarray:
        out, lat, info = [], [], []
        for b, obs in enumerate(obs_batch):
            sd = list(range(k)) if seeds is None else np.asarray(seeds)[b].reshape(-1).tolist()
            req = {"k": k, "seeds": sd, "images": obs["images"], "state": np.asarray(obs["state"], np.float32),
                   "task": obs.get("task") or DEPLOY["pick_prompt"], "steps": None, "latent": True}
            if obs.get("prev_chunk") is not None:
                req["prev_chunk"] = np.asarray(obs["prev_chunk"], np.float32)
                req["inference_delay"] = int(obs.get("inference_delay", 0))
            self.conn.send(req)
            r = self.conn.recv()
            if "error" in r:
                raise RuntimeError(r["error"])
            out.append(r["actions"])
            lat.append(r["latent"])
            info.append({kk: v for kk, v in r.items() if kk not in ("actions", "latent")})
        self.last_info = info
        if lat and lat[0] is not None:
            self.last_latent = np.stack(lat)
            self.latent_dim = self.last_latent.shape[1]
        return np.stack(out)

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--snap", default=None)
    ap.add_argument("--steps", type=int, default=DEPLOY["steps"])
    a = ap.parse_args()
    serve(a.socket, a.snap, a.steps)
