"""Batched MolmoAct2 worker for the residual env (bases.IPCVLABase protocol), on the Spark's GPU in bf16.

Request  obs-NNNNN.npz : state (n,6) our degrees (current calibration), wrist / top (n,480,640,3) uint8,
                         instruction (n,) str
Reply    actions-NNNNN.npz : actions (n,30,6), latency_s (wall time of the batched call)
`ready` is written once the policy is loaded; `done` stops the worker. Processed requests/replies are deleted
(keep_last of each kept for debugging) so long RL runs don't fill the disk.

    HF_HUB_OFFLINE=1 HF_HUB_CACHE=~/molmobench/hf ~/molmo-cuda-venv/bin/python molmo_batch_worker.py \
        --ipc ~/residual_env/ipc_molmo [--ckpt /path/lora.ckpt | --pretrained allenai/MolmoAct2-SO100_101]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from molmo_policy import MolmoBase  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ipc", type=Path, required=True)
ap.add_argument("--pretrained", default="allenai/MolmoAct2-SO100_101")
ap.add_argument("--ckpt", default=None, help="LoRA Lightning checkpoint (overrides --pretrained)")
ap.add_argument("--max_batch", type=int, default=16)
ap.add_argument("--keep_last", type=int, default=3)
args = ap.parse_args()
args.ipc.mkdir(parents=True, exist_ok=True)
for p in list(args.ipc.glob("obs-*.npz")) + list(args.ipc.glob("actions-*.npz")) + [args.ipc / "done", args.ipc / "ready"]:
    p.unlink(missing_ok=True)

base = MolmoBase(None if args.ckpt else args.pretrained, args.ckpt)
info = {"load_s": round(base.load_s, 1), "text_len": base.text_len, "source": base.source}
(args.ipc / "ready").write_text(json.dumps(info))
print("POLICY_READY", json.dumps(info), flush=True)

i = 0
lat = []
while True:
    f = args.ipc / f"obs-{i:05d}.npz"
    while not f.exists():
        if (args.ipc / "done").exists():
            print("WORKER_DONE", json.dumps({"requests": i, "median_latency_s": float(np.median(lat)) if lat else None}),
                  flush=True)
            sys.exit(0)
        time.sleep(0.003)
    time.sleep(0.005)
    d = np.load(f, allow_pickle=False)
    n = d["state"].shape[0]
    t = time.perf_counter()
    outs = []
    for s in range(0, n, args.max_batch):
        e = slice(s, min(n, s + args.max_batch))
        outs.append(base.predict(d["state"][e], d["wrist"][e], d["top"][e], [str(x) for x in d["instruction"][e]]))
    a = np.concatenate(outs, 0).astype(np.float32)
    dt = time.perf_counter() - t
    lat.append(dt)
    tmp = args.ipc / f"actions-{i:05d}.tmp.npz"
    np.savez(tmp, actions=a, latency_s=dt)
    os.replace(tmp, args.ipc / f"actions-{i:05d}.npz")
    old = i - args.keep_last
    if old >= 0:
        (args.ipc / f"obs-{old:05d}.npz").unlink(missing_ok=True)
        (args.ipc / f"actions-{old:05d}.npz").unlink(missing_ok=True)
    if i % 50 == 0:
        print("REQ", i, "n", n, "latency_s", round(dt, 3), flush=True)
    i += 1
