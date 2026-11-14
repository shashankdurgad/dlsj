"""Batched SmolVLA worker for the residual env (IPCVLABase). Same loading as eval/smolvla/smol_worker.py
(Studio Lightning checkpoint, wrist + top, recorder units), but one request carries a batch of envs.

    HF_HOME=~/smolbench/hf ~/isaacsim-venv/bin/python vla_batch_worker.py \
        --ckpt ~/smolbench/ckpt/checkpoints/snapshots/step020000.ckpt --ipc ~/residual_env/ipc
"""
import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/yan/pi05bench")
import joeenv  # noqa: F401,E402  (torch + Joe's physicalai/lerobot site-packages)
import numpy as np  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ipc", type=Path, required=True)
ap.add_argument("--max_batch", type=int, default=16)
args = ap.parse_args()
args.ipc.mkdir(parents=True, exist_ok=True)

from physicalai.data import Observation  # noqa: E402
from physicalai.policies import SmolVLA  # noqa: E402

torch.manual_seed(42)
t0 = time.perf_counter()
ck = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
sig = inspect.signature(SmolVLA.__init__).parameters
hp = {k: v for k, v in ck["hyper_parameters"].items() if k in sig}
hp.update(pretrained_name_or_path=None, load_vlm_weights=False, compile_model=False)
policy = SmolVLA(**hp)
policy.on_load_checkpoint(ck)
missing, unexpected = policy.load_state_dict(ck["state_dict"], strict=False)
print("LOAD", json.dumps({"missing": len(missing), "unexpected": len(unexpected)}), flush=True)
del ck
policy = policy.eval().to("cuda")
policy.requires_grad_(False)
(args.ipc / "ready").write_text(json.dumps({"load_s": time.perf_counter() - t0}))
print("POLICY_READY", flush=True)

i = 0
while True:
    f = args.ipc / f"obs-{i:05d}.npz"
    while not f.exists():
        if (args.ipc / "done").exists():
            sys.exit(0)
        time.sleep(0.005)
    time.sleep(0.01)
    d = np.load(f, allow_pickle=False)
    n = d["state"].shape[0]
    t = time.perf_counter()
    outs = []
    for s in range(0, n, args.max_batch):
        e = slice(s, min(n, s + args.max_batch))
        obs = Observation(
            state=torch.from_numpy(d["state"][e].copy()),
            images={c: torch.from_numpy(d[c][e].copy()).permute(0, 3, 1, 2).float() / 255 for c in ("wrist", "top")},
            task=[str(x) for x in d["instruction"][e]],
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(policy.predict_action_chunk(obs).float().cpu().numpy())
    torch.cuda.synchronize()
    a = np.concatenate(outs, 0)
    tmp = args.ipc / f"actions-{i:05d}.tmp.npz"
    np.savez(tmp, actions=a, latency_s=time.perf_counter() - t)
    os.replace(tmp, args.ipc / f"actions-{i:05d}.npz")
    i += 1
