"""SmolVLA (w3joe/smolvla-junction-so101-3stage-20261114) worker for the Isaac SO-101 harness (file IPC, like
sim/isaac_pi05/policy_worker.py). Cameras wrist + top (full frame); state/actions in the recorder's calibrated units."""
import argparse, inspect, json, os, sys, time
from pathlib import Path

sys.path.insert(0, "/home/yan/pi05bench")
import joeenv  # noqa: F401  (torch from ~/isaacsim-venv + Joe's physicalai/lerobot site-packages)
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ipc", type=Path, required=True)
args = ap.parse_args()
args.ipc.mkdir(parents=True, exist_ok=True)

from physicalai.data import Observation
from physicalai.policies import SmolVLA

torch.manual_seed(42)
t0 = time.perf_counter()
ck = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
sig = inspect.signature(SmolVLA.__init__).parameters
hp = {k: v for k, v in ck["hyper_parameters"].items() if k in sig}
hp.update(pretrained_name_or_path=None, load_vlm_weights=False, compile_model=False)
policy = SmolVLA(**hp)
policy.on_load_checkpoint(ck)
missing, unexpected = policy.load_state_dict(ck["state_dict"], strict=False)
print("LOAD", json.dumps({"missing": len(missing), "unexpected": len(unexpected), "missing_eg": missing[:3],
                          "step": ck.get("global_step")}), flush=True)
del ck
policy = policy.eval().to("cuda")
policy.requires_grad_(False)
print("POLICY_READY", f"load {time.perf_counter() - t0:.1f}s", torch.cuda.get_device_name(0), flush=True)
(args.ipc / "ready").write_text(json.dumps({"load_s": time.perf_counter() - t0}))

i = 0
while True:
    f = args.ipc / f"obs-{i:04d}.npz"
    while not f.exists():
        if (args.ipc / "done").exists():
            print("WORKER_DONE", flush=True)
            sys.exit(0)
        time.sleep(0.02)
    time.sleep(0.02)
    d = np.load(f, allow_pickle=False)
    obs = Observation(
        state=torch.from_numpy(d["state"].copy()).unsqueeze(0),
        images={n: torch.from_numpy(d[n].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255 for n in ["wrist", "top"]},
        task=[str(d["instruction"].item())],
    )
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = policy.predict_action_chunk(obs)
    torch.cuda.synchronize()
    lat = time.perf_counter() - t
    a = a.float().cpu().numpy()[0]
    assert a.ndim == 2 and a.shape[1] == 6 and np.isfinite(a).all(), a.shape
    tmp = args.ipc / f"actions-{i:04d}.tmp.npz"
    np.savez(tmp, actions=a, latency_s=lat)
    os.replace(tmp, args.ipc / f"actions-{i:04d}.npz")
    print(json.dumps({"i": i, "task": str(d["instruction"].item()), "latency_s": round(lat, 4),
                      "a0": np.round(a[0], 1).tolist(), "a_last": np.round(a[-1], 1).tolist()}), flush=True)
    i += 1
