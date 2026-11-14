"""MolmoAct2-SO100_101 zero-shot policy worker for the Isaac SO-101 harness (file IPC, same protocol as pi05).

Joint frame: the checkpoint uses legacy LeRobot SO-100/101 joints. Physical AI Studio's transform
(policies/molmoact2/so101.py, constants.py): checkpoint = sign * runtime + offset with
signs (1, -1, 1, 1, 1, 1), offsets (0, 90, 90, 0, 0, 0) degrees, scale 1 for our degree-mode data; gripper 0-100 both.
Our harness state is in the team's runtime degrees, so: ckpt = S * ours + O, ours = S * (ckpt - O).
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, "/home/yan/pi05bench")
import joeenv  # noqa: F401
import numpy as np
import torch
from PIL import Image

S = np.array([1, -1, 1, 1, 1, 1], np.float32)
O = np.array([0, 90, 90, 0, 0, 0], np.float32)
to_ckpt = lambda s: S * s + O  # noqa: E731
from_ckpt = lambda c: S * (c - O)  # noqa: E731

ap = argparse.ArgumentParser()
ap.add_argument("--ipc", type=Path, required=True)
ap.add_argument("--snap", required=True)
ap.add_argument("--cams", default="top,inspect")
ap.add_argument("--dtype", default="bfloat16")
args = ap.parse_args()
args.ipc.mkdir(parents=True, exist_ok=True)

from transformers import AutoModelForImageTextToText, AutoProcessor

t0 = time.perf_counter()
proc = AutoProcessor.from_pretrained(args.snap, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(args.snap, trust_remote_code=True,
                                                    dtype=getattr(torch, args.dtype)).to("cuda").eval()
stats = json.load(open(Path(args.snap) / "norm_stats.json"))["metadata_by_tag"]["so100_so101_molmoact2"]["state_stats"]
Q01, Q99 = np.array(stats["q01"]), np.array(stats["q99"])
load_s = time.perf_counter() - t0
print("POLICY_READY", f"load {load_s:.1f}s", torch.cuda.get_device_name(0), flush=True)
(args.ipc / "ready").write_text(json.dumps({"model": "molmoact2", "load_s": load_s}))
cams = args.cams.split(",")
i = 0
while True:
    f = args.ipc / f"obs-{i:04d}.npz"
    while not f.exists():
        if (args.ipc / "done").exists():
            print("WORKER_DONE", flush=True); sys.exit(0)
        time.sleep(0.02)
    time.sleep(0.05)
    d = np.load(f, allow_pickle=False)
    ours = d["state"].astype(np.float32)
    ck = to_ckpt(ours)
    imgs = [Image.fromarray(d[c]) for c in cams]
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.inference_mode():
        out = model.predict_action(processor=proc, images=imgs, task=str(d["instruction"].item()), state=ck,
                                   norm_tag="so100_so101_molmoact2", inference_action_mode="continuous",
                                   enable_depth_reasoning=False, num_steps=10, normalize_language=True,
                                   enable_cuda_graph=True)
    torch.cuda.synchronize(); lat = time.perf_counter() - t
    a_ck = out.actions.float().cpu().numpy().reshape(-1, 6)
    a = from_ckpt(a_ck)
    tmp = args.ipc / f"actions-{i:04d}.tmp.npz"
    np.savez(tmp, actions=a.astype(np.float32), latency_s=lat)
    os.replace(tmp, args.ipc / f"actions-{i:04d}.npz")
    print(json.dumps({"i": i, "lat": round(lat, 3), "n": len(a), "state_ours": ours.round(1).tolist(),
                      "state_ckpt": ck.round(1).tolist(),
                      "in_q01_q99": ((ck >= Q01) & (ck <= Q99)).tolist(),
                      "a0_minus_state": (a[0] - ours).round(1).tolist(),
                      "a_min": a.min(0).round(1).tolist(), "a_max": a.max(0).round(1).tolist()}), flush=True)
    i += 1
