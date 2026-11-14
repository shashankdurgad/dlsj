"""MolmoAct2 latency per 30-action chunk (bf16, 10 flow steps, cameras wrist + top at 480x640, deployed settings).

    /opt/isaaclab-venv/bin/python sim/twin/modal/bench_molmo.py --calls 20 --batches 1,8 --out res.json

batch 1: through sim/twin/policies/molmoact2.py's MolmoAct2Client exactly as the twin uses it (worker subprocess,
pickled IPC): `latency_s` = model time in the worker (pre + flow + post, CUDA-synchronised), `roundtrip_s` adds IPC.
Plain chunks and RTC-guided ones (prev_chunk, as the deployed controller re-plans).
batch > 1: the client has no batched call, so the worker's model (`_Worker`, the same construction) is loaded in this
process and fed a stacked batch through the same pre-processor -> predict_action_chunk -> post-processor path.
Images are synthetic (latency does not depend on content: fixed text length, fixed flow steps).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.policies import molmoact2 as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--calls", type=int, default=20)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--batches", default="1,8")
ap.add_argument("--out", default="/tmp/molmo.json")
args = ap.parse_args()
batches = [int(b) for b in args.batches.split(",")]
rng = np.random.default_rng(0)


def obs():
    return {"images": {c: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for c in ("top", "wrist")},
            "state": np.array(M.DEPLOY["ready_pose"], np.float32) + rng.normal(0, 2, 6).astype(np.float32),
            "task": M.DEPLOY["pick_prompt"]}


def stats(x):
    x = np.asarray(x)
    return {"mean_s": round(float(x.mean()), 4), "p50_s": round(float(np.median(x)), 4),
            "p95_s": round(float(np.percentile(x, 95)), 4), "min_s": round(float(x.min()), 4), "n": int(len(x))}


res = {"batches": {}}
t0 = time.perf_counter()
pol = M.MolmoAct2Client(steps=10, dtype="bfloat16")
res["client_ready"] = pol.info
print("[molmo] ready", pol.info, flush=True)
lat, rt, lat_rtc = [], [], []
prev = None
for i in range(args.warmup + args.calls):
    a = pol.act(obs())
    if i >= args.warmup:
        lat.append(pol.last_latency_s)
        rt.append(pol.last_info["roundtrip_s"])
    prev = a
for i in range(args.warmup + args.calls):
    a = pol.act(obs(), prev_chunk=prev[15:], inference_delay=4)
    if i >= args.warmup:
        lat_rtc.append(pol.last_latency_s)
    prev = a
res["batches"]["1"] = {"client_latency": stats(lat), "client_roundtrip": stats(rt),
                       "client_rtc_latency": stats(lat_rtc), "chunk_shape": list(a.shape),
                       "chunk_finite": bool(np.isfinite(a).all())}
print("[molmo] batch 1", res["batches"]["1"], flush=True)
pol.close()

big = [b for b in batches if b > 1]
if big:
    import torch

    cfg = {"snap": M.DEFAULT_SNAP, "cams": ["wrist", "top"], "steps": 10, "device": "cuda", "noise": "random",
           "seed": 0, "dtype": "bfloat16", "rtc_horizon": 15, "rtc_guidance": 3.0, "text_len": None}
    t = time.perf_counter()
    w = M._Worker(cfg)
    res["inproc_load_s"] = round(time.perf_counter() - t, 1)
    w.pre._extractor.image_keys = ["wrist", "top"]
    for B in [1] + big:
        try:
            ls = []
            for i in range(args.warmup + max(5, args.calls // 2)):
                os_ = [obs() for _ in range(B)]
                batch = {"state": torch.as_tensor(np.stack([o["state"] for o in os_])).to(w.dev),
                         "task": [o["task"] for o in os_],
                         "images": {c: torch.from_numpy(np.stack([o["images"][c] for o in os_])).permute(0, 3, 1, 2)
                                    .to(w.dev) for c in ("wrist", "top")}}
                torch.cuda.synchronize()
                t = time.perf_counter()
                x = w.pre(batch)
                a = w.policy.model.predict_action_chunk(x)
                a = w.post({"action": a})["action"].float().cpu().numpy()
                torch.cuda.synchronize()
                if i >= args.warmup:
                    ls.append(time.perf_counter() - t)
            r = {"inproc_latency": stats(ls), "out_shape": list(a.shape),
                 "per_chunk_s": round(float(np.mean(ls)) / B, 4)}
            if B > 1 and a.shape[0] != B:
                r["warning"] = f"expected leading dim {B}"
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {e}"[:600]}
        res["batches"].setdefault(str(B), {}).update(r)
        print(f"[molmo] batch {B}", r, flush=True)
    res["torch_max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    res["device"] = torch.cuda.get_device_name(0)
res["total_s"] = round(time.perf_counter() - t0, 1)
Path(args.out).write_text(json.dumps(res))
print("[molmo] RESULT", json.dumps(res), flush=True)
