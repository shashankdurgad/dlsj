"""TwinEnv throughput: one Isaac process, one station, 3 cameras at 640x480 (the TwinEnvCfg defaults).

    isaaclab.sh -p sim/twin/modal/bench_env.py --gpu 0 --steps 300 [--barrier DIR --n-procs N] --out res.json

--gpu sets CUDA_VISIBLE_DEVICES before Isaac starts: Isaac Lab picks the renderer by CUDA index
(/renderer/multiGpu/activeCudaGpus=0), so physics and RTX rendering both land on that GPU. With --barrier, every
process waits until all N have built their env, so the timed loops run concurrently (startup is timed separately).

Timed (each after --warmup ticks): `step(render=True)` (the RL tick: 4 PhysX substeps + 3 RTX cameras + colour
correction + GPU->CPU copies), `step(render=False)` (physics + servo model only), `render()` alone, and `reset()`.
The arm follows a slow sinusoid around its start pose so the cameras see motion; image change vs the first timed
frame is reported per camera (render follows physics) and proc 0 writes start / moved frames as PNGs.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--warmup", type=int, default=30)
ap.add_argument("--resets", type=int, default=3)
ap.add_argument("--barrier", default="")
ap.add_argument("--n-procs", type=int, default=1)
ap.add_argument("--out", default="/tmp/bench.json")
ap.add_argument("--png-dir", default="", help="write start / moved frames here (one process only)")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
t_start = time.perf_counter()
from sim.twin.env.twin_env import TwinEnv, TwinEnvCfg  # noqa: E402

env = TwinEnv(TwinEnvCfg())
import numpy as np  # noqa: E402
import torch  # noqa: E402

t_init = time.perf_counter() - t_start
obs = env.reset(seed=args.gpu * 100 + os.getpid() % 100)
s0 = obs["state"].copy()
res = {"gpu": args.gpu, "pid": os.getpid(), "device_name": torch.cuda.get_device_name(0), "init_s": round(t_init, 1),
       "cams": sorted(obs["images"]), "shape": list(next(iter(obs["images"].values())).shape)}
print("[bench] env ready", res, flush=True)

if args.barrier:
    b = Path(args.barrier)
    b.mkdir(parents=True, exist_ok=True)
    (b / f"ready_{os.getpid()}").touch()
    t = time.monotonic()
    while len(list(b.glob("ready_*"))) < args.n_procs and time.monotonic() - t < 900:
        time.sleep(0.5)
    res["barrier_wait_s"] = round(time.monotonic() - t, 1)

amp = np.array([12.0, 10.0, -10.0, 10.0, 15.0, 20.0], np.float32)


def action(k):
    a = s0 + amp * np.sin(2 * np.pi * k / 120.0)
    a[5] = np.clip(a[5], 0.0, 100.0)
    return a


k = 0
for _ in range(args.warmup):
    env.step(action(k))
    k += 1

first = {c: im.astype(np.float32) for c, im in env.render().items()}
dt = []
moved, best = None, -1.0
t0 = time.perf_counter()
for i in range(args.steps):
    t = time.perf_counter()
    obs, rew, term, trunc, info = env.step(action(k))
    k += 1
    dt.append(time.perf_counter() - t)
    if i % 10 == 0:
        d = float(np.abs(obs["images"]["top"].astype(np.float32) - first["top"]).mean())
        if d > best:
            best, moved = d, {c: im.copy() for c, im in obs["images"].items()}
wall = time.perf_counter() - t0
res["steps_per_s_render"] = round(args.steps / wall, 2)
res["step_ms_p50"] = round(1e3 * float(np.percentile(dt, 50)), 1)
res["step_ms_p95"] = round(1e3 * float(np.percentile(dt, 95)), 1)
res["img_change_vs_first"] = {c: round(float(np.abs(moved[c].astype(np.float32) - first[c]).mean()), 2)
                              for c in moved}
res["img_mean"] = {c: round(float(moved[c].mean()), 1) for c in moved}

t0 = time.perf_counter()
for _ in range(args.steps):
    env.step(action(k), render=False)
    k += 1
res["steps_per_s_norender"] = round(args.steps / (time.perf_counter() - t0), 2)

n_r = max(20, args.steps // 5)
t0 = time.perf_counter()
for _ in range(n_r):
    env.render()
res["render_ms"] = round(1e3 * (time.perf_counter() - t0) / n_r, 1)

t0 = time.perf_counter()
for j in range(args.resets):
    env.reset(seed=1000 + j)
res["reset_s"] = round((time.perf_counter() - t0) / max(1, args.resets), 3)
res["torch_max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

if args.png_dir:
    import imageio.v3 as iio

    for c in first:
        iio.imwrite(f"{args.png_dir}/bench_{c}_start.png", first[c].astype(np.uint8))
        iio.imwrite(f"{args.png_dir}/bench_{c}_moved.png", moved[c])
Path(args.out).write_text(json.dumps(res))
print("[bench] RESULT", json.dumps(res), flush=True)
env.close()
