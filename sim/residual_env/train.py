"""Residual SAC training entry point.

    # smoke run (replay base = recorded teleop motions, no images, no VLA):
    python train.py --base replay --num_envs 32 --minutes 5
    # with the frozen SmolVLA in the loop (start vla_batch_worker.py first, images rendered):
    python train.py --base smolvla --num_envs 8 --latency 0.074 --minutes 60
"""
import argparse
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", choices=["replay", "hold", "smolvla"], default="replay")
ap.add_argument("--num_envs", type=int, default=32)
ap.add_argument("--minutes", type=float, default=5.0)
ap.add_argument("--latency", type=float, default=0.074)
ap.add_argument("--episode_s", type=float, default=12.0)
ap.add_argument("--warmup_steps", type=int, default=2000, help="transitions before updates start")
ap.add_argument("--updates_per_step", type=int, default=1)
ap.add_argument("--data", default=str(Path.home() / "residual_env/data/duplo_teleop_3phase"))
ap.add_argument("--ipc", default=str(Path.home() / "residual_env/ipc"))
ap.add_argument("--out", default=str(Path.home() / "residual_env/out/train"))
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import launch  # noqa: E402

app = launch(cameras=args.base == "smolvla")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app import make  # noqa: E402
from bases import HoldBase, IPCVLABase, ReplayBase, load_episodes  # noqa: E402
from env import EnvCfg  # noqa: E402
from sac import SAC  # noqa: E402

torch.manual_seed(args.seed)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if args.base == "replay":
    eps = load_episodes(args.data)
    eps = {e: d for e, d in eps.items() if not d["excluded"] and (d["grasp_frame"] or 0) > 5}
    base = ReplayBase(eps, seed=args.seed)
elif args.base == "hold":
    base = HoldBase()
else:
    base = IPCVLABase(args.ipc)
cfg = EnvCfg(num_envs=args.num_envs, cameras=args.base == "smolvla", base_latency_s=args.latency,
             episode_s=args.episode_s)
sim, scene, layout, env = make(cfg, base)
agent = SAC(env.obs_dim, env.act_dim, env.device)
obs = env.reset()
ep_ret = torch.zeros(env.n, device=env.device)
ep_flags = {k: torch.zeros(env.n, dtype=torch.bool, device=env.device) for k in ("grasped", "lifted", "placed")}
log, finished = [], []
t0 = time.perf_counter()
steps = 0
while time.perf_counter() - t0 < args.minutes * 60:
    if agent.size() < args.warmup_steps:
        act = torch.rand(env.n, env.act_dim, device=env.device) * 0.6 - 0.3  # small random residuals
    else:
        act = agent.act(obs)
    obs2, rew, done, info = env.step(act)
    agent.add(obs, act, rew, obs2, done & (env.t < env.max_ticks))  # time-limit ends are not terminal
    ep_ret += rew
    for k in ep_flags:
        ep_flags[k] |= info[k]
    if bool(done.any()):
        ids = torch.nonzero(done).flatten()
        for i in ids.tolist():
            finished.append({"ret": float(ep_ret[i]), **{k: bool(ep_flags[k][i]) for k in ep_flags}})
        ep_ret[ids] = 0
        for k in ep_flags:
            ep_flags[k][ids] = False
        obs2 = env.reset(ids)
    obs = obs2
    steps += 1
    upd = {}
    if agent.size() >= args.warmup_steps:
        for _ in range(args.updates_per_step):
            upd = agent.update()
    if steps % 200 == 0:
        recent = finished[-50:]
        row = {"t_s": round(time.perf_counter() - t0, 1), "steps": steps, "transitions": agent.size(),
               "env_ticks_per_s": round(steps * env.n / (time.perf_counter() - t0), 1),
               "episodes": len(finished),
               "ret_recent": round(float(np.mean([f["ret"] for f in recent])), 2) if recent else None,
               "grasp_rate": round(float(np.mean([f["grasped"] for f in recent])), 2) if recent else None,
               "lift_rate": round(float(np.mean([f["lifted"] for f in recent])), 2) if recent else None,
               **{k: round(v, 4) for k, v in upd.items()}}
        log.append(row)
        print("TRAIN", json.dumps(row), flush=True)
agent.save(out / "residual_sac.pt")
(out / "train_log.json").write_text(json.dumps(log, indent=1))
print("TRAIN_DONE", json.dumps({"steps": steps, "episodes": len(finished)}), flush=True)
app.close()
