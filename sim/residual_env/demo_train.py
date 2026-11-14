"""Residual-RL demo trainer: SAC learns bounded corrections on top of a FROZEN base policy, for the demo task
"grasp the brick, lift it >= 5 cm, hold 1 s" (demo_lib.py), with periodic fixed-seed evaluation.

Bases:
  scripted : ScriptedFailBase, a scripted stand-in reproducing the VLAs' observed miss (fallback demo; no images)
  molmo    : the real MolmoAct2 (zero-shot or our LoRA) via molmo_batch_worker.py (start it first; images rendered)
  replay   : recorded teleop motions (plumbing tests)

    python demo_train.py --base scripted --num_envs 64 --minutes 75 --out ~/residual_env/out/demo_scripted
    python demo_train.py --base molmo --num_envs 8 --minutes 120 --ipc ~/residual_env/ipc_molmo --latency 0.31

Outputs (--out): curve.json (eval success vs transitions/time), eval_base.json, eval_<k>.json, ckpt_<k>.pt,
ckpt_final.pt, train_log.json. The evaluation uses the same seeds every time (base alone once, then base +
deterministic residual at every eval), so the curve is comparable across checkpoints.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", choices=["scripted", "molmo", "replay"], default="scripted")
ap.add_argument("--num_envs", type=int, default=64)
ap.add_argument("--minutes", type=float, default=60.0)
ap.add_argument("--latency", type=float, default=0.074, help="base latency in SIM time (s); MolmoAct2 INT8+RTC on the XPS ~0.31")
ap.add_argument("--ipc", default=str(Path.home() / "residual_env/ipc_molmo"))
ap.add_argument("--eval_every_min", type=float, default=10.0)
ap.add_argument("--eval_reps", type=int, default=1, help="eval episodes = num_envs * eval_reps (fixed seeds)")
ap.add_argument("--eval_seed", type=int, default=1000)
ap.add_argument("--warmup_steps", type=int, default=3000)
ap.add_argument("--updates_per_step", type=int, default=1)
ap.add_argument("--scale_errors", type=float, default=1.0, help="scripted base error scale")
ap.add_argument("--prompts", default="", help="grasp|present|place strings (default: the 3-stage training strings)")
ap.add_argument("--out", default=str(Path.home() / "residual_env/out/demo"))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="/home/joe-spark-2/desmond/datasets/duplo-cal2-456-3stage-20261114/duplo_teleop_cal2_rgb_3stage")
args = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import launch  # noqa: E402

app = launch(cameras=args.base == "molmo")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app import make  # noqa: E402
from bases import IPCVLABase, ReplayBase, ScriptedFailBase, load_episodes  # noqa: E402
from demo_lib import SuccessTracker, demo_cfg, rollout, summarise  # noqa: E402
from sac import SAC  # noqa: E402

torch.manual_seed(args.seed)
out = Path(args.out).expanduser()
out.mkdir(parents=True, exist_ok=True)
if args.base == "scripted":
    base = ScriptedFailBase(scale_errors=args.scale_errors, seed=args.seed)
elif args.base == "replay":
    eps = load_episodes(args.data)
    base = ReplayBase({e: d for e, d in eps.items() if not d["excluded"] and (d["grasp_frame"] or 0) > 5}, seed=args.seed)
else:
    base = IPCVLABase(args.ipc)
cfg = demo_cfg(args.num_envs, cameras=args.base == "molmo", latency=args.latency)
if args.prompts:
    g, p, pl = args.prompts.split("|")
    cfg.instructions = {"grasp": g, "present": p, "place": pl}
sim, scene, layout, env = make(cfg, base)
if hasattr(base, "env"):
    base.env = env
agent = SAC(env.obs_dim, env.act_dim, env.device)
meta = {"base": args.base, "num_envs": env.n, "obs_dim": env.obs_dim, "act_dim": env.act_dim,
        "residual_joints": list(cfg.residual_joints), "residual_scale": list(cfg.residual_scale),
        "latency_s": args.latency, "calib": cfg.calib, "instructions": cfg.instructions,
        "eval_seed": args.eval_seed, "eval_episodes": env.n * args.eval_reps}
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print("META", json.dumps(meta), flush=True)


def evaluate(policy, tag: str) -> dict:
    ms = [rollout(env, base, policy, seed=args.eval_seed + r) for r in range(args.eval_reps)]
    m = {k: np.concatenate([x[k] for x in ms]) for k in ms[0]}
    s = summarise(m)
    (out / f"eval_{tag}.json").write_text(json.dumps({"summary": s, "per_episode": {k: v.tolist() for k, v in m.items()}},
                                                     indent=1))
    print("EVAL", tag, json.dumps(s), flush=True)
    return s


t_start = time.perf_counter()
curve = []
s0 = evaluate(None, "base")
curve.append({"tag": "base", "transitions": 0, "t_min": 0.0, **s0})
(out / "curve.json").write_text(json.dumps(curve, indent=1))

obs = env.reset()
tr = SuccessTracker(env.n, env.device)
ep_ret = torch.zeros(env.n, device=env.device)
ep_succ = torch.zeros(env.n, dtype=torch.bool, device=env.device)
finished, log = [], []
steps, k_eval = 0, 0
t0 = time.perf_counter()
next_eval = args.eval_every_min * 60
while time.perf_counter() - t0 < args.minutes * 60:
    act = (torch.rand(env.n, env.act_dim, device=env.device) * 0.6 - 0.3) if agent.size() < args.warmup_steps \
        else agent.act(obs)
    obs2, rew, done, info = env.step(act)
    s = tr.update(info)
    bonus = (s & ~ep_succ).float() * 10.0  # one-off success bonus for the demo task (lift >= 5 cm, held 1 s)
    rew = rew + bonus
    ep_succ |= s
    agent.add(obs, act, rew, obs2, done & (env.t < env.max_ticks))
    ep_ret += rew
    if bool(done.any()):
        ids = torch.nonzero(done).flatten()
        for i in ids.tolist():
            finished.append({"ret": float(ep_ret[i]), "success": bool(ep_succ[i])})
        ep_ret[ids] = 0
        ep_succ[ids] = False
        tr.reset(ids)
        obs2 = env.reset(ids)
    obs = obs2
    steps += 1
    upd = {}
    if agent.size() >= args.warmup_steps:
        for _ in range(args.updates_per_step):
            upd = agent.update()
    el = time.perf_counter() - t0
    if steps % 100 == 0:
        recent = finished[-100:]
        row = {"t_min": round(el / 60, 2), "steps": steps, "transitions": steps * env.n, "buffer": agent.size(),
               "env_ticks_per_s": round(steps * env.n / el, 1), "episodes": len(finished),
               "episodes_per_h": round(len(finished) / el * 3600, 1),
               "train_success_recent": round(float(np.mean([f["success"] for f in recent])), 3) if recent else None,
               "ret_recent": round(float(np.mean([f["ret"] for f in recent])), 2) if recent else None,
               **{k: round(v, 4) for k, v in upd.items()}}
        log.append(row)
        print("TRAIN", json.dumps(row), flush=True)
    if el >= next_eval:
        k_eval += 1
        agent.save(out / f"ckpt_{k_eval:02d}.pt")
        se = evaluate(lambda o: agent.act(o, deterministic=True), f"{k_eval:02d}")
        curve.append({"tag": f"{k_eval:02d}", "transitions": steps * env.n, "t_min": round(el / 60, 2), **se})
        (out / "curve.json").write_text(json.dumps(curve, indent=1))
        (out / "train_log.json").write_text(json.dumps(log, indent=1))
        obs = env.reset()
        tr.reset(torch.arange(env.n, device=env.device))
        ep_ret.zero_()
        ep_succ.zero_()
        next_eval = (time.perf_counter() - t0) + args.eval_every_min * 60
agent.save(out / "ckpt_final.pt")
sf = evaluate(lambda o: agent.act(o, deterministic=True), "final")
curve.append({"tag": "final", "transitions": steps * env.n, "t_min": round((time.perf_counter() - t0) / 60, 2), **sf})
(out / "curve.json").write_text(json.dumps(curve, indent=1))
(out / "train_log.json").write_text(json.dumps(log, indent=1))
print("DEMO_DONE", json.dumps({"base": s0, "final": sf, "steps": steps, "episodes": len(finished),
                               "wall_min": round((time.perf_counter() - t_start) / 60, 1)}), flush=True)
if hasattr(base, "ipc"):
    (Path(base.ipc) / "done").write_text("1")
app.close()
