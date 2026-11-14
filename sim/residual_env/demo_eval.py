"""Evaluate the scripted VLA-failure stand-in base alone vs base + learned residual on fixed seeds.

    python demo_eval.py --episodes 256 --seed 5000 [--residual out/demo/ckpt_04.pt,out/demo/ckpt_final.pt] [--out f.json]

Base alone is always evaluated first; held-out seeds (not the trainer's eval seed) give an unbiased comparison.
"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--residual", default="")
ap.add_argument("--scale_errors", type=float, default=1.0)
ap.add_argument("--out", default="")
args = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import launch  # noqa: E402

app = launch(cameras=False)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from app import make  # noqa: E402
from bases import ScriptedFailBase  # noqa: E402
from demo_lib import demo_cfg, rollout, summarise  # noqa: E402

base = ScriptedFailBase(scale_errors=args.scale_errors)
sim, scene, layout, env = make(demo_cfg(args.episodes), base)
base.env = env
from sac import SAC  # noqa: E402

results = {}
for res_path in [""] + [r for r in args.residual.split(",") if r]:  # base alone first, then each residual
    policy = None
    if res_path:
        agent = SAC(env.obs_dim, env.act_dim, env.device)
        agent.load(res_path)
        policy = lambda o, a=agent: a.act(o, deterministic=True)  # noqa: E731
    m = rollout(env, base, policy, seed=args.seed)
    s = summarise(m)
    s.update({"residual": res_path or None, "scale_errors": args.scale_errors, "seed": args.seed})
    print("EVAL", json.dumps(s), flush=True)
    results[res_path or "base"] = {"summary": s, "per_episode": {k: np.asarray(v).tolist() for k, v in m.items()}}
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
app.close()
