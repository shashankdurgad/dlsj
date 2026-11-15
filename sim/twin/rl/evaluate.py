"""Fixed-seed evaluation over the same held-out episodes: baselines and trained residuals.

    sim/twin/run.sh sim/twin/rl/evaluate.py --out baselines                     # the three baselines (+ clean replay)
    sim/twin/run.sh sim/twin/rl/evaluate.py --out dice_s0_eval --ckpt sim/twin/out/rl/dice_s0/residual.pt
    ... --pool train   (training episodes), --n_eps 512, --seed 1234
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sim.twin.rl import grasp_env as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default="")
ap.add_argument("--num_envs", type=int, default=512)
ap.add_argument("--n_eps", type=int, default=512)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--pool", default="test")
ap.add_argument("--clamp", type=float, default=None, help="LeRobot clamp override (default: saved cfg / 10)")
ap.add_argument("--joint_offsets_json", default="")
ap.add_argument("--extra", type=int, default=1, help="also: clean replay without brick perturbation")
ap.add_argument("--preset", default=None, choices=list(G.PRESETS),
                help="grasp_env.PRESETS; default: the checkpoint's own saved GraspCfg (baselines without --ckpt: v3). "
                     "With --ckpt an explicit preset overrides the saved cfg and every mismatch is printed")
ap.add_argument("--max_step_deg", type=float, default=None,
                help="deployed controller slew override, deg / tick (default: saved cfg / preset; 0 = off)")
ap.add_argument("--veto", type=int, default=0, help="1: the deploy close-veto (needs a yolo* detector)")
ap.add_argument("--detector", default="yolo", choices=list(G.DETECTORS), help="wrist-detector model (attach_detector)")
args = ap.parse_args()
app = G.launch()


import dataclasses  # noqa: E402

import torch  # noqa: E402

from sim.twin.rl import rollout as RO  # noqa: E402

saved = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("grasp_cfg") if args.ckpt else None
if saved is not None:  # the run's own environment config (what the residual was trained on)
    names = {f.name for f in dataclasses.fields(G.GraspCfg)}
    base = {k: v for k, v in saved.items() if k in names and k not in ("num_envs", "device")}
    if args.preset is not None:
        over = G.PRESETS[args.preset]
        for k, v in over.items():
            if base.get(k) != v:
                print(f"WARNING: --preset {args.preset} overrides the checkpoint's GraspCfg.{k}: {base.get(k)!r} -> "
                      f"{v!r}", flush=True)
        base.update(over)
elif args.ckpt:
    print("WARNING: the checkpoint has no saved GraspCfg (trained before it was saved): using --preset "
          f"{args.preset or 'v3'}; the eval env may differ from the training env", flush=True)
    base = dict(G.PRESETS[args.preset or "v3"])
else:
    base = dict(G.PRESETS[args.preset or "v3"])
base["veto"] = bool(args.veto)
if args.clamp is not None:
    base["max_relative_target"] = args.clamp
if args.max_step_deg is not None:
    if args.ckpt and base.get("max_step_deg", 0.0) != args.max_step_deg:
        print(f"WARNING: --max_step_deg {args.max_step_deg} differs from the training env's "
              f"{base.get('max_step_deg', 0.0)}", flush=True)
    base["max_step_deg"] = args.max_step_deg
if args.joint_offsets_json:
    base["joint_offsets_json"] = args.joint_offsets_json
cfg = G.GraspCfg(num_envs=args.num_envs, **base)
env = G.make(cfg)
G.attach_detector(env, args.detector)
out = G.OUT / args.out
out.mkdir(parents=True, exist_ok=True)
rows = []
kw = dict(n_eps=args.n_eps, seed=args.seed, pool=args.pool)
if not args.ckpt:
    rows.append(RO.summarize(RO.evaluate(env, "base", errors=False, **kw), "base, no errors"))
    rows.append(RO.summarize(RO.evaluate(env, "base", errors=True, **kw), "base, with errors"))
    rows.append(RO.summarize(RO.evaluate(env, "scripted", errors=True, scripted=RO.Scripted(env), **kw),
                             "scripted align-then-close, with errors"))
    if args.extra:
        bp, by = env.g.brick_xy, env.g.brick_yaw_deg
        env.g.brick_xy, env.g.brick_yaw_deg = 0.0, 0.0
        rows.append(RO.summarize(RO.evaluate(env, "base", errors=False, **kw),
                                 "base, no errors, brick at the recorded fingertip"))
        env.g.brick_xy, env.g.brick_yaw_deg = bp, by
else:
    from sim.twin.rl.dice import Dice

    agent, ck = Dice.load(args.ckpt, env.device)
    ks = ck.get("k_state", 4)
    rows.append(RO.summarize(RO.evaluate(env, "base", errors=True, **kw), "base, with errors"))
    det_mode = args.detector
    for label, pol, K, abl in (("residual only, best-of-1", "dice", 1, False),
                               ("best-of-4 selection only", "select", 4, False),
                               ("residual + best-of-4", "dice", 4, False), ("residual + best-of-16", "dice", 16, False),
                               ("residual best-of-1, detector ablated", "dice", 1, True),
                               ("residual + best-of-4, detector ablated", "dice", 4, True)):
        env.g.det_ablate = abl or det_mode == "none"
        rows.append(RO.summarize(RO.evaluate(env, pol, errors=True, agent=agent, K=K, k_state=ks, **kw), label))
    env.g.det_ablate = det_mode == "none"
RO.dump(rows, out / "eval.json")
app.close()
