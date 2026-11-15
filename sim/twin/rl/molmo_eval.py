"""Image-based pick episodes with the MolmoAct2 prior in TwinEnv: the prior-alone baseline that DICE-RL has to raise
(and the end-to-end smoke test of MolmoAct2Prior), or a trained image residual (--ckpt). Protocol: image_task.py.

    sim/twin/run.sh sim/twin/rl/molmo_eval.py --episodes 2 --K 4                                   # smoke
    sim/twin/run.sh sim/twin/rl/molmo_eval.py --episodes 200 --K 1 --out molmo8k_base \
        --snap sim/twin/data/models/molmoact2_ft_8k/step008000.ckpt                                # prior alone
    ... --ckpt sim/twin/out/rl/<run>/residual.pt --K 4                                             # + residual
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim/residual_env"))

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=2)
ap.add_argument("--K", type=int, default=4, help="candidates per request (residual run: best-of-K by min-Q)")
ap.add_argument("--snap", default="sim/twin/data/models/molmoact2_ft_8k/step008000.ckpt",
                help="Studio fine-tune .ckpt (--finetune 1) or an HF snapshot dir / repo id (--finetune 0, zero-shot)")
ap.add_argument("--finetune", type=int, default=1, help="1: --snap is the cal2 fine-tune .ckpt (raises otherwise)")
ap.add_argument("--fixed_noise", type=int, default=1, help="1: the deployed export's fixed flow noise per request")
ap.add_argument("--clutter", default="abl_c", choices=["abl_c", "real"])
ap.add_argument("--instruction", default=None)
ap.add_argument("--crop_top", default="auto")
ap.add_argument("--ckpt", default="", help="trained image residual (train_image.py); empty = prior alone")
ap.add_argument("--delay", type=int, default=10)
ap.add_argument("--horizon_s", type=float, default=3.0, help="unused when start types set the horizon")
ap.add_argument("--start", default="", help="fixed start type (near / approach / frame0 / perturbed / recovery)")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--pool", default="test")
ap.add_argument("--out", default="molmo_eval")
ap.add_argument("--sheets", type=int, default=4, help="episodes with a saved wrist-frame strip")
ap.add_argument("--inject", type=int, default=0, help="1: real-failure injection on the prior (image_task)")
ap.add_argument("--max_step", default="4,8",
                help="deployed controller slew per tick on the final command, body deg,gripper units; empty = off")
ap.add_argument("--controller_delay", type=int, default=1, help="1: the play model's deployed +1 tick command delay")
args = ap.parse_args()

import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.twin.rl.image_task import CLUTTER, ImageGraspTask  # noqa: E402

meta = torch.load(args.ckpt, map_location="cpu", weights_only=False) if args.ckpt else {}
task = ImageGraspTask(args.snap, args.K, args.delay, args.horizon_s, args.seed, args.instruction, args.crop_top,
                      detector=meta.get("detector", "sim"), res_xy_m=meta.get("res_xy_m", 0.025),
                      res_z_m=meta.get("res_z_m", 0.025), veto=bool(meta.get("veto", False)),
                      top_target=bool(meta.get("top_target", False)),
                      inject=bool(args.inject or meta.get("inject", False)),
                      max_step=tuple(float(v) for v in args.max_step.split(",")) if args.max_step else None,
                      controller_delay=bool(args.controller_delay), finetune=bool(args.finetune),
                      fixed_noise=bool(args.fixed_noise), clutter=CLUTTER if args.clutter == "abl_c" else None)
for key, val in (("max_step", args.max_step), ("controller_delay", bool(args.controller_delay))):
    if meta and meta.get(key, "unset") != val:  # eval runs the deploy controller; say when training did not
        print(f"WARNING: --{key} {val!r} differs from the checkpoint's training value {meta.get(key, 'unset')!r}",
              flush=True)
from sim.twin.rl.grasp_env import OUT  # noqa: E402
from sim.twin.rl.image_task import decide, state_of  # noqa: E402
from sim.twin.rl.rollout import wilson  # noqa: E402

out = OUT / args.out
out.mkdir(parents=True, exist_ok=True)
agent = proj = None
if args.ckpt:
    from sim.twin.rl.dice import Dice
    from sim.twin.rl.priors import LatentProjector

    agent, ck = Dice.load(args.ckpt, task.dev)
    if ck.get("latent", True):
        proj = torch.as_tensor(LatentProjector(ck["latent_dim"], ck["proj_dim"]).W, device=task.dev)
print("prior ready", json.dumps(task.prior.client.info), flush=True)

results = []
for n_ep in range(args.episodes):
    task.reset(args.pool, n_ep, start=args.start or None)
    frames, t_wall = [], time.perf_counter()
    rew_sum, done = 0.0, False
    while not done:
        if task.need_request():
            C = task.request()
            k = 0
            if agent is not None and len(C) > 1:
                k = int(decide(task, agent, proj, C))
            task.install(C[k])
            if n_ep < args.sheets and len(frames) < 6:
                frames.append(task.imgs["wrist"])
        if task.t % 5 == 0:
            r = torch.zeros(6, device=task.dev)
            if agent is not None and task.pick:
                with torch.no_grad():
                    r = agent.pi(state_of(task, proj)[None], task.plan_feat()[None])[0]
            rew_sum += task.set_residual(r)
        rew, done = task.tick()
        rew_sum += rew
    oc = task.outcome()
    plan = getattr(task.env, "scene_plan", None) or {}
    n_bricks = 1 + len(plan.get("distractors", []) or [])
    r = {"n": n_ep, "episode": task.ep, "task": task.task, "start": task.start, "n_bricks": n_bricks,
         "outcome": oc, "centred": task.flags["centred"],
         "ticks": task.t, "K": args.K, "return": round(rew_sum, 3), "closes": task.n_close,
         "nn_gap": round(task.nn_gap, 4), "vetoes": task.n_veto, "injection": task.inj,
         "first_close_err_mm": task.first_close_err_mm,
         "first_close_empty": task.first_close_empty if task.first_close_empty is not None else
         (None if task.n_close == 0 else not task.flags["ever_grasp"]),
         "first_attempt_success": oc == "success" and task.attempt_at_success <= 1, "black_frames": task.black,
         "prior_latency_s_median": round(float(np.median(task.lat)), 4), "prior_calls": len(task.lat),
         "wall_s": round(time.perf_counter() - t_wall, 2),
         "mean_abs_delta": (task.sum_abs_o / max(1, task.n_pick)).cpu().numpy().round(4).tolist()}
    results.append(r)
    print("MOLMO", json.dumps(r), flush=True)
    if frames:
        import cv2

        cv2.imwrite(str(out / f"ep{n_ep:03d}_{task.ep}_{oc}_wrist.jpg"),
                    cv2.cvtColor(np.concatenate([cv2.resize(f, (320, 240)) for f in frames], 1), cv2.COLOR_RGB2BGR))
valid = [r for r in results if r["black_frames"] == 0]  # episodes whose prior saw a real image every request
if len(valid) < len(results):
    print(f"WARNING: {len(results) - len(valid)} episodes had black renders and are excluded", flush=True)
results_all, results = results, valid
oc = [r["outcome"] for r in results]
n, ns = len(oc), oc.count("success")
mad = np.mean([r["mean_abs_delta"] for r in results], 0)
not_centred = sum(1 for r in results if r["outcome"] == "success" and not r["centred"])
summary = {"prior": args.snap or "allenai/MolmoAct2-SO100_101 (zero-shot)", "residual": args.ckpt or None,
           "episodes": n, "K": args.K, "crops": task.crops, "pick_success": round(ns / n, 4),
           "pick_success_ci95": wilson(ns, n), "centred": round(sum(r["centred"] for r in results) / n, 4),
           "failures": {k: round(oc.count(k) / n, 4) for k in ("missed", "pushed", "slipped", "tipped", "wrong")},
           "not_centred_given_success": round(not_centred / max(1, ns), 4),
           "mean_abs_delta": {"tip_xyz_mm": (mad[:3] * 1e3).round(2).tolist(), "wrist_deg": mad[3:5].round(2).tolist(),
                              "gripper_units": round(float(mad[5]), 2)},
           "prior_latency_s_median": round(float(np.median([r["prior_latency_s_median"] for r in results])), 4),
           "episode_wall_s_median": round(float(np.median([r["wall_s"] for r in results])), 2),
           "protocol": {"pool": args.pool, "horizon_s": args.horizon_s, "delay_ticks": args.delay, "seed": args.seed,
                        "dr": "off", "clamp": 10}}
summary["excluded_black_render"] = len(results_all) - len(results)
summary["max_step"], summary["controller_delay"], summary["play_dr"] = args.max_step, bool(args.controller_delay), False


def _rate(rows):
    k = sum(r["outcome"] == "success" for r in rows)
    return {"n": len(rows), "success": round(k / max(1, len(rows)), 3), "ci95": wilson(k, len(rows))}


summary["wrong_brick"] = round(oc.count("wrong") / max(1, n), 4)
summary["by_neighbour_gap"] = {}
for lab, lo, hi in (("touching", -1, 0.002), ("<2cm", 0.002, 0.02), ("2-5cm", 0.02, 0.05), (">5cm", 0.05, 9)):
    rr = [r for r in results if lo <= r["nn_gap"] < hi]
    if rr:
        summary["by_neighbour_gap"][lab] = {"n": len(rr),
                                            "success": round(sum(r["outcome"] == "success" for r in rr) / len(rr), 3),
                                            "wrong": round(sum(r["outcome"] == "wrong" for r in rr) / len(rr), 3)}
fc = [r for r in results if r["first_close_empty"] is not None]
errs = sorted(r["first_close_err_mm"] for r in results if r["first_close_err_mm"] is not None)
summary["first_close"] = {"n_with_close": len(fc), "empty_frac": round(sum(r["first_close_empty"] for r in fc) /
                                                                       max(1, len(fc)), 3),
                          "lateral_err_mm_median": errs[len(errs) // 2] if errs else None,
                          "lateral_err_mm_p25_p75": [errs[len(errs) // 4], errs[3 * len(errs) // 4]] if errs else None,
                          "mean_closes": round(sum(r["closes"] for r in results) / max(1, n), 2)}
nf = sum(1 for r in results if r["first_attempt_success"])
summary["first_attempt_success"] = round(nf / max(1, n), 4)
summary["first_attempt_success_ci95"] = wilson(nf, n)
summary["by_start"] = {s: _rate([r for r in results if r["start"] == s]) for s in sorted({r["start"] for r in results})}
summary["by_bricks"] = {lab: _rate([r for r in results if lo <= r["n_bricks"] <= hi])
                        for lab, lo, hi in (("1", 1, 1), ("2-3", 2, 3), ("4-5", 4, 5), ("6-8", 6, 8))
                        if any(lo <= r["n_bricks"] <= hi for r in results)}
(out / "molmo_eval.json").write_text(json.dumps({"summary": summary, "episodes": results_all}, indent=1))
print("MOLMO_SUMMARY", json.dumps(summary), flush=True)
task.close()
