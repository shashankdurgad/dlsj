"""Multi-GPU image-path DICE-RL in one container: N = gpus x procs_per_gpu ranks (one Isaac TwinEnv + one
MolmoAct2 + one learner each; rank i on GPU i % gpus) exchange finished episodes through a shared directory, so every
learner trains on all ranks' data; rank 0's residual is the result (W&B on rank 0). The held-out evaluation rounds
(train_image.py: start, every --eval_every_min, final; nominal + full DR) are split over all ranks in-loop.
Stop early: touch <share>/STOP (printed at the start) or tag the W&B run "stop"; --minutes is the safety cap.

    python sim/twin/rl/multi_image.py --gpus 8 --procs_per_gpu 2 --run abl_c --minutes 90 -- --detector yolo ...
    python sim/twin/rl/multi_image.py --gpus 8 --run abl_a --eval_only 1 --eval_eps 96     # prior alone (molmo_eval)
Prints RL_ABLATION {...} (last merged evaluation round) at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

ap = argparse.ArgumentParser()
ap.add_argument("--gpus", type=int, default=8)
ap.add_argument("--procs_per_gpu", type=int, default=1, help="ranks per GPU (fill the GPU: Isaac + MolmoAct2 each)")
ap.add_argument("--run", required=True)
ap.add_argument("--minutes", type=float, default=20)
ap.add_argument("--eval_eps", type=int, default=96)
ap.add_argument("--eval_seed", type=int, default=4321)
ap.add_argument("--eval_only", type=int, default=0)
ap.add_argument("--stagger_s", type=float, default=25.0, help="spawn spacing (8 x 22 GB checkpoint loads)")
ap.add_argument("--snap", default="sim/twin/data/models/molmoact2_ft_8k/step008000.ckpt")
ap.add_argument("--wandb", default="", help="extra args for rank 0 only (W&B)")
ap.add_argument("--eval_args", default="", help="extra molmo_eval.py args (e.g. --inject 1)")
ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- train_image.py arguments")
args = ap.parse_args()
extra = [a for a in args.rest if a != "--"]
ISAAC = "/workspace/isaaclab/isaaclab.sh"
RUNNER = [ISAAC, "-p"] if os.path.exists(ISAAC) else [sys.executable]
OUT = REPO / "sim/twin/out/rl"


def spawn(cmds: list[list[str]], tag: str) -> list[int]:
    """Run one command per GPU (staggered), wait for all; returns exit codes. Logs -> OUT/<run>_<tag>_r<i>.log."""
    procs = []
    for i, c in enumerate(cmds):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % args.gpus), PYTHONPATH=str(REPO))
        log = open(OUT / f"{args.run}_{tag}_r{i}.log", "w")  # noqa: SIM115 (closed below)
        procs.append((subprocess.Popen(RUNNER + c, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO), log))
        if i < len(cmds) - 1:
            time.sleep(args.stagger_s)
    codes = []
    for p, log in procs:
        codes.append(p.wait())
        log.close()
    return codes


OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
ckpt = None
if not args.eval_only:
    share = Path("/tmp") / f"share_{args.run}"
    shutil.rmtree(share, ignore_errors=True)
    world = args.gpus * args.procs_per_gpu
    print(f"STOP: touch {share / 'STOP'} (or tag the W&B run 'stop')", flush=True)
    cmds = []
    for i in range(world):
        c = ["sim/twin/rl/train_image.py", "--run", f"{args.run}_r{i}", "--rank", str(i), "--world", str(world),
             "--seed", str(i), "--share_dir", str(share), "--minutes", str(args.minutes), "--snap", args.snap, *extra]
        if i == 0 and args.wandb:
            c += args.wandb.split()
        cmds.append(c)
    codes = spawn(cmds, "train")
    print("TRAIN_CODES", codes, flush=True)
    eps = [len(list(share.glob(f"r{i}_e*.pt"))) for i in range(world)]
    info = {"per_rank": eps, "total": sum(eps), "wall_min": round((time.time() - t0) / 60, 1)}
    print("TRAIN_EPISODES", json.dumps(info), flush=True)
    rounds = sorted((OUT / f"{args.run}_r0" / "rounds").glob("round*.json"))
    last = json.loads(rounds[-1].read_text()) if rounds else {}
    last.pop("episodes", None)
    summ = {"run": args.run, "train_codes": codes, **info, "rounds": len(rounds), "last_round": last,
            "residual_rounds": str(OUT / f"{args.run}_r0" / "rounds"),
            "deploy_config": "K=1, deterministic residual, fixed noise, slew 4/8, controller delay, nominal + DR"}
    (OUT / f"{args.run}_summary.json").write_text(json.dumps(summ, indent=1))
    print("RL_ABLATION", json.dumps(summ), flush=True)
    sys.exit(0)
t1 = time.time()
per = -(-args.eval_eps // args.gpus)
cmds = []
for i in range(args.gpus):
    c = ["sim/twin/rl/molmo_eval.py", "--episodes", str(per), "--K", "1", "--seed", str(args.eval_seed + i),
         "--start", "ready", "--snap", args.snap, "--out", f"{args.run}_eval_r{i}", "--sheets", "1"]
    if ckpt is not None:
        c += ["--ckpt", str(ckpt)]
    c += args.eval_args.split()
    cmds.append(c)
codes = spawn(cmds, "eval")
rows = []
for i in range(args.gpus):
    f = OUT / f"{args.run}_eval_r{i}" / "molmo_eval.json"
    if f.exists():
        rows += [r for r in json.loads(f.read_text())["episodes"] if r.get("black_frames", 0) == 0]
from sim.twin.rl.rollout import wilson  # noqa: E402

n = len(rows)
ns = sum(r["outcome"] == "success" for r in rows)
nf = sum(bool(r.get("first_attempt_success")) for r in rows)
summ = {"run": args.run, "eval_codes": codes, "episodes": n, "pick_success": round(ns / max(1, n), 4),
        "pick_success_ci95": wilson(ns, n), "first_attempt_success": round(nf / max(1, n), 4),
        "first_attempt_ci95": wilson(nf, n),
        "failures": {k: round(sum(r["outcome"] == k for r in rows) / max(1, n), 4)
                     for k in ("missed", "pushed", "slipped", "tipped", "wrong")},
        "mean_closes": round(sum(r.get("closes", 0) for r in rows) / max(1, n), 2),
        "first_close_empty": round(sum(bool(r.get("first_close_empty")) for r in rows) /
                                   max(1, sum(r.get("first_close_empty") is not None for r in rows)), 3),
        "first_close_err_mm_median": (sorted(r["first_close_err_mm"] for r in rows if r.get("first_close_err_mm")
                                             is not None) or [None])[sum(r.get("first_close_err_mm") is not None
                                                                         for r in rows) // 2 if rows else 0],
        "wrong_brick": round(sum(r["outcome"] == "wrong" for r in rows) / max(1, n), 4),
        "eval_wall_min": round((time.time() - t1) / 60, 1), "total_wall_min": round((time.time() - t0) / 60, 1),
        "residual": str(ckpt) if ckpt else None, "deploy_config": "K=1, deterministic residual, no critic"}
(OUT / f"{args.run}_summary.json").write_text(json.dumps(summ, indent=1))
print("RL_ABLATION", json.dumps(summ), flush=True)
