"""Paired held-out evaluation of image residuals (and the prior alone) in the deploy configuration: the in-loop
evaluation of train_image.py (eval_episode) for several arms on the SAME episodes / seeds.

Per held-out index i and condition, every arm runs the episode with task.rng = default_rng(eval_seed + 7919 i + c):
    nominal (c = 0): no DR at all = train_image's nominal round eval (same seeds: i < eval_eps reproduces a round);
    deploy  (c = 2): deployment-like: joint calibration drift N(0, --drift_joint_sd) + physics DR + play draw,
                     visual DR off.
K = 1, deterministic residual, fixed deploy noise, slew 4 / 8, controller delay, ready start, the 8k validation pool;
the task settings come from the first residual checkpoint's meta. Arms: "prior" (zero residual) + name=ckpt pairs.
Episodes are interleaved (i outer, condition, arm inner) and appended to <out>/r<rank>.jsonl, so a cut run stays
paired. --merge: Wilson CIs per arm / condition + the paired difference (arm - prior) with a bootstrap CI.

    python sim/twin/rl/heldout_eval.py --arms final=<ckpt>,round3=<ckpt> --n 100 --rank 0 --world 16 --out heldout_c
    python sim/twin/rl/heldout_eval.py --merge --out heldout_c
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim/residual_env"))

ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="", help="name=ckpt,... (the prior alone is always arm 'prior')")
ap.add_argument("--conds", default="nominal,deploy")
ap.add_argument("--n", type=int, default=100, help="held-out indices per condition (all ranks)")
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--world", type=int, default=1)
ap.add_argument("--eval_seed", type=int, default=4321)
ap.add_argument("--drift_joint_sd", default="3,3,3,1,0,0", help="deploy condition: joint calibration error sd, deg")
ap.add_argument("--snap", default="sim/twin/data/models/molmoact2_ft_8k/step008000.ckpt")
ap.add_argument("--out", default="heldout_eval")
ap.add_argument("--merge", action="store_true")
args = ap.parse_args()
OUTD = REPO / "sim/twin/out/rl" / args.out
OUTD.mkdir(parents=True, exist_ok=True)
SEED_OFF = {"nominal": 0, "deploy": 2}


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(c - h, 4), round(c + h, 4)]


METRICS = {"first_attempt_success": lambda e: bool(e["first_attempt_success"]),
           "success": lambda e: e["outcome"] == "success", "wrong_brick": lambda e: e["outcome"] == "wrong"}


def merge():
    import numpy as np

    rows = [json.loads(line) for f in sorted(OUTD.glob("r*.jsonl")) for line in f.read_text().splitlines() if line]
    by = {}
    for r in rows:
        by.setdefault((r["cond"], r["i"]), {})[r["arm"]] = r
    arms = sorted({r["arm"] for r in rows}, key=lambda a: (a != "prior", a))
    rng = np.random.default_rng(0)
    summ = {"arms": arms, "conds": {}}
    for cond in sorted({k[0] for k in by}):
        # paired set: indices where every arm finished without black renders
        keys = sorted(k for k, v in by.items() if k[0] == cond and all(a in v and v[a]["black"] == 0 for a in arms))
        cs = {"n_paired": len(keys), "arms": {}, "paired_vs_prior": {}}
        for a in arms:
            cs["arms"][a] = {}
            for m, f in METRICS.items():
                k = sum(f(by[x][a]) for x in keys)
                cs["arms"][a][m] = {"rate": round(k / max(1, len(keys)), 4), "k": k, "ci95": wilson(k, len(keys))}
        for a in arms:
            if a == "prior":
                continue
            cs["paired_vs_prior"][a] = {}
            for m, f in METRICS.items():
                d = np.array([float(f(by[x][a])) - float(f(by[x]["prior"])) for x in keys])
                if not len(d):
                    continue
                boot = rng.choice(d, (10000, len(d))).mean(1)
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
                cs["paired_vs_prior"][a][m] = {
                    "diff": round(float(d.mean()), 4), "boot_ci95": [round(float(np.quantile(boot, q)), 4)
                                                                     for q in (0.025, 0.975)],
                    "normal_ci95": [round(float(d.mean() - 1.96 * se), 4), round(float(d.mean() + 1.96 * se), 4)],
                    "arm_only": int((d > 0).sum()), "prior_only": int((d < 0).sum())}  # McNemar discordant pairs
        summ["conds"][cond] = cs
    excluded = {c: sum(1 for k, v in by.items() if k[0] == c) - summ["conds"][c]["n_paired"] for c in summ["conds"]}
    summ["excluded_incomplete_or_black"] = excluded
    (OUTD / "summary.json").write_text(json.dumps(summ, indent=1))
    print("HELDOUT", json.dumps(summ), flush=True)


if args.merge:
    merge()
    sys.exit(0)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.twin.rl.dice import Dice  # noqa: E402

arms = [("prior", None)] + [tuple(a.split("=", 1)) for a in args.arms.split(",") if a]
meta = {}
for name, path in arms[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("latent"):
        raise SystemExit(f"{path}: latent residuals are not supported here (the deploy code refuses them)")
    meta = meta or {k: v for k, v in ck.items() if not isinstance(v, dict) or k in ("provenance", "drift")}
    print("ARM", name, path, json.dumps({k: ck.get(k) for k in ("steps", "detector", "top_target", "max_step",
                                                                 "controller_delay", "fixed_noise", "clutter")}))
from sim.twin.rl.image_task import CLUTTER, ImageGraspTask, state_of  # noqa: E402

ms = meta.get("max_step", "4,8")
task = ImageGraspTask(args.snap, 1, seed=args.eval_seed, detector=meta.get("detector", "yolo"),
                      res_xy_m=meta.get("res_xy_m", 0.035), res_z_m=meta.get("res_z_m", 0.02),
                      veto=bool(meta.get("veto", False)), top_target=bool(meta.get("top_target", False)),
                      max_step=tuple(float(v) for v in ms.split(",")) if ms else None,
                      controller_delay=bool(meta.get("controller_delay", True)), finetune=True,
                      fixed_noise=bool(meta.get("fixed_noise", True)),
                      clutter=CLUTTER if meta.get("clutter", "abl_c") == "abl_c" else None,
                      p_near=float(meta.get("p_near", 0.5)),
                      drift={"joint_sd_deg": [float(v) for v in args.drift_joint_sd.split(",")]})
d = task.dev
agents = {name: Dice.load(path, d)[0] for name, path in arms[1:]}
log = (OUTD / f"r{args.rank}.jsonl").open("a")
done = set()
for f in OUTD.glob(f"r{args.rank}.jsonl"):
    for line in f.read_text().splitlines():
        if line:
            r = json.loads(line)
            done.add((r["i"], r["cond"], r["arm"]))


def episode(i: int, cond: str, agent) -> dict:
    """train_image.eval_episode with the deploy condition added (phys + play + drift, no visual DR)."""
    dr = cond == "deploy"
    task.rng, task.K = np.random.default_rng(args.eval_seed + 7919 * i + SEED_OFF[cond]), 1
    task.reset("test", 10 ** 6 + i, randomize=False, start="ready", play_dr=dr, phys_dr=dr)
    t_w, fin = time.perf_counter(), False
    while not fin:
        if task.need_request():
            task.install(task.request()[0])
        if task.t % 5 == 0:
            r = torch.zeros(6, device=d)
            if agent is not None and task.pick:
                with torch.no_grad():
                    r = agent.pi(state_of(task, None)[None], task.plan_feat()[None])[0]
            task.set_residual(r)
        _, fin = task.tick()
    oc = task.outcome()
    return {"i": i, "cond": cond, "episode": task.ep, "outcome": oc, "closes": task.n_close,
            "first_attempt_success": oc == "success" and task.attempt_at_success <= 1, "black": task.black,
            "first_close_err_mm": task.first_close_err_mm, "n_near": task.n_near, "nn_gap": round(task.nn_gap, 4),
            "cal_err": [round(float(v), 3) for v in task.cal_err.tolist()], "ticks": task.t,
            "wall_s": round(time.perf_counter() - t_w, 1)}


conds = args.conds.split(",")
for i in range(args.rank, args.n, args.world):
    for cond in conds:
        for name, _ in arms:
            if (i, cond, name) in done:
                continue
            r = {**episode(i, cond, agents.get(name)), "arm": name}
            log.write(json.dumps(r) + "\n")
            log.flush()
            print("HELDOUT_EP", json.dumps(r), flush=True)
task.close()
