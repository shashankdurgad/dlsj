"""DICE-RL residual on top of the visual MolmoAct2 prior, in TwinEnv (one station, cameras), pick phase only.

Decision every 5 ticks: state = 25-d obs + the prior's pooled prefix latent (fixed random projection to proj_dim),
action = residual conditioned on the plan's next 5 goals (condition_residual_on_base_action). At each prior request
(every 15 ticks, RTC) the K candidates are ranked by min-Q (best-of-K, uniform during the warm-up). Rows go into the
same Replay / Dice as the state-based trainer; gradient steps after every episode at UTD ~1 (one step per new row,
batch 256).

DR (training only): physics (physics.apply: gripper effort, fin stiffness, frictions, brick mass) + the play model's
per-episode draw from episode 0; visual DR (sensor pipeline / lighting / materials at --dr_scale, spotlights at
--spot_scale) ramps linearly 0 -> full over --dr_ramp_eps episodes (all ranks' episodes, --dr ramp).

Held-out evaluation every --eval_every_min (and once at the start = the prior alone, the residual being zero): the
deploy config (K = 1, deterministic residual, fixed deploy noise, slew + controller delay, ready start) on the 8k's
validation episodes with fixed seeds, --eval_eps NOMINAL episodes (no DR) + --eval_dr_eps at full DR, split over all
ranks (multi_image.py). Each round's residual is saved as an exportable checkpoint (deploy meta) in <out>/rounds/ and
as a W&B artifact. Stop: the --minutes cap, a STOP file (--stop_file or <share_dir>/STOP), or the W&B run tag "stop".

    sim/twin/run.sh sim/twin/rl/train_image.py --run img8k --minutes 60 \
        --snap sim/twin/data/models/molmoact2_ft_8k/step008000.ckpt
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim/residual_env"))

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--minutes", type=float, default=90, help="wall-clock safety cap")
ap.add_argument("--snap", default="sim/twin/data/models/molmoact2_ft_8k/step008000.ckpt")
ap.add_argument("--finetune", type=int, default=1, help="1: --snap is the cal2 fine-tune .ckpt (raises otherwise)")
ap.add_argument("--fixed_noise", type=int, default=1, help="1: the deployed export's fixed flow noise per request")
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--proj_dim", type=int, default=128)
ap.add_argument("--latent", type=int, default=1, help="1: the prior's pooled latent in the state; 0: obs only")
ap.add_argument("--rank", type=int, default=0, help="multi-GPU (multi_image.py): this process's rank")
ap.add_argument("--world", type=int, default=1, help="number of ranks (eval shards)")
ap.add_argument("--share_dir", default="", help="shared episode-row exchange dir (all ranks train on all rows)")
ap.add_argument("--start", default="ready", help="episode start (image_task): ready (deploy-like, 12 s) | sampled")
ap.add_argument("--res_xy_m", type=float, default=0.035)
ap.add_argument("--res_z_m", type=float, default=0.02)
ap.add_argument("--veto", type=int, default=0)
ap.add_argument("--top_target", type=int, default=0, help="1: the noisy top-camera target in the obs (+4)")
ap.add_argument("--inject", type=int, default=0, help="1: real-failure injection on the prior (image_task)")
ap.add_argument("--inject_lateral", type=float, default=1.0, help="scale of the injected lateral bias")
ap.add_argument("--max_step", default="4,8",
                help="deployed controller slew per tick on the final command, body deg,gripper units; empty = off")
ap.add_argument("--controller_delay", type=int, default=1, help="1: the play model's deployed +1 tick command delay")
ap.add_argument("--play_dr", type=int, default=1, help="1: draw the play model's parameters per training episode")
ap.add_argument("--phys_dr", type=int, default=1, help="1: physics DR per training episode (from episode 0)")
ap.add_argument("--clutter", default="abl_c", choices=["abl_c", "real"],
                help="distractor counts: abl_c = image_task.CLUTTER (2+ weighted), real = scene_stats.json")
ap.add_argument("--p_near", type=float, default=0.5, help="episodes with 1-2 distractors next to the target")
ap.add_argument("--drift_joint_sd", default="", help="deployment drift (with --phys_dr): per-episode joint "
                "calibration error sd, 6 comma-separated deg (pan,lift,elbow,wflex,roll,grip); empty = off")
ap.add_argument("--drift_top", default="", help="top-camera extrinsic jitter 'pos_m,rot_deg' half-ranges; empty = off")
ap.add_argument("--drift_wrist", default="", help="wrist-mount extrinsic jitter 'pos_m,rot_deg'; empty = off")
ap.add_argument("--detector", default="yolo", choices=["yolo", "yolo_cons", "sim", "none"],
                help="wrist detector in the obs: the fitted real-YOLO model, the residual env's noise model, or none")
ap.add_argument("--bc_weight", type=float, default=2.0)
ap.add_argument("--select_warmup_eps", type=int, default=40)
ap.add_argument("--update_gate", type=int, default=300)
ap.add_argument("--utd", type=float, default=1.0)
ap.add_argument("--expl_noise", type=float, default=0.05)
ap.add_argument("--expert", default="", help="image expert rows (expert.py --latent_snap); RLPD 0.5 -> 0.1")
ap.add_argument("--init", default="", help="warm-start actor + critics from a checkpoint with the same dims")
ap.add_argument("--dr", default="off", choices=["off", "fixed", "ramp"])
ap.add_argument("--dr_scale", type=float, default=0.5, help="full visual DR strength (TwinRandomizer ranges x this)")
ap.add_argument("--spot_scale", type=float, default=1.0)
ap.add_argument("--dr_ramp_eps", default="0,500", help="episodes (all ranks) of the visual DR ramp 0 -> full")
ap.add_argument("--eval_every_min", type=float, default=10.0)
ap.add_argument("--eval_eps", type=int, default=48, help="nominal held-out episodes per round (all ranks)")
ap.add_argument("--eval_dr_eps", type=int, default=24, help="full-DR held-out episodes per round (all ranks)")
ap.add_argument("--eval_seed", type=int, default=4321)
ap.add_argument("--stop_file", default="", help="stop when this file exists (default <share_dir>/STOP)")
ap.add_argument("--gpu_log_s", type=float, default=60.0, help="rank 0: nvidia-smi sample period (W&B)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--wandb_project", default="", help="W&B project (off when empty); key from WANDB_API_KEY")
ap.add_argument("--wandb_entity", default="")
ap.add_argument("--pylibs", default="", help="extra site dir with wandb (appended to sys.path)")
args = ap.parse_args()

randomizer = ramp = None
if args.dr != "off":
    from sim.twin.env.randomize import TwinRandomizer
    from sim.twin.rl.visual_dr import Ramp, scaled_cfg

    r0, r1 = (int(v) for v in args.dr_ramp_eps.split(","))
    ramp = Ramp(args.dr, r0, r1, args.dr_scale, args.spot_scale)
    randomizer = TwinRandomizer(scaled_cfg(0.0))

from sim.twin.rl.image_task import CLUTTER, OBS_DIM, ImageGraspTask  # noqa: E402


def _floats(s: str):
    return [float(v) for v in s.split(",")] if s else None


DRIFT = {"joint_sd_deg": _floats(args.drift_joint_sd), "top": _floats(args.drift_top),
         "wrist": _floats(args.drift_wrist)}

task = ImageGraspTask(args.snap, args.K, seed=args.seed + 99, randomizer=randomizer, detector=args.detector,
                      finetune=bool(args.finetune), fixed_noise=bool(args.fixed_noise),
                      clutter=CLUTTER if args.clutter == "abl_c" else None, p_near=args.p_near, drift=DRIFT,
                      res_xy_m=args.res_xy_m, res_z_m=args.res_z_m, veto=bool(args.veto),
                      top_target=bool(args.top_target), inject=bool(args.inject),
                      inject_lateral=args.inject_lateral,
                      max_step=tuple(float(v) for v in args.max_step.split(",")) if args.max_step else None,
                      controller_delay=bool(args.controller_delay))
OBS = task.obs_dim
START = None if args.start == "sampled" else args.start
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.twin.rl.dice import Dice, DiceCfg, Replay, finalize_returns, mix  # noqa: E402
from sim.twin.rl.grasp_env import OUT  # noqa: E402
from sim.twin.rl.image_task import decide, state_of  # noqa: E402
from sim.twin.rl.priors import LatentProjector  # noqa: E402

out = OUT / args.run
out.mkdir(parents=True, exist_ok=True)
d = task.dev
torch.manual_seed(args.seed)
# latent dim from one request
task.reset("train", 0)
task.request()
latent_dim = int(task.latent.shape[0])
proj = torch.as_tensor(LatentProjector(latent_dim, args.proj_dim).W, device=d) if args.latent else None
s_dim, a_dim = OBS + (args.proj_dim if args.latent else 0), 30
dcfg = DiceCfg(bc_weight=args.bc_weight, k_state=0, qf_warmup=1000,
               expert_ratio=(0.5, 0.1, 10000) if args.expert else (0.0, 0.0, 1))
agent = Dice(s_dim, a_dim, dcfg, d)
if args.init:
    a0, _ = Dice.load(args.init, d)
    agent.actor.load_state_dict(a0.actor.state_dict())
    agent.critic.load_state_dict(a0.critic.state_dict())
    agent.critic_t.load_state_dict(a0.critic_t.state_dict())
buf = Replay(200_000, s_dim, 4, a_dim, 6, d)
exp_buf = None
if args.expert:
    E = torch.load(args.expert, map_location=d)
    for key in ("s", "s2"):  # the expert rows carry obs (25) + projected latent; match this arm's state
        lat = E[key][:, OBS_DIM:]
        ob = E[key][:, :OBS_DIM].clone()
        if args.detector == "none":
            ob[:, 6:10] = 0.0
        if OBS > OBS_DIM:  # no top-camera target logged in the demos: "dropped" (zeros, valid 0)
            ob = torch.cat([ob, torch.zeros(len(ob), OBS - OBS_DIM, device=ob.device)], -1)
        E[key] = torch.cat([ob, lat], -1) if args.latent else ob
    assert E["s"].shape[1] == s_dim, f"expert state dim {E['s'].shape[1]} != {s_dim} (proj_dim / latent mismatch)"
    exp_buf = Replay(len(E["s"]) + 1, s_dim, 4, a_dim, 6, d)
    ix = exp_buf.add(**{k: E[k] for k in ("s", "s2", "A", "A2", "a_sel", "r", "rew", "term", "end", "epi")})
    exp_buf.nxt[ix] = E["nxt"]
    exp_buf.G[ix] = E["G"]
from sim.twin.rl.provenance import provenance  # noqa: E402

PROV = provenance(task.servo_params, getattr(task.env.play, "cfg", None) if task.env.play is not None else None)
(out / "config.json").write_text(json.dumps({"args": vars(args), "dice": asdict(dcfg), "latent_dim": latent_dim,
                                             "provenance": PROV}, indent=1, default=str))
logf = (out / "log.jsonl").open("w")
wb = None
if args.wandb_project:
    if args.pylibs:
        sys.path.append(args.pylibs)
    import wandb

    wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.run, dir=str(out),
                    config={"args": vars(args), "dice": asdict(dcfg), "latent_dim": latent_dim, "path": "image"})
extra = {"obs_dim": OBS, "top_target": bool(args.top_target), "k_state": 0, "chunk": 5, "latent_dim": latent_dim,
         "proj_dim": args.proj_dim,
         "latent": bool(args.latent), "detector": args.detector, "res_xy_m": args.res_xy_m,
         "res_z_m": args.res_z_m, "veto": bool(args.veto), "inject": bool(args.inject),
         "max_step": args.max_step, "controller_delay": bool(args.controller_delay), "play_dr": bool(args.play_dr),
         "phys_dr": bool(args.phys_dr), "fixed_noise": bool(args.fixed_noise), "finetune": bool(args.finetune),
         "clutter": args.clutter, "p_near": args.p_near, "drift": DRIFT, "provenance": PROV,
         "res_scale": [float(v) for v in task.res_scale.tolist()],
         "rate": [float(v) for v in task.rate.tolist()[:5]] + [None],
         "tip_local": [float(v) for v in task.R.tip_local.tolist()],
         "joint_offsets": [float(v) for v in task.env.profile.joint_offsets], "decision_ticks": 5, "fade_ticks": 6}
COLS = ("s", "s2", "A", "A2", "a_sel", "r", "rew", "term", "end")
seen: set = set()
share = Path(args.share_dir) if args.share_dir else None
if share is not None:
    share.mkdir(parents=True, exist_ok=True)


def share_out(ep_rows, n):
    ix = torch.tensor(ep_rows, device=d)
    blob = {k: getattr(buf, k)[ix].cpu() for k in COLS} | {"G": buf.G[ix].cpu()}
    tmp = share / f".r{args.rank}_e{n}.tmp"
    torch.save(blob, tmp)
    tmp.rename(share / f"r{args.rank}_e{n}.pt")


def share_in() -> int:
    """Add the other ranks' finished episodes (rows linked in order, MC returns as computed there)."""
    global imported_eps
    got = 0
    for f in sorted(share.glob("r*_e*.pt")):
        if f.name in seen or f.name.startswith(f"r{args.rank}_"):
            continue
        seen.add(f.name)
        try:
            blob = torch.load(f, map_location=d)
        except Exception:  # noqa: BLE001 (being written)
            seen.discard(f.name)
            continue
        m = len(blob["s"])
        rk, ep = (int(v) for v in f.stem[1:].split("_e"))
        ix = buf.add(**{k: blob[k] for k in COLS}, epi=torch.full((m,), 10 ** 6 * (rk + 1) + ep, device=d))
        if m > 1:
            buf.nxt[ix[:-1]] = ix[1:]
        buf.G[ix] = blob["G"].to(d)
        got += m
        imported_eps += 1
    return got


# ------------------------------------------------------------------ held-out evaluation rounds
EVAL_DIR = share / "eval" if share is not None else out / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
(out / "rounds").mkdir(exist_ok=True)
STOP = Path(args.stop_file) if args.stop_file else (share or out) / "STOP"
eval_agent = Dice(s_dim, a_dim, dcfg, d)
done_rounds: set = set()
pending_rounds: dict = {}  # rank 0: round -> meta, until every rank's shard is in
full_dr = Ramp("fixed", 0, 1, args.dr_scale, args.spot_scale) if randomizer is not None else None


def my_shard(n: int) -> list[int]:
    return [i for i in range(n) if i % args.world == args.rank]


def eval_episode(i: int, use_residual: bool, dr: bool) -> dict:
    """Held-out episode i (fixed seed): deploy config, nominal or full DR; the training rng / K are restored."""
    rng0, k0 = task.rng, task.K
    task.rng, task.K = np.random.default_rng(args.eval_seed + 7919 * i + (1 if dr else 0)), 1
    if dr and full_dr is not None:
        full_dr.apply(randomizer, 0)
    task.reset("test", 10 ** 6 + i, randomize=dr and randomizer is not None, start="ready", play_dr=dr, phys_dr=dr)
    fin = False
    while not fin:
        if task.need_request():
            task.install(task.request()[0])
        if task.t % 5 == 0:
            r = torch.zeros(6, device=d)
            if use_residual and task.pick:
                with torch.no_grad():
                    r = eval_agent.pi(state_of(task, proj)[None], task.plan_feat()[None])[0]
            task.set_residual(r)
        _, fin = task.tick()
    oc = task.outcome()
    res = {"i": i, "episode": task.ep, "dr": dr, "outcome": oc, "closes": task.n_close,
           "first_attempt_success": oc == "success" and task.attempt_at_success <= 1, "black": task.black,
           "n_near": task.n_near, "nn_gap": round(task.nn_gap, 4)}
    task.rng, task.K = rng0, k0
    return res


def start_round(m: int):
    """Rank 0: publish round m's actor (round 0: none = the prior alone) and save it as an exportable checkpoint."""
    blob = {"round": m, "residual": m > 0, "steps": agent.steps, "rows": rows, "t_s": time.perf_counter() - t0,
            "episodes": n_ep_global()}
    if m > 0:
        blob["actor"] = {k: v.cpu() for k, v in agent.actor.state_dict().items()}
        agent.save(out / "rounds" / f"residual_round{m:02d}.pt", extra)
    tmp = EVAL_DIR / f".round{m:02d}.tmp"
    torch.save(blob, tmp)
    tmp.rename(EVAL_DIR / f"round{m:02d}.pt")
    pending_rounds[m] = blob


def run_rounds():
    """Every rank: evaluate its shard of each published round not done yet."""
    for f in sorted(EVAL_DIR.glob("round*.pt")):
        m = int(f.stem[5:])
        if m in done_rounds:
            continue
        blob = torch.load(f, map_location=d, weights_only=False)
        if blob["residual"]:
            eval_agent.actor.load_state_dict(blob["actor"])
        t_e = time.perf_counter()
        res = [eval_episode(i, blob["residual"], False) for i in my_shard(args.eval_eps)]
        res += [eval_episode(i, blob["residual"], True) for i in my_shard(args.eval_dr_eps)]
        (EVAL_DIR / f"round{m:02d}_r{args.rank}.json").write_text(
            json.dumps({"rank": args.rank, "wall_s": round(time.perf_counter() - t_e, 1), "episodes": res}))
        done_rounds.add(m)


def rate(rr, key) -> float:
    return round(sum(key(r) for r in rr) / max(1, len(rr)), 4)


def collect_rounds():
    """Rank 0: merge finished rounds, log them (W&B: eval/* nominal, eval_dr/* full DR, prior baseline lines)."""
    for m in sorted(pending_rounds):
        parts = [EVAL_DIR / f"round{m:02d}_r{k}.json" for k in range(args.world)]
        if not all(p.exists() for p in parts):
            continue
        blob = pending_rounds.pop(m)
        eps_all = [e for p in parts for e in json.loads(p.read_text())["episodes"] if e["black"] == 0]
        summ = {"round": m, "grad_steps": blob["steps"], "episodes_trained": blob["episodes"]}
        for tag, rr in (("eval", [e for e in eps_all if not e["dr"]]), ("eval_dr", [e for e in eps_all if e["dr"]])):
            summ[f"{tag}/n"] = len(rr)
            summ[f"{tag}/first_attempt_success"] = rate(rr, lambda e: e["first_attempt_success"])
            summ[f"{tag}/success"] = rate(rr, lambda e: e["outcome"] == "success")
            summ[f"{tag}/wrong_brick"] = rate(rr, lambda e: e["outcome"] == "wrong")
            summ[f"{tag}/first_attempt_ci95"] = wilson(sum(e["first_attempt_success"] for e in rr), len(rr))
        if m == 0:
            BASE.update({k.replace("eval", "prior"): v for k, v in summ.items() if k.startswith("eval")})
        summ.update(BASE)
        (out / "rounds" / f"round{m:02d}.json").write_text(json.dumps({**summ, "episodes": eps_all}, indent=1))
        print("EVAL_ROUND", json.dumps(summ), flush=True)
        if wb is not None:
            wb.log({k: v for k, v in summ.items() if isinstance(v, (int, float))}, step=rows)
            if m > 0:
                art = wandb.Artifact(f"{args.run}-residual", type="model", metadata=summ)
                art.add_file(str(out / "rounds" / f"residual_round{m:02d}.pt"))
                wb.log_artifact(art, aliases=[f"round{m:02d}"])


def gpu_stats() -> dict:
    """nvidia-smi: utilisation (%) and memory (GiB) of every GPU on the node."""
    import subprocess

    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    o = {}
    for line in q.strip().splitlines():
        try:
            i, u, mem = (float(v) for v in line.split(","))
        except ValueError:
            continue
        o[f"gpu/{int(i)}_util"], o[f"gpu/{int(i)}_mem_gib"] = u, round(mem / 1024, 2)
    return o


def stop_requested() -> bool:
    if STOP.exists():
        return True
    if wb is not None and args.rank == 0:
        try:
            if "stop" in wandb.Api().run(f"{wb.entity}/{wb.project}/{wb.id}").tags:
                STOP.write_text("W&B tag stop\n")
                return True
        except Exception:  # noqa: BLE001 (API hiccup: keep going)
            return False
    return False


imported_eps = 0


def n_ep_global() -> int:
    return n_ep + imported_eps


from sim.twin.rl.rollout import wilson  # noqa: E402

if randomizer is not None:
    from sim.twin.rl.visual_dr import Ramp  # noqa: E402
BASE: dict = {}
t0 = time.perf_counter()
n_ep, rows, recent = 0, 0, []
next_round, t_gpu, gpu = 0, -1e9, {}
if args.rank == 0:
    start_round(0)  # the prior alone (zero residual) on the same held-out set
    next_round = 1
t_next_eval = t0 + args.eval_every_min * 60
ep_times: list = []
while time.perf_counter() - t0 < args.minutes * 60 and not stop_requested():
    run_rounds()
    if args.rank == 0:
        collect_rounds()
        if time.perf_counter() >= t_next_eval:
            start_round(next_round)
            next_round += 1
            t_next_eval = time.perf_counter() + args.eval_every_min * 60
    dr_now = ramp.apply(randomizer, n_ep_global()) if ramp is not None else {}
    task.reset("train", n_ep, randomize=ramp is not None, start=START, play_dr=bool(args.play_dr),
               phys_dr=bool(args.phys_dr))
    ep_rows, prev, pending = [], torch.tensor([-1], device=d), None
    done, t_ep, r_abs = False, time.perf_counter(), []
    while not done:
        if task.need_request():
            C = task.request()
            k = 0
            if len(C) > 1:
                k = int(np.random.randint(len(C))) if n_ep < args.select_warmup_eps else decide(task, agent, proj, C)
            task.install(C[k])
        if task.t % 5 == 0:
            s = state_of(task, proj)
            a = task.plan_feat()
            if pending is not None:  # close the previous decision's row with this state as s2
                ps, pa, pr, prew = pending
                ix = buf.add(s=ps[None], s2=s[None], A=pa.expand(1, 4, a_dim), A2=a.expand(1, 4, a_dim),
                             a_sel=pa[None], r=pr[None], rew=torch.tensor([prew], device=d),
                             term=torch.tensor([False], device=d), end=torch.tensor([False], device=d),
                             epi=torch.tensor([n_ep], device=d))
                buf.link(prev, ix)
                prev = ix
                ep_rows.append(int(ix))
            with torch.no_grad():
                r = agent.pi(s[None], a[None])[0]
            if task.pick:
                r_abs.append(r.abs())
            r = (r + torch.randn_like(r) * args.expl_noise).clamp(-1, 1) if task.pick else torch.zeros_like(r)
            pending = [s, a, r, task.set_residual(r)]
        rew, done = task.tick()
        pending[3] += rew
    ps, pa, pr, prew = pending  # last row: terminal (success / tipped / lost / time limit, see train.py)
    ix = buf.add(s=ps[None], s2=ps[None], A=pa.expand(1, 4, a_dim), A2=pa.expand(1, 4, a_dim), a_sel=pa[None],
                 r=pr[None], rew=torch.tensor([prew], device=d), term=torch.tensor([True], device=d),
                 end=torch.tensor([True], device=d), epi=torch.tensor([n_ep], device=d))
    buf.link(prev, ix)
    ep_rows.append(int(ix))
    finalize_returns(buf, torch.tensor([ep_rows], device=d), torch.tensor([len(ep_rows)], device=d), dcfg.gamma)
    if args.share_dir:  # publish this episode's rows, import the other ranks' new episodes
        share_out(ep_rows, n_ep)
        imported = share_in()
    else:
        imported = 0
    rows += len(ep_rows)
    recent.append((task.outcome(), task.flags["centred"]))
    upd = {}
    if buf.n >= args.update_gate:
        for _ in range(int(round(args.utd * (len(ep_rows) + imported)))):
            ne = int(round(dcfg.batch * agent.expert_ratio())) if exp_buf is not None else 0
            b_on = buf.sample(dcfg.batch - ne, dcfg.gamma, dcfg.n_step)
            upd = agent.update(mix(b_on, exp_buf.sample(ne, dcfg.gamma, dcfg.n_step) if ne else None))
    n_ep += 1
    ep_times.append(time.perf_counter())
    ep_times = [t for t in ep_times if t > ep_times[-1] - 300]
    R = recent[-100:]
    ra = torch.stack(r_abs).mean(0).tolist() if r_abs else [0.0] * 6
    if args.rank == 0 and time.perf_counter() - t_gpu > args.gpu_log_s:
        gpu, t_gpu = gpu_stats(), time.perf_counter()
    row = {"t_s": round(time.perf_counter() - t0, 1), "episodes": n_ep, "rows": rows, "grad_steps": agent.steps,
           "episode_s": round(time.perf_counter() - t_ep, 2), "outcome": task.outcome(),
           "success_last100": round(float(np.mean([x[0] == "success" for x in R])), 3),
           "centred_last100": round(float(np.mean([x[1] for x in R])), 3), **dr_now,
           "episodes_global": n_ep_global(), "episodes_per_min_rank": round(len(ep_times) / 5.0, 2),
           "episodes_per_min_all": round(len(ep_times) / 5.0 * args.world, 2),
           "residual_abs_mean": round(float(np.mean(ra)), 4),
           **{f"residual_abs/{n}": round(float(v), 4) for n, v in zip(("x", "y", "z", "wflex", "wroll", "grip"), ra,
                                                                         strict=True)},
           **{k2: round(v, 5) for k2, v in upd.items()}, **gpu}
    logf.write(json.dumps(row) + "\n")
    logf.flush()
    if wb is not None:
        wb.log({k2: v for k2, v in row.items() if isinstance(v, (int, float))}, step=rows)
    gpu = {}
    if n_ep % 10 == 0:
        print("TRAIN_IMG", json.dumps(row), flush=True)
        agent.save(out / "residual.pt", extra)
agent.save(out / "residual.pt", extra)
# final round: rank 0 publishes it, every rank evaluates its shard (waits up to 20 min for it), rank 0 merges
if args.rank == 0:
    start_round(next_round)
    (EVAL_DIR / "final").write_text(str(next_round))
t_w = time.perf_counter()
while time.perf_counter() - t_w < 1200:
    run_rounds()
    fin_f = EVAL_DIR / "final"
    if fin_f.exists() and int(fin_f.read_text()) in done_rounds:
        if args.rank != 0:
            break
        collect_rounds()
        if not pending_rounds:
            break
    time.sleep(2)
if wb is not None:
    wb.finish()
print("TRAIN_IMG_DONE", json.dumps({"episodes": n_ep, "rows": rows, "grad_steps": agent.steps}), flush=True)
task.close()
