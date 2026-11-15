"""DICE-RL residual training in the batched grasp env (state-based, ReplayErrorPrior base).

    NAME=dlsj-twin-rl setsid nohup sim/twin/run.sh sim/twin/rl/train.py --run dice_s0 --minutes 25 \
        > sim/twin/out/rl/dice_s0.log 2>&1 &

Outputs in sim/twin/out/rl/<run>/: config.json, log.jsonl (one row per log interval), curves.png, residual.pt
(actor + critics), eval.json (held-out test episodes: base / residual-only / best-of-K / detector ablation).
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sim.twin.rl import grasp_env as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--minutes", type=float, default=25.0)
ap.add_argument("--num_envs", type=int, default=512)
ap.add_argument("--K", type=int, default=16, help="candidates per decision at collection (max_q_min)")
ap.add_argument("--select_warmup", type=int, default=100000, help="rows with a uniform candidate choice")
ap.add_argument("--update_gate", type=int, default=10000, help="online rows before updates start")
ap.add_argument("--updates_per_step", type=int, default=1, help="updates (x grad_steps) per vector decision step")
ap.add_argument("--expl_noise", type=float, default=0.05, help="Gaussian noise on the residual at collection")
ap.add_argument("--expert", default="", help="expert rows .pt (sim/twin/rl/expert.py); empty = no RLPD mixing")
ap.add_argument("--dr", default="off", choices=["off", "fixed", "ramp"],
                help="physics DR (sim/twin/physics/apply.py ranges): off | fixed | ramp (0 -> 1 between --dr_ramp)")
ap.add_argument("--dr_ramp", default="3000,9000", help="gradient steps where the DR magnitude ramps 0 -> 1")
ap.add_argument("--eval_eps", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out_root", default="", help="output root (default sim/twin/out/rl; can be a Modal Volume)")
ap.add_argument("--resume", type=int, default=0, help="1: load <out>/residual.pt (nets, optimisers, step) and continue "
                "(the replay is not checkpointed: it refills; RLPD expert rows reload)")
ap.add_argument("--ckpt_every", type=int, default=200, help="vector steps between checkpoints")
ap.add_argument("--wandb_project", default="", help="W&B project (off when empty)")
ap.add_argument("--wandb_entity", default="")
ap.add_argument("--wandb_key_fifo", default="", help="read WANDB_API_KEY from this FIFO (never from disk / argv)")
ap.add_argument("--pylibs", default=str(Path(__file__).resolve().parents[1] / "out/rl/pylibs"),
                help="extra site dir for packages missing from the image (wandb)")
ap.add_argument("--cap", type=int, default=3_000_000)
ap.add_argument("--detector", default="yolo_mix", choices=list(G.DETECTORS),
                help="wrist-detector model for training (grasp_env.attach_detector); evaluation uses nominal yolo")
ap.add_argument("--preset", default="v3", choices=list(G.PRESETS), help="grasp_env.PRESETS; "
                "explicit --play / --distractors / ... flags given on the command line still win")
ap.add_argument("--trunc_bootstrap", type=int, default=0,
                help="1: bootstrap through time-limit truncation. Default 0 (terminal): with t/T in the observation "
                     "the state at t = T never starts a row, so bootstrapping there is pure extrapolation "
                     "(Q drifted to -1.6 against MC returns of -0.08 in the smoke run)")
for f in fields(G.GraspCfg):
    if f.name != "num_envs" and isinstance(f.default, (int, float, str)) and not isinstance(f.default, bool):
        ap.add_argument("--" + f.name, type=type(f.default), default=f.default)
    elif isinstance(f.default, bool):
        ap.add_argument("--" + f.name, type=int, default=int(f.default))
from sim.twin.rl.dice import DiceCfg  # noqa: E402

for f in fields(DiceCfg):
    if isinstance(f.default, (int, float)) and not isinstance(f.default, bool):
        ap.add_argument("--d_" + f.name, type=type(f.default), default=f.default)
args = ap.parse_args()
app = G.launch()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.twin.rl import rollout as RO  # noqa: E402
from sim.twin.rl.dice import Dice, Replay, finalize_returns, mix  # noqa: E402

torch.manual_seed(args.seed)
out = (Path(args.out_root) if args.out_root else G.OUT) / args.run
out.mkdir(parents=True, exist_ok=True)
gkw = {f.name: (bool(getattr(args, f.name)) if isinstance(f.default, bool) else getattr(args, f.name))
       for f in fields(G.GraspCfg) if hasattr(args, f.name) and f.name != "num_envs"}
given = {a.lstrip("-").split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
gkw = {**gkw, **{k: v for k, v in G.PRESETS[args.preset].items() if k not in given}}
gcfg = G.GraspCfg(num_envs=args.num_envs, **gkw)
dkw = {f.name: getattr(args, "d_" + f.name) for f in fields(DiceCfg) if hasattr(args, "d_" + f.name)}
dcfg = DiceCfg(**dkw)
env = G.make(gcfg)
G.attach_detector(env, args.detector)
(out / "config.json").write_text(json.dumps({"args": vars(args), "grasp": asdict(gcfg), "dice": asdict(dcfg),
                                             "provenance": env.deploy_meta()["provenance"]}, indent=1, default=str))
n, d = env.n, env.device
K, KS = args.K, dcfg.k_state
s_dim, a_dim = env.OBS_DIM + KS * env.h * 6, env.h * 6
agent = Dice(s_dim, a_dim, dcfg, d)
SAVE_META = {**env.deploy_meta(), "k_state": KS, "chunk": env.h, "detector": args.detector, "preset": args.preset}
if args.resume and (out / "residual.pt").exists():
    agent, _ = Dice.load(out / "residual.pt", d)
    print(f"resumed from {out / 'residual.pt'} at grad step {agent.steps}", flush=True)
wb = None
if args.wandb_project:
    import hashlib
    import os
    import subprocess

    if args.wandb_key_fifo:
        with open(args.wandb_key_fifo) as fh:
            os.environ["WANDB_API_KEY"] = fh.readline().strip()
        Path(args.wandb_key_fifo).unlink(missing_ok=True)
    if Path(args.pylibs).is_dir():
        sys.path.append(args.pylibs)  # after the image packages: they win on conflicts
    import wandb

    recipe = {"args": {k: v for k, v in vars(args).items() if k != "wandb_key_fifo"}, "grasp": asdict(gcfg),
              "dice": asdict(dcfg)}
    git = subprocess.run(["git", "-C", str(G.REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    fp = hashlib.sha256(json.dumps(recipe, sort_keys=True, default=str).encode()).hexdigest()[:12]
    wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.run, resume="allow",
                    dir=str(out),
                    config={**recipe, "git_commit": git or "unknown", "recipe_fingerprint": fp,
                            "physics_dr": args.dr, "dr_ramp": args.dr_ramp})
buf = Replay(args.cap, s_dim, KS, a_dim, 6, d)
exp_buf = None
if args.expert:
    E = torch.load(args.expert, map_location=d)
    exp_buf = Replay(len(E["s"]) + 1, s_dim, KS, a_dim, 6, d)
    idx = exp_buf.add(**{k: E[k] for k in ("s", "s2", "A", "A2", "a_sel", "r", "rew", "term", "end", "epi")})
    exp_buf.nxt[idx] = E["nxt"]
    exp_buf.G[idx] = E["G"]
    print(f"expert rows: {len(idx)} from {args.expert}", flush=True)

phys_dr = None
if args.dr != "off":
    from sim.twin.physics import apply as phys

    phys_dr = phys
    nominal = phys.nominal_physics()
    ranges = phys._p()["dr"]
r0, r1 = (int(v) for v in args.dr_ramp.split(","))


def apply_dr(ids):
    """Physics DR for the envs in ids: one draw per reset batch, magnitude scaled toward nominal by the ramp."""
    if phys_dr is None or len(ids) == 0:
        return
    mag = 1.0 if args.dr == "fixed" else float(np.clip((agent.steps - r0) / max(1, r1 - r0), 0, 1))
    rng = np.random.default_rng(int(torch.randint(1 << 30, (1,))))
    draw = {}
    for k, (lo, hi) in ranges.items():
        v = rng.uniform(lo, hi)
        draw[k] = nominal[k] + mag * (v - nominal[k])
    phys_dr.apply_physics_dr(env.scene, draw, env_ids=ids.cpu())


gen = torch.Generator(device=d).manual_seed(args.seed)
ids_all = torch.arange(n, device=d)
env.reset_envs(ids_all, env.sample_params(n, gen, "train"))
apply_dr(ids_all)
obs = env.obs()
C = env.candidates(K)
F = env.cand_feat(C)
prev_row = torch.full((n,), -1, dtype=torch.long, device=d)
ep_rows = torch.full((n, env.max_ticks // env.h + 2), -1, dtype=torch.long, device=d)
ep_len = torch.zeros(n, dtype=torch.long, device=d)
ep_id = torch.arange(n, device=d)
next_ep = n
ep_ret = torch.zeros(n, device=d)
recent = []
logf = (out / "log.jsonl").open("a" if args.resume else "w")
t0 = time.perf_counter()
steps = rows = 0
t_env = t_upd = 0.0
upd = {}
while time.perf_counter() - t0 < args.minutes * 60:
    ta = time.perf_counter()
    s = RO.state_of(obs, F, KS)
    with torch.no_grad():
        k, r, q = agent.select(s, F, "max_q_min")
        if rows < args.select_warmup:
            k = torch.randint(K, (n,), device=d)
            sK = s.unsqueeze(1).expand(n, K, s_dim)[ids_all, k]
            r = agent.pi(sK, F[ids_all, k])
        r = (r + torch.randn_like(r) * args.expl_noise).clamp(-1, 1)
    a_sel = F[ids_all, k]
    rew, term, trunc = env.step_chunk(k, r)
    obs2 = env.obs()
    C2 = env.candidates(K)
    F2 = env.cand_feat(C2)
    s2 = RO.state_of(obs2, F2, KS)
    end = term | trunc
    tstore = term if args.trunc_bootstrap else end
    row = buf.add(s=s, s2=s2, A=F[:, :KS], A2=F2[:, :KS], a_sel=a_sel, r=r, rew=rew, term=tstore, end=end, epi=ep_id)
    buf.link(prev_row, row)
    prev_row = row
    ep_rows[ids_all, ep_len.clamp(max=ep_rows.shape[1] - 1)] = row
    ep_len += 1
    ep_ret += rew
    rows += n
    steps += 1
    if bool(end.any()):
        di = torch.nonzero(end).flatten()
        finalize_returns(buf, ep_rows[di], ep_len[di], dcfg.gamma)
        oc = env.outcome()[di]
        for j, i in enumerate(di.tolist()):
            recent.append((float(ep_ret[i]), int(oc[j]), bool(env.centred[i])))
        recent = recent[-2000:]
        env.reset_envs(di, env.sample_params(len(di), gen, "train"))
        apply_dr(di)
        ep_rows[di] = -1
        ep_len[di] = 0
        ep_ret[di] = 0
        prev_row[di] = -1
        ep_id[di] = next_ep + torch.arange(len(di), device=d)
        next_ep += len(di)
        obs_n = env.obs()
        Cn = env.candidates(K)
        obs2 = torch.where(end[:, None], obs_n, obs2)
        C2 = torch.where(end[:, None, None, None], Cn, C2)
        env.cands = C2
        F2 = env.cand_feat(C2)
    obs, C, F = obs2, C2, F2
    torch.cuda.synchronize()
    tb = time.perf_counter()
    t_env += tb - ta
    if buf.n >= args.update_gate:
        for _ in range(args.updates_per_step * dcfg.grad_steps):
            ne = int(round(dcfg.batch * agent.expert_ratio())) if exp_buf is not None else 0
            b_on = buf.sample(dcfg.batch - ne, dcfg.gamma, dcfg.n_step)
            b_ex = exp_buf.sample(ne, dcfg.gamma, dcfg.n_step) if ne else None
            upd = agent.update(mix(b_on, b_ex))
        torch.cuda.synchronize()
    t_upd += time.perf_counter() - tb
    if steps % 20 == 0:
        R = recent[-1000:]
        row_log = {"t_s": round(time.perf_counter() - t0, 1), "steps": steps, "rows": rows, "grad_steps": agent.steps,
                   "env_ticks": rows * env.h, "env_ticks_per_s": round(rows * env.h / max(1e-6, t_env), 1),
                   "wall_ticks_per_s": round(rows * env.h / (time.perf_counter() - t0), 1),
                   "frac_time_env": round(t_env / max(1e-6, t_env + t_upd), 3), "episodes": next_ep - n,
                   "success": round(float(np.mean([x[1] == 0 for x in R])), 4) if R else None,
                   "centred": round(float(np.mean([x[2] for x in R])), 4) if R else None,
                   "ret": round(float(np.mean([x[0] for x in R])), 3) if R else None,
                   "pushed": round(float(np.mean([x[1] == 2 for x in R])), 4) if R else None,
                   "tipped": round(float(np.mean([x[1] == 4 for x in R])), 4) if R else None,
                   "expert_ratio": round(agent.expert_ratio(), 3) if exp_buf is not None else 0.0,
                   "selecting": rows >= args.select_warmup,
                   **{k2: round(v, 5) for k2, v in upd.items()}}
        logf.write(json.dumps(row_log) + "\n")
        logf.flush()
        print("TRAIN", json.dumps(row_log), flush=True)
        if wb is not None:
            wb.log({k2: v for k2, v in row_log.items() if isinstance(v, (int, float))}, step=rows)
    if steps % args.ckpt_every == 0:
        agent.save(out / "residual.pt", SAVE_META)
agent.save(out / "residual.pt", SAVE_META)
logf.close()
print("TRAIN_DONE", json.dumps({"steps": steps, "rows": rows, "grad_steps": agent.steps,
                                "minutes": round((time.perf_counter() - t0) / 60, 2)}), flush=True)

# ------------------------------------------------------------------ held-out evaluation (test episodes, fixed seed)
if phys_dr is not None:
    phys_dr.apply_physics_dr(env.scene, nominal)
rows_eval = []
for label, pol, kw, abl in (("base (with errors)", "base", {}, False),
                            ("residual only, best-of-1", "dice", {"K": 1}, False),
                            ("best-of-4 selection only (r=0)", "select", {"K": 4}, False),
                            ("residual + best-of-4", "dice", {"K": 4}, False),
                            ("residual + best-of-16", "dice", {"K": 16}, False),
                            ("residual best-of-1, detector ablated", "dice", {"K": 1}, True),
                            ("residual + best-of-4, detector ablated", "dice", {"K": 4}, True)):
    env.g.det_ablate = abl or args.detector == "none"
    res = RO.evaluate(env, pol, n_eps=args.eval_eps, agent=agent, k_state=KS, **kw)
    rows_eval.append(RO.summarize(res, label))
env.g.det_ablate = args.detector == "none"
RO.dump(rows_eval, out / "eval.json")
if wb is not None:
    cols = ["policy", "n", "pick_success", "ci_lo", "ci_hi", "centred", "missed", "pushed", "slipped", "tipped"]
    tab = wandb.Table(columns=cols, data=[[r["policy"], r["n"], r["pick_success"], *r["pick_success_ci95"],
                                           r["centred"], *(r["failures"][k] for k in cols[6:])] for r in rows_eval])
    wb.log({"eval/table": tab, **{f"eval/{r['policy']}/pick_success": r["pick_success"] for r in rows_eval}})
    wb.finish()
try:
    from sim.twin.rl.plots import curves

    curves(out)
except Exception as e:  # noqa: BLE001
    print("curves failed:", e, flush=True)
print("ALL_DONE", flush=True)
app.close()
