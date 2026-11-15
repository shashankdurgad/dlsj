"""Actor / learner split for DICE-RL on one machine (GB10 dev: 1 GPU, small N; Modal: N GPUs). Local IPC only.

    launcher  spawns 1 learner + N actor processes (+ optional prior server), sets CUDA_VISIBLE_DEVICES per process
    actor     an Isaac process with a batched GraspEnv; best-of-K selection + residual from the latest published
              weights; streams decision rows to the learner (AF_UNIX socket, pickled numpy batches)
    learner   owns the GPU replay (n-step links, MC returns), the DICE-RL updates, checkpoints, W&B hook; publishes
              weights to <out>/weights.pt (atomic rename) that actors reload when it changes
    prior     (image path) sim/twin/rl/prior_server.py: ONE MolmoAct2 worker shared by all actors, batched requests

Everything is config: --env_gpus 0,1,2 (one actor per entry, repeats allowed), --envs_per_proc, --learner_gpu,
--out (a Modal Volume path works: checkpoints + resume), --minutes, --wandb_project.

    sim/twin/run.sh sim/twin/rl/dist.py --role launcher --run dist_smoke --env_gpus 0 --envs_per_proc 256 --minutes 3
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from multiprocessing.connection import Client, Listener, wait
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

ap = argparse.ArgumentParser()
ap.add_argument("--role", choices=["launcher", "learner", "actor"], required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--out_root", default=str(REPO / "sim/twin/out/rl"))
ap.add_argument("--env_gpus", default="0")
ap.add_argument("--envs_per_proc", type=int, default=256)
ap.add_argument("--learner_gpu", default="0")
ap.add_argument("--minutes", type=float, default=3.0)
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--select_warmup", type=int, default=100000)
ap.add_argument("--update_gate", type=int, default=10000)
ap.add_argument("--grad_steps_per_batch", type=int, default=20, help="learner grad steps per received actor batch")
ap.add_argument("--publish_every", type=int, default=200, help="grad steps between weight publications")
ap.add_argument("--expl_noise", type=float, default=0.05)
ap.add_argument("--bc_weight", type=float, default=2.0)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--expert", default="")
ap.add_argument("--resume", type=int, default=0)
ap.add_argument("--wandb_project", default="")
ap.add_argument("--wandb_entity", default="")
ap.add_argument("--pylibs", default="", help="extra site dir with wandb (appended to sys.path)")
ap.add_argument("--actor_id", type=int, default=0)
ap.add_argument("--preset", default="v3", choices=["v1", "v3", "v4"])
ap.add_argument("--detector", default="yolo_mix")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
out = Path(args.out_root) / args.run
out.mkdir(parents=True, exist_ok=True)
SOCK = str(out / "learner.sock")
KEYS = ("s", "s2", "A", "A2", "a_sel", "r", "rew", "term", "end", "env", "ep", "done_oc", "done_cen")


# ---------------------------------------------------------------------------------------------------- launcher
def launcher():
    procs = []
    base = [sys.executable, str(Path(__file__).resolve()), "--run", args.run, "--out_root", args.out_root]
    common = ["--minutes", str(args.minutes), "--K", str(args.K), "--envs_per_proc", str(args.envs_per_proc),
              "--bc_weight", str(args.bc_weight), "--batch", str(args.batch), "--expert", args.expert,
              "--select_warmup", str(args.select_warmup), "--update_gate", str(args.update_gate),
              "--grad_steps_per_batch", str(args.grad_steps_per_batch), "--publish_every", str(args.publish_every),
              "--resume", str(args.resume), "--wandb_project", args.wandb_project,
              "--wandb_entity", args.wandb_entity, "--expl_noise", str(args.expl_noise), "--pylibs", args.pylibs,
              "--preset", args.preset, "--detector", args.detector]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.learner_gpu)
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    procs.append(subprocess.Popen(base + ["--role", "learner"] + common, env=env))
    t = time.time()
    while not os.path.exists(SOCK) and time.time() - t < 300:
        time.sleep(0.5)
    for i, g in enumerate(args.env_gpus.split(",")):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        procs.append(subprocess.Popen(base + ["--role", "actor", "--actor_id", str(i), "--seed", str(args.seed + i)]
                                      + common, env=env))

    def stop(*_):
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for p in procs[1:]:
        p.wait()
    procs[0].wait(timeout=600)
    print("LAUNCHER_DONE", flush=True)


# ---------------------------------------------------------------------------------------------------- learner
def learner():
    import numpy as np
    import torch

    from sim.twin.rl.dice import Dice, DiceCfg, Replay, finalize_returns, mix

    d = "cuda:0"
    meta = None
    listener = Listener(SOCK + ".tmp", family="AF_UNIX", backlog=64)
    os.replace(SOCK + ".tmp", SOCK)
    listener._listener._unlink.cancel()
    conns, buf, agent, exp_buf = [], None, None, None
    last_row, ep_rows, ep_len = {}, {}, {}
    rows, recent, t0 = 0, [], time.perf_counter()
    logf = (out / "log.jsonl").open("a" if args.resume else "w")
    wb = None
    if args.wandb_project:
        if args.pylibs:
            sys.path.append(args.pylibs)
        import wandb

        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.run, resume="allow")
    sock = listener._listener._socket
    t_upd, n_batches, upd, last_pub = 0.0, 0, {}, -1
    end_t = t0 + args.minutes * 60 + 120
    while time.perf_counter() < end_t:
        ready = wait([sock] + conns, timeout=1.0)
        for c in ready:
            if c is sock:
                conns.append(listener.accept())
                continue
            try:
                msg = c.recv()
            except (EOFError, OSError):
                conns.remove(c)
                continue
            if msg.get("hello"):
                if meta is None:
                    meta = msg
                    dcfg = DiceCfg(bc_weight=args.bc_weight, batch=args.batch, k_state=msg["k_state"])
                    agent = Dice(msg["s_dim"], msg["a_dim"], dcfg, d)
                    (out / "config.json").write_text(json.dumps({"args": vars(args), "deploy": msg["deploy"]},
                                                                indent=1, default=str))
                    if args.resume and (out / "residual.pt").exists():
                        agent, _ = Dice.load(out / "residual.pt", d)
                    buf = Replay(3_000_000, msg["s_dim"], msg["k_state"], msg["a_dim"], 6, d)
                    if args.expert:
                        E = torch.load(args.expert, map_location=d)
                        exp_buf = Replay(len(E["s"]) + 1, msg["s_dim"], msg["k_state"], msg["a_dim"], 6, d)
                        cols = ("s", "s2", "A", "A2", "a_sel", "r", "rew", "term", "end", "epi")
                        ix = exp_buf.add(**{k: E[k] for k in cols})
                        exp_buf.nxt[ix] = E["nxt"]
                        exp_buf.G[ix] = E["G"]
                    publish(agent, rows)
                c.send({"ok": True})
                continue
            a = msg["actor"]
            B = {k: torch.as_tensor(msg[k], device=d) for k in KEYS}
            n = len(B["s"])
            epi = B["ep"] + a * 10 ** 9
            idx = buf.add(s=B["s"], s2=B["s2"], A=B["A"], A2=B["A2"], a_sel=B["a_sel"], r=B["r"], rew=B["rew"],
                          term=B["term"], end=B["end"], epi=epi)
            if a not in last_row:
                m = msg["n_envs"]
                last_row[a] = torch.full((m,), -1, dtype=torch.long, device=d)
                ep_rows[a] = torch.full((m, 32), -1, dtype=torch.long, device=d)
                ep_len[a] = torch.zeros(m, dtype=torch.long, device=d)
            env_i = B["env"].long()
            buf.link(last_row[a][env_i], idx)
            last_row[a][env_i] = idx
            ep_rows[a][env_i, ep_len[a][env_i].clamp(max=31)] = idx
            ep_len[a][env_i] += 1
            fin = B["end"]
            if bool(fin.any()):
                di = env_i[fin]
                finalize_returns(buf, ep_rows[a][di], ep_len[a][di], agent.cfg.gamma)
                ep_rows[a][di] = -1
                ep_len[a][di] = 0
                last_row[a][di] = -1
                for oc, cen in zip(B["done_oc"][fin].tolist(), B["done_cen"][fin].tolist(), strict=False):
                    recent.append((oc, cen))
                recent = recent[-2000:]
            rows += n
            n_batches += 1
            c.send({"rows": rows, "grad_steps": agent.steps})
            if buf.n >= args.update_gate:
                tb = time.perf_counter()
                for _ in range(args.grad_steps_per_batch):
                    ne = int(round(agent.cfg.batch * agent.expert_ratio())) if exp_buf is not None else 0
                    bo = buf.sample(agent.cfg.batch - ne, agent.cfg.gamma, agent.cfg.n_step)
                    be = exp_buf.sample(ne, agent.cfg.gamma, agent.cfg.n_step) if ne else None
                    upd = agent.update(mix(bo, be))
                torch.cuda.synchronize()
                t_upd += time.perf_counter() - tb
                if agent.steps - last_pub >= args.publish_every:
                    publish(agent, rows)
                    last_pub = agent.steps
            if n_batches % 20 == 0:
                R = recent[-1000:]
                row = {"t_s": round(time.perf_counter() - t0, 1), "rows": rows, "env_ticks": rows * meta["chunk"],
                       "wall_ticks_per_s": round(rows * meta["chunk"] / (time.perf_counter() - t0), 1),
                       "grad_steps": agent.steps, "learner_busy": round(t_upd / (time.perf_counter() - t0), 3),
                       "actors": len(conns), "success": round(float(np.mean([x[0] == 0 for x in R])), 4) if R else None,
                       "centred": round(float(np.mean([x[1] for x in R])), 4) if R else None,
                       **{k: round(v, 5) for k, v in upd.items()}}
                logf.write(json.dumps(row) + "\n")
                logf.flush()
                print("LEARN", json.dumps(row), flush=True)
                if wb is not None:
                    wb.log({k: v for k, v in row.items() if isinstance(v, (int, float))}, step=rows)
                agent.save(out / "residual.pt", save_meta(meta))
        if meta is not None and not conns and rows > 0:
            break
    if agent is not None:
        agent.save(out / "residual.pt", save_meta(meta))
    print("LEARNER_DONE", json.dumps({"rows": rows, "grad_steps": agent.steps if agent else 0}), flush=True)
    listener.close()


def save_meta(meta: dict) -> dict:
    """Checkpoint extra: dims + the actor process's deploy meta (scales, rates, tip point, GraspCfg)."""
    return {**meta["deploy"], **{k: meta[k] for k in ("obs_dim", "k_state", "chunk")}}


def publish(agent, rows):
    import torch

    tmp = out / "weights.tmp"
    torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(), "steps": agent.steps,
                "rows": rows}, tmp)
    os.replace(tmp, out / "weights.pt")


# ---------------------------------------------------------------------------------------------------- actor
def actor():
    from sim.twin.rl import grasp_env as G

    app = G.launch()
    import torch

    from sim.twin.rl import rollout as RO
    from sim.twin.rl.dice import Dice, DiceCfg

    env = G.make(G.GraspCfg(num_envs=args.envs_per_proc, **G.PRESETS[args.preset]))
    G.attach_detector(env, args.detector)
    n, d, K, KS = env.n, env.device, args.K, 4
    s_dim, a_dim = env.OBS_DIM + KS * env.h * 6, env.h * 6
    agent = Dice(s_dim, a_dim, DiceCfg(k_state=KS), d)
    conn = Client(SOCK, family="AF_UNIX")
    conn.send({"hello": True, "s_dim": s_dim, "a_dim": a_dim, "k_state": KS, "chunk": env.h, "obs_dim": env.OBS_DIM,
               "deploy": {**env.deploy_meta(), "detector": args.detector, "preset": args.preset}})
    conn.recv()
    wfile, wtime, learner_rows = out / "weights.pt", 0.0, 0
    gen = torch.Generator(device=d).manual_seed(args.seed)
    ids = torch.arange(n, device=d)
    env.reset_envs(ids, env.sample_params(n, gen, "train"))
    ep = torch.arange(n, device=d)
    nxt_ep = n
    obs = env.obs()
    F = env.cand_feat(env.candidates(K))
    t0 = time.perf_counter()
    steps = 0
    t_env = 0.0
    while time.perf_counter() - t0 < args.minutes * 60:
        if wfile.exists() and wfile.stat().st_mtime > wtime:
            try:
                w = torch.load(wfile, map_location=d)
                agent.actor.load_state_dict(w["actor"])
                agent.critic.load_state_dict(w["critic"])
                wtime = wfile.stat().st_mtime
            except Exception:  # noqa: BLE001 (being replaced)
                pass
        ta = time.perf_counter()
        s = RO.state_of(obs, F, KS)
        with torch.no_grad():
            k, r, _ = agent.select(s, F, "max_q_min")
            if learner_rows < args.select_warmup:
                k = torch.randint(K, (n,), device=d)
                r = agent.pi(s, F[ids, k])
            r = (r + torch.randn_like(r) * args.expl_noise).clamp(-1, 1)
        a_sel = F[ids, k]
        rew, term, trunc = env.step_chunk(k, r)
        obs2 = env.obs()
        C2 = env.candidates(K)
        F2 = env.cand_feat(C2)
        s2 = RO.state_of(obs2, F2, KS)
        end = term | trunc
        oc = env.outcome()
        msg = {"actor": args.actor_id, "n_envs": n, "s": s, "s2": s2, "A": F[:, :KS], "A2": F2[:, :KS],
               "a_sel": a_sel, "r": r, "rew": rew, "term": end, "end": end, "env": ids, "ep": ep.clone(),
               "done_oc": oc, "done_cen": env.centred.clone()}
        conn.send({k2: (v.cpu().numpy() if hasattr(v, "cpu") else v) for k2, v in msg.items()})
        if bool(end.any()):
            di = torch.nonzero(end).flatten()
            env.reset_envs(di, env.sample_params(len(di), gen, "train"))
            ep[di] = nxt_ep + torch.arange(len(di), device=d)
            nxt_ep += len(di)
            obs_n = env.obs()
            Cn = env.candidates(K)
            obs2 = torch.where(end[:, None], obs_n, obs2)
            C2 = torch.where(end[:, None, None, None], Cn, C2)
            env.cands = C2
            F2 = env.cand_feat(C2)
        obs, F = obs2, F2
        t_env += time.perf_counter() - ta
        learner_rows = conn.recv()["rows"]
        steps += 1
        if steps % 50 == 0:
            print("ACTOR", json.dumps({"actor": args.actor_id, "steps": steps,
                                       "env_ticks_per_s": round(steps * n * env.h / t_env, 1),
                                       "wall_ticks_per_s": round(steps * n * env.h / (time.perf_counter() - t0), 1)}),
                  flush=True)
    conn.close()
    app.close()


if __name__ == "__main__":
    {"launcher": launcher, "learner": learner, "actor": actor}[args.role]()
