"""Policies and the fixed evaluation protocol for the grasp env (shared by baselines, training and eval).

Obs layout (GraspEnv.obs, 25): joints/100 [0:6], detector xyz*10 [6:9], valid [9], load [10], t/T [11],
applied offset / scale [12:18], previous residual [18:24], pick_phase [24].
"""
from __future__ import annotations

import json

import torch

from sim.twin.rl.grasp_env import FAIL, POSES, START_TYPES


def state_of(obs, F, k_state):
    """Critic / actor state: obs + the first k_state candidate features (pi0.5 template: latent + all K chunks)."""
    return torch.cat([obs, F[:, :k_state].flatten(1)], -1)


class Scripted:
    """Sanity residual using only the detector: align x/y on the brick during the approach; when the base starts
    closing, hold the jaw open (+30 units), lower / raise the fingertip to 1 cm above the brick centre and release
    the jaw once aligned (latched); freeze the offsets once the load proxy shows a squeeze."""

    def __init__(self, env, tol_xy=0.006, tol_z=0.005):
        self.env, self.tol_xy, self.tol_z = env, tol_xy, tol_z
        self.released = torch.zeros(env.n, dtype=torch.bool, device=env.device)
        self.frozen = torch.zeros_like(self.released)

    def __call__(self, obs, F):
        env = self.env
        n = obs.shape[0]
        first = obs[:, 11] * env.max_ticks < 0.5  # new episode
        self.released &= ~first
        self.frozen &= ~first
        det, valid = obs[:, 6:9] / 10.0, obs[:, 9] > 0.5
        o = obs[:, 12:18] * env.res_scale
        sel = F[:, 0].view(n, env.h, 6)
        g_base = sel[:, -1, 5] * 20 + obs[:, 5] * 100  # base gripper goal at the chunk end
        closing = g_base < 30
        self.frozen |= obs[:, 10] > 0.15  # squeeze (measured - commanded gripper > 1.5 units)
        tgt = o.clone()
        upd = valid & ~self.frozen
        # aim the brick 6 mm toward the moving jaw (clear of the fixed finger), gain 0.5 against detector noise
        err = det + torch.tensor([-0.006, 0.0, 0.010], device=obs.device)
        tgt[:, :2] = torch.where(upd[:, None], o[:, :2] + 0.5 * err[:, :2], o[:, :2])
        zt = o[:, 2] + 0.5 * err[:, 2]
        tgt[:, 2] = torch.where(upd & closing, zt, o[:, 2])
        aligned = valid & (err[:, :2].norm(dim=-1) < self.tol_xy) & (err[:, 2].abs() < self.tol_z)
        self.released |= closing & aligned
        hold_open = closing & ~self.released & ~self.frozen
        tgt[:, 5] = torch.where(hold_open, torch.full_like(o[:, 5], float(env.res_scale[5])), torch.zeros_like(o[:, 5]))
        tgt[:, 3:5] = 0
        return (tgt / env.res_scale).clamp(-1, 1)


class Recorder:
    """Per-tick joints (ours), brick pose (env frame) and residual offset of a few envs (render_grasps.py)."""

    def __init__(self, ids):
        self.ids, self.q, self.bp, self.bq, self.o = ids, [], [], [], []

    def __call__(self, env):
        i = self.ids
        self.q.append(env.measured(noisy=False)[i].cpu())
        self.bp.append(env.brick_local()[i].cpu())
        self.bq.append(env.bquat_w()[i].cpu())
        self.o.append(env.o[i].cpu())

    def arrays(self):
        return {k: torch.stack(v, 1).numpy() for k, v in
                (("q", self.q), ("brick_pos", self.bp), ("brick_quat", self.bq), ("offset", self.o))}


def run_batch(env, params, policy="base", agent=None, K=1, k_state=4, scripted=None, gen=None, recorder=None):
    """One batch of episodes (len(params['ep']) == env.n). policy: base | scripted | dice | select (best-of-K, r=0)
    | residual (dice with K=1). Returns per-episode tensors."""
    n, d = env.n, env.device
    ids = torch.arange(n, device=d)
    env.reset_envs(ids, params)
    base_hook = env.tick_hook  # e.g. the detector model's 30 Hz camera tick

    def hook(e):
        if base_hook is not None:
            base_hook(e)
        recorder(e)

    if recorder is not None:
        env.tick_hook = hook
    abs_r = torch.zeros(n, 6, device=d)
    n_dec = torch.zeros(n, device=d)
    while not bool(env.done.all()):
        obs = env.obs()
        C = env.candidates(16, gen)  # same draw count for every policy (paired candidate noise)
        F = env.cand_feat(C)
        if policy == "base":
            k, r = torch.zeros(n, dtype=torch.long, device=d), torch.zeros(n, 6, device=d)
        elif policy == "scripted":
            k, r = torch.zeros(n, dtype=torch.long, device=d), scripted(obs, F)
        else:
            s = state_of(obs, F, k_state)
            Fk = F[:, :K]
            k, r, _ = agent.select(s, Fk, "max_q_min" if K > 1 else "first", residual=policy != "select")
        pick = env.pick & ~env.done
        k = torch.where(env.pick, k, torch.zeros_like(k))  # best-of-K only in the pick phase
        abs_r += r.abs() * pick[:, None]
        n_dec += pick.float()
        env.step_chunk(k, r)
    env.tick_hook = base_hook
    oc = env.outcome()
    return {"outcome": oc, "centred": env.centred.clone(),
            "abs_o": env.sum_abs_o / env.n_pick_ticks.clamp(min=1)[:, None],
            "abs_r": abs_r / n_dec.clamp(min=1)[:, None], "kind": env.kind.clone(), "pose": env.pose.clone(),
            "start": env.start_type.clone(), "n_dis": env.n_dis.clone(), "nn_gap": env.nn_gap.clone(),
            "vetoed": env.vetoed.clone(),
            "hover": env.hover.clone(),
            "t_succ": env.t_succ.float(), "lift_diag": env.lift_diag.clone(), "table": env.n_table.clone(),
            "reopen": env.n_reopen.clone()}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (round(c - h, 4), round(c + h, 4))


def summarize(res: dict, label: str) -> dict:
    oc = res["outcome"]
    ns = int((oc == 0).sum())
    nc = int(res["centred"].sum())
    out = {"policy": label, "n": int(len(oc)), "pick_success": round(ns / len(oc), 4),
           "pick_success_ci95": wilson(ns, len(oc)), "success": round(ns / len(oc), 4),
           "centred_ci95": wilson(nc, len(oc)),
           "centred": round(float(res["centred"].float().mean()), 4),
           "centred_given_success": round(float(res["centred"][oc == 0].float().mean()), 4) if (oc == 0).any() else 0.0}
    out["failures"] = {FAIL[k]: round(float((oc == k).float().mean()), 4) for k in range(1, len(FAIL))}
    o = res["abs_o"].mean(0).tolist()
    out["mean_abs_delta"] = {"tip_xyz_mm": [round(v * 1e3, 2) for v in o[:3]], "wrist_flex_deg": round(o[3], 2),
                             "wrist_roll_deg": round(o[4], 2), "gripper_units": round(o[5], 2)}
    out["mean_abs_r"] = [round(v, 3) for v in res["abs_r"].mean(0).tolist()]
    def rate(mask):
        k, n = int((oc[mask] == 0).sum()), int(mask.sum())
        return {"n": n, "success": round(k / n, 3), "ci95": wilson(k, n)}

    out["wrong_brick"] = round(float((oc == 5).float().mean()), 4)
    out["wrong_brick_ci95"] = wilson(int((oc == 5).sum()), len(oc))
    gap = res["nn_gap"]
    out["by_neighbour_gap"] = {}
    for lab, lo, hi in (("touching", -1.0, 0.002), ("<2cm", 0.002, 0.02), ("2-5cm", 0.02, 0.05), (">5cm", 0.05, 9.0)):
        mk = (gap >= lo) & (gap < hi)
        if mk.any():
            n_, s_, w_ = int(mk.sum()), int((oc[mk] == 0).sum()), int((oc[mk] == 5).sum())
            out["by_neighbour_gap"][lab] = {"n": n_, "success": round(s_ / n_, 3), "ci95": wilson(s_, n_),
                                            "wrong": round(w_ / n_, 3)}
    out["veto_ticks"] = round(float(res["vetoed"].mean()), 2)
    out["by_start"] = {nm: rate(res["start"] == k) for k, nm in enumerate(START_TYPES) if (res["start"] == k).any()}
    bricks = res["n_dis"] + 1
    out["by_bricks"] = {lab: rate((bricks >= lo) & (bricks <= hi)) for lab, lo, hi in
                        (("1", 1, 1), ("2-3", 2, 3), ("4-5", 4, 5), ("6-8", 6, 8))
                        if ((bricks >= lo) & (bricks <= hi)).any()}
    for k, nm in enumerate(POSES):
        m = res["pose"] == k
        if m.any():
            out[f"success_{nm}"] = round(float((oc[m] == 0).float().mean()), 3)
            out[f"n_{nm}"] = int(m.sum())
    for k, nm in enumerate(("2x2", "2x4", "curved")):
        m = res["kind"] == k
        if m.any():
            out[f"success_{nm}"] = round(float((oc[m] == 0).float().mean()), 3)
    if res["hover"].any():
        out["success_hover"] = round(float((oc[res["hover"]] == 0).float().mean()), 3)
    ld = res["lift_diag"].sum(0)
    if ld[0] > 0:
        out["while_lifted"] = {k: round(float(ld[i] / ld[0]), 3) for i, k in
                               enumerate(("ticks", "grasped", "y_ok", "z_ok", "tilt_ok", "tilt_deg")) if i}
    out["table_contact_ticks"] = round(float(res["table"].mean()), 2)
    out["reopen_events"] = round(float(res["reopen"].mean()), 3)
    return out


def eval_params(env, n_eps=512, seed=1234, pool="test", errors=True):
    gen = torch.Generator(device=env.device).manual_seed(seed)
    return env.sample_params(n_eps, gen, pool, errors=errors)


def evaluate(env, policy, n_eps=512, seed=1234, pool="test", errors=True, **kw):
    """Fixed protocol: the same n_eps episodes (params from `seed`) in batches of env.n; fixed candidate noise seed."""
    P = eval_params(env, n_eps, seed, pool, errors)
    outs = []
    end = env.g.end_on_success
    env.g.end_on_success = True  # the task is the pick only: scoring stops at the held lift (as in training)
    gen = torch.Generator(device=env.device).manual_seed(seed + 1)
    env.gen.manual_seed(seed + 2)
    try:
        for i in range(0, n_eps, env.n):
            sub = {k: v[i:i + env.n] for k, v in P.items()}
            m = len(sub["ep"])
            if m < env.n:  # pad with repeats, drop afterwards
                sub = {k: torch.cat([v, v[: env.n - m]]) for k, v in sub.items()}
            r = run_batch(env, sub, policy, gen=gen, **kw)
            outs.append({k: v[:m] for k, v in r.items()})
    finally:
        env.g.end_on_success = end
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def dump(rows, path):
    path.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print("EVAL", json.dumps(r), flush=True)
