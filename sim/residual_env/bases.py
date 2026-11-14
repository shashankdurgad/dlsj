"""Frozen base policies for the residual env. All return chunks (n, H, 6) in our units (old-calibration degrees,
gripper 0-100), aligned so chunk[:, j] is the target for tick (request tick + j)."""

from __future__ import annotations

import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_episodes(root: str | Path, episodes: list[int] | None = None) -> dict[int, dict]:
    """{ep: {"action": (T,6), "state": (T,6), "grasp_frame", "inspection_frame", "bowl"}} from a LeRobot dataset."""
    root = Path(root)
    lf = root / "meta/dlsj/labels.jsonl"
    labels = {}
    if lf.exists():
        for line in lf.read_text().splitlines():
            if line.strip():
                lab = json.loads(line)
                labels[int(lab["episode_index"])] = lab
    df = pd.concat(pd.read_parquet(f, columns=["action", "observation.state", "episode_index", "frame_index"])
                   for f in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True)))
    out = {}
    for ep, g in df.groupby("episode_index"):
        ep = int(ep)
        if episodes is not None and ep not in episodes:
            continue
        g = g.sort_values("frame_index")
        lab = labels.get(ep, {})
        out[ep] = {"action": np.stack(g["action"].values).astype(np.float32),
                   "state": np.stack(g["observation.state"].values).astype(np.float32),
                   "grasp_frame": lab.get("grasp_frame"), "inspection_frame": lab.get("inspection_frame"),
                   "bowl": lab.get("bowl_final", lab.get("bowl")),
                   "excluded": lab.get("excluded") or (lab.get("exclude_reason") if lab.get("exclude_from_training") else None)}
    return out


class HoldBase:
    """Keeps the commanded pose (no base motion): pure-residual sanity tests."""

    needs_images = False

    def __init__(self, horizon: int = 50):
        self.H = horizon

    def reset(self, ids):
        pass

    def request(self, ids, obs):
        s = obs["state"]
        return s.unsqueeze(1).expand(len(ids), self.H, 6).clone()


class ReplayBase:
    """Replays recorded teleop actions (one episode per env) as if they were the base policy's chunks: realistic,
    human-like base motion for fast residual-RL plumbing tests without a VLA in the loop."""

    needs_images = False

    def __init__(self, episodes: dict[int, dict], horizon: int = 50, device: str = "cuda:0", seed: int = 0):
        self.eps = list(episodes.values())
        self.H, self.device = horizon, device
        self.rng = np.random.default_rng(seed)
        self.assign: dict[int, int] = {}

    def reset(self, ids):
        for i in ids.tolist():
            self.assign[i] = int(self.rng.integers(len(self.eps)))

    def episode(self, i: int) -> dict:
        return self.eps[self.assign[i]]

    def request(self, ids, obs):
        out = []
        for k, i in enumerate(ids.tolist()):
            a = self.eps[self.assign[i]]["action"]
            t = int(obs["tick"][k])
            idx = np.clip(np.arange(t, t + self.H), 0, len(a) - 1)
            out.append(torch.from_numpy(a[idx]))
        return torch.stack(out).to(self.device)


class IPCVLABase:
    """Frozen VLA in a separate process (its own Python env), batched over envs via npz file IPC.

    Request file obs-XXXXX.npz: state (n,6), wrist/top (n,480,640,3) uint8, instruction (n,) str.
    Reply actions-XXXXX.npz: actions (n,H,6), latency_s. Worker: sim/residual_env/vla_batch_worker.py.
    The env applies the latency model in SIM time (cfg.base_latency_s), independent of the worker's wall time.
    """

    needs_images = True

    def __init__(self, ipc: str | Path, device: str = "cuda:0", timeout_s: float = 600.0):
        self.ipc, self.device, self.timeout = Path(ipc), device, timeout_s
        self.k = 0
        self.lat: list[float] = []
        t_end = time.time() + timeout_s
        while not (self.ipc / "ready").exists():
            if time.time() > t_end:
                raise TimeoutError("VLA worker not ready")
            time.sleep(1)

    def reset(self, ids):
        pass

    def request(self, ids, obs):
        imgs = obs["images"]
        tmp = self.ipc / f"obs-{self.k:05d}.tmp.npz"
        np.savez(tmp, state=obs["state"].cpu().numpy(), wrist=imgs["wrist"].cpu().numpy().astype(np.uint8),
                 top=imgs["top"].cpu().numpy().astype(np.uint8), instruction=np.asarray(obs["instruction"]))
        os.replace(tmp, self.ipc / f"obs-{self.k:05d}.npz")
        f = self.ipc / f"actions-{self.k:05d}.npz"
        t_end = time.time() + self.timeout
        while not f.exists():
            if time.time() > t_end:
                raise TimeoutError("VLA worker timed out")
            time.sleep(0.005)
        time.sleep(0.01)
        r = np.load(f)
        self.lat.append(float(r["latency_s"]))
        self.k += 1
        return torch.from_numpy(r["actions"].astype(np.float32)).to(self.device)


class ScriptedFailBase:
    """Scripted VLA-failure stand-in (NOT a learned policy): a closed-loop differential-IK reach -> grasp -> lift
    toward the TRUE brick centre, with the two failure modes we observed from the VLAs on this rig injected per
    episode: a systematic grasp-position error of a few cm (biased offset + noise) and a mistimed gripper close
    (often too early, while still descending). Uses the sim's own Jacobians (exact kinematics).

    Schedule (30 Hz ticks): 0..T_PRE reach a pre-grasp point 7 cm above the (erroneous) grasp point; T_PRE..T_DESC
    descend; close at T_CLOSE + timing error; lift 12 cm starting 15 ticks after the close; hold.
    Error model (seeded per episode): xy offset = bias + N(0, noise_xy), magnitude clipped to [err_min, err_max];
    z offset U(0, err_z); close-timing error: early U(-35, -15) ticks with p_early, late U(+10, +25) with p_late,
    else on time U(-3, +5).
    """

    needs_images = False
    T_PRE, T_DESC, T_CLOSE, LIFT_DELAY, LIFT_TICKS = 50, 85, 92, 15, 35

    def __init__(self, env=None, horizon: int = 50, seed: int = 0, bias_xy=(0.018, -0.012), noise_xy: float = 0.015,
                 err_min: float = 0.025, err_max: float = 0.05, err_z: float = 0.012, p_early: float = 0.45,
                 p_late: float = 0.15, pre_height: float = 0.07, lift: float = 0.12, open_units: float = 35.0,
                 damping: float = 0.05, scale_errors: float = 1.0):
        self.env, self.H = env, horizon
        self.rng = np.random.default_rng(seed)
        self.p = dict(bias_xy=np.asarray(bias_xy), noise_xy=noise_xy, err_min=err_min, err_max=err_max,
                      err_z=err_z, p_early=p_early, p_late=p_late)
        self.pre_h, self.lift_h, self.open_u, self.lam = pre_height, lift, open_units, damping
        self.scale = scale_errors
        self.err = None  # (n, 3) position error, env frame
        self.t_close = None  # (n,) ticks
        self.p_start = None  # (n, 3) tip at the first request
        self.forced: dict[int, dict] = {}  # env id -> fixed error sample (evaluation seeds)

    def sample(self, rng) -> dict:
        p = self.p
        e = p["bias_xy"] + rng.normal(0, p["noise_xy"], 2)
        m = np.linalg.norm(e)
        e = e / max(m, 1e-6) * np.clip(m, p["err_min"], p["err_max"])
        u = rng.random()
        dt = rng.uniform(-35, -15) if u < p["p_early"] else (
            rng.uniform(10, 25) if u < p["p_early"] + p["p_late"] else rng.uniform(-3, 5))
        return {"err": [float(e[0]), float(e[1]), float(rng.uniform(0, p["err_z"]))], "dt_close": float(dt)}

    def reset(self, ids):
        env = self.env
        n = env.n
        if self.err is None:
            self.err = torch.zeros(n, 3, device=env.device)
            self.t_close = torch.full((n,), float(self.T_CLOSE), device=env.device)
            self.p_start = torch.zeros(n, 3, device=env.device)
            self.started = torch.zeros(n, dtype=torch.bool, device=env.device)
        for i in ids.tolist():
            s = self.forced.pop(i, None) or self.sample(self.rng)
            self.err[i] = torch.tensor(s["err"], device=env.device) * self.scale
            self.t_close[i] = self.T_CLOSE + s["dt_close"] * self.scale
            self.started[i] = False
        self.last = {}

    def _tip_jacobian(self, ids):
        env = self.env
        Jall = env.robot.root_physx_view.get_jacobians()
        if not isinstance(Jall, torch.Tensor):  # Isaac Lab 3 returns a warp array
            import warp as wp
            Jall = wp.to_torch(Jall)
        nb, nd = env.robot.num_bodies, env.robot.num_joints
        bi = env.grip_body - 1 if Jall.shape[1] == nb - 1 else env.grip_body  # fixed base: root body not in J
        c0 = 6 if Jall.shape[-1] == nd + 6 else 0  # floating base: 6 root columns first
        jcols = torch.as_tensor(env.jidx[:5], device=Jall.device) + c0
        J = Jall[ids][:, bi][:, :, jcols]  # (m, 6, 5) arm joints
        link = env.robot.data.body_pos_w[ids, env.grip_body]
        tip = env.tip_w()[ids]
        r = tip - link
        Jl, Ja = J[:, :3], J[:, 3:]
        Jtip = Jl - torch.cross(r.unsqueeze(-1).expand_as(Ja), Ja, dim=1)  # v_tip = v + w x r = v - r x w
        R = _quat_to_mat_xyzw(env.robot.data.body_quat_w[ids, env.grip_body])
        return Jtip, Ja, tip - env.scene.env_origins[ids], R

    def _target(self, ids, tau):
        """Desired tip (m, 3) env frame and gripper units (m,) at tick tau (m,)."""
        env = self.env
        grasp = env.reset_brick[ids] + self.err[ids]
        pre = grasp + torch.tensor([0, 0, self.pre_h], device=env.device)
        tc = self.t_close[ids]
        lift_start = tc.clamp(min=self.T_DESC) + self.LIFT_DELAY
        a = (tau / self.T_PRE).clamp(0, 1).unsqueeze(-1)
        a = 0.5 - 0.5 * torch.cos(math.pi * a)
        p = self.p_start[ids] * (1 - a) + pre * a
        b = ((tau - self.T_PRE) / (self.T_DESC - self.T_PRE)).clamp(0, 1).unsqueeze(-1)
        b = 0.5 - 0.5 * torch.cos(math.pi * b)
        p = torch.where((tau > self.T_PRE).unsqueeze(-1), pre * (1 - b) + grasp * b, p)
        c = ((tau - lift_start) / self.LIFT_TICKS).clamp(0, 1).unsqueeze(-1)
        c = 0.5 - 0.5 * torch.cos(math.pi * c)
        lifted = grasp + torch.tensor([0, 0, self.lift_h], device=env.device)
        p = torch.where((tau > lift_start).unsqueeze(-1), grasp * (1 - c) + lifted * c, p)
        grip = torch.where(tau >= tc, torch.zeros_like(tau), torch.full_like(tau, self.open_u))
        return p, grip

    def request(self, ids, obs):
        env = self.env
        Jt, Ja, tip, R = self._tip_jacobian(ids)
        new = ~self.started[ids]
        if bool(new.any()):
            sel = ids[new]
            self.p_start[sel] = tip[new]
            self.started[sel] = True
        q = env.robot.data.joint_pos[ids][:, env.jidx]  # sim radians (m, 6)
        a_cur = R[:, :, 2]  # gripper local +z in world; fingers point along local -z
        a_des = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(a_cur)
        e_rot = torch.cross(a_cur, a_des, dim=-1)
        Jfull = torch.cat([Jt, Ja * 0.08], 1)  # (m, 6, 5): orientation rows scaled to metres (8 cm lever)
        JJt = Jfull @ Jfull.transpose(1, 2) + (self.lam ** 2) * torch.eye(6, device=env.device)
        t0 = obs["tick"].to(env.device).float()
        chunk = []
        for j in range(self.H):
            p_des, grip = self._target(ids, t0 + j)
            err = torch.cat([p_des - tip, e_rot * 0.08], -1).unsqueeze(-1)
            dq = (Jfull.transpose(1, 2) @ torch.linalg.solve(JJt, err)).squeeze(-1)
            dq = dq.clamp(-0.8, 0.8)
            qt = q.clone()
            qt[:, :5] = q[:, :5] + dq
            ours = env.to_ours(qt)
            ours[:, 5] = grip
            chunk.append(ours)
        return torch.stack(chunk, 1)


def _quat_to_mat_xyzw(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
