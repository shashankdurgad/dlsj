"""Shared pieces of the residual-RL demo (scripted VLA-failure stand-in base + learned residual).

Task: grasp the target brick and lift it >= 5 cm, held for 1 s (30 ticks). Episode 8 s at 30 Hz.
Residual action (6,): gripper +-25 units, shoulder_pan / shoulder_lift / elbow_flex +-8 deg, wrist_flex +-6 deg,
wrist_roll +-4 deg, added to the base target each tick before the LeRobot clamp.
"""
from __future__ import annotations

import numpy as np
import torch

DEMO_RES_JOINTS = (5, 0, 1, 2, 3, 4)
DEMO_RES_SCALE = (25.0, 8.0, 8.0, 8.0, 6.0, 4.0)
LIFT_M = 0.05
HOLD_TICKS = 30
EPISODE_S = 8.0


def demo_cfg(num_envs: int, cameras: bool = False, latency: float = 0.074):
    from env import EnvCfg

    return EnvCfg(num_envs=num_envs, cameras=cameras, base_latency_s=latency, episode_s=EPISODE_S, calib="joe_cal2",
                  residual_joints=DEMO_RES_JOINTS, residual_scale=DEMO_RES_SCALE, lift_height=LIFT_M,
                  w_present=0.0, w_place=0.0)


class SuccessTracker:
    """Per-env: lifted >= LIFT_M with the brick at the grasp point for HOLD_TICKS consecutive ticks."""

    def __init__(self, n, device):
        self.count = torch.zeros(n, dtype=torch.long, device=device)

    def update(self, info):
        self.count = torch.where(info["lifted"], self.count + 1, torch.zeros_like(self.count))
        return self.count >= HOLD_TICKS

    def reset(self, ids):
        self.count[ids] = 0


@torch.no_grad()
def rollout(env, base, policy=None, seed: int = 1000, record=None):
    """All envs from one seeded reset, run to the end (no early reset). Returns per-env metrics (numpy).
    `policy(obs) -> action` or None (base alone). `record(t)` is called after every tick (rendering)."""
    n, dev = env.n, env.device
    env.gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scripted = hasattr(base, "sample") and hasattr(base, "forced")
    if scripted:  # same per-episode base errors for every evaluation with this seed
        for i in range(n):
            base.forced[i] = base.sample(rng)
    obs = env.reset()
    tr = SuccessTracker(n, dev)
    success = torch.zeros(n, dtype=torch.bool, device=dev)
    t_success = torch.full((n,), -1, dtype=torch.long, device=dev)
    max_lift = torch.zeros(n, device=dev)
    err_at_close = torch.full((n,), float("nan"), device=dev)
    z0 = env.reset_brick[:, 2].clone()
    for t in range(env.max_ticks):
        act = policy(obs) if policy is not None else None
        obs, rew, done, info = env.step(act)
        s = tr.update(info)
        newly = s & ~success
        t_success = torch.where(newly, torch.full_like(t_success, t), t_success)
        success |= s
        bz = env.brick_local()[:, 2] - z0
        max_lift = torch.maximum(max_lift, torch.where(info["dist"] < 0.04, bz, torch.zeros_like(bz)))
        at_close = (base.t_close.round().long() == t) if scripted else torch.zeros_like(success)
        if bool(at_close.any()):
            d_xy = torch.linalg.norm((env.tip_w() - env.scene.env_origins)[:, :2] - env.brick_local()[:, :2], dim=-1)
            err_at_close = torch.where(at_close, d_xy, err_at_close)
        if record is not None:
            record(t)
    out = {"success": success.cpu().numpy(), "t_success": t_success.cpu().numpy(),
           "max_lift_m": max_lift.cpu().numpy(), "xy_err_at_close_m": err_at_close.cpu().numpy()}
    if scripted:
        out["base_err_xy_m"] = torch.linalg.norm(base.err[:, :2], dim=-1).cpu().numpy()
        out["base_dt_close"] = (base.t_close - base.T_CLOSE).cpu().numpy()
    return out


def summarise(m: dict) -> dict:
    return {"episodes": int(len(m["success"])), "success_rate": float(np.mean(m["success"])),
            "mean_max_lift_cm": float(np.mean(m["max_lift_m"]) * 100),
            "median_xy_err_at_close_cm": (float(np.nanmedian(m["xy_err_at_close_m"]) * 100)
                                          if np.isfinite(m["xy_err_at_close_m"]).any() else None)}
