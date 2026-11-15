"""Mechanical play, load-dependent compliance, stiction and encoder model of the real SO-101 follower, as a hook for
the twin's servo model (sim/residual_env/env.py ResidualEnv.apply_goal: LeRobot clamp -> first-order servo ->
PhysX PD drive). Identified from the recorded encoders (ident.py, fit.py; parameters in params.json).

    from sim.twin.dynamics.play import Play, PlayCfg, install
    play = install(env, PlayCfg())        # env: ResidualEnv / GraspEnv / TwinEnv.residual (wraps the instance)
    play.sample(env_ids, gen)             # optional per-env domain randomisation (cfg.dr=True draws on install)

What it changes, per control tick (units: LeRobot degrees, gripper 0-100; all tensors (n, 6)):
1. command delay (cfg.controller_delay, off by default): the deployed controllers' traces fit one extra tick on
   pan / wrist_flex / wrist_roll; teleop fits none.
2. servo target = the clamped goal + c0 + (c - c_drive) * tau  (encoder-visible sag of the P-only servo loop; tau =
   static gravity torque at the current joint pose, gravity.py) + link_sag * tau (camera-only compliance, 0 unless
   set), then the hold band: the target only moves once the goal leaves the band +-(d0 + m |tau|) around it
   (direction memory = backlash / deadband / load-dependent gear friction), plus a breakaway margin s when the
   joint was at rest (stiction; extra lost motion at reversals). This REPLACES the servo model's constant gravity
   bias (the env's `bias` is zeroed on install).
   c_drive = the compliance the PhysX arm drive already adds (180/pi / stiffness, default 60 N m/rad -> 0.95 deg /
   N m): it is subtracted so the sag is not counted twice. With cfg.drive="physical" the twin's arm drive is
   expected to run at the real servo stiffness (params.json isaac_drive_physical: the env owner sets it in the scene
   cfg) and the play model adds no sag of its own (gravity and contact loads deflect the arm in PhysX).
3. reported state (`measured(noisy=True)`): the joint minus link_sag * tau (the encoder sits on the servo output,
   not on the link), quantised to the STS3215's 4096 counts / rev (0.0879 deg; gripper 0.0665 units) on a per-env
   grid phase, with rare +-1 count flicker. No Gaussian noise: the real readings do not jitter at rest.
   `measured(noisy=False)` stays the true joint state (the env's FK / privileged uses).
Everything else (first-order K, velocity limit, LeRobot clamp, gripper contact stall) is the env's own servo model.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from sim.twin.dynamics.gravity import Arm

HERE = Path(__file__).resolve().parent
PARAMS = HERE / "params.json"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
RAD2DEG = 180.0 / math.pi


@dataclass
class PlayCfg:
    params: str = str(PARAMS)
    band: bool = True  # hold band / backlash / stiction (2.)
    sag: bool = True  # encoder-visible load sag c * tau (2.)
    link_sag: tuple | None = None  # camera-only compliance, deg per N m on (lift, elbow, wrist_flex); None = params
    encoder: bool = True  # quantised readings instead of Gaussian noise (3.)
    controller_delay: bool = False  # one extra tick on params["delay_joints"] (deployed-controller traces)
    drive: str = "compensate"  # "compensate": subtract the PhysX drive's own sag | "physical": see module doc
    drive_stiffness: float = 60.0  # N m/rad of the twin's arm drive (sim/residual_env/scene.py default)
    dr: bool = False  # draw per-env parameters from params["dr"] on install / sample()
    seed: int = 0
    overrides: dict = field(default_factory=dict)  # {"c": [...], "d0": [...], ...} replaces params values


class Play:
    """Per-env play state. Torch (any device) or numpy; all inputs in ours units (n, 6)."""

    def __init__(self, cfg: PlayCfg, n: int, joint_offsets, device="cpu", arm: Arm | None = None):
        self.cfg, self.n, self.device = cfg, n, device
        p = json.loads(Path(cfg.params).read_text())
        p.update(cfg.overrides)
        self.p = p
        self.arm = arm or Arm()
        t = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float32, device=device)  # noqa: E731
        self.off = t(joint_offsets)[:6]
        self.nominal = {k: t(p[k]) for k in ("c0", "c", "d0", "m", "s")}
        self.nominal["link_sag"] = t(np.r_[0.0, list(cfg.link_sag), 0.0, 0.0] if cfg.link_sag is not None
                                     else p.get("link_sag", [0.0] * 6))
        self.count = t(p["encoder"]["count"])  # (6,) deg (gripper: units)
        self.flicker_p = float(p["encoder"]["flicker_p"])
        self.delay_mask = torch.zeros(6, dtype=torch.bool, device=device)
        if cfg.controller_delay:
            self.delay_mask[[JOINTS.index(j) for j in p["delay_joints"]]] = True
        if cfg.drive == "physical":
            self.c_drive = self.nominal["c"].clone()  # the drive itself sags by c * tau: add none here
        else:
            cd = RAD2DEG / cfg.drive_stiffness
            self.c_drive = t([0.0, cd, cd, cd, cd, 0.0])
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(cfg.seed)
        self.P = {k: v.expand(n, 6).clone() for k, v in self.nominal.items()}
        self.phase = torch.rand(n, 6, generator=self.gen, device=device) * self.count  # encoder grid phase
        self.eff = None  # band state (servo target), ours
        self.rest = torch.ones(n, 6, dtype=torch.bool, device=device)
        self.prev_goal = None
        if cfg.dr:
            self.sample(torch.arange(n, device=device))

    # ------------------------------------------------------------------ parameters
    def sample(self, ids, gen: torch.Generator | None = None):
        """Per-env draw around the fitted values (params["dr"]: multiplicative ranges and additive c0 / link_sag)."""
        g = gen or self.gen
        dr = self.p["dr"]
        m = len(ids)

        def u(lo, hi):
            return lo + (hi - lo) * torch.rand(m, 6, generator=g, device=self.device)

        for k in ("c", "d0", "m", "s"):
            lo, hi = dr[f"{k}_scale"]
            self.P[k][ids] = self.nominal[k] * u(lo, hi)
        a = dr["c0_add_deg"]
        self.P["c0"][ids] = self.nominal["c0"] + u(-a, a)
        lo, hi = dr["link_sag_deg_per_Nm"]
        ls = u(lo, hi) * torch.tensor([0, 1, 1, 1, 0, 0], dtype=torch.float32, device=self.device)
        self.P["link_sag"][ids] = self.nominal["link_sag"] + ls
        self.phase[ids] = torch.rand(m, 6, generator=g, device=self.device) * self.count

    # ------------------------------------------------------------------ dynamics
    def tau(self, q_ours: torch.Tensor) -> torch.Tensor:
        """(n, 6) static gravity torque (N m; gripper / pan / roll ~0) at the joint pose (ours units)."""
        q = torch.deg2rad(q_ours) + self.off
        out = torch.zeros_like(q_ours)
        out[:, :5] = self.arm.gravity_torque(q.double()).to(q_ours.dtype)
        return out

    def reset(self, ids, q_ours: torch.Tensor):
        """Envs `ids` restart at joint state q_ours (m, 6): band state = there, at rest."""
        if self.eff is None:
            self.eff = torch.zeros(self.n, 6, device=self.device)
            self.prev_goal = torch.zeros(self.n, 6, device=self.device)
        self.eff[ids] = q_ours
        self.prev_goal[ids] = q_ours
        self.rest[ids] = True

    def target(self, goal: torch.Tensor, q_true: torch.Tensor) -> torch.Tensor:
        """Clamped goal (n, 6) + current true joints (n, 6) -> the servo model's target (n, 6) (bias already in)."""
        if self.eff is None:
            self.reset(torch.arange(self.n, device=self.device), q_true)
        P, cfg = self.P, self.cfg
        if cfg.controller_delay:
            g = torch.where(self.delay_mask, self.prev_goal, goal)
            self.prev_goal = goal.clone()
            goal = g
        tau = self.tau(q_true)
        u = goal + P["c0"]
        if cfg.sag:
            u = u + (P["c"] - self.c_drive) * tau
        u = u + P["link_sag"] * tau
        if not cfg.band:
            self.eff = u
            return u
        w = P["d0"] + P["m"] * tau.abs()
        wb = w + P["s"] * self.rest  # breakaway margin from rest
        lo, hi = u - wb, u + wb
        out = (self.eff < lo) | (self.eff > hi)  # the goal left the band: the joint moves
        new = torch.where(out, torch.clamp(self.eff, u - w, u + w), self.eff)
        self.rest = ~out
        self.eff = new
        return new

    def measure(self, q_true: torch.Tensor) -> torch.Tensor:
        """Encoder reading (n, 6) of the true joints: link-only sag removed, quantised, rare +-1 count flicker."""
        q = q_true - self.P["link_sag"] * self.tau(q_true) if bool((self.P["link_sag"] != 0).any()) else q_true
        if not self.cfg.encoder:
            return q
        k = self.count
        r = torch.round((q - self.phase) / k) * k + self.phase
        if self.flicker_p > 0:
            f = torch.rand(r.shape, generator=self.gen, device=self.device) < self.flicker_p
            sgn = torch.where(torch.rand(r.shape, generator=self.gen, device=self.device) < 0.5, -1.0, 1.0)
            r = r + f * sgn * k
        return r


def install(env, cfg: PlayCfg | None = None) -> Play:
    """Wrap a ResidualEnv-like instance (ResidualEnv, GraspEnv, TwinEnv.residual) with the play model, in place.

    * env.apply_goal(goal): LeRobot clamp around the encoder reading -> Play.target -> first-order servo (env.K,
      env.vmax) -> PD target WITHOUT the env's constant bias (env.bias is zeroed: the play model replaces it).
    * env.measured(noisy=True) -> Play.measure (quantised encoder); noisy=False -> the true joints (unchanged).
    * resets are detected per env (env.r changed outside apply_goal) and restart the play state there.
    """
    cfg = cfg or PlayCfg()
    play = Play(cfg, env.n, env.off.detach().cpu().numpy(), device=env.device)
    orig_measured = env.measured
    env.bias = torch.zeros_like(env.bias)
    last_r = {"r": None}

    def measured(noisy: bool = True):
        s = orig_measured(noisy=False)
        return play.measure(s) if noisy else s

    def apply_goal(goal: torch.Tensor):
        goal = goal.to(device=env.r.device, dtype=env.r.dtype)
        true = orig_measured(noisy=False)
        if play.eff is None:
            play.reset(torch.arange(env.n, device=env.device), env.r.clone())
        elif last_r["r"] is not None:
            ids = torch.nonzero((env.r != last_r["r"]).any(-1)).flatten()
            if len(ids):
                play.reset(ids, env.r[ids].clone())
        meas = play.measure(true)
        m = env.cfg.max_relative_target
        clamped = torch.maximum(torch.minimum(goal, meas + m), meas - m)
        env.stats["clamp_hits"] += (clamped != goal).any(-1).float()
        tgt = play.target(clamped, true)
        env.r = env.r + torch.clamp(env.K * (tgt - env.r), -env.vmax, env.vmax)
        last_r["r"] = env.r.clone()
        q = env.robot.data.joint_pos_target.clone() if hasattr(env.robot.data, "joint_pos_target") \
            else env.robot.data.default_joint_pos.clone()
        q[:, env.jidx] = env.to_sim(env.r + env.bias)
        env.robot.set_joint_position_target(q)

    env.measured = measured
    env.apply_goal = apply_goal
    env.play = play
    return play
