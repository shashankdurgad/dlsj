"""Simulated wrist-camera brick detector, fitted to the real one (brick-detector-v1 on recorded cal2 wrist frames).

What the robot would get from the real detector during the pick (see README.md): per camera frame, boxes + confidence
(one class). The residual's brick observation is derived from the target's box by `Geometry.box_to_rel` (ray through
the box centre -> table plane -> brick centre relative to the fingertip, gripper_link frame). This model produces the
same thing in the twin, from the geometric visibility of the brick (geometry.py: fins / bases / other bricks / image
edge), so it is blind when and where the real detector is blind:

    s_t = x_t . beta_thr + b_ep + e_t      a box with conf >= thr is output iff s_t > 0      (probit, per threshold)
    x = [1, vis, occ_grip, oof, log(area/1e4), log^2, blur, blur^2, closing, dist, dist^2, grip closing speed]
        (blur = brick-centre image speed / 10 px per frame)
    e_t = rho^k e_{t-k} + sqrt(1 - rho^2k) eps         (k = frames since the last call; misses come in runs)
    b_ep per episode (the brick's colour / glare / kind), drawn from the fitted episodes' empirical distribution
    conf | box = clip(x . gamma + sigma_c n, thr, 0.79)
    wrong_target: the robot's TargetTracker (wrist_target.py) sometimes locks on another brick / a plate edge
    (9 % of pick frames on the recorded data, runs of ~7 frames, 4-12 cm away): a Markov state that, while on,
    reports a detection at the brick's ray-plane point shifted by a per-run table offset.
    box = geometric VISIBLE box (in-frame, unoccluded part), centre shifted and resized by AR(1) noise whose mean and
          std grow with 1 - vis;  pose = Geometry.box_to_rel(box) (same function as on the robot).
Parameters: detector_model.json (fit_model.py). Everything is batched over envs: numpy (`backend="numpy"`) or torch
(`backend="torch"`, any device).

    geo = Geometry(torch_device="cuda:0")
    det = DetectorModel(n_envs, geo=geo)                     # backend follows geo
    det.reset(ids, gen)                                      # new episodes: per-brick difficulty, AR state
    vis = geo.visibility(joints, brick_pos, brick_R, brick_size, others)
    out = det.observe(joints, vis, gen, dt_frames=1)          # dict: detected (n,), conf (n,), box (n, 4) px,
                                                             #   rel (n, 3) brick centre - fingertip (gripper_link),
                                                             #   pos (n, 3) env-frame estimate
    detected, pose_estimate, confidence = out["detected"], out["rel"], out["conf"]

`GraspEnvDetector` adapts it to GraspEnv's hook `env.detector_model = f(env, rel) -> (det, valid)`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sim.twin.perception import geometry as G

MODEL_JSON = Path(__file__).resolve().parent / "detector_model.json"
BRICK_CENTRE_Z = G.TABLE_Z + 0.0096  # the robot assumes a brick standing studs-up on the table


def load_params(path: str | Path = MODEL_JSON) -> dict:
    return json.loads(Path(path).read_text())


class DetectorModel:
    def __init__(
        self,
        n: int,
        params: dict | None = None,
        geo: G.Geometry | None = None,
        threshold: float | None = None,
        latency_frames: int | None = None,
    ):
        self.p = params or load_params()
        self.geo = geo or G.Geometry(camera=self.p.get("camera", "physical"))
        self.xp = xp = self.geo.xp
        self.n = n
        self.threshold = float(self.p["threshold"] if threshold is None else threshold)
        self.latency = int(self.p.get("latency_frames", 1) if latency_frames is None else latency_frames)
        kw = {} if xp is np else {"device": self.geo.a["T"].device, "dtype": self.geo.a["T"].dtype}
        self._kw = kw
        dp = self.p["det"][min(self.p["det"], key=lambda k: abs(float(k) - self.threshold))]
        self.dp = dp
        self.beta = self._t(dp["beta"])
        self.gamma = self._t(self.p["conf"]["gamma"])
        bx = self.p["box"]
        self.box_mean = self._t([bx[k]["mean"] for k in ("dx", "dy", "lw", "lh")])  # (4, 2)
        self.box_std = self._t([bx[k]["std"] for k in ("dx", "dy", "lw", "lh")])
        self.b = self._z(n)
        self.e = self._z(n)
        self.u = self._z(n, 4)
        self.wrong = self._z(n) > 1  # the robot's tracker is locked on another object (wrong_target model)
        self.w_off = self._z(n, 2)
        self.prev_uv = None
        self.prev_grip = None
        self.fresh = xp.ones(n, dtype=bool) if xp is np else xp.ones(n, dtype=xp.bool, device=kw["device"])
        self.fresh_g = self.fresh.copy() if xp is np else self.fresh.clone()

    # ------------------------------------------------------------------ helpers
    def _t(self, v):
        return np.asarray(v, float) if self.xp is np else self.xp.as_tensor(v, **self._kw)

    def _z(self, *s):
        return np.zeros(s) if self.xp is np else self.xp.zeros(*s, **self._kw)

    def _randn(self, gen, *s):
        if self.xp is np:
            return (gen or np.random.default_rng()).standard_normal(s)
        return self.xp.randn(*s, generator=gen, **self._kw)

    def _rand(self, gen, *s):
        if self.xp is np:
            return (gen or np.random.default_rng()).random(s)
        return self.xp.rand(*s, generator=gen, **self._kw)

    # ------------------------------------------------------------------ episode state
    def reset(self, ids=None, gen=None):
        """New episodes for envs `ids` (default all): per-brick difficulty b ~ N(0, sigma_b), AR states drawn from
        their stationary distributions."""
        xp = self.xp
        if ids is None:
            ids = np.arange(self.n) if xp is np else xp.arange(self.n, device=self._kw["device"])
        m = len(ids)
        qs = self._t(self.dp["b_quantiles"])  # empirical per-episode difficulty (skewed), interpolated quantiles
        x = self._rand(gen, m) * (len(qs) - 1)
        i0 = x.astype(int) if self.xp is np else x.long()
        i1 = np.minimum(i0 + 1, len(qs) - 1) if self.xp is np else (i0 + 1).clamp(max=len(qs) - 1)
        b = qs[i0] + (x - i0) * (qs[i1] - qs[i0])
        med = qs[len(qs) // 2]
        # MAP shrinks the per-episode intercepts; b_scale (fitted on the episode-level spread) undoes it
        self.b[ids] = med + self.dp.get("b_scale", 1.0) * (b - med) + self.dp.get("b_shift", 0.0)
        self.e[ids] = self._randn(gen, m)
        self.u[ids] = self._randn(gen, m, 4)
        self.wrong[ids] = False
        self.fresh[ids] = True
        self.fresh_g[ids] = True

    # ------------------------------------------------------------------ one camera frame
    def observe(self, joints_deg, vis: dict, gen=None, dt_frames: float = 1.0, speed_px=None):
        """One detector output per env for the current camera frame.

        joints_deg (n, 6) measured joints (camera pose by FK + the wrist mount); vis = Geometry.visibility(...) of the
        target brick at this frame; dt_frames = camera frames since the previous call (AR(1) correlation and the blur
        speed); speed_px (n,) overrides the image speed of the brick centre (px per frame).
        Returns {"detected", "conf", "box", "rel", "pos", "score"} (box / rel / pos are NaN where not detected)."""
        xp, p = self.xp, self.p
        uv = vis["uv"]
        if speed_px is None:
            if self.prev_uv is None:
                speed_px = self._z(self.n) + 5.0
            else:
                d = uv - self.prev_uv
                speed_px = (d[:, 0] ** 2 + d[:, 1] ** 2) ** 0.5 / max(dt_frames, 1e-6)
                speed_px = xp.where(self.fresh, self._z(self.n) + 5.0, speed_px)
        self.prev_uv = uv.copy() if xp is np else uv.clone()
        self.fresh = self.fresh & False
        sp = _clip(xp, xp.where(xp.isfinite(speed_px), speed_px, speed_px * 0 + 5.0), 0.0, 60.0) / 10.0
        area = _clip(xp, vis["area_px"], 300.0, 3e5)
        grip = joints_deg[:, 5]
        if self.prev_grip is None:
            gclose = grip * 0
        else:
            gclose = xp.where(self.fresh_g, grip * 0, (self.prev_grip - grip) / max(dt_frames, 1e-6))
        self.prev_grip = grip.copy() if xp is np else grip.clone()
        self.fresh_g = self.fresh_g & False
        gc = _clip(xp, gclose, 0.0, 6.0) / 3.0
        la = xp.log(area / 1e4)
        dd = (_clip(xp, vis["dist"], 0.03, 0.4) - 0.12) / 0.05
        X = xp.stack(
            [
                xp.ones_like(sp),
                vis["vis"],
                vis["occ_grip"],
                vis["oof"],
                la,
                la * la,
                sp,
                sp * sp,
                (grip < 12).to(sp.dtype) if xp is not np else (grip < 12).astype(float),
                dd,
                dd * dd,
                gc,
            ],
            -1,
        )
        r = self.dp["rho"] ** dt_frames
        self.e = self.e * r + self._randn(gen, self.n) * math.sqrt(max(1 - r * r, 0.0))
        score = X @ self.beta + self.b + self.e
        cp = p["conf"]
        conf = _clip(xp, X @ self.gamma + self._randn(gen, self.n) * cp["sigma"], self.threshold, cp["max"])
        visible = (vis["vis"] > 0.02) & vis["front"]
        bv = vis["box_vis"]
        # box noise: AR(1) standard normals scaled by mean / std linear in (1 - vis)
        rb = p.get("box_rho", 0.8) ** dt_frames
        self.u = self.u * rb + self._randn(gen, self.n, 4) * math.sqrt(max(1 - rb * rb, 0.0))
        o = 1.0 - vis["vis"]
        A = xp.stack([xp.ones_like(o), o], -1)  # (n, 2)
        mean = A @ self.box_mean.T  # (n, 4)
        std = _clip(xp, A @ self.box_std.T, 0.02, 2.0)
        z = mean + std * self.u
        dx, dy = _clip(xp, z[:, 0], -1.5, 1.5), _clip(xp, z[:, 1], -1.5, 1.5)
        lw, lh = _clip(xp, z[:, 2], -1.5, 1.5), _clip(xp, z[:, 3], -1.5, 1.5)
        bx0 = _clip(xp, bv[:, 0], 0.0, G.W)
        by0 = _clip(xp, bv[:, 1], 0.0, G.H)
        bx1 = _clip(xp, bv[:, 2], 0.0, G.W)
        by1 = _clip(xp, bv[:, 3], 0.0, G.H)
        w, h = bx1 - bx0, by1 - by0
        cx, cy = (bx0 + bx1) / 2 + dx * w, (by0 + by1) / 2 + dy * h
        w2, h2 = w * xp.exp(lw), h * xp.exp(lh)
        box = xp.stack(
            [
                _clip(xp, cx - w2 / 2, 0.0, G.W),
                _clip(xp, cy - h2 / 2, 0.0, G.H),
                _clip(xp, cx + w2 / 2, 0.0, G.W),
                _clip(xp, cy + h2 / 2, 0.0, G.H),
            ],
            -1,
        )
        big = ((box[:, 2] - box[:, 0]) >= p.get("min_side_px", 12)) & (
            (box[:, 3] - box[:, 1]) >= p.get("min_side_px", 12)
        )
        finite = xp.isfinite(box).all(-1)
        detected = visible & finite & big & (score > 0)
        box = xp.where(detected[:, None], xp.nan_to_num(box), box * float("nan"))
        rel, pos = self.geo.box_to_rel(
            xp.nan_to_num(box), joints_deg, BRICK_CENTRE_Z + p["plane_dz"], centre_z=BRICK_CENTRE_Z
        )
        nan = rel * float("nan")
        rel = xp.where(detected[:, None], rel, nan)
        pos = xp.where(detected[:, None], pos, nan)
        conf = xp.where(detected, conf, conf * 0)
        target_detected, target_rel, target_conf = detected, rel, conf
        wt = p.get("wrong_target")
        if wt:  # TargetTracker errors on real data: runs locked on another brick / plate edge, offset ~4-12 cm
            pe = 1.0 - (1.0 - wt["p_enter"]) ** dt_frames
            px = 1.0 - (1.0 - wt["p_exit"]) ** dt_frames
            u = self._rand(gen, self.n)
            new = ~self.wrong & (u < pe)
            self.wrong = (self.wrong & (u >= px)) | new
            r_ = xp.exp(math.log(wt["offset_median_m"]) + wt["offset_log_sd"] * self._randn(gen, self.n))
            a_ = self._rand(gen, self.n) * (2 * math.pi)
            off = xp.stack([r_ * xp.cos(a_), r_ * xp.sin(a_)], -1)
            self.w_off = xp.where(new[:, None], off, self.w_off)
            # base: the brick centre's own pixel through the same ray-plane estimator, then shifted on the table
            c = vis["uv"]
            _, base = self.geo.box_to_rel(
                xp.stack([c[:, 0], c[:, 1], c[:, 0], c[:, 1]], -1),
                joints_deg,
                BRICK_CENTRE_Z + p["plane_dz"],
                centre_z=BRICK_CENTRE_Z,
            )
            wpos = xp.stack([base[:, 0] + self.w_off[:, 0], base[:, 1] + self.w_off[:, 1], base[:, 2]], -1)
            p_tip, Rg = self.geo.tip(joints_deg)
            wrel = ((wpos - p_tip)[:, None, :] @ Rg)[:, 0, :]
            w = self.wrong
            detected = detected | w
            rel = xp.where(w[:, None], wrel, rel)
            pos = xp.where(w[:, None], wpos, pos)
            conf = xp.where(w, conf * 0 + wt.get("conf", 0.6), conf)
            box = xp.where(w[:, None], box * float("nan"), box)
        return {
            "detected": detected,
            "conf": conf,
            "box": box,
            "rel": rel,
            "pos": pos,
            "score": score,
            "wrong": self.wrong if wt else detected & False,
            "target_detected": target_detected,
            "target_rel": target_rel,
            "target_conf": target_conf,
        }


class _EnvDetector:
    """Shared part of the batched env adapters (torch): a 30 Hz camera (`tick`), latency, automatic per-env resets."""

    KEYS = ("detected", "conf", "box", "rel", "pos", "wrong", "vis", "occ_grip", "oof")

    def __init__(
        self,
        env,
        tip_local,
        params: dict | None = None,
        samples: int = 5,
        others: bool = True,
        threshold: float | None = None,
    ):
        p = params or load_params()
        prof = getattr(env, "profile", None)
        self.geo = G.Geometry(
            offsets_rad=list(prof.joint_offsets) if prof is not None else None,
            camera=p.get("camera", "physical"),
            torch_device=str(env.device),
            tip_local=tuple(float(v) for v in tip_local),
            samples=samples,
        )
        # the jaw fin occludes as the REAL camera sees it at the measured reading (geometry.JAW_REAL), not at the
        # twin's own jaw angle (~0.2 rad more open)
        self.model = DetectorModel(env.n, p, self.geo, threshold=threshold)
        self.others = others
        self.queue: list = []  # frames not yet delivered (latency)
        self.last = None
        self._ticked = False
        self._t_prev = None

    # subclasses: (joints (n, 6), brick pos (n, 3) env, R (n, 3, 3), size (n, 3), others tuple | None)
    def _state(self, env):
        raise NotImplementedError

    def _auto_reset(self, env):
        t = env.t.clone()
        if self._t_prev is None:
            self.model.reset(None, env.gen)
        else:
            ids = (t < self._t_prev).nonzero().flatten()
            if len(ids):
                self.reset(ids, env.gen)
        self._t_prev = t

    def reset(self, ids, gen=None):
        """New episodes for envs ids (called automatically when env.t goes back)."""
        self.model.reset(ids, gen)
        for o in self.queue + ([self.last] if self.last is not None else []):
            o["detected"][ids] = False
            o["rel"][ids] = 0.0

    def frame(self, env, dt_frames: float = 1.0) -> dict:
        import torch

        j, pos, R, size, oth = self._state(env)
        vis = self.geo.visibility(j, pos, R, size, oth)
        out = self.model.observe(j, vis, env.gen, dt_frames=dt_frames)
        out.update(vis=vis["vis"], occ_grip=vis["occ_grip"], oof=vis["oof"])
        out["rel"] = torch.nan_to_num(out["rel"], nan=0.0)
        return {k: out[k] for k in self.KEYS}

    def tick(self, env):
        """One wrist-camera frame (call every 30 Hz control tick); delivers the frame `latency` ticks old."""
        self._ticked = True
        self._auto_reset(env)
        self.queue.append(self.frame(env, 1.0))
        while len(self.queue) > self.model.latency:
            self.last = self.queue.pop(0)
        if self.last is None:
            self.last = {k: v.clone() for k, v in self.queue[0].items()}
            self.last["detected"][:] = False

    def output(self, env, dt_frames: float):
        if not self._ticked:  # no per-tick hook: one frame per observation, no latency
            self._auto_reset(env)
            self.last = self.frame(env, dt_frames)
        return self.last["rel"], self.last["detected"]


class GraspEnvDetector(_EnvDetector):
    """sim/twin/rl/grasp_env.py: `env.detector_model = det = GraspEnvDetector(env)` and chain `det.tick` into
    `env.tick_hook` (30 Hz camera, 1-frame latency). `det(env, rel_true) -> (det (n, 3), valid (n,))` = GraspEnv's
    hook contract: the brick centre - fingertip (Kin TIP_LOCAL) in gripper_link estimated from the simulated box,
    zero where not detected. The ACTIVE brick (env.kind: 2x2 / 2x4 / curved 2x2) is the target; the other two pool
    bodies occlude it when they are on the table. `det.last` holds conf / box / vis / occ_grip / oof for logging."""

    def __init__(self, env, **kw):
        super().__init__(env, env.kin.tip[:3].tolist(), **kw)

    def _state(self, env):
        import torch

        j = env.measured(noisy=False)
        o = env.scene.env_origins
        hgt = float(env.layout.target.size[2])
        # GraspEnv KINDS: 2x2 target, 2x4 box, curved-slope 2x2 (as its box; USD origin assumed at the centre)
        sizes = torch.tensor([[0.0318, 0.0318, hgt], [0.0636, 0.0318, hgt], [0.0318, 0.0318, hgt]], device=env.device)
        oth = None
        if self.others:
            ps, Rs, ss, vs = [], [], [], []
            for k, obj in enumerate(env.objs):
                if obj is None:
                    continue
                pk = obj.data.root_pos_w - o
                ps.append(pk)
                Rs.append(G.quat_xyzw_to_R(obj.data.root_quat_w))
                ss.append(sizes[k].expand(env.n, 3))
                vs.append((env.kind != k) & (pk[:, 0] < 0.6))  # parked bodies sit at x = 0.85
            oth = (torch.stack(ps, 1), torch.stack(Rs, 1), torch.stack(ss, 1), torch.stack(vs, 1))
        return j, env.bpos_w() - o, G.quat_xyzw_to_R(env.bquat_w()), sizes[env.kind], oth

    def __call__(self, env, rel_true):
        return self.output(env, float(getattr(env, "h", 1)))


class ResidualEnvDetector(_EnvDetector):
    """sim/residual_env ResidualEnv (and TwinEnv's env.residual): replaces `ResidualEnv.detector()` on the instance,
    no edit of env.py:  `env.residual.detector = ResidualEnvDetector(env.residual)`  (TwinEnv: after construction).
    Returns (det (n, 3), valid (n,) float) like the original; the target is env.brick, env.distractors occlude.
    Called once per observation (dt_frames = ticks per policy step, default 1); call `.tick(renv)` every 30 Hz tick
    instead for the camera rate + latency."""

    def __init__(self, renv, dt_frames: float = 1.0, **kw):
        import sys

        tip = getattr(renv, "tip_local", None)
        if tip is None:
            tip = getattr(sys.modules.get("env"), "TIP_LOCAL", (0.008, 0.0, -0.090))
        tip = tip.tolist() if hasattr(tip, "tolist") else list(tip)
        super().__init__(renv, tip[:3], **kw)
        self.renv, self.dt_frames = renv, dt_frames

    def _state(self, env):
        import torch

        j = env.measured(noisy=False)
        o = env.scene.env_origins
        dev = env.device
        size = torch.as_tensor(env.layout.target.size, dtype=torch.float32, device=dev).expand(env.n, 3)
        oth = None
        if self.others and env.distractors:
            ps = torch.stack([d.data.root_pos_w - o for d in env.distractors], 1)
            Rs = torch.stack([G.quat_xyzw_to_R(d.data.root_quat_w) for d in env.distractors], 1)
            ss = torch.stack(
                [
                    torch.as_tensor(sp.size, dtype=torch.float32, device=dev).expand(env.n, 3)
                    for sp in env.layout.distractors[: len(env.distractors)]
                ],
                1,
            )
            oth = (ps, Rs, ss, (ps[..., :2].abs() < 0.6).all(-1))
        return j, env.brick.data.root_pos_w - o, G.quat_xyzw_to_R(env.brick.data.root_quat_w), size, oth

    def __call__(self):
        rel, det = self.output(self.renv, self.dt_frames)
        return rel, det.float()


def _clip(xp, a, lo, hi):
    return np.clip(a, lo, hi) if xp is np else a.clamp(lo, hi)
