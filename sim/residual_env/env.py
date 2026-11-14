"""Vectorised residual-RL environment for the SO-101 brick task (Isaac Lab 3.0, N cloned envs on the GPU).

Units at the interface are OUR units (LeRobot degrees for the 5 arm joints, gripper 0-100 range units, old-calibration
frame = the training data frame), exactly what the real controller sends. Conversions to sim radians follow Joe's
harness: q = deg2rad(d) + joint_offsets; gripper q = units/100 * upper_limit.

Realism layers (all from data, see sim/realism/):
* servo: commanded goal -> per-joint first-order response r += clip(K * (goal - r), +-vmax/30) at 30 Hz (A1 fit:
  K, velocity limits), plus the measured steady-state (gravity) bias; a stiff PhysX PD drive tracks r so that contact
  still stops the jaw physically (holding a brick);
* LeRobot max_relative_target clamp of the goal around the measured state (default 10 units/tick, all joints);
* measurement noise on observed joints (A1 per-joint std);
* the frozen base policy's chunk arrives L seconds after its observation (sim keeps running), then replaces the plan
  from the current tick on (the deployed controller's `none` blend; optional linear crossfade);
* residual: bounded deltas on gripper (+-15 units) and wrist_flex / wrist_roll (+-5 deg) added after the base target,
  before the clamp;
* perception: a simulated wrist-camera brick detector (true pose + noise, valid only inside the wrist FOV cone);
  a gripper load proxy (commanded-minus-measured gripper, like Present_Current tracking the squeeze).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

JOINT_OFFSETS = [0.0, -0.2756362497772003, 0.056384559242785785, -0.45744995615777856, 0.0, 0.0]  # Joe's camera-fit


def resolve_joint_offsets(source: str) -> list[float]:
    """'joe' (camera-fit.json, the original) | 'fit_old' / 'fit_cal2' (sim/realism/joint_offsets_fit.json: data fit of the
    5 body-joint offsets with Joe's top camera fixed; use fit_old for old-calibration data, fit_cal2 for cal2 states)."""
    if source in ("", "joe"):
        return list(JOINT_OFFSETS)
    import json as _json
    from pathlib import Path as _P

    for cand in (_P(__file__).resolve().parent / "joint_offsets_fit.json",
                 _P(__file__).resolve().parents[1] / "realism" / "joint_offsets_fit.json"):
        if cand.exists():
            fit = _json.loads(cand.read_text())
            return list(fit[source.removeprefix("fit_")]["offsets_rad_for_sim"])
    raise FileNotFoundError("joint_offsets_fit.json not found next to env.py or in sim/realism/")
GRIPPER_UPPER = math.radians(100.000046)
TIP_LOCAL = (0.008, 0.0, -0.090)  # grasp point: centre of a 31.8 mm brick held against the fixed finger pad
# Gripper units: Joe's conversion (q = units/100 * upper) puts the sim jaw on a 31.8 mm brick at ~12.5 units, the real
# gripper holds it at 5.75-5.89 (A1/A2). With the finger pads the sim jaw stalls on the brick at 8.2 deg. Calibrated offset: sim jaw degrees = units + GRIP_OFFSET_UNITS.
GRIP_OFFSET_UNITS = 2.4  # measured: pads stall on a 31.8 mm brick at 8.2 deg sim = real hold 5.8 units
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@dataclass
class EnvCfg:
    num_envs: int = 8
    physics_hz: int = 120
    control_hz: int = 30
    episode_s: float = 30.0
    cameras: bool = False  # only the VLA base needs images
    overview: bool = False  # 1280x720 overview camera for demo videos (needs the app launched with cameras)
    scene_params: str = ""  # sim/realism/scene_params.json (A2)
    camera_source: str = "joe"  # "joe" (camera-fit.json, default) | "a2_refined" (rejected by A2; opt-in only)
    joint_offsets: str = "fit_old"  # data-fitted (sim/realism/joint_offset_fit.md); "joe" = original camera-fit; "fit_cal2" for cal2 states
    # Calibration profile (calib.py): "joe" | "fit_old" | "fit_cal2" | "joe_cal2". Empty = use `joint_offsets` as the
    # profile name (back-compatible). joe_cal2 also sets cameras, inspect camera, gripper scale and robot USD.
    calib: str = "joe_cal2"  # default: cal2 = the current robot calibration and the cal2 datasets
    target_size: tuple | None = None  # override the target brick size (m), e.g. (0.064, 0.032, 0.0192) for a 2x4
    servo_params: str = ""  # sim/realism/servo_params.json (A1)
    max_relative_target: float = 10.0  # LeRobot clamp, units per tick (all joints incl. gripper)
    meas_noise: bool = True
    # base policy timing
    base_latency_s: float = 0.074  # SmolVLA OV on the XPS; MolmoAct2 ~0.34, pi0.5 INT8 ~0.15
    base_chunk_exec: int = 15  # request a new chunk when <= this + latency ticks remain (RTC horizon)
    crossfade_ticks: int = 0
    # residual
    residual_scale: tuple = (15.0, 5.0, 5.0)  # gripper units, wrist_flex deg, wrist_roll deg
    residual_joints: tuple = (5, 3, 4)
    plan_obs_ticks: int = 5
    # detector noise model (assumption: YOLO box centre error ~4 mm at wrist range, 5% dropouts)
    det_noise_m: float = 0.004
    det_dropout: float = 0.05
    det_fov_deg: float = 70.0
    det_max_range_m: float = 0.35
    # rewards
    w_reach: float = 0.5
    w_grasp: float = 2.0
    w_lift: float = 4.0
    w_present: float = 4.0
    w_place: float = 8.0
    w_time: float = 0.005
    w_residual: float = 0.01
    w_jerk: float = 0.005
    w_clamp: float = 0.01
    w_drop: float = 4.0
    lift_height: float = 0.03
    instructions: dict = field(default_factory=lambda: {
        "grasp": "pick up the brick", "present": "bring the brick to the inspection camera",
        "place": "place the brick in the {bowl} bowl"})


def load_servo(path: str):
    """(K, vmax_per_tick, bias, noise) tensors over the 6 joints from A1's servo_params.json (defaults if missing)."""
    import json
    from pathlib import Path

    K = [0.30, 0.25, 0.25, 0.29, 0.29, 0.20]
    vmax = [169, 227, 232, 169, 161, 121]
    bias = [-0.09, 0.66, 1.93, 0.22, -0.09, 1.23]
    noise = [0.13, 0.15, 0.25, 0.12, 0.08, 0.11]
    if path and Path(path).exists():
        p = json.loads(Path(path).read_text())["joints"]
        K = [p[j]["first_order_K_per_tick"] for j in JOINTS]
        vmax = [p[j]["velocity_limit_deg_s"] for j in JOINTS]
        bias = [p[j]["steady_state_bias_deg"] for j in JOINTS]
        noise = [p[j]["meas_noise_std"] for j in JOINTS]
    return K, [v / 30.0 for v in vmax], bias, noise


class ResidualEnv:
    """Batched env. `step(residual)` advances one 30 Hz control tick for all envs.

    Base policy: an object with `reset(env_ids)`, `request(env_ids, obs) -> (n, H, 6)` chunks in our units, and
    `needs_images` (bool). obs = {"state": (n,6) noisy ours, "images": {"wrist","top"} uint8 (n,480,640,3) or None,
    "instruction": [str]*n, "tick": (n,)}.
    """

    def __init__(self, cfg: EnvCfg, base, sim, scene, layout, device: str = "cuda:0"):
        self.cfg, self.base, self.sim, self.scene, self.layout, self.device = cfg, base, sim, scene, layout, device
        self.n = cfg.num_envs
        self.robot = scene["robot"]
        self.brick = scene["brick"]
        self.distractors = [scene[f"distractor{i}"] for i in range(len(layout.distractors))]
        self.bowl_objs = [scene[f"bowl_{b}"] for b in layout.bowls]
        self.substeps = cfg.physics_hz // cfg.control_hz
        self.max_ticks = int(cfg.episode_s * cfg.control_hz)
        d = device
        K, vmax, bias, noise = load_servo(cfg.servo_params)
        self.K = torch.tensor(K, device=d)
        self.vmax = torch.tensor(vmax, device=d)
        self.bias = torch.tensor(bias, device=d)
        self.noise = torch.tensor(noise, device=d) if cfg.meas_noise else torch.zeros(6, device=d)
        from calib import load_profile

        self.profile = load_profile(cfg.calib or cfg.joint_offsets)
        self.off = torch.tensor(self.profile.joint_offsets, device=d)
        self.g_scale, self.g_bias = float(self.profile.grip_scale), float(self.profile.grip_bias)
        self.jidx = [self.robot.joint_names.index(j) for j in JOINTS]
        self.grip_body = self.robot.body_names.index("gripper_link")
        self.tip_local = torch.tensor(TIP_LOCAL, device=d)
        self.bowl_names = list(layout.bowls)
        self.bowl_xy = torch.tensor([[b[0], b[1]] for b in layout.bowls.values()], device=d)
        self.bowl_xy_env = self.bowl_xy.unsqueeze(0).repeat(self.n, 1, 1)  # per-env plate centres (randomised)
        self.bowl_r = torch.tensor([b[2] for b in layout.bowls.values()], device=d)
        ins = self.profile.inspect
        self.inspect_cam = _cam_from_rows(np.array(ins.rows if ins else _INSPECT_ROWS), device=d,
                                          f_px=ins.f_px if ins else None)
        self.table_z = -0.002
        self.res_scale = torch.tensor(cfg.residual_scale, device=d)
        self.res_joints = list(cfg.residual_joints)
        self.lat_ticks = int(math.ceil(cfg.base_latency_s * cfg.control_hz))
        H = 64
        self.plan = torch.zeros(self.n, H, 6, device=d)  # plan[:, j] = target for tick (plan_t0 + j)
        self.plan_t0 = torch.zeros(self.n, dtype=torch.long, device=d)
        self.plan_len = torch.zeros(self.n, dtype=torch.long, device=d)
        self.pending = None  # (env ids, chunks, request ticks, arrival ticks)
        self.r = torch.zeros(self.n, 6, device=d)  # servo reference (ours)
        self.prev_res = torch.zeros(self.n, len(self.res_joints), device=d)
        self.t = torch.zeros(self.n, dtype=torch.long, device=d)
        self.phase = torch.zeros(self.n, dtype=torch.long, device=d)  # 0 grasp, 1 present, 2 place
        self.bowl = torch.zeros(self.n, dtype=torch.long, device=d)
        self.was_grasped = torch.zeros(self.n, dtype=torch.bool, device=d)
        self.present_count = torch.zeros(self.n, dtype=torch.long, device=d)
        self.stats = {"clamp_hits": torch.zeros(self.n, device=d)}
        self.reset_brick = torch.zeros(self.n, 3, device=d)
        self.gen = torch.Generator(device=d)
        self.gen.manual_seed(0)

    # ------------------------------------------------------------------ unit conversions
    def to_sim(self, ours: torch.Tensor) -> torch.Tensor:
        q = torch.deg2rad(ours) + self.off
        q[..., 5] = ours[..., 5] * self.g_scale + self.g_bias
        return q

    def to_ours(self, q: torch.Tensor) -> torch.Tensor:
        s = torch.rad2deg(q - self.off)
        s[..., 5] = (q[..., 5] - self.g_bias) / self.g_scale
        return s

    def measured(self, noisy: bool = True) -> torch.Tensor:
        s = self.to_ours(self.robot.data.joint_pos[:, self.jidx].clone())
        if noisy:
            s = s + torch.randn(s.shape, device=self.device, generator=self.gen) * self.noise
        return s

    # ------------------------------------------------------------------ geometry
    def tip_w(self) -> torch.Tensor:
        p = self.robot.data.body_pos_w[:, self.grip_body]
        q = self.robot.data.body_quat_w[:, self.grip_body]  # xyzw
        return p + _quat_apply_xyzw(q, self.tip_local.expand(self.n, 3))

    def brick_local(self) -> torch.Tensor:
        """Brick centre in the env frame."""
        return self.brick.data.root_pos_w - self.scene.env_origins

    def grasped(self) -> torch.Tensor:
        """Brick between the jaws (near the fingertip point) and the jaw stalled on it (open > 3 units, measured
        above the command = squeeze), like the controller's grasp check plus privileged contact geometry."""
        d = torch.linalg.norm(self.brick.data.root_pos_w - self.tip_w(), dim=-1)
        meas = self.measured(noisy=False)[:, 5]
        gvel = self.robot.data.joint_vel[:, self.jidx[5]].abs()
        squeeze = meas - (self.r[:, 5] + self.bias[5])  # held open by the brick beyond the servo's free bias
        return (d < 0.02) & (meas > 2.0) & (meas < 15.0) & (squeeze > 1.0) & (gvel < 0.2)

    def in_inspect_view(self, p_local: torch.Tensor) -> torch.Tensor:
        """Brick projects inside the inspect camera image, 0.08-0.45 m in front of it."""
        R, t, f = self.inspect_cam
        pc = (p_local - t) @ R  # camera frame (USD: -Z forward)
        z = -pc[:, 2]
        u = 320 + f * pc[:, 0] / z.clamp(min=1e-3)
        v = 240 - f * pc[:, 1] / z.clamp(min=1e-3)
        return (z > 0.08) & (z < 0.45) & (u > 40) & (u < 600) & (v > 30) & (v < 450)

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids: torch.Tensor | None = None, brick_pose: torch.Tensor | None = None,
              joint_deg: torch.Tensor | None = None):
        ids = torch.arange(self.n, device=self.device) if env_ids is None else env_ids
        m = len(ids)
        lay = self.layout
        if joint_deg is None:
            if lay.start_joint is not None:  # data-derived start pose distribution (clipped Gaussian)
                med, std, lo, hi = (torch.tensor(v, device=self.device) for v in lay.start_joint)
                j0 = (med + torch.randn(m, 6, device=self.device, generator=self.gen) * std * 0.5).clamp(lo, hi)
            else:
                j0 = torch.tensor(lay.init_joint_deg, device=self.device).expand(m, 6).clone()
                j0 += (torch.rand(m, 6, device=self.device, generator=self.gen) * 2 - 1) * torch.tensor(
                    lay.init_joint_jitter_deg, device=self.device)
        else:
            j0 = joint_deg.to(self.device).clone()
        q = self.robot.data.default_joint_pos[ids].clone()
        q[:, self.jidx] = self.to_sim(j0)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        self.robot.set_joint_position_target(q, env_ids=ids)
        self.r[ids] = j0
        origins = self.scene.env_origins[ids]
        if brick_pose is None:
            if lay.target_xy_std is not None:  # where the gripper actually grasped in 740 episodes
                lo, hi = (torch.tensor(v, device=self.device) for v in lay.target_xy_box)
                xy = (torch.tensor(lay.target.pos, device=self.device) + torch.randn(
                    m, 2, device=self.device, generator=self.gen) * torch.tensor(lay.target_xy_std, device=self.device)
                      ).clamp(lo, hi)
            else:
                xy = torch.tensor(lay.target.pos, device=self.device) + (
                    torch.rand(m, 2, device=self.device, generator=self.gen) * 2 - 1) * lay.target_xy_jitter
            yaw = (torch.rand(m, device=self.device, generator=self.gen) * 2 - 1) * lay.target_yaw_range
        else:
            xy, yaw = brick_pose[:, :2].to(self.device), brick_pose[:, 2].to(self.device)
        z = torch.full((m,), self.table_z + lay.target.size[2] / 2 + 0.0005, device=self.device)
        pose = torch.cat([origins + torch.stack([xy[:, 0], xy[:, 1], z], -1), _yaw_quat_xyzw(yaw)], -1)
        self.reset_brick[ids] = torch.stack([xy[:, 0], xy[:, 1], z], -1)  # env-frame target centre (for scripted bases)
        self.brick.write_root_pose_to_sim(pose, env_ids=ids)
        self.brick.write_root_velocity_to_sim(torch.zeros(m, 6, device=self.device), env_ids=ids)
        n_dis = torch.randint(lay.distractor_count[0], lay.distractor_count[1] + 1, (m,), device=self.device,
                              generator=self.gen)
        for k, dobj in enumerate(self.distractors):
            b = lay.distractors[k]
            if lay.distractor_xy is not None:
                dm, ds, dlo, dhi = (torch.tensor(v, device=self.device) for v in lay.distractor_xy)
                dxy = (dm + torch.randn(m, 2, device=self.device, generator=self.gen) * ds).clamp(dlo, dhi)
            else:
                dxy = torch.tensor(b.pos, device=self.device) + (
                    torch.rand(m, 2, device=self.device, generator=self.gen) * 2 - 1) * lay.distractor_xy_jitter
            # unused distractors are parked off the table (count per episode follows the data, 1-3)
            dxy = torch.where((k < n_dis).unsqueeze(-1), dxy, torch.tensor([1.5, 1.5], device=self.device))
            # keep distractors >= 6 cm from the target (no overlap at spawn)
            away = dxy - xy
            dist = torch.linalg.norm(away, dim=-1, keepdim=True).clamp(min=1e-4)
            dxy = torch.where(dist < 0.06, xy + away / dist * 0.06, dxy)
            dz = torch.full((m,), self.table_z + b.size[2] / 2 + 0.0005, device=self.device)
            dyaw = (torch.rand(m, device=self.device, generator=self.gen) * 2 - 1) * math.pi
            dobj.write_root_pose_to_sim(torch.cat([origins + torch.stack([dxy[:, 0], dxy[:, 1], dz], -1),
                                                   _yaw_quat_xyzw(dyaw)], -1), env_ids=ids)
            dobj.write_root_velocity_to_sim(torch.zeros(m, 6, device=self.device), env_ids=ids)
        self._place_bowls(ids, xy)
        self.plan[ids] = j0.unsqueeze(1).expand(m, self.plan.shape[1], 6)
        self.plan_t0[ids] = 0
        self.plan_len[ids] = 0
        self.t[ids] = 0
        self.phase[ids] = 0
        self.bowl[ids] = torch.randint(0, len(self.bowl_names), (m,), device=self.device, generator=self.gen)
        self.was_grasped[ids] = False
        self.present_count[ids] = 0
        self.prev_res[ids] = 0
        self.stats["clamp_hits"][ids] = 0
        self.pending = None
        if self.base is not None:
            self.base.reset(ids)
        return self.observe()

    def _place_bowls(self, ids, brick_xy):
        """Random non-overlapping plate layout per episode inside the data's plate-centre box, clear of the target
        brick and the robot base (the plate layout changed every few dozen episodes in the recordings)."""
        lay = self.layout
        if lay.plate_box is None:
            return
        lo = torch.tensor(lay.plate_box[0], device=self.device)
        hi = torch.tensor(lay.plate_box[1], device=self.device)
        m = len(ids)
        placed = []
        for k, obj in enumerate(self.bowl_objs):
            r = self.bowl_r[k]
            best = None
            for _ in range(40):  # rejection sampling, vectorised over envs
                c = lo + torch.rand(m, 2, device=self.device, generator=self.gen) * (hi - lo)
                okk = torch.linalg.norm(c - brick_xy, dim=-1) > r + 0.04
                okk &= torch.linalg.norm(c, dim=-1) > r + 0.10  # robot base
                for (pc, pr) in placed:
                    okk &= torch.linalg.norm(c - pc, dim=-1) > r + pr + 0.01
                best = c if best is None else torch.where(okk.unsqueeze(-1), c, best)
                if bool(okk.all()):
                    break
            placed.append((best, r))
            self.bowl_xy_env[ids, k] = best
            z = torch.full((m, 1), self.table_z + 0.003, device=self.device)
            quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(m, 4)
            obj.write_root_pose_to_sim(torch.cat([self.scene.env_origins[ids] + torch.cat([best, z], -1), quat], -1),
                                       env_ids=ids)

    # ------------------------------------------------------------------ base policy plumbing
    def instructions(self, ids) -> list[str]:
        ins = self.cfg.instructions
        out = []
        for i in ids.tolist():
            ph = int(self.phase[i])
            if ph == 0:
                out.append(ins["grasp"])
            elif ph == 1:
                out.append(ins["present"])
            else:
                out.append(ins["place"].format(bowl=self.bowl_names[int(self.bowl[i])]))
        return out

    def overview_frames(self, ids) -> torch.Tensor:
        """(m, 720, 1280, 3) uint8 RGB from the overview camera (cfg.overview)."""
        self.sim.render()
        self.scene.update(0.0)
        return self.scene["overview_cam"].data.output["rgb"][ids, ..., :3]

    def images(self, ids):
        if not self.cfg.cameras:
            return None
        self.sim.render()
        self.scene.update(0.0)
        return {"wrist": self.scene["wrist_cam"].data.output["rgb"][ids, ..., :3],
                "top": self.scene["top_cam"].data.output["rgb"][ids, ..., :3]}

    def _maybe_request(self):
        """Ask the base for a chunk where the plan runs low (and none is in flight); deliver arrivals."""
        if self.base is None:
            return
        remaining = self.plan_t0 + self.plan_len - self.t
        if self.pending is not None:
            ids, chunks, t_req, t_arr = self.pending
            due = self.t[ids] >= t_arr
            if bool(due.all()):
                for k, i in enumerate(ids.tolist()):
                    self._install(i, chunks[k], int(t_req[k]))
                self.pending = None
        if self.pending is None:
            need = remaining <= (self.cfg.base_chunk_exec + self.lat_ticks)
            ids = torch.nonzero(need).flatten()
            if len(ids):
                obs = {"state": self.measured()[ids], "images": self.images(ids) if self.base.needs_images else None,
                       "instruction": self.instructions(ids), "tick": self.t[ids].clone(), "phase": self.phase[ids]}
                chunks = self.base.request(ids, obs)
                t_req = self.t[ids].clone()
                self.pending = (ids, chunks, t_req, t_req + self.lat_ticks)

    def _install(self, i: int, chunk: torch.Tensor, t_req: int):
        """Chunk action j is for tick t_req + j; drop the ones whose tick passed during inference."""
        now = int(self.t[i])
        start = max(0, now - t_req)
        fresh = chunk[start:]
        H = self.plan.shape[1]
        n = min(len(fresh), H)
        if self.cfg.crossfade_ticks and self.plan_len[i] > 0:
            old = self.target_at(i, now, n)
            c = min(self.cfg.crossfade_ticks, n)
            a = torch.linspace(1.0 / (c + 1), c / (c + 1), c, device=self.device).unsqueeze(-1)
            fresh = fresh.clone()
            fresh[:c] = (1 - a) * old[:c] + a * fresh[:c]
        self.plan[i, :n] = fresh[:n]
        self.plan[i, n:] = fresh[n - 1]
        self.plan_t0[i] = now
        self.plan_len[i] = n

    def target_at(self, i: int, tick: int, k: int = 1) -> torch.Tensor:
        j = torch.clamp(torch.arange(tick, tick + k, device=self.device) - self.plan_t0[i], 0, self.plan.shape[1] - 1)
        return self.plan[i, j]

    def base_targets(self, k: int = 1) -> torch.Tensor:
        """(n, k, 6) planned targets for the next k ticks."""
        j = (self.t - self.plan_t0).unsqueeze(1) + torch.arange(k, device=self.device).unsqueeze(0)
        j = j.clamp(0, self.plan.shape[1] - 1)
        return torch.gather(self.plan, 1, j.unsqueeze(-1).expand(self.n, k, 6))

    # ------------------------------------------------------------------ step
    def apply_goal(self, goal: torch.Tensor):
        """goal (n,6) ours -> LeRobot clamp around the measured state -> servo reference -> PD drive target."""
        meas = self.measured(noisy=False)
        m = self.cfg.max_relative_target
        clamped = torch.maximum(torch.minimum(goal, meas + m), meas - m)
        self.stats["clamp_hits"] += (clamped != goal).any(-1).float()
        step = torch.clamp(self.K * (clamped - self.r), -self.vmax, self.vmax)
        self.r = self.r + step
        q = self.robot.data.joint_pos_target.clone() if hasattr(self.robot.data, "joint_pos_target") \
            else self.robot.data.default_joint_pos.clone()
        q[:, self.jidx] = self.to_sim(self.r + self.bias)
        self.robot.set_joint_position_target(q)

    def physics(self):
        for _ in range(self.substeps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(1.0 / self.cfg.physics_hz)

    def step(self, residual: torch.Tensor | None = None, goal_override: torch.Tensor | None = None):
        """One control tick. residual (n, len(residual_joints)) in [-1, 1] (scaled), or goal_override (n,6) ours
        (open-loop replay: bypasses the base and the residual)."""
        if goal_override is None:
            self._maybe_request()
            goal = self.base_targets(1)[:, 0].clone()
        else:
            goal = goal_override.to(self.device).clone()
        res = torch.zeros(self.n, len(self.res_joints), device=self.device) if residual is None else \
            residual.clamp(-1, 1)
        goal[:, self.res_joints] += res * self.res_scale
        goal[:, 5] = goal[:, 5].clamp(0.0, 100.0)
        self.apply_goal(goal)
        self.physics()
        self.t += 1
        rew, info = self.reward(res)
        self.prev_res = res
        done = self.t >= self.max_ticks
        done |= info["placed"] | info["dropped"]
        return self.observe(), rew, done, info

    # ------------------------------------------------------------------ observations / rewards
    def detector(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Brick centre relative to the fingertip point in the gripper frame (m) + valid flag."""
        q = self.robot.data.body_quat_w[:, self.grip_body]
        rel_w = self.brick.data.root_pos_w - self.tip_w()
        rel = _quat_apply_xyzw(_quat_conj_xyzw(q), rel_w)
        dist = torch.linalg.norm(rel, dim=-1)
        # wrist camera looks roughly along the fingers (-z of gripper_link): FOV cone around -z
        cosang = (-rel[:, 2]) / dist.clamp(min=1e-4)
        valid = (dist < self.cfg.det_max_range_m) & (cosang > math.cos(math.radians(self.cfg.det_fov_deg / 2)))
        valid |= dist < 0.04  # between the jaws: still "seen" as in the real wrist view
        valid &= torch.rand(self.n, device=self.device, generator=self.gen) > self.cfg.det_dropout
        rel = rel + torch.randn(rel.shape, device=self.device, generator=self.gen) * self.cfg.det_noise_m
        return torch.where(valid.unsqueeze(-1), rel, torch.zeros_like(rel)), valid.float()

    def observe(self) -> torch.Tensor:
        s = self.measured()
        plan = self.base_targets(self.cfg.plan_obs_ticks) - s.unsqueeze(1)
        load = ((self.measured(noisy=False)[:, 5] - self.r[:, 5]) / 10.0).unsqueeze(-1)
        det, valid = self.detector()
        ph = torch.nn.functional.one_hot(self.phase, 3).float()
        tf = (self.t.float() / self.max_ticks).unsqueeze(-1)
        return torch.cat([s / 100.0, plan.flatten(1) / 20.0, load, det * 10.0, valid.unsqueeze(-1), ph, tf], -1)

    @property
    def obs_dim(self) -> int:
        return 6 + 6 * self.cfg.plan_obs_ticks + 1 + 3 + 1 + 3 + 1

    @property
    def act_dim(self) -> int:
        return len(self.res_joints)

    def reward(self, res: torch.Tensor):
        c = self.cfg
        tip = self.tip_w()
        bp = self.brick.data.root_pos_w
        d = torch.linalg.norm(bp - tip, dim=-1)
        # carried = brick off the table AND at the grasp point (it only rises if it is held between the pads)
        lifted = (bp[:, 2] - self.scene.env_origins[:, 2] > self.table_z + self.layout.target.size[2] / 2
                  + c.lift_height) & (d < 0.04)
        grasped = self.grasped() | lifted
        local = self.brick_local()
        in_view = self.in_inspect_view(local)
        speed = torch.linalg.norm(self.brick.data.root_lin_vel_w, dim=-1)
        near_insp = torch.zeros_like(in_view)
        if self.layout.inspection_tip is not None:
            near_insp = torch.linalg.norm((tip - self.scene.env_origins) - torch.tensor(
                self.layout.inspection_tip, device=self.device), dim=-1) < 0.05
        presenting = lifted & (in_view | near_insp) & (speed < 0.05)
        self.present_count = torch.where(presenting, self.present_count + 1, torch.zeros_like(self.present_count))
        # phase switching like the controller: grasp detected -> present; held still at the camera -> place
        self.phase = torch.where((self.phase == 0) & grasped, torch.ones_like(self.phase), self.phase)
        self.phase = torch.where((self.phase == 1) & (self.present_count >= 15), torch.full_like(self.phase, 2),
                                 self.phase)
        bowl_xy = self.bowl_xy_env[torch.arange(self.n, device=self.device), self.bowl]
        in_bowl = (torch.linalg.norm(local[:, :2] - bowl_xy, dim=-1) < self.bowl_r[self.bowl] * 0.85) & \
            (local[:, 2] < 0.04) & ~grasped & (speed < 0.05)
        placed = (self.phase == 2) & in_bowl
        dropped = self.was_grasped & ~grasped & (self.phase < 2) & (local[:, 2] < 0.02) & (speed < 0.05)
        self.was_grasped |= grasped
        jerk = (res - self.prev_res).pow(2).sum(-1)
        r = c.w_reach * torch.exp(-d / 0.05) * (self.phase == 0).float()
        r += c.w_grasp * grasped.float() + c.w_lift * lifted.float() + c.w_present * presenting.float() * 0.1
        r += c.w_place * placed.float() - c.w_drop * dropped.float()
        r -= c.w_time + c.w_residual * res.pow(2).sum(-1) + c.w_jerk * jerk
        r -= c.w_clamp * self.stats["clamp_hits"] / self.t.clamp(min=1).float()
        info = {"dist": d, "grasped": grasped, "lifted": lifted, "in_view": in_view, "presenting": presenting,
                "placed": placed, "dropped": dropped, "phase": self.phase.clone()}
        return r, info


# ---------------------------------------------------------------------- math helpers (xyzw quaternions)
_INSPECT_ROWS = [[-0.999530856702204, 0.03062787129655299, 0, 0],
                 [0.0066718593539116895, 0.21773401198020886, 0.9759853412422564, 0],
                 [0.029892353418890183, 0.9755274642606655, -0.2178362083773864, 0],
                 [0.19372239888012727, 0.30782613116852947, 0.09, 1]]


def _cam_from_rows(m, device, f_px=None):
    """USD row-vector camera matrix -> (R with camera axes as columns... used as p_cam = (p - t) @ R, t, f_px)."""
    R = torch.tensor(np.asarray(m)[:3, :3].T, dtype=torch.float32, device=device)  # columns = cam axes in world
    t = torch.tensor(np.asarray(m)[3, :3], dtype=torch.float32, device=device)
    f = f_px if f_px else 20.0 / 27.234043 * 640
    return R, t, f


def _yaw_quat_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return torch.stack([z, z, torch.sin(yaw / 2), torch.cos(yaw / 2)], -1)


def _quat_conj_xyzw(q):
    return torch.cat([-q[..., :3], q[..., 3:]], -1)



def _quat_apply_xyzw(q, v):
    xyz, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)
