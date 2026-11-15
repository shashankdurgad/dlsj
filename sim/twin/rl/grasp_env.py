"""Batched grasp-only residual env for the SO-101 twin (no cameras, privileged physics state for rewards / eval).

Built on the teammate's ResidualEnv (sim/residual_env/env.py: servo model, LeRobot clamp, measurement noise, load
proxy) with the twin calibration profile (sim/twin/calib/profile.py) and the twin physics hooks
(sim/twin/physics/apply.py: CAD finger pads, rigid jaw contact, Duplo masses / materials, plate colliders).

Episode = the PICK phase of one recorded cal2 episode: it starts 0.7-1.2 s before the recorded gripper-close frame
(validate.py's close rule) from the recorded joint state, with the target brick at the FK fingertip of the close
frame, perturbed by xy +-2.5 cm / yaw +-25 deg. Target kind 2x2 / 2x4 / curved 2x3 (LDraw 2302b) and resting pose
(up / upside-down / on its side / on its end) are sampled (GraspCfg.brick_mix, pose_mix; the real target-kind
distribution from sim/twin/scene/scene_stats.json when present). Horizon 3 s at 30 Hz.

Decisions are chunk-level (default 5 ticks = 6 Hz), like the deployed VLA controllers:
  1. `candidates()` -> K base chunks (n, K, h, 6): the recorded actions + this episode's VLA-like systematic error
     (task-space xyz offset through IK, wrist-yaw error, close-timing shift, optional hover) + a per-candidate draw
     (xyz jitter, close-timing jitter, action noise). With the real MolmoAct2 base the K candidates are K flow-noise
     seeds of `MolmoAct2Client.act(obs, seed=...)`.
  2. the learner picks one (best-of-K by min-Q) and outputs a residual r in [-1, 1]^6 for it;
  3. `step_chunk(k, r)` executes the chunk: r sets OFFSET TARGETS relative to the base (fingertip xyz in the gripper
     frame +-2.5 cm, wrist_flex / wrist_roll +-10 deg, gripper +-30 units); the applied offset moves toward the
     target at <= 5 mm / 3 deg per tick, is converted to joints by a 2-step damped-least-squares IK on
     pan / lift / elbow (the fingertip stays put under wrist corrections), and is added to the base goal BEFORE the
     LeRobot clamp and the servo model (ResidualEnv.apply_goal).

Residual gate (`pick_phase`): the residual is applied only while pick_phase is True. When the pick phase ends (held
lift >= 3 cm for 0.5 s here; on the robot also the controller switching prompts) the applied offset fades linearly
to 0 over `fade_ticks` and every later target is 0 (the base runs unmodified).

Observation (what the real robot has): `obs()` (n, 25) = measured joints, wrist-detector brick centre in the gripper
frame (true pose + 4 mm noise, 5 % dropouts, FOV gating; from FK of the measured joints), valid flag, load proxy
(measured - commanded gripper), time since start, applied residual offset, previous residual, pick_phase.
`cand_feat(c)` (n, K, 30) = each candidate's next 5 targets relative to the measured state. `priv()` = true brick
pose / contact state for analysis (not used by the critic: best-of-K runs the critic on the robot).
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "sim/residual_env"
for _p in (str(REPO), str(RES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
OUT = REPO / "sim/twin/out/rl"
SERVO_FIT = REPO / "sim/twin/servo_fit/servo_params_fit.json"  # gripper close rate fitted on real traces (K 0.40)
KINDS = ("2x2", "2x4", "curved_2x3")
# body box (x, y, z) per kind (m): LDraw sizes, long side along x
KIND_SIZE = ((0.0317, 0.0317, 0.0192), (0.0637, 0.0317, 0.0192), (0.0477, 0.0317, 0.0192))
POSES = ("up", "upside_down", "side", "end")
START_TYPES = ("near", "approach", "frame0", "perturbed", "recovery")
# body rotation (xyzw) of each resting pose before the yaw: identity, 180 deg about x, 90 deg about x, 90 deg about y
POSE_QUAT = ((0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0), (0.70710678, 0.0, 0.0, 0.70710678),
             (0.0, 0.70710678, 0.0, 0.70710678))
POSE_HEIGHT_AXIS = (2, 2, 1, 0)  # body axis that ends up vertical


def distractor_count_mix(n_max: int) -> list:
    """P(0..n_max distractors) from scene_stats.json's distractor_count (real frames), else uniform over 0-6."""
    f = REPO / "sim/twin/scene/scene_stats.json"
    try:
        dc = json.loads(f.read_text())["distractor_count"]
        p = [float(dc.get(str(k), dc.get(k, 0.0)) if isinstance(dc, dict) else 0.0) for k in range(n_max + 1)]
        if sum(p) > 0:
            return p
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return [1.0 if k <= 6 else 0.0 for k in range(n_max + 1)]


def target_kind_mix(default=(0.80, 0.18, 0.02)) -> tuple:
    """P(2x2, 2x4, curved 2x3) of the TARGET brick from the scene agent's real-frame stats (scene_stats.json)."""
    f = REPO / "sim/twin/scene/scene_stats.json"
    try:
        tk = json.loads(f.read_text())["target_kind"]
        return tuple(float(tk.get(k, 0.0)) for k in KINDS)
    except (OSError, KeyError, ValueError):
        return default
PICK_PROMPT = "pick up the brick and lift it up"  # the deployed MolmoAct2 pick prompt
FAIL = ("success", "missed", "pushed", "slipped", "tipped", "wrong")


@dataclass
class GraspCfg:
    num_envs: int = 512
    device: str = "cuda:0"
    data: str = str(DATA)
    # calibration: "twin" = sim/twin/calib/profile.twin_profile() (+ the physics layer's gripper bias), or a
    # residual-env calib.py profile name. joint_offsets_json overrides the 6 joint offsets (rad), e.g. the refit in
    # sim/twin/align/joint_fit/: a list, or a dict with "joint_offsets_rad" / "offsets_rad_for_sim" / "offsets_rad".
    profile: str = "twin"
    joint_offsets_json: str = ""
    physics: bool = True  # twin physics hooks (sim/twin/physics/apply.py)
    physics_params: str = ""
    # episode
    horizon_s: float = 3.0
    pre_close_s: tuple = (0.7, 1.2)
    chunk: int = 5
    brick_xy: float = 0.025
    brick_yaw_deg: float = 25.0
    brick_mix: tuple | None = None  # P(2x2), P(2x4), P(curved 2x3); None = target_kind_mix() (real frames)
    # P(up, upside-down, side, end): the grasp gate's hand labels of the 60 held-out demos (wrist frames,
    # sim/twin/grasp_gate/brick_labels.json): up 35 %, upside-down 45 %, side 20 %, no clear end. Side = rolled 90 deg
    # about the body x axis, so a 2x2 is squeezed across its 31.7 mm side (the gate's side_y). The box / hull
    # colliders have no studs, so an upside-down brick rests at the same height as an upright one.
    pose_mix: tuple = (0.35, 0.45, 0.20, 0.0)
    max_relative_target: float = 10.0  # LeRobot clamp as deployed
    # Defaults stay backward compatible (other callers, e.g. the grasp gate, pin them); the RL configs turn the v3
    # features on explicitly: --play 1 --meas_noise 0 --distractors 7 --plates 1 --start_mix 0.3,0.25,0.15,0.2,0.1
    meas_noise: bool = True  # Gaussian state noise (off when the play model's quantised encoder is used)
    play: bool = False  # sim/twin/dynamics/play.py: joint play / hold band / sag / quantised encoders, per-env DR
    # start states: P(near: 0.7-1.2 s before the close, approach: 0.5-4 s before, episode frame 0, perturbed: an
    # approach state + +-perturb_deg per body joint, recovery: the jaw just closed on nothing / brick knocked away)
    start_mix: tuple = (1.0, 0.0, 0.0, 0.0, 0.0)  # v3 RL configs: (0.3, 0.25, 0.15, 0.2, 0.1)
    approach_s: tuple = (0.5, 4.0)
    perturb_deg: float = 7.5
    after_close_s: float = 1.5  # horizon = time to the recorded close + this (capped at max_horizon_s)
    max_horizon_s: float = 8.0
    recovery_horizon_s: float = 2.5
    # clutter: 0-7 distractors (count distribution from scene_stats.json when present), near / touching the target,
    # on plates (41 % in the real frames) or on the table; plates on the table (the residual env's layout sampler)
    distractors: int = 0  # v3 RL configs: 7
    p_near: float = 0.25
    p_on_plate: float = 0.41
    plates: bool = False  # v3 RL configs: True
    # deployed controller: --max-step slew of the final command per 30 Hz tick (after the residual, before the
    # LeRobot clamp / servo), body joints deg / gripper units (failure_match twin_eval: 4 / 8); 0 = off (v1)
    max_step_deg: float = 0.0
    max_step_grip: float = 8.0  # DEPLOY_MAX_STEP[1]
    servo_params: str = str(SERVO_FIT)  # servo model (was sim/realism/servo_params.json, gripper K 0.2046)
    play_controller_delay: bool = True  # play model +1 tick on pan / wflex / roll: belongs with the servo fit
    p_row: float = 0.5  # of the near placements: aligned with the target's sides (rows / side by side), 0-10 mm gap
    # target identity: top-camera target position in the obs (+4: xyz rel. fingertip in the gripper frame, valid),
    # per-episode bias sd top_bias_m + per-frame noise sd top_noise_m, dropouts top_dropout
    top_target: bool = False
    top_bias_m: float = 0.008
    top_noise_m: float = 0.002
    top_dropout: float = 0.03
    w_wrong: float = 1.0  # lifting a non-target brick: failure, episode ends
    w_knock: float = 0.1  # per neighbour displaced > 1.5 cm
    veto: bool = False  # deploy safety rule: hold the jaw open while the wrist tracker is locked on a wrong brick
    end_on_success: bool = True  # training: terminate at success; eval: False (base continues, residual fades)
    # base error model (VLA-like). Per episode = systematic bias of the policy, per candidate = sample spread.
    errors: bool = True
    err_mode: str = "gauss"  # "gauss": N(0, err_xy_m) | "real": the real 8k empty-close miss (lateral 20-32 mm,
    #                          random direction, height sd 3 mm; the replay base closes on the recorded clock)
    err_xy_m: float = 0.010  # std (clipped at 2.5 std)
    err_z_m: float = 0.006
    err_yaw_deg: float = 8.0  # uniform +-
    close_jitter: int = 5  # +- ticks
    hover_p: float = 0.15  # stops 2-4 cm short above the brick
    hover_z: tuple = (0.02, 0.04)
    cand_xyz_m: float = 0.004  # per candidate, constant over its chunk
    cand_close_jitter: int = 2
    act_noise_deg: float = 0.3
    act_noise_grip: float = 1.0
    # residual
    res_xyz_m: float = 0.025  # lateral (gripper x / y) offset bound
    res_z_m: float | None = None  # along the fingers; None = res_xyz_m
    res_wrist_deg: float = 10.0
    res_grip: float = 30.0
    rate_xyz_m: float = 0.005  # per tick
    rate_wrist_deg: float = 3.0
    fade_ticks: int = 6
    # detector (ResidualEnv's model)
    det_noise_m: float = 0.004
    det_dropout: float = 0.05
    det_fov_deg: float = 70.0
    det_max_range_m: float = 0.35
    det_ablate: bool = False  # zero the detector (real-vs-nothing check)
    # reward
    w_success: float = 1.0
    w_centred: float = 0.3
    w_push: float = 0.3
    w_tip: float = 0.5
    w_table: float = 0.01
    w_reopen: float = 0.1
    w_res: float = 0.01
    w_jerk: float = 0.02
    hold_ticks: int = 15
    lift_m: float = 0.03


# The deployed controller's --max-step slew (smolvla_control / loop_control, run_loop.sh): the 8k real traces
# (real_rollouts pi05_traces 2026-11-14 15:18-15:29) record max_step 4.0 / max_step_gripper 8.0 and their per-tick
# command steps saturate at exactly 4 deg (body) and 8 units (gripper).
DEPLOY_MAX_STEP = (4.0, 8.0)
DEPLOY_CTRL = {"max_step_deg": DEPLOY_MAX_STEP[0], "max_step_grip": DEPLOY_MAX_STEP[1], "play_controller_delay": True}
# The RL training / evaluation preset with the robustness features on (the GraspCfg defaults stay v1-compatible).
PRESETS = {"v1": {},
           "v3": {"play": True, "meas_noise": False, "distractors": 7, "plates": True,
                  "start_mix": (0.3, 0.25, 0.15, 0.2, 0.1), **DEPLOY_CTRL},
           # v4: target identity + the real failure mode (lateral misses), wider lateral residual bound
           "v4": {"play": True, "meas_noise": False, "distractors": 7, "plates": True, "p_near": 0.5,
                  "start_mix": (0.3, 0.25, 0.15, 0.2, 0.1), "top_target": True, "err_mode": "real",
                  "res_xyz_m": 0.035, "res_z_m": 0.02, **DEPLOY_CTRL}}


def close_frame(grip, insp, thr=12.0, stay=15.0, open_min=15.0):
    """validate.py --grasp_rule gripper: first frame the gripper reading falls through 12 and stays <= 15."""
    g = grip[:insp] if insp else grip
    if len(g) == 0 or g.max() < open_min:
        return None
    for f in range(1, len(g)):
        if g[f - 1] > thr >= g[f] and (g[f:] <= stay).all():
            return f
    return None


def load_offsets(path: str) -> list:
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict):
        for k in ("joint_offsets_rad", "offsets_rad_for_sim", "offsets_rad"):
            if k in d:
                d = d[k]
                break
    assert isinstance(d, list) and len(d) == 6, f"{path}: expected 6 joint offsets (rad)"
    return [float(v) for v in d]


def load_profile(cfg: GraspCfg, phys=None):
    import calib

    if cfg.profile == "twin":
        from sim.twin.calib.profile import twin_profile

        prof = twin_profile()
    else:
        prof = calib.load_profile(cfg.profile)
    if phys is not None:
        phys.patch_profile(prof)  # gripper bias matching the CAD pads (0.105 rad)
    if cfg.joint_offsets_json:
        prof = replace(prof, joint_offsets=load_offsets(cfg.joint_offsets_json), name=prof.name + "+joint_fit")
    return prof


def launch():
    """Headless Isaac app without cameras (GB10 render flags harmless)."""
    from app import KIT_ARGS
    from isaaclab.app import AppLauncher

    return AppLauncher(headless=True, enable_cameras=False, kit_args=KIT_ARGS).app


def make(cfg: GraspCfg):
    """Scene + GraspEnv (the app must be running)."""
    import isaaclab.sim as sim_utils
    import scene as rscene
    from env import EnvCfg
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveScene

    phys = None
    if cfg.physics:
        from sim.twin.physics import apply as phys

        phys.load_params(cfg.physics_params or None)
    prof = load_profile(cfg, phys)
    rscene.ROBOT_USD = prof.robot_usd
    layout = rscene.load_layout(str(REPO / "sim/realism/scene_params.json"), "joe")
    t2 = rscene.Brick("d2x2", KIND_SIZE[0], (1.5, 1.5), (0.5, 0.5, 0.5), 0.00675)
    t4 = rscene.Brick("d2x4", KIND_SIZE[1], (1.5, 1.5), (0.5, 0.5, 0.5), 0.012)
    layout.distractors = [t2 if k < 5 else t4 for k in range(cfg.distractors)]  # slots 0-4 2x2, 5-6 2x4
    if not cfg.plates:
        layout.bowls = {"purple": layout.bowls["purple"]}  # one plate, parked off the table
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=cfg.device))
    scfg = rscene.build_scene_cfg(cfg.num_envs, layout, False, profile=prof)
    t = layout.target

    def park(k):
        return RigidObjectCfg.InitialStateCfg(pos=(0.85, -0.45 + 0.3 * k, -0.002 + t.size[2] / 2))

    # brick variants: a 2x4 box (patched by the physics layer like a distractor) and the curved-slope 2x2 USD
    scfg.distractor_2x4 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Brick2x4", spawn=sim_utils.CuboidCfg(
        size=KIND_SIZE[1], rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.012), collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.64, 0.83))), init_state=park(0))
    if phys is not None:
        scfg = phys.patch_scene_cfg(scfg, prof)
    if cfg.brick_mix is None:
        cfg.brick_mix = target_kind_mix()
    if cfg.brick_mix[2] > 0 and phys is not None:
        b = scfg.brick.spawn
        scfg.curved = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/BrickCurved", spawn=sim_utils.UsdFileCfg(
            usd_path=phys.brick_usd("curved_2x3"), rigid_props=b.rigid_props, collision_props=b.collision_props,
            mass_props=sim_utils.MassPropertiesCfg(mass=float(phys._p()["bricks"]["mass_kg"]["curved_2x3"])),
            physics_material=b.physics_material), init_state=park(1))
    scene = InteractiveScene(scfg)
    if phys is not None:
        print("physics post_spawn:", json.dumps(phys.post_spawn(scene))[:400], flush=True)
    else:
        import app as rapp

        rapp.finger_pads(cfg.num_envs)
    sim.reset()
    scene.reset()
    rcfg = EnvCfg(num_envs=cfg.num_envs, cameras=False, scene_params=str(REPO / "sim/realism/scene_params.json"),
                  calib="joe_cal2", servo_params=cfg.servo_params,
                  max_relative_target=cfg.max_relative_target, meas_noise=cfg.meas_noise, episode_s=cfg.horizon_s,
                  det_noise_m=cfg.det_noise_m, det_dropout=cfg.det_dropout, det_fov_deg=cfg.det_fov_deg,
                  det_max_range_m=cfg.det_max_range_m)
    env = GraspEnv(rcfg, cfg, sim, scene, layout, prof)
    if cfg.play:
        from sim.twin.dynamics.play import PlayCfg, install

        env.play_cfg = PlayCfg(dr=True, controller_delay=cfg.play_controller_delay)
        install(env, env.play_cfg)
    return env


DETECTORS = ("sim", "yolo", "yolo_cons", "yolo_mix", "none")


def attach_detector(env, mode: str = "yolo_mix"):
    """Wrist-detector model for the residual's observation (explicit, per config; GraspEnv's default stays "sim"):
    sim        the residual env's model (true pose + 4 mm noise, 5 % dropouts, FOV gate);
    yolo       sim/twin/perception's model of the REAL wrist YOLO (fin occlusion, 1-frame latency, 30 Hz), nominal;
    yolo_cons  the same, conservative (b_shift -0.3: more misses near the close);
    yolo_mix   per episode uniformly between nominal and conservative (training default);
    none       no detector (ablation / fallback: valid = 0, det = 0)."""
    if mode not in DETECTORS:
        raise ValueError(f"detector {mode!r} not in {DETECTORS}")
    env.g.det_ablate = mode == "none"
    if mode in ("sim", "none"):
        return None
    from sim.twin.perception.detector_model import GraspEnvDetector

    # others=False: GraspEnvDetector's occluder path feeds Isaac Lab 3.0 data arrays to a numpy routine (fails on
    # GPU); the inactive kind bodies are parked anyway and the distractor slots are not modelled as occluders yet
    det = GraspEnvDetector(env, others=False)
    lo, hi = {"yolo": (0.0, 0.0), "yolo_cons": (-0.3, -0.3), "yolo_mix": (-0.3, 0.0)}[mode]
    orig = det.reset

    def reset(ids, gen=None):  # per-episode difficulty shift (conservative <-> nominal)
        orig(ids, gen)
        if lo != hi or lo != 0.0:
            u = torch.rand(len(ids), device=env.device, generator=gen)
            det.model.b[ids] += lo + (hi - lo) * u

    det.reset = reset
    env.detector_model = det
    prev = env.tick_hook

    def tick(e):
        if prev is not None:
            prev(e)
        det.tick(e)

    env.tick_hook = tick
    return det


def _quat_mul_xyzw(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def _quat_z_xyzw(q):
    """World z-axis of a body (third column of its rotation)."""
    x, y, z, w = q.unbind(-1)
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)


from env import ResidualEnv, _quat_apply_xyzw, _quat_conj_xyzw  # noqa: E402

from sim.twin.contact.hook import rest_lift_m  # noqa: E402
from sim.twin.rl.gate import is_pick_prompt  # noqa: E402


class GraspEnv(ResidualEnv):
    def __init__(self, rcfg, gcfg: GraspCfg, sim, scene, layout, profile):
        super().__init__(rcfg, None, sim, scene, layout, gcfg.device)
        from sim.twin.rl.kin import Kin

        d = self.device
        self.g = g = gcfg
        # the twin profile (ResidualEnv loaded joe_cal2): offsets, gripper mapping, cameras
        self.profile = profile
        self.off = torch.tensor(profile.joint_offsets, dtype=torch.float32, device=d)
        self.g_scale, self.g_bias = float(profile.grip_scale), float(profile.grip_bias)
        # ONE fingertip point: ResidualEnv.tip_local = env.TIP_LOCAL as set by the physics layer's patch_profile
        # (fin-ray grasp point); used by tip_w(), the success checks, the detector model and this FK / IK
        self.kin = Kin(profile.joint_offsets, d, tip_local=tuple(float(v) for v in self.tip_local.tolist()))
        self.objs = [self.brick, scene["distractor_2x4"], scene["curved"] if "curved" in scene.rigid_objects else None]
        self.park_xy = torch.tensor([[0.85, -0.45], [0.85, -0.15], [0.85, 0.15]], device=d)
        self.h = g.chunk
        self.max_ticks = int(round(g.max_horizon_s * 30))  # time normaliser + upper bound
        self.T_env = torch.full((self.n,), int(round(g.horizon_s * 30)), dtype=torch.long, device=self.device)
        self.start_type = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.n_dis = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self._load_pool()
        n = self.n

        def z(*s, dt=torch.float32):
            return torch.zeros(n, *s, dtype=dt, device=d)

        def b():
            return z(dt=torch.bool)

        self.ep, self.t0, self.kind, self.gshift = (z(dt=torch.long) for _ in range(4))
        self.err_p, self.err_yaw, self.hover = z(3), z(), b()
        self.b0, self.o, self.o_tgt, self.o_hand = z(3), z(6), z(6), z(6)
        self.pose, self.bz0 = z(dt=torch.long), z(3)  # resting pose index, initial body z-axis (world)
        self.prev_cmd = z(6)  # last command sent (max_step slew state)
        self.kind_size = torch.tensor(KIND_SIZE, device=d)
        self.pose_quat = torch.tensor(POSE_QUAT, device=d)
        self.pose_axis = torch.tensor(POSE_HEIGHT_AXIS, device=d)
        self.prev_r, self.fade = z(6), z(dt=torch.long)
        self.pick = torch.ones(n, dtype=torch.bool, device=d)
        self.hold, self.c_streak = z(dt=torch.long), z(dt=torch.long)
        self.succ, self.centred, self.tipped, self.pushed, self.lost = b(), b(), b(), b(), b()
        self.ever_grasp, self.ever_lift, self.closing, self.done = b(), b(), b(), b()
        self.phi, self.min_grip, self.prev_goal_g = z(), z(), z()
        self.n_reopen, self.n_table, self.t_succ = z(), z(), z(dt=torch.long)
        self.sum_abs_o, self.n_pick_ticks = z(6), z()
        self.lift_diag = z(6)  # over lifted ticks: count, grasped, |y|<5mm, z in pad span, tilt<15, mean tilt
        rz = g.res_xyz_m if g.res_z_m is None else g.res_z_m
        self.res_scale = torch.tensor([g.res_xyz_m] * 2 + [rz] + [g.res_wrist_deg] * 2 + [g.res_grip], device=d)
        self.OBS_DIM = 25 + (4 if g.top_target else 0)
        self.top_bias = torch.zeros(self.n, 3, device=d)
        self.dis_xy0 = torch.zeros(self.n, max(1, g.distractors), 3, device=d)
        self.wrong = torch.zeros(self.n, dtype=torch.bool, device=d)
        self.knocked = torch.zeros(self.n, max(1, g.distractors), dtype=torch.bool, device=d)
        self.nn_gap = torch.full((self.n,), 1.0, device=d)  # nearest-neighbour gap (m) between bounding circles
        self.vetoed = torch.zeros(self.n, device=d)
        self.rate = torch.tensor([g.rate_xyz_m] * 3 + [g.rate_wrist_deg] * 2 + [1e9], device=d)
        self.cands = None
        self.dis_count_mix = torch.tensor(distractor_count_mix(g.distractors), device=self.device)
        self.tick_hook = None  # f(env) after every physics tick (recording for render_grasps.py)
        # integration points for sim/twin/perception (occlusion-aware detector) and sim/twin/dynamics (play / sag):
        self.detector_model = None  # f(env, rel (n,3) true brick centre, gripper frame) -> (det (n,3), valid (n,))
        self.goal_hook = None  # f(env, goal (n,6) ours) -> goal, applied after the residual, before clamp + servo

    def deploy_meta(self) -> dict:
        """What a deployed residual must reproduce (saved in every checkpoint, required by export_residual.py)."""
        from dataclasses import asdict

        from sim.twin.rl.provenance import provenance

        return {"res_scale": [float(v) for v in self.res_scale.tolist()],
                "rate": [float(v) for v in self.rate.tolist()[:5]] + [None],
                "tip_local": [float(v) for v in self.tip_local.tolist()],
                "joint_offsets": [float(v) for v in self.profile.joint_offsets],
                "decision_ticks": int(self.h), "fade_ticks": int(self.g.fade_ticks), "obs_dim": int(self.OBS_DIM),
                "grasp_cfg": asdict(self.g),
                "provenance": provenance(self.g.servo_params, getattr(self, "play_cfg", None))}

    # ------------------------------------------------------------------ data
    def _load_pool(self):
        from bases import load_episodes

        eps = load_episodes(self.g.data)
        pre_max = int(math.ceil(self.g.pre_close_s[1] * 30))
        keep = []
        for e, dd in sorted(eps.items()):
            cf = close_frame(dd["state"][:, 5], dd["inspection_frame"])
            if dd["excluded"] or cf is None or cf < pre_max + 1 or not dd["inspection_frame"]:
                continue
            keep.append((e, dd, cf))
        E, T = len(keep), max(len(dd["action"]) for _, dd, _ in keep)
        A = np.zeros((E, T, 6), np.float32)
        S = np.zeros((E, T, 6), np.float32)
        for i, (_, dd, _) in enumerate(keep):
            L = len(dd["action"])
            A[i, :L], A[i, L:] = dd["action"], dd["action"][-1]
            S[i, :L], S[i, L:] = dd["state"], dd["state"][-1]
        d = self.device
        self.ep_ids = [e for e, _, _ in keep]
        self.A, self.S = torch.tensor(A, device=d), torch.tensor(S, device=d)
        self.LEN = torch.tensor([len(dd["action"]) for _, dd, _ in keep], device=d)
        self.CF = torch.tensor([cf for _, _, cf in keep], device=d)
        p, R = self.kin.fk(self.S[torch.arange(E, device=d), self.CF])
        self.tip_close, self.yaw_close = p, torch.atan2(R[:, 1, 0], R[:, 0, 0])
        perm = np.random.default_rng(0).permutation(E)
        k = int(round(E * 2 / 3))
        self.pool = {"train": torch.tensor(np.sort(perm[:k]), device=d),
                     "test": torch.tensor(np.sort(perm[k:]), device=d),
                     "all": torch.arange(E, device=d)}
        print(f"[GraspEnv] {E} cal2 episodes with a close frame: train {k} / test {E - k} (2:1, seed 0)", flush=True)

    # ------------------------------------------------------------------ episode parameters
    def sample_params(self, m: int, gen: torch.Generator, pool: str = "train", errors: bool | None = None) -> dict:
        g, d = self.g, self.device
        errors = g.errors if errors is None else errors

        def u(*s):
            return torch.rand(*s, device=d, generator=gen)

        P = self.pool[pool]
        out = {"ep": P[torch.randint(len(P), (m,), device=d, generator=gen)],
               "pre": torch.randint(int(g.pre_close_s[0] * 30), int(g.pre_close_s[1] * 30) + 1, (m,), device=d,
                                    generator=gen),
               "dxy": (u(m, 2) * 2 - 1) * g.brick_xy, "dyaw": (u(m) * 2 - 1) * math.radians(g.brick_yaw_deg),
               "kind": torch.multinomial(torch.tensor(g.brick_mix, device=d).expand(m, 3), 1, generator=gen)[:, 0],
               "pose": torch.multinomial(torch.tensor(g.pose_mix, device=d).expand(m, 4), 1, generator=gen)[:, 0]}
        if self.objs[2] is None:
            out["kind"] = out["kind"].clamp(max=1)
        sd = torch.tensor([g.err_xy_m, g.err_xy_m, g.err_z_m], device=d)
        e_p = (torch.randn(m, 3, device=d, generator=gen) * sd).clamp(-2.5 * sd, 2.5 * sd)
        if g.err_mode == "real":
            a = u(m) * 2 * math.pi
            mag = 0.020 + 0.012 * u(m)
            e_p = torch.stack([torch.cos(a) * mag, torch.sin(a) * mag, torch.randn(m, device=d, generator=gen) * 0.003],
                              -1)
        hover = u(m) < g.hover_p
        e_p[:, 2] += hover * (g.hover_z[0] + u(m) * (g.hover_z[1] - g.hover_z[0]))
        out.update(err_p=e_p, err_yaw=(u(m) * 2 - 1) * g.err_yaw_deg, hover=hover,
                   gshift=torch.randint(-g.close_jitter, g.close_jitter + 1, (m,), device=d, generator=gen))
        if not errors:
            out.update(err_p=torch.zeros_like(e_p), err_yaw=torch.zeros(m, device=d),
                       hover=torch.zeros(m, dtype=torch.bool, device=d),
                       gshift=torch.zeros(m, dtype=torch.long, device=d))
        out["errors"] = torch.full((m,), bool(errors), device=d)
        out["start"] = torch.multinomial(torch.tensor(g.start_mix, device=d).expand(m, 5), 1, generator=gen)[:, 0]
        out["pre_s"] = g.approach_s[0] + u(m) * (g.approach_s[1] - g.approach_s[0])
        out["jit"] = (u(m, 5) * 2 - 1) * g.perturb_deg
        ang = u(m) * 2 * math.pi
        rec = out["start"] == 4  # recovery: the brick 1.5-3 cm from where the jaw closes, 30 % knocked on its side
        mag = 0.015 + u(m) * 0.015
        out["dxy"] = torch.where(rec[:, None], torch.stack([torch.cos(ang), torch.sin(ang)], -1) * mag[:, None],
                                 out["dxy"])
        out["pose"] = torch.where(rec & (u(m) < 0.3), torch.full_like(out["pose"], 2), out["pose"])
        cm = self.dis_count_mix.expand(m, -1)
        out["n_dis"] = torch.multinomial(cm, 1, generator=gen)[:, 0].clamp(max=g.distractors)
        out["dis_u"] = u(m, max(1, g.distractors), 6)  # placement randoms per slot
        return out

    # ------------------------------------------------------------------ reset
    def _place_bowls(self, ids, brick_xy):
        """Plates on the table (the residual env's sampler, clear of the target) or parked (plates=False)."""
        if self.g.plates:
            return ResidualEnv._place_bowls(self, ids, brick_xy)
        for obj in self.bowl_objs:
            m = len(ids)
            pose = torch.tensor([[1.6, 1.6, 0.003, 0, 0, 0, 1.0]], device=self.device).expand(m, 7).clone()
            pose[:, :3] += self.scene.env_origins[ids]
            obj.write_root_pose_to_sim(pose, env_ids=ids)

    def _place_distractors(self, ids, params, txy):
        """Slots k < n_dis: near / touching the target (p_near), on a plate (p_on_plate) or on the table >= 6 cm from
        the target; other slots parked. Random yaw, pose from pose_mix."""
        if not self.distractors or "n_dis" not in params:
            return
        g, d, m = self.g, self.device, len(ids)
        o = self.scene.env_origins[ids]
        U = params["dis_u"]
        pm = torch.tensor(g.pose_mix, device=d).cumsum(0)
        # footprint radius (half diagonal) of the target per env, for non-overlapping placement
        t_rad = self.kind_size[self.kind[ids], :2].norm(dim=-1) / 2
        placed = []  # (xy, radius, active) of the slots placed so far
        n_ok = torch.zeros(m, dtype=torch.long, device=d)
        nn = torch.ones(m, device=d)  # nearest-neighbour gap
        for k, obj in enumerate(self.distractors):
            u = U[:, k]
            act = k < params["n_dis"]
            ang = u[:, 0] * 2 * math.pi
            # rows / side by side: along one of the target's sides (its yaw + k * 90 deg), 0-10 mm gap
            row = u[:, 5] < g.p_row
            tyaw = self.yaw_close[self.ep[ids]]
            ang = torch.where(row, tyaw + torch.floor(u[:, 0] * 4) * (math.pi / 2), ang)
            size = torch.tensor(self.layout.distractors[k].size, device=d)
            rad = float(size[:2].norm() / 2)
            # near / touching: 2-20 mm gap between the bounding circles
            t_half = self.kind_size[self.kind[ids], 0] / 2  # side-by-side: half widths + gap
            gap = torch.where(row[:, None], 0.010 * u[:, 1:2], 0.002 + 0.018 * u[:, 1:2])
            reach = torch.where(row[:, None], (t_half + float(size[0]) / 2)[:, None], t_rad[:, None] + rad)
            near = txy + torch.stack([torch.cos(ang), torch.sin(ang)], -1) * (reach + gap)
            if len(self.bowl_objs):
                b = (u[:, 2] * len(self.bowl_objs)).long().clamp(max=len(self.bowl_objs) - 1)
                pc = self.bowl_xy_env[ids, b]
                pr = self.bowl_r[b] * 0.55 * u[:, 1]
                plate = pc + torch.stack([torch.cos(ang), torch.sin(ang)], -1) * pr[:, None]
            else:
                plate = near
            table = torch.stack([0.10 + 0.28 * u[:, 1], -0.25 + 0.45 * u[:, 2]], -1)
            away = table - txy
            dist = away.norm(dim=-1, keepdim=True).clamp(min=1e-4)
            clear = t_rad[:, None] + rad + 0.01
            table = torch.where(dist < clear, txy + away / dist * clear, table)
            mode = u[:, 3]
            xy = torch.where((mode < g.p_near)[:, None], near,
                             torch.where((mode < g.p_near + g.p_on_plate)[:, None], plate, table))
            on_plate = (mode >= g.p_near) & (mode < g.p_near + g.p_on_plate) & (len(self.bowl_objs) > 0)
            pose_i = (u[:, 4:5] > pm).sum(-1).clamp(max=3)
            # no interpenetration with the slots placed before (PhysX would launch them): drop the slot instead
            for pxy, prad, pact in placed:
                act = act & ~(pact & ((xy - pxy).norm(dim=-1) < prad + rad + 0.003))
            sep = (xy - txy).norm(dim=-1)
            act = act & (sep >= torch.where(row, t_half + float(size[0]) / 2 - 0.001, t_rad + rad))
            placed.append((xy, rad, act))
            n_ok += act.long()
            half = size[self.pose_axis[pose_i]] / 2
            z = self.table_z + half + 0.0015 + on_plate.float() * 0.006 + \
                (pose_i == POSES.index("upside_down")).float() * rest_lift_m("down")
            yaw = torch.where(row, tyaw, u[:, 5] * 2 * math.pi)  # rows keep the target's orientation
            qy = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)], -1)
            q = _quat_mul_xyzw(qy, self.pose_quat[pose_i])
            pos = torch.cat([xy, z[:, None]], -1)
            pos = torch.where(act[:, None], pos, torch.tensor([1.5 + 0.1 * k, 1.5, 0.02], device=d).expand(m, 3))
            self.dis_xy0[ids, k] = pos
            nn_k = torch.where(act, sep - torch.where(row, t_half + float(size[0]) / 2, t_rad + rad),
                               torch.full_like(sep, 1.0))
            nn = torch.minimum(nn, nn_k)
            obj.write_root_pose_to_sim(torch.cat([pos + o, q], -1), env_ids=ids)
            obj.write_root_velocity_to_sim(torch.zeros(m, 6, device=d), env_ids=ids)
        self.n_dis[ids] = n_ok
        self.nn_gap[ids] = nn.clamp(min=0.0)

    def observe(self):  # ResidualEnv.reset calls this; GraspEnv uses obs()
        return None

    def reset_envs(self, ids: torch.Tensor, params: dict):
        if len(ids) == 0:
            return
        d = self.device
        ep = params["ep"]
        st = params.get("start", torch.zeros_like(ep))
        cf = self.CF[ep]
        pre = torch.where(st == 1, (params["pre_s"] * 30).long(), params["pre"]) if "pre_s" in params else \
            params["pre"]
        t0 = (cf - pre).clamp(min=0)
        t0 = torch.where(st == 2, torch.zeros_like(t0), t0)  # episode frame 0
        t0 = torch.where(st == 4, torch.minimum(cf + 3, self.LEN[ep] - 1), t0)  # recovery: just after the close
        T = (cf - t0 + int(self.g.after_close_s * 30)).clamp(max=self.max_ticks)
        T = torch.where(st == 4, torch.full_like(T, int(self.g.recovery_horizon_s * 30)), T)
        T = torch.where(st == 0, torch.full_like(T, int(round(self.g.horizon_s * 30))), T)  # near: fixed 3 s
        self.T_env[ids], self.start_type[ids] = T, st
        self.ep[ids], self.t0[ids], self.kind[ids] = ep, t0, params["kind"]
        self.err_p[ids], self.err_yaw[ids] = params["err_p"], params["err_yaw"].float()
        self.hover[ids] = params["hover"]
        self.gshift[ids] = params["gshift"]
        self._errors_on = bool(params["errors"].any())
        xy = self.tip_close[ep, :2] + params["dxy"]
        # long-side kinds (2x4, curved 2x3) lie across the jaws so they close on the 31.7 mm width
        yaw = self.yaw_close[ep] + params["dyaw"] + (params["kind"] >= 1).float() * (math.pi / 2)
        pose_i = params.get("pose", torch.zeros_like(params["kind"]))
        self.pose[ids] = pose_i
        qyaw = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)], -1)
        quat = _quat_mul_xyzw(qyaw, self.pose_quat[pose_i])
        half_h = self.kind_size[params["kind"], self.pose_axis[pose_i]] / 2
        j0 = self.S[ep, t0].clone()
        if "jit" in params:  # perturbed starts, kept only where the fingertip stays >= 3 cm above the table
            jp = j0.clone()
            jp[:, :5] += params["jit"]
            ok = (st == 3) & (self.kin.fk(jp)[0][:, 2] > self.table_z + 0.03)
            j0 = torch.where(ok[:, None], jp, j0)
        super().reset(env_ids=ids, brick_pose=torch.cat([xy, yaw[:, None]], -1), joint_deg=j0)
        m = len(ids)
        zq = torch.zeros(m, 6, device=d)
        # upside-down bricks stand on their studs with the shell / studs colliders (else 4.4 mm inside the table)
        z_rest = self.table_z + half_h + 0.0005 + (pose_i == POSES.index("upside_down")).float() * rest_lift_m("down")
        ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=d).expand(m, 4)
        for k, obj in enumerate(self.objs):
            if obj is None:
                continue
            act = params["kind"] == k
            pose = torch.zeros(m, 7, device=d)
            pose[:, :2] = torch.where(act[:, None], xy, self.park_xy[k].expand(m, 2))
            pose[:, 2] = torch.where(act, z_rest, self.table_z + self.kind_size[k, 2] / 2 + 0.0005)
            pose[:, 3:] = torch.where(act[:, None], quat, ident)
            pose[:, :3] += self.scene.env_origins[ids]
            obj.write_root_pose_to_sim(pose, env_ids=ids)
            obj.write_root_velocity_to_sim(zq, env_ids=ids)
        self.b0[ids] = torch.cat([xy, z_rest[:, None]], -1)
        self.wrong[ids], self.knocked[ids], self.vetoed[ids] = False, False, 0.0
        self.nn_gap[ids] = 1.0
        self.top_bias[ids] = torch.randn(len(ids), 3, device=d, generator=self.gen) * self.g.top_bias_m
        self.top_bias[ids, 2] = 0.0
        self._place_distractors(ids, params, xy)
        play = getattr(self, "play", None)
        if play is not None:
            play.sample(ids, self.gen)
        self.bz0[ids] = _quat_z_xyzw(quat)
        for name in ("o", "o_tgt", "o_hand", "prev_r", "sum_abs_o", "lift_diag"):
            getattr(self, name)[ids] = 0
        for name in ("fade", "hold", "c_streak", "t_succ"):
            getattr(self, name)[ids] = 0
        for name in ("succ", "centred", "tipped", "pushed", "lost", "ever_grasp", "ever_lift", "closing", "done"):
            getattr(self, name)[ids] = False
        self.prompt = getattr(self, "prompt", PICK_PROMPT)
        self.prev_cmd[ids] = j0  # slew starts from the start pose
        self.pick[ids] = is_pick_prompt(self.prompt)  # residual gate = the prompt filter the robot uses
        self.phi[ids], self.n_reopen[ids], self.n_table[ids], self.n_pick_ticks[ids] = 0, 0, 0, 0
        self.min_grip[ids] = 100.0
        self.prev_goal_g[ids] = self.A[ep, t0, 5]

    # ------------------------------------------------------------------ active brick
    def bpos_w(self):
        p = self.objs[0].data.root_pos_w
        for k in (1, 2):
            if self.objs[k] is not None:
                p = torch.where((self.kind == k)[:, None], self.objs[k].data.root_pos_w, p)
        return p

    def bquat_w(self):
        q = self.objs[0].data.root_quat_w
        for k in (1, 2):
            if self.objs[k] is not None:
                q = torch.where((self.kind == k)[:, None], self.objs[k].data.root_quat_w, q)
        return q

    def brick_local(self):
        return self.bpos_w() - self.scene.env_origins

    def grasped(self):  # ResidualEnv.grasped on the active brick
        dist = torch.linalg.norm(self.bpos_w() - self.tip_w(), dim=-1)
        meas = self.measured(noisy=False)[:, 5]
        gvel = self.robot.data.joint_vel[:, self.jidx[5]].abs()
        squeeze = meas - (self.r[:, 5] + self.bias[5])
        return (dist < 0.02) & (meas > 2.0) & (meas < 15.0) & (squeeze > 1.0) & (gvel < 0.2)

    # ------------------------------------------------------------------ base candidates
    def base_targets(self, k: int = 1, tick=None):
        """(n, k, 6) recorded targets with this episode's close-timing shift (no other error)."""
        tau = (self.t if tick is None else tick).unsqueeze(1) + torch.arange(k, device=self.device)
        L = self.LEN[self.ep].unsqueeze(1)
        idx = torch.minimum(self.t0.unsqueeze(1) + tau, L - 1)
        a = self.A[self.ep.unsqueeze(1), idx]
        gi = (idx - self.gshift.unsqueeze(1)).clamp(min=0)
        gi = torch.minimum(gi, L - 1)
        a[..., 5] = self.A[self.ep.unsqueeze(1), gi, 5]
        return a

    def candidates(self, K: int, gen: torch.Generator | None = None) -> torch.Tensor:
        """K base chunks (n, K, h, 6) for the next decision (absolute goals, our units)."""
        g, d, n, h = self.g, self.device, self.n, self.h
        gen = gen or self.gen
        base = self.base_targets(h)  # (n, h, 6)
        c = base.unsqueeze(1).expand(n, K, h, 6).clone()
        if not self._errors_on:
            self.cands = c
            return c
        tau = self.t.unsqueeze(1) + torch.arange(h, device=d)
        L = self.LEN[self.ep].view(n, 1, 1)
        idx = torch.minimum(self.t0.view(n, 1, 1) + tau.unsqueeze(1), L - 1)  # (n, 1, h)
        js = torch.randint(-g.cand_close_jitter, g.cand_close_jitter + 1, (n, K, 1), device=d, generator=gen)
        gi = torch.minimum((idx - self.gshift.view(n, 1, 1) - js).clamp(min=0), L - 1)
        c[..., 5] = self.A[self.ep.view(n, 1, 1).expand(n, K, h), gi, 5]
        p0, _ = self.kin.fk(c)
        dp = self.err_p.view(n, 1, 1, 3) + torch.randn(n, K, 1, 3, device=d, generator=gen) * torch.tensor(
            [g.cand_xyz_m, g.cand_xyz_m, g.cand_xyz_m / 2], device=d)
        c[..., 4] += self.err_yaw.view(n, 1, 1)
        c = self.kin.ik_tip(c, p0 + dp)
        c[..., :5] += torch.randn(n, K, h, 5, device=d, generator=gen) * g.act_noise_deg
        c[..., 5] = (c[..., 5] + torch.randn(n, K, h, device=d, generator=gen) * g.act_noise_grip).clamp(0, 100)
        self.cands = c
        return c

    # ------------------------------------------------------------------ observations
    def detector_fk(self):
        """ResidualEnv.detector from FK of the measured (true) joints: brick centre - fingertip, gripper frame."""
        c = self.cfg
        p, R = self.kin.fk(self.measured(noisy=False))
        rel = (R.transpose(-1, -2) @ (self.brick_local() - p).unsqueeze(-1)).squeeze(-1)
        if self.detector_model is not None:
            det, valid = self.detector_model(self, rel)
            valid = valid.float() * (0.0 if self.g.det_ablate else 1.0)
            return det * valid.unsqueeze(-1), valid
        dist = torch.linalg.norm(rel, dim=-1)
        cosang = (-rel[:, 2]) / dist.clamp(min=1e-4)
        valid = (dist < c.det_max_range_m) & (cosang > math.cos(math.radians(c.det_fov_deg / 2)))
        valid |= dist < 0.04
        valid &= torch.rand(self.n, device=self.device, generator=self.gen) > c.det_dropout
        rel = rel + torch.randn(rel.shape, device=self.device, generator=self.gen) * c.det_noise_m
        if self.g.det_ablate:
            valid = torch.zeros_like(valid)
        return torch.where(valid.unsqueeze(-1), rel, torch.zeros_like(rel)), valid.float()

    def obs(self) -> torch.Tensor:
        s = self.measured()
        det, valid = self.detector_fk()
        load = ((self.measured(noisy=False)[:, 5] - self.r[:, 5]) / 10.0).unsqueeze(-1)
        tf = (self.t.float() / self.max_ticks).unsqueeze(-1)  # time since start / max horizon
        self._s_last = s
        parts = [s / 100.0, det * 10.0, valid[:, None], load, tf, self.o / self.res_scale, self.prev_r,
                 self.pick.float()[:, None]]
        if self.g.top_target:  # the loop's top-camera target (table position), relative to the fingertip
            p, R = self.kin.fk(self.measured(noisy=False))
            tgt = self.b0 + self.top_bias + torch.randn(self.n, 3, device=self.device, generator=self.gen) * \
                self.g.top_noise_m * torch.tensor([1.0, 1.0, 0.0], device=self.device)
            rel = (R.transpose(-1, -2) @ (tgt - p).unsqueeze(-1)).squeeze(-1)
            ok = torch.rand(self.n, device=self.device, generator=self.gen) > self.g.top_dropout
            parts += [torch.where(ok[:, None], rel * 10.0, torch.zeros_like(rel)), ok.float()[:, None]]
        ob = torch.cat(parts, -1)
        self.nonfinite = ~torch.isfinite(ob).all(-1)  # a PhysX blow-up: these envs are reset as failures
        return torch.nan_to_num(ob, nan=0.0, posinf=0.0, neginf=0.0)

    def cand_feat(self, c: torch.Tensor) -> torch.Tensor:
        """(n, K, h*6): candidate goals relative to the last observed (noisy) state, /20."""
        f = ((c - self._s_last.view(self.n, 1, 1, 6)) / 20.0).flatten(2)
        return torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    def priv(self) -> torch.Tensor:
        qg = self.robot.data.body_quat_w[:, self.grip_body]
        rel = _quat_apply_xyzw(_quat_conj_xyzw(qg), self.bpos_w() - self.robot.data.body_pos_w[:, self.grip_body])
        return rel

    # ------------------------------------------------------------------ residual application
    def _update_offsets(self):
        tgt = torch.where(self.pick[:, None], self.o_tgt, torch.zeros_like(self.o_tgt))
        lim = torch.where(self.pick[:, None], self.rate.expand(self.n, 6),
                          torch.maximum(self.rate, self.o_hand.abs() / max(1, self.g.fade_ticks)))
        self.o = self.o + (tgt - self.o).clamp(-lim, lim)

    def apply_residual(self, goal: torch.Tensor) -> torch.Tensor:
        o = self.o
        g2 = goal.clone()
        g2[:, 3:5] += o[:, 3:5]
        need = o[:, :5].abs().sum(-1) > 1e-7
        if bool(need.any()):
            p, R = self.kin.fk(goal)
            gi = self.kin.ik_tip(g2, p + (R @ o[:, :3].unsqueeze(-1)).squeeze(-1))
            g2 = torch.where(need[:, None], gi, g2)
        g2[:, 5] = (goal[:, 5] + o[:, 5]).clamp(0.0, 100.0)
        return g2

    # ------------------------------------------------------------------ step
    def step_chunk(self, k_sel: torch.Tensor, r: torch.Tensor):
        """Execute candidate k_sel (n,) with residual r (n, 6) in [-1, 1] for one chunk.
        Returns reward (n,), terminated (n,), truncated (n,) for the episode state at the chunk start."""
        n, g = self.n, self.g
        r = r.clamp(-1, 1)
        r = torch.where(self.pick[:, None], r, torch.zeros_like(r))  # residual gate
        chunk = self.cands[torch.arange(n, device=self.device), k_sel]
        self.o_tgt = r * self.res_scale
        alive = ~self.done
        rew = torch.zeros(n, device=self.device)
        rew -= alive * (g.w_res * r.pow(2).mean(-1) + g.w_jerk * (r - self.prev_r).pow(2).mean(-1))
        self.prev_r = r
        term = torch.zeros(n, dtype=torch.bool, device=self.device)
        for j in range(self.h):
            self._update_offsets()
            goal = self.apply_residual(chunk[:, j])
            if self.goal_hook is not None:
                goal = self.goal_hook(self, goal)
            if g.veto and self.detector_model is not None and getattr(self.detector_model, "last", None):
                # close-veto: the wrist tracker is locked on another brick -> keep the jaw where it is (no close)
                wrong = self.detector_model.last.get("wrong")
                if wrong is not None:
                    meas_g = self.measured(noisy=False)[:, 5]
                    hold = wrong.bool() & (goal[:, 5] < meas_g) & self.pick
                    goal[:, 5] = torch.where(hold, meas_g, goal[:, 5])
                    self.vetoed += hold.float()
            if g.max_step_deg:  # the deployed controller's slew (_step): residual first, then this, then the clamp
                lim = torch.tensor([g.max_step_deg] * 5 + [g.max_step_grip], device=self.device)
                goal = self.prev_cmd + (goal - self.prev_cmd).clamp(-lim, lim)
                self.prev_cmd = goal.clone()
            self.apply_goal(goal)
            self.physics()
            self.t += 1
            rt, tj = self._tick(goal)
            if self.tick_hook is not None:
                self.tick_hook(self)
            live = alive & ~term
            rew += rt * live
            term |= tj & live
        trunc = alive & ~term & (self.t >= self.T_env)
        self.done |= term | trunc
        return rew, term, trunc

    def _tick(self, goal):
        g, d = self.g, self.device
        o = self.scene.env_origins
        bw = self.bpos_w()
        bp = bw - o
        tip = self.tip_w()
        dist = torch.linalg.norm(bw - tip, dim=-1)
        height = bp[:, 2] - self.b0[:, 2]
        near = dist < 0.03
        grasped = self.grasped()
        lifted = (height > g.lift_m) & near
        bz = _quat_z_xyzw(self.bquat_w())  # tilt relative to the resting pose (side / end / upside-down starts)
        tilt = torch.rad2deg(torch.acos((bz * self.bz0).sum(-1).clamp(-1, 1)))
        rel = self.priv()  # brick centre in the gripper_link frame
        # centred: brick centre between the pads across the finger width (|y| < 5 mm), at pad height, held level
        # (ResidualEnv.grasped() is not required: it is true on only ~7 % of held ticks, see README)
        centred_now = (rel[:, 1].abs() < 0.005) & (rel[:, 2] > -0.104) & (rel[:, 2] < -0.068) & (tilt < 15)
        self.lift_diag += lifted[:, None] * torch.stack([torch.ones_like(tilt), grasped.float(),
                                                         (rel[:, 1].abs() < 0.005).float(),
                                                         ((rel[:, 2] > -0.104) & (rel[:, 2] < -0.068)).float(),
                                                         (tilt < 15).float(), tilt], -1)
        self.hold = torch.where(lifted, self.hold + 1, torch.zeros_like(self.hold))
        self.c_streak = torch.where(lifted, self.c_streak + centred_now.long(), torch.zeros_like(self.c_streak))
        succ_ev = (self.hold >= g.hold_ticks) & ~self.succ
        cen_ev = succ_ev & (self.c_streak * 2 >= self.hold)
        self.succ |= succ_ev
        self.centred |= cen_ev
        self.t_succ = torch.where(succ_ev, self.t, self.t_succ)
        disp = torch.linalg.norm(bp[:, :2] - self.b0[:, :2], dim=-1)
        push_ev = ~self.ever_grasp & (disp > 0.015) & ~self.pushed
        self.pushed |= push_ev
        self.ever_grasp |= grasped | ((height > 0.01) & near)
        self.ever_lift |= lifted
        tip_ev = (tilt > 35) & (height < 0.01) & ~self.succ & ~self.tipped
        self.tipped |= tip_ev
        blown = ~torch.isfinite(bp).all(-1) | ~torch.isfinite(self.robot.data.joint_pos).all(-1)
        lost = ((disp > 0.08) | (bp[:, 2] < -0.05) | blown) & ~self.lost
        self.lost |= lost
        # fixed-finger pad bottom below the table top (+1 mm)
        qg = self.robot.data.body_quat_w[:, self.grip_body]
        pad = self.robot.data.body_pos_w[:, self.grip_body] + _quat_apply_xyzw(
            qg, torch.tensor([-0.010, 0.0, -0.104], device=d).expand(self.n, 3))
        table = (pad[:, 2] - o[:, 2]) < self.table_z + 0.001
        self.n_table += table.float() * self.pick
        # re-grasp oscillation: commanded gripper re-opens >= 10 units after the close started
        gg = goal[:, 5]
        self.closing |= gg < 15
        self.min_grip = torch.where(self.closing, torch.minimum(self.min_grip, gg), self.min_grip)
        reopen = self.closing & (gg > self.min_grip + 10) & ~self.succ
        self.n_reopen += reopen.float()
        self.min_grip = torch.where(reopen, gg, self.min_grip)
        phi = 0.2 * (height / g.lift_m).clamp(0, 1) * near + 0.1 * grasped
        rew = (phi - self.phi) + g.w_success * succ_ev + g.w_centred * cen_ev - g.w_push * push_ev - g.w_tip * tip_ev \
            - g.w_table * table - g.w_reopen * reopen
        self.phi = phi
        # pick phase ends at success: residual fades to 0 (hand-off to the next stage)
        hand = succ_ev & self.pick
        self.o_hand = torch.where(hand[:, None], self.o, self.o_hand)
        self.pick &= ~hand
        self.sum_abs_o += self.o.abs() * self.pick[:, None]
        self.n_pick_ticks += self.pick.float()
        wrong_ev = torch.zeros_like(tip_ev)
        if self.distractors:
            dz = torch.stack([obj.data.root_pos_w[:, 2] - o[:, 2] for obj in self.distractors], 1)
            dxy = torch.stack([obj.data.root_pos_w[:, :2] - o[:, :2] for obj in self.distractors], 1)
            lifted_d = (dz - self.dis_xy0[:, : len(self.distractors), 2]) > g.lift_m
            wrong_ev = lifted_d.any(-1) & ~self.wrong & ~self.succ
            moved = ((dxy - self.dis_xy0[:, : len(self.distractors), :2]).norm(dim=-1) > 0.015) & \
                (self.dis_xy0[:, : len(self.distractors), 0] < 1.0)
            knock_ev = moved & ~self.knocked[:, : len(self.distractors)]
            self.knocked[:, : len(self.distractors)] |= moved
            rew = rew - g.w_knock * knock_ev.float().sum(-1)
        self.wrong |= wrong_ev
        rew = rew - g.w_wrong * wrong_ev
        term = tip_ev | lost | wrong_ev
        if g.end_on_success:
            term |= succ_ev
        return rew, term

    # ------------------------------------------------------------------ outcomes
    def outcome(self) -> torch.Tensor:
        """0 success, 1 missed, 2 pushed, 3 slipped, 4 tipped (FAIL names)."""
        oc = torch.ones(self.n, dtype=torch.long, device=self.device)
        oc = torch.where(self.pushed, torch.full_like(oc, 2), oc)
        oc = torch.where(self.ever_grasp | self.ever_lift, torch.full_like(oc, 3), oc)
        oc = torch.where(self.tipped, torch.full_like(oc, 4), oc)
        oc = torch.where(self.wrong, torch.full_like(oc, 5), oc)
        return torch.where(self.succ, torch.zeros_like(oc), oc)
