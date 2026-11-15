"""Image-based pick episodes in TwinEnv (one station, cameras via VisualSync) with a visual prior (MolmoAct2), the
deployed controller timing, the residual gate and the residual's offset application. Shared by molmo_eval.py
(prior-alone / fixed policies) and train_image.py (DICE-RL with the visual prior).

Episode protocol = the batched grasp env's: held-out (or train) split of the cal2 episodes (2:1, seed 0), start
0.7-1.2 s before the recorded gripper close from the recorded joints, 2x2 target at the close-frame fingertip
perturbed by xy +-2.5 cm / yaw +-25 deg, 3 s, success = lift >= 3 cm held 0.5 s. Controller: 30-action chunks,
RTC re-plan when 15 remain, inference delay d ticks (sim time pauses during inference; the old plan's next d actions
run first, as on the robot), LeRobot clamp 10. For a fine-tune .ckpt the prompt is a trained one,
"pick up the <colour> brick", with the twin's target recoloured to that colour; top camera cropped to 560x480.

Residual: offsets of the fingertip (gripper frame, +-2.5 cm), wrist flex / roll (+-10 deg) and gripper (+-30 units),
rate-limited (5 mm / 3 deg per tick) and IK-converted exactly as GraspEnv.apply_residual; gated by
is_pick_prompt(task), faded out over 6 ticks at the hand-off (success).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from sim.twin.rl.gate import is_pick_prompt
from sim.twin.rl.grasp_env import DATA, DEPLOY_MAX_STEP, PICK_PROMPT, START_TYPES, close_frame
from sim.twin.rl.priors import FT_BRICK_COLOURS, FT_PICK, FT_TOP_CROP, TWIN_PALETTE, MolmoAct2Prior

H_DEC = 5  # ticks per residual decision
START_MIX = (0.3, 0.25, 0.15, 0.2, 0.1)  # GraspCfg.start_mix
# the deployed controller's ready pose (8k fine-tune finetune_cal2_456.json) and the deploy-like horizon: MolmoAct2
# closes on a clock 1.5-2.5 s after it and regrasps every ~2.4 s, so 12 s allows several attempts
READY_POSE = (26.3, -51.8, 30.1, 73.1, 15.9, 37.7)
READY_HORIZON = 330  # 11 s: end only on success (or a wrong brick) or this timeout
STALE_FIRST = 12  # deployed controller: the arm waits for the first inference, then its first ~12 actions are stale
REAL_BRICKS = Path(__file__).resolve().parents[1] / "out/failure_match/real_brick_xy.json"
# The 8k fine-tune's held-out episodes (molmoact2_ft_8k/plan.json validation_episodes; canonical cal2 indices: their
# frame total 9,726 = plan train_frames 95,656 subtracted from the 105,382 used, exactly). The image-path TEST pool;
# the 8k trained on the rest (409) and never saw 77 / 131 / 284 (dropped there, excluded here too).
FT_PLAN = Path(__file__).resolve().parents[1] / "data/models/molmoact2_ft_8k/plan.json"
FT_VALIDATION = (3, 12, 13, 15, 16, 44, 47, 52, 57, 71, 80, 82, 102, 111, 112, 113, 115, 120, 126, 142, 144, 174, 176,
                 216, 218, 231, 260, 281, 290, 304, 305, 311, 330, 335, 350, 360, 361, 364, 375, 378, 382, 428, 434,
                 451)
FT_DROPPED = (77, 131, 284)
# Clutter of the ablation-(c) scenes (failure-match verdict: the real 43 % first-attempt rate matches 2+ distractors):
# distractor count P(0-1) 0.15, P(2-3) 0.45, P(4-7) 0.40; with p_near, 1-2 of them sit next to the target
CLUTTER = {0: 0.075, 1: 0.075, 2: 0.225, 3: 0.225, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1}
DEPLOY_NOISE_SEED = 0  # the export's flow noise is a fixed-seed op: the same start noise on every request
OBS_DIM = 25


def ft_split(keep) -> dict:
    """train / test pools: test = the 8k's validation episodes (checked against its plan.json when present)."""
    if FT_PLAN.exists():
        import json as _json

        val = tuple(_json.loads(FT_PLAN.read_text())["validation_episodes"])
        if val != FT_VALIDATION:
            raise ValueError(f"{FT_PLAN} validation_episodes differ from image_task.FT_VALIDATION")
    return {"train": [e for e in keep if e not in FT_VALIDATION], "test": [e for e in keep if e in FT_VALIDATION]}


class ImageGraspTask:
    def __init__(self, snap=None, K=4, delay=10, horizon_s=3.0, seed=1234, instruction=None, crop_top="auto",
                 randomizer=None, detector: str = "yolo", res_xy_m: float = 0.035, res_z_m: float = 0.02,
                 veto: bool = False, top_target: bool = False, inject: bool = False, inject_lateral: float = 1.0,
                 max_step: tuple | None = DEPLOY_MAX_STEP, controller_delay: bool = True, finetune: bool = True,
                 fixed_noise: bool = True, servo_params: str | None = None, clutter: dict | None = CLUTTER,
                 p_near: float = 0.5, drift: dict | None = None):
        """detector: "yolo" = sim/twin/perception's model of the real wrist YOLO (ResidualEnvDetector, 30 Hz ticks,
        fin occlusion), "sim" = the residual env's noise model, "none" = no detector (ablation).
        max_step: the deployed controller's slew (body deg, gripper units) per tick on the final command (residual
        first, then the slew, then the LeRobot clamp), on every start type; None = off.
        controller_delay: the play model's +1 tick on pan / wrist_flex / wrist_roll (deployed-controller traces).
        finetune: the prior is the cal2 fine-tune (.ckpt: top crop 560x480, "pick up the <colour> brick" prompts);
        False = zero-shot. Raises when it does not match snap.
        fixed_noise: candidate j starts the flow from seed DEPLOY_NOISE_SEED + j on every request (candidate 0 = the
        deployed export's fixed noise); False = fresh seeded noise per request.
        servo_params: servo model file (None = TwinEnvCfg's default, the twin servo fit).
        clutter: distractor-count distribution {n: p} (None = the real-data statistics, scene_stats.json); p_near:
        per episode, 1-2 table distractors moved next to the target (side-aligned, 0-10 mm gap, as GraspEnv).
        drift: deployment-drift DR (drawn with phys_dr, i.e. training + the DR eval; nominal eval = none):
          "joint_sd_deg": (6,) per-episode joint calibration error e ~ N(0, sd): the arm physically sits at
            command + e while every robot-side reading (prior state, residual obs, slew) is physical - e;
          "top" / "wrist": (pos half-range m, rot half-range deg) camera extrinsic jitter (uniform, as the camera DR
            of sim/twin/env/randomize.py, focal off), applied after the visual DR (it sets these camera poses)."""
        import sys

        from sim.twin.env.twin_env import TwinEnv, TwinEnvCfg

        repo = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repo / "sim/residual_env"))
        from bases import load_episodes

        from sim.twin.rl.kin import Kin

        is_ckpt = bool(snap) and str(snap).endswith(".ckpt")
        if bool(finetune) != is_ckpt:
            raise ValueError(f"finetune={finetune} but snap={snap!r} is {'a' if is_ckpt else 'not a'} Studio .ckpt: "
                             "pass the fine-tune .ckpt with finetune=True, or a zero-shot snapshot with False")
        if not REAL_BRICKS.exists():
            raise FileNotFoundError(f"{REAL_BRICKS} (the ready-start brick positions) is missing: refusing to change "
                                    "the placement silently")
        kw = {} if servo_params is None else {"servo_params": servo_params}
        kw["play_cfg"] = {"controller_delay": bool(controller_delay)}  # belongs with the fitted servo file
        if randomizer is None:
            self.env = TwinEnv(TwinEnvCfg(cameras=("top", "wrist"), randomize=False, **kw))
        else:
            self.env = TwinEnv(TwinEnvCfg(cameras=("top", "wrist"), randomize=True, randomizers=[randomizer], **kw))
        self.servo_params = str(self.env.cfg.servo_params)
        if clutter is not None:
            self.env.scene_stats["distractor_count"] = {str(k): float(v) for k, v in clutter.items()}
        self.p_near = float(p_near)
        self.drift = dict(drift or {})
        self.joint_sd = torch.tensor(list(self.drift.get("joint_sd_deg") or [0.0] * 6), dtype=torch.float32)
        self.cal_err = torch.zeros(6)
        cams = {c: (float(v[0]), float(v[1]), 0.0) for c in ("top", "wrist") if (v := self.drift.get(c)) and any(v)}
        self.cam_drift, self.cam_drifted = None, False
        if cams:
            from sim.twin.env.randomize import TwinRandomizer

            off = {"enabled": False}
            self.cam_drift = TwinRandomizer({"camera": {"enabled": True, "pose": True, "focal": False,
                                                        "cams": list(cams), "half_range": cams, "scale": 1.0},
                                             "light": off, "material": off, "clutter": off, "image": off})
        self.K, self.delay, self.T = K, delay, int(horizon_s * 30)
        self.instruction = instruction
        self.finetune, self.fixed_noise = bool(finetune), bool(fixed_noise)
        if crop_top == "auto":
            crops = {"top": FT_TOP_CROP} if self.finetune else None
        else:
            crops = {"top": tuple(int(v) for v in crop_top.split(","))} if crop_top else None
        self.crops = crops
        self.prior = MolmoAct2Prior(snap=snap, crops=crops)
        R = self.R = self.env.residual
        self.dev = self.env.device
        self.max_step = None if not max_step else torch.tensor([max_step[0]] * 5 + [max_step[1]], device=self.dev)
        # ONE fingertip point with the residual env / physics layer (fin-ray TIP_LOCAL), see GraspEnv
        self.kin = Kin(self.env.profile.joint_offsets, self.dev,
                       tip_local=tuple(float(v) for v in self.env.residual.tip_local.tolist()))
        self.eps = load_episodes(DATA)
        keep = []
        for e, d in sorted(self.eps.items()):  # GraspEnv._load_pool's rule
            cf = close_frame(d["state"][:, 5], d["inspection_frame"])
            if not d["excluded"] and cf is not None and cf >= 37 and d["inspection_frame"] and e not in FT_DROPPED:
                keep.append(e)
        self.pool = ft_split(keep)
        self.rng = np.random.default_rng(seed)
        self.res_scale = torch.tensor([res_xy_m, res_xy_m, res_z_m, 10.0, 10.0, 30.0], device=self.dev)
        self.veto, self.n_veto = veto, 0
        # the loop's top-camera target (table xy, per-episode bias sd 8 mm + 2 mm per-frame noise, 3 % dropouts)
        self.top_target = top_target
        # real-failure injection on the prior (explicit flag): per-episode lateral bias of the commanded fingertip
        # (70 % of episodes, |d| ~ 20-32 mm, median ~22, random table direction, z +-5 mm) + 3 mm per-chunk jitter,
        # applied through the IK; time-locked close 1.5-2.5 s after the start (median 1.8 s) whatever the alignment,
        # 1 s closed, then regrasp every 2.4 s while nothing is held (the real 8k / 5.5k traces)
        self.inject, self.inject_lateral = inject, float(inject_lateral)
        import json as _json

        self.real_xy = [r["xy_m"] for r in _json.loads(REAL_BRICKS.read_text())]  # real deploy runs (not episodes)
        self.phys_drawn = False
        self.obs_dim = OBS_DIM + (4 if top_target else 0)
        self.rate = torch.tensor([0.005] * 3 + [3.0] * 2 + [1e9], device=self.dev)
        self.rest_z = R.table_z + R.layout.target.size[2] / 2 + 0.0005
        self.det_mode, self.det = detector, None
        if detector in ("yolo", "yolo_cons"):
            from sim.twin.perception import geometry as PG
            from sim.twin.perception.detector_model import ResidualEnvDetector

            class _Det(ResidualEnvDetector):  # Isaac Lab 3.0 data arrays -> torch before the perception geometry
                """The robot's wrist tracker (wrist_target.py default): it follows the brick NEAREST the fingertip,
                which can be a distractor; then last["wrong"] is set (wrong-brick lifts stay failures)."""
                nearest_is_target = True

                def _state(self, env):
                    o = env.scene.env_origins
                    dev = env.device
                    z4, z3 = torch.zeros(4, device=dev), torch.zeros(3, device=dev)
                    pos, q = env.brick.data.root_pos_w - o, env.brick.data.root_quat_w - z4  # proxy - tensor
                    size = torch.as_tensor(env.layout.target.size, dtype=torch.float32, device=dev).expand(env.n, 3)
                    tip = env.tip_w() - o
                    best = (pos - tip).norm(dim=-1)
                    is_t = torch.ones(env.n, dtype=torch.bool, device=dev)
                    for k, d in enumerate(env.distractors or []):
                        dp = d.data.root_pos_w - o
                        dist = (dp - tip).norm(dim=-1)
                        on = (dp[:, :2].abs() < 0.6).all(-1) & (dist < best)  # parked bodies sit off the table
                        ds = torch.as_tensor(env.layout.distractors[k].size, dtype=torch.float32, device=dev)
                        pos = torch.where(on[:, None], dp, pos)
                        q = torch.where(on[:, None], d.data.root_quat_w - z4, q)
                        size = torch.where(on[:, None], ds.expand(env.n, 3) + z3, size)
                        best = torch.where(on, dist, best)
                        is_t = is_t & ~on
                    self.nearest_is_target = is_t
                    return env.measured(noisy=False), pos, PG.quat_xyzw_to_R(q), size, None

                def frame(self, env, dt_frames):
                    f = super().frame(env, dt_frames)  # _state above ran inside: nearest_is_target is this frame's
                    f["wrong"] = f["wrong"] | ~self.nearest_is_target
                    return f

            self.det = R.detector = _Det(R, dt_frames=1, others=False)
            if detector == "yolo_cons":  # conservative: more misses near the close (b_shift -0.3 per episode)
                orig = self.det.reset

                def reset_cons(ids, gen=None):
                    orig(ids, gen)
                    self.det.model.b[ids] += -0.3

                self.det.reset = reset_cons

    # ------------------------------------------------------------------ episode
    def reset(self, pool="test", n_ep=0, randomize=False, start: str | None = None, play_dr: bool = False,
              phys_dr: bool = False):
        """start: near | approach | frame0 | perturbed | recovery (None: sampled from START_MIX), as GraspEnv; or
        "ready": the deployed ready pose with a 12 s horizon (deploy-like protocol of the failure-match agent).
        play_dr: draw the play model's parameters (training); False = the fitted nominal values (evaluation).
        phys_dr: draw the physics DR (physics.apply: gripper effort, fin stiffness, frictions, brick mass); False =
        nominal (undoes an earlier draw)."""
        env, R, rng = self.env, self.R, self.rng
        ep = int(rng.choice(self.pool[pool]))
        d = self.eps[ep]
        cf = close_frame(d["state"][:, 5], d["inspection_frame"])
        S = d["state"]
        p, Rm = self.kin.fk(torch.tensor(S[cf], device=self.dev))
        self.start = start or START_TYPES[int(rng.choice(len(START_TYPES), p=np.asarray(START_MIX) / sum(START_MIX)))]
        dxy = rng.uniform(-0.025, 0.025, 2)
        dyaw = rng.uniform(-25, 25)
        if self.start == "recovery":  # the jaw closes where the brick was: 1.5-3 cm away now
            a = rng.uniform(0, 2 * math.pi)
            dxy = np.array([math.cos(a), math.sin(a)]) * rng.uniform(0.015, 0.03)
        brick = (float(p[0]) + dxy[0], float(p[1]) + dxy[1],
                 float(math.atan2(float(Rm[1, 0]), float(Rm[0, 0]))) + math.radians(dyaw))
        self.upside_down = False
        if self.start == "ready" and self.real_xy:  # the real deploy brick positions +-1 cm, 20 % upside-down
            bx, by = self.real_xy[int(rng.integers(len(self.real_xy)))]
            brick = (bx + rng.uniform(-0.01, 0.01), by + rng.uniform(-0.01, 0.01), rng.uniform(-math.pi, math.pi))
            self.upside_down = bool(rng.random() < 0.2)
        t0 = {"ready": 0, "near": cf - int(rng.integers(21, 37)), "approach": cf - int(rng.uniform(0.5, 4.0) * 30),
              "frame0": 0, "perturbed": cf - int(rng.uniform(0.5, 4.0) * 30),
              "recovery": min(cf + 3, len(S) - 1)}[self.start]
        t0 = max(0, t0)
        j0 = np.array(READY_POSE if self.start == "ready" else S[t0], np.float32)
        # deployment drift: joint calibration error (robot frame = physical - e); none in the nominal eval
        self.cal_err = (torch.as_tensor(rng.normal(size=6), dtype=torch.float32) * self.joint_sd
                        if phys_dr else torch.zeros(6))
        if self.start == "perturbed":
            jp = j0.copy()
            jp[:5] += rng.uniform(-7.5, 7.5, 5)
            if float(self.kin.fk(torch.tensor(jp, device=self.dev))[0][2]) > self.R.table_z + 0.03:
                j0 = jp
        # full pick phase from this start: to the recorded close + 1.5 s (<= 8 s); recovery 2.5 s
        self.T = 75 if self.start == "recovery" else int(min(cf - t0 + 45, 240))
        if self.start == "ready":
            self.T = READY_HORIZON
        # TwinEnv's brick pool places the distractors / plates from the real-data statistics (1-8 bricks)
        env.reset(seed=ep, joint_deg=j0 + self.cal_err.numpy(), brick_pose=brick, render=False, randomize=randomize)
        self.cal_err = self.cal_err.to(self.dev)
        if self.cam_drift is not None and phys_dr:
            self.cam_drift(env, rng)
            self.cam_drifted = True
        elif self.cam_drifted:
            self.cam_drift.restore(env)
            self.cam_drifted = False
        self.n_near = self._near_neighbours() if rng.random() < self.p_near else 0
        if self.upside_down:  # 180 deg about the body x axis (box collider: same rest height)
            pose = torch.cat([R.brick.data.root_pos_w[:1] - torch.zeros(3, device=self.dev),
                              R.brick.data.root_quat_w[:1] - torch.zeros(4, device=self.dev)], -1)
            q = pose[0, 3:]
            x = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.dev)  # xyzw of 180 deg about x
            pose[0, 3:] = torch.stack([q[3] * x[0] + q[0] * x[3] + q[1] * x[2] - q[2] * x[1],
                                       q[3] * x[1] - q[0] * x[2] + q[1] * x[3] + q[2] * x[0],
                                       q[3] * x[2] + q[0] * x[1] - q[1] * x[0] + q[2] * x[3],
                                       q[3] * x[3] - q[0] * x[0] - q[1] * x[1] - q[2] * x[2]])
            R.brick.write_root_pose_to_sim(pose)
            R.brick.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.dev))
        from sim.twin.physics import apply as phys

        if phys_dr:
            self.phys_draw = phys.apply_physics_dr(R.scene, phys.sample_physics_dr(rng), [0])
            self.phys_drawn = True
        elif self.phys_drawn:
            self.phys_draw = phys.apply_physics_dr(R.scene, phys.nominal_physics(), [0])
            self.phys_drawn = False
        play = getattr(env, "play", None)
        if play is not None and play_dr:
            play.sample(torch.zeros(1, dtype=torch.long, device=self.dev), R.gen)
        elif play is not None:  # nominal play parameters (undo any earlier draw)
            for k, v in play.nominal.items():
                play.P[k][0] = v
        if self.instruction:
            self.task = self.instruction
        elif self.finetune:
            from sim.twin.visuals.twin_scene import set_brick_color

            colour = FT_BRICK_COLOURS[int(rng.integers(len(FT_BRICK_COLOURS)))]
            set_brick_color(0, "Brick", TWIN_PALETTE[colour])
            self.task = FT_PICK.format(colour=colour)
        else:
            self.task = PICK_PROMPT
        self.pick = is_pick_prompt(self.task)
        self.ep, self.n_ep = ep, n_ep
        self.b0 = R.brick_local()[0].clone()
        self.top_bias = torch.tensor([*rng.normal(0, 0.008, 2), 0.0], dtype=torch.float32, device=self.dev)
        # neighbours (the brick pool's distractors on the table): start poses, nearest gap between bounding circles
        o0 = env.scene.env_origins[0]
        self.nb0 = [(d.data.root_pos_w[0] - o0).clone() for d in R.distractors]
        self.nb0 = [p for p in self.nb0 if abs(float(p[0])) < 1.0 and abs(float(p[1])) < 1.0]
        gaps = [float(torch.linalg.norm(p[:2] - self.b0[:2])) - 0.045 for p in self.nb0]
        self.nn_gap = max(0.0, min(gaps)) if gaps else 1.0
        self.plan: list = []
        self.t = self.hold = self.c_streak = self.fade = 0
        self.flags = dict.fromkeys(("success", "centred", "pushed", "tipped", "ever_grasp", "lost", "wrong"), False)
        self.o = torch.zeros(6, device=self.dev)
        self.o_tgt = torch.zeros(6, device=self.dev)
        self.o_hand = torch.zeros(6, device=self.dev)
        self.prev_r = torch.zeros(6, device=self.dev)
        self.phi = 0.0
        self.latent = None
        self.lat: list = []
        self.black = 0
        self.n_close, self.closing, self.attempt_at_success = 0, False, 0
        self.prev_cmd = None
        self.first_close_err_mm, self.first_close_empty = None, None
        self.inj = None
        if self.inject:
            biased = self.rng.random() < 0.7
            mag = float(np.clip(self.rng.normal(0.022, 0.005), 0.015, 0.032)) * self.inject_lateral if biased else 0.0
            ang = self.rng.uniform(0, 2 * math.pi)
            self.inj = {"bias_m": [mag * math.cos(ang), mag * math.sin(ang),
                                   float(self.rng.uniform(-0.005, 0.005)) if biased else 0.0],
                        "t_close": int(round(float(np.clip(self.rng.normal(1.8, 0.3), 1.5, 2.5)) * 30)),
                        "period": int(round(self.rng.uniform(2.3, 2.5) * 30)), "closed_ticks": 30}
        self.sum_abs_o = torch.zeros(6, device=self.dev)
        self.n_pick = 0

    def _near_neighbours(self) -> int:
        """Move 1-2 table distractors next to the target: aligned with one of its sides (yaw = target's, 0 / 90 deg),
        0-10 mm gap, +-1 cm along the side; kept only where it clears the other bricks and the plates."""
        from sim.twin.scene.bricks import SIZES, TABLE_Z, _quat

        env, R, rng = self.env, self.R, self.rng
        pool = env.bricks.current["distractors"] if env.bricks is not None else []
        cands = [(d, obj) for _n, d, obj in pool if not d.get("on_plate")]
        if not cands:
            return 0
        o = env.scene.env_origins[:1]
        tq = R.brick.data.root_quat_w[:1] - torch.zeros(4, device=self.dev)
        tyaw = 2.0 * math.atan2(float(tq[0, 2]), float(tq[0, 3]))
        txy = R.brick_local()[0, :2].detach().cpu().numpy()
        others = [(obj.data.root_pos_w[:1] - o)[0, :2].detach().cpu().numpy() for _n, _d, obj in pool]
        plates = [(R.bowl_xy_env[0, k].detach().cpu().numpy(), float(R.bowl_r[k])) for k in range(len(R.bowl_objs))
                  if abs(float(R.bowl_xy_env[0, k, 0])) < 1.0]
        th = np.array(SIZES["2x2"][:2]) / 2
        moved = 0
        for d, obj in cands[: int(rng.integers(1, 3))]:
            rot = int(rng.integers(2))  # 0: same yaw, 1: +90 deg
            dh = np.array(SIZES[d["kind"]][:2])[::-1 if rot else 1] / 2
            ax = int(rng.integers(2))
            sgn = float(rng.choice([-1.0, 1.0]))
            local = np.zeros(2)
            local[ax] = sgn * (th[ax] + dh[ax] + rng.uniform(0.0, 0.01))
            local[1 - ax] = rng.uniform(-0.01, 0.01)
            c, s = math.cos(tyaw), math.sin(tyaw)
            p = txy + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])
            me = (obj.data.root_pos_w[:1] - o)[0, :2].detach().cpu().numpy()
            if any(np.linalg.norm(p - q) < 0.045 for q in others if np.linalg.norm(q - me) > 1e-6) or \
                    any(np.linalg.norm(p - pc) < r + 0.03 for pc, r in plates):
                continue
            z = TABLE_Z + SIZES[d["kind"]][2] / 2 + 0.0005
            pose = torch.tensor([[p[0], p[1], z, *_quat(tyaw + rot * math.pi / 2, d.get("upside_down", False))]],
                                dtype=torch.float32, device=self.dev)
            pose[:, :3] += o
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.dev))
            others = [q if np.linalg.norm(q - me) > 1e-6 else p for q in others]
            moved += 1
        return moved

    def request(self):
        """Render, ask the prior for K chunks (RTC on the current plan's tail); returns (K, 30, 6) numpy."""
        imgs = self.env.render()
        for _ in range(3):  # a renderer starved of GPU memory returns black frames: retry, then count it
            if min(float(v.mean()) for v in imgs.values()) > 2.0:
                break
            imgs = self.env.render()
        self.black += int(min(float(v.mean()) for v in imgs.values()) <= 2.0)
        obs = {"images": imgs, "state": self.env.state() - self.cal_err.cpu().numpy(), "task": self.task}
        if self.plan:
            obs["prev_chunk"] = np.asarray(self.plan, np.float32)
            obs["inference_delay"] = self.delay
        k = self.K if self.pick else 1  # best-of-K only in the pick phase
        base = DEPLOY_NOISE_SEED if self.fixed_noise else 1000 * self.n_ep + 37 * self.t
        C = self.prior.sample([obs], k, seeds=[[base + j for j in range(k)]])[0]
        self.lat.append(self.prior.last_info[0]["latency_s"])
        if self.prior.last_latent is not None:
            self.latent = torch.as_tensor(self.prior.last_latent[0], device=self.dev)
        self.imgs = imgs
        return C

    def install(self, chunk):
        chunk = np.asarray(chunk, np.float32)
        if self.inj is not None and any(self.inj["bias_m"]):  # lateral bias + per-chunk jitter through the IK
            c = torch.as_tensor(chunk, device=self.dev)
            p, _ = self.kin.fk(c)
            d = torch.tensor(self.inj["bias_m"], device=self.dev) + torch.tensor(
                [*self.rng.normal(0, 0.003, 2), 0.0], dtype=torch.float32, device=self.dev)
            chunk = self.kin.ik_tip(c, p + d).cpu().numpy()
        new = list(chunk)
        if not self.plan and self.start == "ready":
            new = new[STALE_FIRST:]  # the arm held still during the first inference: its first actions are stale
        self.plan = (self.plan[: self.delay] + new[self.delay:]) if self.plan else new

    def need_request(self) -> bool:
        return len(self.plan) <= 15

    # ------------------------------------------------------------------ observations
    def obs(self) -> torch.Tensor:
        """GraspEnv.obs layout (25)."""
        R = self.R
        s = R.measured()[0] - self.cal_err  # the robot's (miscalibrated) reading
        self._s_last = s
        if self.det is not None or self.det_mode == "none":
            if self.det is not None:
                d, v = R.detector()
                det, valid = d[0], bool(v[0] > 0.5)
            else:
                det, valid = torch.zeros(3, device=self.dev), False
            q = R.measured(noisy=False)
            return self._obs_from(s, det, valid, q)
        q = R.measured(noisy=False)
        p, Rm = self.kin.fk(q)
        rel = (Rm[0].T @ (R.brick_local()[0] - p[0]))
        dist = float(rel.norm())
        cosang = -float(rel[2]) / max(dist, 1e-4)
        valid = (dist < R.cfg.det_max_range_m and cosang > math.cos(math.radians(R.cfg.det_fov_deg / 2))) or dist < 0.04
        valid = valid and self.rng.random() > R.cfg.det_dropout
        det = (rel + torch.randn(3, device=self.dev) * R.cfg.det_noise_m) if valid else torch.zeros(3, device=self.dev)
        return self._obs_from(s, det, valid, q)

    def _obs_from(self, s, det, valid, q):
        load = (q[0, 5] - self.R.r[0, 5]) / 10.0
        tf = torch.tensor([self.t / 240.0], device=self.dev)  # time since start / max horizon (GraspEnv obs)
        parts = [s / 100.0, det * 10.0, torch.tensor([float(valid)], device=self.dev), load[None], tf,
                 self.o / self.res_scale, self.prev_r, torch.tensor([float(self.pick)], device=self.dev)]
        if self.top_target:  # relative to the fingertip, gripper frame (as GraspEnv top_target)
            p, Rm = self.kin.fk(q)
            noise = torch.tensor([*self.rng.normal(0, 0.002, 2), 0.0], dtype=torch.float32, device=self.dev)
            rel = Rm[0].T @ (self.b0 + self.top_bias + noise - p[0])
            ok = self.rng.random() > 0.03
            parts += [rel * 10.0 if ok else torch.zeros(3, device=self.dev),
                      torch.tensor([float(ok)], device=self.dev)]
        return torch.cat(parts)

    def plan_feat(self, chunk=None) -> torch.Tensor:
        """(30,) next H_DEC goals of a chunk (default: the current plan) relative to the last observed state, /20."""
        c = self.plan[:H_DEC] if chunk is None else list(chunk[:H_DEC])
        while len(c) < H_DEC:
            c.append(c[-1])
        c = torch.as_tensor(np.asarray(c, np.float32), device=self.dev)
        return ((c - self._s_last) / 20.0).flatten()

    # ------------------------------------------------------------------ residual + tick
    def set_residual(self, r: torch.Tensor):
        r = r.clamp(-1, 1) if self.pick else torch.zeros_like(r)
        self.o_tgt = r * self.res_scale
        pen = 0.01 * float(r.pow(2).mean()) + 0.02 * float((r - self.prev_r).pow(2).mean())
        self.prev_r = r
        return -pen

    def _apply(self, goal):
        if self.pick:
            lim = self.rate
            tgt = self.o_tgt
        else:  # hand-off: fade to zero over 6 ticks
            lim = torch.maximum(self.rate, self.o_hand.abs() / 6)
            tgt = torch.zeros_like(self.o)
        self.o = self.o + (tgt - self.o).clamp(-lim, lim)
        g = torch.as_tensor(goal, device=self.dev).clone()
        if float(self.o[:5].abs().sum()) > 1e-7:
            p, Rm = self.kin.fk(g)
            g2 = g.clone()
            g2[3:5] += self.o[3:5]
            g = self.kin.ik_tip(g2[None], (p + Rm @ self.o[:3])[None])[0]
        g[5] = (torch.as_tensor(goal[5], device=self.dev) + self.o[5]).clamp(0, 100)
        last = getattr(self.det, "last", None) if self.det is not None else None
        if self.veto and self.pick and last is not None and last.get("wrong") is not None and bool(last["wrong"][0]):
            meas_g = self.R.measured(noisy=False)[0, 5]  # close-veto: tracker locked on another brick -> no close
            if float(g[5]) < float(meas_g):
                g[5] = meas_g
                self.n_veto += 1
        if self.max_step is not None:  # the deployed controller's --max-step slew of the final command (as _step)
            prev = self.prev_cmd if self.prev_cmd is not None else self.R.measured(noisy=False)[0] - self.cal_err
            g = prev + (g - prev).clamp(-self.max_step, self.max_step)
            self.prev_cmd = g.clone()
        return g + self.cal_err  # robot frame -> physical joints

    def tick(self):
        """Execute the plan's next action (+ residual). Returns (reward, terminated)."""
        from env import _quat_apply_xyzw, _quat_conj_xyzw

        R, f = self.R, self.flags
        nxt = np.array(self.plan.pop(0), np.float32)
        if self.inj is not None and not self.flags["ever_grasp"]:  # time-locked close and regrasp clock
            ph = self.t - self.inj["t_close"]
            nxt[5] = 0.0 if ph >= 0 and ph % self.inj["period"] < self.inj["closed_ticks"] else max(float(nxt[5]), 45.0)
        goal = self._apply(nxt)
        gcmd = float(goal[5])  # close attempts: the commanded jaw falls below 15 after being open (> 30)
        if gcmd < 15 and not self.closing:
            self.n_close += 1
            self.closing = True
            if self.n_close == 1:  # lateral miss at the first close (brick centre vs fingertip, table plane)
                d = (self.R.brick.data.root_pos_w - self.R.tip_w())[0, :2]
                self.first_close_err_mm = round(float(torch.linalg.norm(d)) * 1e3, 1)
        elif gcmd > 30:
            if self.closing and self.n_close == 1 and self.first_close_empty is None:
                self.first_close_empty = not self.flags["ever_grasp"]
            self.closing = False
        self.env.step(goal.cpu().numpy(), render=False)
        self.t += 1
        if self.det is not None:
            self.det.tick(R)  # 30 Hz wrist camera + 1-frame latency
        if self.pick:
            self.sum_abs_o += self.o.abs()
            self.n_pick += 1
        bp = R.brick_local()[0]
        h = float(bp[2] - self.b0[2])
        near = float(torch.linalg.norm(R.brick.data.root_pos_w - R.tip_w(), dim=-1)[0]) < 0.03
        bz = _quat_apply_xyzw(R.brick.data.root_quat_w, torch.tensor([[0.0, 0, 1]], device=self.dev))[0]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(bz[2])))))
        qg = R.robot.data.body_quat_w[:, R.grip_body]
        rel = _quat_apply_xyzw(_quat_conj_xyzw(qg), R.brick.data.root_pos_w - R.robot.data.body_pos_w[:, R.grip_body])
        rel = rel[0]
        centred_now = abs(float(rel[1])) < 0.005 and -0.104 < float(rel[2]) < -0.068 and tilt < 15
        disp = float(torch.linalg.norm(bp[:2] - self.b0[:2]))
        rew = 0.0
        if not f["ever_grasp"] and disp > 0.015 and not f["pushed"]:
            f["pushed"] = True
            rew -= 0.3
        grasped = bool(R.grasped()[0])
        f["ever_grasp"] |= (h > 0.01 and near) or grasped
        term = False
        tilt0 = 180.0 - tilt if self.upside_down else tilt  # relative to the resting pose
        if tilt0 > 35 and h < 0.01 and not f["success"] and not f["tipped"]:
            f["tipped"] = True
            rew -= 0.5
            term = self.start != "ready"  # deploy-like protocol: end only on success / wrong brick / timeout
        if disp > 0.08 or float(bp[2]) < -0.05:
            f["lost"] = True
            term = term or self.start != "ready" or float(bp[2]) < -0.05
        o0 = self.env.scene.env_origins[0]
        live = [d for d in R.distractors if abs(float((d.data.root_pos_w[0] - o0)[0])) < 1.0]
        if not f["success"] and any(float((d.data.root_pos_w[0] - o0)[2]) - float(p[2]) > 0.03
                                    for d, p in zip(live, self.nb0, strict=False)):
            f["wrong"] = term = True  # a non-target brick lifted: failure
            rew -= 1.0
        lifted = h > 0.03 and near
        self.hold = self.hold + 1 if lifted else 0
        self.c_streak = self.c_streak + int(centred_now) if lifted else 0
        phi = 0.2 * min(max(h / 0.03, 0.0), 1.0) * near + 0.1 * grasped
        rew += phi - self.phi
        self.phi = phi
        if self.hold >= 15 and not f["success"]:
            f["success"] = True
            self.attempt_at_success = self.n_close
            f["centred"] = self.c_streak * 2 >= self.hold
            rew += 1.0 + 0.3 * f["centred"]
            self.o_hand = self.o.clone()
            self.pick = False  # hand-off: the controller switches to the bring prompt; the residual fades out
            term = True
        return rew, term or self.t >= self.T

    def outcome(self) -> str:
        f = self.flags
        return ("success" if f["success"] else "wrong" if f["wrong"] else "tipped" if f["tipped"] else
                "slipped" if f["ever_grasp"] else "pushed" if f["pushed"] else "missed")

    def close(self):
        self.prior.close()
        self.env.close()


def state_of(task: ImageGraspTask, proj: torch.Tensor | None) -> torch.Tensor:
    """Residual / critic state: the 25-d observation + the prior's pooled latent projected to proj_dim."""
    s = task.obs()
    if proj is None:
        return s
    z = task.latent @ proj if task.latent is not None else torch.zeros(proj.shape[1], device=task.dev)
    return torch.cat([s, z])


@torch.no_grad()
def decide(task: ImageGraspTask, agent, proj, C) -> int:
    """Best-of-K at a request: argmax_k min-Q(s, a_k, pi(s, a_k)); a_k = candidate k's first actions that will
    actually run (after the d delay ticks that the old plan still covers)."""
    s = state_of(task, proj)
    A = torch.stack([task.plan_feat(c[task.delay:] if task.plan else c) for c in C])[None]
    return int(agent.select(s[None], A)[0][0])
