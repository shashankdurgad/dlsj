"""TwinEnv: one SO-101 station in Isaac Lab 3.0 with the twin's look, driven like the real follower.

    from sim.twin.env import TwinEnv, TwinEnvCfg
    env = TwinEnv(TwinEnvCfg())                    # launches Isaac (headless, cameras) if not running yet
    obs = env.reset(seed=0)                        # randomize=None -> cfg.randomize (False: no DR hooks)
    obs, reward, terminated, truncated, info = env.step(action)   # one 30 Hz control tick
    env.close()

Conventions (the team's runtime LeRobot units, exactly the recorded `observation.state` / `action` of the cal2 data):
* state / action (6,) float32: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll in degrees of the
  cal2 calibration frame, gripper 0-100. Sim radians follow the joe_cal2 profile (q = deg2rad(d) + offsets,
  gripper q = units * scale + bias); `twin_profile()` adds the measured cameras and the twin robot USD.
* one `step` = one control tick at 30 Hz (4 PhysX substeps at 120 Hz): the action is the goal the controller would
  send; it goes through the residual env's LeRobot `max_relative_target` clamp around the measured state and its
  servo model (first-order response + velocity limit + gravity bias, tracked by a stiff PD drive), exactly as the
  residual env applies its base policy's targets (`ResidualEnv.step(goal_override=...)`).
* obs = {"images": {"top", "wrist", "inspect"}: uint8 (H, W, 3), "state": float32 (6,), "task": str}. Images are
  rendered from physics-free visual proxies synced from the physics state before every render (GB10 render path,
  sim/twin/visuals/usd_sync.py) and colour-corrected per camera (visuals.twin_scene.color_correct).
* task: "pick up the brick and hold it in front of the inspection camera" until the brick has been held still in
  the inspect view for 15 ticks (residual env phase 2), then "place the brick in the <bowl> bowl" (the recorded
  datasets' two task strings).
* reward / terminated / info: the residual env's shaped reward (reach, grasp, lift, present, place, penalties);
  terminated = placed in the sampled bowl or dropped; truncated = episode time limit. info["is_success"] = placed.

Randomisation: the residual env's own layout sampling (target brick where the gripper grasped in the data,
distractors, plate layout, start pose; seeded by `seed`) runs on every reset unless `cfg.layout="nominal"`.
Scene content (sim/twin/scene): with `cfg.bricks="data"` (default) the bricks are a pool of real Duplo kinds (2x2,
2x4, curved 2x3; LDraw geometry) and every reset draws the target / distractor kinds, colours, count, on-plate
placement and the number of plates on the table from the real-data statistics (scene/scene_stats.json);
`bricks="legacy"` keeps the residual env's 2x2 target + three distractors. A reset with an explicit `brick_pose`
(open-loop replays) uses a 2x2 target. `reset(layout=layout_from_episode(ep))` reproduces a recorded episode's first
frame: plates, target (colour, kind, pose), distractors, start pose and target bowl. `cfg.props` adds the station's
static props (inspection camera on its tripod, cables, white box, overhead tripod, table edge / floor, base clamp).
`randomize=True` additionally calls every hook in `cfg.randomizers` as `f(env, rng)` after the layout reset and
before the settle / render (default: `[TwinRandomizer()]` from sim/twin/env/randomize.py when that file exists).
Physics hooks: sim/twin/physics/apply.py `patch_scene_cfg(scene_cfg, profile)` before the InteractiveScene and
`post_spawn(env)` before `sim.reset()`, both optional.
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RESIDUAL = REPO / "sim/residual_env"
for _p in (str(REPO), str(RESIDUAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CAMS = ("top", "wrist", "inspect")
SENSOR = {"top": "top_cam", "wrist": "wrist_cam", "inspect": "inspect_cam"}
PICK_TASK = "pick up the brick and hold it in front of the inspection camera"
PLACE_TASK = "place the brick in the {bowl} bowl"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
# frame-0 joint states of the 456 cal2 episodes (LeRobot units, cal2 frame). scene_params.json's start_joint_state_deg
# is in OLD-calibration degrees, so the twin replaces it (fallback when the dataset is not on disk).
START_CAL2 = {"median": [26.42, -51.69, 30.11, 73.23, 15.56, 37.23], "std": [18.98, 37.91, 37.05, 21.02, 17.59, 14.42],
              "p5": [2.29, -102.51, -49.65, 35.23, -4.33, 17.95], "p95": [64.48, 22.95, 82.09, 101.56, 51.96, 61.75]}


@dataclass
class TwinEnvCfg:
    # images
    width: int = 640
    height: int = 480
    cameras: tuple = CAMS  # which cameras to return ("inspect" optional)
    color_correct: bool = True
    render_passes: int = 1  # sim.render() calls per returned frame (validate_twin.py checks 1 is lag-free)
    # control / episode
    episode_s: float = 30.0
    max_relative_target: float = 10.0  # LeRobot clamp, units per tick (1000 = off, as in teleop replays)
    meas_noise: bool = False  # Gaussian state noise; off: the play model's quantised encoder replaces it
    play: bool = True  # joint play / hold band / sag / encoder model fitted from data (sim/twin/dynamics)
    # PlayCfg overrides, e.g. {"controller_delay": True} when TwinEnv stands in for the deployed VLA controller
    # (+1 tick on pan / wrist_flex / wrist_roll, fitted on its traces); teleop replays keep the default (no delay)
    play_cfg: dict = field(default_factory=dict)
    settle_ticks: int = 4  # physics ticks after a reset (bricks drop 0.5 mm, arm holds) before the first obs
    # scene
    layout: str = "sample"  # "sample": residual env layout distributions (seeded) | "nominal": fixed layout
    start_pose: str = "recorded"  # "recorded": a random cal2 episode's frame-0 state + start_jitter_deg noise |
    #                               "gaussian": the residual env's clipped Gaussian over START_CAL2
    start_jitter_deg: float = 1.0
    spread_plates: bool = True  # re-sample the plates without overlap (the residual env's sampler lets them overlap)
    scene_params: str = str(REPO / "sim/realism/scene_params.json")
    # servo model: the twin fit (sim/twin/servo_fit/: gripper close rate fitted on real traces) when present
    servo_params: str = str(REPO / "sim/twin/servo_fit/servo_params_fit.json") if (
        REPO / "sim/twin/servo_fit/servo_params_fit.json").exists() else str(REPO / "sim/realism/servo_params.json")
    target_size: tuple | None = None  # e.g. (0.0636, 0.0318, 0.0192) for a 2x4 target
    look: Any = None  # visuals.twin_scene.TwinLook (None = default)
    finger_pads: bool = True  # the residual env's box finger pads (its grasp model)
    bricks: str = "data"  # "data": brick pool, kinds / colours / counts from the real data | "legacy"
    brick_pool: dict | None = None  # {"2x2": n, "2x4": n, "curved_2x3": n} incl. the target (None = scene default)
    scene_stats: str = str(REPO / "sim/twin/scene/scene_stats.json")
    drop_plates: bool = True  # "data" sampling: as often as in the real frames only 2 plates are on the table
    real_plates: bool = True  # "data" sampling: plate arrangements of real first frames (+ 1 cm jitter)
    props: bool = True  # static props (sim/twin/scene/props.py)
    instructions: dict = field(default_factory=lambda: {"grasp": PICK_TASK, "present": PICK_TASK,
                                                        "place": PLACE_TASK})
    task: str | None = None  # fixed task string (overrides the phase-dependent one)
    # randomisation
    randomize: bool = False  # default for reset(randomize=None): DR off for evaluation
    randomizers: list | None = None  # list of f(env, rng); None = [TwinRandomizer()] if randomize.py exists
    physics_hooks: bool = True  # use sim/twin/physics/apply.py if present
    physics_dr: bool = True  # with randomizers=None: also apply.py's sample_physics_dr / apply_physics_dr
    # runtime
    num_envs: int = 1  # TwinEnv returns env 0 only; >1 only for batched use of env.residual
    device: str = "cuda:0"
    headless: bool = True
    close_app: bool = True  # close the Kit app in close() if TwinEnv launched it
    residual_kw: dict = field(default_factory=dict)  # extra ResidualEnv EnvCfg fields


_APP = None


def launch_app(headless: bool = True, cameras: bool = True):
    """Start Isaac Sim with the GB10 render fix (no-op if already running). Returns (app, launched_here)."""
    global _APP
    if _APP is not None:
        return _APP, False
    try:  # already launched by the caller (e.g. a script that started AppLauncher itself)
        import omni.kit.app  # noqa: F401
        import omni.usd

        if omni.usd.get_context() is not None:
            return None, False
    except ImportError:
        pass
    # the twin's validated render path: Fabric scene delegate OFF + VisualSync (sim/twin/visuals/usd_sync.py).
    # sim/residual_env/app.py defaults to the delegate ON for the residual env; the twin pins its own flags.
    os.environ.setdefault("DLSJ_KIT_ARGS", "--/app/useFabricSceneDelegate=false "
                          "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false")
    from app import KIT_ARGS
    from isaaclab.app import AppLauncher

    _APP = AppLauncher(headless=headless, enable_cameras=cameras, kit_args=KIT_ARGS).app
    return _APP, True


def _optional(module: str):
    """Import a teammate's optional module if its file exists; warn (not fail) if it is broken."""
    path = REPO / (module.replace(".", "/") + ".py")
    if not path.exists():
        return None
    try:
        return importlib.import_module(module)
    except Exception as e:  # noqa: BLE001
        print(f"[TwinEnv] WARNING: {module} exists but failed to import ({type(e).__name__}: {e}); ignoring",
              flush=True)
        return None


class PhysicsDR:
    """Randomizer hook around sim/twin/physics/apply.py's physics DR (brick friction / mass, jaw friction, effort)."""

    def __init__(self, mod):
        self.mod = mod
        self.last = None

    def __call__(self, env, rng):
        self.last = self.mod.apply_physics_dr(env.scene, self.mod.sample_physics_dr(rng))
        return self.last

    def restore(self, env):
        """Nominal physics (params.json values) after DR episodes."""
        P = self.mod._p()
        nominal = {"brick_friction": P["bricks"]["material"]["static_friction"], "brick_mass_scale": 1.0,
                   "jaw_friction": P["gripper"]["material"]["static_friction"],
                   "gripper_effort_nm": P["gripper"]["effort_nm"]}
        self.mod.apply_physics_dr(env.scene, {k: v for k, v in nominal.items() if k in (self.last or nominal)})
        self.last = None


class TwinEnv:
    """Single-station twin env (see module doc). Attributes: sim, scene, stage, visuals, profile, cfg, layout,
    residual (the wrapped ResidualEnv), sync (VisualSync)."""

    def __init__(self, cfg: TwinEnvCfg | None = None):
        self.cfg = cfg = cfg or TwinEnvCfg()
        if "av" in sys.modules:
            print("[TwinEnv] WARNING: PyAV (av) is imported in this process; on the GB10 that has produced all-black "
                  "camera renders. Decode videos in a subprocess instead.", flush=True)
        self.app, self._own_app = launch_app(cfg.headless, cameras=True)

        import app as rapp
        import isaaclab.sim as sim_utils
        import omni.usd
        import scene as rscene
        import torch
        from isaaclab.scene import InteractiveScene

        from env import EnvCfg, ResidualEnv, _cam_from_rows
        from sim.twin.calib.profile import twin_profile
        from sim.twin.visuals.twin_scene import TwinLook, apply_twin_visuals, twin_scene_cfg, wrist_cam_rel
        from sim.twin.visuals.usd_sync import VisualSync

        self.torch = torch
        self.device = cfg.device
        self.profile = twin_profile()
        rscene.ROBOT_USD = self.profile.robot_usd
        self.look = cfg.look or TwinLook()

        # layout (residual env distributions from stage A2's scene_params.json)
        layout = rscene.load_layout(cfg.scene_params, "joe")
        if cfg.target_size:
            t = layout.target
            sx, sy, sz = cfg.target_size
            mass = t.mass * (sx * sy * sz) / (t.size[0] * t.size[1] * t.size[2])
            layout.target = rscene.Brick(t.name, tuple(cfg.target_size), t.pos, t.color, mass)
        if cfg.bricks == "data" and not cfg.target_size:
            from sim.twin.scene.bricks import pool_specs

            layout.distractors = pool_specs(rscene.Brick, cfg.brick_pool)
        st = START_CAL2
        layout.start_joint = tuple(tuple(st[k]) for k in ("median", "std", "p5", "p95"))
        layout.init_joint_deg = tuple(st["median"])
        self._starts = self._recorded_starts() if cfg.start_pose == "recorded" else None
        self.layout = layout

        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=cfg.device,
                                                                       use_fabric=False))
        scfg = twin_scene_cfg(cfg.num_envs, layout, self.profile, self.look)
        for cam in CAMS:  # resolution + rgb only (depth costs render time)
            c = getattr(scfg, SENSOR[cam], None)
            if c is None:
                continue
            c.width, c.height, c.data_types = cfg.width, cfg.height, ["rgb"]
            c.spawn.vertical_aperture = c.spawn.horizontal_aperture * cfg.height / cfg.width  # square pixels
        if cfg.physics_hooks:
            self._physics = _optional("sim.twin.physics.apply")
        else:
            self._physics = None
        if self._physics is not None:
            try:
                if hasattr(self._physics, "patch_profile"):
                    self._physics.patch_profile(self.profile)
                if hasattr(self._physics, "patch_scene_cfg"):
                    scfg = self._physics.patch_scene_cfg(scfg, self.profile) or scfg
            except Exception as e:
                # a failed physics patch leaves the stock gripper bias / pad tip / contacts: never silently
                if os.environ.get("DLSJ_ALLOW_PHYSICS_FAIL") != "1":
                    raise
                print(f"[TwinEnv] WARNING: physics patch_scene_cfg failed ({type(e).__name__}: {e})", flush=True)
        self.scene_cfg = scfg
        self.scene = InteractiveScene(scfg)
        self.stage = omni.usd.get_context().get_stage()
        self.visuals = apply_twin_visuals(cfg.num_envs, layout, self.profile, self.look)
        if cfg.finger_pads:
            rapp.finger_pads(cfg.num_envs)
        self.props_info = None
        self._keep_out = np.zeros((0, 2))  # prop footprints plates must clear (tripod feet / hub, white box)
        if cfg.props:
            from sim.twin.scene.props import PROPS, add_props, tripod_geometry

            self.props_info = add_props(cfg.num_envs, self.profile)
            g = tripod_geometry(self.profile)
            self._keep_out = np.r_[g["feet"], [g["hub_xy"]], [PROPS["box_center"]]]
        self.residual = None
        self.physics_info = None
        if self._physics is not None and hasattr(self._physics, "post_spawn"):
            try:
                self.physics_info = self._physics.post_spawn(self)
            except Exception as e:
                if os.environ.get("DLSJ_ALLOW_PHYSICS_FAIL") != "1":
                    raise
                print(f"[TwinEnv] WARNING: physics post_spawn failed ({type(e).__name__}: {e})", flush=True)
        self.sim.reset()
        self.scene.reset()
        self.sync = VisualSync(self.scene, wrist_cam_rel=wrist_cam_rel())

        rcfg = EnvCfg(num_envs=cfg.num_envs, cameras=False, scene_params=cfg.scene_params, calib="joe_cal2",
                      servo_params=cfg.servo_params, max_relative_target=cfg.max_relative_target,
                      meas_noise=cfg.meas_noise, episode_s=cfg.episode_s, instructions=dict(cfg.instructions),
                      target_size=cfg.target_size, **cfg.residual_kw)
        self.residual = ResidualEnv(rcfg, None, self.sim, self.scene, layout, cfg.device)
        # the twin's measured cameras (the residual env loaded joe_cal2's inspect camera for its in-view check)
        self.residual.profile = self.profile
        # gripper unit mapping of the twin profile (the physics layer may re-solve the bias for its finger colliders)
        self.residual.g_scale, self.residual.g_bias = float(self.profile.grip_scale), float(self.profile.grip_bias)
        ins = self.profile.inspect
        self.residual.inspect_cam = _cam_from_rows(np.array(ins.rows), device=cfg.device, f_px=ins.f_px)
        # joint offsets of the twin profile (fit W): the residual env loaded joe_cal2's for to_sim / to_ours / measured
        self.residual.off = torch.tensor(self.profile.joint_offsets, dtype=torch.float32, device=cfg.device)
        self.play = None
        if cfg.play:
            from sim.twin.dynamics.play import PlayCfg, install

            self.play = install(self.residual, PlayCfg(**cfg.play_cfg))

        self.bricks = None
        if cfg.bricks == "data" and not cfg.target_size:
            from sim.twin.scene.bricks import BrickPool, load_stats

            self.bricks = BrickPool(self)
            self.scene_stats = load_stats(Path(cfg.scene_stats))
        self.scene_plan: dict = {}
        if cfg.randomizers is None:
            self.randomizers: list[Callable] = []
            rmod = _optional("sim.twin.env.randomize")
            if rmod is not None and hasattr(rmod, "TwinRandomizer"):
                self.randomizers.append(rmod.TwinRandomizer())
            ph = self._physics
            if cfg.physics_dr and ph is not None and hasattr(ph, "sample_physics_dr") and \
                    hasattr(ph, "apply_physics_dr"):
                self.randomizers.append(PhysicsDR(ph))
        else:
            self.randomizers = list(cfg.randomizers)
        self.rng = np.random.default_rng(0)
        self.randomizer_errors: dict[str, str] = {}
        self.dr_draw: dict = {}  # set by TwinRandomizer (randomize.py) on a DR reset
        self._dr_active = False
        self._images: dict[str, np.ndarray] = {}
        self._task = cfg.task or cfg.instructions["grasp"]
        self.t = 0
        self.last_info: dict = {}
        print(f"[TwinEnv] profile {self.profile.name} | {cfg.width}x{cfg.height} {list(cfg.cameras)} | "
              f"randomizers {[type(r).__name__ for r in self.randomizers]} | physics hooks "
              f"{'apply.py' if self._physics else 'none'}", flush=True)

    @staticmethod
    def _recorded_starts():
        """Frame-0 states of the cal2 episodes (pyarrow only; PyAV is deliberately not imported here)."""
        try:
            import pyarrow.compute as pc
            import pyarrow.parquet as pq

            out = []
            for f in sorted((DATA / "data").glob("*/*.parquet")):
                t = pq.read_table(f, columns=["frame_index", "observation.state"])
                t = t.filter(pc.equal(t["frame_index"], 0))
                out += t["observation.state"].to_pylist()
            return np.asarray(out, np.float32) if out else None
        except Exception as e:  # noqa: BLE001
            print(f"[TwinEnv] recorded start poses unavailable ({type(e).__name__}: {e}); Gaussian start", flush=True)
            return None

    # ------------------------------------------------------------------ rendering
    def render(self) -> dict[str, np.ndarray]:
        """Sync the visual proxies from physics, render, return {cam: uint8 (H, W, 3)} (colour-corrected)."""
        from sim.twin.visuals.twin_scene import color_correct

        self.sync()
        for _ in range(max(1, self.cfg.render_passes)):
            self.sim.render()
        self.scene.update(0.0)
        out = {}
        for cam in self.cfg.cameras:
            key = SENSOR[cam]
            if key not in self.scene.sensors:
                continue
            img = self.scene.sensors[key].data.output["rgb"][0, ..., :3]
            if self.cfg.color_correct:
                img = color_correct(img, cam)
            if self._dr_active:  # image-space DR of the randomizers (randomize.py: after color_correct)
                for f in self.randomizers:
                    if hasattr(f, "postprocess"):
                        img = f.postprocess(img, cam)
            out[cam] = img.detach().cpu().numpy().astype(np.uint8)
        self._images = out
        return out

    # ------------------------------------------------------------------ obs
    def state(self, noisy: bool | None = None) -> np.ndarray:
        """Measured joint state (6,) in LeRobot units (noise per cfg.meas_noise unless given)."""
        n = self.cfg.meas_noise if noisy is None else noisy
        return self.residual.measured(noisy=n)[0].detach().cpu().numpy().astype(np.float32)

    def task(self) -> str:
        if self.cfg.task:
            return self.cfg.task
        return self.residual.instructions(self.torch.zeros(1, dtype=self.torch.long, device=self.device))[0]

    def _obs(self, render: bool = True) -> dict:
        imgs = self.render() if render else self._images
        return {"images": imgs, "state": self.state(), "task": self.task()}

    # ------------------------------------------------------------------ reset / step
    def reset(self, seed: int | None = None, randomize: bool | None = None, *, joint_deg=None, brick_pose=None,
              bowl: str | None = None, render: bool = True, layout: dict | None = None) -> dict:
        """New episode. seed: seeds the layout sampling (residual env generator) and the randomizers' rng.
        joint_deg (6,) start pose in LeRobot units, brick_pose (x, y, yaw) env frame, bowl: target bowl name.
        layout: a real-episode layout (sim/twin/scene/layout.layout_from_episode): plates, target, distractors, start
        pose and bowl from the recorded first frame (explicit joint_deg / brick_pose / bowl still win)."""
        torch = self.torch
        r = self.residual
        if seed is not None:
            r.gen.manual_seed(int(seed))
            self.rng = np.random.default_rng(int(seed))
        dev = self.device
        n = self.cfg.num_envs
        if layout is not None:
            if joint_deg is None and layout.get("start_state") is not None:
                joint_deg = layout["start_state"]
            t = layout.get("target")
            if brick_pose is None and t:
                brick_pose = (t["x"], t["y"], t.get("yaw", 0.0))
            if bowl is None and layout.get("bowl") in r.bowl_names:
                bowl = layout["bowl"]
        plan = self._brick_plan(layout, replay=brick_pose is not None and layout is None)
        jd = bp = None
        if joint_deg is not None:
            jd = torch.as_tensor(np.asarray(joint_deg, np.float32), device=dev).reshape(-1, 6).expand(n, 6).clone()
        elif self.cfg.layout == "nominal":
            jd = torch.tensor(self.layout.init_joint_deg, dtype=torch.float32, device=dev).expand(n, 6).clone()
        elif self._starts is not None:
            k = self.rng.integers(len(self._starts), size=n)
            j = self._starts[k] + self.rng.normal(0.0, self.cfg.start_jitter_deg, (n, 6)).astype(np.float32)
            j[:, 5] = np.clip(j[:, 5], 0.0, 100.0)
            jd = torch.as_tensor(j, device=dev)
        if brick_pose is not None:
            bp = torch.as_tensor(np.asarray(brick_pose, np.float32), device=dev).reshape(-1, 3).expand(n, 3).clone()
        elif self.cfg.layout == "nominal":
            bp = torch.tensor([[*self.layout.target.pos, 0.0]], dtype=torch.float32, device=dev).expand(n, 3).clone()
        r.reset(joint_deg=jd, brick_pose=bp)
        if bowl is not None:
            r.bowl[:] = r.bowl_names.index(bowl)
        real_arrangement = False
        if layout is not None:
            self._layout_plates(layout)
        elif self.cfg.layout == "sample" and self.cfg.spread_plates:
            real_arrangement = bool(plan) and self.cfg.real_plates and self._real_plates()
            if not real_arrangement:
                self._spread_plates()
        if plan:
            if layout is None and not real_arrangement and self.cfg.drop_plates and \
                    plan.get("n_plates", 3) < len(r.bowl_objs):
                self._drop_plates(len(r.bowl_objs) - max(1, plan["n_plates"]))
            self.bricks.place(plan, self.rng, exact=layout is not None)
            self.scene_plan = dict(self.bricks.describe(), plates_on_table=[
                nm for k, nm in enumerate(r.bowl_names) if abs(float(r.bowl_xy_env[0, k, 0])) < 1.0],
                source=f"episode {layout.get('episode')}" if layout is not None else "sampled")
        do_dr = self.cfg.randomize if randomize is None else bool(randomize)
        if not do_dr and self._dr_active:  # evaluation reset after DR episodes: back to the nominal twin
            for f in self.randomizers:
                if hasattr(f, "restore"):
                    try:
                        f.restore(self)
                    except Exception as e:  # noqa: BLE001
                        print(f"[TwinEnv] WARNING: restore of {type(f).__name__} failed ({e})", flush=True)
            self.dr_draw = {}
        self._dr_active = do_dr
        if do_dr:
            for f in self.randomizers:
                try:
                    f(self, self.rng)
                except Exception as e:  # noqa: BLE001  (a teammate's hook must not kill a rollout; logged once)
                    name = getattr(f, "__name__", type(f).__name__)
                    if name not in self.randomizer_errors:
                        print(f"[TwinEnv] WARNING: randomizer {name} failed ({type(e).__name__}: {e}); "
                              "continuing without it for this reset", flush=True)
                    self.randomizer_errors[name] = f"{type(e).__name__}: {e}"
        self.settle(self.cfg.settle_ticks)
        self.t = 0
        self.last_info = {}
        return self._obs(render)

    def _brick_plan(self, layout: dict | None, replay: bool) -> dict:
        """The brick pool assignment for this reset (empty with cfg.bricks="legacy")."""
        if self.bricks is None:
            return {}
        from sim.twin.scene.bricks import plan_from_layout, sample_plan

        if layout is not None:
            plan = plan_from_layout(layout)
        elif replay or self.cfg.layout == "nominal":  # open-loop replays / nominal: the residual env's 2x2 target
            plan = sample_plan(self.rng, self.scene_stats, self.bricks.counts)
            plan["target"] = {"kind": "2x2", "color": "light_blue"}
            if self.cfg.layout == "nominal":
                plan["distractors"] = [{"kind": "2x4", "color": "light_blue", "on_plate": False},
                                       {"kind": "2x4", "color": "yellow", "on_plate": False},
                                       {"kind": "2x2", "color": "cream", "on_plate": False}]
                plan["n_plates"] = 3
        else:
            plan = sample_plan(self.rng, self.scene_stats, self.bricks.counts)
        self.bricks.assign(plan)
        return plan

    def _set_plate(self, k: int, xy):
        torch = self.torch
        r = self.residual
        o = self.scene.env_origins
        xy_t = torch.as_tensor(np.asarray(xy, np.float32), device=self.device)
        r.bowl_xy_env[:, k] = xy_t
        pose = torch.cat([o[:, :2] + xy_t, o[:, 2:] + r.table_z + 0.003,
                          torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(len(o), 4)], -1)
        r.bowl_objs[k].write_root_pose_to_sim(pose)

    def _park_plate(self, k: int):
        """Plate off the table and out of every camera (kinematic body; the reward only uses the target bowl)."""
        self._set_plate(k, (2.0 + 0.3 * k, 2.0))

    def _layout_plates(self, layout: dict):
        """Plates where the recorded first frame shows them; plates not seen are parked."""
        r = self.residual
        for k, name in enumerate(r.bowl_names):
            p = layout.get("plates", {}).get(name)
            if p is not None:
                self._set_plate(k, (p["x"], p["y"]))
            else:
                self._park_plate(k)

    def _real_plates(self, tries: int = 40, jitter: float = 0.01) -> bool:
        """Plates as in a random real first frame that has the target bowl (layout statistics), jittered; must clear
        the target brick, the gripper, the props and each other. Plates not in that frame are parked."""
        r = self.residual
        lays = [L for L in self.scene_stats.get("plate_layouts", []) if r.bowl_names[int(r.bowl[0])] in L]
        if not lays:
            return False
        rad = dict(zip(r.bowl_names, r.bowl_r.detach().cpu().numpy(), strict=True))
        o0 = self.scene.env_origins[0]
        brick = r.brick_local()[0, :2].detach().cpu().numpy()
        tip = (r.tip_w()[0] - o0)[:2].detach().cpu().numpy()
        for _ in range(tries):
            L = lays[int(self.rng.integers(len(lays)))]
            pos = {k: np.asarray(v, float) + self.rng.normal(0.0, jitter, 2) for k, v in L.items() if k in rad}
            ok = all(np.linalg.norm(p - brick) > rad[k] + 0.03 and np.linalg.norm(p - tip) > rad[k] + 0.02 and
                     np.linalg.norm(p) > rad[k] + 0.09 and
                     all(np.linalg.norm(p - q) > rad[k] + 0.03 for q in self._keep_out) for k, p in pos.items())
            ks = list(pos)
            ok &= all(np.linalg.norm(pos[a] - pos[b]) > rad[a] + rad[b] + 0.003
                      for i, a in enumerate(ks) for b in ks[i + 1:])
            if ok:
                for k, name in enumerate(r.bowl_names):
                    if name in pos:
                        self._set_plate(k, pos[name])
                    else:
                        self._park_plate(k)
                return True
        return False

    def _drop_plates(self, n_drop: int):
        """Park `n_drop` random non-target plates."""
        r = self.residual
        tgt = int(r.bowl[0])
        others = [k for k in range(len(r.bowl_objs)) if k != tgt]
        for k in self.rng.permutation(others)[:n_drop]:
            self._park_plate(int(k))

    def _spread_plates(self, tries: int = 4000, margin: float = 0.04):
        """Non-overlapping plate centres (as on the real table: >= r1 + r2 apart) inside the data's plate-centre box
        (+ margin), clear of the target brick and the robot base; keeps the residual env's draw if none fits."""
        torch = self.torch
        r, lay = self.residual, self.layout
        if lay.plate_box is None or not r.bowl_objs:
            return
        lo, hi = np.array(lay.plate_box[0]) - margin, np.array(lay.plate_box[1]) + margin
        rad = r.bowl_r.detach().cpu().numpy()
        o0 = self.scene.env_origins[0]
        brick = r.brick_local()[0, :2].detach().cpu().numpy()
        tip = (r.tip_w()[0] - o0)[:2].detach().cpu().numpy()
        dis = [(d.data.root_pos_w[0] - o0).detach().cpu().numpy() for d in r.distractors]
        dis = [p[:2] for p in dis if abs(p[0]) < 1.0 and abs(p[1]) < 1.0]  # on the table (unused ones are parked)
        c = lo + self.rng.random((tries, len(rad), 2)) * (hi - lo)
        ok = (np.linalg.norm(c - brick, axis=-1) > rad + 0.03).all(1)
        ok &= (np.linalg.norm(c, axis=-1) > rad + 0.09).all(1)  # robot base
        ok &= (np.linalg.norm(c - tip, axis=-1) > rad + 0.02).all(1)  # gripper at the start pose
        for q in self._keep_out:  # inspection tripod, white box
            ok &= (np.linalg.norm(c - q, axis=-1) > rad + 0.03).all(1)
        for i in range(len(rad)):
            for j in range(i + 1, len(rad)):
                ok &= np.linalg.norm(c[:, i] - c[:, j], axis=-1) > rad[i] + rad[j] + 0.005
        ok_d = ok.copy()
        for p in dis:  # distractor bricks not under a plate rim (relaxed if nothing fits)
            ok_d &= (np.linalg.norm(c - p, axis=-1) > rad + 0.025).all(1)
        idx = np.flatnonzero(ok_d) if ok_d.any() else np.flatnonzero(ok)
        if not len(idx):
            return
        best = torch.as_tensor(c[idx[0]], dtype=torch.float32, device=self.device)
        o = self.scene.env_origins
        for k, obj in enumerate(r.bowl_objs):
            r.bowl_xy_env[:, k] = best[k]
            pose = torch.cat([o[:, :2] + best[k], o[:, 2:] + r.table_z + 0.003,
                              torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(len(o), 4)], -1)
            obj.write_root_pose_to_sim(pose)

    def settle(self, ticks: int):
        """Hold the current servo reference for `ticks` control ticks without advancing the episode clock."""
        r = self.residual
        for _ in range(ticks):
            r.apply_goal(r.r.clone())
            r.physics()

    def step(self, action, render: bool = True):
        """One 30 Hz tick. action (6,) LeRobot units (the goal sent to the follower). render=False skips the
        cameras (obs["images"] then holds the last rendered frames)."""
        torch = self.torch
        a = torch.as_tensor(np.asarray(action, np.float32).reshape(-1)[:6], device=self.device)
        goal = a.unsqueeze(0).expand(self.cfg.num_envs, 6).clone()
        _, rew, done, info = self.residual.step(goal_override=goal)
        self.t += 1
        placed = bool(info["placed"][0])
        dropped = bool(info["dropped"][0])
        truncated = bool(self.residual.t[0] >= self.residual.max_ticks) and not (placed or dropped)
        out = {k: (v[0].item() if hasattr(v, "dim") else v) for k, v in info.items()}
        out.update(is_success=placed, tick=self.t, task=self.task(),
                   clamp_hits=float(self.residual.stats["clamp_hits"][0]),
                   brick_pos=(self.residual.brick_local()[0]).detach().cpu().numpy().round(4).tolist(),
                   bowl=self.residual.bowl_names[int(self.residual.bowl[0])])
        self.last_info = out
        return self._obs(render), float(rew[0]), placed or dropped, truncated, out

    # ------------------------------------------------------------------ misc
    @property
    def num_envs(self) -> int:
        return self.cfg.num_envs

    @property
    def max_episode_steps(self) -> int:
        return int(self.residual.max_ticks)

    def close(self):
        if self._own_app and self.cfg.close_app and self.app is not None:
            global _APP
            self.app.close()
            _APP = None
