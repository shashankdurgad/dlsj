"""Run the residual env's realism gate (sim/residual_env/validate.py, unchanged) in the twin: twin calibration profile
(twin robot USD converted from the SO-101 URDF, measured cameras), A1 servo / A2 scene params from sim/realism/, and
optionally the twin physics layer (sim/twin/physics/apply.py).

    sim/twin/run.sh sim/twin/physics/realism_gate.py --physics off --out sim/twin/out/physics/gate_baseline
    sim/twin/run.sh sim/twin/physics/realism_gate.py --physics on  --out sim/twin/out/physics/gate_physics
    ... --params sim/twin/physics/params.json --set gripper.collider=pads_cad   (override params, dotted keys)

--physics off = the residual env as the teammate left it (app.finger_pads on the twin USD).
"""
import argparse
import json
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "sim/residual_env"
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"

ap = argparse.ArgumentParser()
ap.add_argument("--physics", choices=["off", "on"], default="on")
ap.add_argument("--params", default=str(REPO / "sim/twin/physics/params.json"))
ap.add_argument("--set", action="append", default=[], help="override a params.json entry: dotted.key=json_value")
ap.add_argument("--out", required=True)
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--num_envs", type=int, default=20)
ap.add_argument("--physics_hz", type=int, default=0, help="override EnvCfg.physics_hz (0 = default 120)")
ap.add_argument("--skip", default="")
ap.add_argument("--table_dz", type=float, default=0.0, help="experiment: move the table plane by dz (m)")
ap.add_argument("--trace", type=int, default=0, help="dump per-tick gripper q / brick pose in the gripper frame")
ap.add_argument("--park_plates", type=int, default=1,
                help="1: plates parked out of reach (the replay does not know each episode's real plate layout, and "
                     "random plates with a 23 mm rim get hit by the replayed arm); 0: the env's random layout")
ap.add_argument("--data", default=str(DATA))
args = ap.parse_args()

sys.path.insert(0, str(RES))
import app  # noqa: E402  (stdlib only at module level)
import calib  # noqa: E402

from sim.twin.calib.profile import twin_profile  # noqa: E402  (binds the original calib.load_profile)

app.DEFAULT_SCENE_PARAMS = str(REPO / "sim/realism/scene_params.json")
app.DEFAULT_SERVO_PARAMS = str(REPO / "sim/realism/servo_params.json")

phys = None
if args.physics == "on":
    from sim.twin.physics import apply as phys  # noqa: E402

    phys.load_params(args.params, overrides=args.set)

_orig_load = calib.load_profile
_cache = {}


def _load_profile(name, *a, **kw):
    if name == "twin_measured":
        if "p" not in _cache:
            p = twin_profile()
            _cache["p"] = phys.patch_profile(p) if phys else p
        return _cache["p"]
    return _orig_load(name, *a, **kw)


calib.load_profile = _load_profile

if args.physics_hz:
    import env as renv

    _init = renv.EnvCfg.__init__

    def _cfg_init(self, *a, **kw):
        _init(self, *a, **kw)
        self.physics_hz = args.physics_hz

    renv.EnvCfg.__init__ = _cfg_init


if args.park_plates:
    import env as renv
    import torch

    def _park(self, ids, brick_xy):
        m = len(ids)
        for k, obj in enumerate(self.bowl_objs):
            xy = torch.tensor([0.75, -0.45 + 0.45 * k], device=self.device).expand(m, 2)
            self.bowl_xy_env[ids, k] = xy
            z = torch.full((m, 1), self.table_z + 0.003, device=self.device)
            quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(m, 4)
            obj.write_root_pose_to_sim(torch.cat([self.scene.env_origins[ids] + torch.cat([xy, z], -1), quat], -1),
                                       env_ids=ids)

    renv.ResidualEnv._place_bowls = _park


if args.trace:
    import env as renv
    import numpy as _np
    import torch

    _step = renv.ResidualEnv.step
    _rec = []

    def _traced(self, *a, **kw):
        o = _step(self, *a, **kw)
        gq = self.robot.data.joint_pos[:, self.jidx[5]]
        q = self.robot.data.body_quat_w[:, self.grip_body]
        rel = renv._quat_apply_xyzw(renv._quat_conj_xyzw(q), self.brick.data.root_pos_w -
                                    self.robot.data.body_pos_w[:, self.grip_body])
        tau = self.robot.data.applied_torque[:, self.jidx[5]] if hasattr(self.robot.data, "applied_torque") \
            else torch.zeros_like(gq)
        bz = renv._quat_apply_xyzw(self.brick.data.root_quat_w, torch.tensor([0.0, 0, 1], device=self.device)
                                   .expand(self.n, 3))
        gz = renv._quat_apply_xyzw(q, torch.tensor([0.0, 0, 1], device=self.device).expand(self.n, 3))
        tilt = torch.rad2deg(torch.acos((bz * gz).sum(-1).abs().clamp(-1, 1)))  # brick vs gripper z
        bzw = (self.brick.data.root_pos_w - self.scene.env_origins)[:, 2]
        _rec.append(torch.cat([gq[:, None], self.r[:, 5:6], rel, tau[:, None],
                               self.robot.data.joint_pos_target[:, self.jidx[5]][:, None], tilt[:, None],
                               bzw[:, None], (self.tip_w() - self.scene.env_origins)[:, 2:3]], -1).cpu().numpy())
        if len(_rec) % 50 == 0:
            _np.save(Path(args.out) / "trace.npy", _np.stack(_rec))
        return o

    renv.ResidualEnv.step = _traced


def make(cfg, base=None, device: str = "cuda:0"):
    """app.make with the physics hooks (same order as app.make)."""
    import os

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    from env import ResidualEnv
    from scene import build_scene_cfg, load_layout

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / cfg.physics_hz, device=device))
    layout = load_layout(cfg.scene_params or app.DEFAULT_SCENE_PARAMS, getattr(cfg, "camera_source", "joe"))
    profile = calib.load_profile(cfg.calib)
    print("calibration profile:", profile.name, "| robot", profile.robot_usd, "| gripper q = units *",
          round(profile.grip_scale, 5), "+", round(profile.grip_bias, 4), "| physics", args.physics, flush=True)
    import scene as rscene

    rscene.TABLE_TOP_Z = -0.002 + args.table_dz
    scfg = build_scene_cfg(cfg.num_envs, layout, cfg.cameras, profile=profile)
    if args.table_dz:
        p0 = scfg.table.init_state.pos
        scfg.table.init_state.pos = (p0[0], p0[1], p0[2] + args.table_dz)
    if phys:
        scfg = phys.patch_scene_cfg(scfg, profile)
    scene = InteractiveScene(scfg)
    if phys:
        print("physics post_spawn:", json.dumps(phys.post_spawn(scene)), flush=True)
    elif os.environ.get("FINGER_PADS", "1") == "1":
        print("finger pads:", app.finger_pads(cfg.num_envs), flush=True)
    sim.reset()
    scene.reset()
    if not cfg.servo_params:
        cfg.servo_params = app.DEFAULT_SERVO_PARAMS
    env = ResidualEnv(cfg, base, sim, scene, layout, device)
    env.table_z += args.table_dz
    return sim, scene, layout, env


app.make = make
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if phys:
    (out / "params_used.json").write_text(json.dumps(phys.PARAMS, indent=1))
sys.argv = [str(RES / "validate.py"), "--calib", "twin_measured", "--data", args.data, "--out", str(out),
            "--episodes", str(args.episodes), "--num_envs", str(args.num_envs)] + \
           (["--skip", args.skip] if args.skip else [])
runpy.run_path(str(RES / "validate.py"), run_name="__main__")
