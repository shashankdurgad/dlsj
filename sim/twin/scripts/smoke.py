"""Twin smoke test: Isaac Lab 3.0 and Physical AI Studio in one Python, the SO-101 URDF converted to USD, and the
residual env's scene (joe_cal2 cameras) rendered once from every camera.

    sim/twin/run.sh sim/twin/scripts/smoke.py [--reconvert]

Writes sim/twin/out/smoke_<camera>.png and sim/twin/assets/so101/usd/ (git-ignored).
"""

import argparse
import dataclasses
import importlib.metadata as md
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TWIN = REPO / "sim/twin"
OUT = TWIN / "out"
sys.path.insert(0, str(REPO / "sim/residual_env"))
from app import KIT_ARGS  # noqa: E402  (GB10 render fix; app.py imports nothing from Isaac at module level)

ap = argparse.ArgumentParser()
ap.add_argument("--reconvert", action="store_true", help="redo the URDF -> USD conversion")
ap.add_argument("--steps", type=int, default=30)
args = ap.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

simapp = AppLauncher(headless=True, enable_cameras=True, kit_args=KIT_ARGS).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

for pkg in ("isaacsim", "isaaclab", "physicalai-train", "torch", "numpy", "gymnasium", "warp-lang"):
    try:
        print(f"{pkg:18s} {md.version(pkg)}", flush=True)
    except md.PackageNotFoundError:
        print(f"{pkg:18s} (not installed)", flush=True)
from physicalai.gyms import Gym  # noqa: E402,F401  (Studio's gym interface imports next to Isaac Lab)

print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

# 1. SO-101 URDF -> USD (same description as Joe's so101_new_calib USD)
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

drive = UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=60.0, damping=2.0))
conv = UrdfConverter(UrdfConverterCfg(
    asset_path=str(TWIN / "assets/so101/urdf/so101_new_calib.urdf"), usd_dir=str(TWIN / "assets/so101/usd"),
    usd_file_name="so101_new_calib.usda", fix_base=True, merge_fixed_joints=False, joint_drive=drive,
    force_usd_conversion=args.reconvert))
usd = conv.usd_path
print("robot usd:", usd, flush=True)

# 2. the residual env's scene with our robot USD and the joe_cal2 calibration
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

import scene as rscene  # noqa: E402
from app import paint_robot  # noqa: E402
from calib import load_profile  # noqa: E402

rscene.ROBOT_USD = usd
profile = load_profile("joe_cal2")
profile = dataclasses.replace(profile, robot_usd=usd) if dataclasses.is_dataclass(profile) else profile
if not dataclasses.is_dataclass(profile):
    profile.robot_usd = usd
layout = rscene.load_layout(str(REPO / "sim/realism/scene_params.json"), "joe")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
scn = InteractiveScene(rscene.build_scene_cfg(1, layout, True, profile=profile))
paint_robot(1)
sim.reset()
scn.reset()
for _ in range(args.steps):
    sim.step()
    scn.update(sim.get_physics_dt())

robot = scn["robot"]
print("joints:", robot.data.joint_names, flush=True)
print("joint pos (deg):", np.round(np.rad2deg(robot.data.joint_pos[0].cpu().numpy()), 1).tolist(), flush=True)

import omni.usd  # noqa: E402

stage = omni.usd.get_context().get_stage()
paths = [str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith("/World/envs/env_0/Robot")]
print("robot prims:", len(paths), "| gripper link found:",
      any(p.endswith(rscene.GRIPPER_LINK.split("Robot/", 1)[1]) for p in paths),
      "| jaw link found:", any(p.endswith(rscene.JAW_LINK.split("Robot/", 1)[1]) for p in paths), flush=True)

import imageio.v3 as iio  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
for name, sensor in scn.sensors.items():
    out = getattr(sensor.data, "output", None)
    if not out or "rgb" not in out:
        continue
    rgb = out["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
    iio.imwrite(OUT / f"smoke_{name}.png", rgb)
    print(f"camera {name}: {rgb.shape} mean {rgb.mean():.1f} -> {OUT / f'smoke_{name}.png'}", flush=True)
print("SMOKE_OK", flush=True)
simapp.close()
