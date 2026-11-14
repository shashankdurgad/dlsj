"""Spawn N envs, settle, print joint/body info, save top + wrist images of env 0."""
import argparse
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=4)
ap.add_argument("--cameras", type=int, default=1)
ap.add_argument("--out", default=str(Path.home() / "residual_env/out/smoke"))
ap.add_argument("--plain_app", type=int, default=0, help="launch via isaacsim.SimulationApp")
ap.add_argument("--kit_args", default="--/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false")
args = ap.parse_args()

if args.plain_app:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": 640, "height": 480})
else:
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=bool(args.cameras), kit_args=args.kit_args).app

import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene import JOINTS, build_scene_cfg, load_layout  # noqa: E402

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
lay = load_layout(Path.home() / "residual_env/scene_params.json", "joe")
scene = InteractiveScene(build_scene_cfg(args.num_envs, lay, bool(args.cameras)))
sim.reset()
scene.reset()
robot = scene["robot"]
robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
robot.set_joint_position_target(robot.data.default_joint_pos.clone())
print("joint_names", robot.joint_names)
print("body_names", robot.body_names)
lim = robot.data.joint_pos_limits[0].cpu().numpy() if hasattr(robot.data, "joint_pos_limits") else None
print("limits_deg", None if lim is None else np.round(np.rad2deg(lim), 1).tolist())
for i in range(240):
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim.get_physics_dt())
for _ in range(5):
    sim.render()
    scene.update(sim.get_physics_dt())
print("render_interval", getattr(sim.cfg, "render_interval", None), "has_rtx", sim.has_rtx_sensors() if hasattr(sim, "has_rtx_sensors") else "?")
print("joint_pos_deg env0", np.round(np.rad2deg(robot.data.joint_pos[0].cpu().numpy()), 2).tolist())
print("brick pos env0", scene["brick"].data.root_pos_w[0].cpu().numpy().round(4).tolist(),
      "env origin", scene.env_origins[0].cpu().numpy().round(3).tolist())
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if args.cameras:
    for name in ("top_cam", "wrist_cam"):
        rgb = scene[name].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(rgb).save(out / f"{name}_env0.png")
        d = scene[name].data.output["distance_to_image_plane"][0].cpu().numpy()
        cd = scene[name].data
        print(name, "pos_w", cd.pos_w[0].cpu().numpy().round(3).tolist(), "quat_w", cd.quat_w_world[0].cpu().numpy().round(3).tolist() if hasattr(cd, "quat_w_world") else cd.quat_w_ros[0].cpu().numpy().round(3).tolist(), "| origin", scene.env_origins[0].cpu().numpy().round(3).tolist())
        dd = np.where(np.isfinite(d), d, np.nan); print(name, "depth p5/p50/p95", np.nanpercentile(dd, [5, 50, 95]).round(4).tolist(), "dtype", d.dtype, "shape", d.shape)
        print(name, rgb.shape, "rgb mean", float(rgb.mean()), "| depth finite frac", float(np.isfinite(d).mean()),
              "median", float(np.nanmedian(np.where(np.isfinite(d), d, np.nan))))
import omni.replicator.core as rep
anns = {}
for name, path in (("top", "/World/envs/env_0/TopCam"),
                   ("wrist", "/World/envs/env_0/Robot/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link/WristCam")):
    rp = rep.create.render_product(path, (640, 480))
    a = rep.AnnotatorRegistry.get_annotator("rgb"); a.attach([rp]); anns[name] = a
for _ in range(30):
    rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
    if all(np.asarray(a.get_data()).ndim == 3 for a in anns.values()):
        break
for name, a in anns.items():
    img = np.asarray(a.get_data())[..., :3]
    Image.fromarray(img).save(out / f"rep_{name}_env0.png")
    print("REP", name, img.shape, "mean", float(img.mean()))
print("SMOKE_SCENE_OK")
app.close()
