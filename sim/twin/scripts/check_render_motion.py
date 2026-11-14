"""Does a camera see physics motion? Render the residual env's scene (its own launch flags) at two very different
arm poses with the standard step/render loop, no visual-sync workaround, and compare the top-camera images.

    sim/twin/run.sh sim/twin/scripts/check_render_motion.py
"""
import dataclasses
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "sim/residual_env"))
from app import KIT_ARGS  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

simapp = AppLauncher(headless=True, enable_cameras=True, kit_args=KIT_ARGS).app
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

import scene as rscene  # noqa: E402
from calib import load_profile  # noqa: E402

usd = str(REPO / "sim/twin/assets/so101/usd/so101_new_calib/so101_new_calib.usda")
rscene.ROBOT_USD = usd
prof = dataclasses.replace(load_profile("joe_cal2"), robot_usd=usd)
layout = rscene.load_layout(str(REPO / "sim/realism/scene_params.json"), "joe")
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
scn = InteractiveScene(rscene.build_scene_cfg(1, layout, True, profile=prof))
sim.reset(); scn.reset()
robot, cam = scn["robot"], scn["top_cam"]
imgs, gl = [], []
for pose_deg in ([0, -60, 60, 60, 0, 10], [60, 20, -40, 30, 90, 60]):
    q = torch.tensor([np.deg2rad(pose_deg)], dtype=torch.float32, device="cuda:0")
    for _ in range(60):
        robot.write_joint_state_to_sim(q, torch.zeros_like(q)); robot.set_joint_position_target(q); robot.write_data_to_sim()
        sim.step(); scn.update(sim.get_physics_dt())
    imgs.append(cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.float32))
    gl.append(robot.data.body_link_pose_w[0, robot.data.body_names.index("gripper_link"), :3].cpu().numpy())
d = np.abs(imgs[0] - imgs[1]).mean(-1)
print(f"physics gripper position: pose A {np.round(gl[0], 3).tolist()}  pose B {np.round(gl[1], 3).tolist()}")
print(f"top image change between poses: mean {d.mean():.2f}, pixels changed >20: {(d > 20).mean() * 100:.2f}%")
print("RENDER_SEES_MOTION" if (d > 20).mean() > 0.005 else "RENDER_FROZEN")
simapp.close()
