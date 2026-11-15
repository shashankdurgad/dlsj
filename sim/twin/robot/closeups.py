"""Close-up renders of the twin's wrist (mount, camera, gripper body) at a recorded joint state, plus the wrist view.

    sim/twin/run.sh sim/twin/robot/closeups.py [--ep 7 --frame 92] [--wrist mount|joe] [--tag _x]
writes sim/twin/out/robot/closeups_ep<EP>_f<F><tag>.jpg: four world-fixed close-up cameras placed around the
gripper (side +x, camera side +y, from above-behind like the phone photos, from the fingertips) and the twin's wrist
camera. Joint state -> sim angles exactly as the env (joe_cal2 offsets / gripper mapping); the close-up cameras are
placed from the physics gripper pose after the warm-up (their USD xforms, like VisualSync's wrist camera). Visual
check only; no physics changes.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim/residual_env"))
from app import KIT_ARGS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ep", type=int, default=7)
ap.add_argument("--frame", type=int, default=92)
ap.add_argument("--wrist", choices=("mount", "joe"), default="mount")
ap.add_argument("--tag", default="")
ap.add_argument("--warmup", type=int, default=40)
args = ap.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

simapp = AppLauncher(headless=True, enable_cameras=True, kit_args=KIT_ARGS).app

import cv2  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import scene as rscene  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from pxr import Gf  # noqa: E402

from sim.twin.calib.profile import twin_profile  # noqa: E402
from sim.twin.robot.with_mount import patch  # noqa: E402
from sim.twin.visuals.twin_scene import (
    TwinLook,
    apply_twin_visuals,
    color_correct,
    twin_scene_cfg,
    wrist_cam_rel,  # noqa: E402
)
from sim.twin.visuals.usd_sync import VisualSync, set_local  # noqa: E402

OUT = REPO / "sim/twin/out/robot"
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
# close-up eyes in the gripper_link frame (m), all looking at the mount / gripper body
VIEWS = {"side +x": (0.20, 0.05, -0.02), "camera side +y": (0.02, 0.24, -0.05),
         "above-behind": (0.06, 0.16, 0.16), "side -x": (-0.20, 0.08, -0.03)}
TARGET = np.array([0.0, 0.03, -0.035])


def recorded_state(ep: int, frame: int) -> np.ndarray:
    for f in sorted(DATA.glob("data/*/*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pandas()
        r = t[(t.episode_index == ep) & (t.frame_index == frame)]
        if len(r):
            return np.asarray(r["observation.state"].iloc[0], float)
    raise KeyError(f"episode {ep} frame {frame} not in {DATA}")


def look_rows(eye, target, up=(0, 0, 1.0)) -> np.ndarray:
    """USD camera row matrix (x right, y up, looks along -z) at `eye` looking at `target` (world)."""
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = x, y, -f, eye
    return m


patch(args.wrist)
prof = twin_profile()
rscene.ROBOT_USD = prof.robot_usd
state = recorded_state(args.ep, args.frame)
q = np.r_[np.deg2rad(state[:5]) + np.asarray(prof.joint_offsets[:5]), state[5] * prof.grip_scale + prof.grip_bias]
layout = rscene.load_layout(str(REPO / "sim/realism/scene_params.json"), "joe")
look = TwinLook()
cfg = twin_scene_cfg(1, layout, prof, look)
for k in range(len(VIEWS)):
    setattr(cfg, f"close{k}", CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Close{k}", update_period=0.0, height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.005, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.5, 0.5, 0.5), convention="opengl")))
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0", use_fabric=False))
scn = InteractiveScene(cfg)
print("visuals:", apply_twin_visuals(1, layout, prof, look), flush=True)
sim.reset()
scn.reset()
sync = VisualSync(scn, wrist_cam_rel=wrist_cam_rel())
robot = scn["robot"]
for name in [k for k in scn.rigid_objects]:  # park bricks / plates off the table: robot-only close-ups
    o = scn[name]
    o.write_root_pose_to_sim(torch.tensor([[5.0, 5.0, -2.0, 0, 0, 0, 1.0]], device=robot.device))
jidx = [robot.joint_names.index(j) for j in rscene.JOINTS]
qq = robot.data.default_joint_pos.clone()
qq[0, jidx] = torch.tensor(q, device=robot.device, dtype=torch.float32)


def step(n):
    for _ in range(n):
        robot.write_joint_state_to_sim(qq, torch.zeros_like(qq))
        robot.set_joint_position_target(qq)
        scn.write_data_to_sim()
        sim.step(render=False)
        scn.update(sim.get_physics_dt())
        sync()
        sim.render()
        scn.update(0.0)


step(args.warmup)
gi = robot.body_names.index("gripper_link")
x, y, z, w = robot.data.body_quat_w[0, gi].cpu().numpy()  # xyzw
G = np.eye(4)
G[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
G[:3, 3] = (robot.data.body_pos_w[0, gi] - scn.env_origins[0]).cpu().numpy()
stage = omni.usd.get_context().get_stage()
for k, eye_g in enumerate(VIEWS.values()):
    m = look_rows(G[:3, :3] @ eye_g + G[:3, 3], G[:3, :3] @ TARGET + G[:3, 3])
    set_local(stage.GetPrimAtPath(f"/World/envs/env_0/Close{k}"), Gf.Matrix4d(*m.ravel().tolist()))
step(8)
tiles = []
for k, name in enumerate(VIEWS):
    img = scn[f"close{k}"].data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
    tiles.append(cv2.putText(img.copy(), name, (10, 28), 0, 0.8, (255, 255, 0), 2))
w = color_correct(scn["wrist_cam"].data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8), "wrist")
tiles.append(cv2.putText(np.ascontiguousarray(w), f"sim wrist ({args.wrist})", (10, 28), 0, 0.8, (255, 255, 0), 2))
tiles.append(np.zeros_like(tiles[0]))
grid = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
OUT.mkdir(parents=True, exist_ok=True)
path = OUT / f"closeups_ep{args.ep:03d}_f{args.frame:03d}{args.tag}.jpg"
cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
print("wrote", path, "| state", np.round(state, 1).tolist(), flush=True)
simapp.close()
