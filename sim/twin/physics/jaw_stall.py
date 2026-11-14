"""Jaw stall bench (gravity off): a 31.7 mm brick placed between the open pads at different depths, the jaw commanded
closed past contact; where does it stop, and does the gripper effort matter? Decides whether the held reading is set
by contact geometry (real servo) or the pads sink into the brick.

    sim/twin/run.sh sim/twin/physics/jaw_stall.py [--set gripper.collider=sdf]
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--target_q", type=float, default=0.12)
ap.add_argument("--kinematic", type=int, default=0, help="kinematic brick (contacts must stop the jaw)")
args = ap.parse_args()
sys.path.insert(0, str(REPO / "sim/residual_env"))
from app import launch  # noqa: E402

app = launch(cameras=False)
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

import scene as rscene  # noqa: E402
from sim.twin.calib.profile import twin_profile  # noqa: E402
from sim.twin.physics import apply as phys  # noqa: E402

phys.load_params(None, args.set)
prof = twin_profile()
layout = rscene.load_layout(str(REPO / "sim/realism/scene_params.json"))
DEPTHS = [-0.080, -0.088, -0.095, -0.100, -0.105]  # brick centre z in the gripper frame (pad bottom ~ -0.104)
EFFORTS = [0.6, 2.9]
CASES = [(z, e) for e in EFFORTS for z in DEPTHS]
n = len(CASES)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0", gravity=(0.0, 0.0, 0.0)))
scfg = phys.patch_scene_cfg(rscene.build_scene_cfg(n, layout, False, profile=prof), prof)
if args.kinematic:
    scfg.brick.spawn.rigid_props.kinematic_enabled = True
scene = InteractiveScene(scfg)
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

_st = omni.usd.get_context().get_stage()
for p in Usd.PrimRange(_st.GetPrimAtPath("/World/envs/env_0/Brick")):
    print("BRICKPRIM", p.GetPath(), p.GetTypeName(), p.HasAPI(UsdPhysics.CollisionAPI),
          UsdGeom.Cube(p).GetSizeAttr().Get() if p.IsA(UsdGeom.Cube) else "",
          [str(o.GetOpName()) + str(o.Get()) for o in UsdGeom.Xformable(p).GetOrderedXformOps()] if p.IsA(UsdGeom.Xformable) else "",
          flush=True)
print("post_spawn", json.dumps(phys.post_spawn(scene)), flush=True)
sim.reset()
scene.reset()
dev = sim.device
rob, brick = scene["robot"], scene["brick"]
gi = rob.joint_names.index("gripper")
bg = rob.body_names.index("gripper_link")
ids = torch.arange(n, device=dev)
rob.write_joint_effort_limit_to_sim_index(limits=torch.tensor([[e] for _, e in CASES], device=dev), joint_ids=[gi],
                                          env_ids=ids)
# open jaw, arm in the init pose, lifted well above the table
jp = rob.data.default_joint_pos.clone()
jp[:, gi] = 0.45
rob.write_joint_state_to_sim(jp, torch.zeros_like(jp))
rob.set_joint_position_target(jp)
for _ in range(5):
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(1 / 120)


def qapply(q, v):  # xyzw
    xyz, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


pg, qg = rob.data.body_pos_w[:, bg], rob.data.body_quat_w[:, bg]
local = torch.tensor([[0.0079 - 0.0079 + 0.00795 + 0.0001, 0.0, z] for z, _ in CASES], device=dev)  # face at fixed pad
local[:, 0] = -0.0079 + 0.0317 / 2 + 0.0002
pos = pg + qapply(qg, local)
# the parked plates / distractors are far away; the table is below the init pose
brick.write_root_pose_to_sim(torch.cat([pos, qg], -1), env_ids=ids)
brick.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=ids)
hist = []
for t in range(120):
    tgt = jp.clone()
    tgt[:, gi] = max(args.target_q, 0.45 - 0.03 * (t + 1))  # ~3.6 rad/s closing, like the servo
    rob.set_joint_position_target(tgt)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(1 / 120)
    pgb, qgb = rob.data.body_pos_w[:, bg], rob.data.body_quat_w[:, bg]
    conj = torch.cat([-qgb[:, :3], qgb[:, 3:]], -1)
    rel = qapply(conj, brick.data.root_pos_w - pgb)
    hist.append(torch.cat([rob.data.joint_pos[:, gi:gi + 1], rel], -1).cpu().numpy())
h = np.stack(hist)
out = []
for k, (z, e) in enumerate(CASES):
    q_end, rel = float(h[-1, k, 0]), h[-1, k, 1:]
    out.append({"brick_z": z, "effort": e, "q_stall": round(q_end, 4), "brick_rel_end": np.round(rel, 4).tolist(),
                "brick_moved_mm": round(float(np.linalg.norm(rel - h[0, k, 1:])) * 1000, 1)})
    print(out[-1], flush=True)
res = {"target_q": args.target_q, "set": args.set, "cases": out}
(REPO / "sim/twin/out/physics" / f"jaw_stall{'_' + '_'.join(s.split('=')[1] for s in args.set) if args.set else ''}.json"
 ).write_text(json.dumps(res, indent=1))
print("STALL", json.dumps(res), flush=True)
app.close()
