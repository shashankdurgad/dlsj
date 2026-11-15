"""Inspect the twin robot USD's gripper: prims, instancing, collision approximations, jaw geometry in the
gripper_link frame (to place finger pads), drive / limit properties after sim.reset().

    sim/twin/run.sh sim/twin/physics/inspect_gripper.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "sim/residual_env"))
from app import launch  # noqa: E402

app = launch(cameras=False)

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import scene as rscene  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from sim.twin.calib.profile import twin_profile  # noqa: E402

prof = twin_profile()
layout = rscene.load_layout(str(REPO / "sim/realism/scene_params.json"))
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
scene = InteractiveScene(rscene.build_scene_cfg(1, layout, False, profile=prof))
stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
grip = jaw = None
rows = []
for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
    path = str(p.GetPath())
    if "gripper_link" not in path:
        continue
    if p.GetName() == "gripper_link":
        grip = p
    if p.GetName() == "moving_jaw_so101_v1_link":
        jaw = p
    info = dict(path=path.replace("/World/envs/env_0/Robot/", ""), type=p.GetTypeName(), inst=p.IsInstance(),
                proxy=p.IsInstanceProxy(), coll=p.HasAPI(UsdPhysics.CollisionAPI),
                purpose=UsdGeom.Imageable(p).GetPurposeAttr().Get() if p.IsA(UsdGeom.Imageable) else None)
    if p.HasAPI(UsdPhysics.MeshCollisionAPI):
        info["approx"] = UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()
    rows.append(info)
for r in rows:
    print("PRIM", json.dumps(r), flush=True)

g_w = UsdGeom.Xformable(grip).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
g_inv = g_w.GetInverse()
geo = {}
for p in Usd.PrimRange(grip, Usd.TraverseInstanceProxies()):
    if not p.IsA(UsdGeom.Mesh):
        continue
    m = UsdGeom.Mesh(p)
    pts = np.array(m.GetPointsAttr().Get())
    if pts.size == 0:
        continue
    T = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * g_inv
    M = np.array(T)  # row-vector
    q = np.c_[pts, np.ones(len(pts))] @ M
    q = q[:, :3]
    sel = (q[:, 2] > -0.106) & (q[:, 2] < -0.07)
    d = dict(n=len(q), bbox_min=np.round(q.min(0), 4).tolist(), bbox_max=np.round(q.max(0), 4).tolist(),
             coll=p.HasAPI(UsdPhysics.CollisionAPI), purpose=UsdGeom.Imageable(p).GetPurposeAttr().Get())
    if sel.any():
        s = q[sel]
        d["tip_region"] = dict(x=[round(float(s[:, 0].min()), 4), round(float(s[:, 0].max()), 4)],
                               y=[round(float(s[:, 1].min()), 4), round(float(s[:, 1].max()), 4)])
        # inner faces: points with |x| small (near the gap). histogram of x
        h, e = np.histogram(s[:, 0], bins=40)
        d["x_hist"] = [(round(float(e[i]), 4), int(h[i])) for i in range(40) if h[i]]
    geo[str(p.GetPath()).replace("/World/envs/env_0/Robot/", "")] = d
for k, v in geo.items():
    print("MESH", k, json.dumps(v), flush=True)

sim.reset()
scene.reset()
rob = scene["robot"]
print("JOINTS", rob.joint_names, flush=True)
print("BODIES", rob.body_names, flush=True)
print("LIMITS", np.round(rob.data.joint_pos_limits[0].cpu().numpy(), 4).tolist(), flush=True)
for attr in ("joint_stiffness", "joint_damping", "joint_effort_limits", "joint_vel_limits", "joint_armature",
             "joint_friction_coeff"):
    v = getattr(rob.data, attr, None)
    if v is not None:
        print(attr.upper(), np.round(v[0].cpu().numpy(), 4).tolist(), flush=True)
import warp as wp  # noqa: E402

masses = np.round(wp.to_torch(rob.root_view.get_masses())[0].cpu().numpy(), 4).tolist()
print("MASSES", dict(zip(rob.body_names, masses, strict=False)), flush=True)
app.close()
