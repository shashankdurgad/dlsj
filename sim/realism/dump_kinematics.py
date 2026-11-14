"""Dump the SO-101 kinematic chain + finger/tip points + camera poses from Joe's scene (read-only) for numpy FK,
plus Isaac FK of gripper_link/moving_jaw at random joint vectors for verification.

    OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 ~/isaacsim-venv/bin/python dump_kinematics.py --out kin.json
"""
import argparse, json
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import numpy as np, omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
ROOT = Path("/home/joe-spark-2/pi05-isaac-so101")
omni.usd.get_context().open_stage(str(ROOT / "outputs/scene-final.usda"))
for _ in range(20): app.update()
st = omni.usd.get_context().get_stage()
M = lambda p: np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).tolist()
joints = {}
for p in st.Traverse():
    if p.IsA(UsdPhysics.RevoluteJoint) and str(p.GetPath()).startswith("/World/SO101"):
        j = UsdPhysics.RevoluteJoint(p)
        b0 = j.GetBody0Rel().GetTargets(); b1 = j.GetBody1Rel().GetTargets()
        q0 = j.GetLocalRot0Attr().Get(); q1 = j.GetLocalRot1Attr().Get()
        joints[p.GetName()] = {"body0": str(b0[0]) if b0 else None, "body1": str(b1[0]) if b1 else None,
            "pos0": list(j.GetLocalPos0Attr().Get()), "rot0": [q0.GetReal(), *q0.GetImaginary()],
            "pos1": list(j.GetLocalPos1Attr().Get()), "rot1": [q1.GetReal(), *q1.GetImaginary()],
            "axis": j.GetAxisAttr().Get(), "lower": j.GetLowerLimitAttr().Get(), "upper": j.GetUpperLimitAttr().Get()}
links = {p.GetName(): str(p.GetPath()) for p in st.Traverse() if str(p.GetPath()).startswith("/World/SO101") and p.GetName().endswith("_link")}
world = World(stage_units_in_meters=1.0); world.reset()
arm = SingleArticulation("/World/SO101"); arm.initialize()
names = arm.dof_names
def finger_pts(prefixes, frame_prim, n=300):
    Fi = np.linalg.inv(np.array(UsdGeom.Xformable(frame_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())))
    out = []
    for m in st.Traverse():
        if str(m.GetPath()).startswith("/World/SO101") and m.IsA(UsdGeom.Mesh) and m.GetName().startswith(prefixes):
            P = np.asarray(UsdGeom.Mesh(m).GetPointsAttr().Get(), float)
            P = P[np.linspace(0, len(P) - 1, min(n, len(P))).astype(int)]
            T = np.array(UsdGeom.Xformable(m).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            W = np.c_[P, np.ones(len(P))] @ T @ Fi
            out.append(W[:, :3])
    return np.concatenate(out).tolist()
grip = st.GetPrimAtPath(links["gripper_link"]); jaw = st.GetPrimAtPath(links["moving_jaw_so101_v1_link"])
q = arm.get_joint_positions().copy() * 0; arm.set_joint_positions(q); world.step(render=False); world.step(render=False)
res = {"dof_names": names, "joints": joints, "links": links, "root_world": M(st.GetPrimAtPath("/World/SO101")),
       "base_world": M(st.GetPrimAtPath(links["base_link"])),
       "fixed_finger_pts_gripper": finger_pts(("wrist_roll_follower",), grip),
       "jaw_pts_jaw": finger_pts(("moving_jaw",), jaw), "tip_local_gripper": [-0.008, 0.0, -0.10]}
for cam in ("TopCamera", "InspectCamera"):
    c = UsdGeom.Camera(st.GetPrimAtPath("/World/" + cam))
    res[cam] = {"world": M(c.GetPrim()), "focal": c.GetFocalLengthAttr().Get(), "hap": c.GetHorizontalApertureAttr().Get(),
                "vap": c.GetVerticalApertureAttr().Get()}
rng = np.random.default_rng(0); lim = np.asarray(arm._articulation_view.get_dof_limits())[0]; checks = []
for _ in range(12):
    q = rng.uniform(lim[:, 0], lim[:, 1]); arm.set_joint_positions(q); arm.set_joint_velocities(q * 0)
    world.step(render=False); world.step(render=False)
    av = arm._articulation_view
    lt = np.asarray(av._physics_view.get_link_transforms())[0]  # (links, 7): pos xyz, quat xyzw
    bn = list(av.body_names)
    def tf(name):
        r = lt[bn.index(name)]; x, y, z, w = r[3:7]
        m = np.eye(4); m[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        m[:3, 3] = r[:3]; return m.T.tolist()  # stored row-vector like USD
    checks.append({"q": arm.get_joint_positions().tolist(), "q_cmd": q.tolist(), "gripper_link": tf("gripper_link"),
                   "jaw": tf("moving_jaw_so101_v1_link"), "body_names": bn, "all": {n: tf(n) for n in bn}})
res["fk_checks"] = checks
Path(a.out).write_text(json.dumps(res)); print("KIN_DONE", list(joints), names); app.close()
