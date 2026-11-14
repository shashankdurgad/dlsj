"""Robot surface points in the robot-base frame at recorded joint states (exactly the sim's kinematics).

For each sampled (episode, frame) of the cal2 dataset: q = deg2rad(state[:5]) + profile offsets, gripper
q = units * grip_scale + grip_bias (as sim/residual_env/env.py), written to the twin's SO-101; then every link's
world pose is read back and the link's mesh points (sampled once in the link frame) are moved with it.

    sim/twin/run.sh sim/twin/align/export_points.py --samples sim/twin/out/align/samples.json
Writes sim/twin/out/align/points.npz: pts (F, P, 3) float32, link (P,) int, link_names, samples, q (F, 6).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "sim/residual_env"))
from app import KIT_ARGS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True, help="json list of [episode, frame]")
ap.add_argument("--profile", default="joe_cal2")
ap.add_argument("--points-per-link", type=int, default=1500)
ap.add_argument("--out", default=str(REPO / "sim/twin/out/align/points.npz"))
args = ap.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

simapp = AppLauncher(headless=True, enable_cameras=False, kit_args=KIT_ARGS).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

sys.path.insert(0, str(REPO / "sim/twin/align"))
from calib import load_profile  # noqa: E402
from frames import states  # noqa: E402

USD = str(REPO / "sim/twin/assets/so101/usd/so101_new_calib/so101_new_calib.usda")
prof = load_profile(args.profile)
off = np.array(prof.joint_offsets[:5])

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0", gravity=(0.0, 0.0, 0.0)))
robot = Articulation(ArticulationCfg(
    prim_path="/World/Robot", spawn=sim_utils.UsdFileCfg(usd_path=USD),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=1e4, damping=1e2)}))
sim.reset()
names = robot.data.body_names
print("joints:", robot.data.joint_names, "| bodies:", names, flush=True)

# body prims and the meshes that belong to each (nearest ancestor with RigidBodyAPI)
stage = omni.usd.get_context().get_stage()
body_paths = {}
TRAV = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot"), TRAV):
    if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() in names:
        body_paths[str(p.GetPath())] = p.GetName()
xc = UsdGeom.XformCache(Usd.TimeCode.Default())
rng = np.random.default_rng(0)
local, link_of = [], []
meshes = [p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot"), TRAV) if p.IsA(UsdGeom.Mesh)]
print("bodies found:", body_paths, "| meshes:", len(meshes), [str(m.GetPath()) for m in meshes[:6]], flush=True)
for p in meshes:
    if not p.IsA(UsdGeom.Mesh) or "collision" in str(p.GetPath()).lower():
        continue
    a = p
    while a and str(a.GetPath()) not in body_paths:
        a = a.GetParent()
    if not a:
        continue
    pts = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), dtype=np.float64)
    if len(pts) == 0:
        continue
    M = np.array(xc.GetLocalToWorldTransform(p)).T  # USD row-vector -> column
    B = np.array(xc.GetLocalToWorldTransform(a)).T
    w = (np.c_[pts, np.ones(len(pts))] @ (np.linalg.inv(B) @ M).T)[:, :3]
    local.append((body_paths[str(a.GetPath())], w))
by_link = {}
for n, w in local:
    by_link.setdefault(n, []).append(w)
link_names = sorted(by_link, key=names.index)
L, idx = [], []
for i, n in enumerate(link_names):
    w = np.concatenate(by_link[n])
    w = w[rng.choice(len(w), min(args.points_per_link, len(w)), replace=False)]
    L.append(w)
    idx += [i] * len(w)
    print(f"link {n}: {len(np.concatenate(by_link[n]))} mesh vertices -> {len(w)} points", flush=True)
L = np.concatenate(L)
idx = np.array(idx)
body_ids = [names.index(n) for n in link_names]

samples = json.loads(Path(args.samples).read_text())
jn = robot.data.joint_names
order = [jn.index(j) for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
cache, out, Q = {}, [], []
for ep, fr in samples:
    if ep not in cache:
        cache[ep] = states(ep)
    s = cache[ep][fr]
    q6 = np.r_[np.deg2rad(s[:5]) + off, s[5] * prof.grip_scale + prof.grip_bias]
    q = torch.zeros(1, len(jn), device="cuda:0")
    q[0, order] = torch.tensor(q6, dtype=torch.float32, device="cuda:0")
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    robot.write_data_to_sim()
    sim.step(render=False)
    robot.update(sim.get_physics_dt())
    pose = robot.data.body_link_pose_w[0].cpu().numpy() if hasattr(robot.data, "body_link_pose_w") \
        else robot.data.body_state_w[0, :, :7].cpu().numpy()
    root = pose[0]  # base: pose 0 = base_link; express everything in the robot-base frame
    W = []
    for i, b in enumerate(body_ids):
        p, qw = pose[b, :3], pose[b, 3:7]  # Isaac Lab quaternion convention checked below
        W.append((p, qw))
    Q.append(q6)
    out.append(W)

# quaternion convention: Isaac Lab 3.0 is xyzw; detect by comparing link 0 at rest with the USD xform cache


def R_of(qv, xyzw):
    x, y, z, w = (qv if xyzw else np.r_[qv[1:], qv[0]])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


rest_B = np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath([k for k, v in body_paths.items() if v == link_names[0]][0]))).T
q0 = out[0][0][1]
xyzw = np.abs(R_of(q0, True) - rest_B[:3, :3]).max() < np.abs(R_of(q0, False) - rest_B[:3, :3]).max()
print("quaternion convention:", "xyzw" if xyzw else "wxyz", flush=True)
F = np.zeros((len(out), len(L), 3), np.float32)
for f, W in enumerate(out):
    for i, (p, qv) in enumerate(W):
        m = idx == i
        F[f, m] = L[m] @ R_of(qv, xyzw).T + p
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.out, pts=F, link=idx, link_names=np.array(link_names), samples=np.array(samples),
                    q=np.array(Q), profile=args.profile)
print(f"EXPORT_OK {F.shape} -> {args.out}", flush=True)
simapp.close()
