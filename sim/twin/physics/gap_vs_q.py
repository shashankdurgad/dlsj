"""Jaw gap vs gripper joint angle in the twin (PhysX joint frame, not the USD authored pose): where do the fitted
CAD pads (apply.py, pads_cad) hold a 31.7 mm Duplo, and which gripper bias makes that read the real 5.75?

    sim/twin/run.sh sim/twin/physics/gap_vs_q.py
Writes sim/twin/out/physics/gap_vs_q.json.
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
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from sim.twin.calib.profile import twin_profile  # noqa: E402
from sim.twin.physics import apply as phys  # noqa: E402

phys.load_params()
prof = twin_profile()
layout = rscene.load_layout(str(REPO / "sim/realism/scene_params.json"))
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0", gravity=(0.0, 0.0, 0.0)))
scfg = phys.patch_scene_cfg(rscene.build_scene_cfg(1, layout, False, profile=prof), prof)
scene = InteractiveScene(scfg)
print("post_spawn", phys.post_spawn(scene))
stage = omni.usd.get_context().get_stage()
root = "/World/envs/env_0/Robot/" + rscene.GRIPPER_LINK.split("/", 1)[1]
grip_p = stage.GetPrimAtPath(root)
jaw_p = stage.GetPrimAtPath(root + "/moving_jaw_so101_v1_link")
pad_m = stage.GetPrimAtPath(str(jaw_p.GetPath()) + "/TwinMovingPad")
pad_f = stage.GetPrimAtPath(root + "/TwinFixedPad")


def corners(prim, frame):
    T = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * \
        UsdGeom.Xformable(frame).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    c = np.array([[x, y, z, 1] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    return (c @ np.asarray(T))[:, :3]


mov_in_jaw = corners(pad_m, jaw_p)  # pad corners in the jaw link frame (rigid)
fix_in_grip = corners(pad_f, grip_p)
authored_jaw_in_grip = np.asarray(UsdGeom.Xformable(jaw_p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) *
                                  UsdGeom.Xformable(grip_p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                                  .GetInverse())
sim.reset()
scene.reset()
rob = scene["robot"]
gi = rob.joint_names.index("gripper")
bg, bj = rob.body_names.index("gripper_link"), rob.body_names.index("moving_jaw_so101_v1_link")


def quat_mat(q):  # xyzw
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


x_fix = fix_in_grip[:, 0].max()
# the moving pad's inner face = the 4 corners nearest the fixed finger at the authored pose (q = 0)
_auth = np.c_[mov_in_jaw, np.ones(8)] @ authored_jaw_in_grip
INNER = np.argsort(_auth[:, 0])[:4]
rows = []
for q in np.linspace(-0.17, 0.6, 32).tolist() + [0.19, 0.2, 0.21, 0.22, 0.23]:
    jp = rob.data.default_joint_pos.clone()
    jp[:, gi] = float(q)
    rob.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    rob.set_joint_position_target(jp)
    for _ in range(3):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(1 / 120)
    pg, qg = rob.data.body_pos_w[0, bg].cpu().numpy(), rob.data.body_quat_w[0, bg].cpu().numpy()
    pj, qj = rob.data.body_pos_w[0, bj].cpu().numpy(), rob.data.body_quat_w[0, bj].cpu().numpy()
    Rg, Rj = quat_mat(qg), quat_mat(qj)
    if len(rows) == 0 or abs(q) < 0.02:
        rel = np.eye(4)
        rel[:3, :3] = (Rg.T @ Rj).T
        rel[3, :3] = Rg.T @ (pj - pg)
    w = mov_in_jaw @ Rj.T + pj  # world
    g = (w - pg) @ Rg  # gripper frame
    zlo = g[:, 2].min()
    # inner face of the moving pad at heights above its lowest corner
    def gap_at(dz, g=g, zlo=zlo):
        # interpolate the pad's inner face x at z = zlo + dz (inner face = the 4 corners with the smallest x)
        inner = g[INNER]
        zz, xx = inner[:, 2], inner[:, 0]
        A = np.c_[zz, np.ones(4)]
        k = np.linalg.lstsq(A, xx, rcond=None)[0]
        return float(k[0] * (zlo + dz) + k[1] - x_fix)
    rows.append({"q": round(float(q), 4), "real_q": round(float(rob.data.joint_pos[0, gi]), 4),
                 "gap_tip_mm": round(gap_at(0.0) * 1000, 2), "gap_5mm_mm": round(gap_at(0.005) * 1000, 2),
                 "gap_10mm_mm": round(gap_at(0.010) * 1000, 2), "gap_20mm_mm": round(gap_at(0.020) * 1000, 2)})
    print(rows[-1], flush=True)
q0 = rob.data.default_joint_pos[0, gi].item()
res = {"rows": rows, "authored_jaw_in_grip_rows": authored_jaw_in_grip.round(5).tolist(), "fixed_inner_x": x_fix}
for key in ("gap_tip_mm", "gap_5mm_mm", "gap_10mm_mm"):
    order = np.argsort([r["real_q"] for r in rows])
    qs = np.array([rows[i]["real_q"] for i in order])
    gs = np.array([rows[i][key] for i in order])
    qc = float(np.interp(31.7, gs, qs))
    res[f"q_contact_31.7mm_{key}"] = round(qc, 4)
    res[f"bias_for_hold_5.75_{key}"] = round(qc - 5.75 * prof.grip_scale, 4)
print("GAP", json.dumps({k: v for k, v in res.items() if k != "rows"}), flush=True)
(REPO / "sim/twin/out/physics/gap_vs_q.json").write_text(json.dumps(res, indent=1))
app.close()
