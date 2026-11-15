"""Fin-ray grasp bench (gravity off): a 31.7 mm brick between the open fins at several depths, the jaw commanded
closed past contact (as the servo does). Reports where the jaw stalls (reading in gripper units), how far the fins
sink into the brick (compliant contact = the TPU give), and the 120 Hz buzz of the jaw and brick after it settles.

    sim/twin/run.sh sim/twin/gripper/bench.py --tag k4000 [--efforts 0.4 0.7] \
        [--set gripper.finray.material.compliant_contact_stiffness=4000]
    sim/twin/run.sh sim/twin/gripper/bench.py --tag e --empty 2.9 0 -3   # closed EMPTY at these commands (units)
    sim/twin/run.sh sim/twin/gripper/bench.py --tag lat --depths 15 --lateral 0 10 20 30 [--yaw 0 45] --lift_g 2
        # brick offset along gripper y (the fin-width axis) / yawed about gripper z; --lift_g pulls the brick toward
        # the fin tips with that many g after the squeeze (1 s) and reports whether it stays (`held`)
Writes sim/twin/out/gripper/bench_<tag>.json (--out to change the folder). Every case also reports the fin tip-to-tip
distance (hinged fins: the distal segments' tips; rigid fins: the CAD tips).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--tag", default="run")
ap.add_argument("--efforts", type=float, nargs="*", default=[], help="gripper effort limits (N m) to test")
ap.add_argument("--target_units", type=float, default=2.7, help="closing command (real hold command ~2.7)")
ap.add_argument("--dr_check", action="store_true", help="apply one physics DR draw, print the fin stiffness, exit")
ap.add_argument("--empty", type=float, nargs="*", default=None,
                help="closed-empty bench: no brick, one env per closing command (gripper units)")
ap.add_argument("--depths", type=float, nargs="*", default=[], help="brick centre above the fixed fin tip (mm)")
ap.add_argument("--out", default="sim/twin/out/gripper", help="output folder (repo-relative)")
ap.add_argument("--lateral", type=float, nargs="*", default=[], help="brick offsets along gripper y (mm)")
ap.add_argument("--yaw", type=float, nargs="*", default=[], help="brick yaw about gripper z (deg)")
ap.add_argument("--lift_g", type=float, default=0.0,
                help="after the squeeze, pull the brick toward the fin tips with this many g for 1 s (0 = off)")
args = ap.parse_args()
sys.path.insert(0, str(REPO / "sim/residual_env"))
from app import launch

app = launch(cameras=False)
import isaaclab.sim as sim_utils
import numpy as np
import scene as rscene
import torch
from isaaclab.scene import InteractiveScene

from sim.twin.calib.profile import twin_profile
from sim.twin.gripper import model
from sim.twin.physics import apply as phys

phys.load_params(None, args.set)
prof = phys.patch_profile(twin_profile())
layout = rscene.load_layout(str(REPO / "sim/realism/scene_params.json"))
DEPTHS = [d / 1e3 for d in args.depths] or [0.006, 0.012, 0.020, 0.030, 0.040]  # brick centre above the tip (m)
EFFORTS = args.efforts or [phys.grip_effort()]
EMPTY = args.empty is not None
CASES = [(0.02, e, u) for e in EFFORTS for u in (args.empty or [args.target_units])] if EMPTY else \
    [(d, e, args.target_units) for e in EFFORTS for d in DEPTHS]
CASES = [(*c, la / 1e3, yw) for c in CASES for yw in (args.yaw or [0.0]) for la in (args.lateral or [0.0])]
n = len(CASES)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0", gravity=(0.0, 0.0, 0.0)))
scfg = phys.patch_scene_cfg(rscene.build_scene_cfg(n, layout, False, profile=prof), prof)
scene = InteractiveScene(scfg)
print("post_spawn", json.dumps(phys.post_spawn(scene)), flush=True)
sim.reset()
scene.reset()
dev = sim.device
rob, brick = scene["robot"], scene["brick"]
gi = rob.joint_names.index("gripper")
if args.dr_check:
    draw = phys.apply_physics_dr(scene, phys.sample_physics_dr(np.random.default_rng(0)))
    fin_ids = [i for i, nm in enumerate(rob.joint_names) if nm.startswith("fin_")]
    k = getattr(rob.data.joint_stiffness, "torch", rob.data.joint_stiffness)
    print("DR_CHECK", json.dumps(draw), "fin stiffness env0", [round(float(v), 3) for v in k[0, fin_ids]], flush=True)
    raise SystemExit(0)
bg = rob.body_names.index("gripper_link")
ids = torch.arange(n, device=dev)
rob.write_joint_effort_limit_to_sim_index(limits=torch.tensor([[e] for _, e, *_r in CASES], device=dev), joint_ids=[gi],
                                          env_ids=ids)
jp = rob.data.default_joint_pos.clone()
jp[:, gi] = 0.5
rob.write_joint_state_to_sim(jp, torch.zeros_like(jp))
rob.set_joint_position_target(jp)
for _ in range(5):
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(1 / 120)


def qapply(q, v):
    """Rotate v by quaternion q (xyzw)."""
    xyz, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


tip_z = float(model.geometry()["fixed_tip"][2])
tip_y = float(model.geometry()["fixed_tip"][1])
local = torch.tensor([[model.fixed_face_x(tip_z + d) + model.BRICK_W / 2 + 0.0005, tip_y, tip_z + d]
                      for d, *_r in CASES], device=dev, dtype=torch.float32)
local[:, 1] += torch.tensor([c[3] for c in CASES], device=dev)
if EMPTY:  # brick parked 0.3 m to the side (along the jaw axis), clear of the fins
    local[:, 1] = 0.3
pg, qg = rob.data.body_pos_w[:, bg], rob.data.body_quat_w[:, bg]
half = torch.tensor([[0.0, 0.0, np.sin(np.radians(c[4]) / 2), np.cos(np.radians(c[4]) / 2)] for c in CASES],
                    device=dev, dtype=torch.float32)  # yaw about gripper z (xyzw)
qb = torch.cat([qg[:, 3:] * half[:, :3] + half[:, 3:] * qg[:, :3] + torch.cross(qg[:, :3], half[:, :3], dim=-1),
                qg[:, 3:] * half[:, 3:] - (qg[:, :3] * half[:, :3]).sum(-1, keepdim=True)], -1)
brick.write_root_pose_to_sim(torch.cat([pg + qapply(qg, local), qb], -1), env_ids=ids)
brick.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=ids)
q_cmd_t = torch.tensor([u * prof.grip_scale + prof.grip_bias for _, _, u, *_r in CASES], device=dev)
q_cmd = float(q_cmd_t[0])
TIP_OFF = {}


def tip_points():
    """World points of the two fin tips (hinged: on the distal segment bodies; rigid: on the links)."""
    geo = model.geometry()
    bend = phys.fin_bend()
    bj = rob.body_names.index("moving_jaw_so101_v1_link")
    pts = {}
    if bend is None:
        for w, b in (("fixed", bg), ("jaw", bj)):
            loc = torch.tensor(geo[f"{w}_tip"], device=dev, dtype=torch.float32).expand(n, 3)
            pts[w] = rob.data.body_pos_w[:, b] + qapply(rob.data.body_quat_w[:, b], loc)
        return pts
    h = int(bend["hinges"])
    last = f"TwinFinSeg_{model.SEG_NAMES[h - 1]}"
    if not TIP_OFF:
        for w in ("fixed", "jaw"):
            ch = model.segments(w, h, float(phys._p()["gripper"]["finray"].get("extend", 0.0)),
                                bool(bend.get("slices")), float(bend.get("root_len", 0.012)))
            TIP_OFF[w] = torch.tensor(geo[f"{w}_tip"] - ch[-1]["origin"], device=dev, dtype=torch.float32)
        # the fixed fin's distal body sits on the fixed side of the gripper (smaller x in gripper_link)
        cand = [i for i, nm in enumerate(rob.body_names) if nm in (last, last + "_0")]  # both fins (Isaac dedups)
        conj0 = torch.cat([-qg[:, :3], qg[:, 3:]], -1)
        xs = [float(qapply(conj0, rob.data.body_pos_w[:, i] - pg)[0, 0]) for i in cand]
        TIP_OFF["bodies"] = (cand[int(np.argmin(xs))], cand[int(np.argmax(xs))])
        print("fin tip bodies", [rob.body_names[i] for i in TIP_OFF["bodies"]], [round(x * 1e3, 1) for x in xs],
              flush=True)
    for w, b in zip(("fixed", "jaw"), TIP_OFF["bodies"], strict=True):
        pts[w] = rob.data.body_pos_w[:, b] + qapply(rob.data.body_quat_w[:, b], TIP_OFF[w].expand(n, 3))
    return pts


hist, gaps = [], []
for t in range(300):
    tgt = jp.clone()
    tgt[:, gi] = torch.clamp(q_cmd_t, min=0.5 - 0.03 * (t + 1))  # ~3.6 rad/s closing, like the servo
    rob.set_joint_position_target(tgt)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(1 / 120)
    pgb, qgb = rob.data.body_pos_w[:, bg], rob.data.body_quat_w[:, bg]
    conj = torch.cat([-qgb[:, :3], qgb[:, 3:]], -1)
    rel = qapply(conj, brick.data.root_pos_w - pgb)
    hist.append(torch.cat([rob.data.joint_pos[:, gi:gi + 1], rel], -1).cpu().numpy())
    tp = tip_points()
    gaps.append(torch.linalg.norm(tp["fixed"] - tp["jaw"], dim=-1).cpu().numpy())
h = np.stack(hist)
gp = np.stack(gaps)
pull = None
if args.lift_g and not EMPTY:  # pull test: the brick's weight x lift_g toward the fin tips (gripper -z), 1 s
    m = float(phys._p()["bricks"]["mass_kg"]["2x2"])
    down = qapply(rob.data.body_quat_w[:, bg], torch.tensor([[0.0, 0.0, -1.0]], device=dev).expand(n, 3))
    brick.permanent_wrench_composer.set_forces_and_torques_index(
        forces=(down * m * 9.81 * args.lift_g)[:, None, :].contiguous(), body_ids=[0], env_ids=ids, is_global=True)
    for _t in range(120):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(1 / 120)
    pgb, qgb = rob.data.body_pos_w[:, bg], rob.data.body_quat_w[:, bg]
    conj = torch.cat([-qgb[:, :3], qgb[:, 3:]], -1)
    pull = qapply(conj, brick.data.root_pos_w - pgb).cpu().numpy()
fin_ids = [i for i, nm in enumerate(rob.joint_names) if nm.startswith("fin_")]  # bending-fin hinges, if any
fin_q = rob.data.joint_pos[:, fin_ids].cpu().numpy() if fin_ids else np.zeros((n, 0))
print("joint_names", rob.joint_names, flush=True)
out = []
for k, (d, e, u, la, yw) in enumerate(CASES):
    tail = h[-60:, k]
    q_end, rel = float(tail[:, 0].mean()), tail[:, 1:].mean(0)
    depth_end = float(rel[2]) - tip_z
    tip_gap = {"tip_gap_mm": round(float(gp[-60:, k].mean()) * 1e3, 2),
               "tip_gap_p2p_um": round(float(np.ptp(gp[-60:, k])) * 1e6, 1)}
    if EMPTY:
        geo = model.geometry()
        rigid = float(np.linalg.norm(geo["fixed_tip"] - model.jaw_to_grip(geo["jaw_tip"], q_end)))
        out.append({"cmd_units": u, "effort": e, "q_end": round(q_end, 4),
                    "reading_units": round((q_end - prof.grip_bias) / prof.grip_scale, 2), **tip_gap,
                    "rigid_tip_gap_mm": round(rigid * 1e3, 2),
                    "q_buzz_p2p_mrad": round(float(np.ptp(tail[:, 0])) * 1e3, 3),
                    "fin_deg": {rob.joint_names[i]: round(float(np.degrees(fin_q[k, j])), 1)
                                for j, i in enumerate(fin_ids)}})
        print(out[-1], flush=True)
        continue
    extra = {}
    if args.lateral or args.yaw:
        extra = {"lateral_mm": round(la * 1e3, 1), "yaw_deg": yw}
    if pull is not None:
        slip = float(np.linalg.norm(pull[k] - rel))
        extra.update(pull_slip_mm=round(slip * 1e3, 1), held=bool(slip < 0.005))
    out.append({"depth_mm": round(d * 1e3, 1), "effort": e, "q_stall": round(q_end, 4), **tip_gap, **extra,
                "reading_units": round((q_end - prof.grip_bias) / prof.grip_scale, 2),
                "depth_end_mm": round(depth_end * 1e3, 1),
                "sink_total_mm": round((model.BRICK_W - model.gap(q_end, depth_end)) * 1e3, 2),
                "q_buzz_p2p_mrad": round(float(np.ptp(tail[:, 0])) * 1e3, 3),
                "brick_buzz_rms_um": round(float(np.sqrt(((tail[:, 1:] - rel) ** 2).sum(1).mean())) * 1e6, 1),
                "brick_moved_mm": round(float(np.linalg.norm(rel - h[0, k, 1:])) * 1e3, 1),
                "fin_deg": {rob.joint_names[i]: round(float(np.degrees(fin_q[k, j])), 1)
                            for j, i in enumerate(fin_ids)}})
    print(out[-1], flush=True)
res = {"q_cmd": round(q_cmd, 4), "grip_bias": prof.grip_bias, "set": args.set, "tip_local": list(phys.tip_local()),
       "cases": out}
dst = REPO / args.out
dst.mkdir(parents=True, exist_ok=True)
tmp = dst / f".bench_{args.tag}.json.tmp"
tmp.write_text(json.dumps(res, indent=1))
tmp.replace(dst / f"bench_{args.tag}.json")
print("BENCH", json.dumps(res), flush=True)
app.close()
