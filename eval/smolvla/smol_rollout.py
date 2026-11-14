"""SmolVLA 3-stage benchmark (copy of sim/isaac_pi05/rollout.py: wrist+top only, stage switching, wrist video).
Original: π0.5 pick-and-present benchmark in Joe's SO-101 Isaac scene (read-only use of his USD; own Isaac instance).

Joe's conventions (pi05-isaac-so101/rollout_reference.py): state = rad2deg(q - joint_offsets), gripper = q/upper*100;
actions (50,6) -> target = deg2rad(a) + offsets, gripper = clip(a/100,0,1)*upper; sim paused while the policy thinks;
30 Hz control = 2 physics steps (1/60 s) per action. Policy runs in policy_worker.py (file IPC).
"""
import argparse, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ipc", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--chunks", type=int, default=10)
ap.add_argument("--jitter", type=float, default=0.015, help="brick xy randomisation (m, uniform +-)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--instruction", default="pick up the brick")
ap.add_argument("--execute", type=int, default=25, help="actions executed per chunk before re-planning")
ap.add_argument("--bowl", default="orange")
ap.add_argument("--video-every", type=int, default=2, help="save an overview frame every N actions")
args = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 640, "height": 480})

import numpy as np
from PIL import Image
import omni.usd, omni.timeline, omni.replicator.core as rep
from pxr import Gf, Usd, UsdGeom
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction

ROOT = Path("/home/joe-spark-2/pi05-isaac-so101")
args.out.mkdir(parents=True, exist_ok=True)
args.ipc.mkdir(parents=True, exist_ok=True)
omni.usd.get_context().open_stage(str(ROOT / "outputs/scene-final.usda"))
for _ in range(20): app.update()
stage = omni.usd.get_context().get_stage()
fit = json.loads((ROOT / "assets/camera-fit.json").read_text())
OFF = np.array(fit["joint_offsets_rad"], np.float32)
INIT_DEG = np.array(fit["initial_joint_state"], np.float32)
BRICK0 = np.array(fit["objects"]["target"]["position"], np.float64)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
INSPECT_CAM = np.array(UsdGeom.Xformable(stage.GetPrimAtPath("/World/InspectCamera"))
                       .ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())

world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
world.reset()
arm = SingleArticulation("/World/SO101"); arm.initialize()
idx = [arm.dof_names.index(j) for j in JOINTS]
lim = np.asarray(arm._articulation_view.get_dof_limits())[0]  # (n, 2) rad
GUP = float(lim[idx[5], 1])
brick = SingleRigidPrim("/World/Brick")
grip_prim = next(p for p in stage.Traverse() if p.GetName() == "gripper_link")
wrist_cam = next(str(p.GetPath()) for p in stage.Traverse() if p.GetName() == "WristCamera")
cams = {"wrist": wrist_cam, "top": "/World/TopCamera", "inspect": "/World/InspectCamera", "overview": "/World/OverviewCamera"}
ann = {}
for n, path in cams.items():
    rp = rep.create.render_product(path, (640, 480))
    a = rep.AnnotatorRegistry.get_annotator("rgb"); a.attach([rp]); ann[n] = a
timeline = omni.timeline.get_timeline_interface()

def grab(names, fresh=True):
    for _ in range(30):
        if fresh:
            rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
        out = {n: np.asarray(ann[n].get_data()) for n in names}
        if all(v.ndim == 3 for v in out.values()):
            return {n: v[:, :, :3].copy() for n, v in out.items()}
        fresh = True
    raise RuntimeError("annotator returned no image")

def q_full():
    return np.asarray(arm.get_joint_positions(), np.float32).copy()

def joint_state():
    q = q_full()[idx]
    s = np.rad2deg(q - OFF)
    s[5] = q[5] / GUP * 100
    return s.astype(np.float32)

def to_target(a):
    t = np.deg2rad(a).astype(np.float32) + OFF
    t[5] = np.clip(a[5] / 100, 0, 1) * GUP
    return np.clip(t, lim[idx, 0], lim[idx, 1])

TIP_LOCAL = Gf.Vec3d(-0.008, 0.0, -0.10)  # between the fingertips, gripper_link frame (measured jaw bounds)


def tip_pos():
    m = UsdGeom.Xformable(grip_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m.Transform(TIP_LOCAL))


def pos(prim):
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())

rng = np.random.default_rng(args.seed)
k = 0
summary = []
for ep in range(args.episodes):
    world.reset()
    arm.initialize(); brick.initialize()
    world.step(render=False)
    q = q_full()
    init = np.deg2rad(INIT_DEG).astype(np.float32) + OFF
    init[5] = INIT_DEG[5] / 100 * GUP
    q[idx] = init
    arm.set_joint_positions(q); arm.set_joint_velocities(np.zeros_like(q))
    arm.apply_action(ArticulationAction(joint_positions=q))
    b0 = BRICK0.copy(); b0[:2] += rng.uniform(-args.jitter, args.jitter, 2)
    yaw = rng.uniform(-0.3, 0.3)
    brick.set_world_pose(position=b0, orientation=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
    brick.set_linear_velocity(np.zeros(3)); brick.set_angular_velocity(np.zeros(3))
    for _ in range(30): world.step(render=True)
    b_start = np.array(brick.get_world_pose()[0])
    epdir = args.out / f"ep{ep}"; (epdir / "frames").mkdir(parents=True, exist_ok=True)
    lat, bpos, gdist, gstate, fi = [], [], [], [], 0
    stage, stage_switch = 0, []
    (epdir / "wrist").mkdir(exist_ok=True)
    t_ep = time.perf_counter()
    for c in range(args.chunks):
        timeline.pause()
        lifted_now = len(bpos) > 0 and (bpos[-1][2] - b_start[2]) > 0.03
        near_cam = len(bpos) > 0 and float(np.linalg.norm(bpos[-1] - INSPECT_CAM)) < 0.12
        if stage == 0 and lifted_now:
            stage, stage_switch = 1, stage_switch + [(c, "lifted")]
        elif stage == 1 and near_cam:
            stage, stage_switch = 2, stage_switch + [(c, "at_camera")]
        instr = [args.instruction, "bring the brick to the inspection camera", f"place the brick in the {args.bowl} bowl"][stage]
        imgs = grab(["wrist", "top"])
        if c == 0:
            for n, v in imgs.items(): Image.fromarray(v).save(epdir / f"obs0_{n}.png")
        tmp = args.ipc / f"obs-{k:04d}.tmp.npz"
        np.savez(tmp, state=joint_state(), instruction=np.asarray(instr), **imgs)
        os.replace(tmp, args.ipc / f"obs-{k:04d}.npz")
        af = args.ipc / f"actions-{k:04d}.npz"
        t_wait = time.monotonic() + 900
        while not af.exists():
            assert time.monotonic() < t_wait, "policy worker timed out"
            time.sleep(0.02)
        time.sleep(0.05)
        r = np.load(af); acts = r["actions"]; lat.append(float(r["latency_s"])); k += 1
        timeline.play()
        for j, a in enumerate(acts[:args.execute]):
            full = q_full(); full[idx] = to_target(a)
            arm.apply_action(ArticulationAction(joint_positions=full))
            world.step(render=False); world.step(render=True)
            bp = np.array(brick.get_world_pose()[0]); bpos.append(bp)
            gdist.append(float(np.linalg.norm(tip_pos() - bp))); gstate.append(joint_state())
            if j % args.video_every == 0:
                fr = grab(["overview", "wrist"], fresh=False)
                Image.fromarray(fr["overview"]).save(epdir / "frames" / f"{fi:05d}.jpg", quality=85)
                Image.fromarray(fr["wrist"]).save(epdir / "wrist" / f"{fi:05d}.jpg", quality=85)
                fi += 1
        print(f"EP {ep} CHUNK {c} stage {stage} lat {lat[-1]:.3f}s brick_z {bpos[-1][2]:.3f} tip {gdist[-1]:.3f}", flush=True)
    timeline.pause()
    for n, v in grab(["wrist", "top", "inspect", "overview"]).items():
        Image.fromarray(v).save(epdir / f"final_{n}.png")
    bpos = np.array(bpos); gs = np.array(gstate)
    lift = float(bpos[:, 2].max() - b_start[2])
    disp = float(np.max(np.linalg.norm(bpos[:, :2] - b_start[:2], axis=1)))
    closed_near = bool(np.any((gs[:, 5] < 20) & (np.array(gdist) < 0.03)))
    d_cam = float(np.min(np.linalg.norm(bpos - INSPECT_CAM, axis=1)))
    res = {"episode": ep, "brick_start": b_start.round(4).tolist(), "lift_m": round(lift, 4), "lifted": lift > 0.03,
           "max_xy_displacement_m": round(disp, 4), "min_fingertip_brick_dist_m": round(float(np.min(gdist)), 4),
           "gripper_closed_near_brick": closed_near, "min_brick_to_inspect_cam_m": round(d_cam, 3),
           "stages_reached": stage, "stage_switches": stage_switch, "latency_s": lat, "wall_s": round(time.perf_counter() - t_ep, 1),
           "joint_min": gs.min(0).round(1).tolist(), "joint_max": gs.max(0).round(1).tolist()}
    np.savez(epdir / "arrays.npz", brick=bpos, gripper_dist=np.array(gdist), state=gs)
    (epdir / "result.json").write_text(json.dumps(res, indent=1))
    summary.append(res)
    print("EPISODE_RESULT", json.dumps({k2: v for k2, v in res.items() if k2 != "latency_s"}), flush=True)
(args.ipc / "done").write_text("1")
(args.out / "summary.json").write_text(json.dumps(summary, indent=1))
print("ROLLOUT_DONE", flush=True)
app.close()
