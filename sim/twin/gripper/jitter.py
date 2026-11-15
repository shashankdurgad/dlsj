"""Brick jitter while grasped / held: is it physics (contact / solver) or the rendered copy (VisualSync)?

Replays a recorded cal2 episode in TwinEnv (as replay_video.py) and logs
  * the brick and gripper_link physics poses after EVERY PhysX substep (120 Hz),
  * at every rendered frame (30 Hz): the physics poses and the poses of the visual proxies the cameras see
    (TwinVis/Brick and RobotVisual/.../gripper_link read back from USD after the render).
Jitter = high-pass part (centred moving average over 0.1 s removed) of the brick pose in the gripper frame during
the hold (grasp settled .. inspection frame): RMS position (mm) and rotation (deg).

    sim/twin/run.sh sim/twin/gripper/jitter.py --ep 18 --tag before [--set gripper.collider=pads_cad ...] [--video 1]
    python sim/twin/gripper/jitter.py --encode before      # hold video from the saved frames (plain python, PyAV)
Writes sim/twin/out/gripper/jitter_<tag>.{json,npz} (+ jitter_<tag>.mp4 after --encode).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/gripper"
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
ap = argparse.ArgumentParser()
ap.add_argument("--ep", type=int, default=18)
ap.add_argument("--tag", default="run")
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--video", type=int, default=0)
ap.add_argument("--encode", default="")
ap.add_argument("--extra_hold", type=int, default=45, help="ticks to keep holding at the inspection pose")
args = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)

if args.encode:
    import av
    import cv2

    d = np.load(OUT / f"jitter_{args.encode}_frames.npz")
    fr = d["frames"]
    path = OUT / f"jitter_{args.encode}.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=30)
        s.width, s.height, s.pix_fmt = fr.shape[2], fr.shape[1], "yuv420p"
        s.options = {"crf": "20", "preset": "medium", "movflags": "+faststart"}
        for k, f in enumerate(fr):
            f = f.copy()
            cv2.putText(f, f"{args.encode}  hold frame {k}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    print("ENCODE_OK", path, len(fr))
    sys.exit(0)

from sim.twin.physics import apply as phys

phys.load_params(None, args.set)
import torch

from sim.twin.env import TwinEnv, TwinEnvCfg

W, H = 640, 480
env = TwinEnv(TwinEnvCfg(width=W, height=H, cameras=("wrist", "inspect"), meas_noise=False,
                         max_relative_target=1000.0))
from bases import load_episodes
from env import _quat_apply_xyzw
from pxr import Usd, UsdGeom

d = load_episodes(DATA)[args.ep]
A, S = d["action"], d["state"]
T, insp = len(A), int(d["inspection_frame"])
g = S[:insp, 5]
gf = next((f for f in range(1, len(g)) if g[f - 1] > 12 >= g[f] and (g[f:] <= 15).all()), None)
R = env.residual
R.noise.zero_()
env.reset(seed=args.ep, joint_deg=S[gf], brick_pose=(0.8, 0.5, 0.0), render=False)
tip = (R.tip_w() - env.scene.env_origins)[0]
q = R.robot.data.body_quat_w[:1, R.grip_body]
xa = _quat_apply_xyzw(q, torch.tensor([[1.0, 0, 0]], device=env.device))[0]
env.reset(seed=args.ep, joint_deg=S[0], brick_pose=(float(tip[0]), float(tip[1]), float(torch.atan2(xa[1], xa[0]))),
          render=False)
o = env.scene.env_origins[:1]
for k, obj in enumerate(list(R.bowl_objs) + list(R.distractors)):
    pose = torch.tensor([[0.9, -0.5 + 0.15 * k, 0.02, 0, 0, 0, 1.0]], device=env.device)
    pose[:, :3] += o
    obj.write_root_pose_to_sim(pose)
    obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
env.settle(2)

gi = R.jidx[5]
sub = []


def snap():
    b = R.brick.data
    return np.concatenate([b.root_pos_w[0].cpu().numpy(), b.root_quat_w[0].cpu().numpy(),
                           R.robot.data.body_pos_w[0, R.grip_body].cpu().numpy(),
                           R.robot.data.body_quat_w[0, R.grip_body].cpu().numpy(),
                           [float(R.robot.data.joint_pos[0, gi]), float(R.robot.data.joint_pos_target[0, gi]),
                            float(R.robot.data.joint_vel[0, gi])]])


def physics():  # ResidualEnv.physics with a log after every substep
    for _ in range(R.substeps):
        R.scene.write_data_to_sim()
        R.sim.step(render=False)
        R.scene.update(1.0 / R.cfg.physics_hz)
        sub.append(snap())


R.physics = physics
brick_vis = next(p for k, p, _ in env.sync.rigid[0] if k == "brick")
grip_vis = next(p for i, p in env.sync.links[0] if i == R.grip_body)


def usd_pose(prim):
    M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    qd = M.ExtractRotationQuat()
    return np.array([*M.ExtractTranslation(), *qd.GetImaginary(), qd.GetReal()])


rend, frames, lifted = [], [], []
n_ticks = min(T, insp + 1) + args.extra_hold
for t in range(n_ticks):
    a = A[min(t, insp)]
    obs, _, _, _, info = env.step(a, render=True)
    lifted.append(bool(info["lifted"]))
    rend.append(np.concatenate([snap()[:14], usd_pose(brick_vis), usd_pose(grip_vis)]))
    if args.video and t >= gf - 5:
        frames.append(np.concatenate([obs["images"]["wrist"], obs["images"]["inspect"]], 1))
sub, rend = np.stack(sub), np.stack(rend)


def qmat(q):
    x, y, z, w = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def rel(bp, bq, gp, gq):
    Rg, Rb = qmat(gq), qmat(bq)
    p = np.einsum("nji,nj->ni", Rg, bp - gp)
    Rr = np.einsum("nji,njk->nik", Rg, Rb)
    return p, Rr


def rotvec(Rr):
    from scipy.spatial.transform import Rotation
    ref = Rotation.from_matrix(Rr[len(Rr) // 2])
    return (ref.inv() * Rotation.from_matrix(Rr)).as_rotvec()


def highpass(x, win):
    k = np.ones(win) / win
    pad = np.pad(x, ((win // 2, win // 2), (0, 0)), mode="edge")
    return x - np.stack([np.convolve(pad[:, i], k, mode="valid")[: len(x)] for i in range(x.shape[1])], 1)


def stats(p, Rr, rate):
    win = max(3, int(round(0.1 * rate)) | 1)
    hp, hr = highpass(p, win), highpass(rotvec(Rr), win)
    return {"pos_rms_mm": round(float(np.sqrt((hp ** 2).sum(1).mean())) * 1e3, 4),
            "pos_p2p_mm": [round(float(v) * 1e3, 3) for v in np.ptp(hp, 0)],
            "rot_rms_deg": round(float(np.degrees(np.sqrt((hr ** 2).sum(1).mean()))), 4),
            "drift_mm": round(float(np.linalg.norm(p[-1] - p[0])) * 1e3, 3), "n": len(p)}


lifted = np.array(lifted)
t0 = gf + 20  # grasp settled
res = {"ep": args.ep, "tag": args.tag, "set": args.set, "grasp_frame": gf, "inspection_frame": insp,
       "lifted": bool(lifted.any()), "lifted_ticks": int(lifted.sum())}
windows = {"hold_moving": (t0, insp), "hold_static": (insp + 10, n_ticks)}
for name, (a, b) in windows.items():
    if b - a < 8:
        continue
    s = sub[a * 4:b * 4]
    p, Rr = rel(s[:, :3], s[:, 3:7], s[:, 7:10], s[:, 10:14])
    r = rend[a:b]
    pp, Rp = rel(r[:, :3], r[:, 3:7], r[:, 7:10], r[:, 10:14])
    pv, Rv = rel(r[:, 14:17], r[:, 17:21], r[:, 21:24], r[:, 24:28])
    res[name] = {"physics_120hz": stats(p, Rr, 120), "physics_at_render_30hz": stats(pp, Rp, 30),
                 "visual_at_render_30hz": stats(pv, Rv, 30),
                 "visual_minus_physics_world_mm_max": round(float(np.abs(r[:, 14:17] - r[:, :3]).max()) * 1e3, 4),
                 "visual_minus_physics_rel_mm_max": round(float(np.abs(pv - pp).max()) * 1e3, 4),
                 "brick_world_120hz": stats(s[:, :3], qmat(s[:, 3:7]), 120),
                 "gripper_world_120hz": stats(s[:, 7:10], qmat(s[:, 10:14]), 120),
                 "gripper_q_rms_mrad": round(float(highpass(s[:, 14:15], 13).std()) * 1e3, 4),
                 "gripper_q_minus_target_rad": round(float((s[:, 14] - s[:, 15]).mean()), 4),
                 "held_reading_units": round(float(((s[:, 14] - R.g_bias) / R.g_scale).mean()), 2),
                 "brick_rel_grip_mm": [round(float(v) * 1e3, 2) for v in p.mean(0)]}
print("JITTER", json.dumps(res), flush=True)
(OUT / f"jitter_{args.tag}.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(OUT / f"jitter_{args.tag}.npz", sub=sub, rend=rend, lifted=lifted)
if args.video and frames:
    np.savez_compressed(OUT / f"jitter_{args.tag}_frames.npz", frames=np.stack(frames))
env.close()
