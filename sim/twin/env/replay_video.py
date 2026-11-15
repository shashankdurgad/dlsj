"""Real-vs-sim replay video of a recorded cal2 episode (top / wrist / inspect; real row above sim row).

Three stages, because importing PyAV in the Isaac process blacks out the cameras:
    python  sim/twin/env/replay_video.py real   --ep 18       # decode real frames + scene layout (plain python)
    sim/twin/run.sh sim/twin/env/replay_video.py sim --ep 18  # replay recorded actions in TwinEnv, render each tick
    python  sim/twin/env/replay_video.py encode --ep 18     # compose + H.264 mp4 -> sim/twin/out/env/replay_ep018.mp4
Scene from the real first top frame (sim/twin/scene/layout.layout_from_episode: plates, target colour / kind,
distractors, target bowl), target brick at the fingertip of the gripper-close frame (as validate_twin.py), no clamp,
no measurement noise.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/env"
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
CAMS = ("top", "wrist", "inspect")
W, H = 320, 240

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=("real", "sim", "encode"))
ap.add_argument("--ep", type=int, default=18)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--dr", action="store_true", help="sim stage: domain randomisation on (writes sim_dr.npz)")
ap.add_argument("--dr-seed", type=int, default=7)
args = ap.parse_args()
tmp = OUT / f"video_ep{args.ep:03d}"
tmp.mkdir(parents=True, exist_ok=True)


if args.stage == "real":
    import cv2

    from sim.twin.scene.layout import layout_from_episode
    from sim.twin.visuals.realdata import Dataset

    ds = Dataset(str(DATA))
    from sim.twin.align.frames import states

    T = len(states(args.ep, str(DATA)))
    frames = {c: ds.frames(args.ep, c, list(range(T))) for c in CAMS}
    np.savez_compressed(tmp / "real.npz", **{c: np.stack([cv2.resize(f, (W, H)) for f in frames[c]]) for c in CAMS})
    lay = layout_from_episode(args.ep, DATA)
    (tmp / "layout.json").write_text(json.dumps(dict(lay, T=T)))
    print("REAL_OK", T, "frames | plates", sorted(lay["plates"]), "| target", lay["target"], flush=True)

elif args.stage == "sim":
    import torch

    from sim.twin.env import TwinEnv, TwinEnvCfg

    env = TwinEnv(TwinEnvCfg(width=W * 2, height=H * 2, cameras=CAMS, meas_noise=False, max_relative_target=1000.0))
    from bases import load_episodes  # residual env (on sys.path once TwinEnv is built)

    from env import _quat_apply_xyzw

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
    brick = (float(tip[0]), float(tip[1]), float(torch.atan2(xa[1], xa[0])))
    lay = json.loads((tmp / "layout.json").read_text())
    env.reset(
        seed=args.dr_seed if args.dr else args.ep,
        joint_deg=S[0],
        brick_pose=brick,
        render=False,
        randomize=bool(args.dr),
        layout=lay,
    )
    import cv2

    sim = {c: np.zeros((T, H, W, 3), np.uint8) for c in CAMS}
    lifted = False
    for t in range(T):
        obs, _, _, _, info = env.step(A[t], render=True)
        lifted |= bool(info["lifted"])
        for c in CAMS:
            sim[c][t] = cv2.resize(obs["images"][c], (W, H), interpolation=cv2.INTER_AREA)
    np.savez_compressed(tmp / ("sim_dr.npz" if args.dr else "sim.npz"), **sim)
    print("SIM_OK", T, "ticks | brick lifted:", lifted, flush=True)
    env.close()

else:
    import av
    import cv2

    real, sim = np.load(tmp / "real.npz"), np.load(tmp / "sim.npz")
    dr = np.load(tmp / "sim_dr.npz") if (tmp / "sim_dr.npz").exists() else None
    T = min(len(real["top"]), len(sim["top"]), len(dr["top"]) if dr is not None else 10**9)

    def lab(img, text):
        img = img.copy()
        cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    path = OUT / f"replay_ep{args.ep:03d}.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=args.fps)
        s.width, s.height, s.pix_fmt = W * 3, H * (3 if dr is not None else 2), "yuv420p"
        s.options = {"crf": "23", "preset": "medium", "movflags": "+faststart"}
        for t in range(T):
            top = np.concatenate([lab(real[k][t], f"real {k}") for k in CAMS], 1)
            bot = np.concatenate([lab(sim[k][t], f"sim {k}") for k in CAMS], 1)
            rows = [top, bot] + (
                [np.concatenate([lab(dr[k][t], f"sim randomised {k}") for k in CAMS], 1)] if dr is not None else []
            )
            fr = np.concatenate(rows, 0)
            cv2.putText(
                fr,
                f"episode {args.ep}  t={t / args.fps:4.1f}s",
                (W * 3 - 230, fr.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            for p in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    print("ENCODE_OK", path, T, "frames", flush=True)
