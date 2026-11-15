"""Real vs sim close-frame images of gate runs: wrist and top camera at close - 8, close, close + 12, close + 40.

    sim/twin/run.sh sim/twin/grasp_gate/closeups.py --stage render --tag r3_W [--episodes 3 7 ...] [--n 6]
        (Isaac, TwinEnv cameras) the gate trace's recorded joints / brick pose written kinematically into the
        real-episode layout (scene/layout.layout_from_episode: plates, distractors), rendered -> closeups/<tag>/sim.npz
    sim/twin/run.sh sim/twin/grasp_gate/closeups.py --stage sheet --tag r3_W     (plain python, PyAV)
        -> closeups/<tag>/ep<NNN>.jpg: rows real wrist / sim wrist / real top / sim top (top cropped around the grasp)
Default episodes: the first --n failures and 2 successes of the run. The sim arm uses the run's joint offsets.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUTG = REPO / "sim/twin/out/grasp_gate"
DT = (-8, 0, 12, 40)

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["render", "sheet"], required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--episodes", type=int, nargs="*")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--warmup", type=int, default=8, help="renders after each reset before the first kept frame")
ap.add_argument("--passes", type=int, default=3, help="renders per kept frame (the last is kept)")
args = ap.parse_args()
out = OUTG / "closeups" / args.tag
out.mkdir(parents=True, exist_ok=True)
res = json.loads((OUTG / args.tag / "result.json").read_text())
per = {d["episode"]: d for d in res["episodes"]}
if args.episodes:
    show = args.episodes
else:
    fails = [e for e, d in per.items() if not d["success"]]
    show = fails[: args.n] + [e for e, d in per.items() if d["success"]][:2]


def crop_top(img, uv, half=(120, 90)):
    """Crop (2 * half) around pixel uv, scaled x2 (top camera close-up)."""
    import cv2

    h, w = img.shape[:2]
    x0 = int(np.clip(uv[0] - half[0], 0, w - 2 * half[0]))
    y0 = int(np.clip(uv[1] - half[1], 0, h - 2 * half[1]))
    return cv2.resize(img[y0:y0 + 2 * half[1], x0:x0 + 2 * half[0]], (320, 240))


if args.stage == "render":
    from sim.twin.env.twin_env import TwinEnv, TwinEnvCfg
    from sim.twin.rl.grasp_env import load_offsets
    from sim.twin.scene.layout import layout_from_episode

    tr = np.load(OUTG / args.tag / "trace.npz")
    row = {int(e): k for k, e in enumerate(tr["episodes"])}
    env = TwinEnv(TwinEnvCfg(cameras=("top", "wrist"), randomize=False, meas_noise=False, layout="sample"))
    import torch

    R = env.residual
    jo = res["summary"]["args"].get("joint_offsets")
    if jo:
        R.off = torch.tensor(load_offsets(str(REPO / jo)), dtype=torch.float32, device=env.device)
    o = env.scene.env_origins[:1]
    imgs, meta = {}, {}
    for ep in show:
        k = row[ep]
        cf = int(tr["cf"][k] - tr["start"][k])
        q, bp, bq = tr["q"][k], tr["bp"][k], tr["bq"][k]
        try:
            lay = layout_from_episode(ep)
        except Exception as ex:  # noqa: BLE001 - a missing layout only loses the clutter
            print(ep, "no layout:", ex, flush=True)
            lay = None
        env.reset(seed=0, joint_deg=q[0], brick_pose=(float(bp[0, 0]), float(bp[0, 1]), 0.0), render=False,
                  layout=lay)
        for _ in range(args.warmup):  # new materials / textures need a few rendered frames to settle
            env.render()
        for t in DT:
            f = int(np.clip(cf + t, 0, len(q) - 1))
            full = R.robot.data.default_joint_pos[:1].clone()
            full[:, R.jidx] = R.to_sim(torch.as_tensor(q[f], device=env.device)[None])
            R.robot.write_joint_state_to_sim(full, torch.zeros_like(full))
            R.robot.set_joint_position_target(full)
            pose = torch.cat([torch.as_tensor(bp[f], device=env.device)[None] + o,
                              torch.as_tensor(bq[f], device=env.device)[None]], -1)
            R.brick.write_root_pose_to_sim(pose)
            R.brick.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(1.0 / 120)
            for _ in range(args.passes):
                im = env.render()
            for cam in ("wrist", "top"):
                imgs[f"{ep}_{cam}_{t}"] = im[cam]
        meta[str(ep)] = {"cf_frame": int(tr["cf"][k]), "outcome": per[ep]["outcome"],
                         "fin_tip_mm": per[ep]["fin_tip_above_table_mm"], "tip_close": tr["tip"][k, cf].tolist()}
        print(ep, meta[str(ep)], flush=True)
    np.savez_compressed(out / "sim.npz", **imgs)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    env.close()
else:
    import cv2

    from sim.twin.align import frames as F
    from sim.twin.grasp_gate.rung0 import cameras, project

    sim = np.load(out / "sim.npz")
    meta = json.loads((out / "meta.json").read_text())
    cam = cameras()["fit"]
    for ep_s, m in meta.items():
        ep, cf = int(ep_s), m["cf_frame"]
        idx = [max(cf + t, 0) for t in DT]
        rw, rt = F.video_frames(ep, "wrist", idx), F.video_frames(ep, "top", idx)
        uv = project(cam, np.array(m["tip_close"]))[0]
        rows = []
        for lab in ("real wrist", "sim wrist", "real top", "sim top"):
            cam_name = lab.split()[1]
            cells = []
            for t, i in zip(DT, idx, strict=True):
                if lab.startswith("real"):
                    im = (rw if cam_name == "wrist" else rt)[i]
                else:
                    im = sim[f"{ep}_{cam_name}_{t}"]
                im = crop_top(im, uv) if "top" in lab else cv2.resize(im, (320, 240))
                im = np.ascontiguousarray(im)
                cv2.putText(im, f"{lab} close{t:+d}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cells.append(im)
            rows.append(np.concatenate(cells, 1))
        sheet = np.concatenate(rows, 0)
        hdr = np.zeros((24, sheet.shape[1], 3), np.uint8)
        cv2.putText(hdr, f"{args.tag} ep {ep}: sim {m['outcome']} | fin tip {m['fin_tip_mm']} mm above table at close",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"ep{ep:03d}.jpg"), np.concatenate([hdr, sheet], 0)[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", out / f"ep{ep:03d}.jpg", flush=True)
