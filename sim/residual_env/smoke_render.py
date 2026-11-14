"""Rendering smoke test for a new machine (e.g. a Modal container): do the wrist / top / overview cameras render the
simulated scene and FOLLOW the arm (not blank, not frozen), and how fast is the env with cameras?

    python smoke_render.py --out /tmp/smoke [--num_envs 8] [--ticks 90]

Writes smoke.json (verdict + timings) and frames_t0.jpg / frames_t1.jpg (overview | wrist | top of env 0).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="smoke_render")
ap.add_argument("--num_envs", type=int, default=8)
ap.add_argument("--ticks", type=int, default=90)
args = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
t_launch = time.perf_counter()
from app import launch  # noqa: E402

app = launch(cameras=True)
launch_s = time.perf_counter() - t_launch

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from app import make  # noqa: E402
from bases import ScriptedFailBase  # noqa: E402
from demo_lib import demo_cfg  # noqa: E402

res = {"launch_s": round(launch_s, 1), "num_envs": args.num_envs, "gpu": torch.cuda.get_device_name(0)}
base = ScriptedFailBase()
cfg = demo_cfg(args.num_envs, cameras=True)
cfg.overview = True
t = time.perf_counter()
sim, scene, layout, env = make(cfg, base)
base.env = env
res["make_s"] = round(time.perf_counter() - t, 1)
ids = torch.arange(env.n, device=env.device)


def grab():
    ov = env.overview_frames(ids).cpu().numpy()
    im = env.images(ids)
    return ov, im["wrist"].cpu().numpy(), im["top"].cpu().numpy()


env.reset()
t = time.perf_counter()
f0 = grab()
res["first_render_s"] = round(time.perf_counter() - t, 2)
q0 = env.robot.data.joint_pos.clone()
t = time.perf_counter()
for _ in range(args.ticks):
    env.step(None)
torch.cuda.synchronize()
res["physics_ticks_per_s"] = round(args.ticks * env.n / (time.perf_counter() - t), 1)
t = time.perf_counter()
for _ in range(5):
    f1 = grab()
torch.cuda.synchronize()
res["render_wrist_top_overview_s"] = round((time.perf_counter() - t) / 5, 3)
res["joint_motion_rad"] = round(float((env.robot.data.joint_pos - q0).abs().max()), 3)
for k, a, b in zip(("overview", "wrist", "top"), f0, f1):
    res[f"{k}_mean"] = round(float(b.mean()), 1)
    res[f"{k}_std"] = round(float(b.std()), 1)
    res[f"{k}_diff_t0_t1"] = round(float(np.abs(b.astype(np.float32) - a.astype(np.float32)).mean()), 2)
blank = any(res[f"{k}_std"] < 3 for k in ("overview", "wrist", "top"))
frozen = res["joint_motion_rad"] > 0.05 and max(res[f"{k}_diff_t0_t1"] for k in ("wrist", "top")) < 0.5
res["verdict"] = "BLANK" if blank else ("FROZEN" if frozen else "OK")
for tag, f in (("t0", f0), ("t1", f1)):
    ov = cv2.resize(f[0][0], (640, 360))
    cams = cv2.resize(np.concatenate([f[1][0], f[2][0]], 1), (640, 240))  # wrist | top
    row = np.concatenate([ov, cams], 0)
    cv2.imwrite(str(out / f"frames_{tag}.jpg"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
(out / "smoke.json").write_text(json.dumps(res, indent=1))
print("SMOKE_RESULT", json.dumps(res), flush=True)
app.close()
