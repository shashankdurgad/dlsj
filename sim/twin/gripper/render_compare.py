"""Wrist-camera check of the finger visuals: render the twin's wrist camera at recorded joint states and compare the
red finger silhouettes with the real frames (fit_wrist.py frames; no PyAV here).

    sim/twin/run.sh sim/twin/gripper/render_compare.py [--wrist profile|fitted] [--set gripper.collider=pads_cad]
--wrist fitted renders with the wrist-camera pose fitted on the fixed fin (fit_wrist.json) instead of the profile's
pose (calib/profile.py: sim/twin/robot/wrist_mount.json by default). Writes
sim/twin/out/gripper/wrist_compare_<tag>.jpg (real | sim | real with the sim's red outline) and
wrist_compare_<tag>.json (IoU of the red masks).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/gripper"
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
ap = argparse.ArgumentParser()
ap.add_argument("--wrist", choices=("profile", "fitted"), default="profile")
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--tag", default="")
ap.add_argument("--n", type=int, default=12)
args = ap.parse_args()
tag = args.tag or args.wrist

from sim.twin.calib import profile as tprofile
from sim.twin.gripper.fit_wrist import red_mask
from sim.twin.physics import apply as phys

phys.load_params(None, args.set)
if args.wrist == "fitted":
    fit = json.loads((OUT / "fit_wrist.json").read_text())
    _cams = tprofile._cams

    def _fitted_cams():
        c = _cams()
        c["wrist"] = (np.array(fit["cam_local_to_cv_fitted"]), np.array(fit["cam_position_local_fitted"]),
                      c["wrist"][2])
        return c

    tprofile._cams = _fitted_cams

from sim.twin.env import TwinEnv, TwinEnvCfg

env = TwinEnv(TwinEnvCfg(cameras=("wrist",), meas_noise=False, color_correct=True))
from bases import load_episodes

frames = np.load(OUT / "fit_wrist_frames.npz")
keys = sorted((k for k in frames.files if not k.endswith("_grip")), key=lambda k: float(frames[k + "_grip"]))
keys = [keys[i] for i in np.linspace(0, len(keys) - 1, args.n).astype(int)]
eps = load_episodes(DATA)
rows, ious = [], []
for key in keys:
    ep, fr = (int(v) for v in key.split("_"))
    state = eps[ep]["state"][fr]
    env.reset(seed=0, joint_deg=state, brick_pose=(0.8, 0.5, 0.0), render=False)
    env.residual.r[:] = env.torch.as_tensor(state, device=env.device, dtype=env.residual.r.dtype)
    sim = env.render()["wrist"]
    real = frames[key]
    ms, mr = red_mask(sim), red_mask(real)
    iou = float((ms & mr).sum() / max((ms | mr).sum(), 1))
    ious.append(iou)
    over = real.copy()
    cs, _ = cv2.findContours(ms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cs, -1, (0, 255, 0), 2)
    for img, text in ((real, f"real ep{ep} f{fr} grip {state[5]:.1f}"), (sim, f"sim ({tag})"),
                      (over, f"sim red outline, IoU {iou:.2f}")):
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    rows.append(np.concatenate([real, sim, over], 1))
half = (len(rows) + 1) // 2
pad = [np.zeros_like(rows[0])] * (2 * half - len(rows))
sheet = np.concatenate([np.concatenate(rows[:half], 0), np.concatenate(rows[half:] + pad, 0)], 1)
sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA)
cv2.imwrite(str(OUT / f"wrist_compare_{tag}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 88])
res = {"tag": tag, "wrist": args.wrist, "set": args.set, "red_iou_median": round(float(np.median(ious)), 3),
       "red_iou": [round(v, 3) for v in ious], "frames": keys}
(OUT / f"wrist_compare_{tag}.json").write_text(json.dumps(res, indent=1))
print("WRIST_COMPARE", json.dumps(res), flush=True)
env.close()
