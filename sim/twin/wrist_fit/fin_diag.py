"""Which fin does the twin's wrist camera see? (Isaac) Renders audit samples, hides each fin group in turn and
projects the fin prims' bounding boxes into the wrist camera.

    NAME=dlsj-twin-wristfit sim/twin/run.sh sim/twin/wrist_fit/fin_diag.py [--samples 3,2,0]
Writes sim/twin/out/wrist_fit/fin_diag.{jpg,json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.audit.common import load_set  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--samples", default="3,2,0")
ap.add_argument("--layouts", default=str(REPO / "sim/twin/out/audit/runs/20261115-015346/layouts.json"))
a = ap.parse_args()
OUT = REPO / "sim/twin/out/wrist_fit"
OUT.mkdir(parents=True, exist_ok=True)
S = load_set()["samples"]
LAY = json.loads(Path(a.layouts).read_text())["layouts"]

from sim.twin.env.twin_env import TwinEnv, TwinEnvCfg  # noqa: E402

env = TwinEnv(TwinEnvCfg(cameras=("wrist",), meas_noise=False, max_relative_target=1000.0, render_passes=2))
import cv2  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from sim.twin.visuals.twin_scene import wrist_cam_rel  # noqa: E402

ENV0 = "/World/envs/env_0"
H, W = env.cfg.height, env.cfg.width


def vis(paths, on):
    for p in paths:
        im = UsdGeom.Imageable(env.stage.GetPrimAtPath(p))
        im.MakeVisible() if on else im.MakeInvisible()


def render():
    env.render()
    return env.render()["wrist"].copy()


rows, info = [], {}
for si in [int(x) for x in a.samples.split(",")]:
    s = S[si]
    env.reset(seed=s["ep"], layout=LAY.get(str(s["ep"])), joint_deg=list(s["state"]), render=False)
    fins = {"fixed": [], "jaw": []}
    allp = []
    for p in Usd.PrimRange(env.stage.GetPrimAtPath(ENV0)):
        n = p.GetName()
        if n.startswith("Finray_"):
            path = str(p.GetPath())
            allp.append(path)
            fins["jaw" if "moving_jaw" in path else "fixed"].append(path)
    base = render()
    cam = env.stage.GetPrimAtPath(f"{ENV0}/{wrist_cam_rel()}")
    M = np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    c = UsdGeom.Camera(cam)
    fx = c.GetFocalLengthAttr().Get() / c.GetHorizontalApertureAttr().Get() * W
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    proj = {}
    for k, ps in fins.items():
        pts = []
        for p in ps:
            r = bc.ComputeWorldBound(env.stage.GetPrimAtPath(p)).ComputeAlignedRange()
            mn, mx = np.array(r.GetMin()), np.array(r.GetMax())
            for corner in np.array(np.meshgrid(*zip(mn, mx, strict=False))).T.reshape(-1, 3):
                loc = (corner - M[3, :3]) @ np.linalg.inv(M[:3, :3])
                pts.append([W / 2 + fx * loc[0] / -loc[2], H / 2 - fx * loc[1] / -loc[2], -loc[2]])
        proj[k] = np.round(np.array(pts), 1).tolist()
    panels = [base]
    for k in ("fixed", "jaw"):
        vis(fins[k], False)
        img = render()
        vis(fins[k], True)
        d = (np.abs(img.astype(int) - base.astype(int)).max(-1) > 20)
        ov = base.copy()
        ov[d] = (0, 255, 0) if k == "fixed" else (255, 0, 255)
        cv2.putText(ov, f"{k} hidden: {int(d.sum())} px", (10, 30), 0, 0.8, (255, 255, 255), 2)
        panels.append(ov)
        info.setdefault(str(si), {})[k] = dict(n_prims=len(fins[k]), changed_px=int(d.sum()), proj=proj[k])
    info[str(si)]["cam_world"] = M.round(4).tolist()
    info[str(si)]["fx"] = fx
    rows.append(np.concatenate(panels, 1))
    print("[fin]", si, s["phase"], {k: (len(v), info[str(si)][k]["changed_px"]) for k, v in fins.items()},
          {k: np.array(v)[:, :2].mean(0).round(0).tolist() if v else None for k, v in proj.items()}, flush=True)
cv2.imwrite(str(OUT / "fin_diag.jpg"), cv2.cvtColor(cv2.resize(np.concatenate(rows, 0), None, fx=0.5, fy=0.5),
                                                     cv2.COLOR_RGB2BGR))
(OUT / "fin_diag.json").write_text(json.dumps(info, indent=1))
print("FIN_OK", flush=True)
env.close()
