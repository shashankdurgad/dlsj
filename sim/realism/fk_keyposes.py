"""Stage A2 FK: recorded key joint states -> fingertip position in the sim/table frame (Joe's scene, read-only).

Uses Joe's conversion (camera-fit.json joint_offsets_rad): q = deg2rad(state) + offsets, gripper q = s/100 * upper.
The fingertip point is the harness's TIP_LOCAL (between the fingertips, gripper_link frame).

    OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 ~/isaacsim-venv/bin/python fk_keyposes.py \
        --joints joints.json --out fk.json
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--joints", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--keys", default="start,grasp,inspect,release")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

ROOT = Path("/home/joe-spark-2/pi05-isaac-so101")
omni.usd.get_context().open_stage(str(ROOT / "outputs/scene-final.usda"))
for _ in range(20):
    app.update()
stage = omni.usd.get_context().get_stage()
OFF = np.array(json.loads((ROOT / "assets/camera-fit.json").read_text())["joint_offsets_rad"], np.float32)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TIP_LOCAL = Gf.Vec3d(-0.008, 0.0, -0.10)
world = World(stage_units_in_meters=1.0)
world.reset()
arm = SingleArticulation("/World/SO101")
arm.initialize()
idx = [arm.dof_names.index(j) for j in JOINTS]
lim = np.asarray(arm._articulation_view.get_dof_limits())[0]
grip = next(p for p in stage.Traverse() if p.GetName() == "gripper_link" and str(p.GetPath()).startswith("/World/SO101"))
base = next(p for p in stage.Traverse() if p.GetName() == "base_link" and str(p.GetPath()).startswith("/World/SO101"))


def set_pose(deg):
    q = arm.get_joint_positions().copy()
    t = np.deg2rad(np.asarray(deg[:5], np.float32)) + OFF[:5]
    for k, j in enumerate(idx[:5]):
        q[j] = t[k]
    q[idx[5]] = np.clip(deg[5] / 100.0, 0, 1) * lim[idx[5], 1]
    arm.set_joint_positions(q)
    arm.set_joint_velocities(np.zeros_like(q))
    world.step(render=False)
    world.step(render=False)


def tip():
    m = UsdGeom.Xformable(grip).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m.Transform(TIP_LOCAL))


eps = json.loads(Path(args.joints).read_text())
keys = args.keys.split(",")
out = []
for i, e in enumerate(eps):
    r = {"set": e["set"], "episode": e["episode"], "bowl": e["bowl"]}
    for k in keys:
        if e.get(k) is None:
            continue
        set_pose(e[k])
        r[k + "_tip"] = tip().round(4).tolist()
    out.append(r)
    if i % 100 == 0:
        print("FK", i, r, flush=True)
bm = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
Path(args.out).write_text(json.dumps({"base_origin": list(bm.ExtractTranslation()), "episodes": out}))
print("FK_DONE", len(out), flush=True)
app.close()
