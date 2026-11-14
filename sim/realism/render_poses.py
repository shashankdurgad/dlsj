"""Stage A2 camera check: render Joe's sim top/wrist cameras at recorded joint states (Joe's scene, read-only).

Input: extract_episodes.json (from extract_episodes.py). States are raw per dataset; cal2 states are converted to the old
calibration (minus recal_2026-11-14 offset) before Joe's conversion. Also writes the camera-frame pose of the gripper
link and finger points per render, for reprojection fitting.

    OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 ~/isaacsim-venv/bin/python render_poses.py \
        --episodes extract_episodes.json --out renders
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--keys", default="start,mid2,inspect")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 640, "height": 480})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

ROOT = Path("/home/joe-spark-2/pi05-isaac-so101")
omni.usd.get_context().open_stage(str(ROOT / "outputs/scene-final.usda"))
for _ in range(20):
    app.update()
stage = omni.usd.get_context().get_stage()
OFF = np.array(json.loads((ROOT / "assets/camera-fit.json").read_text())["joint_offsets_rad"], np.float32)
RECAL = np.array([20.35, -7.96, 0.18, 0.04, 0.0, 0.0])
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
world = World(stage_units_in_meters=1.0)
world.reset()
arm = SingleArticulation("/World/SO101")
arm.initialize()
idx = [arm.dof_names.index(j) for j in JOINTS]
lim = np.asarray(arm._articulation_view.get_dof_limits())[0]
wrist_cam = next(str(p.GetPath()) for p in stage.Traverse() if p.GetName() == "WristCamera")
ann = {}
for n, path in (("top", "/World/TopCamera"), ("wrist", wrist_cam)):
    rp = rep.create.render_product(path, (640, 480))
    a = rep.AnnotatorRegistry.get_annotator("rgb")
    a.attach([rp])
    ann[n] = a
# finger meshes (red) for 3D reprojection points
fingers = [p for p in stage.Traverse() if str(p.GetPath()).startswith("/World/SO101") and p.IsA(UsdGeom.Mesh)
           and p.GetName().startswith(("wrist_roll_follower", "moving_jaw"))]


def grab(a):
    for _ in range(30):
        rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
        d = np.asarray(a.get_data())
        if d.ndim == 3:
            return d[:, :, :3].copy()
    raise RuntimeError("no image")


def set_pose(deg):
    q = arm.get_joint_positions().copy()
    t = np.deg2rad(np.asarray(deg[:5], np.float32)) + OFF[:5]
    for k, j in enumerate(idx[:5]):
        q[j] = t[k]
    q[idx[5]] = np.clip(deg[5] / 100.0, 0, 1) * lim[idx[5], 1]
    arm.set_joint_positions(q)
    arm.set_joint_velocities(np.zeros_like(q))
    for _ in range(2):
        world.step(render=True)


def finger_points(n=400):
    pts = []
    for m in fingers:
        mesh = UsdGeom.Mesh(m)
        P = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
        if len(P) > n:
            P = P[np.linspace(0, len(P) - 1, n).astype(int)]
        T = np.array(UsdGeom.Xformable(m).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        W = np.c_[P, np.ones(len(P))] @ T  # row-vector convention
        pts.append(W[:, :3])
    return np.concatenate(pts, 0)


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
eps = json.loads(Path(args.episodes).read_text())
meta = []
for e in eps:
    off = RECAL if e["set"] == "cal2" else np.zeros(6)
    for k in args.keys.split(","):
        if k not in e["state"]:
            continue
        st = np.asarray(e["state"][k]) - off
        set_pose(st)
        tag = f"{e['set']}_{e['episode']:03d}_{k}"
        Image.fromarray(grab(ann["top"])).save(out / f"{tag}_top_sim.png")
        if k == "start":
            Image.fromarray(grab(ann["wrist"])).save(out / f"{tag}_wrist_sim.png")
        np.save(out / f"{tag}_fingers.npy", finger_points().astype(np.float32))
        meta.append({"tag": tag, "state_old": st.round(2).tolist()})
    print("RENDERED", e["set"], e["episode"], flush=True)
(out / "meta.json").write_text(json.dumps(meta))
print("RENDER_DONE", len(meta), flush=True)
app.close()
