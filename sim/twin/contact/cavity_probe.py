"""Is an upside-down brick's cavity open in the collider? Bricks are settled studs-down on the table, then a 0.5 mm grid
of PhysX scene-query rays is cast straight down over each one: the collision surface height map (what a fin tip
lowered vertically would touch).

    sim/twin/run.sh sim/twin/contact/cavity_probe.py --tag shell --set contact.brick_collider=\"shell\"
-> sim/twin/out/contact/cavity_<tag>.npz: heights (mm above the table) per brick kind on the brick-local x / y grid,
   the brick pose, plus a summary line per kind: open-cavity area (cells below the rim by > 5 mm), floor height.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--step_mm", type=float, default=0.5)
args = ap.parse_args()
if args.set:
    os.environ["TWIN_PHYSICS_SET"] = ";".join([o for o in (os.environ.get("TWIN_PHYSICS_SET", ""), *args.set) if o])

from sim.twin.rl import grasp_env as G  # noqa: E402

app = G.launch()
import carb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omni.physx import get_physx_scene_query_interface  # noqa: E402

from sim.twin.contact import bricks as B  # noqa: E402

KINDS = ("2x2", "2x4", "curved_2x3")
cfg = G.GraspCfg(num_envs=3, device="cpu", meas_noise=False, errors=False, horizon_s=60.0,
                 **{k: v for k, v in (("play", False), ("distractors", 0), ("plates", False))
                    if k in G.GraspCfg.__dataclass_fields__})
env = G.make(cfg)
dev = env.device
phys = sys.modules["sim.twin.physics.apply"]
variant = (phys._p().get("contact") or {}).get("brick_collider") or "box"
o = env.scene.env_origins
ids = torch.arange(3, device=dev)
xy = (0.30, 0.20)
for e in range(3):
    for kk, obj in enumerate(env.objs):
        if kk == e:
            h = B.rest_heights(KINDS[e], variant)["down"]
            pose = [xy[0] + float(o[e, 0]), xy[1] + float(o[e, 1]), float(env.table_z) + h + 0.002 + float(o[e, 2]),
                    1, 0, 0, 0]
        else:
            pose = [0.85 + float(o[e, 0]), -0.45 + 0.3 * kk + float(o[e, 1]), 0.02 + float(o[e, 2]), 0, 0, 0, 1]
        obj.write_root_pose_to_sim(torch.tensor([pose], device=dev), env_ids=ids[e:e + 1])
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev), env_ids=ids[e:e + 1])
for _ in range(90):
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(1.0 / 120)
sq = get_physx_scene_query_interface()
out, summ = {}, {}
for e, kind in enumerate(KINDS):
    obj = env.objs[e]
    p = obj.data.root_pos_w[e].cpu().numpy().astype(float)
    hx, hy, hz = (v / 2 for v in B.BODY[kind])
    s = args.step_mm / 1e3
    gx = np.arange(-hx - 0.003, hx + 0.003 + 1e-9, s)
    gy = np.arange(-hy - 0.003, hy + 0.003 + 1e-9, s)
    H = np.full((len(gy), len(gx)), np.nan)
    top = float(env.table_z) + float(o[e, 2]) + 0.06
    names = set()
    for j, yy in enumerate(gy):
        for i, xx in enumerate(gx):
            # brick upside down = rotation pi about x: local (x, y) -> world (x, -y)
            r = sq.raycast_closest(carb.Float3(p[0] + xx, p[1] - yy, top), carb.Float3(0, 0, -1), 0.2)
            if r["hit"]:
                H[j, i] = (r["position"][2] - float(env.table_z) - float(o[e, 2])) * 1e3
                names.add(str(r["collision"]).split("/")[-1])
    out[kind] = H
    rim = float(np.nanmax(H))
    inside = (np.abs(gx)[None, :] < hx - 0.002) & (np.abs(gy)[:, None] < hy - 0.002)
    deep = (H < rim - 5.0) & inside
    summ[kind] = {"centre_mm": np.round((p - o[e].cpu().numpy()) * 1e3, 2).tolist(),
                  "rim_mm": round(rim, 2), "floor_mm": round(float(np.nanmin(H[inside])), 2),
                  "open_area_frac_of_interior": round(float(deep.sum() / inside.sum()), 3),
                  "max_depth_below_rim_mm": round(rim - float(np.nanmin(H[inside])), 2), "hit_shapes": sorted(names)}
    print(kind, json.dumps(summ[kind]), flush=True)
dst = REPO / "sim/twin/out/contact"
dst.mkdir(parents=True, exist_ok=True)
np.savez_compressed(dst / f"cavity_{args.tag}.npz", **{k: v for k, v in out.items()},
                    summary=json.dumps(summ), variant=variant, step_mm=args.step_mm)
print("CAVITY", variant, json.dumps(summ), flush=True)
app.close()
