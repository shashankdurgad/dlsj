"""Resting-pose bench for the brick colliders: every kind (2x2, 2x4, curved 2x3) in every pose (up, down = on its
studs, side_x, side_y) is dropped from 3 mm above its nominal resting height onto the table and settled for 1.5 s.

    sim/twin/run.sh sim/twin/contact/rest_test.py --tag box                                   # params.json bricks
    sim/twin/run.sh sim/twin/contact/rest_test.py --tag studs --set contact.brick_collider=\"studs\"
-> sim/twin/out/contact/rest_<tag>.json: settled centre height (mm above the table), tilt from the placed pose (deg),
   xy drift (mm), final speed, plus the effective contact / rest offsets of env_0's shapes and the hook's summary.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--seconds", type=float, default=1.5)
args = ap.parse_args()
if args.set:
    os.environ["TWIN_PHYSICS_SET"] = ";".join([o for o in (os.environ.get("TWIN_PHYSICS_SET", ""), *args.set) if o])

from sim.twin.rl import grasp_env as G  # noqa: E402

app = G.launch()
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.twin.contact import bricks as B  # noqa: E402

S = math.sqrt(0.5)
POSES = {"up": (0, 0, 0, 1), "down": (1, 0, 0, 0), "side_x": (0, S, 0, S), "side_y": (S, 0, 0, S)}
CURVED_SIDE = (0.5, 0.5, 0.5, 0.5)  # on a flat side, long axis horizontal (gate_run.py --c_curved)
KINDS = ("2x2", "2x4", "curved_2x3")
combos = [(k, p) for k in range(3) for p in POSES]
cfg = G.GraspCfg(num_envs=len(combos), meas_noise=False, errors=False, horizon_s=60.0,
                 **{k: v for k, v in (("play", False), ("distractors", 0), ("plates", False))
                    if k in G.GraspCfg.__dataclass_fields__})
env = G.make(cfg)
dev = env.device
assert env.objs[2] is not None, "no curved brick in this GraspEnv"
phys = sys.modules["sim.twin.physics.apply"]
variant = (phys._p().get("contact") or {}).get("brick_collider") or "box"
o = env.scene.env_origins
N = len(combos)
ids = torch.arange(N, device=dev)
xy = torch.tensor([0.30, 0.20], device=dev)
nominal = []


def quat(k, pn):
    if k == 2 and pn.startswith("side"):
        return CURVED_SIDE
    return POSES["side_y"] if k == 1 and pn == "side_x" else POSES[pn]  # 2x4: on its long side (as gate.py)


for e, (k, pn) in enumerate(combos):
    h = B.rest_heights(KINDS[k], variant)[pn]
    nominal.append(h)
    for kk, obj in enumerate(env.objs):
        if kk == k:
            pose = torch.tensor([[*(xy + o[e, :2]).tolist(), float(env.table_z) + h + 0.003 + float(o[e, 2]),
                                  *quat(k, pn)]], device=dev)
        else:
            pose = torch.tensor([[0.85 + float(o[e, 0]), -0.45 + 0.3 * kk + float(o[e, 1]), 0.02 + float(o[e, 2]),
                                  0, 0, 0, 1]], device=dev)
        obj.write_root_pose_to_sim(pose, env_ids=ids[e:e + 1])
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev), env_ids=ids[e:e + 1])
steps = int(args.seconds * 120)
for _ in range(steps):
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(1.0 / 120)
rows = []
for e, (k, pn) in enumerate(combos):
    obj = env.objs[k]
    p = (obj.data.root_pos_w[e] - o[e]).cpu().numpy()
    q = obj.data.root_quat_w[e].cpu().numpy()
    v = float(torch.linalg.norm(obj.data.root_lin_vel_w[e]))
    q0 = np.array(quat(k, pn))
    dot = abs(float(np.dot(q, q0)))
    rows.append({"kind": KINDS[k], "pose": pn, "height_mm": round(float(p[2] - float(env.table_z)) * 1e3, 2),
                 "nominal_mm": round(nominal[e] * 1e3, 2),
                 "tilt_deg": round(math.degrees(2 * math.acos(min(dot, 1.0))), 1),
                 "drift_mm": round(float(np.linalg.norm(p[:2] - xy.cpu().numpy())) * 1e3, 1),
                 "speed_mm_s": round(v * 1e3, 2)})
    print(rows[-1], flush=True)
import omni.usd  # noqa: E402

from sim.twin.contact import hook  # noqa: E402

out = REPO / "sim/twin/out/contact"
out.mkdir(parents=True, exist_ok=True)
res = {"tag": args.tag, "variant": variant, "set": args.set, "rows": rows,
       "offsets_env0": hook.describe_offsets(omni.usd.get_context().get_stage())}
(out / f"rest_{args.tag}.json").write_text(json.dumps(res, indent=1))
print("REST", json.dumps(rows), flush=True)
app.close()
