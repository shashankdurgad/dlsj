"""Does VisualSync pose every RobotVisual link from the right physics body? (Isaac)

    NAME=dlsj-twin-wristfit sim/twin/run.sh sim/twin/wrist_fit/sync_diag.py
Prints the physics body names / link paths, the VisualSync mapping, and per Finray prim the world position of the
visual vs the physics prim at the same relative path (an audit grasp-phase state).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.audit.common import load_set  # noqa: E402
from sim.twin.env.twin_env import TwinEnv, TwinEnvCfg  # noqa: E402

env = TwinEnv(TwinEnvCfg(cameras=("wrist",), meas_noise=False, max_relative_target=1000.0, render_passes=1))
from pxr import Usd, UsdGeom  # noqa: E402

s = load_set()["samples"][3]
env.reset(seed=s["ep"], joint_deg=list(s["state"]), render=False)
env.render()
rob = env.sync.robot
print("[sync] body_names", rob.body_names, flush=True)
try:
    print("[sync] link_paths", rob.root_physx_view.link_paths[0], flush=True)
except Exception as ex:  # noqa: BLE001
    print("[sync] link_paths n/a", ex, flush=True)
print("[sync] mapping", [(i, str(p.GetPath())) for i, p in env.sync.links.get(0, [])], flush=True)
bp = rob.data.body_pos_w[0].detach().cpu().numpy()
ENV0 = "/World/envs/env_0"
for p in Usd.PrimRange(env.stage.GetPrimAtPath(ENV0)):
    n = p.GetName()
    if n.startswith("TwinFinSeg") or n in ("gripper_link", "moving_jaw_so101_v1_link"):
        w = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))[3, :3]
        idx = [i for i, b in enumerate(rob.body_names) if b == n]
        print("[sync] prim", str(p.GetPath()).replace(ENV0, ""), "usd", w.round(4).tolist(), "phys",
              [bp[i].round(4).tolist() for i in idx], flush=True)
print("SYNC_OK", flush=True)
env.close()
