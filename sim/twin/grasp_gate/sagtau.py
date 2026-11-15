"""Static gravity torque on lift / elbow / wrist_flex for every frame of every kept cal2 demo (states and actions),
for the gate's arm-sag factor (gate.py --sag a,b,c: angle += sag * tau, the joint fit's model,
align/joint_fit/model.py).

    sim/twin/run.sh sim/twin/grasp_gate/sagtau.py        (plain python; needs sim/twin/align/joint_fit/fk.py)
    -> sim/twin/out/grasp_gate/sag_tau.npz {episodes, tau_state (E, T, 3), tau_action (E, T, 3)} in N m, rows in
       GraspEnv's keep order, padded with the last frame like GraspEnv's A / S.
Torques use the joint fit's current offsets (result.json) and its wrist-camera mass on gripper_link.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
JF = REPO / "sim/twin/align/joint_fit"
sys.path.insert(0, str(JF))
from fk import SO101, sim_q  # noqa: E402

from sim.twin.grasp_gate import common as C  # noqa: E402

CAM_MASS = (0.02, np.array([0.0, 0.06, -0.04]))  # = align/joint_fit/model.py CAM_MASS


def main():
    res = json.loads((JF / "result.json").read_text())
    off = np.array(res["joint_offsets_rad_for_sim"], float)
    demos = C.load_demos()
    arm = SO101()
    keep = demos["keep"]
    T = max(len(demos["eps"][e]["action"]) for e in keep)
    out = {"tau_state": np.zeros((len(keep), T, 3), np.float32), "tau_action": np.zeros((len(keep), T, 3), np.float32)}
    for i, e in enumerate(keep):
        d = demos["eps"][e]
        for key, arr in (("tau_state", d["state"]), ("tau_action", d["action"])):
            L = len(arr)
            q = np.array([sim_q(a, off, 0.0, 0.0) for a in arr])  # gripper angle barely changes the arm's torques
            tau = np.array([arm.gravity_torque(q[f], {"gripper_link": CAM_MASS})[1:4] for f in range(L)])
            out[key][i, :L], out[key][i, L:] = tau, tau[-1]
        if i % 50 == 0:
            print(i, e, np.round(out["tau_state"][i, int(d["cf"])], 3).tolist(), flush=True)
    np.savez_compressed(C.OUT / "sag_tau.npz", episodes=np.array(keep), **out)
    print("wrote", C.OUT / "sag_tau.npz", flush=True)


if __name__ == "__main__":
    main()
