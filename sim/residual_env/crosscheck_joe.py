"""Cross-check against Joe's cal2 replays (pi05-isaac-so101/cal2-calibration/replay.py) in OUR env.

Joe's protocol: the recorded action commands of one cal2 episode, dynamic physics, no attachments; the target is a
2x4 brick (64x32x19.2 mm) placed from the episode's initial top image at his reference position/yaw. Metrics as his:
max lift, lift held over the last 1 s, closest fingertip (our TIP_LOCAL) to the initial target, mean arm tracking
(|measured - command|, deg), final gripper. Runs each requested profile on the same episodes.

    python crosscheck_joe.py --profiles joe_cal2,fit_cal2 --out ~/residual_env/out/crosscheck_joe
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--profiles", default="joe_cal2,fit_cal2")
ap.add_argument("--cases", default="ep101:reference-cal2.json,ep120:reference-ep120-heldout.json,"
                "ep120ca:reference-ep120-camera-adapted.json")
ap.add_argument("--profile", default="", help="internal: run one profile in this process")
ap.add_argument("--out", default=str(Path.home() / "residual_env/out/crosscheck_joe"))
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

if not args.profile:  # one Isaac process per profile (a scene is built once per process)
    import subprocess
    res = {}
    for prof in args.profiles.split(","):
        r = subprocess.run([sys.executable, __file__, "--profile", prof, "--cases", args.cases, "--out", args.out],
                           capture_output=True, text=True)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("CROSSCHECK ")]
        res[prof] = json.loads(line[-1][11:]) if line else {"error": (r.stdout + r.stderr)[-1500:]}
    joe = {}
    for name, f in [("ep101", "result-ep101-joe.json"), ("ep120", "result-ep120-heldout-joe.json"),
                    ("ep120ca", "result-ep120-camera-adapted-joe.json")]:
        p = HERE / "assets/joe_cal2" / f
        if p.exists():
            x = json.loads(p.read_text())
            joe[name] = {"max_lift_m": round(x["max_lift_m"], 4), "hold_over5cm": x["hold_over5cm"],
                         "closest_tip_to_target_m": round(x["closest_fk_to_initial_target_m"], 4),
                         "mean_arm_tracking_deg": [round(v, 2) for v in x["mean_arm_tracking_deg"]],
                         "final_gripper_rad": round(x["final_gripper_rad"], 3)}
    res["joe_own_sim"] = joe
    (out / "crosscheck.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0)

sys.path.insert(0, str(HERE))
from app import launch  # noqa: E402

app = launch(cameras=False)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from app import make  # noqa: E402
from env import EnvCfg  # noqa: E402

cases = [c.split(":") for c in args.cases.split(",")]
cfg = EnvCfg(num_envs=len(cases), max_relative_target=1000.0, meas_noise=False, episode_s=60, calib=args.profile,
             target_size=(0.064, 0.032, 0.0192))
sim, scene, layout, env = make(cfg)
dev = env.device
n = len(cases)
recs, brick = [], torch.zeros(env.n, 3, device=dev)
for k, (name, ref_file) in enumerate(cases):
    ref = json.loads((HERE / "assets/joe_cal2" / ref_file).read_text())
    ep = int(ref.get("replay_episode", 101))
    d = np.load(HERE / f"assets/joe_cal2/episode{ep}.npz")
    recs.append((name, ep, d["states"].astype(np.float32), d["actions"].astype(np.float32)))
    tp = ref["objects"]["target"]["position"]
    brick[k] = torch.tensor([tp[0], tp[1], np.deg2rad(ref["target_yaw_deg"])], device=dev)
T = min(180, min(len(r[3]) for r in recs))  # Joe replays the first 180 frames
s0 = torch.tensor(np.stack([r[2][0] for r in recs]), device=dev)
env.reset(joint_deg=s0, brick_pose=brick)
init = (env.brick.data.root_pos_w - scene.env_origins)[:n].clone()
lift, tipd, track = np.zeros((n, T)), np.full(n, np.inf), np.zeros((n, T, 5))
for t in range(T):
    goal = torch.tensor(np.stack([r[3][t] for r in recs]), device=dev)
    env.step(goal_override=goal)
    pos = (env.brick.data.root_pos_w - scene.env_origins)[:n]
    lift[:, t] = (pos[:, 2] - init[:, 2]).cpu().numpy()
    tip = (env.tip_w() - scene.env_origins)[:n]
    tipd = np.minimum(tipd, torch.linalg.norm(tip - init, dim=1).cpu().numpy())
    track[:, t] = np.abs(env.measured(noisy=False)[:n, :5].cpu().numpy() - goal[:, :5].cpu().numpy())
res = {}
for k, (name, ep, _, _) in enumerate(recs):
    res[name] = {"episode": ep, "max_lift_m": round(float(lift[k].max()), 4),
                 "hold_over5cm": bool(lift[k, -30:].min() > 0.05),
                 "closest_tip_to_target_m": round(float(tipd[k]), 4),
                 "mean_arm_tracking_deg": np.round(track[k].mean(0), 2).tolist(),
                 "final_gripper_units": round(float(env.measured(noisy=False)[k, 5]), 2),
                 "final_gripper_rad": round(float(env.robot.data.joint_pos[k, env.jidx[5]]), 3)}
print("CROSSCHECK " + json.dumps(res), flush=True)
app.close()
