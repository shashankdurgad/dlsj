"""Contact sheets of base vs residual-corrected grasps.

Stage 1 (batched GraspEnv, no cameras): run the same held-out episodes with the base (with errors) and with the
trained residual, record per-tick joints / brick pose.  Stage 2 (TwinEnv, cameras): play the recorded states back
kinematically (write joints + brick pose, one physics substep, VisualSync, render) and tile wrist / top frames.

    sim/twin/run.sh sim/twin/rl/render_grasps.py --stage record --ckpt sim/twin/out/rl/dice_b2/residual.pt --out dice_b2
    sim/twin/run.sh sim/twin/rl/render_grasps.py --stage render --out dice_b2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sim.twin.rl import grasp_env as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["record", "render"], required=True)
ap.add_argument("--ckpt", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--n_show", type=int, default=4)
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--ticks", default="0,15,30,45,60,75,89")
args = ap.parse_args()
out = G.OUT / args.out
out.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402

if args.stage == "record":
    app = G.launch()
    import torch  # noqa: E402

    from sim.twin.rl import rollout as RO  # noqa: E402
    from sim.twin.rl.dice import Dice  # noqa: E402

    env = G.make(G.GraspCfg(num_envs=256, brick_mix=(1.0, 0.0, 0.0)))
    agent, ck = Dice.load(args.ckpt, env.device)
    res_all = {}
    recs = {}
    for label, pol, K in (("base", "base", 1), ("residual", "dice", args.K)):
        rec = RO.Recorder(torch.arange(env.n, device=env.device))
        res = RO.evaluate(env, pol, n_eps=env.n, seed=777, agent=agent, K=K, k_state=ck.get("k_state", 4),
                          recorder=rec)
        res_all[label] = res
        recs[label] = rec.arrays()
    oc_b, oc_r = res_all["base"]["outcome"].cpu().numpy(), res_all["residual"]["outcome"].cpu().numpy()
    fixed = np.nonzero((oc_b != 0) & (oc_r == 0))[0]
    both = np.nonzero((oc_b == 0) & (oc_r == 0))[0]
    lost = np.nonzero((oc_b == 0) & (oc_r != 0))[0]
    show = list(fixed[: max(1, args.n_show - 1)]) + list(both[:1]) + list(lost[:1])
    show = show[: args.n_show + 1]
    np.savez(out / "grasps.npz", show=np.array(show), oc_base=oc_b[show], oc_res=oc_r[show],
             **{f"{lab}_{k}": v[show] for lab, a in recs.items() for k, v in a.items()})
    summ = {"episodes": int(env.n), "base_success": float((oc_b == 0).mean()),
            "residual_success": float((oc_r == 0).mean()),
            "fixed_by_residual": int(len(fixed)), "lost_by_residual": int(len(lost)), "shown": [int(v) for v in show]}
    (out / "grasps.json").write_text(json.dumps(summ, indent=1))
    print("RECORD", json.dumps(summ), flush=True)
    app.close()
else:
    from sim.twin.env.twin_env import TwinEnv, TwinEnvCfg

    env = TwinEnv(TwinEnvCfg(cameras=("top", "wrist"), randomize=False, meas_noise=False, layout="nominal"))
    import torch  # noqa: E402

    R = env.residual
    z = np.load(out / "grasps.npz")
    ticks = [int(v) for v in args.ticks.split(",")]
    names = [G.FAIL[int(v)] for v in z["oc_base"]], [G.FAIL[int(v)] for v in z["oc_res"]]
    o = env.scene.env_origins[:1]
    for k2, obj in enumerate(R.distractors + R.bowl_objs):  # clutter / plates out of view
        pose = torch.tensor([[0.88, -0.55 + 0.12 * k2, 0.02, 0, 0, 0, 1.0]], device=env.device)
        pose[:, :3] += o
        obj.write_root_pose_to_sim(pose)
    rows = {"wrist": [], "top": []}
    import cv2  # noqa: E402

    for e in range(len(z["show"])):
        for lab, nm in (("base", names[0][e]), ("residual", names[1][e])):
            q, bp, bq = z[f"{lab}_q"][e], z[f"{lab}_brick_pos"][e], z[f"{lab}_brick_quat"][e]
            env.reset(seed=0, joint_deg=q[0], brick_pose=(float(bp[0, 0]), float(bp[0, 1]), 0.0), render=False)
            cells = {"wrist": [], "top": []}
            for t in ticks:
                t = min(t, len(q) - 1)
                qq = torch.as_tensor(q[t], device=env.device)[None]
                full = R.robot.data.default_joint_pos[:1].clone()
                full[:, R.jidx] = R.to_sim(qq)
                R.robot.write_joint_state_to_sim(full, torch.zeros_like(full))
                R.robot.set_joint_position_target(full)
                pose = torch.cat([torch.as_tensor(bp[t], device=env.device)[None] + o,
                                  torch.as_tensor(bq[t], device=env.device)[None]], -1)
                R.brick.write_root_pose_to_sim(pose)
                R.brick.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(1.0 / 120)
                img = env.render()
                for cam in ("wrist", "top"):
                    im = cv2.resize(img[cam], (320, 240))
                    if cam == "top":  # crop around the brick: top camera projection is not needed, use a centre crop
                        im = cv2.resize(img[cam][60:420, 80:560], (320, 240))
                    im = np.ascontiguousarray(im)
                    cv2.putText(im, f"t={t / 30:.1f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cells[cam].append(im)
            for cam in cells:
                row = np.concatenate(cells[cam], 1)
                lbl = np.zeros((22, row.shape[1], 3), np.uint8)
                cv2.putText(lbl, f"episode {int(z['show'][e])} | {lab}: {nm}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1)
                rows[cam].append(np.concatenate([lbl, row], 0))
    for cam, rr in rows.items():
        sheet = np.concatenate(rr, 0)
        cv2.imwrite(str(out / f"grasps_{cam}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", out / f"grasps_{cam}.jpg", sheet.shape, flush=True)
    env.close()
