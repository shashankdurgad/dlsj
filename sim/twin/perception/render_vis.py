"""Rendered visibility of the target brick in the twin's wrist camera (instance segmentation), for calibrating the
camera-free occlusion proxy (geometry.py) and for side-by-side figures.

    python sim/twin/perception/render_vis.py --prep [--episodes 60]       # plain python (PyAV ok): jobs json
    NAME=dlsj-twin-percep sim/twin/run.sh sim/twin/perception/render_vis.py [--jobs ...] [--shard i,n]   # Isaac
    -> sim/twin/out/perception/render/vis_<shard>.npz  (+ seg_<k>.png / rgb_<k>.jpg for the first --keep jobs)

Jobs: recorded cal2 pick-phase states (joints at frames 45 .. -10 relative to the gripper close) with the target
brick at its wrist-fitted pose (gt_real.py) and the real layout's other bricks (out/scene/layouts) where cached; plus
perturbed copies (brick shifted / turned, gripper opening changed) to cover more occlusion configurations.

Per job, in TwinEnv (wrist camera only, colour correction off, the twin's centred-pinhole camera):
  pass A: full scene -> the target's visible pixels (instance id of /World/envs/env_0/TwinVis/<target>);
  pass B: robot visual + every other brick hidden -> the target's unoccluded in-frame pixels;
  the state actually rendered (joints after the 2-tick settle, brick poses) is saved so the proxy can be evaluated
  on exactly the same scene. Occluding pixels are attributed to fins / bases / mount / robot / other bricks / plates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "sim/residual_env")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
OUT = REPO / "sim/twin/out/perception"
LAYOUTS = REPO / "sim/twin/out/scene/layouts/duplo_teleop_cal2_rgb_456"
KIND_TWIN = {"2x2": "2x2", "2x3": "curved_2x3", "2x4": "2x4"}


def prep(n_eps: int, seed: int = 0):
    from sim.twin.align import frames as F

    fits = json.loads((OUT / "real_fit.json").read_text())["episodes"]
    rng = np.random.default_rng(seed)
    eps = sorted(int(e) for e, f in fits.items() if f.get("src") == "wrist")
    cached = {int(p.stem[2:]) for p in LAYOUTS.glob("ep*.json")} if LAYOUTS.exists() else set()
    eps = sorted(set(rng.choice(eps, size=min(n_eps, len(eps)), replace=False).tolist()))
    jobs = []
    for ep in eps:
        f = fits[str(ep)]
        st = F.states(ep)
        lay = None
        lp = LAYOUTS / f"ep{ep:03d}.json"
        if ep in cached and lp.exists():
            lay = json.loads(lp.read_text())
        for tt in (45, 25, 12, 6, 3, 0, -4, -10):
            fr = f["cf"] - tt
            if not 0 <= fr < len(st):
                continue
            base = dict(
                ep=ep,
                frame=int(fr),
                tt=tt,
                joints=st[fr].round(3).tolist(),
                kind=KIND_TWIN[f["kind"]],
                brick=[f["x"], f["y"], f["yaw"]],
                layout=lay,
                perturbed=False,
            )
            jobs.append(base)
            if tt <= 12:
                for _ in range(2):
                    j = dict(base, perturbed=True)
                    j["brick"] = [
                        f["x"] + rng.uniform(-0.02, 0.02),
                        f["y"] + rng.uniform(-0.02, 0.02),
                        f["yaw"] + rng.uniform(-0.6, 0.6),
                    ]
                    jt = np.array(base["joints"])
                    jt[5] = float(np.clip(jt[5] + rng.uniform(-8, 12), 0, 45))
                    jt[3] += rng.uniform(-4, 4)
                    jt[4] += rng.uniform(-8, 8)
                    j["joints"] = jt.round(3).tolist()
                    jobs.append(j)
    (OUT / "render").mkdir(parents=True, exist_ok=True)
    (OUT / "render/jobs.json").write_text(json.dumps(jobs))
    print("jobs", len(jobs), "episodes", len(eps), "with layout", sum(j["layout"] is not None for j in jobs) // 1)


def _patch_camera_cfg():
    """TwinEnv sets data_types = ["rgb"] on every camera cfg; add instance segmentation to the wrist camera."""
    from isaaclab.sensors import CameraCfg

    orig = CameraCfg.__setattr__

    def setattr_(self, name, value):
        if name == "data_types" and "WristCam" in str(getattr(self, "prim_path", "")):
            value = list(value) + ["instance_id_segmentation_fast"]
            orig(self, "colorize_instance_id_segmentation", False)
        orig(self, name, value)

    CameraCfg.__setattr__ = setattr_


def _layout(job):
    lay = json.loads(json.dumps(job["layout"])) if job["layout"] else {"plates": {}, "distractors": []}
    x, y, yaw = job["brick"]
    lay["target"] = {
        "kind": job["kind"],
        "color": (lay.get("target") or {}).get("color", "cream"),
        "x": x,
        "y": y,
        "yaw": yaw,
    }
    lay["start_state"] = job["joints"]
    lay["episode"] = job["ep"]
    # keep distractors that do not overlap the (possibly perturbed) target
    lay["distractors"] = [d for d in lay.get("distractors", []) if np.hypot(d["x"] - x, d["y"] - y) > 0.04]
    return lay


def render(args):
    from sim.twin.env.twin_env import launch_app

    launch_app(True, True)
    _patch_camera_cfg()
    import torch
    from pxr import UsdGeom

    from sim.twin.env import TwinEnv, TwinEnvCfg

    jobs = json.loads(Path(args.jobs).read_text())
    i, n = (int(v) for v in args.shard.split(","))
    idx = list(range(len(jobs)))[i::n][: args.limit or None]
    env = TwinEnv(
        TwinEnvCfg(cameras=("wrist",), meas_noise=False, color_correct=False, settle_ticks=2, randomize=False)
    )
    stage = env.stage
    cam = env.scene.sensors["wrist_cam"]
    r = env.residual
    out = {
        k: []
        for k in (
            "job",
            "ep",
            "frame",
            "tt",
            "perturbed",
            "joints",
            "bpos",
            "bquat",
            "bsize",
            "kind",
            "n_vis",
            "n_amodal",
            "box_vis_r",
            "box_amodal_r",
            "occ_by",
            "others",
        )
    }
    OCC = ("fin", "base", "mount", "robot", "brick", "plate", "other")
    imgdir = OUT / "render/img"
    imgdir.mkdir(parents=True, exist_ok=True)

    def seg():
        env.sync()
        env.sim.render()
        env.scene.update(0.0)
        s = cam.data.output["instance_id_segmentation_fast"][0, ..., 0].detach().cpu().numpy()
        info = (
            cam.data.info[0]["instance_id_segmentation_fast"]
            if isinstance(cam.data.info, list)
            else cam.data.info["instance_id_segmentation_fast"]
        )
        lab = {int(k): str(v) for k, v in info["idToLabels"].items()}
        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        return s, lab, rgb

    def box(m):
        ys, xs = np.nonzero(m)
        return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)] if len(xs) else [np.nan] * 4

    robot_vis = stage.GetPrimAtPath("/World/envs/env_0/RobotVisual")
    for k in idx:
        job = jobs[k]
        lay = _layout(job)
        env.reset(seed=k, layout=lay, joint_deg=job["joints"], brick_pose=tuple(job["brick"]), render=False)
        r.r[:] = torch.as_tensor(job["joints"], device=env.device, dtype=r.r.dtype)
        tgt_name = r.brick.cfg.prim_path.rsplit("/", 1)[1]
        tgt_vis = f"/World/envs/env_0/TwinVis/{tgt_name}"
        s, lab, rgb = seg()
        ids_t = [i_ for i_, p in lab.items() if p.startswith(tgt_vis)]
        mA = np.isin(s, ids_t)
        # pass B: hide the robot visual and every other TwinVis prim
        hidden = []
        for p in [robot_vis] + list(stage.GetPrimAtPath("/World/envs/env_0/TwinVis").GetChildren()):
            if p and str(p.GetPath()) != tgt_vis:
                im = UsdGeom.Imageable(p)
                if im.ComputeVisibility() != UsdGeom.Tokens.invisible:
                    im.MakeInvisible()
                    hidden.append(im)
        s2, lab2, _ = seg()
        ids_t2 = [i_ for i_, p in lab2.items() if p.startswith(tgt_vis)]
        mB = np.isin(s2, ids_t2)
        for im in hidden:
            im.MakeVisible()
        occ = np.zeros(len(OCC))
        occl = mB & ~mA
        for i_ in np.unique(s[occl]):
            p = lab.get(int(i_), "").lower()
            c = (
                "fin"
                if "finray_fin" in p
                else "base"
                if "finray_base" in p
                else "mount"
                if ("mount" in p or "wristcam" in p or "camera" in p)
                else "robot"
                if "robotvisual" in p
                else "brick"
                if "/twinvis/" in p and ("brick" in p or "distractor" in p)
                else "plate"
                if "/twinvis/" in p
                else "other"
            )
            occ[OCC.index(c)] += float((s[occl] == i_).sum())
        meas = r.measured(noisy=False)[0].detach().cpu().numpy()
        o = env.scene.env_origins[0]
        bp = (r.brick.data.root_pos_w[0] - o).detach().cpu().numpy()
        bq = r.brick.data.root_quat_w[0].detach().cpu().numpy()
        others = []
        for obj in r.distractors:
            p_ = (obj.data.root_pos_w[0] - o).detach().cpu().numpy()
            if abs(p_[0]) < 1.0 and abs(p_[1]) < 1.0:
                others.append([*p_.tolist(), *obj.data.root_quat_w[0].detach().cpu().numpy().tolist()])
        size = list(env.layout.target.size)
        vals = dict(
            job=k,
            ep=job["ep"],
            frame=job["frame"],
            tt=job["tt"],
            perturbed=job["perturbed"],
            joints=meas,
            bpos=bp,
            bquat=bq,
            bsize=size,
            kind=job["kind"],
            n_vis=int(mA.sum()),
            n_amodal=int(mB.sum()),
            box_vis_r=box(mA),
            box_amodal_r=box(mB),
            occ_by=occ,
            others=json.dumps(others),
        )
        for key in out:
            out[key].append(vals[key])
        if len(out["job"]) <= args.keep:
            import cv2

            cv2.imwrite(str(imgdir / f"rgb_{k:04d}.jpg"), cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
            m = np.zeros(s.shape, np.uint8)
            m[mB] = 100
            m[mA] = 255
            cv2.imwrite(str(imgdir / f"mask_{k:04d}.png"), m)
        if len(out["job"]) % 40 == 0:  # partial results survive a stop
            np.savez_compressed(
                OUT / f"render/vis_{i}.npz", occ_names=np.array(OCC), **{k_: np.array(v) for k_, v in out.items()}
            )
        print(
            "job",
            k,
            job["ep"],
            job["tt"],
            "vis",
            round(mA.sum() / max(mB.sum(), 1), 3),
            "occ",
            dict(zip(OCC, occ.astype(int).tolist(), strict=True)),
            flush=True,
        )
    np.savez_compressed(
        OUT / f"render/vis_{i}.npz", occ_names=np.array(OCC), **{k: np.array(v) for k, v in out.items()}
    )
    print("RENDER_DONE", len(out["job"]), flush=True)
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--jobs", default=str(OUT / "render/jobs.json"))
    ap.add_argument("--shard", default="0,1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep", type=int, default=60)
    a = ap.parse_args()
    if a.prep:
        prep(a.episodes)
    else:
        render(a)


if __name__ == "__main__":
    main()
