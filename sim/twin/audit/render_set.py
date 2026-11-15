"""Render the audit set in the CURRENT twin: TwinEnv at the recorded joint state of every sample, real layout.

    NAME=dlsj-twin-audit sim/twin/run.sh sim/twin/audit/render_set.py --run sim/twin/out/audit/runs/<stamp>

The env is the default TwinEnv (default look, props, physics hooks, colour correction ON, randomisation OFF): the
images are exactly what a policy would be fed as obs["images"]. Per sample: reset(layout=<episode layout>,
joint_deg=<recorded state>), the servo reference corrected until the measured state matches the recorded one (the
plain reset's sag is reported as hold_err); the target brick's visual proxy is posed explicitly, on the table where the
layout has it before the grasp, and rigidly attached to the gripper afterwards (grasp / lift / inspect / place): its
pose in the gripper frame is taken at the frame the gripper closes, with the brick resting on the table under the
fingertip point and the detected yaw. The distance between that fingertip point and the detected target brick is
recorded per episode (`grasp_miss_m`: kinematic / camera calibration + detection error).
Region masks from the sim's own geometry (hide a group, difference the render): bits table 1, plates 2, bricks 4,
robot 8, props 16. Writes <run>/sim.npz ({cam}: (N, H, W, 3) uint8, mask_{cam}: (N, H, W) uint8) + render_meta.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.audit.common import BITS, CAMS, HELD, jdump, load_set

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--limit", type=int, default=0, help="only the first N samples (smoke test)")
ap.add_argument("--no-masks", action="store_true")
ap.add_argument("--start", type=int, default=0, help="batch mode: samples [start, stop) -> <run>/parts/")
ap.add_argument("--stop", type=int, default=0)
ap.add_argument("--physics-set", default="", help="params.json overrides (TWIN_PHYSICS_SET), e.g. rigid fins: "
                "gripper.finray.bend.hinges=0")
args = ap.parse_args()
if args.physics_set:
    import os

    os.environ["TWIN_PHYSICS_SET"] = args.physics_set
RUN = Path(args.run)
meta = load_set()
samples = meta["samples"][: args.limit] if args.limit else meta["samples"]
PART = bool(args.stop)
if PART:  # one batch of a batched render (run_audit.sh): a PhysX hang only costs this batch, which is retried
    samples = meta["samples"][args.start: args.stop]
LAY = json.loads((RUN / "layouts.json").read_text())["layouts"]

from sim.twin.env.twin_env import TwinEnv, TwinEnvCfg

t_start = time.time()
env = TwinEnv(TwinEnvCfg(cameras=CAMS, meas_noise=False, max_relative_target=1000.0, render_passes=2))
import cv2
from pxr import Usd, UsdGeom

from sim.twin.visuals.twin_scene import wrist_cam_rel
from sim.twin.visuals.usd_sync import _mat, _parent_world, set_local

try:
    from sim.twin.scene.bricks import PLATE_FLOOR
except Exception:  # noqa: BLE001
    PLATE_FLOOR = 0.004
TABLE_RECT = (-0.06, 0.90, -0.65, 0.65)  # real table wood (env frame), as the visual-fit harness
R = env.residual
H, W = env.cfg.height, env.cfg.width
ENV0 = "/World/envs/env_0"
t_env = time.time() - t_start


# ------------------------------------------------------------------ quaternions (xyzw, numpy)
def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qrot(q, v):
    return qmul(qmul(q, np.array([*v, 0.0])), qconj(q))[:3]


def yaw_q(yaw):
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def grip():
    p = R.robot.data.body_pos_w[0, R.grip_body].detach().cpu().numpy().astype(np.float64)
    q = R.robot.data.body_quat_w[0, R.grip_body].detach().cpu().numpy().astype(np.float64)
    tip = R.tip_w()[0].detach().cpu().numpy().astype(np.float64)
    return p, q, tip


# ------------------------------------------------------------------ explicit pose of the target brick's proxy
target = {"pose": None, "prim": "Brick"}
_sync = env.sync


def sync():
    n = _sync()
    if target["pose"] is not None:
        prim = env.stage.GetPrimAtPath(f"{ENV0}/TwinVis/{target['prim']}")
        if prim:
            set_local(prim, _mat(*target["pose"]) * _parent_world(prim).GetInverse())
    return n


env.sync = sync
origin = env.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)


def target_prim() -> str:
    cur = getattr(env.bricks, "current", None) if env.bricks is not None else None
    return cur["target"][0] if cur and cur.get("target") else "Brick"


def brick_z(on_plate: bool) -> float:
    return float(R.table_z) + float(env.layout.target.size[2]) / 2 + 0.0005 + (PLATE_FLOOR if on_plate else 0.0)


def reset_at(ep: int, lay, joint, **kw) -> dict:
    """reset() so that the twin's MEASURED joint state matches `joint` (the recorded state): the reset settles the
    arm under the servo model for a few ticks, and gravity / the servo fit leave it off by up to several degrees
    (run 20261115-000228: elbow +2.1 deg mean, lift -7.5 .. +5.6 deg). The servo reference is corrected by the
    remaining error (fixed point, <= 4 resets) so the cameras see the recorded pose. Returns the hold error of the
    plain reset (a twin discrepancy in its own right) and of the corrected one."""
    want = np.asarray(joint, np.float64)
    ref = want.copy()
    first = best = None
    for _ in range(4):
        env.reset(seed=ep, layout=lay, joint_deg=ref.tolist(), render=False, **kw)
        err = env.state(noisy=False).astype(np.float64) - want
        if first is None:
            first = err.copy()
        if best is None or np.abs(err[:5]).max() < np.abs(best[1][:5]).max():
            best = (ref.copy(), err.copy())
        if np.abs(err[:5]).max() < 0.25:
            break
        ref[:5] -= err[:5]
    if best[1] is not err:  # a contact / limit made the correction worse: go back to the best reference
        env.reset(seed=ep, layout=lay, joint_deg=best[0].tolist(), render=False, **kw)
        err = env.state(noisy=False).astype(np.float64) - want
    return dict(hold_err=first.round(3).tolist(), state_err=err.round(3).tolist())


def episode_grasp(ep: int, s: dict):
    """Target brick pose in the gripper frame at the gripper-close frame + the fingertip-to-brick miss."""
    lay = LAY.get(str(ep))
    reset_at(ep, lay, s["grasp_state"], brick_pose=(0.8, 0.5, 0.0))
    p, q, tip = grip()
    t = (lay or {}).get("target") or {}
    z = brick_z(bool(t.get("on_plate")))
    pb = np.array([tip[0], tip[1], origin[2] + z])
    miss = None
    if t.get("x") is not None:
        miss = float(np.hypot(t["x"] - (tip[0] - origin[0]), t["y"] - (tip[1] - origin[1])))
    rel_p = qrot(qconj(q), pb - p)
    rel_q = qmul(qconj(q), yaw_q(float(t.get("yaw", 0.0))))
    return dict(rel_p=rel_p, rel_q=rel_q, miss=miss, tip=(tip - origin).tolist(),
                tip_z_above_table=float(tip[2] - origin[2] - float(R.table_z)))


def pose(s: dict, g: dict) -> dict:
    ep = s["ep"]
    lay = LAY.get(str(ep))
    held = s["phase"] in HELD
    t = (lay or {}).get("target") or {}
    if held or t.get("x") is None:
        hold = reset_at(ep, lay, s["state"], brick_pose=(0.8, 0.5, 0.0))
    else:
        hold = reset_at(ep, lay, s["state"])
    target["prim"] = target_prim()
    if held:
        p, q, _ = grip()
        target["pose"] = (p + qrot(q, g["rel_p"]), qmul(q, g["rel_q"]))
    elif t.get("x") is not None:
        target["pose"] = (origin + np.array([t["x"], t["y"], brick_z(bool(t.get("on_plate")))]),
                          yaw_q(float(t.get("yaw", 0.0))))
    else:
        target["pose"] = None
    return dict(**hold, held=held, target_prim=target["prim"],
                scene_plan=getattr(env, "scene_plan", None))


# ------------------------------------------------------------------ rendering with a black-frame guard
BLACK = {"retries": 0}


def is_black(img: dict) -> bool:
    return min(float(v.mean()) for v in img.values()) <= 2.0


def render_ok(tries: int = 8) -> dict:
    """env.render(), retried while any camera comes back black (run 20261115-000228: all 63 samples x 3 cameras were
    all-zero on a GPU / memory-starved host, and nothing noticed). Raises if it stays black."""
    for k in range(tries):
        img = {c: v.copy() for c, v in env.render().items()}
        if not is_black(img):
            return img
        BLACK["retries"] += 1
        time.sleep(min(2.0 * (k + 1), 10.0))
    raise RuntimeError("black render: " + ", ".join(f"{c} mean {float(v.mean()):.2f}" for c, v in img.items()))


# ------------------------------------------------------------------ masks
def set_vis(paths, on: bool):
    for p in paths:
        prim = env.stage.GetPrimAtPath(p)
        if prim:
            im = UsdGeom.Imageable(prim)
            im.MakeVisible() if on else im.MakeInvisible()


def groups() -> dict:
    tv = env.stage.GetPrimAtPath(f"{ENV0}/TwinVis")
    kids = [str(c.GetPath()) for c in tv.GetChildren()] if tv else []
    g = {BITS["robot"]: [f"{ENV0}/RobotVisual"], BITS["bricks"]: [k for k in kids if "/Bowl_" not in k],
         BITS["plates"]: [k for k in kids if "/Bowl_" in k]}
    if env.stage.GetPrimAtPath(f"{ENV0}/TwinProps"):
        g[BITS["props"]] = [f"{ENV0}/TwinProps"]
    return g


def table_rays(cam: str):
    rel = {"top": "TopCam", "inspect": "InspectCam", "wrist": wrist_cam_rel()}[cam]
    prim = env.stage.GetPrimAtPath(f"{ENV0}/{rel}")
    M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    c = UsdGeom.Camera(prim)
    fx = c.GetFocalLengthAttr().Get() / c.GetHorizontalApertureAttr().Get() * W
    v, u = np.mgrid[:H, :W]
    d = np.stack([(u + 0.5 - W / 2) / fx, -(v + 0.5 - H / 2) / fx, -np.ones_like(u, float)], -1)
    dw = d @ M[:3, :3]
    pos = M[3, :3] - origin
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (float(R.table_z) - pos[2]) / dw[..., 2]
    P = pos + t[..., None] * dw
    x0, x1, y0, y1 = TABLE_RECT
    return (t > 0) & (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)


def masks(base: dict) -> dict:
    out = {c: np.zeros((H, W), np.uint8) for c in base}
    for bit, paths in groups().items():
        if not paths:
            continue
        set_vis(paths, False)
        img = render_ok()
        set_vis(paths, True)
        for c in base:
            d = np.abs(img[c].astype(np.int16) - base[c].astype(np.int16)).max(-1) > 22
            d = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            out[c][d > 0] |= bit
    for c in base:
        occ = cv2.dilate((out[c] > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        try:
            out[c][table_rays(c) & ~occ] |= BITS["table"]
        except Exception:  # noqa: BLE001
            pass
    return out


# ------------------------------------------------------------------ main loop
imgs = {c: [] for c in CAMS}
mks = {c: [] for c in CAMS}
rows, grasp_info, errors = [], {}, []
t0 = time.time()
for n, s in enumerate(samples):
    try:
        ep = s["ep"]
        if ep not in grasp_info:
            grasp_info[ep] = episode_grasp(ep, s)
        info = pose(s, grasp_info[ep])
        env.render()  # first frame after the teleport (renderer history)
        img = render_ok()
        m = None
        if not args.no_masks:
            try:
                m = masks(img)
            except Exception:  # noqa: BLE001
                errors.append(dict(i=s["i"], where="masks", error=traceback.format_exc(limit=3)))
    except Exception:  # noqa: BLE001
        errors.append(dict(i=s["i"], where="pose/render", error=traceback.format_exc(limit=5)))
        print("[audit] sample", s["i"], "FAILED\n", errors[-1]["error"], flush=True)
        info, img, m = dict(failed=True), {c: np.zeros((H, W, 3), np.uint8) for c in CAMS}, None
    for c in CAMS:
        imgs[c].append(img[c])
        mks[c].append(m[c] if m is not None else np.zeros((H, W), np.uint8))
    g = grasp_info.get(s["ep"], {})
    rows.append(dict(i=s["i"], ep=s["ep"], phase=s["phase"], grasp_miss_m=g.get("miss"), **info))
    if n % 10 == 0:
        print(f"[audit] rendered {n + 1}/{len(samples)} ({time.time() - t0:.0f}s)", flush=True)

OUT_DIR = RUN / "parts" if PART else RUN
OUT_DIR.mkdir(exist_ok=True)
SUF = f"_{args.start:03d}" if PART else ""
np.savez_compressed(OUT_DIR / f"sim{SUF}.npz", **{c: np.stack(v) for c, v in imgs.items()},
                    **{"mask_" + c: np.stack(v) for c, v in mks.items()})
look = env.look
jdump(OUT_DIR / f"render_meta{SUF}.json", dict(physics_set=args.physics_set,
    n=len(samples), black_retries=BLACK["retries"], env_start_s=round(t_env, 1), render_s=round(time.time() - t0, 1),
    errors=errors, samples=rows,
    grasp={str(k): dict(miss_m=v["miss"], tip=v["tip"], tip_z_above_table=v["tip_z_above_table"])
           for k, v in grasp_info.items()},
    profile=getattr(env.profile, "name", None), visuals=env.visuals, props=env.props_info,
    physics_hooks=env._physics.__name__ if env._physics else None, physics_info=env.physics_info,
    look={f.name: getattr(look, f.name) for f in dataclasses.fields(look)} if dataclasses.is_dataclass(look)
    else repr(look),
    cfg={f.name: getattr(env.cfg, f.name) for f in dataclasses.fields(env.cfg)
         if f.name not in ("look", "randomizers")}))
n_black = sum(1 for c in CAMS for v in imgs[c] if float(v.mean()) <= 2.0)
if n_black or errors:
    print(f"RENDER_FAILED {len(errors)} errors, {n_black} black camera images (black retries {BLACK['retries']})",
          flush=True)
else:
    print("RENDER_OK", len(samples), "samples", 0, "errors", f"(black retries {BLACK['retries']})", flush=True)
env.close()
