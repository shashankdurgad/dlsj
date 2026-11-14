"""Render the twin (with visuals/) at the recorded joint state of real frames and write real | sim side-by-sides.

    sim/twin/run.sh sim/twin/visuals/render_compare.py [--episodes 0 76 190 304 418] [--frame 0]
        [--calibrate] [--no-layout] [--plain] [--profile joe_cal2]

Per episode: the joint state of the frame (radians = deg2rad(state[:5]) + profile.joint_offsets[:5], gripper rad =
units * grip_scale + grip_bias, exactly the residual env's to_sim), plates and bricks placed where they are detected in
that real top frame (measure_real.detect_layout; rough), then top / wrist / inspect renders next to the real frames:
sim/twin/out/visuals/compare_ep<NNN>_f<FFF>.jpg (rows: top, wrist, inspect; columns: real | sim | 50 % blend).

--calibrate: scale the light intensities so the table renders like the real (empty-table) median in the top camera,
             then write sim/twin/visuals/lighting.json (exposure_scale) used by TwinLook by default.
--plain:     the residual env's original look (no visuals/); NOTE its renders keep the spawn pose (no proxies).
"""
import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TWIN = REPO / "sim/twin"
OUT = TWIN / "out/visuals"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim/residual_env"))
from app import KIT_ARGS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, nargs="+", default=[0, 76, 190, 304, 418])
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--profile", default="joe_cal2")
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--no-layout", action="store_true")
ap.add_argument("--plain", action="store_true")
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--warmup", type=int, default=60)
ap.add_argument("--tag", default="")
ap.add_argument("--no-ccm", action="store_true", help="skip the per-camera colour transform (wrist / inspect)")
args = ap.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

simapp = AppLauncher(headless=True, enable_cameras=True, kit_args=KIT_ARGS).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

import scene as rscene  # noqa: E402
from calib import load_profile  # noqa: E402
from sim.twin.visuals.measure_real import detect_layout, persistent_mask  # noqa: E402
from sim.twin.visuals.realdata import Dataset  # noqa: E402
from sim.twin.visuals.usd_sync import VisualSync  # noqa: E402
from sim.twin.visuals.table_texture import TABLE_TOP_Z, backproject, camera_from_profile  # noqa: E402
from sim.twin.visuals.twin_scene import LIGHTING_JSON, TwinLook, apply_twin_visuals, set_brick_color, \
    color_correct, twin_scene_cfg, wrist_cam_rel  # noqa: E402

drive = UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=60.0, damping=2.0))
usd = UrdfConverter(UrdfConverterCfg(
    asset_path=str(TWIN / "assets/so101/urdf/so101_new_calib.urdf"), usd_dir=str(TWIN / "assets/so101/usd"),
    usd_file_name="so101_new_calib.usda", fix_base=True, merge_fixed_joints=False, joint_drive=drive)).usd_path
profile = dataclasses.replace(load_profile(args.profile), robot_usd=usd)
layout = rscene.load_layout(str(REPO / "sim/realism/scene_params.json"), "joe")
# more bricks than the env's default so the detected real layout can be reproduced: 5 x 2x2 + 3 x 2x4
B = rscene.Brick
s22, s24 = (0.0318, 0.0318, 0.0192), (0.0636, 0.0318, 0.0192)
W = (0.8, 0.8, 0.8)  # plain-look colour; the twin look binds palette colours instead
layout.distractors = [B("d0", s24, (0.24, -0.07), W, 0.018), B("d1", s24, (0.24, 0.0), W, 0.018),
                      B("d2", s22, (0.2, 0.0), W, 0.0093), B("d3", s22, (0.2, 0.05), W, 0.0093),
                      B("d4", s22, (0.2, -0.05), W, 0.0093), B("d5", s22, (0.15, 0.05), W, 0.0093),
                      B("d6", s24, (0.15, -0.05), W, 0.018)]
SLOTS = {"2x2": ["Brick", "Distractor2", "Distractor3", "Distractor4", "Distractor5"],
         "2x4": ["Distractor0", "Distractor1", "Distractor6"]}

look = TwinLook()
# The GB10 render path (KIT_ARGS: Fabric scene delegate off) renders plain USD and PhysX does not write poses back to
# USD -> the cameras render physics-free visual proxies moved by VisualSync (usd_sync.py) before every render.
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0", use_fabric=False))
if args.plain:
    from app import paint_robot

    cfg = rscene.build_scene_cfg(1, layout, True, profile=profile)
    scn = InteractiveScene(cfg)
    paint_robot(1)
    info = {"plain": True}
else:
    scn = InteractiveScene(twin_scene_cfg(1, layout, profile, look))
    info = apply_twin_visuals(1, layout, profile, look)
print("visuals:", info, flush=True)
sim.reset()
scn.reset()
sync = VisualSync(scn, wrist_cam_rel=wrist_cam_rel()) if not args.plain else (lambda: 0)

robot = scn["robot"]
dev = robot.device
jidx = [robot.joint_names.index(j) for j in rscene.JOINTS]
off = torch.tensor(profile.joint_offsets, device=dev, dtype=torch.float32)


def to_sim(state):
    s = torch.tensor(state, device=dev, dtype=torch.float32)
    q = torch.deg2rad(s) + off
    q[5] = s[5] * profile.grip_scale + profile.grip_bias
    return q


def set_joints(state):
    q = robot.data.default_joint_pos.clone()
    q[0, jidx] = to_sim(state)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)


def place(obj, x, y, z, yaw):
    pose = torch.tensor([[x, y, z, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]], device=dev)
    pose[:, :3] += scn.env_origins[:1]
    obj.write_root_pose_to_sim(pose)
    obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))


def apply_layout(lay):
    used = set()
    for name in layout.bowls:
        obj = scn[f"bowl_{name}"]
        if name in lay["plates"]:
            x, y, _ = lay["plates"][name]
            place(obj, x, y, TABLE_TOP_Z + 0.003, 0.0)
        else:
            place(obj, 10.0 + len(used), 10.0, -2.0, 0.0)  # not in this frame: park far below the table
            used.add(name)
    free = {k: list(v) for k, v in SLOTS.items()}
    placed = []
    for b in sorted(lay["bricks"], key=lambda b: -b["area"]):
        slots = free[b["kind"]] or free["2x2" if b["kind"] == "2x4" else "2x4"]
        if not slots:
            continue
        pname = slots.pop(0)
        key = "brick" if pname == "Brick" else f"distractor{pname[len('Distractor'):]}"
        z = TABLE_TOP_Z + 0.0192 / 2 + 0.0005 + (0.006 if b["on_plate"] else 0.0)
        place(scn[key], b["x"], b["y"], z, b["yaw"])
        if not args.plain:
            set_brick_color(0, pname, b["color"])
        placed.append((pname, b["color"], round(b["x"], 3), round(b["y"], 3)))
    k = 0
    for kind in free:
        for pname in free[kind]:
            key = "brick" if pname == "Brick" else f"distractor{pname[len('Distractor'):]}"
            place(scn[key], -10.0 - 0.1 * k, 10.0, -2.0, 0.0)
            k += 1
    return placed


def render(state, n):
    for _ in range(n):
        set_joints(state)
        scn.write_data_to_sim()
        sim.step(render=False)
        scn.update(sim.get_physics_dt())
        sync()  # physics -> visual proxies (the GB10 render path reads USD; see usd_sync.py)
        sim.render()
        scn.update(0.0)
    out = {}
    for name, sensor in scn.sensors.items():
        o = sensor.data.output
        if "rgb" in o:
            out[name] = o["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
        if "distance_to_image_plane" in o:
            out[name + "_depth"] = o["distance_to_image_plane"][0].cpu().numpy()[..., 0]
    return out


def lin(x):
    c = np.asarray(x, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def table_ratio(sim_rgb, sim_depth, real_rgb):
    """median linear sim/real ratio over pixels where the sim sees the bare table (depth == table plane)."""
    r_cv, pos, K = camera_from_profile(args.profile)
    v, u = np.mgrid[:480, :640]
    uv = np.c_[u.ravel() + 0.5, v.ravel() + 0.5]
    P = backproject(uv, r_cv, pos, K, TABLE_TOP_Z + 0.0003)
    z_plane = ((P - pos) @ np.asarray(r_cv).T)[:, 2].reshape(480, 640)
    m = (np.abs(sim_depth - z_plane) < 0.002) & (v > 80)
    real_hsv = cv2.cvtColor(real_rgb, cv2.COLOR_RGB2HSV)
    m &= real_hsv[..., 1] < 90  # real wood (not a plate / brick in the median)
    a, b = lin(sim_rgb[m]).mean(1), lin(real_rgb[m]).mean(1)
    return float(np.median(a) / max(np.median(b), 1e-6)), int(m.sum())


ds = Dataset()
OUT.mkdir(parents=True, exist_ok=True)
IGNORE = persistent_mask()

if args.calibrate and not args.plain:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    med = cv2.imread(str(OUT / "top_median.png"))[..., ::-1].copy()
    apply_layout({"plates": {}, "bricks": []})
    set_joints(ds.state(args.episodes[0], 0))
    scale = look.scale()
    for it in range(4):
        img = render(ds.state(args.episodes[0], 0), args.warmup if it == 0 else args.steps * 2)
        ratio, npx = table_ratio(img["top_cam"], img["top_cam_depth"], med)
        print(f"calibrate it {it}: scale {scale:.4f} table sim/real {ratio:.3f} over {npx} px", flush=True)
        if abs(ratio - 1) < 0.02:
            break
        for path in ("/World/Light", "/World/Sun"):
            a = stage.GetPrimAtPath(path).GetAttribute("inputs:intensity")
            a.Set(a.Get() / ratio)
        scale /= ratio
    # backdrop gain: inspect camera, pixels on the billboard vs the real inspect median
    imed = cv2.imread(str(OUT / "inspect_median.png"))[..., ::-1].copy()
    bd = stage.GetPrimAtPath("/World/Looks/Twin/Backdrop/Tex")
    gain = look.bd_gain()
    if bd and "inspect_cam" in img:
        for it in range(3):
            img = render(ds.state(args.episodes[0], 0), args.steps * 2)
            s_rgb = img["inspect_cam"]
            m = np.zeros(s_rgb.shape[:2], bool)
            m[20:200, :] = True  # upper image: billboard only (table and robot are below)
            gr = float(np.median(lin(s_rgb[m]).mean(1)) / max(np.median(lin(imed[m]).mean(1)), 1e-6))
            print(f"backdrop it {it}: gain {gain:.3f} sim/real {gr:.3f}", flush=True)
            if abs(gr - 1) < 0.03:
                break
            gain /= gr
            from pxr import Gf
            bd.GetAttribute("inputs:scale").Set(Gf.Vec4f(gain, gain, gain, 1.0))
    LIGHTING_JSON.write_text(json.dumps({"exposure_scale": scale, "backdrop_gain": gain, "dome_intensity": look.dome_intensity,
                                         "key_intensity": look.key_intensity, "table_ratio_last": ratio,
                                         "note": "linear tonemap; scale so the textured table renders like the real "
                                                 "top-camera median (render_compare.py --calibrate)"}, indent=1))
    cv2.imwrite(str(OUT / "calibrate_top.jpg"), np.concatenate([med, img["top_cam"]], 1)[..., ::-1])
    print("wrote", LIGHTING_JSON, flush=True)

first = True
summary = []
for ep in args.episodes:
    st = ds.state(ep, args.frame)
    real = {c: ds.frame(ep, c, args.frame) for c in ("top", "wrist", "inspect")}
    placed = []
    if not args.no_layout:
        r_cv, pos, K = camera_from_profile(args.profile)
        lay = detect_layout(real["top"], r_cv, pos, K, ignore=IGNORE)
        placed = apply_layout(lay)
    img = render(st, args.warmup if first else args.steps)
    first = False
    rows = []
    for cam, key in (("top", "top_cam"), ("wrist", "wrist_cam"), ("inspect", "inspect_cam")):
        if key not in img:
            continue
        r, s = real[cam], img[key]
        if not args.plain and not args.no_ccm:
            s = color_correct(s, cam)  # per-camera colour response (identity for top)
        blend = (0.5 * r.astype(np.float32) + 0.5 * s.astype(np.float32)).astype(np.uint8)
        row = np.concatenate([r, s, blend], 1)
        cv2.putText(row, f"real {cam}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(row, f"sim {cam}", (648, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rows.append(row)
    grid = np.concatenate(rows, 0)
    tag = ("_plain" if args.plain else "") + args.tag
    p = OUT / f"compare_ep{ep:03d}_f{args.frame:03d}{tag}.jpg"
    cv2.imwrite(str(p), grid[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    stats = {c: [round(float(v), 1) for v in img[k].reshape(-1, 3).mean(0)] for c, k in
             (("top", "top_cam"), ("wrist", "wrist_cam"), ("inspect", "inspect_cam")) if k in img}
    rstats = {c: [round(float(v), 1) for v in real[c].reshape(-1, 3).mean(0)] for c in stats}
    summary.append(dict(episode=ep, frame=args.frame, state=[round(float(v), 2) for v in st], placed=placed,
                        sim_mean_rgb=stats, real_mean_rgb=rstats, image=str(p)))
    print(json.dumps(summary[-1]), flush=True)
(OUT / f"compare_summary{('_plain' if args.plain else '') + args.tag}.json").write_text(json.dumps(summary, indent=1))
print("COMPARE_OK", flush=True)
simapp.close()
