"""Visual domain randomisation check: render a grid of randomised draws next to real frames, and time them.

    sim/twin/run.sh sim/twin/env/test_randomize.py [--draws 12] [--seed 0] [--frames 3]
    sim/twin/run.sh sim/twin/env/test_randomize.py --real-stats      # no Isaac: real-frame statistics only
    sim/twin/run.sh sim/twin/env/test_randomize.py --check           # does each knob change the rendered image?

Outputs (sim/twin/out/dr/):
* dr_grid.jpg      first row: real top | wrist | inspect frames of two episodes; then one tile (top | wrist | inspect,
                   after color_correct + TwinRandomizer.postprocess) per randomised draw, two tiles per row;
* dr_draws.json    every draw (what TwinRandomizer returned), its timing and per-camera image statistics;
* real_stats.json  (--real-stats) brightness / white balance / sharpness spread of the real cal2 frames, which the
                   ranges in randomize.py are chosen to bracket; the grid run prints sim vs real side by side.

Stand-in env: built like visuals/render_compare.py (twin_scene_cfg + apply_twin_visuals + VisualSync), wrapped in a
SimpleNamespace with the attributes the core env exposes (sim, scene, stage, visuals, profile; + sync). Each draw puts
the arm at the recorded joint state of a real frame and the plates / bricks where they are detected in it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "sim/twin/out/dr"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim/residual_env"))

ap = argparse.ArgumentParser()
ap.add_argument("--draws", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--frames", type=int, default=3, help="render calls per draw (timing + RTX settling)")
ap.add_argument("--episodes", type=int, nargs="+", default=[0, 76, 190, 304, 418, 120])
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--real-stats", action="store_true")
ap.add_argument("--check", action="store_true")
ap.add_argument("--restore-frames", type=int, default=30, help="--check: render calls after restore()")
ap.add_argument("--n-real", type=int, default=48, help="episodes sampled for --real-stats")
ap.add_argument("--only", nargs="*", default=None, help="--check: only these knobs")
ap.add_argument("--dump-real", default=None, help=argparse.SUPPRESS)  # subprocess: decode real frames -> npz
args = ap.parse_args()

import numpy as np  # noqa: E402

CAMS = ("top", "wrist", "inspect")
# top-camera patch that is bare table in (almost) every episode (right / bottom of the image, away from the plates)
TABLE_PATCH = (slice(380, 470), slice(380, 630))


def img_stats(img: np.ndarray, cam: str) -> dict:
    """Global brightness, white balance (R/G, B/G of the mean) and sharpness (variance of the Laplacian)."""
    import cv2

    x = img[..., :3].astype(np.float32)
    m = x.reshape(-1, 3).mean(0)
    g = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    d = {"luma": float(0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]), "r_g": float(m[0] / max(m[1], 1)),
         "b_g": float(m[2] / max(m[1], 1)), "sharp": float(cv2.Laplacian(g, cv2.CV_32F).var())}
    if cam == "top":
        p = x[TABLE_PATCH]
        hsv = cv2.cvtColor(img[TABLE_PATCH][..., :3], cv2.COLOR_RGB2HSV)
        p = p[hsv[..., 1] < 90]  # wood, not a plate / brick
        if len(p) > 500:
            t = p.mean(0)
            d.update(table_luma=float(0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]), table_r_g=float(t[0] / t[1]),
                     table_b_g=float(t[2] / t[1]))
    return d


def summarize(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r], float)
        out[k] = {"p5": round(float(np.percentile(v, 5)), 3), "p50": round(float(np.median(v)), 3),
                  "p95": round(float(np.percentile(v, 95)), 3), "n": int(len(v))}
    return out


def real_stats():
    from sim.twin.visuals.realdata import Dataset

    ds = Dataset()
    eps = np.linspace(0, len(ds) - 1, args.n_real).astype(int)
    per = {c: [] for c in CAMS}
    for ep in eps:
        n = len(ds.states(int(ep)))
        idx = [0, n // 2]  # start (arm parked) and mid-episode (arm moving: wrist motion blur)
        for c in CAMS:
            for f in ds.frames(int(ep), c, idx):
                per[c].append(img_stats(f, c))
    res = {c: summarize(v) for c, v in per.items()}
    res["episodes"] = [int(e) for e in eps]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "real_stats.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if args.real_stats:
    real_stats()
    sys.exit(0)


def dump_real(path):
    """Real frames, joint states and detected layouts of args.episodes -> npz. Runs in a subprocess: PyAV must never
    be imported into the Isaac process (every camera then renders zeros)."""
    from sim.twin.visuals.measure_real import detect_layout, persistent_mask
    from sim.twin.visuals.realdata import Dataset
    from sim.twin.visuals.table_texture import camera_from_profile

    ds = Dataset()
    r_cv, pos, K = camera_from_profile("joe_cal2")
    ignore = persistent_mask()
    out, lays = {}, []
    for ep in args.episodes:
        for c in CAMS:
            out[f"{ep}_{c}"] = ds.frame(ep, c, args.frame)
        out[f"{ep}_state"] = ds.state(ep, args.frame)
        lays.append(detect_layout(out[f"{ep}_top"], r_cv, pos, K, ignore=ignore))
    out["layouts"] = np.array(json.dumps(lays, default=float))
    np.savez_compressed(path, **out)


if args.dump_real:
    dump_real(args.dump_real)
    sys.exit(0)

import subprocess  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
REAL_NPZ = OUT / "real_frames.npz"
subprocess.run([sys.executable, __file__, "--dump-real", str(REAL_NPZ), "--frame", str(args.frame), "--episodes",
                *map(str, args.episodes)], check=True)

# ------------------------------------------------------------------------------------------------ Isaac part
from app import KIT_ARGS  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

simapp = AppLauncher(headless=True, enable_cameras=True, kit_args=KIT_ARGS).app

import dataclasses  # noqa: E402
import math  # noqa: E402
import types  # noqa: E402

import cv2  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

import scene as rscene  # noqa: E402
from sim.twin.calib.profile import twin_profile  # noqa: E402
from sim.twin.env.randomize import TwinRandomizer, postprocess  # noqa: E402
from sim.twin.visuals.table_texture import TABLE_TOP_Z  # noqa: E402
from sim.twin.visuals.twin_scene import TwinLook, apply_twin_visuals, color_correct, set_brick_color, \
    twin_scene_cfg, wrist_cam_rel  # noqa: E402
from sim.twin.visuals.usd_sync import VisualSync  # noqa: E402

TWIN = REPO / "sim/twin"
drive = UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=60.0, damping=2.0))
usd = UrdfConverter(UrdfConverterCfg(
    asset_path=str(TWIN / "assets/so101/urdf/so101_new_calib.urdf"), usd_dir=str(TWIN / "assets/so101/usd"),
    usd_file_name="so101_new_calib.usda", fix_base=True, merge_fixed_joints=False, joint_drive=drive)).usd_path
profile = dataclasses.replace(twin_profile(), robot_usd=usd)
layout = rscene.load_layout(str(REPO / "sim/realism/scene_params.json"), "joe")
B = rscene.Brick
s22, s24 = (0.0318, 0.0318, 0.0192), (0.0636, 0.0318, 0.0192)
W = (0.8, 0.8, 0.8)
layout.distractors = [B("d0", s24, (0.24, -0.07), W, 0.018), B("d1", s24, (0.24, 0.0), W, 0.018),
                      B("d2", s22, (0.2, 0.0), W, 0.0093), B("d3", s22, (0.2, 0.05), W, 0.0093),
                      B("d4", s22, (0.2, -0.05), W, 0.0093), B("d5", s22, (0.15, 0.05), W, 0.0093),
                      B("d6", s24, (0.15, -0.05), W, 0.018)]
SLOTS = {"2x2": ["Brick", "Distractor2", "Distractor3", "Distractor4", "Distractor5"],
         "2x4": ["Distractor0", "Distractor1", "Distractor6"]}

look = TwinLook()
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0", use_fabric=False))
scn = InteractiveScene(twin_scene_cfg(1, layout, profile, look))
visuals = apply_twin_visuals(1, layout, profile, look)
sim.reset()
scn.reset()
sync = VisualSync(scn, wrist_cam_rel=wrist_cam_rel())
stage = omni.usd.get_context().get_stage()
env = types.SimpleNamespace(sim=sim, scene=scn, stage=stage, visuals=visuals, profile=profile, sync=sync)

robot = scn["robot"]
dev = robot.device
jidx = [robot.joint_names.index(j) for j in rscene.JOINTS]
off = torch.tensor(profile.joint_offsets, device=dev, dtype=torch.float32)


def set_joints(state):
    s = torch.tensor(state, device=dev, dtype=torch.float32)
    q_ = torch.deg2rad(s) + off
    q_[5] = s[5] * profile.grip_scale + profile.grip_bias
    q = robot.data.default_joint_pos.clone()
    q[0, jidx] = q_
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)


def place(obj, x, y, z, yaw):
    pose = torch.tensor([[x, y, z, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]], device=dev)
    pose[:, :3] += scn.env_origins[:1]
    obj.write_root_pose_to_sim(pose)
    obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))


def key_of(pname):
    return "brick" if pname == "Brick" else f"distractor{pname[len('Distractor'):]}"


def apply_layout(lay):
    k = 0
    for name in layout.bowls:
        if name in lay["plates"]:
            x, y, _ = lay["plates"][name]
            place(scn[f"bowl_{name}"], x, y, TABLE_TOP_Z + 0.003, 0.0)
        else:
            place(scn[f"bowl_{name}"], 10.0 + k, 10.0, -2.0, 0.0)
            k += 1
    free = {kk: list(v) for kk, v in SLOTS.items()}
    for b in sorted(lay["bricks"], key=lambda b: -b["area"]):
        slots = free[b["kind"]] or free["2x2" if b["kind"] == "2x4" else "2x4"]
        if not slots:
            continue
        pname = slots.pop(0)
        z = TABLE_TOP_Z + 0.0192 / 2 + 0.0005 + (0.006 if b["on_plate"] else 0.0)
        place(scn[key_of(pname)], b["x"], b["y"], z, b["yaw"])
        set_brick_color(0, pname, b["color"])  # the env's own colour choice; the randomiser keeps it (+ jitter)
    for kind in free:
        for pname in free[kind]:
            place(scn[key_of(pname)], -10.0 - 0.1 * k, 10.0, -2.0, 0.0)
            k += 1


def render(state, n):
    for _ in range(n):
        set_joints(state)
        scn.write_data_to_sim()
        sim.step(render=False)
        scn.update(sim.get_physics_dt())
        sync()
        sim.render()
        scn.update(0.0)
    return {c: scn.sensors[f"{c}_cam"].data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
            for c in CAMS if f"{c}_cam" in scn.sensors}


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


TW, TH = 320, 240


def tile(imgs: dict, tag: str) -> np.ndarray:
    return np.concatenate([label(cv2.resize(imgs[c], (TW, TH), interpolation=cv2.INTER_AREA), f"{tag} {c}")
                           for c in CAMS], 1)


RD = np.load(REAL_NPZ)
LAYS = json.loads(str(RD["layouts"]))
scenes = [(ep, RD[f"{ep}_state"], {c: RD[f"{ep}_{c}"] for c in CAMS}, LAYS[i]) for i, ep in enumerate(args.episodes)]

if args.check:
    # Each knob alone at an exaggerated setting -> mean abs pixel change per camera vs the nominal render.
    ep, st, real, lay = scenes[0]
    apply_layout(lay)
    base = render(st, 60)
    again = render(st, args.restore_frames)  # noise floor: the same nominal scene rendered again
    print("noise_floor", json.dumps({c: round(float(np.abs(again[c].astype(float) - base[c]).mean()), 2)
                                     for c in again}), flush=True)
    knobs = {
        "top_pose": {"camera": {"cams": ["top"], "focal": False, "scale": 5.0}},
        "top_focal": {"camera": {"cams": ["top"], "pose": False, "scale": 10.0}},
        "inspect_pose": {"camera": {"cams": ["inspect"], "focal": False, "scale": 2.0}},
        "wrist_pose": {"camera": {"cams": ["wrist"], "focal": False, "scale": 2.0}},
        "wrist_focal": {"camera": {"cams": ["wrist"], "pose": False, "scale": 5.0}},
        "light": {"light": {"exposure": (1.6, 1.6)}},
        "material": {"material": {"table_gain": (0.6, 0.6)}},
        "clutter": {"clutter": {"p_cable": 1.0, "p_blob": 1.0}},
        # everything at once with the photo texture variants and the textured dome (restore must be exact)
        "full": {"camera": {}, "light": {"p_dome_texture": 1.0}, "clutter": {},
                 "material": {"table_variants": {"photo_full": 1.0}, "backdrop_variants": {"pano0": 1.0}}},
    }
    def snapshot():
        from pxr import Usd, UsdShade

        snap = {}
        roots = ["/World/Light", "/World/Sun", "/World/Looks/Twin", "/World/envs/env_0/TableTopVisual",
                 "/World/envs/env_0/BackdropVisual", "/World/envs/env_0/TopCam", "/World/envs/env_0/InspectCam",
                 "/World/envs/env_0/TwinVis"]
        for r in roots:
            pr = stage.GetPrimAtPath(r)
            if not pr:
                continue
            for q in Usd.PrimRange(pr):
                for a in q.GetAttributes():
                    if a.HasAuthoredValue():
                        snap[str(a.GetPath())] = repr(a.Get())
                b = UsdShade.MaterialBindingAPI(q).GetDirectBinding().GetMaterialPath()
                if b:
                    snap[str(q.GetPath()) + ":binding"] = str(b)
        return snap

    snap0 = snapshot()
    res = {}
    for name, sub in knobs.items():
        if args.only and name not in args.only:
            continue
        cfg = {k: {"enabled": False} for k in ("camera", "light", "material", "clutter", "image")}
        for k, v in sub.items():
            cfg[k] = dict(v, enabled=True)
        base = render(st, args.restore_frames)  # per-knob baseline (the bricks may still be settling)
        dr = TwinRandomizer(cfg)
        seed = next(k for k in range(1000) if np.random.default_rng(k).uniform() > 0.97)  # first draw near the edge
        dr(env, np.random.default_rng(seed))
        img = render(st, args.frames)
        res[name] = {c: round(float(np.abs(img[c].astype(float) - base[c]).mean()), 2) for c in img}
        dr.restore(env)
        snap1 = snapshot()
        dif = sorted(k for k in set(snap0) | set(snap1) if snap0.get(k) != snap1.get(k))
        print(name, "usd diffs after restore:", len(dif), [(k, snap0.get(k), snap1.get(k)) for k in dif[:12]],
              flush=True)
        img2 = render(st, args.restore_frames)
        res[name]["restored"] = {c: round(float(np.abs(img2[c].astype(float) - base[c]).mean()), 2) for c in img2}
        dimg = {c: np.clip(np.abs(img2[c].astype(int) - base[c]).astype(np.float32) * 8, 0, 255).astype(np.uint8)
                for c in img2}
        cv2.imwrite(str(OUT / f"check_{name}.jpg"), np.concatenate([tile(base, "nominal"), tile(img, name),
                                                                    tile(img2, "restored"), tile(dimg, "|diff| x8")],
                                                                   0)[..., ::-1])
        print(name, json.dumps(res[name]), flush=True)
    (OUT / "check.json").write_text(json.dumps(res, indent=1))
    print("CHECK_OK", flush=True)
    simapp.close()
    sys.exit(0)

dr = TwinRandomizer()
rng = np.random.default_rng(args.seed)
tiles, draws, sim_stats = [], [], {c: [] for c in CAMS}
# warm-up at the first scene (shader compile, first material creation) so timings are steady-state
apply_layout(scenes[0][3])
render(scenes[0][1], 60)
dr(env, np.random.default_rng(12345))
render(scenes[0][1], args.frames)
for i in range(args.draws):
    ep, st, real, lay = scenes[i % len(scenes)]
    apply_layout(lay)
    t0 = time.perf_counter()
    d = dr(env, rng)
    t1 = time.perf_counter()
    img = render(st, args.frames)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    out = {}
    for c in CAMS:
        s = color_correct(img[c], c)  # per-camera colour response (identity for top), then image-space DR
        out[c] = postprocess(s, c, d)
        sim_stats[c].append(img_stats(out[c], c))
    t3 = time.perf_counter()
    tm = {"randomize_s": round(t1 - t0, 4), "render_s": round(t2 - t1, 4), "render_calls": args.frames,
          "postprocess_s": round(t3 - t2, 4)}
    print(f"draw {i:2d} ep {ep}: {json.dumps(tm)}", flush=True)
    draws.append({"draw": i, "episode": ep, "timing": tm, "draw_values": d})
    tiles.append(tile(out, f"#{i}"))

real_row = [tile(scenes[k][2], f"real ep{scenes[k][0]}") for k in range(2)]
rows = [np.concatenate(real_row, 1)]
for k in range(0, len(tiles), 2):
    pair = tiles[k:k + 2]
    if len(pair) == 1:
        pair.append(np.zeros_like(pair[0]))
    rows.append(np.concatenate(pair, 1))
grid = np.concatenate(rows, 0)
cv2.line(grid, (0, TH), (grid.shape[1], TH), (255, 255, 255), 3)
cv2.imwrite(str(OUT / "dr_grid.jpg"), grid[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
sim_sum = {c: summarize(v) for c, v in sim_stats.items()}
real_sum = json.loads((OUT / "real_stats.json").read_text()) if (OUT / "real_stats.json").exists() else {}
print("image statistics, sim DR draws vs real cal2 frames (p5 / p50 / p95):", flush=True)
for c in CAMS:
    for k, v in sim_sum[c].items():
        r = real_sum.get(c, {}).get(k)
        rs = f"{r['p5']:.3g} / {r['p50']:.3g} / {r['p95']:.3g}" if r else "-"
        print(f"  {c:7s} {k:10s} sim {v['p5']:.3g} / {v['p50']:.3g} / {v['p95']:.3g}   real {rs}", flush=True)
rt = np.array([d["timing"]["randomize_s"] for d in draws])
rr = np.array([d["timing"]["render_s"] for d in draws])
timing = {"randomize_s_mean": round(float(rt.mean()), 4), "randomize_s_max": round(float(rt.max()), 4),
          "render_s_mean": round(float(rr.mean()), 4), "render_calls_per_draw": args.frames,
          "randomize_plus_render_s_mean": round(float((rt + rr).mean()), 4)}
print("timing:", json.dumps(timing), flush=True)
(OUT / "dr_draws.json").write_text(json.dumps({"timing": timing, "sim_stats": sim_sum, "draws": draws}, indent=1))
print("wrote", OUT / "dr_grid.jpg", "DR_OK", flush=True)
simapp.close()
