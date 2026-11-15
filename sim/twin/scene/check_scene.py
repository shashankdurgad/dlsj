"""Check the twin's scene content against the real frames (Isaac).

    sim/twin/run.sh sim/twin/scene/check_scene.py [--only grid,bricks] [--episodes ...]

grid:   8 sampled layouts (TwinEnv bricks="data", seeds 100-107) next to 8 real first top frames, and the same 8 real
        episodes re-created with layout_from_episode -> out/scene/grid_sampled_vs_real.jpg, grid_layouts_real_vs_sim.jpg
bricks: Duplo size in pixels, real vs sim, at matched poses: the target brick on the table (top camera, first
        frame), held between the jaws (wrist camera, 10 frames after the gripper closes) and at the inspection mark
        (inspect camera); the brick is placed at the fingertip in the gripper frame at the recorded joint state.
        Sim mask = render difference with / without the brick; real mask = GrabCut seeded by the sim mask (so it
        measures size where the poses agree); top view also in metres (back-projected rectangle, as layout.py).
        Also: visual mesh vs collider vs mass. -> out/scene/bricks_real_vs_sim.jpg, check_scene.json
Real frames are decoded in a subprocess (PyAV in the Isaac process blacks out the cameras).
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/scene"
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="grid,bricks")
ap.add_argument("--episodes", type=int, nargs="*", default=[0, 40, 80, 120, 160, 200, 280, 400])
ap.add_argument("--brick_episodes", type=int, nargs="*", default=[6, 51, 80, 102, 165, 195])
ap.add_argument("--plate_episodes", type=int, nargs="*", default=[6, 15, 24, 33, 45, 54])
ap.add_argument("--legacy_plate", action="store_true", help="visual plates with the pre-fit profile (before image)")
ap.add_argument("--tag", default="")
args = ap.parse_args()
only = set(args.only.split(","))

from sim.twin.env.twin_env import TwinEnv, TwinEnvCfg  # noqa: E402

if args.legacy_plate:  # twin_scene imports plate_mesh at build time: the pre-fit profile for the visual plates
    import functools

    from sim.twin.scene.plate_fit import legacy_profile_params
    from sim.twin.visuals import assets as _assets

    _assets.plate_mesh = functools.partial(_assets.plate_mesh, **legacy_profile_params())
env = TwinEnv(TwinEnvCfg(meas_noise=False))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.twin.scene.layout import grasp_frame, layout_from_episode  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
report_path = OUT / f"check_scene{args.tag}.json"
report = json.loads(report_path.read_text()) if report_path.exists() else {}


def real_frames(ep, want: dict) -> dict:
    """{cam: [frame indices]} -> {cam: [rgb]} decoded in a subprocess, plus the episode's states."""
    with tempfile.TemporaryDirectory() as td:
        code = ("import sys, json, numpy as np; sys.path.insert(0, %r); from sim.twin.visuals.realdata import Dataset; "
                "ds = Dataset(%r); w = json.loads(%r); "
                "np.savez(%r, states=ds.states(%d), **{c: np.stack(ds.frames(%d, c, v)) for c, v in w.items()})"
                % (str(REPO), str(DATA), json.dumps(want), td + "/f.npz", ep, ep))
        subprocess.run([sys.executable, "-c", code], check=True)
        z = np.load(td + "/f.npz")
        return {k: z[k] for k in z.files}


def label(img, text):
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def small(img, w=320):
    return cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)


def save(path, rgb):
    cv2.imwrite(str(path), np.ascontiguousarray(rgb)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])


def dump():
    report_path.write_text(json.dumps(report, indent=1))


def grid(tiles, cols=4):
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    return np.concatenate(rows, 0)


# ------------------------------------------------------------------ grid
if "grid" in only:
    sampled, plans = [], []
    for s in range(8):
        obs = env.reset(seed=100 + s)
        bowl = env.residual.bowl_names[int(env.residual.bowl[0])]
        t = env.scene_plan["target"]
        sampled.append(label(small(obs["images"]["top"]), f"sim seed {100 + s}: {t['color']} {t['kind']} -> {bowl}"))
        plans.append(env.scene_plan)
    real, recreated, views = [], [], []
    for ep in args.episodes:
        lay = layout_from_episode(ep)
        fr = real_frames(ep, {"top": [0], "wrist": [0], "inspect": [0]})
        t = lay["target"] or {}
        real.append(label(small(fr["top"][0]), f"real ep {ep}: {t.get('color')} {t.get('kind')} -> {lay['bowl']}"))
        obs = env.reset(seed=ep, layout=lay)
        recreated.append(label(small(obs["images"]["top"]), f"sim layout ep {ep}"))
        if len(views) < 3:  # all three cameras, real over sim, for the first episodes
            views.append(np.concatenate([np.concatenate([label(small(fr[c][0]), f"real {c} ep {ep}") for c in
                                                         ("top", "wrist", "inspect")], 1),
                                         np.concatenate([label(small(obs["images"][c]), f"sim {c}") for c in
                                                         ("top", "wrist", "inspect")], 1)], 0))
    gap = np.full((8, 320 * 4, 3), 255, np.uint8)
    save(OUT / f"grid_sampled_vs_real{args.tag}.jpg", np.concatenate([grid(sampled), gap, grid(real)], 0))
    pairs = [np.concatenate([a, b], 0) for a, b in zip(real, recreated, strict=True)]
    save(OUT / f"grid_layouts_real_vs_sim{args.tag}.jpg", grid(pairs))
    save(OUT / f"layouts_all_cameras{args.tag}.jpg", np.concatenate(views, 0))
    report["grid"] = {"sampled_plans": plans, "episodes": args.episodes}
    print("GRID ok", flush=True)
    dump()


# ------------------------------------------------------------------ bricks
def rect(mask):
    """(area px, [long, short] px of the minAreaRect, centroid) of a boolean mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    (_, _), (w, h), _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
    return {"area": int(len(xs)), "rect": [round(max(w, h), 1), round(min(w, h), 1)],
            "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]}


def sim_mask(R, img_with, cam):
    """Exact sim brick mask: render with the brick moved away and diff."""
    pose = R.brick.data.root_pose_w[:1].clone() if hasattr(R.brick.data, "root_pose_w") else \
        R.brick.data.root_state_w[:1, :7].clone()
    away = pose.clone()
    away[:, 2] += 5.0
    R.brick.write_root_pose_to_sim(away)
    img_without = env.render()[cam]
    R.brick.write_root_pose_to_sim(pose)
    d = np.abs(img_with.astype(np.int16) - img_without.astype(np.int16)).max(-1) > 12
    d = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(d)
    if n < 2:
        return None
    return lb == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def real_mask(img, smask, grow: int = 25):
    """Real brick region by GrabCut seeded with the sim brick: its eroded core is probably brick, a ring of `grow` px
    around it probably background, everything further out background (measures size where the poses agree)."""
    H, W = smask.shape
    sm = smask.astype(np.uint8)
    gc = np.full((H, W), cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(sm, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0] = cv2.GC_PR_BGD
    gc[sm > 0] = cv2.GC_PR_FGD
    if int(sm.sum()) < 30:
        return None
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(np.ascontiguousarray(img[..., ::-1]), gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    return lb == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def metric_rect(mask, cam, z):
    """minAreaRect (long, short) in metres of a top-view mask back-projected to the plane z (sides included)."""
    from sim.twin.scene.layout import backproject

    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    P = backproject(max(cs, key=cv2.contourArea)[:, 0, :].astype(float), cam, z)[:, :2].astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(P)
    return [round(float(max(w, h)), 4), round(float(min(w, h)), 4)]


def sim_top_camera():
    """The twin's top camera (profile rows, centred principal point) as (R_world_to_cv, pos, K)."""
    rows = np.asarray(env.profile.top.rows, float)
    f = env.profile.top.f_px * env.cfg.width / 640.0
    K = np.array([[f, 0, env.cfg.width / 2], [0, f, env.cfg.height / 2], [0, 0, 1.0]])
    return np.stack([rows[0, :3], -rows[1, :3], -rows[2, :3]]), rows[3, :3].copy(), K


def crop(img, c, half=70, masks=()):
    img = np.ascontiguousarray(img.copy())
    for m, col in masks:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, col, 1)
    x, y = int(c[0]), int(c[1])
    pad = np.pad(img, ((half, half), (half, half), (0, 0)))
    return cv2.resize(pad[y:y + 2 * half, x:x + 2 * half], (240, 240), interpolation=cv2.INTER_NEAREST)


def hold_brick(R):
    """Brick at the fingertip with the gripper's orientation (sides on the jaws, studs toward the wrist)."""
    tip = R.tip_w()[0]
    q = R.robot.data.body_quat_w[0, R.grip_body]
    R.brick.write_root_pose_to_sim(torch.cat([tip, q]).unsqueeze(0))
    R.brick.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))


if "bricks" in only:
    from sim.twin.visuals import assets

    R = env.residual
    labels = {json.loads(s)["episode_index"]: json.loads(s) for s in
              (DATA / "meta/dlsj/labels.jsonl").read_text().splitlines() if s.strip()}
    out, rows = [], []
    for ep in args.brick_episodes:
        lay = layout_from_episode(ep)
        insp = int(labels[ep]["inspection_frame"])
        fr0 = real_frames(ep, {"top": [0]})
        S = fr0["states"]
        th = min(grasp_frame(S, insp) + 10, insp)
        fr = real_frames(ep, {"wrist": [th], "inspect": [insp]})
        real = {"top": fr0["top"][0], "wrist": fr["wrist"][0], "inspect": fr["inspect"][0]}
        res = {"episode": ep, "target": {k: lay["target"].get(k) for k in ("color", "kind", "x", "y")},
               "hold_frame": th, "inspection_frame": insp}
        tiles = []
        for cam, t in (("top", 0), ("wrist", th), ("inspect", insp)):
            env.reset(seed=ep, layout=lay, joint_deg=S[t], render=False)
            if cam != "top":
                hold_brick(R)
            img = env.render()[cam]
            sm = sim_mask(R, img, cam)
            rm = real_mask(real[cam], sm) if sm is not None else None
            res[cam] = {"sim": rect(sm) if sm is not None else None, "real": rect(rm) if rm is not None else None}
            if cam == "top" and sm is not None:  # metric size with the layout detector's method (sides included)
                res[cam]["sim_m"] = metric_rect(sm, sim_top_camera(), -0.002 + 0.0192)
                res[cam]["real_m_detector"] = [lay["target"].get("long"), lay["target"].get("short")]
            if sm is not None and rm is not None:
                res[cam]["long_side_ratio_sim_over_real"] = round(res[cam]["sim"]["rect"][0] /
                                                                  res[cam]["real"]["rect"][0], 3)
                res[cam]["area_ratio_sim_over_real"] = round(res[cam]["sim"]["area"] / res[cam]["real"]["area"], 3)
            if sm is not None:
                c = rect(sm)["centroid"]
                half = {"top": 40, "wrist": 130, "inspect": 80}[cam]
                ms = [(sm, (0, 255, 0))] + ([(rm, (255, 0, 255))] if rm is not None else [])
                tiles += [label(crop(real[cam], c, half, ms[1:]), f"real {cam} t={t}"),
                          label(crop(img, c, half, ms[:1]), f"sim {cam}")]
        out.append(res)
        print("BRICK", json.dumps(res), flush=True)
        if tiles:
            rows.append(label(np.concatenate(tiles, 1), f"ep {ep} {res['target']['color']} {res['target']['kind']}"))
    if rows:
        w = max(r.shape[1] for r in rows)
        save(OUT / f"bricks_real_vs_sim{args.tag}.jpg",
             np.concatenate([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows], 0))
    # visual mesh vs collider vs mass
    cons = {}
    for _name, obj, spec, kind in env.bricks.items if env.bricks else []:
        if kind in cons:
            continue
        m = obj.root_view.get_masses()
        try:
            import warp as wp

            m = wp.to_torch(m)
        except Exception:  # noqa: BLE001
            pass
        p = assets.brick_mesh(kind)[0]
        body = p[p[:, 2] <= p[:, 2].min() + spec.size[2] + 1e-6]
        cons[kind] = {"collider_box_m": [round(v, 4) for v in spec.size],
                      "visual_body_m": np.round(body.max(0) - body.min(0), 4).tolist(),
                      "visual_with_studs_m": np.round(p.max(0) - p.min(0), 4).tolist(),
                      "mass_kg_sim": round(float(torch.as_tensor(m).flatten()[0]), 5)}
    report["bricks"] = {"episodes": out, "mesh_collider_mass": cons}
    print("CONSISTENCY", json.dumps(cons), flush=True)
    dump()

# ------------------------------------------------------------------ plates (close-ups, inspect + top)
if "plates" in only:
    from sim.twin.scene.layout import project, top_camera
    from sim.twin.scene.plate_fit import inspect_camera

    cams = {"top": top_camera(), "inspect": inspect_camera()}
    tiles = []
    for ep in args.plate_episodes:
        lay = layout_from_episode(ep)
        fr = real_frames(ep, {"top": [0], "inspect": [0]})
        obs = env.reset(seed=ep, layout=lay, joint_deg=fr["states"][0])
        for name, p in lay["plates"].items():
            row = []
            for cam in ("inspect", "top"):
                uv, z = project(np.array([[p["x"], p["y"], 0.01]]), cams[cam])
                half = int(min(160, 1.3 * p["diam"] / 2 * cams[cam][2][0, 0] / max(z[0], 0.05)))
                row += [label(crop(fr[cam][0], uv[0], half), f"real {cam} ep {ep} {name}"),
                        label(crop(obs["images"][cam], uv[0], half), f"sim {cam}{args.tag}")]
            tiles.append(np.concatenate(row, 1))
    save(OUT / f"plates_real_vs_sim{args.tag}.jpg", np.concatenate(tiles[:8], 0))
    print("PLATES ok", len(tiles), flush=True)

print("CHECK_SCENE_DONE", flush=True)
env.close()
