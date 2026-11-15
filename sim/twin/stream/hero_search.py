"""Search for the live mirror's hero viewpoint over representative REAL poses (cal2 recordings).

    sim/twin/run.sh sim/twin/stream/hero_search.py [--top 6] [--stage2 24]

Poses (LeRobot degrees, cal2): rest, mid-reach, grasps at table height near the real deploy brick positions
(out/failure_match/real_brick_xy.json; the recorded grasp whose fingertip lies nearest), lift / carry, present to the
inspect camera, and place over each plate (yellow / orange / purple), from the cal2 dataset's labelled frames.

Stage 1 (geometry, no rendering): the robot visual meshes' vertices (sampled per link, posed from the physics bodies)
projected into every candidate camera: azimuth (15 deg steps, full circle) x elevation x focal, looking at the
centroid of the working volume, distance solved so that the arm's median bbox height at working poses is ~60 %.
Terms: (a) clipping, (b) height, (d) bricks / plates of the grasp area in frame, (f) centring + 3/4 angle +
elevation. Stage 2 (renders, real look, dark background) for the best stage-1 candidates: arm mask = full render vs
robot hidden, gripper mask = full vs gripper subtree hidden; (a) / (b) from the arm mask, (c) visible gripper area at
grasp poses, (e) bright clutter (luma > 185) behind the arm in the robot-hidden render. Writes
out/stream/hero_search/{stage1.json, stage2.json, ranked.json, <cand>_<pose>.jpg} and out/stream/hero_candidates.jpg.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim.twin.stream.live import OUT, Mirror, look_at_rows, parse_args, set_hero
from sim.twin.stream.server import Shared

DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
BRICKS = REPO / "sim/twin/out/failure_match/real_brick_xy.json"
STATS = REPO / "sim/twin/scene/scene_stats.json"
DIR = OUT / "hero_search"
W, H, HA = 960, 540, 20.955
TABLE_Z = 0.0
WORK = ("mid_reach", "grasp_1", "grasp_2", "grasp_3", "grasp_4", "lift", "inspect", "place_yellow", "place_orange",
        "place_purple")  # rest excluded from the height target
GRASP = ("grasp_1", "grasp_2", "grasp_3", "grasp_4")
SHEET_POSES = ("grasp_1", "lift", "place_orange", "mid_reach")


def log(*a):
    print("[hero_search]", *a, flush=True)


# ---------------------------------------------------------------------------------------------- poses
def real_poses(m: Mirror) -> tuple[dict, dict]:
    """{name: deg (6,)} and {name: fingertip xyz} from the cal2 dataset."""
    from bases import load_episodes  # residual env (on sys.path once TwinEnv is built)

    eps = load_episodes(DATA)
    r = m.env.residual
    tips, frames = [], []
    for ep, d in eps.items():
        S, gf, insp = d["state"], d["grasp_frame"], d["inspection_frame"]
        if gf is None or insp is None or d.get("excluded") or not (0 < gf < insp < len(S) - 5):
            continue
        m.pose(S[gf])
        tips.append((r.tip_w() - m.env.scene.env_origins)[0].detach().cpu().numpy())
        frames.append((ep, int(gf), int(insp)))
    tips = np.asarray(tips)
    log(f"{len(frames)} labelled episodes; grasp tips x {tips[:, 0].min():.3f}..{tips[:, 0].max():.3f} "
        f"y {tips[:, 1].min():.3f}..{tips[:, 1].max():.3f}")
    deploy = np.asarray([b["xy_m"] for b in json.loads(BRICKS.read_text())], float)
    # four grasp targets spanning the deploy positions: nearest / farthest / leftmost / rightmost
    rad = np.linalg.norm(deploy, axis=1)
    targets = [deploy[rad.argmin()], deploy[rad.argmax()], deploy[deploy[:, 1].argmin()], deploy[deploy[:, 1].argmax()]]
    poses, tipd = {}, {}
    used = set()
    for k, t in enumerate(targets, 1):
        order = np.argsort(np.linalg.norm(tips[:, :2] - t, axis=1))
        i = next(int(j) for j in order if frames[int(j)][0] not in used)
        ep, gf, insp = frames[i]
        used.add(ep)
        poses[f"grasp_{k}"] = eps[ep]["state"][gf]
        tipd[f"grasp_{k}"] = tips[i]
        log(f"grasp_{k}: deploy xy {np.round(t, 3).tolist()} -> episode {ep} frame {gf}, tip {np.round(tips[i], 3)}")
    ep, gf, insp = frames[int(np.argsort(np.linalg.norm(tips[:, :2] - deploy.mean(0), axis=1))[0])]
    S = eps[ep]["state"]
    poses["rest"] = S[0]
    poses["mid_reach"] = S[gf // 2]
    poses["lift"] = S[(gf + insp) // 2]
    poses["inspect"] = S[insp]
    for bowl in ("yellow", "orange", "purple"):
        best = None
        for ep2, _gf2, insp2 in frames:
            if eps[ep2]["bowl"] != bowl:
                continue
            g = eps[ep2]["state"][:, 5]
            rel = next((f for f in range(insp2 + 10, len(g)) if g[f] > g[insp2] + 10), None)
            if rel is not None:
                best = (ep2, max(insp2, rel - 3))
                break
        if best:
            poses[f"place_{bowl}"] = eps[best[0]]["state"][best[1]]
            log(f"place_{bowl}: episode {best[0]} frame {best[1]}")
    for k, q in poses.items():
        m.pose(q)
        tipd[k] = (r.tip_w() - m.env.scene.env_origins)[0].detach().cpu().numpy()
    return {k: np.asarray(v, float) for k, v in poses.items()}, tipd


# ---------------------------------------------------------------------------------------------- geometry
def link_points(m: Mirror, per_link: int = 400, seed: int = 0) -> list:
    """[(body index, (n,4) homogeneous points in the link frame)] from the RobotVisual meshes (child links pruned)."""
    from pxr import Usd, UsdGeom

    rng = np.random.default_rng(seed)
    m.env.sync()
    links = m.env.sync.links[0]
    link_paths = {str(p.GetPath()) for _, p in links}
    out = []
    for bi, lp in links:
        lw = np.array(UsdGeom.Xformable(lp).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        inv = np.linalg.inv(lw)
        pts = []
        it = iter(Usd.PrimRange(lp))
        for prim in it:
            if prim != lp and str(prim.GetPath()) in link_paths:
                it.PruneChildren()
                continue
            if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
                continue
            p = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if not p:
                continue
            p = np.asarray(p, float)
            mw = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            pts.append(np.c_[p, np.ones(len(p))] @ mw @ inv)
        if pts:
            p = np.concatenate(pts)
            if len(p) > per_link:
                p = p[rng.choice(len(p), per_link, replace=False)]
            out.append((bi, p))
    return out


def body_mats(m: Mirror) -> np.ndarray:
    from sim.twin.visuals.usd_sync import _mat

    o = m.env.scene.env_origins[0].detach().cpu().numpy()
    bp = m.robot.data.body_pos_w[0].detach().cpu().numpy() - o
    bq = m.robot.data.body_quat_w[0].detach().cpu().numpy()
    return np.stack([np.array(_mat(bp[i], bq[i])) for i in range(len(bp))])


def project(pts: np.ndarray, eye, target, focal: float):
    """(n,3) world -> (u, v) pixels and depth for a look-at camera."""
    m = look_at_rows(eye, target)
    pc = (pts - m[3, :3]) @ m[:3, :3].T
    z = -pc[:, 2]
    f = focal / HA * W
    zz = np.maximum(z, 1e-3)
    return W / 2 + f * pc[:, 0] / zz, H / 2 - f * pc[:, 1] / zz, z


def bbox(u, v):
    return np.array([u.min(), v.min(), u.max(), v.max()])  # exact mesh vertices: no outliers


def clipped(uvz, margin: float = 0.03) -> bool:
    u, v, z = uvz
    return bool((z < 0.05).any() or u.min() < margin * W or v.min() < margin * H or u.max() > (1 - margin) * W or
                v.max() > (1 - margin) * H)


def cam_eye(target, az_deg, el_deg, dist):
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.asarray(target) + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def stage1(pose_pts: dict, scene_pts: np.ndarray, target, reach_dir_deg: float):
    res = []
    for focal in (14.0, 18.0, 24.0):
        for el in (15.0, 22.0, 30.0, 40.0, 50.0):
            for az in range(0, 360, 15):
                dist = 0.8
                for _ in range(3):  # solve the distance for a 60 % median bbox height at working poses
                    eye = cam_eye(target, az, el, dist)
                    hs = []
                    for k in WORK:
                        u, v, z = project(pose_pts[k], eye, target, focal)
                        if (z < 0.05).any():
                            hs.append(2.0)
                            continue
                        b = bbox(u, v)
                        hs.append((b[3] - b[1]) / H)
                    dist *= float(np.median(hs)) / 0.60
                    dist = float(np.clip(dist, 0.25, 2.6))
                for _ in range(20):  # (a) is a hard constraint: back off until no pose clips (3 % margin)
                    eye = cam_eye(target, az, el, dist)
                    if not any(clipped(project(p, eye, target, focal)) for p in pose_pts.values()):
                        break
                    dist = min(2.6, dist * 1.04)
                eye = cam_eye(target, az, el, dist)
                if eye[2] < TABLE_Z + 0.08 or np.hypot(eye[0] - 0.3, eye[1]) > 2.8:
                    continue
                clip, hs, cx, cy = 0, [], [], []
                for k, p in pose_pts.items():
                    u, v, z = project(p, eye, target, focal)
                    b = bbox(u, v)
                    clip += int(clipped((u, v, z)))
                    if k in WORK:
                        hs.append((b[3] - b[1]) / H)
                        cx.append((b[0] + b[2]) / 2 / W)
                        cy.append((b[1] + b[3]) / 2 / H)
                u, v, z = project(scene_pts, eye, target, focal)
                vis = float(((u > 0) & (u < W) & (v > 0) & (v < H) & (z > 0)).mean())
                h_med = float(np.median(hs))
                ang = abs((az - reach_dir_deg + 180) % 360 - 180)  # 0 = looking along the reach from in front
                s = (3.0 * clip / len(pose_pts) + 4.0 * max(0.0, abs(h_med - 0.6) - 0.05) + 1.5 * (1 - vis) +
                     1.0 * (abs(np.mean(cx) - 0.5) + abs(np.mean(cy) - 0.5)) + 0.6 * abs(ang - 60) / 60 +
                     0.4 * abs(el - 30) / 30)
                res.append(dict(az=az, el=el, focal=focal, dist=round(dist, 3), eye=np.round(eye, 4).tolist(),
                                target=np.round(target, 4).tolist(), clip=clip, h_med=round(h_med, 3),
                                vis=round(vis, 3),
                                cx=round(float(np.mean(cx)), 3), cy=round(float(np.mean(cy)), 3), ang=ang,
                                s1=round(float(s), 4)))
    res.sort(key=lambda d: d["s1"])
    return res


# ---------------------------------------------------------------------------------------------- stage 2
def visible(prim, on: bool):
    from pxr import UsdGeom

    im = UsdGeom.Imageable(prim)
    im.MakeVisible() if on else im.MakeInvisible()


def shot(m: Mirror, passes: int = 4):
    img = None
    for _ in range(12):
        for _ in range(passes):
            img = m.render()
        if img is not None:
            return img
    return np.zeros((H, W, 3), np.uint8)


def masks(a, b):
    """Pixels where render a differs from render b (the hidden part): strong difference only (the glossy table's
    reflections and soft shadows of the arm stay below 60), opened, small specks dropped."""
    import cv2

    d = (np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1) > 60).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 150
    return keep[lab]


class WhiteArm:
    """Temporarily paint the whole robot visual flat white (the black arm against the dark room would not show up in
    a render difference)."""

    def __init__(self, stage, robot_prim):
        from pxr import Gf, Sdf, UsdShade

        self.root = robot_prim
        self.mat = UsdShade.Material.Define(stage, "/World/Looks/HeroSearchWhite")
        sh = UsdShade.Shader.Define(stage, "/World/Looks/HeroSearchWhite/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
        self.mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

    def __call__(self, on: bool):
        from pxr import UsdShade

        api = UsdShade.MaterialBindingAPI.Apply(self.root)
        if on:
            api.Bind(self.mat, UsdShade.Tokens.strongerThanDescendants)
        else:
            api.UnbindDirectBinding()


def arm_height(m: Mirror, poses: dict, keys, robot_prim, white) -> float:
    """Median rendered arm-mask height (fraction of H) over `keys` (distance calibration pre-pass)."""
    hs = []
    white(True)
    for k in keys:
        m.pose(poses[k])
        a = shot(m)
        visible(robot_prim, False)
        n = shot(m)
        visible(robot_prim, True)
        ys = np.nonzero(masks(a, n))[0]
        if len(ys):
            hs.append((ys.max() - ys.min()) / H)
    white(False)
    return float(np.median(hs)) if hs else 0.0


def stage2(m: Mirror, cands: list, poses: dict, tips: dict, robot_prim, grip_prim):
    import cv2

    white = WhiteArm(m.stage, robot_prim)
    names = [k for k in ("rest", "mid_reach", "grasp_1", "grasp_2", "grasp_4", "lift", "inspect", "place_yellow",
                         "place_orange", "place_purple") if k in poses]
    out = []
    for ci, c in enumerate(cands):
        set_hero(m.stage, c["eye"], c["target"], c["focal"])
        h0 = arm_height(m, poses, ("grasp_2", "mid_reach", "place_orange"), robot_prim, white)
        if h0 > 0.62:  # rendered taller than the proxy said: back off (never move closer: stage 1 cleared clipping)
            c = dict(c, dist=round(float(np.clip(c["dist"] * h0 / 0.60, 0.25, 2.6)), 3))
            c["eye"] = np.round(cam_eye(c["target"], c["az"], c["el"], c["dist"]), 4).tolist()
            set_hero(m.stage, c["eye"], c["target"], c["focal"])
        f_px = c["focal"] / HA * W
        rec = {"clip": 0, "h": {}, "gvis": {}, "clutter": [], "cx": [], "cy": []}
        for k in names:
            m.pose(poses[k])
            a = shot(m)  # the real look (contact sheet)
            white(True)
            w = shot(m)
            visible(grip_prim, False)
            b = shot(m)
            visible(grip_prim, True)
            visible(robot_prim, False)
            n = shot(m)
            visible(robot_prim, True)
            white(False)
            arm = masks(w, n)
            grip = masks(w, b)
            cv2.imwrite(str(DIR / f"c{ci:02d}_{k}.jpg"), a[..., ::-1])
            ys, xs = np.nonzero(arm)
            if len(xs) < 50:
                rec["clip"] += 1
                continue
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            e = 3
            rec["clip"] += int(arm[:e].any() or arm[-e:].any() or arm[:, :e].any() or arm[:, -e:].any())
            if k in WORK:
                rec["h"][k] = round(float((y1 - y0) / H), 3)
                rec["cx"].append(float((x0 + x1) / 2 / W))
                rec["cy"].append(float((y0 + y1) / 2 / H))
            # visible gripper area in m^2 at the fingertip depth (0.006 m^2 ~ the whole gripper side-on)
            _, _, z = project(tips[k][None], c["eye"], c["target"], c["focal"])
            rec["gvis"][k] = round(float(grip.sum() * (z[0] / f_px) ** 2 / 0.006), 3)
            # bright clutter in the robot-hidden render behind the arm (bbox grown by 25 %)
            gx, gy = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
            sl = n[int(max(0, y0 - gy)):int(min(H, y1 + gy)), int(max(0, x0 - gx)):int(min(W, x1 + gx))]
            luma = sl @ np.array([0.299, 0.587, 0.114])
            rec["clutter"].append(float((luma > 185).mean()))
        h_med = float(np.median(list(rec["h"].values()))) if rec["h"] else 0.0
        g = [min(1.0, rec["gvis"].get(k, 0.0)) for k in GRASP if k in rec["gvis"]]
        gv = float(np.mean(g)) if g else 0.0
        clutter = float(np.mean(rec["clutter"])) if rec["clutter"] else 1.0
        cen = abs(np.mean(rec["cx"]) - 0.5) + abs(np.mean(rec["cy"]) - 0.5) if rec["cx"] else 1.0
        s2 = (5.0 * rec["clip"] + 4.0 * max(0.0, abs(h_med - 0.6) - 0.05) + 2.0 * (1 - gv) +
              4.0 * clutter + 1.0 * cen + 0.6 * abs(c["ang"] - 60) / 60 + 0.4 * abs(c["el"] - 30) / 30 +
              1.0 * (1 - c["vis"]))
        c2 = dict(c, idx=ci, clip2=rec["clip"], h2=round(h_med, 3), gvis=round(gv, 3), clutter=round(clutter, 4),
                  cen2=round(float(cen), 3), s2=round(float(s2), 4), heights=rec["h"], gvis_pose=rec["gvis"])
        out.append(c2)
        log(f"cand {ci:02d} az {c['az']:3d} el {c['el']:4.1f} f {c['focal']:4.1f} d {c['dist']:.2f}: "
            f"clip {rec['clip']} "
            f"h {h_med:.2f} grip {gv:.2f} clutter {clutter:.3f} cen {cen:.2f} -> s2 {s2:.3f}")
    out.sort(key=lambda d: d["s2"])
    return out


def contact_sheet(ranked: list, path: Path):
    import cv2

    tiles = []
    for li, c in enumerate(ranked):
        quad = []
        for k in SHEET_POSES:
            f = DIR / f"c{c['idx']:02d}_{k}.jpg"
            img = cv2.imread(str(f)) if f.exists() else np.zeros((H, W, 3), np.uint8)
            quad.append(cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
        tile = np.vstack([np.hstack(quad[:2]), np.hstack(quad[2:])])
        lab = (f"{'ABCDEF'[li]}  az {c['az']} el {c['el']:.0f} f {c['focal']:.0f}mm d {c['dist']:.2f}  "
               f"h {c['h2']:.2f} grip {c['gvis']:.2f} clutter {c['clutter']:.3f} score {c['s2']:.2f}")
        cv2.rectangle(tile, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(tile, lab, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, " | ".join(SHEET_POSES), (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                    cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--stage2", type=int, default=20)
    a = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args(["--look", "real", "--bg", "dark"])
    m = Mirror(args, Shared(args.width, args.height))
    t0 = time.time()
    poses, tips = real_poses(m)
    log("poses:", {k: np.round(v, 1).tolist() for k, v in poses.items()})
    lp = link_points(m)
    log(f"{len(lp)} visual links, {sum(len(p) for _, p in lp)} sample points")
    pose_pts = {}
    for k, q in poses.items():
        m.pose(q)
        mats = body_mats(m)
        pose_pts[k] = np.concatenate([(p @ mats[bi])[:, :3] for bi, p in lp])
    allp = np.concatenate([pose_pts[k] for k in WORK if k in pose_pts])
    target = np.array([np.percentile(allp[:, 0], 50), np.percentile(allp[:, 1], 50),
                       0.5 * (np.percentile(allp[:, 2], 5) + np.percentile(allp[:, 2], 95))])
    reach = math.degrees(math.atan2(target[1], target[0]))  # base (origin) -> working centroid
    deploy = np.asarray([b["xy_m"] for b in json.loads(BRICKS.read_text())], float)
    stats = json.loads(STATS.read_text())
    plates = {}
    for L in stats.get("plate_layouts", []):
        for k, v in L.items():
            plates.setdefault(k, []).append(v)
    ring = np.c_[np.cos(np.linspace(0, 2 * np.pi, 9)[:-1]), np.sin(np.linspace(0, 2 * np.pi, 9)[:-1])] * 0.09
    scene_pts = [np.c_[deploy, np.full(len(deploy), 0.01)]]
    for v in plates.values():
        c = np.median(np.asarray(v), 0)
        scene_pts.append(np.c_[c + ring, np.zeros(8)])
    scene_pts = np.concatenate(scene_pts)
    log(f"working centroid {np.round(target, 3).tolist()}, reach direction {reach:.0f} deg")
    s1 = stage1(pose_pts, scene_pts, target, reach)
    (DIR / "stage1.json").write_text(json.dumps(s1, indent=1))
    log(f"stage 1: {len(s1)} candidates in {time.time() - t0:.0f} s; best {s1[0]}")
    picked, per_az = [], {}
    for c in s1:  # diversity: at most 2 per azimuth
        if per_az.get(c["az"], 0) >= 2 or c["clip"] > 0:
            continue
        per_az[c["az"]] = per_az.get(c["az"], 0) + 1
        picked.append(c)
        if len(picked) >= a.stage2:
            break
    robot_prim = m.stage.GetPrimAtPath("/World/envs/env_0/RobotVisual")
    gname = "gripper_link"
    gi = m.robot.body_names.index(gname)
    grip_prim = next(p for bi, p in m.env.sync.links[0] if bi == gi)
    s2 = stage2(m, picked, poses, tips, robot_prim, grip_prim)
    (DIR / "stage2.json").write_text(json.dumps(s2, indent=1))
    top = s2[:a.top]
    (DIR / "ranked.json").write_text(json.dumps(top, indent=1))
    contact_sheet(top, OUT / "hero_candidates.jpg")
    for li, c in enumerate(top):
        log(f"{'ABCDEF'[li]}: eye {c['eye']} target {c['target']} focal {c['focal']} (az {c['az']} el {c['el']} "
            f"d {c['dist']}) s2 {c['s2']}")
    log(f"done in {time.time() - t0:.0f} s")
    sys.stdout.flush()
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
