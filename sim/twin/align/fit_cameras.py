"""Target-free camera calibration from recorded episodes: the robot is the calibration object.

For sampled frames (make_samples.py) we know the joint state, hence the robot's 3D surface (export_points.py, the
sim's own kinematics). Each camera's pose is fitted so the projected robot overlays the real robot:
  real mask   = pixels that differ from the episode's median background AND are dark (black arm) or red (jaws);
  sim outline = per-link convex hulls of the projected surface points (moving links only; the base is static);
  loss        = mean distance of projected points to the real mask + mean distance of real-mask pixels near the
                sim robot (a band around it, so people / clutter elsewhere don't count) to the sim outline.
Intrinsics are the D415 factory values (calib/realsense_intrinsics.json); the start pose is the joe_cal2 camera.
Episodes are split 2:1 into fit / held-out; the report compares joe_cal2 as-is, joe_cal2 pose + factory intrinsics,
and the fitted pose, by held-out IoU and loss.

    docker run ... python sim/twin/align/fit_cameras.py [--cams top inspect]
Writes sim/twin/calib/cameras_fit.json and sim/twin/out/align/overlay_<cam>.jpg.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
TWIN = HERE.parent
REPO = TWIN.parents[1]
sys.path.insert(0, str(HERE))
from frames import states, video_frames  # noqa: E402

SCALE = 0.5  # masks / distance transforms at 320x240
MOVING = ("upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", "moving_jaw_so101_v1_link")
CAP = 25.0  # px (at SCALE) cap on per-point distances


def load_joe(cam: str) -> dict:
    ref = json.loads((REPO / "sim/residual_env/assets/joe_cal2/reference-cal2.json").read_text())
    c = ref["camera" if cam == "top" else "inspect_camera"]
    W, H = c.get("resolution", [640, 480])
    pp = c.get("principal_point", [W / 2, H / 2])
    return {
        "R": np.array(c["world_to_cv"], float),
        "t": np.array(c["position"], float),
        "fx": c["focal_px"],
        "fy": c["focal_px"],
        "cx": pp[0],
        "cy": pp[1],
    }


def factory(cam: str) -> dict:
    k = json.loads((TWIN / "calib/realsense_intrinsics.json").read_text())[cam]
    return {"fx": k["fx"], "fy": k["fy"], "cx": k["cx"], "cy": k["cy"]}


def project(P: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    Xc = (P - cam["t"]) @ cam["R"].T
    z = Xc[..., 2]
    ok = z > 0.02
    u = cam["fx"] * Xc[..., 0] / np.where(ok, z, 1) + cam["cx"]
    v = cam["fy"] * Xc[..., 1] / np.where(ok, z, 1) + cam["cy"]
    return np.stack([u, v], -1), ok


def real_mask(fr: np.ndarray, bg: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(fr, cv2.COLOR_RGB2HSV)
    diff = np.abs(fr.astype(np.int16) - bg.astype(np.int16)).max(-1) > 40
    dark = hsv[..., 2] < 70
    red = ((hsv[..., 0] < 8) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 70)
    m = (diff & (dark | red)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.resize(m, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    return m


def sim_mask(uv: np.ndarray, ok: np.ndarray, link: np.ndarray, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for li in np.unique(link):
        s = (link == li) & ok
        if s.sum() < 3:
            continue
        hull = cv2.convexHull((uv[s] * SCALE).astype(np.float32))
        if np.abs(hull).max() > 1e5:
            continue
        cv2.fillConvexPoly(m, np.round(hull).astype(np.int32), 1)
    return m


class Problem:
    def __init__(self, cam_name, items, pts, link):
        self.cam_name, self.items, self.pts, self.link = cam_name, items, pts, link
        self.h, self.w = int(480 * SCALE), int(640 * SCALE)

    def loss(self, cam, return_iou=False):
        losses, ious = [], []
        for f, rm, dt_real in self.items:
            uv, ok = project(self.pts[f], cam)
            sm = sim_mask(uv, ok, self.link, (self.h, self.w))
            p = uv[ok] * SCALE
            inside = (p[:, 0] >= 0) & (p[:, 0] < self.w) & (p[:, 1] >= 0) & (p[:, 1] < self.h)
            if inside.sum() < 50 or rm.sum() < 30:
                continue
            pi = np.clip(p[inside].astype(int), 0, [self.w - 1, self.h - 1])
            a = np.minimum(dt_real[pi[:, 1], pi[:, 0]], CAP).mean()
            band = cv2.dilate(sm, np.ones((31, 31), np.uint8)) > 0
            rb = (rm > 0) & band
            if rb.sum() > 0:
                dt_sim = cv2.distanceTransform((1 - sm).astype(np.uint8), cv2.DIST_L2, 3)
                b = np.minimum(dt_sim[rb], CAP).mean()
            else:
                b = CAP
            losses.append(a + b)
            if return_iou:
                inter = ((sm > 0) & (rm > 0) & band).sum()
                union = ((sm > 0) | ((rm > 0) & band)).sum()
                ious.append(inter / max(union, 1))
        if return_iou:
            return float(np.mean(losses)), float(np.mean(ious)), len(losses)
        return float(np.mean(losses)) if losses else 1e3


def with_delta(base, x):
    R = Rotation.from_rotvec(x[:3]).as_matrix() @ base["R"]
    return {**base, "R": R, "t": base["t"] + x[3:6]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", nargs="+", default=["top", "inspect"])
    ap.add_argument("--points", default=str(TWIN / "out/align/points.npz"))
    args = ap.parse_args()
    D = np.load(args.points, allow_pickle=True)
    names = list(D["link_names"])
    keep = np.isin(D["link"], [names.index(n) for n in MOVING if n in names])
    pts, link = D["pts"][:, keep].astype(np.float64), D["link"][keep]
    samples = [tuple(map(int, s)) for s in D["samples"]]
    split = json.loads((TWIN / "out/align/samples_split.json").read_text())
    n_top = split["n_top"]
    eps = sorted({e for e, _ in samples})
    held = set(eps[2::3])
    out_json = TWIN / "calib/cameras_fit.json"
    result = json.loads(out_json.read_text()) if out_json.exists() else {}
    for cam_name in args.cams:
        idx = list(range(n_top)) if cam_name == "top" else list(range(n_top, len(samples)))
        by_ep = {}
        for f in idx:
            by_ep.setdefault(samples[f][0], []).append(f)
        fit_items, val_items, frames_rgb = [], [], {}
        for e, fs in by_ep.items():
            n = len(states(e))
            bgi = sorted(set(np.linspace(0, n - 1, 25).astype(int).tolist()))
            want = sorted(set(bgi) | {samples[f][1] for f in fs})
            vf = video_frames(e, cam_name, want)
            bg = np.median(np.stack([vf[i] for i in bgi if i in vf]), 0).astype(np.uint8)
            for f in fs:
                fr = vf.get(samples[f][1])
                if fr is None:
                    continue
                rm = real_mask(fr, bg)
                dt = cv2.distanceTransform((1 - rm).astype(np.uint8), cv2.DIST_L2, 3)
                (val_items if e in held else fit_items).append((f, rm, dt))
                frames_rgb[f] = (fr, rm)
        fitP, valP = Problem(cam_name, fit_items, pts, link), Problem(cam_name, val_items, pts, link)
        optP = Problem(cam_name, fit_items[:: max(1, len(fit_items) // 80)], pts, link)
        joe = load_joe(cam_name)
        joe_fac = {**joe, **factory(cam_name)}
        print(
            f"[{cam_name}] fit frames {len(fit_items)} | held-out frames {len(val_items)} (episodes {len(held)})",
            flush=True,
        )
        x0 = np.zeros(6)
        best = None
        for start in (x0, np.r_[0.02, -0.02, 0.0, 0.01, 0.01, 0.01]):
            r = minimize(
                lambda x, prob=optP, cam=joe_fac: prob.loss(with_delta(cam, x)),
                start,
                method="Powell",
                options={"xtol": 1e-4, "ftol": 1e-4, "maxfev": 1500},
            )
            if best is None or r.fun < best.fun:
                best = r
            print(
                f"[{cam_name}]   start {np.round(start, 3).tolist()} -> loss {r.fun:.2f} ({r.nfev} evals)", flush=True
            )
        fitted = with_delta(joe_fac, best.x)
        rows = {}
        for label, cam in (
            ("joe_cal2 as-is", joe),
            ("joe_cal2 pose + factory intrinsics", joe_fac),
            ("fitted", fitted),
        ):
            rows[label] = {"fit": fitP.loss(cam, True)[:2], "held_out": valP.loss(cam, True)[:2]}
            print(
                f"[{cam_name}] {label:38s} fit loss {rows[label]['fit'][0]:6.2f} IoU {rows[label]['fit'][1]:.3f} | "
                f"held-out loss {rows[label]['held_out'][0]:6.2f} IoU {rows[label]['held_out'][1]:.3f}",
                flush=True,
            )
        dR = Rotation.from_matrix(fitted["R"] @ joe["R"].T).as_euler("xyz", degrees=True)
        print(
            f"[{cam_name}] fitted vs joe_cal2: rotation {np.round(dR, 2).tolist()} deg, "
            f"translation {np.round((fitted['t'] - joe['t']) * 1000, 1).tolist()} mm",
            flush=True,
        )
        result[cam_name] = {
            "world_to_cv": fitted["R"].tolist(),
            "position": fitted["t"].tolist(),
            "fx": fitted["fx"],
            "fy": fitted["fy"],
            "cx": fitted["cx"],
            "cy": fitted["cy"],
            "resolution": [640, 480],
            "frame": "robot base (sim world, env origin)",
            "method": "silhouette fit to recorded cal2 episodes, factory intrinsics, start joe_cal2",
            "metrics": rows,
            "n_fit_frames": len(fit_items),
            "n_held_out_frames": len(val_items),
        }
        # overlay: held-out frames, outline of joe_cal2 (red) and fitted (green), real mask tinted blue
        tiles = []
        for f, rm, _ in val_items[:: max(1, len(val_items) // 8)][:8]:
            fr, _ = frames_rgb[f]
            img = fr.copy()
            img[cv2.resize(rm, (640, 480), interpolation=cv2.INTER_NEAREST) > 0] //= 2
            img[cv2.resize(rm, (640, 480), interpolation=cv2.INTER_NEAREST) > 0] += np.array([0, 0, 110], np.uint8)
            for cam, col in ((joe, (255, 60, 60)), (fitted, (60, 255, 60))):
                uv, ok = project(pts[f], cam)
                sm = cv2.resize(sim_mask(uv, ok, link, (fitP.h, fitP.w)), (640, 480), interpolation=cv2.INTER_NEAREST)
                cs, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, col, 2)
            cv2.putText(
                img,
                f"ep {samples[f][0]} fr {samples[f][1]}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            tiles.append(cv2.resize(img, (320, 240)))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.concatenate([np.concatenate(tiles[i : i + 4], 1) for i in range(0, len(tiles), 4)], 0)
        (TWIN / "out/align").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(TWIN / f"out/align/overlay_{cam_name}.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    out_json.write_text(json.dumps(result, indent=1))
    print("FIT_OK ->", out_json, flush=True)


if __name__ == "__main__":
    main()
