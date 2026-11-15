"""Rung 0: where was the picked brick on the table, measured by the top camera, vs where FK puts the grasp point?

    python sim/twin/grasp_gate/rung0.py [--pool test|keep] [--n 0] [--sheets 16]     (plain python, PyAV)
    -> sim/twin/out/grasp_gate/rung0/{rung0.json, sheet_*.jpg, rung0.md}

Per demo: the brick the arm picks is the image region that changes between the first top frame (arm at home, brick
on the table) and the frame of the operator's inspection mark (brick in the gripper, away from the table), nearest
to the pixel of the FK grasp point at the gripper-close frame. A box of each Duplo kind (2x2, 2x3 curved, 2x4; body
+ studs, standing on the table) is fitted to that region (silhouette IoU over x, y, yaw) with the top camera model.
Reported: brick xy / yaw / kind, and its offset from the FK grasp point of the close frame in the gripper's frame
(x = jaw axis, y = across the fingers). The top camera was calibrated through the arm's FK (joe_cal2), so the offset
is the local inconsistency the replay sees, not an absolute error. Two camera models are evaluated: "joe" (joe_cal2
pose, its focal 583 px, centred) and "fit" (calib/cameras_fit.json: silhouette-fitted pose, factory intrinsics).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.align import frames as F
from sim.twin.calib.profile import twin_profile
from sim.twin.grasp_gate import common as C
from sim.twin.rl.kin import Kin

TABLE_Z = -0.002
FIN_TIP = (-0.00821, 0.00097, -0.10522)  # fixed fin tip in gripper_link (sim/twin/gripper/assets/finray_so101.json)
BODY_H, STUD_H = 0.0192, 0.0044
LONG = {"2x2": 0.0317, "2x3": 0.0477, "2x4": 0.0635}
W, HS = 0.0317, BODY_H + STUD_H  # brick width; body + studs
# resting poses: (footprint long side, footprint short side, height). up = studs up (or down), side = lying on a long
# side face (studs horizontal), end = standing on a 2-stud end face
POSES = {}
for _k, _l in LONG.items():
    POSES[f"{_k}/up"] = (_l, W, HS)
    POSES[f"{_k}/side"] = (max(_l, HS), min(_l, HS), W) if _k != "2x2" else (W, HS, W)
    if _k != "2x2":
        POSES[f"{_k}/end"] = (W, HS, _l)


def cameras() -> dict:
    """The two top-camera models: R (rows = OpenCV axes in the world), position, fx, fy, cx, cy."""
    ref = json.loads((REPO / "sim/residual_env/assets/joe_cal2/reference-cal2.json").read_text())["camera"]
    fit = json.loads((REPO / "sim/twin/calib/cameras_fit.json").read_text())["top"]
    f = float(ref["focal_px"])
    return {"joe": dict(R=np.array(ref["world_to_cv"]), t=np.array(ref["position"]), fx=f, fy=f, cx=320.0, cy=240.0),
            "fit": dict(R=np.array(fit["world_to_cv"]), t=np.array(fit["position"]),
                        **{k: float(fit[k]) for k in ("fx", "fy", "cx", "cy")})}


def project(cam: dict, p: np.ndarray) -> np.ndarray:
    pc = (np.atleast_2d(p) - cam["t"]) @ cam["R"].T
    return np.stack([cam["cx"] + cam["fx"] * pc[:, 0] / pc[:, 2], cam["cy"] + cam["fy"] * pc[:, 1] / pc[:, 2]], 1)


def backproject(cam: dict, uv, z: float) -> np.ndarray:
    """World point at height z on the ray of pixel uv."""
    d = cam["R"].T @ np.array([(uv[0] - cam["cx"]) / cam["fx"], (uv[1] - cam["cy"]) / cam["fy"], 1.0])
    s = (z - cam["t"][2]) / d[2]
    return cam["t"] + s * d


def box_corners(x, y, yaw, size) -> np.ndarray:
    hx, hy = size[0] / 2, size[1] / 2
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for z in (TABLE_Z, TABLE_Z + size[2]):
        for a, b in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            out.append((x + c * a - s * b, y + s * a + c * b, z))
    return np.array(out)


def silhouette(cam, x, y, yaw, size, shape, org) -> np.ndarray:
    uv = project(cam, box_corners(x, y, yaw, size)) - org
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv * 8).astype(np.int32)), 1, shift=3)
    return m


def fit_box(cam, mask, org) -> dict:
    """Best (kind / resting pose, x, y, yaw) by silhouette IoU with the mask (crop with pixel origin org). "up" is
    kept unless another resting pose is better by 0.03 IoU. Also returns the best IoU of every pose."""
    ys, xs = np.nonzero(mask)
    area = float(mask.sum())
    fits = {}
    for name, size in POSES.items():
        cen = backproject(cam, (xs.mean() + org[0], ys.mean() + org[1]), TABLE_Z + size[2] / 2)
        period = math.pi / 2 if abs(size[0] - size[1]) < 1e-4 else math.pi

        def cost(v, size=size):
            s = silhouette(cam, v[0], v[1], v[2], size, mask.shape, org)
            inter = float((s & mask).sum())
            return 1.0 - inter / max(area + float(s.sum()) - inter, 1.0)

        starts = sorted(((cost((cen[0], cen[1], a)), a) for a in np.arange(0, period, math.radians(15))))[:2]
        for _, a in starts:
            r = minimize(cost, (cen[0], cen[1], a), method="Nelder-Mead",
                         options={"initial_simplex": [(cen[0], cen[1], a), (cen[0] + 0.004, cen[1], a),
                                                      (cen[0], cen[1] + 0.004, a), (cen[0], cen[1], a + 0.15)],
                                  "xatol": 2e-4, "fatol": 1e-4, "maxiter": 160})
            if name not in fits or r.fun < fits[name]["cost"]:
                fits[name] = {"xy": [float(r.x[0]), float(r.x[1])], "yaw_long": float(r.x[2] % math.pi),
                              "cost": float(r.fun)}
    name = min(fits, key=lambda k: fits[k]["cost"] + (0.0 if k.endswith("/up") else 0.03))
    best = dict(fits[name])
    best.update(kind=name.split("/")[0], pose=name.split("/")[1], height=POSES[name][2], iou=round(1.0 - best.pop("cost"), 3),
                iou_all={k: round(1.0 - v["cost"], 3) for k, v in fits.items()})
    return best


def brick_mask(i0, i1, uv_tip, radius=85):
    """Changed region between the two frames nearest to the grasp pixel. Returns (mask crop, crop origin) or None."""
    x0, y0 = int(max(uv_tip[0] - radius, 0)), int(max(uv_tip[1] - radius, 0))
    x1, y1 = int(min(uv_tip[0] + radius, i0.shape[1])), int(min(uv_tip[1] + radius, i0.shape[0]))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    a = cv2.GaussianBlur(i0[y0:y1, x0:x1], (5, 5), 0).astype(np.int16)
    b = cv2.GaussianBlur(i1[y0:y1, x0:x1], (5, 5), 0).astype(np.int16)
    d = (np.abs(a - b).max(-1) > 28).astype(np.uint8)
    d = cv2.morphologyEx(cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE,
                         np.ones((5, 5), np.uint8))
    n, lab, stats, cen = cv2.connectedComponentsWithStats(d, 8)
    best, bd = None, 1e9
    for k in range(1, n):
        if not 150 <= stats[k, cv2.CC_STAT_AREA] <= 6000:
            continue
        dist = float(np.hypot(cen[k, 0] + x0 - uv_tip[0], cen[k, 1] + y0 - uv_tip[1]))
        if dist < bd:
            best, bd = k, dist
    if best is None or bd > 60:
        return None
    return (lab == best).astype(np.uint8), np.array([x0, y0], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="test")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--sheets", type=int, default=16)
    ap.add_argument("--shard", default="", help="'i,n': demos i::n -> rung0_shard<i>.json (run n of them, then --merge)")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    out = C.OUT / "rung0"
    out.mkdir(parents=True, exist_ok=True)
    demos = C.load_demos()
    eps = demos[args.pool][: args.n or None]
    if args.shard:
        i, n = (int(v) for v in args.shard.split(","))
        eps = eps[i::n]
    cams = cameras()
    kin = Kin(twin_profile().joint_offsets, "cpu")
    kin_fin = Kin(twin_profile().joint_offsets, "cpu", tip_local=FIN_TIP)
    res, tiles = {}, []
    if args.merge:
        for f in sorted(out.glob("rung0_shard*.json")):
            res.update(json.loads(f.read_text()))
        eps = []
    for ep in eps:
        d = demos["eps"][ep]
        cf, ins = int(d["cf"]), int(d["inspection_frame"])
        p, R = kin.fk(torch.tensor(d["state"][cf]))
        p, R = p.numpy().astype(float), R.numpy().astype(float)
        fr = F.video_frames(ep, "top", [0, cf, ins])
        if len(fr) < 3:
            print(ep, "frames missing", flush=True)
            continue
        zf = kin_fin.fk(torch.tensor(d["state"][max(cf - 10, 0):cf + 25]))[0][:, 2].numpy() - TABLE_Z
        row = {"cf": cf, "fin_tip_fk_mm": round(float(zf.min()) * 1e3, 1), "tip_fk": np.round(p, 4).tolist(), "yaw_close": round(math.atan2(R[1, 0], R[0, 0]), 4)}
        for name, cam in cams.items():
            uv = project(cam, [p[0], p[1], TABLE_Z + BODY_H])[0]
            bm = brick_mask(fr[0], fr[ins], uv)
            if bm is None:
                continue
            fit = fit_box(cam, *bm)
            rel = R.T @ (np.array([*fit["xy"], p[2]]) - p)
            fit.update(off_grip_mm=np.round(rel[:2] * 1e3, 1).tolist(),
                       off_world_mm=np.round((np.array(fit["xy"]) - p[:2]) * 1e3, 1).tolist(),
                       mask_px=int(bm[0].sum()))
            row[name] = fit
            if name == "joe" and len(tiles) < args.sheets:
                x0, y0 = bm[1].astype(int)
                crop = [fr[k][y0:y0 + 170, x0:x0 + 170].copy() for k in (0, cf)]
                for c in crop:
                    cnt, _ = cv2.findContours(bm[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(c, cnt, -1, (255, 255, 0), 1)
                    hull = cv2.convexHull(np.round(project(cam, box_corners(*fit["xy"], fit["yaw_long"],
                                                                            POSES[fit["kind"] + "/" + fit["pose"]])) - bm[1]).astype(np.int32))
                    cv2.polylines(c, [hull], True, (0, 255, 0), 1)
                    cv2.drawMarker(c, tuple(np.round(uv - bm[1]).astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 9, 1)
                t = cv2.resize(np.concatenate(crop, 1), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                t = cv2.copyMakeBorder(t, 0, 340 - t.shape[0], 0, 680 - t.shape[1], cv2.BORDER_CONSTANT)
                cv2.putText(t, f"ep {ep} {fit['kind']}/{fit['pose']} iou {fit['iou']} off {fit['off_grip_mm']}", (4, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                tiles.append(t)
        if "joe" in row:
            row.update({k: row["joe"][k] for k in ("kind", "pose", "height", "xy", "yaw_long", "iou")})
            res[str(ep)] = row
        print(ep, json.dumps({k: row.get(k) for k in ("kind", "iou")}), row.get("joe", {}).get("off_grip_mm"),
              row.get("fit", {}).get("off_grip_mm"), flush=True)
    if args.shard:
        C.dump(out / f"rung0_shard{args.shard.split(',')[0]}.json", res)
    for i in range(0, len(tiles), 8):
        sheet = np.concatenate([np.concatenate(tiles[i:i + 8][j:j + 2] + [np.zeros_like(tiles[0])] *
                                               (2 - len(tiles[i:i + 8][j:j + 2])), 1) for j in range(0, 8, 2)
                                if tiles[i:i + 8][j:j + 2]], 0)
        cv2.imwrite(str(out / f"sheet_{args.shard.split(',')[0] or 'x'}_{i // 8}.jpg"), sheet[..., ::-1])
    if args.shard:
        return
    stats = {"n_found": len(res)}
    md = [f"Rung 0 (top camera vs FK grasp point at the close frame), {len(res)} demos with a brick region\n",
          "| camera | n | jaw-axis x mm (p10 / median / p90) | across-finger y mm (p10 / median / p90) | "
          "\\|xy\\| mm median / p90 | kinds |", "|---|---|---|---|---|---|"]
    for name in cams:
        o = np.array([r[name]["off_grip_mm"] for r in res.values() if name in r and r[name]["iou"] >= 0.6])
        if not len(o):
            continue
        kinds = [r[name]["kind"] for r in res.values() if name in r and r[name]["iou"] >= 0.6]
        q = np.percentile(o, [10, 50, 90], 0)
        nrm = np.linalg.norm(o, axis=1)
        stats[name] = {"n": len(o), "x_p10_50_90": q[:, 0].round(1).tolist(), "y_p10_50_90": q[:, 1].round(1).tolist(),
                       "norm_median": round(float(np.median(nrm)), 1), "norm_p90": round(float(np.percentile(nrm, 90)), 1),
                       "kinds": {k: kinds.count(k) for k in LONG}}
        md.append(f"| {name} | {len(o)} | {q[0, 0]:.1f} / {q[1, 0]:.1f} / {q[2, 0]:.1f} | {q[0, 1]:.1f} / {q[1, 1]:.1f} "
                  f"/ {q[2, 1]:.1f} | {np.median(nrm):.1f} / {np.percentile(nrm, 90):.1f} | {stats[name]['kinds']} |")
    both = [(r["joe"]["xy"], r["fit"]["xy"]) for r in res.values() if "fit" in r and "joe" in r]
    if both:
        dd = np.linalg.norm(np.array([a for a, _ in both]) - np.array([b for _, b in both]), axis=1) * 1e3
        stats["joe_vs_fit_mm"] = {"median": round(float(np.median(dd)), 1), "p90": round(float(np.percentile(dd, 90)), 1)}
        md.append(f"\nThe two camera models put the same brick {np.median(dd):.1f} mm apart (median; p90 "
                  f"{np.percentile(dd, 90):.1f} mm).")
    good = [r for r in res.values() if r["iou"] >= 0.6]
    md.append("\n| top-camera resting pose | n | FK fin-tip height above the table at close, mm (p10 / median / p90) |\n|---|---|---|")
    stats["pose_vs_fk_height"] = {}
    for pose in ("up", "side", "end"):
        z = np.array([r["fin_tip_fk_mm"] for r in good if r["pose"] == pose])
        if len(z):
            q = np.percentile(z, [10, 50, 90]).round(1).tolist()
            stats["pose_vs_fk_height"][pose] = {"n": len(z), "p10_50_90": q}
            md.append(f"| {pose} | {len(z)} | {q[0]} / {q[1]} / {q[2]} |")
    C.dump(out / "rung0.json", {"stats": stats, "episodes": res})
    (out / "rung0.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
