"""Calibrate from capture.py output.

* wrist intrinsics: phone-board views (any pose) -> fx, fy, cx, cy, distortion, RMS reprojection error;
* inspect pose: frames where top AND inspect see the same board lying still (same capture tick); board pose from
  the top camera (factory intrinsics + the confirmed joe_cal2 pose) chained with the board pose seen by inspect;
* top check: a board lying on the table should come out at z ~ 0 (table plane) in the robot frame.

    python calibrate.py <capture dir> --phone-square-mm 15 [--laptop-square-mm 30]
Writes <capture dir>/cameras_measured.json (same camera format as calib/cameras_fit.json).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from boards import board, detect  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("capture", type=Path)
ap.add_argument("--phone-square-mm", type=float, required=True)
ap.add_argument("--laptop-square-mm", type=float, default=None)
ap.add_argument("--intrinsics", type=Path, default=HERE / "realsense_intrinsics.json")
ap.add_argument("--top-pose", type=Path, default=HERE.parents[1] / "residual_env/assets/joe_cal2/reference-cal2.json")
args = ap.parse_args()
cap = args.capture.expanduser()
log = json.loads((cap / "detections.json").read_text())
boards = {"phone": board("phone", args.phone_square_mm / 1000)}
if args.laptop_square_mm:
    boards["laptop"] = board("laptop", args.laptop_square_mm / 1000)
img = lambda cam, f: cv2.cvtColor(cv2.imread(str(cap / cam / f)), cv2.COLOR_BGR2RGB)
res = {}

# 1. wrist intrinsics
views = [r for r in log.get("wrist", []) if r["phone"] >= 8]
views = views[:: max(1, len(views) // 60)]
obj, imgp = [], []
for r in views:
    d = detect(img("wrist", r["file"]), boards["phone"], 8)
    if d:
        obj.append(d[0].astype(np.float32)); imgp.append(d[1].astype(np.float32))
if len(obj) >= 10:
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, imgp, (640, 480), None, None)
    res["wrist"] = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "dist": dist.ravel().tolist(),
                    "rms_px": rms, "views": len(obj), "hfov_deg": float(np.degrees(2 * np.arctan(320 / K[0, 0])))}
    print(f"wrist: {len(obj)} views, RMS {rms:.2f} px, fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} c ({K[0, 2]:.1f}, {K[1, 2]:.1f}), "
          f"dist {np.round(dist.ravel(), 3).tolist()}", flush=True)
else:
    print(f"wrist: only {len(obj)} usable phone-board views (need >= 10)", flush=True)

# 2. inspect pose via the top camera, 3. top table check
intr = json.loads(args.intrinsics.read_text())
Kof = lambda c: np.array([[intr[c]["fx"], 0, intr[c]["cx"]], [0, intr[c]["fy"], intr[c]["cy"]], [0, 0, 1]])
ref = json.loads(args.top_pose.read_text())["camera"]
R_top, c_top = np.array(ref["world_to_cv"]), np.array(ref["position"])  # x_cv = R (X - c)


def pose(cam, f, b):
    d = detect(img(cam, f), boards[b], 6)
    if not d:
        return None
    ok, rv, tv = cv2.solvePnP(d[0], d[1], Kof(cam), None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    err = np.linalg.norm(cv2.projectPoints(d[0], rv, tv, Kof(cam), None)[0].reshape(-1, 2) - d[1], axis=1).mean()
    return cv2.Rodrigues(rv)[0], tv.ravel(), err, len(d[2])


pairs = []
for rt, ri in zip(log.get("top", []), log.get("inspect", [])):
    for b in boards:
        if rt[b] >= 6 and ri[b] >= 6 and rt["file"] == ri["file"]:
            pt, pi = pose("top", rt["file"], b), pose("inspect", ri["file"], b)
            if pt and pi:
                # board -> world via top: X_w = R_top^T (R_tb X_b + t_tb) + c_top
                R_wb = R_top.T @ pt[0]; t_wb = R_top.T @ pt[1] + c_top
                # inspect: x_cv = R_ib X_b + t_ib = R_ib R_wb^T (X_w - t_wb) + t_ib -> R = R_ib R_wb^T, c = t_wb - R^T t_ib
                R_i = pi[0] @ R_wb.T; c_i = t_wb - R_i.T @ pi[1]
                pairs.append({"file": rt["file"], "board": b, "board_z": float(t_wb[2]),
                              "board_normal_z": float(R_wb[:, 2][2]), "R": R_i, "c": c_i, "err_top": pt[2], "err_inspect": pi[2]})
if pairs:
    C = np.array([p["c"] for p in pairs])
    from scipy.spatial.transform import Rotation
    Rm = Rotation.from_matrix(np.array([p["R"] for p in pairs])).mean().as_matrix()
    res["inspect"] = {"world_to_cv": Rm.tolist(), "position": np.median(C, 0).tolist(),
                      "position_spread_mm": (np.std(C, 0) * 1000).tolist(), "n_pairs": len(pairs),
                      **{k: intr["inspect"][k] for k in ("fx", "fy", "cx", "cy")}}
    res["top_check"] = {"board_z_mm_median": float(np.median([p["board_z"] for p in pairs]) * 1000),
                        "board_normal_z_median": float(np.median([abs(p["board_normal_z"]) for p in pairs]))}
    print(f"inspect: {len(pairs)} top+inspect pairs, position {np.round(np.median(C, 0), 3).tolist()} m, "
          f"spread {np.round(np.std(C, 0) * 1000, 1).tolist()} mm", flush=True)
    print(f"top check: board z {res['top_check']['board_z_mm_median']:.1f} mm (table = 0, minus screen thickness), "
          f"|normal z| {res['top_check']['board_normal_z_median']:.3f} (1 = flat)", flush=True)
else:
    print("inspect: no frames where top and inspect both see the board", flush=True)
(cap / "cameras_measured.json").write_text(json.dumps(res, indent=1, default=float))
print("CALIB_DONE ->", cap / "cameras_measured.json", flush=True)
