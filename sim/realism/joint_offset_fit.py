"""Fit the sim SO-101 joint offsets (our LeRobot degrees -> Joe's USD joint radians) from recorded data.

sim_rad = deg2rad(state_deg + delta_deg) + joe_offset_rad, delta fitted for the 5 body joints (gripper excluded), with
Joe's top camera FIXED (camera-fit.json). Terms (robust, summed):
  1. top-camera finger keypoints: real red-finger mask (window around the Joe-offset projection) centroid + tip
     (far end from the robot base pixel) vs the projected FK finger points (numpy FK of Joe's USD, so101_fk.py);
  2. grasp height: FK fingertip z at each episode's grasp_frame inside [0.8, 3.5] cm (hinge), below-table penalised hard;
  3. inspection: fingertip at inspection_frame near the inspect camera's optical axis (weak);
  + prior toward delta = 0 (Joe). 70/30 episode split; bootstrap over episodes for the confidence of delta.
Fits the old-calibration set and cal2 separately; cal2 - old should be ~ -recal_2026-11-14 (pan -20.35, lift +7.96).

    python joint_offset_fit.py jofit/   (kin.json, camera-fit.json, jofit_data.json, frames/)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from camera_check import nelder_mead, red_mask  # noqa: E402
from so101_fk import SO101FK  # noqa: E402
from table_plane import TopCamera  # noqa: E402

RECAL = np.array([20.35, -7.96, 0.18, 0.04, 0.0])
BASE_PX = np.array([336.0, 65.0])
JNAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
HUBER_PX, W_GRASP, W_INSP, W_PRIOR = 40.0, 1.0, 0.05, 1.0


class Model:
    def __init__(self, d: Path):
        self.fk = SO101FK(d / "kin.json")
        fit = json.load(open(d / "camera-fit.json"))
        self.cam = TopCamera(fit)
        self.off = np.array(fit["joint_offsets_rad"], float)
        jaw = self.fk.j["gripper"]
        self.jaw_upper = np.deg2rad(jaw["upper"])
        # subsample finger points for speed
        n_f, n_j = len(self.fk.fixed_pts), len(self.fk.jaw_pts)
        self.fk.fixed_pts = self.fk.fixed_pts[np.linspace(0, n_f - 1, 120).astype(int)]
        self.fk.jaw_pts = self.fk.jaw_pts[np.linspace(0, n_j - 1, 80).astype(int)]
        ic = self.fk.k["InspectCamera"]
        Mi = np.array(ic["world"]).T
        self.insp_pos, self.insp_fwd = Mi[:3, 3], -Mi[:3, 2] / np.linalg.norm(Mi[:3, 2])  # USD camera looks down -Z
        self.insp_half_fov = np.arctan(0.5 * ic["hap"] / ic["focal"])

    def q(self, state, delta):
        s = np.asarray(state, float)
        q = np.zeros(6)
        q[:5] = np.deg2rad(s[:5] + delta) + self.off[:5]
        q[5] = np.clip(s[5] / 100.0, 0, 1) * self.jaw_upper
        return q

    def keypoints(self, state, delta):
        P, tip = self.fk.points(self.q(state, delta))
        uv = self.cam.project(P)
        d = np.linalg.norm(uv - BASE_PX, axis=1)
        far = uv[d >= np.percentile(d, 95)].mean(0)
        return uv.mean(0), far, uv, tip


def real_keypoints(img, uv_prior):
    """Real red-finger mask restricted to a window around the prior projection; centroid + far end from the base."""
    m = red_mask(img)
    x0, y0 = np.maximum(uv_prior.min(0) - 90, 0).astype(int)
    x1, y1 = np.minimum(uv_prior.max(0) + 90, [639, 479]).astype(int)
    win = np.zeros_like(m)
    win[y0:y1 + 1, x0:x1 + 1] = True
    m &= win
    ys, xs = np.nonzero(m)
    if len(xs) < 150:
        return None
    P = np.c_[xs, ys].astype(float)
    d = np.linalg.norm(P - BASE_PX, axis=1)
    return P.mean(0), P[d >= np.percentile(d, 95)].mean(0), len(xs)


def build(d: Path, model: Model, set_name: str):
    data = json.load(open(d / "jofit_data.json"))
    zero = np.zeros(5)
    frames = []
    for f in data["frames"]:
        if f["set"] != set_name:
            continue
        uv_prior_delta = -RECAL if set_name == "cal2" else zero  # prior window: Joe offsets on the old frame
        c, far, uv, _ = model.keypoints(f["state"], uv_prior_delta)
        if not ((uv[:, 0] > 0) & (uv[:, 0] < 640) & (uv[:, 1] > 0) & (uv[:, 1] < 480)).mean() > 0.8:
            continue
        img = np.asarray(Image.open(d / "frames" / f"{f['tag']}.jpg").convert("RGB"))
        rk = real_keypoints(img, uv)
        if rk is None:
            continue
        frames.append({"ep": f["episode"], "state": f["state"], "real_c": rk[0], "real_far": rk[1], "tag": f["tag"]})
    eps = [e for e in data["episodes"] if e["set"] == set_name]
    return frames, eps


def terms(model: Model, delta, frames, eps):
    kp = []
    for f in frames:
        c, far, _, _ = model.keypoints(f["state"], delta)
        kp.append(0.5 * (np.linalg.norm(c - f["real_c"]) + np.linalg.norm(far - f["real_far"])))
    kp = np.asarray(kp)
    gz, ia = [], []
    for e in eps:
        _, tip = model.fk.points(model.q(e["grasp"], delta))
        gz.append(tip[2])
        _, tip_i = model.fk.points(model.q(e["inspect"], delta))
        v = tip_i - model.insp_pos
        ia.append(np.degrees(np.arccos(np.clip(v @ model.insp_fwd / np.linalg.norm(v), -1, 1))))
    return kp, np.asarray(gz), np.asarray(ia)


def loss(model, delta, frames, eps):
    kp, gz, ia = terms(model, delta, frames, eps)
    l_kp = np.mean(np.where(kp < HUBER_PX, kp ** 2, 2 * HUBER_PX * kp - HUBER_PX ** 2)) / 100.0  # Huber, (10 px)^2 units
    cm = gz * 100
    l_g = np.mean(np.maximum(0.8 - cm, 0) ** 2 + np.maximum(cm - 3.5, 0) ** 2 + 25 * np.maximum(-cm, 0) ** 2)
    l_i = np.mean(np.maximum(ia - np.degrees(model.insp_half_fov) * 0.6, 0) ** 2) / 25.0
    l_p = np.sum((np.asarray(delta) / 5.0) ** 2)
    return l_kp + W_GRASP * l_g + W_INSP * l_i + W_PRIOR * l_p


def summarize(model, delta, frames, eps):
    kp, gz, ia = terms(model, delta, frames, eps)
    return {"kp_median_px": round(float(np.median(kp)), 1), "kp_p90_px": round(float(np.percentile(kp, 90)), 1),
            "grasp_z_median_cm": round(float(np.median(gz) * 100), 2),
            "grasp_z_frac_in_band": round(float(np.mean((gz > 0.008) & (gz < 0.035))), 3),
            "grasp_z_frac_below_table": round(float(np.mean(gz < 0)), 3),
            "inspect_angle_median_deg": round(float(np.median(ia)), 1),
            "inspect_frac_in_fov": round(float(np.mean(ia < np.degrees(model.insp_half_fov))), 3)}


def fit(model, frames, eps, x0=None):
    x0 = np.zeros(5) if x0 is None else x0
    xb, _ = nelder_mead(lambda x: loss(model, x, frames, eps), x0, [3.0] * 5, iters=800, tol=1e-5)
    return xb


def split(items, key, train_eps):
    return [i for i in items if i[key] in train_eps], [i for i in items if i[key] not in train_eps]


def main(d: Path):
    model = Model(d)
    report = {}
    for set_name, base in (("old", np.zeros(5)), ("cal2", -RECAL)):
        frames, eps = build(d, model, set_name)
        all_eps = sorted({e["episode"] for e in eps})
        rng = np.random.default_rng(0)
        tr_eps = set(rng.choice(all_eps, size=int(0.7 * len(all_eps)), replace=False).tolist())
        f_tr, f_te = split(frames, "ep", tr_eps)
        e_tr, e_te = split(eps, "episode", tr_eps)
        # cal2 baseline = Joe's offsets applied after removing the recal (what the env did for cal2 data)
        delta = fit(model, f_tr, e_tr, base.copy())
        boot = []
        for b in range(8):
            r = np.random.default_rng(100 + b)
            pick = set(r.choice(sorted(tr_eps), size=len(tr_eps), replace=True).tolist())
            fb = [f for f in f_tr if f["ep"] in pick]
            ebb = [e for e in e_tr if e["episode"] in pick]
            boot.append(fit(model, fb, ebb, delta.copy()))
        boot = np.asarray(boot)
        report[set_name] = {
            "n_frames": len(frames), "n_episodes": len(eps),
            "delta_deg": dict(zip(JNAMES, np.round(delta, 2).tolist())),
            "delta_minus_baseline_deg": dict(zip(JNAMES, np.round(delta - base, 2).tolist())),
            "delta_boot_std_deg": dict(zip(JNAMES, np.round(boot.std(0), 2).tolist())),
            "train_before": summarize(model, base, f_tr, e_tr), "train_after": summarize(model, delta, f_tr, e_tr),
            "heldout_before": summarize(model, base, f_te, e_te), "heldout_after": summarize(model, delta, f_te, e_te),
            "big_pull_joints": [j for j, v in zip(JNAMES, delta - base) if abs(v) > 10],
            "offsets_rad_for_sim": (model.off[:5] + np.deg2rad(delta)).round(5).tolist() + [0.0],
        }
        report[set_name]["_delta"] = delta.tolist()
        print(set_name, json.dumps({k: v for k, v in report[set_name].items() if not k.startswith("_")}, indent=1), flush=True)
    d_old, d_cal2 = np.array(report["old"]["_delta"]), np.array(report["cal2"]["_delta"])
    report["consistency"] = {"cal2_minus_old_deg": dict(zip(JNAMES, np.round(d_cal2 - d_old, 2).tolist())),
                             "expected_minus_recal_deg": dict(zip(JNAMES, (-RECAL).tolist()))}
    print("consistency", json.dumps(report["consistency"]))
    for s in ("old", "cal2"):
        report[s].pop("_delta")
    (HERE / "joint_offsets_fit.json").write_text(json.dumps(report, indent=1))
    overlay(d, model, report)


def overlay(d, model, report):
    """Before (red) / after (green) projected FK finger points on real frames."""
    data = json.load(open(d / "jofit_data.json"))
    tiles = []
    for f in [f for f in data["frames"] if f["set"] == "old"][::37][:8]:
        img = np.asarray(Image.open(d / "frames" / f"{f['tag']}.jpg").convert("RGB")).copy()
        for delta, col in ((np.zeros(5), (255, 0, 0)), (np.array(list(report["old"]["delta_deg"].values())), (0, 255, 0))):
            _, _, uv, _ = model.keypoints(f["state"], delta)
            for u, v in uv.astype(int):
                if 0 <= u < 640 and 0 <= v < 480:
                    img[max(v - 1, 0):v + 2, max(u - 1, 0):u + 2] = col
        tiles.append(Image.fromarray(img).resize((320, 240)))
    s = Image.new("RGB", (320 * 4, 240 * 2))
    for i, t in enumerate(tiles):
        s.paste(t, ((i % 4) * 320, (i // 4) * 240))
    s.save(HERE / "joint_offset_overlay.jpg")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
