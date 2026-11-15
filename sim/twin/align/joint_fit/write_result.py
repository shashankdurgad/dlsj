"""Collect the current best joint-fit estimates into sim/twin/align/joint_fit/result.json (machine readable).

Reads out/align/joint_fit/{fit,stage2,stage3}.json (whatever exists): offsets of fit C (or the latest fit available)
with bootstrap sd and the spread across fit variants, sag coefficients (stage3), wrist-mount and inspect-camera poses.

    python sim/twin/align/joint_fit/write_result.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402

OUT = M.TWIN / "out/align/joint_fit"
JN = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def load(name):
    """JSON from the output folder, or {} if it is not there yet."""
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    fit, s2, s3, s4 = load("fit.json"), load("stage2.json"), load("stage3.json"), load("stage4.json")
    th_all = dict(fit.get("theta", {}))
    mount = "joe_cal2 wrist mount, refitted (fit C)"
    if "W" in s4.get("theta", {}):
        from stage4 import MOUNT, use_wrist_mount
        use_wrist_mount()
        th_all["W"] = s4["theta"]["W"]
        mount = f"fixed to {MOUNT.relative_to(M.REPO)}"
        s3 = {"theta": {**s3.get("theta", {}), **s4["theta"]}, "heldout": {**s3.get("heldout", {}),
                                                                           **s4.get("heldout", {})}}
    best = next(k for k in ("W", "C", "C0", "B3", "B2", "B1") if k in th_all)
    th = M.zero_theta()
    th.update({k: np.array(v, float) for k, v in th_all[best].items()})
    tot = M.JOE_OFFSETS + th["joints"]
    variants = {k: (M.JOE_OFFSETS + np.array(v["joints"])).tolist() for k, v in th_all.items() if "joints" in v}
    spread = np.ptp(np.array([variants[k] for k in variants if k in ("B3", "C0", "C", "W")]), 0) \
        if len(variants) > 1 else np.full(5, np.nan)
    boot = s2.get("bootstrap", {})
    sd = np.array(boot["x"])[:, :5].std(0) if len(boot.get("x", [])) > 1 else np.full(5, np.nan)
    cams = M.cams_of(th)
    w, ins = cams["wrist"], cams["inspect"]
    res = {
        "_source": f"sim/twin/align/joint_fit (fit '{best}'; see README.md). Units: rad, m. "
                   "Silhouettes of the arm links "
                   "and gripper body only (the real TPU fin-ray fingers are not modelled: URDF jaws excluded).",
        "fit": best,
        "joint_offsets_rad_for_sim": tot.tolist() + [0.0],
        "joint_offsets_deg": dict(zip(JN, np.round(np.degrees(tot), 3).tolist(), strict=True)),
        "delta_vs_joe_cal2_deg": dict(zip(JN, np.round(np.degrees(th["joints"]), 3).tolist(), strict=True)),
        "bootstrap_sd_deg": dict(zip(JN, np.round(np.degrees(sd), 3).tolist(), strict=True)),
        "n_bootstrap": len(boot.get("x", [])),
        "spread_across_fits_deg (B3 top+inspect, C0, C, W)": dict(
            zip(JN, np.round(np.degrees(spread), 3).tolist(), strict=True)),
        "joe_cal2_offsets_rad": M.JOE_OFFSETS.tolist() + [0.0],
        "all_fits_offsets_rad": variants,
        "wrist_camera_mount": {"source": mount, "frame": "gripper_link", "local_to_cv": w["R"].tolist(),
                               "position_local": w["t"].tolist(),
                               "fx": w["fx"], "fy": w["fy"], "cx": w["cx"], "cy": w["cy"], "dist": list(w["dist"]),
                               "note": "physical camera rotation; use with the measured intrinsics + distortion"},
        "inspect_camera": {"world_to_cv": ins["R"].tolist(), "position": ins["t"].tolist(),
                           "fx": ins["fx"], "fy": ins["fy"], "cx": ins["cx"], "cy": ins["cy"],
                           "note": "fitted on the recorded episodes (2026-11-14), start = ChArUco measurement"},
    }
    cands = {}
    for k, d in s3.get("theta", {}).items():
        if k not in ("D9", "S9", "E", "W9", "WS9", "WE"):
            continue
        t = M.zero_theta()
        t.update({kk: np.array(v, float) for kk, v in d.items()})
        off = M.JOE_OFFSETS + t["joints"]
        off[[0, 4]] += t["jpr"]
        h = s3.get("heldout", {}).get(k, {})
        cands[k] = {"joint_offsets_rad_for_sim": off.tolist() + [0.0],
                    "joint_offsets_deg": dict(zip(JN, np.round(np.degrees(off), 3).tolist(), strict=True)),
                    "sag_rad_per_Nm_lift_elbow_wflex": t["sag"].tolist(),
                    "fingertip_dz_vs_joe_cal2_mm_heldout_mean": h.get("grasp", {}).get("iou"),
                    "heldout_loss": {c: h[c]["loss"] for c in ("top", "inspect", "table") if c in h},
                    "wrist_camera_local_to_cv": M.cams_of(t)["wrist"]["R"].tolist(),
                    "wrist_camera_position_local": M.cams_of(t)["wrist"]["t"].tolist()}
    if cands:
        hc = s3.get("heldout", {}).get("W", s3.get("heldout", {}).get("C", {}))
        res["grasp_depth_candidates"] = {
            "_note": "W9 / WS9 (D9 / S9): fitted with a grasp-depth term (fingertip 9 mm below joe_cal2 at every "
                     "gripper-close frame) plus the cameras; WE (E): offsets + sag from the cameras alone. "
                     "sag: q[lift, elbow, wrist_flex] "
                     "+= sag * static gravity torque (fk.SO101.gravity_torque, URDF masses + 20 g camera on "
                     "gripper_link). Compare heldout_loss with C (cameras only).",
            "best_heldout_loss": {c: hc[c]["loss"] for c in ("top", "inspect", "table") if c in hc},
            "best_fingertip_dz_mm": hc.get("grasp", {}).get("iou"), **cands}
    tip = load("tip_height.json").get("dz_mm_per_deg")
    s2p = s2.get("profiles")
    if tip:
        an = {"_note": "single-parameter changes on top of the fit that put the fingertip GRASP_DROP lower at the "
                       "gripper-close frames (median over 456 episodes, linearised); camera cost from the 1-D "
                       "profiles around C (stage2.json, loss at +2 deg vs 0, px)",
              "fingertip_dz_mm_per_deg": tip}
        drop = 1e3 * M.GRASP_DROP
        for k, j in (("lift", 1), ("elbow", 2), ("wflex", 3)):
            d = -drop / tip[k]
            off = tot.copy()
            off[j] += np.radians(d)
            row = {"delta_deg": round(float(d), 2), "joint_offsets_rad_for_sim": off.tolist() + [0.0]}
            if s2p:
                L = np.array(s2p["loss"][k])
                g = s2p["grid_deg"]
                row["camera_loss_change_at_+2deg"] = dict(zip(s2p["cams"], np.round(L[g.index(2.0)] - L[g.index(0.0)],
                                                                                 2).tolist(), strict=True))
            an[f"{k}_only"] = row
        for k, j in (("sag_lift", 0), ("sag_elbow", 1)):
            an[f"{k}_only"] = {"sag_deg_per_Nm": round(float(-drop / tip[k]), 2),
                               "sag_rad_per_Nm_lift_elbow_wflex": [float(np.radians(-drop / tip[k])) if i == j else 0.0
                                                                    for i in range(3)],
                               "joint_offsets_rad_for_sim": tot.tolist() + [0.0]}
        res["grasp_depth_analytic"] = an
    if "heldout" in fit:
        ho = {**fit["heldout"], **s4.get("heldout", {})}
        keep = ("joe_cal2 (current twin)", "C", "joe_cal2 offsets + wrist_mount", best)
        res["heldout_per_camera"] = {k: ho[k] for k in keep if k in ho}
    (HERE / "result.json").write_text(json.dumps(res, indent=1))
    print("RESULT_OK", best, np.round(np.degrees(tot), 2).tolist(), "sd", np.round(np.degrees(sd), 2).tolist())


if __name__ == "__main__":
    main()
