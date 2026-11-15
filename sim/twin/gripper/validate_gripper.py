"""Realism gate (20 cal2 replays, sim/twin/physics/realism_gate.py) for the fin-ray fingers vs the stock-jaw pads.

    sim/twin/run.sh sim/twin/gripper/validate_gripper.py                       # before (pads_cad) + finray
    sim/twin/run.sh sim/twin/gripper/validate_gripper.py --configs finray,finray_rigid
    sim/twin/run.sh sim/twin/gripper/validate_gripper.py --analyse_only
One Isaac process per config (subprocess.run, sequential). Metrics as sim/twin/physics/validate_physics.py (held
reading on lifted episodes, brick tilt in the grip, tracking RMSE, throughput) plus, for the fins, where the brick
sits (height above the fin tip) and how far the fins are inside it at the held angle. Writes
sim/twin/out/gripper/validate_gripper.{json,md}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.gripper import model
from sim.twin.physics import validate_physics as vp

OUT = REPO / "sim/twin/out/gripper"
JOINT_FIT = REPO / "sim/twin/align/joint_fit/result.json"
# the joint fit's W9 offsets (align/joint_fit/result.json grasp_depth_candidates; this copy for boxes whose result.json
# predates them)
W9_FALLBACK = [-0.0819432805284035, 0.017274543575871923, -0.08054798740044437, -0.03023370954585218,
               -1.6803820368077729, 0.0]
W9 = json.dumps(((json.loads(JOINT_FIT.read_text()) if JOINT_FIT.exists() else {}).get("grasp_depth_candidates") or {})
                .get("W9", {}).get("joint_offsets_rad_for_sim") or W9_FALLBACK)
FIN = "gripper.finray."
CONFIGS = {
    "before_pads": ["--physics", "on", "--set", "gripper.collider=pads_cad"],  # stock jaw, CAD-fitted box pads
    "finray": ["--physics", "on"],  # params.json
    "finray_final": ["--physics", "on"],  # params.json again, after plates.rim_h, curved_2x3 mass, wrist mount
    # the rigid fins as before the bending default (joe_cal2 bias, 0.4 N m), for runs on the same joint offsets
    "finray_rigid_fins": ["--physics", "on", "--set", FIN + "bend.hinges=0", "--set", FIN + "bias_rad=null",
                          "--set", FIN + "effort_nm=0.4"],
    "finray_rigid": ["--physics", "on", "--set", FIN + "material.compliant_contact_stiffness=0"],
    "finray_effort_1.2": ["--physics", "on", "--set", FIN + "effort_nm=1.2"],
    "finray_effort_0.5": ["--physics", "on", "--set", FIN + "effort_nm=0.5"],
    "finray_k50k": ["--physics", "on", "--set", FIN + "material.compliant_contact_stiffness=50000",
                    "--set", FIN + "material.compliant_contact_damping=250"],
    "finray_k8k": ["--physics", "on", "--set", FIN + "material.compliant_contact_stiffness=8000",
                   "--set", FIN + "material.compliant_contact_damping=60"],
    "finray_k10k_effort_0.5": ["--physics", "on", "--set", FIN + "material.compliant_contact_stiffness=10000",
                               "--set", FIN + "material.compliant_contact_damping=60", "--set", FIN + "effort_nm=0.5"],
    "finray_k10k_effort_0.4": ["--physics", "on", "--set", FIN + "material.compliant_contact_stiffness=10000",
                               "--set", FIN + "material.compliant_contact_damping=60", "--set", FIN + "effort_nm=0.4"],
    "finray_mu_0.6": ["--physics", "on", "--set", FIN + "material.static_friction=0.6",
                      "--set", FIN + "material.dynamic_friction=0.5"],
    "finray_mu_1.2": ["--physics", "on", "--set", FIN + "material.static_friction=1.2",
                      "--set", FIN + "material.dynamic_friction=1.0"],
    "finray_depth_25mm": ["--physics", "on", "--set", FIN + "grasp_depth=0.025"],
    "finray_extend_5mm": ["--physics", "on", "--set", FIN + "extend=0.005"],
    "finray_extend_10mm": ["--physics", "on", "--set", FIN + "extend=0.010"],
    "finray_bias_0.075": ["--physics", "on", "--set", FIN + "bias_rad=0.075"],
    "finray_bias_0.135": ["--physics", "on", "--set", FIN + "bias_rad=0.135"],
    # LeRobot 0.7 so_follower gripper limits on a 12 V STS3215 (stall ~2.94 N m): Max_Torque_Limit 500 (50 %,
    # 1.47 N m peak), Overload_Torque 25 (0.74 N m once the overload protection trips during a sustained hold)
    "finray_lerobot_0.74": ["--physics", "on", "--set", FIN + "effort_nm=0.74"],
    "finray_lerobot_1.47": ["--physics", "on", "--set", FIN + "effort_nm=1.47"],
    "finray_lerobot_0.74_k5k": ["--physics", "on", "--set", FIN + "effort_nm=0.74",
                                "--set", FIN + "material.compliant_contact_stiffness=5000",
                                "--set", FIN + "material.compliant_contact_damping=40"],
    # bias so the rigid fin tips meet (q -0.178) at the real closed-on-nothing reading 2.8
    "finray_touch_bias_0.74": ["--physics", "on", "--set", FIN + "bias_rad=-0.242", "--set", FIN + "effort_nm=0.74"],
    "finray_touch_bias_1.47": ["--physics", "on", "--set", FIN + "bias_rad=-0.242", "--set", FIN + "effort_nm=1.47"],
    # bending fins (model.segments): hinges per fin, hinge stiffness (N m/rad), tips touching when closed, 0.74 N m
    **{f"finray_bend{h}_k{k}": ["--physics", "on", "--set", FIN + "bias_rad=-0.242", "--set", FIN + "effort_nm=0.74",
                                "--set", FIN + f"bend.hinges={h}", "--set", FIN + f"bend.stiffness_nm_per_rad={k}"]
       for h in (2, 3) for k in (0.5, 0.7, 1.0, 1.5, 2.5)},
    # the moving fin stiffer at its root than the fixed fin (real hold frames: fixed fin root ~15 deg, moving fin
    # root ~3 deg with its bend spread along the fin)
    "finray_bend3_jawroot3": ["--physics", "on", "--set", FIN + "bend.stiffness_jaw_nm_per_rad=[3.0, 0.7, 0.7]"],
    "finray_bend3_jawroot3_fix0.5": ["--physics", "on", "--set", FIN + "bend.stiffness_nm_per_rad=[0.5, 1.0, 1.0]",
                                     "--set", FIN + "bend.stiffness_jaw_nm_per_rad=[3.0, 0.6, 0.6]"],
    # arm joint limits opened by 8 deg (robot.limit_margin_deg), with everything else as params.json
    "finray_limits8": ["--physics", "on", "--set", "robot.limit_margin_deg=8"],
    # real closes put the CAD fin tips 5 (W) / 8 (W9) mm below the table: the arm presses the soft fins into it. The
    # 4-hinge fins, on W or W9 (align/joint_fit result.json grasp_depth_candidates), with a soft table contact
    # standing in for the tip deflection
    **{f"finray_n4{w}{t}": ["--physics", "on", "--set", FIN + "bend.slices=true", "--set", FIN + "bend.hinges=4",
                              "--set", FIN + "bend.stiffness_nm_per_rad=1.45", "--set", FIN + "bend.curl_gearing=0",
                              "--set", "robot.limit_margin_deg=8", *wo, *to]
       for w, wo in (("", []), ("_w9", ["--set", "robot.joint_offsets_rad=" + W9]))
       for t, to in (("", []), ("_table2k", ["--set", "table.compliant_contact_stiffness=2000",
                                              "--set", "table.compliant_contact_damping=40"]),
                     ("_table5k", ["--set", "table.compliant_contact_stiffness=5000",
                                   "--set", "table.compliant_contact_damping=60"]))},
    **{f"finray_n4_w9_{t}": ["--physics", "on", "--set", FIN + "bend.slices=true", "--set", FIN + "bend.hinges=4",
                             "--set", FIN + "bend.stiffness_nm_per_rad=1.45", "--set", FIN + "bend.curl_gearing=0",
                             "--set", "robot.limit_margin_deg=8", "--set", "robot.joint_offsets_rad=" + W9, *to]
       for t, to in (("table500", ["--set", "table.compliant_contact_stiffness=500",
                                   "--set", "table.compliant_contact_damping=20"]),
                     ("softarm", ["--set", 'robot.arm_stiffness={"shoulder_pan": 60.0, "shoulder_lift": 20.0, '
                                  '"elbow_flex": 20.0, "wrist_flex": 20.0, "wrist_roll": 60.0}']))},
    # linear servo map (bias -0.10, wrist-frame fits) + 4 sliced hinges at k 2.5 (bench: held reading 5.9 with the
    # brick 20 mm up), W9, joint limits +8; curl = distal hinges pre-curled toward the other fin (deg)
    **{f"finray_b010_n4_w9{t}": ["--physics", "on", "--set", FIN + "bias_rad=-0.10", "--set", FIN + "bend.slices=true",
                                 "--set", FIN + "bend.hinges=4", "--set", FIN + "bend.stiffness_nm_per_rad=2.5",
                                 "--set", FIN + "bend.curl_gearing=0", "--set", FIN + f"bend.rest_curl_deg={c}",
                                 "--set", "robot.limit_margin_deg=8", "--set", "robot.joint_offsets_rad=" + W9]
       for t, c in (("", "0"), ("_curl8", "[0, 0, 8, 8]"), ("_curl15", "[0, 0, 15, 15]"))},
    # hinge-count x fin-ray coupling grid: fin sliced into a 12 mm clamped root + N equal segments, hinge stiffness
    # 1.6 x the equal-compliance value (bench: held reading ~5.7); "on" = the distal half of the hinges mimic the
    # root hinge with gearing 1 (they curl toward the brick as much as the root bends away)
    **{f"finray_grid_n{n}_{c}": ["--physics", "on", "--set", FIN + "bend.slices=true",
                                 "--set", FIN + f"bend.hinges={n}", "--set", FIN + f"bend.stiffness_nm_per_rad={k}",
                                 "--set", FIN + f"bend.curl_gearing={1.0 if c == 'on' else 0.0}"]
       for n, k in ((2, 1.05), (3, 1.25), (4, 1.45), (5, 1.7)) for c in ("off", "on")},
}


def wilson(p: float, n: int, z: float = 1.96) -> list:
    """Wilson 95 % interval of a proportion p observed on n trials."""
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(c - h), 3), round(float(c + h), 3)]


def fin_metrics(out: Path, name: str) -> dict:
    """Brick height above the fin tip and fin-in-brick depth while the brick is carried (trace.npy columns:
    gripper q, command, brick xyz in the gripper frame, torque, target, tilt, brick z, tip z)."""
    path = out / f"gate_{name}" / "trace.npy"
    if not path.exists() or not name.startswith("finray"):
        return {}
    tr = np.load(path)
    tip_z = float(model.geometry()["fixed_tip"][2])
    depth, sink = [], []
    for k in range(tr.shape[1]):
        held = tr[:, k, 8] > 0.035
        if held.any():
            d = float(np.median(tr[held, k, 4])) - tip_z
            depth.append(d)
            sink.append(model.BRICK_W - model.gap(float(np.median(tr[held, k, 0])), d))
    if not depth:
        return {}
    return {"brick_above_fin_tip_mm": round(float(np.median(depth)) * 1e3, 1),
            "fins_inside_brick_mm": round(float(np.median(sink)) * 1e3, 2)}


def main():
    """Run / analyse the configs and write the comparison table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="before_pads,finray")
    ap.add_argument("--analyse_only", action="store_true")
    ap.add_argument("--episodes", type=int, default=20, help="replayed episodes (= envs; validate.py's seed-0 list)")
    args = ap.parse_args()
    out = OUT if args.episodes == 20 else OUT / f"n{args.episodes}"
    out.mkdir(parents=True, exist_ok=True)
    vp.OUT = out
    vp.CONFIGS.update(CONFIGS)
    names = args.configs.split(",")
    if not args.analyse_only:
        for name in names:
            vp.run_config(name, ["--episodes", str(args.episodes), "--num_envs", str(args.episodes)])
    rows = []
    done = sorted(p.name[5:] for p in out.glob("gate_*") if p.is_dir())
    for name in [n for n in CONFIGS if n in done] + [n for n in done if n not in CONFIGS]:
        r = vp.analyse(name)
        if r is not None:
            r.update(fin_metrics(out, name))
            r["lift_wilson95"] = wilson(r["lift"], args.episodes)
            rows.append(r)
    md = ("| config | grasp | lift >= 3 cm (Wilson 95 %) | inspect view | held reading (lifted eps) / real | brick "
          "tilt in grip (deg) | brick above fin tip (mm) / fins inside brick (mm) | RMSE pan/lift/elbow/wflex/wroll/"
          "grip (deg) | ticks/s (all envs) |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        t = r["tracking_rmse_deg"]
        md += (f"| {r['config']} | {r['grasp']:.0%} | {r['lift']:.0%} ({r['lift_wilson95'][0]:.0%}-"
               f"{r['lift_wilson95'][1]:.0%}) | {r['inspect_view']:.0%} | "
               f"{r['hold_sim_lifted_median']} / {r['hold_real_same_eps_median']} | {r.get('brick_tilt_in_grip_deg')} "
               f"| {r.get('brick_above_fin_tip_mm')} / {r.get('fins_inside_brick_mm')} | "
               + " / ".join(f"{t[j]:.2f}" for j in vp.J) + f" | {r.get('ticks_per_s_all_envs')} |\n")
    (out / "validate_gripper.json").write_text(json.dumps(rows, indent=1))
    (out / "validate_gripper.md").write_text(md)
    print(md, flush=True)


if __name__ == "__main__":
    main()
