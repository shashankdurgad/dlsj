"""Realism gate with vs without the twin physics layer, on the same 20 cal2 episodes, plus the drop test.

    sim/twin/run.sh sim/twin/physics/validate_physics.py                       # baseline + physics + drop test
    sim/twin/run.sh sim/twin/physics/validate_physics.py --configs baseline,physics,bias_old --no_drop
    sim/twin/run.sh sim/twin/physics/validate_physics.py --analyse_only        # re-read the logs / traces

Each config is one Isaac process (realism_gate.py: residual env validate.py in the twin, plates parked out of reach).
Extra metrics from the saved traces (validate.py's own `gripper_hold_median_sim` mixes in episodes where the jaw
closed on nothing): the sim gripper reading while the brick is actually carried (episodes lifted >= 3 cm, window
close+10 .. inspection), vs the real reading in the same windows. Writes sim/twin/out/physics/validate_physics.json,
.md and replay_*.png.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = REPO / "sim/twin/out/physics"
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# name -> realism_gate args
CONFIGS = {
    "baseline": ["--physics", "off"],  # residual env as left (app.finger_pads on the twin USD, joe_cal2 re-solved bias)
    "physics": ["--physics", "on"],  # params.json
    "baseline_random_plates": ["--physics", "off", "--park_plates", "0"],  # validate.py's own protocol
    "physics_random_plates": ["--physics", "on", "--park_plates", "0"],
    # ablations
    "hull": ["--physics", "on", "--set", "gripper.collider=hull"],
    "bias_resolved": ["--physics", "on", "--set", "gripper.bias_rad=-0.0263"],
    "effort_2.9": ["--physics", "on", "--set", "gripper.effort_nm=2.9"],
    "effort_0.6": ["--physics", "on", "--set", "gripper.effort_nm=0.6"],
    "jaw_mu_0.9": ["--physics", "on", "--set", "gripper.material.static_friction=0.9",
                   "--set", "gripper.material.dynamic_friction=0.8"],
    "convex_decomposition": ["--physics", "on", "--set", "gripper.collider=convex_decomposition"],
    "sdf": ["--physics", "on", "--set", "gripper.collider=sdf"],
    "old_bricks": ["--physics", "on", "--set", 'bricks.mass_kg={"2x2":0.0093,"2x4":0.018,"curved_2x2":0.0075}',
                   "--set", 'bricks.material={"static_friction":0.6,"dynamic_friction":0.5,"restitution":0.0}'],
    "physics_240hz": ["--physics", "on", "--physics_hz", "240"],
    "table_-5mm": ["--physics", "on", "--table_dz", "-0.005"],
    "table_-10mm": ["--physics", "on", "--table_dz", "-0.010"],
    "sdf_table_-5mm": ["--physics", "on", "--set", "gripper.collider=sdf", "--table_dz", "-0.005"],
    "sdf_table_-10mm": ["--physics", "on", "--set", "gripper.collider=sdf", "--table_dz", "-0.010"],
    "baseline_table_-10mm": ["--physics", "off", "--table_dz", "-0.010"],
    "rigid_joe_bias": ["--physics", "on", "--set", "bricks.max_depenetration_velocity=1000.0",
                       "--set", "robot.solver_position_iterations=64", "--set", "bricks.solver_position_iterations=64"],
    "rigid_bias_fit": ["--physics", "on", "--set", "bricks.max_depenetration_velocity=1000.0",
                       "--set", "robot.solver_position_iterations=64", "--set", "bricks.solver_position_iterations=64",
                       "--set", "gripper.bias_rad=0.072"],
    "rigid_eff2.9": ["--physics", "on", "--set", "bricks.max_depenetration_velocity=1000.0",
                     "--set", "robot.solver_position_iterations=64", "--set", "bricks.solver_position_iterations=64",
                     "--set", "gripper.effort_nm=2.9"],
    "rigid_iter32": ["--physics", "on", "--set", "bricks.max_depenetration_velocity=1000.0",
                     "--set", "robot.solver_position_iterations=32", "--set", "bricks.solver_position_iterations=32"],
    "rigid_sdf": ["--physics", "on", "--set", "bricks.max_depenetration_velocity=1000.0",
                  "--set", "robot.solver_position_iterations=64", "--set", "bricks.solver_position_iterations=64",
                  "--set", "gripper.collider=sdf"],
    "rigid_table_+5mm": ["--physics", "on", "--set", "bricks.max_depenetration_velocity=1000.0", "--set", "robot.solver_position_iterations=64", "--set", "bricks.solver_position_iterations=64", "--table_dz", "0.005"],
    "rigid_table_+8mm": ["--physics", "on", "--set", "bricks.max_depenetration_velocity=1000.0", "--set", "robot.solver_position_iterations=64", "--set", "bricks.solver_position_iterations=64", "--table_dz", "0.008"],
    "rigid_table_+12mm": ["--physics", "on", "--set", "bricks.max_depenetration_velocity=1000.0", "--set", "robot.solver_position_iterations=64", "--set", "bricks.solver_position_iterations=64", "--table_dz", "0.012"],
    "baseline_table_+8mm": ["--physics", "off", "--table_dz", "0.008"],
    "bias_fit": ["--physics", "on", "--set", "gripper.bias_rad=0.072"],
    "bias_fit_thick": ["--physics", "on", "--set", "gripper.bias_rad=0.072", "--set", "gripper.pad.max_thickness=0.016"],
    "bias_fit_sdf": ["--physics", "on", "--set", "gripper.bias_rad=0.072", "--set", "gripper.collider=sdf"],
    "bias_fit_eff2.9": ["--physics", "on", "--set", "gripper.bias_rad=0.072", "--set", "gripper.effort_nm=2.9"],
    "bias_fit_eff0.6": ["--physics", "on", "--set", "gripper.bias_rad=0.072", "--set", "gripper.effort_nm=0.6"],
    "bias_fit_mu0.9": ["--physics", "on", "--set", "gripper.bias_rad=0.072", "--set", "gripper.material.static_friction=0.9",
                       "--set", "gripper.material.dynamic_friction=0.8"],
}


def run_config(name: str, extra: list[str]) -> Path:
    out = OUT / f"gate_{name}"
    log = OUT / f"gate_{name}.log"
    cmd = [sys.executable, str(HERE / "realism_gate.py"), "--skip", "step", "--trace", "1", "--out", str(out)] + CONFIGS[name] + extra
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{name}] exit {r.returncode} in {time.time() - t0:.0f} s -> {out}", flush=True)
    return out


def close_frame(grip, insp, thr=12.0, stay=15.0, open_min=15.0):  # validate.py's rule
    g = grip[:insp] if insp else grip
    if len(g) == 0 or g.max() < open_min:
        return None
    for f in range(1, len(g)):
        if g[f - 1] > thr >= g[f] and (g[f:] <= stay).all():
            return f
    return None


def analyse(name: str) -> dict | None:
    d = OUT / f"gate_{name}"
    if not (d / "validate.json").exists():
        return None
    rep = json.loads((d / "validate.json").read_text())["replay"]
    tr = np.load(d / "replay_traces.npz")
    sim, real, eps = tr["sim"], tr["real"], tr["episodes"].tolist()
    labels = {r["episode_index"]: r for r in map(json.loads, (DATA / "meta/dlsj/labels.jsonl").read_text().splitlines())
              if r}
    zmax = np.array(rep["brick_max_height_m"])
    lifted = zmax > 0.0096 + 0.03
    hold_sim, hold_real, per = [], [], {}
    for k, e in enumerate(eps):
        L = int(np.isfinite(real[k, :, 0]).sum())
        insp = labels[e]["inspection_frame"]
        cf = close_frame(real[k, :L, 5], insp)
        if cf is None:
            continue
        w = slice(cf + 10, insp + 1)
        hs, hr = float(np.median(sim[k, w, 5])), float(np.median(real[k, w, 5]))
        per[e] = {"lifted": bool(lifted[k]), "brick_max_z": float(zmax[k]), "hold_sim": round(hs, 2),
                  "hold_real": round(hr, 2)}
        if lifted[k]:
            hold_sim.append(hs)
            hold_real.append(hr)
    tilt = gap_held = embedded = None
    if (d / "trace.npy").exists():
        tr = np.load(d / "trace.npy")
        vals = []
        for k in range(len(eps)):
            held = tr[:, k, 8] > 0.035
            if held.any():
                vals.append(float(np.median(tr[held, k, 7])))
        tilt = round(float(np.median(vals)), 1) if vals else None
        gq = OUT / "gap_vs_q.json"
        if gq.exists():  # jaw gap (pads, 5 mm above the tip) at the held angle: < 31.7 mm = pad inside the brick
            rows_g = sorted(json.loads(gq.read_text())["rows"], key=lambda r: r["real_q"])
            qs = np.array([r["real_q"] for r in rows_g])
            gs = np.array([r["gap_5mm_mm"] for r in rows_g])
            gaps = []
            for k in range(len(eps)):
                held = tr[:, k, 8] > 0.035
                if held.any():
                    gaps.append(float(np.interp(np.median(tr[held, k, 0]), qs, gs)))
            gap_held = round(float(np.median(gaps)), 1) if gaps else None
            embedded = int(sum(g < 29.7 for g in gaps))
    p = (d / "params_used.json")
    return {"config": name, "grasp": rep["grasp_detected_frac"], "lift": rep["lifted_3cm_frac"],
            "inspect_view": rep["in_inspect_view_at_mark_frac"],
            "tracking_rmse_deg": {k: round(v, 2) for k, v in rep["tracking_rmse_deg_sim_vs_real"].items()},
            "hold_sim_lifted_median": round(float(np.median(hold_sim)), 2) if hold_sim else None,
            "hold_real_same_eps_median": round(float(np.median(hold_real)), 2) if hold_real else None,
            "hold_sim_all_median_validate": round(rep["gripper_hold_median_sim"], 2),
            "ticks_per_s_all_envs": rep.get("ticks_per_s_all_envs"),
            "brick_tilt_in_grip_deg": tilt, "jaw_gap_held_mm": gap_held, "held_with_pad_inside_brick": embedded,
            "lifted_episodes": [e for k, e in enumerate(eps) if lifted[k]], "per_episode": per,
            "params": json.loads(p.read_text()) if p.exists() else None}


def table(rows: list[dict]) -> str:
    h = ("| config | grasp | lift >= 3 cm | inspect view | held reading (lifted eps) / real | brick tilt in grip (deg) | jaw gap held (mm, brick 31.7) / pad inside brick "
         "| RMSE pan/lift/elbow/wflex/wroll/grip (deg) |\n")
    h += "|---|---|---|---|---|---|---|---|\n"
    for r in rows:
        t = r["tracking_rmse_deg"]
        h += (f"| {r['config']} | {r['grasp']:.0%} | {r['lift']:.0%} | {r['inspect_view']:.0%} | "
              f"{r['hold_sim_lifted_median']} / {r['hold_real_same_eps_median']} | {r.get('brick_tilt_in_grip_deg')} | "
              f"{r.get('jaw_gap_held_mm')} / {r.get('held_with_pad_inside_brick')} | "
              + " / ".join(f"{t[j]:.2f}" for j in J) + " |\n")
    return h


def plot(rows: list[dict], names: list[str]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = [r for r in rows if r["config"] in names]
    if len(sel) < 2:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    eps = sorted(sel[0]["per_episode"])
    w = 0.8 / len(sel)
    for i, r in enumerate(sel):
        z = [r["per_episode"].get(e, {}).get("brick_max_z", 0) * 100 for e in eps]
        ax[0].bar(np.arange(len(eps)) + i * w, z, w, label=r["config"])
    ax[0].axhline(3.96, color="k", lw=0.8, ls="--")
    ax[0].set_xticks(np.arange(len(eps)) + 0.4 - w / 2, [str(e) for e in eps], rotation=60, fontsize=8)
    ax[0].set_ylabel("brick max height (cm)")
    ax[0].set_title("open-loop replay of 20 cal2 episodes: brick lifted?")
    ax[0].legend(fontsize=8)
    # gripper traces of one lifted episode
    e = next((e for e in sel[-1]["lifted_episodes"]), eps[0])
    for r in sel:
        tr = np.load(OUT / f"gate_{r['config']}" / "replay_traces.npz")
        k = tr["episodes"].tolist().index(e)
        ax[1].plot(tr["sim"][k, :, 5], label=f"sim {r['config']}")
    ax[1].plot(tr["real"][k, :, 5], "k", lw=1.5, label="real")
    ax[1].set_ylim(0, 40)
    ax[1].set_xlabel("tick (30 Hz)")
    ax[1].set_ylabel("gripper reading (0-100)")
    ax[1].set_title(f"episode {e}: gripper (real hold ~5.8)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "replay_compare.png", dpi=110)
    print("wrote", OUT / "replay_compare.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="baseline,physics")
    ap.add_argument("--analyse_only", action="store_true")
    ap.add_argument("--no_drop", action="store_true")
    ap.add_argument("--extra", default="", help="extra realism_gate args for every config (space separated)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = [c for c in a.configs.split(",") if c]
    if not a.analyse_only:
        for n in names:
            run_config(n, a.extra.split() if a.extra else [])
    rows = [r for r in (analyse(n) for n in names) if r]
    prev = OUT / "validate_physics.json"
    allrows = {r["config"]: r for r in json.loads(prev.read_text())} if prev.exists() else {}
    allrows.update({r["config"]: r for r in rows})
    prev.write_text(json.dumps(list(allrows.values()), indent=1))
    md = table(list(allrows.values()))
    (OUT / "validate_physics.md").write_text(md)
    print(md, flush=True)
    plot(list(allrows.values()), [n for n in ("baseline", "physics") if n in allrows])
    if not a.no_drop and not a.analyse_only:
        for phys in ("off", "on"):
            with open(OUT / f"drop_{phys}.log", "w") as f:
                subprocess.run([sys.executable, str(HERE / "drop_test.py"), "--physics", phys], stdout=f,
                               stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, str(HERE / "drop_test.py"), "--plot_only"])


if __name__ == "__main__":
    main()
