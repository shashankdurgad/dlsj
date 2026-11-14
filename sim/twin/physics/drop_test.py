"""Drop test: Duplo 2x2 bricks released over a plate at increasing radial offsets (floor, sloped wall, rim, outside),
plus contact-stability cases (stacked pair, touching pair on the table and in the plate, bounce on the table).

    sim/twin/run.sh sim/twin/physics/drop_test.py --physics off     # residual env plate: flat 6 mm cylinder
    sim/twin/run.sh sim/twin/physics/drop_test.py --physics on      # twin physics: profile collider, materials
    sim/twin/run.sh sim/twin/physics/drop_test.py --plot_only       # sim/twin/out/physics/drop_test.png

Real expectation: a brick released over the floor stays put; over the sloped wall it slides / tips into the plate;
centred on the rim (r ~ R) it tips to whichever side holds its centre of mass; outside it lands on the table.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "sim/twin/out/physics"
ap = argparse.ArgumentParser()
ap.add_argument("--physics", choices=["off", "on"], default="on")
ap.add_argument("--params", default=str(Path(__file__).resolve().parent / "params.json"))
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--seconds", type=float, default=3.0)
ap.add_argument("--plot_only", action="store_true")
args = ap.parse_args()

PLATE_XY = (0.30, -0.15)
H_DROP = 0.10  # brick bottom above the table (A2: release tip 10-12 cm above the plate)
R_DROP = [0.0, 0.04, 0.06, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.115]
R_LOW = [0.06, 0.075, 0.085, 0.095]  # gentle placement: 3 cm release
H_LOW = 0.03
CASES = ([("drop", r, H_DROP) for r in R_DROP] + [("drop_low", r, H_LOW) for r in R_LOW] +
         [("stack", None, None), ("touch_table", None, None), ("touch_plate", None, None), ("bounce", None, None)])


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(REPO))
    from sim.twin.visuals import assets

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    res = {}
    for ax, phys in zip(axes, ("off", "on")):
        f = OUT / f"drop_{phys}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        res[phys] = meta["summary"]
        tz = meta["table_z"]
        if phys == "off":
            R = meta["plate_radius"]
            ax.fill([0, R, R, 0], [tz, tz, tz + 0.006, tz + 0.006], color="0.75", label="plate collider")
        else:
            pr = assets.plate_profile(meta["plate_radius"], rim_h=meta["rim_h"], floor_frac=meta["floor_frac"])
            ax.fill(pr[:, 0], pr[:, 1] + tz + 0.003, color="0.75", label="plate collider (profile)")
        ax.axhline(tz, color="saddlebrown", lw=1)
        cmap = plt.get_cmap("viridis")
        for i, (kind, r0, h0) in enumerate(CASES):
            if kind not in ("drop", "drop_low"):
                continue
            tr = d["target"][:, i]  # (T, 3) plate-relative x, y, z(world-env)
            rr = np.hypot(tr[:, 0], tr[:, 1])
            ax.plot(rr, tr[:, 2], color=cmap(r0 / 0.12), lw=1.2 if kind == "drop" else 0.8,
                    ls="-" if kind == "drop" else "--")
            ax.plot(rr[-1], tr[-1, 2], "o", color=cmap(r0 / 0.12), ms=4)
        ax.set_xlim(-0.005, 0.135)
        ax.set_ylim(tz - 0.005, tz + 0.06)
        ax.set_xlabel("brick centre distance from plate centre (m)")
        ax.set_title(f"physics {phys}: {'flat 6 mm disc' if phys == 'off' else 'measured plate profile'}"
                     f" -> in plate {meta['summary']['in_plate']}, rim {meta['summary']['on_rim']}, "
                     f"table {meta['summary']['table']}", fontsize=9)
        ax.set_aspect("equal")
    axes[0].set_ylabel("brick centre z (m), solid: 10 cm drop, dashed: 3 cm")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "drop_test.png", dpi=120)
    (OUT / "drop_test.json").write_text(json.dumps(res, indent=1))
    print("wrote", OUT / "drop_test.png", json.dumps(res), flush=True)


if args.plot_only:
    plot()
    sys.exit(0)

sys.path.insert(0, str(REPO / "sim/residual_env"))
from app import launch  # noqa: E402

app = launch(cameras=False)
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

import scene as rscene  # noqa: E402
from sim.twin.calib.profile import twin_profile  # noqa: E402

phys = None
if args.physics == "on":
    from sim.twin.physics import apply as phys  # noqa: E402

    phys.load_params(args.params, args.set)
prof = twin_profile()
layout = rscene.load_layout(str(REPO / "sim/realism/scene_params.json"))
n = len(CASES)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
scfg = rscene.build_scene_cfg(n, layout, False, profile=prof)
if phys:
    scfg = phys.patch_scene_cfg(scfg, prof)
scene = InteractiveScene(scfg)
info = phys.post_spawn(scene) if phys else {}
sim.reset()
scene.reset()
dev = sim.device
if phys:  # physics DR smoke test: one draw, then back to nominal
    try:
        dr = phys.apply_physics_dr(scene, phys.sample_physics_dr(np.random.default_rng(0)))
        phys.apply_physics_dr(scene, phys.nominal_physics())
        info["dr_smoke"] = {"ok": True, "draw": dr}
    except Exception as ex:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        info["dr_smoke"] = {"ok": False, "error": f"{type(ex).__name__}: {ex}"}
O = scene.env_origins
TZ = rscene.TABLE_TOP_Z
bowls = list(layout.bowls)
plate_obj = scene["bowl_purple"]
R_plate = phys.PARAMS["plates"]["radius"] if phys else layout.bowls["purple"][2]
brick, dis = scene["brick"], [scene[f"distractor{i}"] for i in range(len(layout.distractors))]
hb = layout.target.size[2] / 2


def pose(xyz, yaw=0.0):
    p = torch.tensor(xyz, dtype=torch.float32, device=dev)
    q = torch.tensor([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=torch.float32, device=dev)
    return torch.cat([p, q])


def place(obj, rows):
    obj.write_root_pose_to_sim(torch.stack(rows), env_ids=torch.arange(n, device=dev))
    obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=torch.arange(n, device=dev))


# plates: purple at PLATE_XY, the others parked far away on the table
for k, b in enumerate(bowls):
    xy = PLATE_XY if b == "purple" else (0.8, -0.4 + 0.4 * k)
    place(scene[f"bowl_{b}"], [torch.cat([O[e] + torch.tensor([xy[0], xy[1], TZ + 0.003], device=dev),
                                          torch.tensor([0, 0, 0, 1.0], device=dev)]) for e in range(n)])
px, py = PLATE_XY
floor_top = TZ + (0.003 if phys else 0.006)
rows_b, rows_d = [], {i: [] for i in range(len(dis))}
for e, (kind, r, h) in enumerate(CASES):
    o = O[e].tolist()
    park = [(0.85, 0.45), (0.85, 0.30), (0.70, 0.45)]
    dpos = {i: [o[0] + park[i][0], o[1] + park[i][1], TZ + hb + 0.0005] for i in range(len(dis))}
    if kind in ("drop", "drop_low"):
        b = [o[0] + px + r, o[1] + py, TZ + h + hb]
        yaw = 0.3
    elif kind == "stack":  # cream 2x2 (distractor2) on top of the target, slightly offset
        b = [o[0] + 0.25, o[1] + 0.25, TZ + hb + 0.0005]
        dpos[2] = [o[0] + 0.25 + 0.002, o[1] + 0.25, TZ + 3 * hb + 0.001]
        yaw = 0.0
    elif kind == "touch_table":  # 2x4 (distractor0) touching the target's side
        b = [o[0] + 0.25, o[1] + 0.25, TZ + hb + 0.0005]
        dpos[0] = [o[0] + 0.25 + 0.0318 / 2 + 0.0636 / 2 + 0.0002, o[1] + 0.25, TZ + hb + 0.0005]
        yaw = 0.0
    elif kind == "touch_plate":  # resting on the plate floor, touching a 2x2
        b = [o[0] + px - 0.02, o[1] + py, floor_top + hb + 0.0005]
        dpos[2] = [o[0] + px - 0.02 + 0.0318 + 0.0002, o[1] + py, floor_top + hb + 0.0005]
        yaw = 0.0
    else:  # bounce: 10 cm drop on the open table
        b = [o[0] + 0.25, o[1] + 0.25, TZ + H_DROP + hb]
        yaw = 0.0
    rows_b.append(pose(b, yaw))
    for i in range(len(dis)):
        rows_d[i].append(pose(dpos[i], yaw if kind != "touch_table" else 0.0))
place(brick, rows_b)
for i, dobj in enumerate(dis):
    place(dobj, rows_d[i])
# hold the arm still at its init pose
rob = scene["robot"]
rob.set_joint_position_target(rob.data.default_joint_pos.clone())
T = int(args.seconds * 120)
tgt = np.zeros((T, n, 3), np.float32)
dtr = np.zeros((T, len(dis), n, 3), np.float32)
for t in range(T):
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(1 / 120)
    p = (brick.data.root_pos_w - O).cpu().numpy()
    tgt[t] = p - np.array([px, py, 0.0])
    for i, dobj in enumerate(dis):
        dtr[t, i] = (dobj.data.root_pos_w - O).cpu().numpy()
summ = {"in_plate": 0, "on_rim": 0, "table": 0, "fell_through": 0}
cases = []
r_in = R_plate - 0.003
rim_top = TZ + (0.023 if phys else 0.006)
for e, (kind, r, h) in enumerate(CASES):
    fin = tgt[-1, e]
    rf = float(np.hypot(fin[0], fin[1]))
    zf = float(fin[2])
    c = {"case": kind, "r0": r, "h0": h, "r_final": round(rf, 4), "z_final": round(zf, 4)}
    if kind in ("drop", "drop_low"):
        if zf < TZ - 0.01:
            cls = "fell_through"
        elif rf < r_in - 0.01 and zf < TZ + 0.025:  # on the floor (flat, or tilted against the wall)
            cls = "in_plate"
        elif rf > R_plate:  # outside: on the table (flat, or leaning on the plate's outer wall)
            cls = "table"
        else:
            cls = "on_rim"
        c["result"] = cls
        summ[cls] += 1
        zz = tgt[:, e, 2]
        hit = np.argmax(zz < zz.min() + 0.002)
        c["bounce_m"] = round(float(zz[hit:].max() - zz[-1]), 4) if hit < T - 1 else 0.0
    else:
        last = tgt[T // 3:, e]
        c["drift_mm"] = round(float(np.linalg.norm(last[-1] - last[0])) * 1000, 3)
        c["std_mm"] = round(float(last.std(0).max()) * 1000, 4)
        if kind == "bounce":
            zz = tgt[:, e, 2]
            land = np.argmax(zz < TZ + hb + 0.002)
            c["bounce_m"] = round(float(zz[land:].max() - (TZ + hb)), 4)
        for i in range(len(dis)):
            dd = dtr[T // 3:, i, e]
            if np.linalg.norm(dd[0, :2] - (tgt[0, e, :2] + [px, py])) < 0.08:
                c[f"distractor{i}_drift_mm"] = round(float(np.linalg.norm(dd[-1] - dd[0])) * 1000, 3)
                c[f"distractor{i}_std_mm"] = round(float(dd.std(0).max()) * 1000, 4)
    cases.append(c)
meta = {"physics": args.physics, "plate_radius": R_plate, "table_z": TZ, "summary": summ, "cases": cases,
        "rim_h": phys.PARAMS["plates"]["rim_h"] if phys else 0.006,
        "floor_frac": phys.PARAMS["plates"]["floor_frac"] if phys else 1.0, "post_spawn": info}
OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / f"drop_{args.physics}.npz", target=tgt, distractors=dtr, meta=json.dumps(meta))
print("DROP", json.dumps(meta), flush=True)
app.close()
