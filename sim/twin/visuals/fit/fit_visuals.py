"""Fit the twin's look to the real frames: render -> measure -> adjust (coordinate descent on the fit episodes).

    sim/twin/run.sh sim/twin/visuals/fit/fit_visuals.py [--stages materials,backdrop] [--n 45] [--rounds 2]
                                                        [--tag fit1] [--no-write]

Stages (each renders the fit samples of the evaluation set with the candidate look; nothing here sees the held-out
episodes; run render_eval.py afterwards for the held-out metrics):
  materials coordinate descent over the material parameters (MATERIAL_PARAMS): table lacquer (clear-coat, roughness),
            plate / brick gloss and value, robot body value / roughness. Objective = top-camera score
            (eval_visuals.score: region colour dE, highlight strength, brightness histogram, SSIM) + the highlight
            terms of the wrist and inspect cameras (their colours belong to the camera response) + the table's
            luminance spread in the inspect view (the lacquer's grazing sheen). A step is multiplicative or additive
            per parameter; it is halved when neither direction improves. Writes materials.json.
  backdrop  gain of the unlit inspect billboard / hall cylinder so the background of the inspect view (pixels outside
            every region mask) has the real mean luminance. Writes materials.json.
  lights    (owned by the lighting agent; not run by default) the same descent over exposure, key / ceiling-light
            intensity and warmth (LIGHT_PARAMS). Writes lighting.json.
  ccm, blur (owned by the sensor agent; not run by default) per-camera linear-RGB 3x3 + offset from pairs of sim /
            real region means (table, every plate separately, robot, background), ridge-regularised towards the
            identity, and a gaussian blur only where the twin's image is SHARPER than the real one. Write
            camera_color.json.
Every evaluation is appended to sim/twin/out/visual_fit/fit_<tag>.jsonl.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
ap = argparse.ArgumentParser()
ap.add_argument("--stages", default="materials,backdrop")
ap.add_argument("--n", type=int, default=45, help="fit samples rendered per evaluation")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--tag", default="fit")
ap.add_argument("--no-write", action="store_true")
args = ap.parse_args()

from sim.twin.visuals.fit.harness import OUT, Harness  # noqa: E402

h = Harness()
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sim.twin.visuals import twin_scene as ts  # noqa: E402
from sim.twin.visuals.fit import eval_visuals as ev  # noqa: E402

CAMS = ev.CAMS
samples, real_all = ev.load_eval()
fit_ids = [i for i, s in enumerate(samples) if s["split"] == "fit"]
ids = fit_ids[:: max(1, len(fit_ids) // args.n)][: args.n]
sub = [dict(samples[i], split="fit") for i in ids]
real = {c: real_all[c][ids] for c in CAMS}
look = h.env.look
log_path = OUT / f"fit_{args.tag}.jsonl"
stages = args.stages.split(",")

print(f"[fit] {len(ids)} fit samples; masks ...", flush=True)
first = h.render_set(ids, masks=True, progress=False)
masks = {c: first["mask_" + c] for c in CAMS}

# name: (kind, step, lo, hi); kinds: mul (x *= step), add (x += step)
LIGHT_PARAMS = {
    "exposure": ("mul", 1.12, 0.3, 2.0),
    "key_intensity": ("mul", 1.5, 20.0, 3000.0),
    "ceiling": ("mul", 1.6, 5.0, 3000.0),
    "warmth": ("add", 0.04, -0.25, 0.25),
}
MATERIAL_PARAMS = {
    "table_value": ("mul", 1.1, 0.5, 1.5),  # first: the lacquer adds a specular veil the albedo must give back
    "table_clearcoat": ("add", 0.3, 0.0, 1.0),
    "table_clearcoat_roughness": ("mul", 1.7, 0.02, 0.6),
    "table_roughness": ("add", 0.12, 0.15, 0.95),
    "plate_roughness": ("add", 0.07, 0.03, 0.8),
    "plate_clearcoat": ("add", 0.25, 0.0, 1.0),
    "plate_clearcoat_roughness": ("mul", 1.8, 0.01, 0.5),
    "plate_value": ("mul", 1.08, 0.6, 1.5),
    "brick_roughness": ("add", 0.07, 0.03, 0.8),
    "brick_clearcoat": ("add", 0.25, 0.0, 1.0),
    "robot_value": ("mul", 1.3, 0.2, 4.0),
    "robot_body_roughness": ("add", 0.12, 0.15, 0.95),
}
plate_rgb0 = copy.deepcopy(look.plate_rgb)
body0, servo0 = tuple(look.robot_body_rgb), tuple(look.robot_servo_rgb)
x = {"exposure": look.scale(), "key_intensity": look.key_intensity,
     "ceiling": float(np.mean([c["intensity"] for c in look.ceiling_lights])) if look.ceiling_lights else 0.0,
     "warmth": 0.0, "table_roughness": look.table_roughness, "table_clearcoat": look.table_clearcoat,
     "table_value": float(np.mean(look.table_tint)),
     "table_clearcoat_roughness": look.table_clearcoat_roughness, "plate_roughness": look.plate_roughness,
     "plate_clearcoat": look.plate_clearcoat, "plate_clearcoat_roughness": look.plate_clearcoat_roughness,
     "plate_value": 1.0, "brick_roughness": look.brick_roughness, "brick_clearcoat": look.brick_clearcoat,
     "robot_value": 1.0, "robot_body_roughness": look.robot_body_roughness}


def apply(p: dict):
    """Set the look to parameter vector `p` on the live stage."""
    look.exposure_scale = p["exposure"]
    look.key_intensity = p["key_intensity"]
    for c in look.ceiling_lights:
        c["intensity"] = p["ceiling"]
    look.dome_rgb = (1.0 + p["warmth"], 1.0, 1.0 - p["warmth"])
    look.key_rgb = look.dome_rgb
    for k in ("table_roughness", "table_clearcoat", "table_clearcoat_roughness", "plate_roughness",
              "plate_clearcoat", "plate_clearcoat_roughness", "brick_roughness",
              "brick_clearcoat", "robot_body_roughness"):
        setattr(look, k, p[k])
    look.table_tint = (p["table_value"],) * 3
    pv = p["plate_value"]
    look.plate_rgb = {k: tuple(float(np.clip(v * pv, 0, 255)) for v in rgb) for k, rgb in plate_rgb0.items()}
    look.robot_body_rgb = tuple(float(np.clip(v * p["robot_value"], 0, 255)) for v in body0)
    look.robot_servo_rgb = tuple(float(np.clip(v * p["robot_value"], 0, 255)) for v in servo0)
    ts.refresh_look(look)


def hl_term(m: dict) -> float:
    """Mean |highlight strength real - sim| over the regions of one camera."""
    c = m["color"]
    return float(np.mean([abs(v["highlight_real"] - v["highlight_sim"]) for v in c.values()])) if c else 0.0


def objective(p: dict, note: str = "") -> float:
    """Render the fit samples with `p` and return the loss (logged to fit_<tag>.jsonl)."""
    apply(p)
    out = h.render_set(ids, progress=False)
    sim = ev.apply_ccm({c: out[c] for c in ("wrist", "inspect")})
    sim["top"] = out["top"]  # materials are matched to the top camera: no camera response while fitting them
    res = ev.evaluate(real, sim, masks, sub, clf=False)
    j = ev.score(res["top"]["fit"]) + 0.3 * (hl_term(res["wrist"]["fit"]) + hl_term(res["inspect"]["fit"]))
    sheen = res["inspect"]["fit"]["color"].get("table", {}).get("std_ratio_L", 1.0)
    j += 4.0 * abs(np.log(max(sheen, 1e-3)))  # lacquer: spread of the table's luminance at the grazing view
    row = dict(t=round(time.time() - T0), note=note, J=round(j, 3), top=round(ev.score(res["top"]["fit"]), 3),
               sheen_inspect=sheen,
               hl_wrist=round(hl_term(res["wrist"]["fit"]), 2), hl_inspect=round(hl_term(res["inspect"]["fit"]), 2),
               dE_top={k: v["dE"] for k, v in res["top"]["fit"]["color"].items()},
               params={k: round(v, 4) for k, v in p.items()})
    with open(log_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("[fit]", json.dumps(row), flush=True)
    return j


def step(v, kind, s, sign):
    """One coordinate step: multiplicative ('mul') or additive ('add')."""
    return v * (s if sign > 0 else 1 / s) if kind == "mul" else v + sign * s


def lin(c):
    """sRGB 0-255 -> linear 0-1."""
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def write_json(path: Path, upd: dict):
    """Merge `upd` into a JSON file."""
    d = json.loads(path.read_text()) if path.exists() else {}
    d.update(upd)
    path.write_text(json.dumps(d, indent=1))


T0 = time.time()
PARAMS = (LIGHT_PARAMS if "lights" in stages else {}) | (MATERIAL_PARAMS if "materials" in stages else {})
if PARAMS:
    best = objective(x, "start")
    steps = {k: v[1] for k, v in PARAMS.items()}
    for rnd in range(args.rounds):
        for name, (kind, _, lo, hi) in PARAMS.items():
            if name == "ceiling" and not look.ceiling_lights:
                continue
            moved = False
            for sign in (+1, -1):
                while True:
                    cand = dict(x)
                    cand[name] = float(np.clip(step(x[name], kind, steps[name], sign), lo, hi))
                    if cand[name] == x[name]:
                        break
                    j = objective(cand, f"r{rnd} {name} {'+' if sign > 0 else '-'}")
                    if j < best - 0.01:
                        x, best, moved = cand, j, True
                    else:
                        break
                if moved:
                    break
            if not moved:
                steps[name] = steps[name] ** 0.5 if kind == "mul" else steps[name] / 2
    apply(x)
    print("[fit] look done", json.dumps({k: round(v, 4) for k, v in x.items()}), "J", round(best, 3), flush=True)
    if not args.no_write and "lights" in stages:
        write_json(ts.LIGHTING_JSON, dict(exposure_scale=look.exposure_scale, key_intensity=look.key_intensity,
                                          dome_rgb=list(look.dome_rgb), key_rgb=list(look.key_rgb),
                                          ceiling_lights=look.ceiling_lights))
    if not args.no_write and "materials" in stages:
        write_json(ts.MATERIALS_JSON, dict(
            table_tint=list(look.table_tint), table_roughness=look.table_roughness,
            table_clearcoat=look.table_clearcoat,
            table_clearcoat_roughness=look.table_clearcoat_roughness, plate_roughness=look.plate_roughness,
            plate_clearcoat=look.plate_clearcoat, plate_clearcoat_roughness=look.plate_clearcoat_roughness,
            plate_rgb={k: [round(v, 1) for v in rgb] for k, rgb in look.plate_rgb.items()},
            brick_roughness=look.brick_roughness, brick_clearcoat=look.brick_clearcoat,
            robot_body_rgb=[round(v, 1) for v in look.robot_body_rgb], robot_body_roughness=look.robot_body_roughness,
            robot_servo_rgb=[round(v, 1) for v in look.robot_servo_rgb],
            note="flat TwinLook field names; sRGB 0-255 colours as the top camera sees them; fitted by "
                 "fit/fit_visuals.py (stage materials)"))

if "backdrop" in stages:
    for it in range(3):
        out = h.render_set(ids, progress=False)
        bg = masks["inspect"] == 0
        bg[:, 300:] = False  # upper image: billboard / hall
        r = float(lin(out["inspect"][bg]).mean() / max(lin(real["inspect"][bg]).mean(), 1e-6))
        print(f"[fit] backdrop it {it}: gain {look.bd_gain():.3f} hall {look.hall_gain:.3f} sim/real {r:.3f}",
              flush=True)
        if abs(r - 1) < 0.03:
            break
        look.backdrop_gain = look.bd_gain() / r
        look.hall_gain = look.hall_gain / r
        ts.refresh_look(look)
    if not args.no_write:
        write_json(ts.MATERIALS_JSON, dict(backdrop_gain=look.bd_gain(), hall_gain=look.hall_gain))

raw = h.render_set(ids, progress=False) if ("ccm" in stages or "blur" in stages) else None
ccm = json.loads(ts.CAMERA_COLOR_JSON.read_text()) if ts.CAMERA_COLOR_JSON.exists() else {}
if "ccm" in stages:
    for cam in CAMS:
        A, B, Wt = [], [], []
        for k in range(len(ids)):
            m, s_img, r_img = masks[cam][k], lin(raw[cam][k]), lin(real[cam][k])
            regs = []
            for bit, w in ((1, 3.0), (8, 1.5)):
                regs.append((ev.region_mask(m, bit), w))
            n, lab = cv2.connectedComponents(ev.region_mask(m, 2).astype(np.uint8))
            regs += [((lab == q), 2.0) for q in range(1, n)]
            regs.append(((cv2.erode((m == 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0), 1.0))
            for rm, w in regs:
                if rm.sum() < 400:
                    continue
                a, b = s_img[rm].mean(0), r_img[rm].mean(0)
                if b.max() > 0.92 or a.max() > 0.92:  # clipped
                    continue
                A.append(a)
                B.append(b)
                Wt.append(w)
        A, B, Wt = np.array(A), np.array(B), np.sqrt(np.array(Wt))[:, None]
        X = np.c_[A, np.ones(len(A))]
        lam = 0.02 * len(A) * float((A ** 2).mean())
        prior = np.r_[np.eye(3), np.zeros((1, 3))]
        reg = np.diag([lam, lam, lam, lam * 25])  # offset kept small
        sol = np.linalg.solve((X * Wt).T @ (X * Wt) + reg, (X * Wt).T @ (B * Wt) + reg @ prior)
        M, off = sol[:3].T, sol[3]
        err0 = float(np.abs(A - B).mean())
        err1 = float(np.abs(A @ M.T + off - B).mean())
        prev = ccm.get(cam, {})
        ccm[cam] = dict(M_linear=M.round(5).tolist(), offset_linear=off.round(5).tolist(), n_pairs=int(len(A)),
                        mean_abs_err_linear=[round(err0, 4), round(err1, 4)],
                        **({"blur_sigma": prev["blur_sigma"]} if "blur_sigma" in prev else {}))
        print(f"[fit] ccm {cam}: {len(A)} pairs, |err| {err0:.4f} -> {err1:.4f}, M diag {np.diag(M).round(3)}, "
              f"offset {off.round(4)}", flush=True)
    ccm["_note"] = ("per-camera response applied by twin_scene.color_correct to the RAW render: linear RGB "
                    "out = M_linear @ in + offset_linear, then gaussian blur_sigma (px at 640 wide); "
                    "fitted by fit/fit_visuals.py on sim / real region means")
    if not args.no_write:
        ts.CAMERA_COLOR_JSON.write_text(json.dumps(ccm, indent=1))

if "blur" in stages:
    tmp = OUT / f"ccm_{args.tag}.json"
    for cam in CAMS:
        if cam not in ccm:
            continue
        rs = float(np.median([ev.lap_var(ev.gray(f)) for f in real[cam]]))
        best_s, best_e, trace = 0.0, 1e9, {}
        for sg in (0.0, 0.3, 0.45, 0.6, 0.75, 0.9, 1.1, 1.3, 1.6):
            ccm[cam]["blur_sigma"] = sg
            tmp.write_text(json.dumps(ccm))
            ss = float(np.median([ev.lap_var(ev.gray(ts.color_correct(f, cam, tmp))) for f in raw[cam]]))
            trace[sg] = round(ss, 1)
            e = abs(np.log(ss / rs))
            if e < best_e - 0.02:
                best_s, best_e = sg, e
        ccm[cam]["blur_sigma"] = best_s
        ccm[cam]["sharpness_real_vs_sigma"] = dict(real=round(rs, 1), sim=trace)
        print(f"[fit] blur {cam}: real {rs:.1f}, sim by sigma {trace} -> sigma {best_s}", flush=True)
    tmp.unlink(missing_ok=True)
    if not args.no_write:
        ts.CAMERA_COLOR_JSON.write_text(json.dumps(ccm, indent=1))

print("FIT_OK", round(time.time() - T0), "s", flush=True)
h.close()
