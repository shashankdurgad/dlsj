"""Render a few candidate material looks on a small set of fit samples in ONE Isaac process (fast A/B of gloss /
tint settings; the coordinate descent of fit_visuals.py is too slow while the lights are still changing).

    sim/twin/run.sh sim/twin/visuals/fit/sweep_looks.py --looks sim/twin/out/visual_fit/looks_<tag>.json --tag <tag>
                                                         [--n 12] [--split fit]

looks_<tag>.json: [{"name": "B", "table_clearcoat": 0.15, "table_tint": [1, 0.95, 0.87], ...}, ...] -- TwinLook field
overrides on top of the current default look (lighting.json / materials.json). Extra keys: "plate_value" (x all plate
colours), "robot_value" (x body / servo colours), "plate_ring_amp" (m; rebuilds the visual plate meshes with that
floor ripple, assets.plate_mesh), "table_texture_file" (a wood_tone.py variant), "table_specular" / "plate_specular"
/ "brick_specular" (base-layer specular level, 0.5 = ior 1.5), "brick_value" / "brick_chroma" (x the brick
palette's value / saturation). Every look is applied on the live stage
(twin_scene.refresh_look).

Per look and camera, on renders through each camera's response (twin_scene.color_correct, --ccm):
* region Lab of real / sim, dE, dL, chroma distance dC (dE without lightness), highlight strength (p99 - p50 of L);
* per plate colour (connected components of the plate mask, labelled by the nearest palette colour of the RAW
  render's mean): the same, plus linear-RGB means of the real frame and of the raw render (plate_albedo.py's input);
* per brick colour: real / sim Lab where both show a brick (brick_stats);
* plate ring contrast (ring_energy: RMS of the azimuthally averaged high-pass luma over the elliptical radius of the
  plate floor, minus its noise floor). Not reliable: dominated by sim / real plate misregistration;
* table / plate luma mean / std and chroma (lighting/metrics.contrast); background luma p5 / p50 / mean (bg_stats).
Writes sweep_<tag>.json, sweep_<tag>_<cam>.jpg (rows = samples: real | look 1 | look 2 ...), sweep_<tag>_plates.jpg
(plate close-ups, top and inspect) and sweep_<tag>.npz (raw and camera-response renders, masks).
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
ap.add_argument("--looks", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--split", default="fit")
ap.add_argument("--rows", type=int, default=4, help="samples shown in the sheets")
ap.add_argument("--ccm", default=None, help="camera_color.json for the camera response (default: the repo's)")
ap.add_argument("--skip-eps", type=int, nargs="*", default=[], help="episodes left out (e.g. a pose that hangs PhysX)")
ap.add_argument("--moments", nargs="*", default=None, help="only these moments (e.g. start reach end)")
ap.add_argument("--offset", type=int, default=0, help="first pool index of the evenly spaced sample choice")
ap.add_argument("--physics-set", nargs="*", default=["gripper.finray.bend.hinges=0"],
                help="physics/params.json overrides (TWIN_PHYSICS_SET); default rigid fins (renders need no bending)")
ap.add_argument("--trace-s", type=float, default=0, help="dump every thread's stack every N s (hang diagnosis)")
args = ap.parse_args()
if args.trace_s:
    import faulthandler

    faulthandler.dump_traceback_later(args.trace_s, repeat=True)

if args.physics_set:
    import os

    os.environ["TWIN_PHYSICS_SET"] = ";".join(args.physics_set)
from sim.twin.visuals.fit.harness import OUT, Harness  # noqa: E402

h = Harness()
h.verbose = bool(args.trace_s)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sim.twin.lighting import metrics as lmetrics  # noqa: E402
from sim.twin.visuals import twin_scene as ts  # noqa: E402
from sim.twin.visuals.fit import eval_visuals as ev  # noqa: E402

looks = json.loads(Path(args.looks).read_text())
samples, real_all = ev.load_eval()
pool = [i for i, s in enumerate(samples) if s["split"] == args.split and s["ep"] not in args.skip_eps
        and (args.moments is None or s["moment"] in args.moments)]
ids = pool[args.offset:: max(1, len(pool) // args.n)][: args.n]
real = {c: real_all[c][ids] for c in ev.CAMS}
look = h.env.look
base = copy.deepcopy({k: getattr(look, k) for k in look.__dataclass_fields__})


def set_look(over: dict):
    """Base look + overrides, applied on the live stage."""
    for k, v in base.items():
        setattr(look, k, copy.deepcopy(v))
    for k, v in over.items():
        if k == "name":
            continue
        if k in ("plate_ring_amp", "table_texture_file") or k.endswith("_specular"):
            continue
        if k in ("brick_value", "brick_chroma"):
            continue
        if k == "plate_value":
            look.plate_rgb = {n: tuple(float(np.clip(c * v, 0, 255)) for c in rgb) for n, rgb in look.plate_rgb.items()}
        elif k == "robot_value":
            look.robot_body_rgb = tuple(float(np.clip(c * v, 0, 255)) for c in look.robot_body_rgb)
            look.robot_servo_rgb = tuple(float(np.clip(c * v, 0, 255)) for c in look.robot_servo_rgb)
        elif k in ("plate_rgb", "brick_rgb"):
            getattr(look, k).update({n: tuple(c) for n, c in v.items()})
        else:
            setattr(look, k, tuple(v) if isinstance(v, list) else v)
    bv, bc = float(over.get("brick_value", 1.0)), float(over.get("brick_chroma", 1.0))
    if bv != 1.0 or bc != 1.0:
        look.brick_rgb = {n: scale_color(rgb, bv, bc) for n, rgb in look.brick_rgb.items()}
    ts.refresh_look(look)
    from sim.twin.visuals.assets import PLATE_VIS_RING_AMP

    set_ring_amp(float(over.get("plate_ring_amp", PLATE_VIS_RING_AMP)))
    if "table_texture_file" in over:  # a fresh file per look (switching back to a used file can keep a low mip)
        import omni.usd
        from pxr import Sdf

        tex = omni.usd.get_context().get_stage().GetPrimAtPath(f"{ts.LOOKS}/TableTop/Tex")
        tex.GetAttribute("inputs:file").Set(Sdf.AssetPath(str(Path(over["table_texture_file"]).resolve())))
    set_specular({k: over.get(f"{k}_specular", getattr(look, f"{k}_specular", 0.5))
                  for k in ("table", "plate", "brick")})


def brick_stats(R, S, raw, M, acc):
    """Per brick palette colour: Lab of real / sim on the pixels where BOTH show a brick (sim brick mask, no robot
    in front, real chroma C* > 20: the real brick, not the table it may be misplaced on), labelled by the nearest
    palette colour of the raw render's mean over the connected component."""
    m = cv2.erode((M == 4).astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lab_ = cv2.connectedComponents(m)
    pal = {k: np.array(v, np.float32) for k, v in look.brick_rgb.items()}
    lr, ls = ev.lab(R), ev.lab(S)
    sat = np.hypot(lr[..., 1], lr[..., 2]) > 20
    for q in range(1, n):
        cm = lab_ == q
        if cm.sum() < 60:
            continue
        mean_s = raw[cm].mean(0)
        k = min(pal, key=lambda c: float(np.abs(pal[c] / max(pal[c].sum(), 1) - mean_s / max(mean_s.sum(), 1)).sum()))
        both = cm & sat
        if both.sum() < 40:
            continue
        a = acc.setdefault(k, dict(real=[], sim=[]))
        a["real"].append(lr[both].mean(0))
        a["sim"].append(ls[both].mean(0))


def scale_color(rgb, value: float, chroma: float) -> tuple:
    """sRGB 0-255 colour with its value x `value` and its saturation (distance from its grey) x `chroma`."""
    c = np.asarray(rgb, float)
    g = c.mean()
    return tuple(float(v) for v in np.clip((g + chroma * (c - g)) * value, 0, 255))


def bg_stats(img, M) -> dict:
    """Luma p5 / p50 / mean of the background (no region bit, upper 60 % of the image: billboard / hall / props)."""
    m = M == 0
    m[int(0.6 * m.shape[0]):] = False
    if m.sum() < 2000:
        return {}
    y = np.asarray(img, np.float32)[m] @ np.array([0.299, 0.587, 0.114], np.float32)
    return dict(p5=float(np.percentile(y, 5)), p50=float(np.median(y)), mean=float(y.mean()))


def set_specular(levels: dict):
    """Dielectric specular level of the base layer (materials.preview_material `specular`: 0.5 = ior 1.5) of the
    table / plate / brick materials."""
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    pre = {"table": "TableTop", "plate": "Plate_", "brick": "Brick_"}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ts.LOOKS)):
        if prim.GetName() != "Shader":
            continue
        mat = prim.GetParent().GetName()
        for k, p in pre.items():
            if mat.startswith(p) and prim.GetAttribute("inputs:ior"):
                f0 = 0.08 * float(levels[k])
                prim.GetAttribute("inputs:ior").Set((1 + f0 ** 0.5) / max(1 - f0 ** 0.5, 1e-3))


def set_ring_amp(amp: float):
    """Rebuild the visual plate meshes (env 0) with floor ripple amplitude `amp` (same topology)."""
    import omni.usd
    from pxr import Usd, UsdGeom, Vt

    from sim.twin.visuals.assets import plate_mesh

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath("/World/envs/env_0")
    for prim in Usd.PrimRange(root):
        if prim.GetName() != "Dish" or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts0 = np.asarray(mesh.GetPointsAttr().Get())
        r = float(np.hypot(pts0[:, 0], pts0[:, 1]).max())
        pts, nrm, _c, _i = plate_mesh(r, ring_amp=amp)
        if len(pts) != len(pts0):
            continue
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(pts, np.float32)))
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(nrm, np.float32)))


def plate_components(M, min_px=1500):
    """[(component bool mask, ellipse)] of the plates in a region-mask image (ellipse from the outer contour)."""
    n, lab_ = cv2.connectedComponents(((M & 2) > 0).astype(np.uint8))
    out = []
    for q in range(1, n):
        comp = lab_ == q
        if comp.sum() < min_px:
            continue
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        if len(c) >= 20:
            out.append((comp, cv2.fitEllipse(c)))
    return out


def ring_energy(img, M) -> float | None:
    """Ring contrast of the plates in one image (see the module doc); None without a usable plate."""
    g = ev.gray(img)
    hp = g - cv2.GaussianBlur(g, (0, 0), 4)
    clean = cv2.erode((M == 2).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # plate only, no occluder
    vals = []
    for comp, ((cx, cy), (w, h), ang) in plate_components(M):
        yy, xx = np.nonzero(comp & clean)
        if len(xx) < 800:
            continue
        a = np.deg2rad(ang)
        dx, dy = xx - cx, yy - cy
        u, v = dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)
        rho = np.sqrt((u / (w / 2)) ** 2 + (v / (h / 2)) ** 2)
        sel = (rho > 0.1) & (rho < 0.7)
        b = np.clip(((rho[sel] - 0.1) / 0.6 * 40).astype(int), 0, 39)
        x = hp[yy[sel], xx[sel]]
        cnt = np.bincount(b, minlength=40)
        ok = cnt >= 8
        if ok.sum() < 15:
            continue
        mu = np.bincount(b, x, 40)[ok] / cnt[ok]
        var_in = (np.bincount(b, x * x, 40)[ok] / cnt[ok] - mu ** 2)
        noise = float(np.mean(var_in / cnt[ok]))
        vals.append(max(float(mu.var()) - noise, 0.0))
    return float(np.sqrt(np.median(vals))) if vals else None


def plate_crop(img, M, size=200):
    """Square crop around the largest plate of the region mask (None without one)."""
    comps = plate_components(M)
    if not comps:
        return None
    comp, _ = max(comps, key=lambda t: t[0].sum())
    ys, xs = np.nonzero(comp)
    cy, cx = int(ys.mean()), int(xs.mean())
    half = int(max(np.ptp(ys), np.ptp(xs)) * 0.6) + 4
    y0, x0 = max(cy - half, 0), max(cx - half, 0)
    crop = img[y0:cy + half, x0:cx + half]
    return cv2.resize(np.ascontiguousarray(crop), (size, size), interpolation=cv2.INTER_CUBIC)


def hl(v):
    p50, p99 = np.percentile(v, [50, 99])
    return float(p99 - p50)


def lin(img):
    """sRGB uint8 pixels -> linear 0-1."""
    c = np.asarray(img, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def plate_stats(R, S, raw, M, acc):
    """Per plate colour: real / sim Lab mean and highlight strength (connected components of the plate mask)."""
    m = ev.region_mask(M, 2, 7).astype(np.uint8)
    n, lab_ = cv2.connectedComponents(m)
    pal = {k: np.array(v, np.float32) for k, v in look.plate_rgb.items()}
    lr, ls = ev.lab(R), ev.lab(S)
    for q in range(1, n):
        cm = lab_ == q
        if cm.sum() < 800:
            continue
        mean_s = raw[cm].mean(0)  # labelled on the RAW render (the albedo's chroma; no camera response)
        k = min(pal, key=lambda c: float(np.abs(pal[c] / max(pal[c].sum(), 1) - mean_s / max(mean_s.sum(), 1)).sum()))
        a = acc.setdefault(k, dict(real=[], sim=[], hr=[], hs=[], real_lin=[], raw_lin=[]))
        a["real_lin"].append(lin(R[cm]).mean(0))
        a["raw_lin"].append(lin(raw[cm]).mean(0))
        a["real"].append(lr[cm].mean(0))
        a["sim"].append(ls[cm].mean(0))
        a["hr"].append(hl(lr[cm][:, 0]))
        a["hs"].append(hl(ls[cm][:, 0]))


T0 = time.time()
print(f"[sweep] {len(ids)} {args.split} samples, {len(looks)} looks; masks ...", flush=True)
first = h.render_set(ids, masks=True, progress=False)
masks = {c: first["mask_" + c] for c in ev.CAMS}
report, shown = {}, {c: [] for c in ev.CAMS}
crops = {c: [] for c in ("top", "inspect")}  # [look][sample] plate close-ups
raws = {c: [] for c in ev.CAMS}  # [look] camera-response-applied renders (sweep_<tag>.npz: <cam>_<look index>)
raw_out = {c: [] for c in ev.CAMS}  # [look] RAW renders (raw_<cam>_<look index>)


def rings(imgs, cam):
    """Median ring contrast over the samples (None if no plate was usable)."""
    v = [ring_energy(imgs[k], masks[cam][k]) for k in range(len(ids))]
    v = [x for x in v if x is not None]
    return round(float(np.median(v)), 3) if v else None


ring_real = {c: rings(real[c], c) for c in ev.CAMS}
print("[sweep] ring contrast real", ring_real, flush=True)
for lk in looks:
    set_look(lk)
    out = h.render_set(ids, progress=False)
    sim = ev.apply_ccm({c: out[c] for c in ev.CAMS}, args.ccm)  # every camera's response (top included)
    rep = {}
    for cam in ev.CAMS:
        regs = {}
        for rn, bit in ev.REGIONS.items():
            Lr, Ls, hr, hs = [], [], [], []
            for k in range(len(ids)):
                m = ev.region_mask(masks[cam][k], bit)
                if m.sum() < 300:
                    continue
                a, b = ev.lab(real[cam][k])[m], ev.lab(sim[cam][k])[m]
                Lr.append(a.mean(0))
                Ls.append(b.mean(0))
                hr.append(hl(a[:, 0]))
                hs.append(hl(b[:, 0]))
            if len(Lr) < 3:
                continue
            a, b = np.array(Lr), np.array(Ls)
            d = a - b
            regs[rn] = dict(real=a.mean(0).round(1).tolist(), sim=b.mean(0).round(1).tolist(),
                            dE=round(float(np.linalg.norm(d, axis=1).mean()), 2),
                            dL=round(float(-d[:, 0].mean()), 2),
                            dC=round(float(np.linalg.norm(d[:, 1:], axis=1).mean()), 2),
                            hl=[round(float(np.mean(hr)), 1), round(float(np.mean(hs)), 1)])
        acc = {}
        for k in range(len(ids)):
            plate_stats(real[cam][k], sim[cam][k], out[cam][k], masks[cam][k], acc)
        bacc = {}
        for k in range(len(ids)):
            brick_stats(real[cam][k], sim[cam][k], out[cam][k], masks[cam][k], bacc)
        regs["brick_by_colour"] = {c: dict(n=len(v["real"]), real=np.mean(v["real"], 0).round(1).tolist(),
                                           sim=np.mean(v["sim"], 0).round(1).tolist()) for c, v in bacc.items()}
        regs["plate_by_colour"] = {
            c: dict(n=len(v["real"]), real=np.mean(v["real"], 0).round(1).tolist(),
                    sim=np.mean(v["sim"], 0).round(1).tolist(),
                    hl=[round(float(np.mean(v["hr"])), 1), round(float(np.mean(v["hs"])), 1)],
                    real_lin=np.mean(v["real_lin"], 0).round(5).tolist(),
                    raw_lin=np.mean(v["raw_lin"], 0).round(5).tolist())
            for c, v in acc.items()}
        gr = [ev.lap_var(ev.gray(f)) for f in real[cam]]
        gs = [ev.lap_var(ev.gray(f)) for f in sim[cam]]
        regs["sharp"] = [round(float(np.median(gs)), 1), round(float(np.median(gr)), 1)]
        regs["rings"] = [ring_real[cam], rings(sim[cam], cam)]  # [real, sim]
        bgr = [bg_stats(real[cam][k], masks[cam][k]) for k in range(len(ids))]
        bgs = [bg_stats(sim[cam][k], masks[cam][k]) for k in range(len(ids))]
        regs["background"] = {q: [round(float(np.mean([b[q] for b in bgr if b])), 1),
                                  round(float(np.mean([b[q] for b in bgs if b])), 1)]
                              for q in ("p5", "p50", "mean")} if any(bgr) else {}
        regs["contrast"] = dict(real=lmetrics.contrast(list(real[cam]), list(masks[cam])),
                                sim=lmetrics.contrast(list(sim[cam]), list(masks[cam])))
        if cam in crops:
            crops[cam].append([plate_crop(sim[cam][k], masks[cam][k]) for k in range(min(args.rows, len(ids)))])
        rep[cam] = regs
        shown[cam].append(sim[cam][: args.rows])
        raws[cam].append(sim[cam])
        raw_out[cam].append(out[cam])
    report[lk.get("name", str(len(report)))] = dict(look=lk, metrics=rep)
    t = rep["top"]
    print(f"[sweep] {lk.get('name')}: top " + " ".join(
        f"{r}: dE {v['dE']} dL {v['dL']} dC {v['dC']} hl {v['hl']}" for r, v in t.items() if r in ev.REGIONS)
        + " | hl real/sim " + json.dumps({c: {r: rep[c][r]["hl"] for r in ("table", "plates", "bricks") if r in rep[c]}
                                          for c in ("wrist", "inspect")})
        + " | inspect bg [real, sim] " + json.dumps(rep["inspect"].get("background", {}))
        + " | table contrast real/sim " + json.dumps(
            {c: [rep[c]["contrast"]["real"].get("table"), rep[c]["contrast"]["sim"].get("table")] for c in ev.CAMS})
        + f" | t {time.time() - T0:.0f}s", flush=True)
(OUT / f"sweep_{args.tag}.json").write_text(json.dumps(dict(ids=ids, looks=report), indent=1))
np.savez_compressed(OUT / f"sweep_{args.tag}.npz", ids=np.array(ids), **{f"mask_{c}": masks[c] for c in ev.CAMS},
                    **{f"{c}_{i}": np.stack(v) for c, vs in raws.items() for i, v in enumerate(vs)},
                    **{f"raw_{c}_{i}": np.stack(v) for c, vs in raw_out.items() for i, v in enumerate(vs)})
names = [lk.get("name", str(i)) for i, lk in enumerate(looks)]
for cam in ev.CAMS:
    rows = []
    for k in range(min(args.rows, len(ids))):
        tiles = [real[cam][k]] + [s[k] for s in shown[cam]]
        caps = ["real"] + names
        row = []
        for im, cap in zip(tiles, caps, strict=True):
            im = np.ascontiguousarray(cv2.resize(im, (480, 360), interpolation=cv2.INTER_AREA))
            cv2.putText(im, cap, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, cap, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(im)
        rows.append(np.concatenate(row, 1))
    cv2.imwrite(str(OUT / f"sweep_{args.tag}_{cam}.jpg"), np.concatenate(rows, 0)[..., ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 88])
rows = []
for cam in crops:
    for k in range(min(args.rows, len(ids))):
        tiles = [plate_crop(real[cam][k], masks[cam][k])] + [c[k] for c in crops[cam]]
        if tiles[0] is None:
            continue
        row = []
        for im, cap in zip(tiles, [f"real {cam}"] + names, strict=True):
            im = np.ascontiguousarray(im if im is not None else np.zeros((200, 200, 3), np.uint8))
            cv2.putText(im, cap[:22], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, cap[:22], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(im)
        rows.append(np.concatenate(row, 1))
if rows:
    cv2.imwrite(str(OUT / f"sweep_{args.tag}_plates.jpg"), np.concatenate(rows, 0)[..., ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 90])
print("SWEEP_OK", args.tag, round(time.time() - T0), "s", flush=True)
h.close()
