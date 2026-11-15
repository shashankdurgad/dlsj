"""Real fin bending from the wrist camera: fit a hinged fin chain to the red TPU silhouettes (plain python).

The wrist camera is rigid with gripper_link, so in the real wrist frames the fixed fin, the moving fin and their
bending are seen directly. For each frame the CAD fins are cut into the bending chain of model.segments (slices=True,
`--hinges` N), projected with the fitted wrist mount (robot/wrist_mount.json, measured intrinsics + distortion) and
the jaw angle q plus every hinge angle are fitted to the red mask (symmetric boundary chamfer, px). Angles follow
the sim joints: + = about the joint axis (Y of gripper_link for the fixed fin, Z of the jaw link for the moving
one); `toward` in the outputs is signed so that + = the fin face moving toward the other fin (curling in).

    python sim/twin/gripper/fin_shape.py frames [--n_hold 30 --n_closed 12]   # PyAV, real frames -> npz
    python sim/twin/gripper/fin_shape.py fit --hinges 3                        # per-frame chain fit -> json
    python sim/twin/gripper/fin_shape.py mapping [--n_eps 40]                  # reading -> jaw angle, open frames
    python sim/twin/gripper/fin_shape.py score --tag x --hinges 3 --q 0.05 --fixed a b c --jaw a b c
                                                          # contour error of a sim fin shape on the hold frames
Writes sim/twin/out/gripper/fin_shape_*.{npz,json,jpg}.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.gripper import model  # noqa: E402
from sim.twin.gripper.fit_wrist import WristCam, chamfer, edge_dt, red_mask  # noqa: E402

OUT = REPO / "sim/twin/out/gripper"
FRAMES = OUT / "fin_shape_frames.npz"
SCALE, BIAS = 0.02307670501098437, -0.242  # gripper mapping (rad per reading unit, touch-calibrated bias)
CAM = None


def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Fin:
    """One fin as a hinged chain (visual mesh split at the hinges) posed by hinge angles, in its link frame."""

    def __init__(self, which: str, hinges: int, slices: bool = True):
        self.which, self.axis = which, model.HINGE_AXIS[which]
        self.chain = model.segments(which, hinges, slices=slices and hinges > 0)
        self.parts = model._split_fin_visual(which, self.chain, 0.0)

    def mesh(self, angles) -> tuple[np.ndarray, np.ndarray]:
        R, o, prev = np.eye(3), np.zeros(3), np.zeros(3)
        vs, fs, n = [], [], 0
        for j, (seg, (v, f)) in enumerate(zip(self.chain, self.parts, strict=True)):
            if j:
                o = o + R @ (seg["origin"] - prev)
                R = R @ _rot(self.axis, float(angles[j - 1]))
                prev = seg["origin"]
            vs.append(o + v.astype(float) @ R.T)
            fs.append(f + n)
            n += len(v)
        return np.concatenate(vs), np.concatenate(fs)


def toward_sign(which: str) -> float:
    """+1 if a positive hinge angle moves this fin's face toward the other fin (checked on the rigid geometry)."""
    fin = Fin(which, 1)
    tip0 = fin.mesh([0.0])[0]
    tip1 = fin.mesh([0.05])[0]
    if which == "fixed":  # the fixed face looks along +x (towards the jaw)
        return float(np.sign(tip1[:, 0].mean() - tip0[:, 0].mean()))
    g0, g1 = model.jaw_to_grip(tip0, 0.2), model.jaw_to_grip(tip1, 0.2)
    return float(np.sign(g0[:, 0].mean() - g1[:, 0].mean()))


def render(fins, q: float, af, aj) -> np.ndarray:
    vf, ff = fins[0].mesh(af)
    vj, fj = fins[1].mesh(aj)
    return CAM.silhouette(vf, ff) | CAM.silhouette(model.jaw_to_grip(vj, q), fj)


def _init():
    global CAM
    CAM = WristCam()


def fit_frame(args):
    """(key, q, fixed angles, jaw angles, chamfer px, rigid chamfer px) for one frame."""
    key, img, grip, hinges = args
    real = red_mask(img)
    real_dt = edge_dt(real)
    fins = (Fin("fixed", hinges), Fin("jaw", hinges))
    q0 = SCALE * grip + BIAS

    def cost(x):
        return chamfer(render(fins, x[0], x[1:1 + hinges], x[1 + hinges:]), real, real_dt) + \
            2.0 * float(np.sum(x[1:] ** 2))  # weak pull to the CAD shape (2 px per rad^2)

    rigid = minimize(lambda x: chamfer(render(fins, x[0], np.zeros(hinges), np.zeros(hinges)), real, real_dt),
                     [q0], method="Powell", options={"xtol": 1e-3, "ftol": 1e-3, "maxfev": 300})
    best = None
    for start in (np.r_[rigid.x[0], np.zeros(2 * hinges)], np.r_[q0, np.zeros(2 * hinges)]):
        r = minimize(cost, start, method="Powell", options={"xtol": 1e-3, "ftol": 1e-3, "maxfev": 400 * (1 + hinges)})
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    ch = chamfer(render(fins, x[0], x[1:1 + hinges], x[1 + hinges:]), real, real_dt)
    return key, float(x[0]), x[1:1 + hinges].tolist(), x[1 + hinges:].tolist(), float(ch), float(rigid.fun)


def stage_frames(n_hold: int, n_closed: int):
    from sim.twin.align import frames as F

    labels = F.labels()
    rng = np.random.default_rng(1)
    eps = rng.permutation(sorted(labels))
    keep, nh, nc = {}, 0, 0
    for ep in eps:
        if nh >= n_hold and nc >= n_closed:
            break
        s = F.states(int(ep))
        g = s[:, 5]
        want = []
        if nc < n_closed and g[0] < 3.2:
            want.append(("closed", 0))
        insp = int(labels[ep].get("inspection_frame") or len(g) - 1)
        hold = np.flatnonzero((g > 4.3) & (g < 7.5))
        hold = hold[hold <= insp]
        if nh < n_hold and len(hold) > 20:  # a settled hold: the middle of the held stretch before inspection
            want.append(("hold", int(hold[len(hold) // 2])))
        if not want:
            continue
        fr = F.video_frames(int(ep), "wrist", [i for _, i in want])
        for kind, i in want:
            keep[f"{kind}_{ep}_{i}"] = fr[i]
            keep[f"{kind}_{ep}_{i}_grip"] = np.float32(g[i])
            nh += kind == "hold"
            nc += kind == "closed"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FRAMES, **keep)
    print("FRAMES_OK hold", nh, "closed", nc)


def load_frames():
    d = np.load(FRAMES)
    keys = sorted(k for k in d.files if not k.endswith("_grip"))
    return keys, [d[k] for k in keys], [float(d[k + "_grip"]) for k in keys]


def stage_fit(hinges: int, workers: int):
    keys, imgs, grips = load_frames()
    with Pool(workers, initializer=_init) as pool:
        res = pool.map(fit_frame, [(k, im, g, hinges) for k, im, g in zip(keys, imgs, grips, strict=True)])
    sf, sj = toward_sign("fixed"), toward_sign("jaw")
    rows = [{"key": k, "grip": round(g, 2), "q": round(q, 4), "fixed_deg": np.degrees(af).round(1).tolist(),
             "jaw_deg": np.degrees(aj).round(1).tolist(), "chamfer_px": round(c, 2), "rigid_chamfer_px": round(rc, 2)}
            for (k, q, af, aj, c, rc), g in zip(res, grips, strict=True)]
    out = {"hinges": hinges, "toward_sign": {"fixed": sf, "jaw": sj}, "frames": rows}
    for kind in ("hold", "closed"):
        sel = [r for r in rows if r["key"].startswith(kind)]
        if not sel:
            continue
        out[kind] = {
            "n": len(sel),
            "chamfer_px_median": round(float(np.median([r["chamfer_px"] for r in sel])), 2),
            "rigid_chamfer_px_median": round(float(np.median([r["rigid_chamfer_px"] for r in sel])), 2),
            "q_median": round(float(np.median([r["q"] for r in sel])), 4),
            "grip_median": round(float(np.median([r["grip"] for r in sel])), 2),
            # + = toward the other fin (curling in), per hinge root -> tip
            "fixed_toward_deg_median": (sf * np.median([r["fixed_deg"] for r in sel], 0)).round(1).tolist(),
            "jaw_toward_deg_median": (sj * np.median([r["jaw_deg"] for r in sel], 0)).round(1).tolist(),
        }
    (OUT / f"fin_shape_fit_{hinges}.json").write_text(json.dumps(out, indent=1))
    print("FIT", json.dumps({k: v for k, v in out.items() if k != "frames"}))
    sheet(keys, imgs, {r["key"]: r for r in rows}, hinges, f"fin_shape_fit_{hinges}.jpg")


def sheet(keys, imgs, fits: dict, hinges: int, name: str, sim: tuple | None = None, slices: bool = True):
    """Contact sheet: real frame, fitted chain outline (green), optional sim shape (yellow)."""
    global CAM
    CAM = CAM or WristCam()
    fins = (Fin("fixed", hinges, slices), Fin("jaw", hinges, slices))
    idx = [i for i, k in enumerate(keys) if k.startswith("hold")][:8] + \
        [i for i, k in enumerate(keys) if k.startswith("closed")][:4]
    tiles = []
    for i in idx:
        im = imgs[i].copy()
        r = fits.get(keys[i])
        layers = []
        if r is not None:
            layers.append((render(fins, r["q"], np.radians(r["fixed_deg"]), np.radians(r["jaw_deg"])), (0, 255, 0)))
        if sim is not None:
            layers.append((render(fins, *sim), (255, 255, 0)))
        for m, col in layers:
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, cs, -1, col, 1)
        txt = keys[i] + (f" {r['chamfer_px']:.1f}px" if r else "")
        cv2.putText(im, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        tiles.append(cv2.resize(im, (320, 240)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r + 4], 1) for r in range(0, len(tiles), 4)]
    cv2.imwrite(str(OUT / name), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def fit_q_frame(args):
    """(key, reading, fitted jaw angle, chamfer px) of one open-gripper frame, rigid fins."""
    key, img, grip = args
    real = red_mask(img)
    real_dt = edge_dt(real)
    fins = (Fin("fixed", 0), Fin("jaw", 0))

    def cost(q):
        return chamfer(render(fins, float(np.atleast_1d(q)[0]), [], []), real, real_dt)

    grid = np.arange(-0.2, 1.2, 0.03)
    qb = grid[int(np.argmin([cost(q) for q in grid]))]
    r = minimize(cost, [qb], method="Powell", options={"xtol": 1e-3, "ftol": 1e-3, "maxfev": 200})
    return key, grip, float(np.atleast_1d(r.x)[0]), float(r.fun)


def stage_mapping(n_eps: int, workers: int):
    """Gripper reading -> jaw angle across the opening range: rigid-fin fits on frames before the first close (no
    brick between the fins), then linear and quadratic fits of q(u)."""
    from sim.twin.align import frames as F

    _init()
    rng = np.random.default_rng(2)
    eps = rng.choice(sorted(F.labels()), n_eps, replace=False)
    jobs = []
    for ep in eps:
        g = F.states(int(ep))[:, 5]
        opened = np.flatnonzero(g > 12)
        if not len(opened):
            continue
        close = opened[0] + int(np.argmax(g[opened[0]:] < 8)) if (g[opened[0]:] < 8).any() else len(g)
        idx = sorted({int(np.argmin(np.abs(g[:close] - t))) for t in (3, 8, 14, 20, 26, 32, 40)})
        fr = F.video_frames(int(ep), "wrist", idx)
        jobs += [(f"{ep}_{i}", fr[i], float(g[i])) for i in idx]
    with Pool(workers, initializer=_init) as pool:
        res = pool.map(fit_q_frame, jobs)
    u = np.array([r[1] for r in res])
    q = np.array([r[2] for r in res])
    c = np.array([r[3] for r in res])
    good = c < np.percentile(c, 70)
    lin = np.polyfit(u[good], q[good], 1)
    quad = np.polyfit(u[good], q[good], 2)
    bins = [(lo, hi) for lo, hi in ((0, 5), (5, 10), (10, 16), (16, 22), (22, 28), (28, 36), (36, 50))]
    table = []
    for lo, hi in bins:
        m = good & (u >= lo) & (u < hi)
        if m.sum():
            table.append({"u": [lo, hi], "n": int(m.sum()), "u_median": round(float(np.median(u[m])), 2),
                          "q_fit_median": round(float(np.median(q[m])), 4),
                          "q_touch_mapping": round(float(SCALE * np.median(u[m]) + BIAS), 4)})
    out = {"n_frames": len(res), "n_good": int(good.sum()),
           "chamfer_px_median_good": round(float(np.median(c[good])), 2),
           "linear": {"scale": round(float(lin[0]), 5), "bias": round(float(lin[1]), 4)},
           "quadratic": [round(float(v), 6) for v in quad], "bins": table,
           "per_frame": [{"key": r[0], "u": round(r[1], 2), "q": round(r[2], 4), "chamfer_px": round(r[3], 2)}
                         for r in res]}
    (OUT / "fin_mapping.json").write_text(json.dumps(out, indent=1))
    print("MAPPING", json.dumps({k: v for k, v in out.items() if k != "per_frame"}))


def stage_score(tag: str, hinges: int, q: float, af, aj, slices: bool = True):
    """Contour error of one sim fin shape (jaw angle q, hinge angles in deg) on every hold frame."""
    _init()
    keys, imgs, _ = load_frames()
    fins = (Fin("fixed", hinges, slices), Fin("jaw", hinges, slices))
    pred = render(fins, q, np.radians(af), np.radians(aj))
    errs = {}
    for k, im in zip(keys, imgs, strict=True):
        if k.startswith("hold"):
            real = red_mask(im)
            errs[k] = round(chamfer(pred, real, edge_dt(real)), 2)
    out = {"tag": tag, "hinges": hinges, "q": q, "fixed_deg": list(af), "jaw_deg": list(aj),
           "chamfer_px_median": round(float(np.median(list(errs.values()))), 2), "per_frame": errs}
    (OUT / f"fin_shape_score_{tag}.json").write_text(json.dumps(out, indent=1))
    print("SCORE", tag, out["chamfer_px_median"])
    sheet(keys, imgs, {}, hinges, f"fin_shape_score_{tag}.jpg", sim=(q, np.radians(af), np.radians(aj)), slices=slices)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("frames", "fit", "score", "mapping"))
    ap.add_argument("--n_eps", type=int, default=40)
    ap.add_argument("--n_hold", type=int, default=30)
    ap.add_argument("--n_closed", type=int, default=12)
    ap.add_argument("--hinges", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", default="sim")
    ap.add_argument("--q", type=float, default=0.0)
    ap.add_argument("--fixed", type=float, nargs="*", default=[])
    ap.add_argument("--jaw", type=float, nargs="*", default=[])
    ap.add_argument("--hull", action="store_true", help="score: the CAD-piece chain (bend.slices false)")
    a = ap.parse_args()
    if a.stage == "frames":
        stage_frames(a.n_hold, a.n_closed)
    elif a.stage == "mapping":
        stage_mapping(a.n_eps, a.workers)
    elif a.stage == "fit":
        stage_fit(a.hinges, a.workers)
    else:
        stage_score(a.tag, a.hinges, a.q, a.fixed, a.jaw, not a.hull)
