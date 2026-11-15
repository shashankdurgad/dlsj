"""Ground truth for the real detector runs: where was the target brick, and how visible was it in every wrist frame?

    python sim/twin/perception/gt_real.py            (plain python; after real_detect.py)
    -> sim/twin/out/perception/real_table.npz   one row per (episode, frame) of the pick phase
       sim/twin/out/perception/real_fit.json    per-episode brick pose fits

Brick pose per episode (static on the table until the grasp): kind / resting pose / yaw prior from the grasp gate's
rung 0 (top camera, sim/twin/out/grasp_gate/rung0/rung0.json), position fitted to the WRIST view: the brick's
predicted visible box (geometry.Geometry.visibility with FK of the recorded joints and the fitted wrist mount) is
matched to the real detector's boxes over the frames 2.5 s .. 0.33 s before the gripper closes (vis >= 0.9), x / y /
yaw by Nelder-Mead on the summed box-corner error (Huber), started from a grid around the top-camera position AND the
FK fingertip at the close frame (the brick ends up between the fins). The top camera alone is 16 mm (median) from the
FK grasp point (rung0.md), which moves the brick ~40-80 px in the wrist view near the close, so it is only the prior.

After the close the brick is placed at the real median held pose in the gripper (HELD_LOCAL; it is pulled into
the fins while they close, so the pre-grasp table pose is not used there).

Per frame: geometry features (vis, occ_grip, oof, area_px, dist, ang, centre pixel speed), the target's predicted
visible / amodal boxes, and the real detections matched to it (IoU with the predicted visible box >= 0.3; best
confidence), plus the fixed / jaw fin zones (for the false positives on the fins).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.perception import geometry as G  # noqa: E402

OUT = REPO / "sim/twin/out/perception"
RUNG0 = REPO / "sim/twin/out/grasp_gate/rung0/rung0.json"
LONG = {"2x2": 0.0318, "2x3": 0.0477, "2x4": 0.0636}
BW, BH = 0.0318, 0.0192
# median held-brick centre in gripper_link, fitted to the real wrist boxes 6-20 frames after the close
# (held_brick.py, 599 frames / 45 episodes): bottom face ~7.6 mm above the fixed-fin tip
HELD_LOCAL = np.array([-0.0008, 0.0028, -0.0880])


def iou(a, b):
    """IoU of boxes a (..., 4) and b (..., 4)."""
    x0, y0 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    x1, y1 = np.minimum(a[..., 2], b[..., 2]), np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ar = lambda t: (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])  # noqa: E731
    return inter / np.maximum(ar(a) + ar(b) - inter, 1e-9)


def brick_dims(kind: str, pose: str):
    """Body size (L, W, H) in the brick's axes and the centre height above the table for a resting pose."""
    L = LONG[kind]
    hz = {"up": BH, "side": BW, "end": L}[pose]
    return np.array([L, BW, BH]), hz / 2


def clip_box(b):
    return np.stack(
        [
            np.clip(b[..., 0], 0, G.W),
            np.clip(b[..., 1], 0, G.H),
            np.clip(b[..., 2], 0, G.W),
            np.clip(b[..., 3], 0, G.H),
        ],
        -1,
    )


class Episode:
    def __init__(self, geo, ep, rec, states, r0, lag: float = 0.0, geo_fit=None):
        self.geo, self.ep, self.cf = geo, ep, rec["cf"]
        self.geo_fit = geo_fit or geo
        self.frames = np.array(rec["frames"])
        self.dets = [np.array(d, float).reshape(-1, 5) for d in rec["dets"]]
        self.J = state_at(states, self.frames - lag)
        self.Jcf = state_at(states, np.array([self.cf - lag]))[0]
        self.kind, self.pose = r0.get("kind", "2x2"), r0.get("pose", "up")
        self.size, self.hc = brick_dims(self.kind, self.pose)
        self.yaw0 = float(r0.get("yaw_long", 0.0))
        self.xy_top = np.array(r0["xy"]) if "xy" in r0 else None
        p, R = geo.tip(states[self.cf][None])
        self.xy_tip = p[0, :2]
        self.yaw_grip = math.atan2(R[0, 1, 0], R[0, 0, 0])

    def vis(self, x, y, yaw, idx=None, fit=False):
        J = self.J if idx is None else self.J[idx]
        fr = self.frames if idx is None else self.frames[idx]
        n = len(J)
        pos = np.tile([x, y, G.TABLE_Z + self.hc], (n, 1))
        R = np.repeat(G.yaw_R(np.array([yaw]), self.pose), n, 0)
        after = fr > self.cf
        if after.any():  # held: at the real held pose in the gripper (held_brick.py fit), axes = gripper axes
            Mf = self.geo.gripper_pose(J[after])
            pos[after] = (Mf[:, :3, :3] @ HELD_LOCAL) + Mf[:, :3, 3]
            R[after] = Mf[:, :3, :3] @ G.yaw_R(np.array([0.0]), self.pose)[0] if self.pose != "up" else Mf[:, :3, :3]
        return (self.geo_fit if fit else self.geo).visibility(J, pos, R, np.tile(self.size, (n, 1)))

    def match(self, box_pred, k, thr=0.3, conf=0.05):
        """Real box of frame k that is the target: IoU with the predicted visible box >= thr, or (thr > 0) centre
        within 0.5 of the predicted box's larger side; the most confident such box. thr == 0: the best-IoU box."""
        d = self.dets[k]
        d = d[d[:, 4] >= conf]
        if not len(d) or not np.isfinite(box_pred).all():
            return None, 0.0
        io = iou(d[:, :4], box_pred[None])
        if thr <= 0:
            j = int(np.argmax(io))
            return d[j], float(io[j])
        c = (d[:, :2] + d[:, 2:4]) / 2 - (box_pred[:2] + box_pred[2:]) / 2
        side = max(box_pred[2] - box_pred[0], box_pred[3] - box_pred[1])
        ok = (io >= thr) | (np.linalg.norm(c, axis=1) < 0.5 * side)
        if not ok.any():
            return None, float(io.max())
        j = int(np.argmax(np.where(ok, d[:, 4], -1)))
        return d[j], float(io[j])

    def cost(self, v, idx):
        """Mean over frames of (1 - IoU) between the predicted visible box and the best real box (conf >= 0.25),
        capped at 0.9 (a missed / other-object frame), + a prior pulling the brick towards the FK fingertip at the
        close frame (sigma 15 mm: the fins close around it)."""
        out = self.vis(v[0], v[1], v[2], idx, fit=True)
        bp = clip_box(out["box_vis"])
        tot, n = 0.0, 0
        for i, k in enumerate(idx):
            d = self.dets[k]
            d = d[d[:, 4] >= 0.25]
            if not np.isfinite(bp[i]).all():
                tot += 0.9 if len(d) else 0.0
                n += 1
                continue
            if not len(d):
                continue
            tot += float(min(1.0 - iou(d[:, :4], bp[i][None]).max(), 0.9))
            n += 1
        prior = float(np.sum(((np.array(v[:2]) - self.xy_tip) / 0.015) ** 2)) * 0.02
        return tot / max(n, 1) + prior + (0.0 if n >= 5 else 1.0)

    def fit(self):
        tt = self.cf - self.frames
        cand_idx = np.where((tt >= 3) & (tt <= 75))[0]
        sub = cand_idx[:: max(1, len(cand_idx) // 12)]
        best = None
        kinds = [(self.kind, self.pose)] + [(k, "up") for k in LONG if (k, "up") != (self.kind, self.pose)]
        for kind, pose in kinds:
            self.size, self.hc = brick_dims(kind, pose)
            self.pose = pose
            starts = []
            for base in [b for b in (self.xy_tip, self.xy_top) if b is not None]:
                for dx in np.linspace(-0.02, 0.02, 5):
                    for dy in np.linspace(-0.02, 0.02, 5):
                        for yaw in (self.yaw0, self.yaw_grip, self.yaw_grip + math.pi / 4):
                            starts.append((base[0] + dx, base[1] + dy, yaw))
            scored = sorted(((self.cost(s, sub), s) for s in starts), key=lambda t: t[0])[:2]
            for _, s in scored:
                r = minimize(
                    self.cost,
                    s,
                    args=(cand_idx,),
                    method="Nelder-Mead",
                    options={
                        "initial_simplex": [
                            s,
                            (s[0] + 0.006, s[1], s[2]),
                            (s[0], s[1] + 0.006, s[2]),
                            (s[0], s[1], s[2] + 0.3),
                        ],
                        "xatol": 3e-4,
                        "fatol": 1e-4,
                        "maxiter": 150,
                    },
                )
                if best is None or r.fun < best[0].fun - 0.01 * (kind != kinds[0][0]):
                    best = (r, kind, pose)
        r, self.kind, self.pose = best
        self.size, self.hc = brick_dims(self.kind, self.pose)
        x, y, yaw = r.x
        out = self.vis(x, y, yaw, cand_idx)
        good = cand_idx[(out["vis"] >= 0.9) & (tt[cand_idx] >= 10)]
        bp = clip_box(out["box_vis"])
        ious = [self.match(bp[list(cand_idx).index(k)], k, 0.0, 0.25)[1] for k in good]
        return {
            "x": float(x),
            "y": float(y),
            "yaw": float(yaw),
            "cost": float(r.fun),
            "fit_iou_median": float(np.median(ious)) if ious else 0.0,
            "n_fit": int(len(good)),
            "d_top_mm": None if self.xy_top is None else float(np.linalg.norm([x, y] - self.xy_top) * 1e3),
            "d_tip_mm": float(np.linalg.norm([x, y] - self.xy_tip) * 1e3),
        }


def state_at(states, t):
    """Joint states at (fractional) frame times t (linear interpolation, clamped)."""
    t = np.clip(np.asarray(t, float), 0, len(states) - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(states) - 1)
    a = (t - i0)[:, None]
    return states[i0] * (1 - a) + states[i1] * a


def fin_zones(geo, J):
    """Pixel boxes of the fixed / jaw fins (full CAD fin mesh) per frame: (n, 2, 4)."""
    g = np.load(G.FINRAY)
    M = geo.gripper_pose(J)
    Rw, C = geo.camera_pose(J)
    out = np.zeros((len(J), 2, 4))
    for i, part in enumerate(("fixed_fin", "jaw_fin")):
        v = g[part + "_v"].astype(float)[::4]
        for k in range(len(J)):
            vv = v
            if part.startswith("jaw"):
                vv = v @ geo.jaw_rot(np.array([J[k, 5]]))[0].T + G.JAW_XYZ
            vw = vv @ M[k, :3, :3].T + M[k, :3, 3]
            uv, z = geo.project(Rw[k : k + 1], C[k : k + 1], vw[None])
            uv = uv[0][z[0] > 0.005]
            out[k, i] = clip_box(np.r_[uv.min(0), uv.max(0)])
    return out


_W = {}


def _init(lag):
    from sim.twin.align import frames as F

    _W.update(
        geo=G.Geometry(),
        geo_fit=G.Geometry(parts=(), samples=4),
        F=F,
        lag=lag,
        r0=json.loads(RUNG0.read_text())["episodes"],
    )


def process(path):
    """One episode: pose fit + per-frame rows (dict of lists)."""
    rec = json.loads(Path(path).read_text())
    ep = rec["episode"]
    if str(ep) not in _W["r0"]:
        return None
    geo = _W["geo"]
    E = Episode(geo, ep, rec, _W["F"].states(ep), _W["r0"][str(ep)], _W["lag"], _W["geo_fit"])
    kind0 = E.kind
    fit = E.fit()
    fit.update(kind=E.kind, pose=E.pose, cf=E.cf, kind_top=kind0)
    ok = fit["fit_iou_median"] >= 0.6 and fit["n_fit"] >= 8
    if not ok:  # detector-independent fallback: the brick between the fins at the close frame (FK fingertip)
        E.kind, E.pose = kind0, "up"
        E.size, E.hc = brick_dims(E.kind, E.pose)
        fit.update(x=float(E.xy_tip[0]), y=float(E.xy_tip[1]), yaw=E.yaw_grip, src="tip", kind=E.kind, pose="up")
    else:
        fit["src"] = "wrist"
    out = E.vis(fit["x"], fit["y"], fit["yaw"])
    bv, ba = clip_box(out["box_vis"]), out["box_amodal"]
    uv = out["uv"]
    sp = np.r_[np.nan, np.linalg.norm(np.diff(uv, axis=0), axis=1)]
    sp[1:][np.diff(E.frames) != 1] = np.nan
    fz = fin_zones(geo, E.J)
    rows = {k: [] for k in COLS + BOXES}
    for k, fr in enumerate(E.frames):
        m, io = E.match(bv[k], k)
        vals = dict(
            ep=ep,
            frame=fr,
            tt=E.cf - fr,
            grip=E.J[k, 5],
            u=uv[k, 0],
            v=uv[k, 1],
            speed=sp[k],
            matched=m is not None,
            conf=m[4] if m is not None else 0.0,
            iou=io,
            n_dets=int((E.dets[k][:, 4] >= 0.5).sum()),
            fit_ok=ok,
            box_vis=bv[k],
            box_amodal=ba[k],
            det_box=m[:4] if m is not None else np.full(4, np.nan),
            fin_fixed=fz[k, 0],
            fin_jaw=fz[k, 1],
            **{key: out[key][k] for key in ("vis", "occ_grip", "oof", "area_px", "dist", "ang")},
        )
        for key in rows:
            rows[key].append(vals[key])
    return ep, fit, rows


COLS = [
    "ep",
    "frame",
    "tt",
    "grip",
    "vis",
    "occ_grip",
    "oof",
    "area_px",
    "dist",
    "ang",
    "u",
    "v",
    "speed",
    "matched",
    "conf",
    "iou",
    "n_dets",
    "fit_ok",
]
BOXES = ["box_vis", "box_amodal", "det_box", "fin_fixed", "fin_jaw"]


def main():
    import argparse
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--lag", type=float, default=0.0, help="video frame f shows the arm at state f - lag")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    files = sorted((OUT / "real_dets").glob("ep*.json"))
    if a.episodes:
        files = [f for f in files if int(f.stem[2:]) in a.episodes]
    cols = {k: [] for k in COLS + BOXES}
    fits = {}
    with Pool(a.workers, initializer=_init, initargs=(a.lag,)) as pool:
        for res in pool.imap(process, [str(f) for f in files]):
            if res is None:
                continue
            ep, fit, rows = res
            fits[ep] = fit
            for k in cols:
                cols[k] += rows[k]
            print(ep, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fit.items()}), flush=True)
    tag = a.tag or ("" if not a.episodes else "_test")
    np.savez_compressed(OUT / f"real_table{tag}.npz", **{k: np.array(v) for k, v in cols.items()})
    (OUT / f"real_fit{tag}.json").write_text(json.dumps({"lag": a.lag, "episodes": fits}, indent=1))
    ok = [v for v in fits.values() if v["fit_iou_median"] >= 0.6 and v["n_fit"] >= 8]
    print(
        f"lag {a.lag}: fits ok {len(ok)} / {len(fits)}; median fit IoU "
        f"{np.median([v['fit_iou_median'] for v in fits.values()]):.3f}, cost "
        f"{np.median([v['cost'] for v in fits.values()]):.3f}; median d_tip "
        f"{np.median([v['d_tip_mm'] for v in ok]) if ok else float('nan'):.1f} mm",
        flush=True,
    )


if __name__ == "__main__":
    main()
