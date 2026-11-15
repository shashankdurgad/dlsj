"""Refit the top camera pose (6-DoF, factory intrinsics) so it is consistent with the twin's W9 joint offsets at
TABLE level, by minimising jointly:
  (a) the arm-silhouette loss of the joint fit (align/joint_fit/model.py: arm links + gripper body, fins and URDF
      jaws don't-care; px at 320x240), with the arm posed by the W9 offsets (held fixed);
  (b) the table-level reprojection error of the picked-brick <-> FK fin-ray tip pairs (corr.py): the brick pixel is
      back-projected to the brick-top plane z = 0.0172 m and compared with the tip's xy (robust, in cm x 4, ~ px at
      320x240 at the table's depth).
Held out: episodes[2::3] of the correspondences and the joint fit's own silhouette split (episodes[2::3]).

    sim/twin/run.sh sim/twin/align/top_refit/refit.py            -> sim/twin/out/top_refit/result.json, overlays
    sim/twin/run.sh sim/twin/align/top_refit/refit.py --write    -> also writes calib/cameras_fit.json["top"]
                                                                    (old file kept as cameras_fit_prev.json)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
TWIN = HERE.parents[1]
REPO = TWIN.parents[1]
for p in (REPO, HERE, HERE.parent, HERE.parent / "joint_fit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import model as M  # noqa: E402
from corr import BRICK_TOP_Z, FIN_TIP, w9_offsets  # noqa: E402
from fk import sim_q  # noqa: E402

OUT = TWIN / "out/top_refit"
CAMS_FIT = TWIN / "calib/cameras_fit.json"
TABLE_SCALE = 4.0  # table term in cm x 4 ~ px at 320x240 at the table (f 604 px, depth ~0.75 m, half resolution)
HUBER = 1.0  # cm


def cam_from_json(c):
    return dict(R=np.array(c["world_to_cv"], float), t=np.array(c["position"], float),
                **{k: float(c[k]) for k in ("fx", "fy", "cx", "cy")})


def perturb(cam, x):
    return {**cam, "R": Rotation.from_rotvec(x[:3]).as_matrix() @ cam["R"], "t": cam["t"] + x[3:6]}


def backproject(uv, cam, z):
    d = np.c_[(uv[:, 0] - cam["cx"]) / cam["fx"], (uv[:, 1] - cam["cy"]) / cam["fy"], np.ones(len(uv))] @ cam["R"]
    s = (z - cam["t"][2]) / d[:, 2]
    return cam["t"] + s[:, None] * d


def table_err_cm(cam, C):
    xy = backproject(C["uv"], cam, BRICK_TOP_Z)[:, :2]
    return 100 * np.linalg.norm(xy - C["tip"][:, :2], axis=1)


def huber(e, d=HUBER):
    return np.where(e < d, 0.5 * e * e / d, e - 0.5 * d)


class Sil:
    """Top-camera silhouette loss with the arm posed by fixed joint offsets (FK cached per frame)."""

    def __init__(self, robot, items, offsets):
        self.robot, self.items = robot, items
        self.T = [robot.arm.fk(sim_q(it["state"], offsets, M.GRIP_SCALE0, M.GRIP_BIAS0)) for it in items]

    def frame(self, k, cam, want_mask=False):
        it, T = self.items[k], self.T[k]
        sm, ig, uv = self.robot.render("arm", T, cam)
        if len(uv) < 50 or sm.sum() < 30:
            return None
        a = np.minimum(M.bilinear(it["dt"], uv), M.CAP).mean()
        band = cv2.dilate(sm, np.ones((31, 31), np.uint8)) > 0
        rb = (it["real"] > 0) & band & (ig == 0) & (it["dc"] == 0)
        if rb.sum() >= 30:
            b = np.minimum(cv2.distanceTransform((1 - sm).astype(np.uint8), cv2.DIST_L2, 3)[rb], M.CAP).mean()
        else:
            b = M.CAP
        iou = ((sm > 0) & rb).sum() / max(((sm > 0) | rb).sum(), 1)
        return (float(a + b), float(iou), sm) if want_mask else (float(a + b), float(iou))

    def loss(self, cam, idx=None):
        r = [self.frame(k, cam) for k in (range(len(self.items)) if idx is None else idx)]
        r = np.array([x for x in r if x is not None])
        if len(r) < 5:
            return 1e3, 0.0, len(r)
        return float(r[:, 0].mean()), float(r[:, 1].mean()), len(r)


def pose_delta(c0, c1):
    dR = Rotation.from_matrix(c1["R"] @ c0["R"].T)
    p_tab = np.array([[0.25, 0.0, BRICK_TOP_Z], [0.2, -0.15, BRICK_TOP_Z], [0.2, 0.15, BRICK_TOP_Z]])
    uv0 = M.project(p_tab, c0)[0]
    moved = backproject(uv0, c1, BRICK_TOP_Z) - p_tab  # where the old camera's pixels land under the new one
    return dict(rotation_deg=float(np.degrees(dR.magnitude())),
                rotation_rotvec_cam_deg=np.degrees(dR.as_rotvec()).round(3).tolist(),
                translation_mm=((c1["t"] - c0["t"]) * 1000).round(1).tolist(),
                translation_norm_mm=float(np.linalg.norm(c1["t"] - c0["t"]) * 1000),
                table_shift_cm=(moved[:, :2] * 100).round(2).tolist(), table_points=p_tab[:, :2].tolist())


def stats(e):
    return dict(median_cm=float(np.median(e)), p90_cm=float(np.percentile(e, 90)), mean_cm=float(e.mean()), n=len(e))


def load_corr(max_tip_dist):
    D = json.loads((OUT / "corr.json").read_text())
    rows = [r for r in D["rows"] if "skip" not in r and r["dist_old_cam"] < max_tip_dist]
    eps = sorted(r["ep"] for r in rows)
    held = set(eps[2::3])
    off = w9_offsets()
    assert np.allclose(D["fin_tip"], FIN_TIP)

    def pack(rs):
        from corr import fin_tip

        return dict(uv=np.array([r["uv"] for r in rs]), tip=np.array([fin_tip(r["state_close"], off) for r in rs]),
                    ep=[r["ep"] for r in rs], rows=rs)

    return pack([r for r in rows if r["ep"] not in held]), pack([r for r in rows if r["ep"] in held])


def overlays(cams, Cv, sil_v, path_tab, path_sil):
    from sim.twin.visuals.realdata import Dataset

    ds = Dataset()
    tiles = []
    for k in np.linspace(0, len(Cv["rows"]) - 1, 12).astype(int):
        r = Cv["rows"][k]
        img = np.ascontiguousarray(ds.frame(r["ep"], "top", r["brick_frame"]))
        cv2.circle(img, tuple(int(v) for v in r["uv"]), 6, (255, 255, 255), 2)
        for (_, cam), col in zip(cams.items(), ((255, 60, 60), (60, 255, 60)), strict=True):
            uv = M.project(Cv["tip"][k : k + 1] * [1, 1, 0] + [0, 0, BRICK_TOP_Z], cam)[0][0]
            cv2.drawMarker(img, (int(uv[0]), int(uv[1])), col, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(img, f"ep {r['ep']} {r['color']} {r['kind']}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (320, 240)))
    grid = np.concatenate([np.concatenate(tiles[i : i + 4], 1) for i in range(0, 12, 4)], 0)
    cv2.imwrite(str(path_tab), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    tiles = []
    for k in np.linspace(0, len(sil_v.items) - 1, 8).astype(int):
        it = sil_v.items[k]
        img = np.repeat((it["real"] * 120)[..., None], 3, 2).astype(np.uint8)
        for (_, cam), col in zip(cams.items(), ((255, 60, 60), (60, 255, 60)), strict=True):
            r = sil_v.frame(k, cam, want_mask=True)
            if r is None:
                continue
            cs, _ = cv2.findContours(r[2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, col, 1)
        cv2.putText(img, f"ep {it['ep']} fr {it['fr']}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(img)
    grid = np.concatenate([np.concatenate(tiles[i : i + 4], 1) for i in range(0, 8, 4)], 0)
    cv2.imwrite(str(path_sil), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def write_cameras_fit(res):
    old = json.loads(CAMS_FIT.read_text())
    shutil.copy2(CAMS_FIT, CAMS_FIT.with_name("cameras_fit_prev.json"))
    n = res["new_top"]
    new = dict(old)
    new["top"] = {**old["top"], "world_to_cv": n["world_to_cv"], "position": n["position"],
                  "method": "joint refit (align/top_refit): top-arm silhouettes + table-level brick <-> FK fin-ray tip "
                            "pairs, W9 joint offsets held fixed, factory intrinsics",
                  "metrics": res["metrics"], "n_fit_frames": res["n_sil_fit"], "n_held_out_frames": res["n_sil_held"],
                  "status": res["status"], "previous": "cameras_fit_prev.json"}
    tmp = CAMS_FIT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new, indent=1))
    os.replace(tmp, CAMS_FIT)
    print("WROTE", CAMS_FIT, "(previous -> cameras_fit_prev.json)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, nargs="+", default=[1.0])
    ap.add_argument("--max-tip-dist", type=float, default=0.10)
    ap.add_argument("--n-opt", type=int, default=80, help="silhouette frames used inside the optimiser")
    ap.add_argument("--maxfev", type=int, default=700)
    ap.add_argument("--write", action="store_true", help="write the result (from result.json) to cameras_fit.json")
    a = ap.parse_args()
    if a.write:
        write_cameras_fit(json.loads((OUT / "result.json").read_text()))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    Cf, Cv = load_corr(a.max_tip_dist)
    print(f"correspondences: fit {len(Cf['ep'])}, held-out {len(Cv['ep'])}", flush=True)
    items, _ = M.load_data()
    top = [it for it in items if it["cam"] == "top"]
    off = w9_offsets()
    rob = M.Robot()
    sil_f = Sil(rob, [it for it in top if not it["held"]], off)
    sil_v = Sil(rob, [it for it in top if it["held"]], off)
    opt_idx = list(range(0, len(sil_f.items), max(1, len(sil_f.items) // a.n_opt)))
    print(f"silhouette frames: fit {len(sil_f.items)} (optimiser {len(opt_idx)}), held-out {len(sil_v.items)}",
          flush=True)
    old = cam_from_json(json.loads(CAMS_FIT.read_text())["top"])
    joe = dict(M.BASE["top"])
    start = joe  # W9 was fitted against the joe_cal2 top pose: start the refit there

    def report(cam):
        return dict(table_fit=stats(table_err_cm(cam, Cf)), table_held=stats(table_err_cm(cam, Cv)),
                    sil_fit=sil_f.loss(cam)[:2], sil_held=sil_v.loss(cam)[:2])

    metrics = {"cameras_fit.json (current)": report(old), "joe_cal2 pose + factory": report(joe)}
    for k, v in metrics.items():
        print(f"[{k}] {json.dumps(v)}", flush=True)
    fits = {}
    for lam in a.lam:
        def f(x, lam=lam):
            cam = perturb(start, x)
            t = TABLE_SCALE * huber(table_err_cm(cam, Cf)).mean() if lam > 0 else 0.0
            s = sil_f.loss(cam, opt_idx)[0] if lam < 1e3 else 0.0
            return s + lam * t
        best, t0 = None, time.time()
        for x0 in (np.zeros(6),):
            r = minimize(f, x0, method="Powell", options={"xtol": 1e-4, "ftol": 1e-4, "maxfev": a.maxfev})
            print(f"[lam {lam}] start {np.round(x0, 3).tolist()} -> {r.fun:.3f} ({r.nfev} evals, "
                  f"{time.time() - t0:.0f} s)", flush=True)
            if best is None or r.fun < best.fun:
                best = r
        cam = perturb(start, best.x)
        fits[lam] = (cam, best.x)
        metrics[f"refit lam={lam}"] = report(cam)
        metrics[f"refit lam={lam}"]["pose_change_vs_current"] = pose_delta(old, cam)
        metrics[f"refit lam={lam}"]["pose_change_vs_joe_cal2"] = pose_delta(joe, cam)
        print(f"[refit lam={lam}] {json.dumps(metrics[f'refit lam={lam}'])}", flush=True)
    lam0 = a.lam[0]
    cam, x = fits[lam0]
    m_new, m_old = metrics[f"refit lam={lam0}"], metrics["cameras_fit.json (current)"]
    clean = m_new["table_held"]["median_cm"] <= 1.2 and m_new["sil_held"][0] <= m_old["sil_held"][0] + 0.05
    res = dict(lam=lam0, start="joe_cal2 top pose + factory intrinsics",
               x=x.tolist(), new_top=dict(world_to_cv=cam["R"].tolist(), position=cam["t"].tolist()),
               metrics=metrics, n_corr_fit=len(Cf["ep"]), n_corr_held=len(Cv["ep"]), held_eps=Cv["ep"],
               n_sil_fit=len(sil_f.items), n_sil_held=len(sil_v.items), clean=bool(clean),
               status=f"held-out table error {m_old['table_held']['median_cm']:.2f} -> "
                      f"{m_new['table_held']['median_cm']:.2f} cm median (p90 {m_new['table_held']['p90_cm']:.2f}); "
                      f"held-out top silhouette loss {m_old['sil_held'][0]:.2f} -> {m_new['sil_held'][0]:.2f}")
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    overlays({"old": old, "new": cam}, Cv, sil_v, OUT / "overlay_table.jpg", OUT / "overlay_sil.jpg")
    print(res["status"], "| clean:", clean, flush=True)
    print("REFIT_OK", flush=True)


if __name__ == "__main__":
    main()
