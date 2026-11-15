"""Joint plate albedo over the three cameras' responses (plain Python, no Isaac).

    python sim/twin/visuals/fit/plate_albedo.py --sweep sim/twin/out/visual_fit/sweep_<tag>.json --look <name>
                                                [--camera-color sim/twin/visuals/camera_color.json]

One physical albedo per plate must reproduce what EACH camera recorded after that camera's response
(twin_scene.color_correct: linear 3x3 M + offset; the top camera has none). From one look of a sweep_looks.py run
(per plate colour and camera: linear-RGB mean of the real frames r_c and of the RAW render s_c with the current albedo
x0) the render is modelled as diffuse, s_c = D_c * x (D_c = s_c / x0 elementwise: the light that reaches the plate in
camera c's view), so camera c records
    y_c(x) = g_c * (M_c @ (D_c * x) + o_c)
with g_c a per-camera exposure gain (the real D415s ran auto exposure; brightness is the lighting / sensor agents'):
g_top = the real / sim luminance ratio of the TABLE in the top view (so the albedo keeps the real plate / table
contrast instead of absorbing the exposure), g_wrist / g_inspect free. Alternating least squares over x (per
plate) and g_wrist, g_inspect (shared by the plates). Prints, per plate, the albedo (sRGB as the top camera would
see it under the current light) and the per-camera residual dE76 of
the top-only and the joint solutions; writes sim/twin/out/visual_fit/plate_albedo_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
CAMS = ("top", "wrist", "inspect")


def lin(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb(x):
    x = np.clip(np.asarray(x, np.float64), 0, 1)
    return 255 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_W = _M @ np.ones(3)


def lab_of_lin(x):
    """Linear RGB (0-1) -> CIE Lab (D65)."""
    t = _M @ np.clip(np.asarray(x, np.float64), 1e-6, None) / _W
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--look", required=True)
    ap.add_argument("--camera-color", default=str(REPO / "sim/twin/visuals/camera_color.json"))
    a = ap.parse_args()
    sw = json.loads(Path(a.sweep).read_text())["looks"][a.look]
    ccm = json.loads(Path(a.camera_color).read_text())
    from sim.twin.visuals.twin_scene import TwinLook

    base = TwinLook()
    pal = dict(base.plate_rgb) | {k: tuple(v) for k, v in sw["look"].get("plate_rgb", {}).items()}
    resp = {}
    for c in CAMS:
        f = ccm.get(c) or {}
        resp[c] = (np.asarray(f.get("M_linear", np.eye(3)), float),
                   np.asarray(f.get("offset_linear", np.zeros(3)), float))
    data = {}  # plate -> cam -> (D, r)
    for plate, rgb in pal.items():
        x0 = lin(rgb)
        for c in CAMS:
            st = sw["metrics"][c]["plate_by_colour"].get(plate)
            if not st or st["n"] < 2:
                continue
            D = np.asarray(st["raw_lin"]) / np.maximum(x0, 1e-4)
            data.setdefault(plate, {})[c] = (D, np.asarray(st["real_lin"]))

    def solve_x(obs, g):
        A = np.concatenate([g[c] * resp[c][0] * D[None, :] for c, (D, _r) in obs.items()])
        b = np.concatenate([r - g[c] * resp[c][1] for c, (_D, r) in obs.items()])
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return np.clip(x, 1e-4, 0.95)

    def pred(x, c, D, g):
        M, o = resp[c]
        return g[c] * (M @ (D * x) + o)

    tab = sw["metrics"]["top"]["table"]
    g0 = float((((tab["real"][0] + 16) / 116) ** 3) / (((tab["sim"][0] + 16) / 116) ** 3))
    g = dict.fromkeys(CAMS, 1.0) | {"top": g0}
    for _ in range(30):
        xs = {p: solve_x(obs, g) for p, obs in data.items()}
        for c in ("wrist", "inspect"):
            num = den = 0.0
            for p, obs in data.items():
                if c in obs:
                    D, r = obs[c]
                    y = resp[c][0] @ (D * xs[p]) + resp[c][1]
                    num, den = num + float(y @ r), den + float(y @ y)
            g[c] = num / den if den > 0 else 1.0
    out = {"gains": {c: round(v, 3) for c, v in g.items()}, "plates": {}}
    print("per-camera exposure gains (top from the table):", out["gains"])
    for p, obs in data.items():
        x_top = solve_x({"top": obs["top"]}, g) if "top" in obs else None
        row = {"albedo_srgb_now": list(pal[p]), "joint_srgb": srgb(xs[p]).round(1).tolist(),
               "top_only_srgb": srgb(x_top).round(1).tolist() if x_top is not None else None, "dE": {}}
        for c, (D, r) in obs.items():
            lr = lab_of_lin(r)
            row["dE"][c] = dict(
                now=round(float(np.linalg.norm(lab_of_lin(pred(lin(pal[p]), c, D, g)) - lr)), 1),
                joint=round(float(np.linalg.norm(lab_of_lin(pred(xs[p], c, D, g)) - lr)), 1),
                top_only=round(float(np.linalg.norm(lab_of_lin(pred(x_top, c, D, g)) - lr)), 1)
                if x_top is not None else None,
                real_lab=lab_of_lin(r).round(1).tolist(), joint_lab=lab_of_lin(pred(xs[p], c, D, g)).round(1).tolist())
        out["plates"][p] = row
        print(p, json.dumps(row))
    tag = Path(a.sweep).stem.replace("sweep_", "")
    dst = REPO / f"sim/twin/out/visual_fit/plate_albedo_{tag}.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
