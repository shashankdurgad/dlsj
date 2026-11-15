"""Ceiling-light directions from the specular highlights the real top camera sees (plain Python).

    python sim/twin/visuals/fit/estimate_lights.py

The plate floors and the table are (nearly) horizontal mirrors of low quality: a highlight at table point P seen from
the camera at C means a light in the mirrored view direction l = (-v_x, -v_y, v_z), v = (C - P) / |C - P|.
* plates: over the evaluation set's start frames, every plate-floor pixel votes its mirror direction into a
  "visited" map and, if it is a highlight (much brighter and less saturated than the plate), into a "hit" map; lights
  are the peaks of hit / visited (plates sit at different places in different episodes, so the maps fill in).
* table: the bright blotches of the top-camera median (sheen of the worn wood) give the same directions, broadened by
  the wood's roughness (cross-check).
Writes sim/twin/out/visual_fit/lights_estimate.json (+ lights_estimate.png: hit ratio over (l_x / l_z, l_y / l_z)).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/visual_fit"
TABLE_Z = -0.002
RANGE, BINS = 1.6, 96  # tan(elevation complement) range of the direction map


def main():
    """Vote mirror directions of highlights, write the peaks."""
    from sim.twin.visuals.fit.make_evalset import top_camera

    r_cv, pos, K = top_camera()
    meta = json.loads((OUT / "evalset.json").read_text())
    top = np.load(OUT / "evalset.npz")["top"]
    v, u = np.mgrid[:480, :640]
    d = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1) @ r_cv
    hit, vis = np.zeros((BINS, BINS)), np.zeros((BINS, BINS))

    def mirror(P):
        w = pos - P
        w /= np.linalg.norm(w, axis=-1, keepdims=True)
        return np.stack([-w[..., 0] / w[..., 2], -w[..., 1] / w[..., 2]], -1)

    def vote(acc, m):
        h, _, _ = np.histogram2d(m[:, 0], m[:, 1], BINS, [[-RANGE, RANGE], [-RANGE, RANGE]])
        acc += h

    n = 0
    for s in meta["samples"]:
        if s["moment"] != "start":
            continue
        img = top[s["i"]]
        lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        z = TABLE_Z + 0.003
        P = pos + ((z - pos[2]) / d[..., 2])[..., None] * d
        for _name, (x, y, diam) in s["plates"].items():
            m = np.hypot(P[..., 0] - x, P[..., 1] - y) < 0.62 * diam / 2
            if m.sum() < 800:
                continue
            L0, s0 = np.median(lab[..., 0][m]), np.median(hsv[..., 1][m])
            if s0 < 60:  # not a plate (misdetection)
                continue
            ok = m & (hsv[..., 1] > 0.5 * s0) & (np.abs(lab[..., 0] - L0) < 14)  # unoccluded plate floor
            hl = m & (lab[..., 0] > L0 + 22) & (hsv[..., 1] < 0.6 * s0)
            vote(vis, mirror(P[ok | hl]))
            vote(hit, mirror(P[hl]))
            n += 1
    ratio = cv2.GaussianBlur(hit, (0, 0), 1.0) / np.maximum(cv2.GaussianBlur(vis, (0, 0), 1.0), 30.0)
    ratio[cv2.GaussianBlur(vis, (0, 0), 1.0) < 30] = 0
    peaks, r = [], ratio.copy()
    for _ in range(8):
        i, j = np.unravel_index(np.argmax(r), r.shape)
        if r[i, j] < 0.03:
            break
        tx, ty = (-RANGE + (i + 0.5) * 2 * RANGE / BINS), (-RANGE + (j + 0.5) * 2 * RANGE / BINS)
        vec = np.array([tx, ty, 1.0])
        vec /= np.linalg.norm(vec)
        peaks.append(dict(dir=vec.round(4).tolist(), strength=round(float(r[i, j]), 3),
                          elevation_deg=round(float(np.degrees(np.arcsin(vec[2]))), 1),
                          azimuth_deg=round(float(np.degrees(np.arctan2(vec[1], vec[0]))), 1)))
        cv2.circle(r, (int(j), int(i)), 7, 0.0, -1)
    # table sheen: bright low-saturation blotches of the top median
    med = cv2.imread(str(REPO / "sim/twin/out/visuals/top_median.png"))[..., ::-1]
    g = cv2.cvtColor(med, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = cv2.GaussianBlur(g, (0, 0), 12) - cv2.GaussianBlur(g, (0, 0), 70)
    hsvm = cv2.cvtColor(med, cv2.COLOR_RGB2HSV)
    wood = (hsvm[..., 1] < 90) & (hsvm[..., 2] > 80)
    hp[~wood] = 0
    Pt = pos + ((TABLE_Z - pos[2]) / d[..., 2])[..., None] * d
    table = []
    for _ in range(3):
        i, j = np.unravel_index(np.argmax(hp), hp.shape)
        if hp[i, j] < 5:
            break
        t = mirror(Pt[i, j][None])[0]
        vec = np.array([t[0], t[1], 1.0])
        vec /= np.linalg.norm(vec)
        table.append(dict(px=[int(j), int(i)], table_xy=Pt[i, j, :2].round(3).tolist(), dir=vec.round(4).tolist(),
                          contrast=round(float(hp[i, j]), 1),
                          elevation_deg=round(float(np.degrees(np.arcsin(vec[2]))), 1),
                          azimuth_deg=round(float(np.degrees(np.arctan2(vec[1], vec[0]))), 1)))
        cv2.circle(hp, (int(j), int(i)), 70, 0.0, -1)
    res = dict(plates_used=n, plate_peaks=peaks, table_sheen=table,
               note="unit directions TOWARDS the light, env frame; strength = fraction of mirror-aligned plate-floor "
                    "pixels that were highlights")
    (OUT / "lights_estimate.json").write_text(json.dumps(res, indent=1))
    viz = cv2.resize((np.clip(ratio / max(ratio.max(), 1e-6), 0, 1) * 255).astype(np.uint8), (480, 480),
                     interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / "lights_estimate.png"), viz)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
