"""Presentation prep (plain python, the image venv; PyAV allowed here, never in Isaac):

    python sim/twin/presentation/prep.py candidates [--n 40]       # episode list + layouts -> out/presentation/work/
    python sim/twin/presentation/prep.py sheets --episodes 18 76     # real contact sheets (privacy / quality check)
    python sim/twin/presentation/prep.py real --episodes 18 76       # full-res real frames -> work/epNNN/real_<cam>.npy

Candidates: cal2 episodes with a clean gripper close before the inspection mark (the replay protocol of
env/validate_twin.py), their real first-frame layouts (scene/layout.py, cached), ranked by the number of distractors
on the table (cluttered scenes first) with a seeded shuffle inside equal counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim/residual_env"))
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
WORK = REPO / "sim/twin/out/presentation/work"
CAMS = ("top", "wrist", "inspect")


def close_frame(grip, insp, thr=12.0, stay=15.0, open_min=15.0):
    """Gripper-close frame before the inspection mark (residual validate.py --grasp_rule gripper)."""
    g = grip[:insp] if insp else grip
    if len(g) == 0 or g.max() < open_min:
        return None
    for f in range(1, len(g)):
        if g[f - 1] > thr >= g[f] and (g[f:] <= stay).all():
            return f
    return None


def candidates(n: int):
    from bases import load_episodes

    from sim.twin.scene.layout import layout_from_episode

    eps = load_episodes(DATA)
    ok = []
    for e, d in eps.items():
        if d["excluded"] or not d["inspection_frame"]:
            continue
        gf = close_frame(d["state"][:, 5], d["inspection_frame"])
        if gf and gf > 5:
            ok.append((e, gf, int(d["inspection_frame"]), len(d["action"]), d["bowl"]))
    rng = np.random.default_rng(0)
    pick = [ok[i] for i in rng.permutation(len(ok))[: 3 * n]]
    rows = []
    for e, gf, insp, T, bowl in pick:
        try:
            lay = layout_from_episode(e, DATA)
        except Exception as ex:  # noqa: BLE001
            print("layout failed", e, ex, flush=True)
            continue
        if not lay.get("target") or lay.get("bowl") not in (lay.get("plates") or {}):
            continue
        rows.append(dict(episode=int(e), grasp_frame=gf, inspection_frame=insp, T=T, bowl=bowl,
                         n_distractors=len(lay["distractors"]), target=lay["target"]["color"],
                         kind=lay["target"]["kind"], plates=sorted(lay["plates"])))
    rows.sort(key=lambda r: -r["n_distractors"])
    rows = rows[:n]
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "candidates.json").write_text(json.dumps(rows, indent=1))
    print("CANDIDATES", len(ok), "usable episodes;", [r["episode"] for r in rows], flush=True)


def sheets(episodes):
    import cv2

    from sim.twin.visuals.realdata import Dataset

    ds = Dataset(str(DATA))
    out = WORK / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    for e in episodes:
        T = len(ds.states(e))
        ts = np.linspace(0, T - 1, 6).astype(int).tolist()
        rows = []
        for c in CAMS:
            fr = ds.frames(e, c, ts)
            rows.append(np.concatenate([cv2.resize(f, (320, 240)) for f in fr], 1))
        img = np.concatenate(rows, 0)
        cv2.putText(img, f"ep {e} T={T} t={ts}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"real_ep{e:03d}.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
        print("SHEET", e, flush=True)


def real(episodes):
    from sim.twin.visuals.realdata import Dataset

    ds = Dataset(str(DATA))
    for e in episodes:
        d = WORK / f"ep{e:03d}"
        d.mkdir(parents=True, exist_ok=True)
        T = len(ds.states(e))
        for c in CAMS:
            np.save(d / f"real_{c}.npy", np.stack(ds.frames(e, c, range(T))))
        print("REAL_OK", e, T, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("candidates", "sheets", "real"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--episodes", type=int, nargs="*", default=[])
    a = ap.parse_args()
    if a.stage == "candidates":
        candidates(a.n)
    elif a.stage == "sheets":
        sheets(a.episodes)
    else:
        real(a.episodes)
