"""Evaluation set for the visual fit: real frames + recorded joint states at five moments of ~30 cal2 episodes.

    python sim/twin/visuals/fit/make_evalset.py [--n 30]        # plain python (PyAV); never import in Isaac

Moments per episode: start (frame 0), reach (60 % of the way to the grasp), grasp (gripper closed on the brick),
inspect (the operator's in-view mark), place (just before the gripper opens over the plate). Split 2:1 by episode
(every third episode is held out). Also stores the layout detected in the real frame-0 top image (plates, bricks:
measure_real.detect_layout) and the state at the gripper-close frame (to find the target brick in the sim).
Writes sim/twin/out/visual_fit/evalset.npz (frames uint8) and evalset.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/visual_fit"
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
CAMS = ("top", "wrist", "inspect")
MOMENTS = ("start", "reach", "grasp", "inspect", "place")


def top_camera():
    """(R world->cv, position, K) of the physical top camera (reference-cal2 pose, factory intrinsics)."""
    ref = json.loads((REPO / "sim/residual_env/assets/joe_cal2/reference-cal2.json").read_text())["camera"]
    k = json.loads((REPO / "sim/twin/calib/realsense_intrinsics.json").read_text())["top"]
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
    return np.array(ref["world_to_cv"]), np.array(ref["position"]), K


def moments(S, lab):
    """({moment: frame}, gripper-close frame) of an episode from its states and label, or None if unusable."""
    T, insp = len(S), lab.get("inspection_frame")
    if insp is None or insp >= T - 5:
        return None
    g = S[:insp, 5]
    gf = next((f for f in range(1, len(g)) if g[f - 1] > 12 >= g[f] and (g[f:] <= 15).all()), None)
    if gf is None:
        gf = lab.get("grasp_frame")
    if gf is None or gf < 10 or gf >= insp - 3:
        return None
    op = next((f for f in range(insp, T) if S[f, 5] > 25), None)
    place = (op - 4) if op is not None and op - 4 > insp + 5 else (insp + T) // 2
    return {"start": 0, "reach": int(0.6 * gf), "grasp": min(gf + 6, insp - 1), "inspect": int(insp),
            "place": int(place)}, int(gf)


def main():
    """Build the evaluation set."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    from sim.twin.align.frames import labels, states
    from sim.twin.visuals.measure_real import detect_layout, persistent_mask
    from sim.twin.visuals.realdata import Dataset

    OUT.mkdir(parents=True, exist_ok=True)
    ds, L = Dataset(str(DATA)), labels(str(DATA))
    cam = top_camera()
    ignore = persistent_mask()
    cand = [int(v) for v in np.linspace(2, len(ds) - 3, a.n).round()]
    samples, frames, used = [], {c: [] for c in CAMS}, []
    for k, ep0 in enumerate(cand):
        ep, m = ep0, None
        for ep in range(ep0, ep0 + 6):  # next usable episode
            lab = L.get(ep, {})
            if not lab.get("success", True) or lab.get("excluded"):
                continue
            S = states(ep, str(DATA))
            m = moments(S, lab)
            if m:
                break
        if not m:
            continue
        mom, gf = m
        idx = [mom[k_] for k_ in MOMENTS]
        fr = {c: ds.frames(ep, c, idx) for c in CAMS}
        lay = detect_layout(fr["top"][0], *cam, ignore=ignore)
        split = "held" if k % 3 == 2 else "fit"
        for j, name in enumerate(MOMENTS):
            samples.append(dict(i=len(samples), ep=int(ep), frame=int(idx[j]), moment=name, split=split,
                                state=[float(v) for v in S[idx[j]]], grasp_state=[float(v) for v in S[gf]],
                                bowl=lab.get("bowl_final", lab.get("bowl")),
                                plates={k_: [float(x) for x in v] for k_, v in lay["plates"].items()},
                                bricks=lay["bricks"]))
            for c in CAMS:
                frames[c].append(fr[c][j])
        used.append(ep)
        print(ep, split, mom, "plates", sorted(lay["plates"]), "bricks", len(lay["bricks"]), flush=True)
    np.savez_compressed(OUT / "evalset.npz", **{c: np.stack(v) for c, v in frames.items()})
    (OUT / "evalset.json").write_text(json.dumps(dict(episodes=used, samples=samples), indent=0))
    print("EVALSET_OK", len(used), "episodes", len(samples), "samples")


if __name__ == "__main__":
    main()
