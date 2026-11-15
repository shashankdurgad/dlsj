"""Fixed audit set: ~63 timesteps of 21 cal2 episodes (3 phases each, 9 per phase) + neighbours + a disjoint reference.

    python sim/twin/audit/make_set.py            # plain python (PyAV); never import in an Isaac process

Episodes: evenly spread over the 456 cal2 episodes, successful, with a clean grasp / inspection mark, and NOT one of
the visual-fit agent's evaluation episodes (sim/twin/visuals/fit/make_evalset.py: linspace(2, 453, 30) + up to 5), so
the audit is held out from the appearance fit. Phases (frame index within the episode):
    start 0 | reach 60 % of the way to the grasp | pregrasp 8 frames before the gripper closes | grasp 6 frames after
    | lift halfway between grasp and the inspection mark | inspect the operator's in-view mark | place 4 frames before
    the gripper opens over the plate.
Per sample also: the next recorded frame ("nbr": the real-vs-real floor of every paired metric) and the same phase of
a different, disjoint episode ("ref": real-vs-real distribution baseline and the wrong-scene ceiling).
Writes sim/twin/out/audit/set/{set.json, real.npz, nbr.npz, ref.npz} (frames uint8 480x640, cameras top/wrist/inspect).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.audit.common import CAMS, DATA, PHASES, SET, jdump

N_EP, PER_EP = 21, 3


def moments(S: np.ndarray, lab: dict):
    T, insp = len(S), lab.get("inspection_frame")
    if insp is None or insp >= T - 8:
        return None
    g = S[:insp, 5]
    gf = next((f for f in range(1, len(g)) if g[f - 1] > 12 >= g[f] and (g[f:] <= 15).all()), None)
    if gf is None:
        gf = lab.get("grasp_frame")
    if gf is None or gf < 20 or gf >= insp - 12:
        return None
    op = next((f for f in range(insp, T) if S[f, 5] > 25), None)
    place = (op - 4) if op is not None and op - 4 > insp + 5 else (insp + T) // 2
    grasp = min(gf + 6, insp - 2)
    reach = int(0.6 * gf)
    m = {"start": 0, "reach": reach, "pregrasp": max(gf - 8, reach + 1), "grasp": grasp,
         "lift": (grasp + insp) // 2, "inspect": int(insp), "place": int(min(place, T - 2))}
    return m, int(gf)


def main():
    from sim.twin.align import frames as F

    root = str(DATA)
    L = F.labels(root)
    tab = F.frames_table(root)
    n_total = int(tab.episode_index.max()) + 1
    banned = {int(v) + d for v in np.linspace(2, n_total - 3, 30).round() for d in range(6)}
    used: set[int] = set()

    def pick(start: int):
        for ep in list(range(start, n_total)) + list(range(start)):
            lab = L.get(ep, {})
            if ep in banned or ep in used or not lab.get("success", True) or lab.get("excluded"):
                continue
            S = F.states(ep, root)
            m = moments(S, lab)
            if m:
                used.add(ep)
                return ep, S, m[0], m[1], lab
        raise RuntimeError("no usable episode")

    centres = [int(v) for v in np.linspace(8, n_total - 14, N_EP).round()]
    main_eps = [pick(c) for c in centres]
    ref_eps = [pick(c + 10) for c in centres]
    samples = []
    frames = {k: {c: [] for c in CAMS} for k in ("real", "nbr", "ref")}
    for k, ((ep, S, mom, gf, lab), (rep, rS, rmom, _rgf, _rlab)) in enumerate(zip(main_eps, ref_eps, strict=True)):
        phases = [PHASES[(PER_EP * k + j) % len(PHASES)] for j in range(PER_EP)]
        idx = [mom[p] for p in phases]
        nbr = [i + 1 if i + 1 < len(S) else i - 1 for i in idx]
        ridx = [rmom[p] for p in phases]
        A = np.stack(tab[tab.episode_index == ep].sort_values("frame_index")["action"].to_numpy()).astype(np.float64)
        for c in CAMS:
            got = F.video_frames(ep, c, sorted(set(idx + nbr)), root)
            rgot = F.video_frames(rep, c, sorted(set(ridx)), root)
            for i, n, r in zip(idx, nbr, ridx, strict=True):
                frames["real"][c].append(got[i])
                frames["nbr"][c].append(got[n])
                frames["ref"][c].append(rgot[r])
        for p, i, n, r in zip(phases, idx, nbr, ridx, strict=True):
            fut = A[i:i + 30]
            fut = np.concatenate([fut, np.repeat(fut[-1:], 30 - len(fut), 0)]) if len(fut) < 30 else fut
            samples.append(dict(i=len(samples), ep=int(ep), frame=int(i), phase=p, n_frames=int(len(S)),
                                state=S[i].round(4).tolist(), nbr_frame=int(n), nbr_state=S[n].round(4).tolist(),
                                grasp_frame=int(gf), grasp_state=S[gf].round(4).tolist(),
                                inspection_frame=int(lab["inspection_frame"]), bowl=lab.get("bowl"),
                                pick_task=lab.get("pick_task"), place_task=lab.get("place_task"),
                                future_actions=fut.round(4).tolist(),
                                ref=dict(ep=int(rep), frame=int(r), state=rS[r].round(4).tolist())))
        print(f"ep {ep:3d} (ref {rep:3d}) phases {phases} frames {idx}", flush=True)
    SET.mkdir(parents=True, exist_ok=True)
    for k, d in frames.items():
        np.savez_compressed(SET / f"{k}.npz", **{c: np.stack(v) for c, v in d.items()})
    jdump(SET / "set.json", dict(dataset=DATA.name, episodes=[e[0] for e in main_eps],
                                 ref_episodes=[e[0] for e in ref_eps], phases=list(PHASES), samples=samples))
    print("SET_OK", len(samples), "samples", {p: sum(s["phase"] == p for s in samples) for p in PHASES})


if __name__ == "__main__":
    main()
