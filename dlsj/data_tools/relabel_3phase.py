"""Relabel a duplo teleop dataset (LeRobot v3) into the 3-phase canonical instructions, as a NEW dataset.

Phases (per episode, from the gripper signal and the operator's inspection mark):
  1  start -> grasp_frame          "grasp the brick"
  2  grasp_frame -> inspection     "lift the brick and hold it still in front of the inspection camera"
  3  inspection -> end             "place the brick in the {bowl} bowl and release it"   (bowl = corrected label)

grasp_frame = the frame the gripper closes on the brick: the first downward crossing of `thr` (deg) after the gripper
was open, such that it stays closed (<= `stay`) until the inspection mark. Checked against 123 operator marks in
duplo_teleop_cal2: consistently 21-32 frames before the operator's "pickup done" mark (which comes after the lift),
i.e. the closing moment. Episodes whose gripper never opened before inspection (brick held from frame 0) get no
phase-1 frames (grasp_frame = 0). The bowl applies meta/dlsj/plate_corrections.json; "uncertain" plates are excluded.

Output: <dst>/ with data/*.parquet rewritten (task_index), meta/tasks.parquet (5 tasks), meta/episodes/* (tasks
lists), info.json (total_tasks), stats.json copied, meta/dlsj/labels.jsonl copied with `grasp_frame`,
`grasp_marked_by="gripper_rule"`, `bowl_final`, `excluded`, and videos/ symlinked to the source (no 17 GB copy).
Joe's `phase_data.PhaseDataModule` reads grasp_frame/inspection_frame from that labels file unchanged.

    python -m dlsj.data_tools.relabel_3phase SRC DST [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

GRASP = "grasp the brick"
PRESENT = "lift the brick and hold it still in front of the inspection camera"
PLACE = "place the brick in the {bowl} bowl and release it"
BOWLS = ("yellow", "orange", "purple")
TASKS = [GRASP, PRESENT, *(PLACE.format(bowl=b) for b in BOWLS)]


def grasp_frame(gripper: np.ndarray, inspection: int | None, thr: float = 12.0, stay: float = 15.0,
                open_min: float = 15.0) -> int | None:
    """First close-on-brick frame before `inspection`; 0 if never opened (held from the start); None if unclear."""
    s = gripper[:inspection] if inspection else gripper
    if len(s) == 0:
        return None
    if s.max() < open_min:
        return 0
    for f in range(1, len(s)):
        if s[f - 1] > thr >= s[f] and (s[f:] <= stay).all():
            return f
    return None


def bowl_of(label: dict, corrections: dict) -> tuple[str | None, str | None]:
    """(bowl, exclusion reason)."""
    ep = str(label["episode_index"])
    if ep in corrections.get("uncertain", {}):
        return None, "uncertain plate (plate_corrections.json)"
    if ep in corrections.get("corrections", {}):
        return corrections["corrections"][ep]["actual"], None
    return label.get("bowl"), None


def relabel(src: Path, dst: Path, dry_run: bool = False) -> dict:
    labels = [json.loads(line) for line in (src / "meta/dlsj/labels.jsonl").read_text().splitlines() if line.strip()]
    labels = {int(lab["episode_index"]): lab for lab in labels}
    corr_path = src / "meta/dlsj/plate_corrections.json"
    corrections = json.loads(corr_path.read_text()) if corr_path.exists() else {}
    data_files = sorted(glob.glob(str(src / "data/**/*.parquet"), recursive=True))
    frames = {f: pd.read_parquet(f) for f in data_files}
    by_ep: dict[int, pd.DataFrame] = {}
    for df in frames.values():
        for ep, g in df.groupby("episode_index"):
            by_ep[int(ep)] = g.sort_values("frame_index")

    phase_of: dict[int, tuple[int, int, int]] = {}  # ep -> (grasp, inspection, place task index)
    fallback: dict[int, tuple[int, int, int]] = {}  # excluded episodes: best-effort labels, not for training
    out_labels, stats = [], Counter()
    for ep, g in sorted(by_ep.items()):
        lab = dict(labels.get(ep, {"episode_index": ep}))
        insp = lab.get("inspection_frame")
        bowl, why = bowl_of(lab, corrections)
        grip = np.stack(g["observation.state"].values)[:, 5]
        gf = grasp_frame(grip, insp)
        if why is None and insp is None:
            why = "no inspection_frame"
        if why is None and bowl not in BOWLS:
            why = f"unknown bowl {bowl!r}"
        if why is None and gf is None:
            why = "grasp not found from the gripper"
        if why is None and lab.get("success") is False:
            why = "recorded as unsuccessful"
        lab.update({"grasp_frame": gf, "grasp_marked_by": "gripper_rule", "bowl_final": bowl, "excluded": why,
                    "tasks_3phase": TASKS})
        out_labels.append(lab)
        if why:
            stats["excluded"] += 1
            # still write valid phase labels (recorded bowl) so a full-dataset load never sees an invalid task; the
            # episode stays listed as excluded in labels.jsonl / relabel_3phase.json for the training split
            rec = lab.get("bowl") if lab.get("bowl") in BOWLS else "purple"
            fallback[ep] = (gf if gf is not None else 0, insp if insp is not None else len(g), 2 + BOWLS.index(rec))
            continue
        phase_of[ep] = (gf, insp, 2 + BOWLS.index(bowl))
        stats["phase1_frames"] += gf
        stats["phase2_frames"] += max(0, insp - gf)
        stats["phase3_frames"] += max(0, len(g) - insp)
        stats["episodes"] += 1
        stats[f"bowl_{bowl}"] += 1

    summary = {"src": str(src), "dst": str(dst), "tasks": TASKS, **stats,
               "excluded_episodes": {lab["episode_index"]: lab["excluded"] for lab in out_labels if lab["excluded"]}}
    if dry_run:
        return summary

    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    (dst / "meta/dlsj").mkdir(parents=True)
    for name in ("info.json", "stats.json"):
        shutil.copy2(src / "meta" / name, dst / "meta" / name)
    for f in (src / "meta/dlsj").iterdir():
        if f.name != "labels.jsonl":
            shutil.copy2(f, dst / "meta/dlsj" / f.name)
    os.symlink((src / "videos").resolve(), dst / "videos")

    # per-frame task_index (excluded episodes keep their rows but get phase indices only if usable; drop them)
    for f, df in frames.items():
        df = df.copy()
        keep = df["episode_index"].astype(int).isin(phase_of.keys())
        ti = np.empty(len(df), dtype=np.int64)
        for i, (ep, fr) in enumerate(zip(df["episode_index"].astype(int), df["frame_index"].astype(int))):
            gf, insp, place = phase_of[ep] if ep in phase_of else fallback[ep]
            ti[i] = 0 if fr < gf else (1 if fr < insp else place)
        df["task_index"] = ti
        out = dst / Path(f).relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        # excluded episodes stay on disk (indices contiguous) with best-effort labels; skip them via the split
        # (summary["episodes_used"] / labels "excluded").
        df.to_parquet(out, index=False)
        summary.setdefault("rows_excluded", 0)
        summary["rows_excluded"] += int((~keep).sum())

    pd.DataFrame({"task_index": range(len(TASKS))}, index=pd.Index(TASKS, name="task")).to_parquet(
        dst / "meta/tasks.parquet")
    for f in sorted(glob.glob(str(src / "meta/episodes/**/*.parquet"), recursive=True)):
        e = pd.read_parquet(f)
        e["tasks"] = [
            [TASKS[0], TASKS[1], TASKS[(phase_of.get(int(ep)) or fallback[int(ep)])[2]]] for ep in e["episode_index"]
        ]
        out = dst / Path(f).relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        e.to_parquet(out, index=False)
    info = json.loads((src / "meta/info.json").read_text())
    info["total_tasks"] = len(TASKS)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    (dst / "meta/dlsj/labels.jsonl").write_text("".join(json.dumps(lab) + "\n" for lab in out_labels))
    summary["episodes_used"] = sorted(phase_of)
    (dst / "meta/dlsj/relabel_3phase.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    s = relabel(a.src.expanduser(), a.dst.expanduser(), a.dry_run)
    s.pop("episodes_used", None)
    ex = s.pop("excluded_episodes")
    print(json.dumps(s, indent=1))
    print(f"excluded {len(ex)}:", json.dumps(ex)[:1500])


if __name__ == "__main__":
    main()
