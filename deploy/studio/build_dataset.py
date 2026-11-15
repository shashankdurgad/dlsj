"""Physical AI Studio dataset for a fine-tuned policy: the training snapshot's labels on the original recordings.

    python deploy/studio/build_dataset.py LABELS SOURCE OUT [--full-frame observation.images.top]

LABELS  the prepared training snapshot (only meta/ and data/ are read; its videos are not needed),
        e.g. molmoact2-cal2-456-3stage-20261114 (per-frame 3-stage tasks, explicit train/validation split).
SOURCE  the recording that snapshot was built from (LeRobot v3), e.g. duplo_teleop_cal2 on the XPS.
OUT     a new LeRobot v3 dataset for `physicalai-studio datasets import-dir`:
          - data/ and meta/ from LABELS (the exact task strings and episode set the policy was trained on),
          - the cameras LABELS keeps, with their video files hard-linked from SOURCE (no copy on the same disk),
          - LABELS' MANIFEST / curation / normalisation JSONs under meta/training/ for lineage.

Cameras that were cropped for training (`--full-frame`, MolmoAct2: top cropped to 560x480) stay full-frame here:
the crop is part of the model's preprocessing (the policy contract `top_crop` and the run-time `--crop`), so their
feature shape and image statistics come from SOURCE. Everything else must match SOURCE exactly, and is checked:
episode count, lengths and indices, every per-episode video reference, and every data column except task_index.
SOURCE is never modified.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LINEAGE = ("MANIFEST.json", "SOURCE-CURATION.json", "TRAIN-NORMALIZATION.json", "READY.json", "SHA256SUMS.txt")


def _parquets(root: Path, sub: str) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(root / sub / "*" / "*.parquet")))]


def _episodes(root: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in _parquets(root, "meta/episodes")], ignore_index=True)


def video_keys(info: dict) -> list[str]:
    return [k for k, f in info["features"].items() if f.get("dtype") == "video"]


def check_alignment(labels: Path, source: Path, cams: list[str]) -> None:
    """Raise if LABELS is not a relabelling of SOURCE (same episodes, frames, video refs and data)."""
    li, si = (json.loads((r / "meta/info.json").read_text()) for r in (labels, source))
    for k in ("total_episodes", "total_frames", "fps"):
        if li[k] != si[k]:
            raise ValueError(f"{k}: labels {li[k]} != source {si[k]}")
    missing = [c for c in cams if c not in si["features"]]
    if missing:
        raise ValueError(f"source has no {missing}")
    le, se = _episodes(labels), _episodes(source)
    cols = ["episode_index", "length", "dataset_from_index", "dataset_to_index"]
    for c in cams:
        cols += [f"videos/{c}/{x}" for x in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    for c in cols:
        if not np.array_equal(le[c].to_numpy(), se[c].to_numpy()):
            raise ValueError(f"meta/episodes column {c} differs between labels and source")
    lp, sp = _parquets(labels, "data"), _parquets(source, "data")
    if [p.relative_to(labels) for p in lp] != [p.relative_to(source) for p in sp]:
        raise ValueError("data/ file layout differs between labels and source")
    for a, b in zip(lp, sp):
        da, db = pd.read_parquet(a), pd.read_parquet(b)
        if list(da.columns) != list(db.columns) or len(da) != len(db):
            raise ValueError(f"{a.name}: columns/rows differ")
        for c in da.columns:
            if c == "task_index":
                continue
            x, y = np.stack(da[c].to_numpy()), np.stack(db[c].to_numpy())
            if not np.array_equal(x, y):
                raise ValueError(f"{a.relative_to(labels)}: column {c} differs")


def link_tree(src: Path, dst: Path) -> int:
    """Hard-link every file of src into dst (copy if the link fails, e.g. across disks). Returns the file count."""
    n = 0
    for p in sorted(src.rglob("*")):
        if p.is_dir():
            continue
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(p, q)
        except OSError:
            shutil.copy2(p, q)
        n += 1
    return n


def build(labels: Path, source: Path, out: Path, full_frame: list[str]) -> dict:
    if out.exists():
        raise FileExistsError(out)
    info = json.loads((labels / "meta/info.json").read_text())
    src_info = json.loads((source / "meta/info.json").read_text())
    cams = video_keys(info)
    check_alignment(labels, source, cams)

    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(labels / "meta", tmp / "meta")
    shutil.copytree(labels / "data", tmp / "data")
    (tmp / "meta/training").mkdir()
    for name in LINEAGE:
        if (labels / name).exists():
            shutil.copy2(labels / name, tmp / "meta/training" / name)

    stats = json.loads((tmp / "meta/stats.json").read_text())
    src_stats = json.loads((source / "meta/stats.json").read_text())
    src_eps = _episodes(source).set_index("episode_index")
    for cam in full_frame:
        if cam not in cams:
            raise ValueError(f"--full-frame {cam}: not a camera of the labels ({cams})")
        info["features"][cam] = src_info["features"][cam]
        stats[cam] = src_stats[cam]
        for p in _parquets(tmp, "meta/episodes"):
            df = pd.read_parquet(p)
            for c in [c for c in df.columns if c.startswith(f"stats/{cam}/")]:
                df[c] = src_eps.loc[df["episode_index"], c].to_list()
            df.to_parquet(p, index=False)
    (tmp / "meta/info.json").write_text(json.dumps(info, indent=4))
    (tmp / "meta/stats.json").write_text(json.dumps(stats, indent=4))

    files = {cam: link_tree(source / "videos" / cam, tmp / "videos" / cam) for cam in cams}
    tasks = pd.read_parquet(tmp / "meta/tasks.parquet")
    tmp.rename(out)
    return {"out": str(out), "episodes": info["total_episodes"], "frames": info["total_frames"], "cameras": cams,
            "full_frame": full_frame, "video_files": files, "tasks": sorted(tasks.index.tolist())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", type=Path)
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--full-frame", action="append", default=[],
                    help="camera cropped at training, kept full-frame here (repeatable)")
    a = ap.parse_args()
    try:
        print(json.dumps(build(a.labels.expanduser(), a.source.expanduser(), a.out.expanduser(), a.full_frame),
                         indent=2))
    except (ValueError, FileExistsError) as e:
        sys.exit(f"build_dataset: {e}")


if __name__ == "__main__":
    main()
