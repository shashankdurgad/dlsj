"""Build the SmolVLA training data from the curated snapshot: wrist + top only, train/validation split.

    python prepare_data.py <curated snapshot dir> <output dir>

Input: the 3-stage LeRobot v3 dataset (517 source episodes; wrist, top, inspect, inspect_depth, top_depth;
`top` already cropped to 560x480) with
its `CURATION.json` (exact train/validation episode ids; episodes 1, 10 and 464 excluded). The source is never modified.

Output (<output dir>):
  2cam/        copy with only wrist + top (inspect, inspect_depth, top_depth dropped)
  train/       the 465 training episodes, stats.json recomputed from these episodes only
  validation/  the 51 held-out episodes, carrying the TRAIN stats (same holdout as the pi0.5 run)
  manifest.json counts, task strings and source/output digests

Uses LeRobot's own dataset tools (remove_feature, split_dataset, recompute_stats).
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

CAMERAS = ("wrist", "top")
DROP = ("observation.images.inspect", "observation.images.inspect_depth", "observation.images.top_depth")
JOINTS = [f"{n}.pos" for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


def check_source(info, curation):
    """Validate the curated snapshot before copying anything."""
    if not str(info.get("codebase_version", "")).startswith("v3."):
        raise ValueError("expected a LeRobot v3 dataset")
    if info.get("fps") != 30:
        raise ValueError("expected 30 fps")
    feats = info["features"]
    cams = {k for k in feats if k.startswith("observation.images.")}
    expected = {f"observation.images.{c}" for c in CAMERAS} | set(DROP)
    if cams != expected:
        raise ValueError(f"expected cameras {sorted(expected)}, found {sorted(cams)}")
    if feats["observation.images.top"]["shape"][:2] != [480, 560]:
        raise ValueError("top camera must already be cropped to 560x480 (crop 0,0,560,480@640x480)")
    for key in ("observation.state", "action"):
        if feats[key]["shape"] != [6] or feats[key].get("names") != JOINTS:
            raise ValueError(f"{key}: expected six joints in recorder order")
    train, val = curation["train_episode_ids"], curation["validation_episode_ids"]
    if not train or not val or set(train) & set(val):
        raise ValueError("CURATION.json train/validation ids must be non-empty and disjoint")
    if max(train + val) >= info["total_episodes"]:
        raise ValueError("CURATION.json lists an episode the dataset does not have")
    return sorted(train), sorted(val)


def drop_stale_episode_columns(root, feature):
    """LeRobot 0.7.0's remove_feature keeps the removed camera's `videos/<f>/*` and `stats/<f>/*` columns in
    meta/episodes. split_dataset then fails aggregating them ("Shape of quantile 'min' must be (3,1,1)").
    Drop exactly those columns; nothing else changes."""
    import pyarrow.parquet as pq

    prefixes = (f"videos/{feature}/", f"stats/{feature}/")
    dropped = 0
    for path in sorted((Path(root) / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        stale = [c for c in table.column_names if c.startswith(prefixes)]
        if stale:
            pq.write_table(table.drop_columns(stale), path)
            dropped += len(stale)
    return dropped


def digest(root):
    h = hashlib.sha256()
    for name in ("meta/info.json", "meta/stats.json"):
        h.update((Path(root) / name).read_bytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--keep-2cam", action="store_true", help="keep the intermediate 2cam/ copy")
    args = p.parse_args()
    src, out = args.source.resolve(), args.output.resolve()
    if out.exists():
        p.error(f"{out} exists; choose a new output directory")
    info = json.loads((src / "meta/info.json").read_text())
    curation = json.loads((src / "CURATION.json").read_text())
    train_ids, val_ids = check_source(info, curation)

    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import recompute_stats, remove_feature, split_dataset

    out.mkdir(parents=True)
    source = LeRobotDataset("local/duplo517", root=src)
    two_cam = remove_feature(source, list(DROP), output_dir=out / "2cam", repo_id="local/duplo517_2cam")
    if sum(drop_stale_episode_columns(out / "2cam", f) for f in DROP):
        two_cam = LeRobotDataset("local/duplo517_2cam", root=out / "2cam")
    splits = split_dataset(two_cam, {"train": train_ids, "validation": val_ids}, output_dir=out)
    train, val = splits["train"], splits["validation"]
    recompute_stats(train)  # state/action from train episodes only; image stats are unused (IDENTITY)
    # Validation must be normalised exactly like training: give it the train stats.
    shutil.copyfile(out / "train/meta/stats.json", out / "validation/meta/stats.json")

    manifest = {"source": str(src), "source_digest": digest(src), "dropped_features": list(DROP),
                "cameras": [f"observation.images.{c}" for c in CAMERAS]}
    for name, ds, ids in (("train", train, train_ids), ("validation", val, val_ids)):
        meta = json.loads((out / name / "meta/info.json").read_text())
        cams = sorted(k for k in meta["features"] if k.startswith("observation.images."))
        if cams != sorted(manifest["cameras"]):
            raise RuntimeError(f"{name}: unexpected cameras {cams}")
        expected_frames = sum(source.meta.episodes[i]["length"] for i in ids)
        if meta["total_episodes"] != len(ids) or meta["total_frames"] != expected_frames:
            raise RuntimeError(f"{name}: {meta['total_episodes']} episodes / {meta['total_frames']} frames, "
                               f"expected {len(ids)} / {expected_frames}")
        sample = ds[0]
        shapes = {k: list(sample[k].shape) for k in manifest["cameras"]}
        manifest[name] = {"episodes": len(ids), "frames": expected_frames, "source_episode_ids": ids,
                          "tasks": sorted(ds.meta.tasks.index.tolist()), "first_frame_shapes": shapes,
                          "digest": digest(out / name)}
    if not args.keep_2cam:
        shutil.rmtree(out / "2cam")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("train", "validation")}, indent=2))
    for name in ("train", "validation"):
        m = manifest[name]
        print(f"{name}: {m['episodes']} episodes, {m['frames']} frames, tasks={m['tasks']}")


if __name__ == "__main__":
    main()
