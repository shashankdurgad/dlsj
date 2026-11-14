"""Build the MolmoAct2 training snapshot from a sealed teleop recording (e.g. duplo_teleop_cal2, 100 demos).

    python prepare_data.py <recording dir> <output dir> --calibration <follower.json>

Run in an environment with LeRobot 0.7 and ffmpeg (the SmolVLA scratch venv works). The source is never modified.

The output is one LeRobot v3 dataset that Studio trains on directly:
  - cameras: wrist + top only (inspect and both depth streams removed);
  - top cropped to x 0-560 (560x480), as in the 517-episode run, so deploy with
    `--crop observation.images.top=0,0,560,480@640x480`;
  - meta/dlsj/labels.jsonl copied unchanged (pick/place instruction split at inspection_frame);
  - TRAIN-NORMALIZATION.json: state/action mean/std/min/max/q01/q99 over the TRAIN episodes only, using the exact
    seed-42 episode split Studio's LeRobotDataModule makes (val_split=0.1);
  - calibration/follower.json: the arm calibration, for MolmoAct2's SO-101 frame conversion;
  - SHA256SUMS.txt, MANIFEST.json and READY.json (written last) for the GCS transfer.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

KEEP = ("observation.images.wrist", "observation.images.top")
TOP = "observation.images.top"
CROP = (0, 0, 560, 480)  # x, y, w, h
JOINTS = [f"{n}.pos" for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
PICK = "pick up the brick and hold it in front of the inspection camera"
PLACE_PREFIX = "place the brick in the "
VAL_SPLIT, VAL_SEED = 0.1, 42


def studio_split(n_episodes, val_split=VAL_SPLIT, seed=VAL_SEED):
    """Studio LeRobotDataModule's episode split (pinned 8f13241): sorted(Random(seed).sample(all, n_val))."""
    episodes = list(range(n_episodes))
    n_val = max(1, int(len(episodes) * val_split))
    val = sorted(random.Random(seed).sample(episodes, n_val))
    return [e for e in episodes if e not in set(val)], val


def check_source(info, labels, calibration):
    if not str(info.get("codebase_version", "")).startswith("v3.") or info.get("fps") != 30:
        raise ValueError("expected a 30 fps LeRobot v3 recording")
    feats = info["features"]
    for key in KEEP:
        if key not in feats:
            raise ValueError(f"missing {key}")
    if feats[TOP]["shape"][:2] != [480, 640]:
        raise ValueError(f"{TOP} must be the uncropped 640x480 recording, found {feats[TOP]['shape']}")
    for key in ("observation.state", "action"):
        if feats[key]["shape"] != [6] or feats[key].get("names") != JOINTS:
            raise ValueError(f"{key}: expected six joints in recorder order")
    if sorted(label["episode_index"] for label in labels) != list(range(info["total_episodes"])):
        raise ValueError("need exactly one label per episode")
    bad = [l["episode_index"] for l in labels
           if l.get("success") is not True or l.get("pick_task") != PICK
           or not str(l.get("place_task", "")).startswith(PLACE_PREFIX)
           or not isinstance(l.get("inspection_frame"), int) or not 0 < l["inspection_frame"] < l["n_frames"]]
    if bad:
        raise ValueError(f"episodes with failed/unlabelled takes or bad boundaries: {bad[:10]}")
    if sum(l["n_frames"] for l in labels) != info["total_frames"]:
        raise ValueError("label frame counts disagree with info.json")
    missing = [j for j in (n.removesuffix(".pos") for n in JOINTS) if j not in calibration]
    if missing or any(not {"range_min", "range_max"} <= set(calibration[j]) for j in calibration if j in JOINTS):
        raise ValueError(f"calibration is missing joints/ranges: {missing}")


def drop_stale_episode_columns(root, features):
    """LeRobot 0.7.0 remove_feature leaves `videos/<f>/*` and `stats/<f>/*` columns in meta/episodes."""
    import pyarrow.parquet as pq

    prefixes = tuple(p for f in features for p in (f"videos/{f}/", f"stats/{f}/"))
    for path in sorted((Path(root) / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        stale = [c for c in table.column_names if c.startswith(prefixes)]
        if stale:
            pq.write_table(table.drop_columns(stale), path)


def frame_count(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                         check=True)
    return int(out.stdout.strip())


def crop_top(root, workers=8):
    """Crop every top-camera video file to CROP (same ffmpeg settings as scripts/prepare_train_set.py)."""
    x, y, w, h = CROP
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    feat = info["features"][TOP]
    gop = feat.get("info", {}).get("video.g") or 2
    files = sorted((root / "videos" / TOP).glob("*/*.mp4"))
    before = {f: frame_count(f) for f in files}

    def one(path):
        tmp = path.with_suffix(".crop.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path), "-vf",
                        f"crop={w}:{h}:{x}:{y}", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-g", str(gop),
                        "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-an", str(tmp)], check=True)
        tmp.replace(path)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, files))
    if {f: frame_count(f) for f in files} != before:
        raise RuntimeError("cropping changed frame counts")
    feat["shape"] = [h, w, feat["shape"][2]]
    feat.setdefault("info", {}).update({"video.height": h, "video.width": w, "video.codec": "h264",
                                        "video.pix_fmt": "yuv420p"})
    info_path.write_text(json.dumps(info, indent=4))
    return len(files)


def train_normalization(root, train_ids):
    """Exact per-dimension stats over the train episodes (q01/q99 by np.quantile, not merged estimates)."""
    import pyarrow.parquet as pq

    cols = {"observation.state": [], "action": []}
    train = set(train_ids)
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", *cols])
        keep = np.isin(table.column("episode_index").to_numpy(), list(train))
        for key in cols:
            cols[key].append(np.stack(table.column(key).to_numpy(zero_copy_only=False))[keep])
    out = {}
    for key, parts in cols.items():
        x = np.concatenate(parts).astype(np.float64)
        out[key] = {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "min": x.min(0).tolist(),
                    "max": x.max(0).tolist(), "q01": np.quantile(x, 0.01, axis=0).tolist(),
                    "q99": np.quantile(x, 0.99, axis=0).tolist(), "count": [int(len(x))]}
    out["train_episodes"] = list(train_ids)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="recording root (meta/, data/, videos/)")
    p.add_argument("output", type=Path)
    p.add_argument("--calibration", type=Path, required=True, help="follower calibration JSON used for recording")
    args = p.parse_args()
    src, out = args.source.resolve(), args.output.resolve()
    if out.exists():
        p.error(f"{out} exists; choose a new output directory")
    info = json.loads((src / "meta/info.json").read_text())
    labels = [json.loads(l) for l in (src / "meta/dlsj/labels.jsonl").read_text().splitlines() if l.strip()]
    calibration = json.loads(args.calibration.read_text())
    check_source(info, labels, calibration)
    drop = sorted(k for k in info["features"] if k.startswith("observation.images.") and k not in KEEP)

    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import remove_feature

    remove_feature(LeRobotDataset("local/source", root=src), drop, output_dir=out, repo_id="local/molmoact2")
    drop_stale_episode_columns(out, drop)
    n_videos = crop_top(out)
    (out / "meta/dlsj").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "meta/dlsj/labels.jsonl", out / "meta/dlsj/labels.jsonl")
    (out / "calibration").mkdir()
    shutil.copyfile(args.calibration, out / "calibration/follower.json")

    train_ids, val_ids = studio_split(info["total_episodes"])
    (out / "TRAIN-NORMALIZATION.json").write_text(json.dumps(train_normalization(out, train_ids), indent=2) + "\n")

    # Reload as LeRobot sees it and check shapes on a real frame.
    ds = LeRobotDataset("local/molmoact2", root=out)
    sample = ds[0]
    shapes = {k: list(sample[k].shape) for k in KEEP}
    if shapes[TOP] != [3, CROP[3], CROP[2]] or ds.meta.total_frames != info["total_frames"]:
        raise RuntimeError(f"unexpected result: {shapes}, {ds.meta.total_frames} frames")
    place = {}
    for label in labels:
        place[label["place_task"]] = place.get(label["place_task"], 0) + 1
    manifest = {"source": str(src), "source_info_sha256": sha256(src / "meta/info.json"),
                "episodes": info["total_episodes"], "frames": info["total_frames"], "cameras": list(KEEP),
                "removed": drop, "top_crop": "0,0,560,480@640x480", "top_files_cropped": n_videos,
                "first_frame_shapes": shapes, "studio_split": {"seed": VAL_SEED, "val_split": VAL_SPLIT,
                                                               "train": train_ids, "validation": val_ids},
                "pick_task": PICK, "place_tasks": place, "calibration_sha256": sha256(args.calibration)}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = sorted(f for f in out.rglob("*") if f.is_file() and f.name not in ("SHA256SUMS.txt", "READY.json"))
    (out / "SHA256SUMS.txt").write_text("".join(f"{sha256(f)}  {f.relative_to(out)}\n" for f in files))
    (out / "READY.json").write_text(json.dumps({"files": len(files),
                                                "sha256_manifest": sha256(out / "SHA256SUMS.txt")}) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "studio_split"}, indent=2))
    print(f"train {len(train_ids)} / validation {len(val_ids)} episodes (Studio seed-42 split): val={val_ids}")


if __name__ == "__main__":
    main()
