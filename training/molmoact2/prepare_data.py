"""Build a MolmoAct2 training snapshot from one of the team's 3-stage LeRobot v3 datasets.

    python prepare_data.py <3-stage dataset> <output dir> [--episodes TOP-250-SHORTLIST.json]
        [--calibration follower.json | --calibration-note "pre-2026-11-14 calibration ..."]

Needs only numpy, pyarrow and ffmpeg (runs on Spark 2). The source is never modified.

Sources are the team's 3-stage builds (labels with grasp_frame/inspection_frame; per-frame task_index already
rewritten to "pick up the <colour> brick" | "bring the brick to the inspection camera" | "place the brick in the
<colour> bowl"). The output is a LeRobot v3 dataset that Studio trains on directly:
  - cameras wrist + top only (inspect and both depth streams dropped from info/stats/episodes metadata);
  - top cropped to x 0-560 (560x480); deploy with `--crop observation.images.top=0,0,560,480@640x480`;
  - every source episode is kept, with its original index, so labels and audits still line up. MANIFEST.json
    lists the episodes to train on: the --episodes selection (all if omitted) minus the team's
    `exclude_from_training` episodes;
  - the train/validation split is the team's explicit one from the source CURATION.json (train_episode_ids /
    validation_episode_ids) restricted to those episodes, as the 431 README's loading contract requires. Only if
    the source has none does it fall back to Studio's seed-42 10% split. train.py hands both lists to Studio;
  - TRAIN-NORMALIZATION.json over the train episodes of that split only;
  - SHA256SUMS.txt, MANIFEST.json and READY.json (written last) for the GCS transfer.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

KEEP = ("observation.images.wrist", "observation.images.top")
TOP = "observation.images.top"
CROP = (0, 0, 560, 480)  # x, y, w, h
JOINTS = [f"{n}.pos" for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
INSPECT_TASK = "bring the brick to the inspection camera"
PICK_RE = re.compile(r"^pick up the( [a-z]+)? brick$")
PLACE_RE = re.compile(r"^place the brick in the (yellow|orange|purple) bowl$")
VAL_SPLIT, VAL_SEED = 0.1, 42
FFMPEG = shutil.which("ffmpeg") or str(Path.home() / ".local/bin/ffmpeg")


def studio_split(episodes, val_split=VAL_SPLIT, seed=VAL_SEED):
    """Studio LeRobotDataModule's split (pinned 8f13241): val = sorted(Random(seed).sample(episodes, n_val))."""
    episodes = list(range(episodes)) if isinstance(episodes, int) else list(episodes)
    n_val = max(1, int(len(episodes) * val_split))
    val = sorted(random.Random(seed).sample(episodes, n_val))
    return [e for e in episodes if e not in set(val)], val


def team_split(curation, episodes):
    """The source CURATION.json split restricted to `episodes`, or None if the source has no explicit split."""
    if not curation or "validation_episode_ids" not in curation:
        return None
    train_ids, val_ids = set(curation["train_episode_ids"]), set(curation["validation_episode_ids"])
    stray = [e for e in episodes if (e in train_ids) == (e in val_ids)]
    if stray:
        raise ValueError(f"episodes in neither or both CURATION splits: {stray[:10]}")
    train = [e for e in episodes if e in train_ids]
    val = [e for e in episodes if e in val_ids]
    if not train or not val:
        raise ValueError("CURATION split leaves no train or no validation episodes")
    return train, val


def load_selection(path):
    """A JSON list of episode ids, or a shortlist/curation JSON (episode_ids_sorted / included_episodes)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        for key in ("episode_ids_sorted", "included_episodes", "episodes"):
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError(f"{path}: no episode list found")
    return sorted({int(e) for e in data})


def select(labels, selection=None):
    """Episodes to train on: the selection (or all) minus the team's exclude_from_training episodes."""
    by_ep = {l["episode_index"]: l for l in labels}
    wanted = selection if selection is not None else sorted(by_ep)
    unknown = [e for e in wanted if e not in by_ep]
    if unknown:
        raise ValueError(f"selected episodes not in the dataset: {unknown[:10]}")
    excluded = {e: by_ep[e].get("exclude_reason") for e in wanted if by_ep[e].get("exclude_from_training")}
    return [e for e in wanted if e not in excluded], excluded


def check_source(info, labels, episodes):
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
    if sorted(l["episode_index"] for l in labels) != list(range(info["total_episodes"])):
        raise ValueError("need exactly one label per episode")
    if sum(l["n_frames"] for l in labels) != info["total_frames"]:
        raise ValueError("label frame counts disagree with info.json")
    by_ep = {l["episode_index"]: l for l in labels}
    bad = []
    for e in episodes:
        l = by_ep[e]
        g, i = l.get("grasp_frame"), l.get("inspection_frame")
        if (l.get("success") is not True or not isinstance(g, int) or not isinstance(i, int)
                or not 0 < g < i < l["n_frames"] or not PICK_RE.match(str(l.get("pick_task")))
                or l.get("inspect_task") != INSPECT_TASK or not PLACE_RE.match(str(l.get("place_task")))):
            bad.append(e)
    if bad:
        raise ValueError(f"selected episodes with failed takes, bad stage boundaries or task strings: {bad[:10]}")


def read_tasks(root):
    import pyarrow.parquet as pq

    table = pq.read_table(root / "meta/tasks.parquet").to_pydict()
    names = table.get("task") or table.get("__index_level_0__")
    return dict(zip(table["task_index"], names))


def check_stage_tasks(root, labels, episodes):
    """Per-frame task_index must follow pick (< grasp) / inspect (< inspection) / place for every used episode."""
    import pyarrow.parquet as pq

    tasks, by_ep, want = read_tasks(root), {l["episode_index"]: l for l in labels}, set(episodes)
    seen, wrong = set(), []
    for path in sorted((root / "data").rglob("*.parquet")):
        t = pq.read_table(path, columns=["episode_index", "frame_index", "task_index"])
        eps, frames, idx = (t.column(c).to_numpy() for c in ("episode_index", "frame_index", "task_index"))
        for e in np.unique(eps):
            if int(e) not in want:
                continue
            l, m = by_ep[int(e)], eps == e
            f = frames[m]
            expected = np.where(f < l["grasp_frame"], 0, np.where(f < l["inspection_frame"], 1, 2))
            actual = [tasks[int(k)] for k in idx[m]]
            stage_task = (l["pick_task"], l["inspect_task"], l["place_task"])
            if any(a != stage_task[s] for a, s in zip(actual, expected)):
                wrong.append(int(e))
            seen.add(int(e))
    if wrong or seen != want:
        raise ValueError(f"task_index does not follow the 3-stage labels: {wrong[:10]}, missing {sorted(want - seen)[:10]}")


def drop_features(root, info, drop):
    """What LeRobot's remove_feature does for video features: data parquet holds none, so only metadata changes."""
    import pyarrow.parquet as pq

    for key in drop:
        info["features"].pop(key)
    if "total_videos" in info:
        info["total_videos"] = sum(1 for _ in (root / "videos").rglob("*.mp4"))
    stats_path = root / "meta/stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats_path.write_text(json.dumps({k: v for k, v in stats.items() if k not in drop}, indent=4))
    prefixes = tuple(p for f in drop for p in (f"videos/{f}/", f"stats/{f}/"))
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        stale = [c for c in table.column_names if c.startswith(prefixes)]
        if stale:
            pq.write_table(table.drop_columns(stale), path)


def frame_count(path):
    """Video packets (= frames) of the first video stream, without decoding (one framecrc line per packet)."""
    out = subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy",
                          "-f", "framecrc", "-"], capture_output=True, text=True, check=True)
    return sum(1 for line in out.stdout.splitlines() if line and not line.startswith("#"))


def crop_top(root, info, workers=8):
    """Crop every top-camera file to CROP (same ffmpeg settings as scripts/prepare_train_set.py)."""
    x, y, w, h = CROP
    feat = info["features"][TOP]
    gop = feat.get("info", {}).get("video.g") or 2
    files = sorted((root / "videos" / TOP).glob("*/*.mp4"))

    def one(path):
        before = frame_count(path)
        tmp = path.with_suffix(".crop.mp4")
        subprocess.run([FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", str(path), "-vf",
                        f"crop={w}:{h}:{x}:{y}", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-g", str(gop),
                        "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-an", str(tmp)], check=True)
        if frame_count(tmp) != before:
            raise RuntimeError(f"cropping changed the frame count of {path}")
        tmp.replace(path)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, files))
    feat["shape"] = [h, w, feat["shape"][2]]
    feat.setdefault("info", {}).update({"video.height": h, "video.width": w, "video.codec": "h264",
                                        "video.pix_fmt": "yuv420p"})
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
    p.add_argument("source", type=Path, help="3-stage dataset root (meta/, data/, videos/)")
    p.add_argument("output", type=Path)
    p.add_argument("--episodes", type=Path, help="JSON episode selection (list, shortlist or curation file)")
    cal = p.add_mutually_exclusive_group(required=True)
    cal.add_argument("--calibration", type=Path, help="follower calibration JSON in force during recording")
    cal.add_argument("--calibration-note", help="when that file isn't available: which calibration was in force")
    args = p.parse_args()
    src, out = args.source.resolve(), args.output.resolve()
    if out.exists():
        p.error(f"{out} exists; choose a new output directory")
    info = json.loads((src / "meta/info.json").read_text())
    labels = [json.loads(l) for l in (src / "meta/dlsj/labels.jsonl").read_text().splitlines() if l.strip()]
    selection = load_selection(args.episodes) if args.episodes else None
    episodes, excluded = select(labels, selection)
    check_source(info, labels, episodes)
    check_stage_tasks(src, labels, episodes)
    drop = sorted(k for k in info["features"] if k.startswith("observation.images.") and k not in KEEP)

    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src / "meta", tmp / "meta")
    shutil.copytree(src / "data", tmp / "data")
    for key in KEEP:
        shutil.copytree(src / "videos" / key, tmp / "videos" / key)  # real copies: the snapshot is self-contained
    drop_features(tmp, info, drop)
    n_videos = crop_top(tmp, info)
    (tmp / "meta/info.json").write_text(json.dumps(info, indent=4))
    for extra in ("CURATION.json",):
        if (src / extra).exists():
            shutil.copyfile(src / extra, tmp / f"SOURCE-{extra}")
    if args.calibration:
        (tmp / "calibration").mkdir()
        shutil.copyfile(args.calibration, tmp / "calibration/follower.json")

    curation = json.loads((src / "CURATION.json").read_text()) if (src / "CURATION.json").exists() else None
    split = team_split(curation, episodes)
    split_source = "source CURATION.json (explicit team split)" if split else "Studio seed-42 10% split"
    train_ids, val_ids = split or studio_split(episodes)
    norm = train_normalization(tmp, train_ids)
    (tmp / "TRAIN-NORMALIZATION.json").write_text(json.dumps(norm, indent=2) + "\n")
    by_ep = {l["episode_index"]: l for l in labels}
    count = lambda key: {v: sum(by_ep[e][key] == v for e in episodes) for v in sorted({by_ep[e][key] for e in episodes})}
    frames = {f"part{k}": sum(by_ep[e][f"part{k}_frames"] for e in episodes if f"part{k}_frames" in by_ep[e])
              for k in (1, 2, 3)}
    manifest = {
        "source": str(src), "source_info_sha256": sha256(src / "meta/info.json"),
        "selection": str(args.episodes.resolve()) if args.episodes else None,
        "dataset_episodes": info["total_episodes"], "dataset_frames": info["total_frames"],
        "episodes": episodes, "excluded": excluded,
        "used_frames": sum(by_ep[e]["n_frames"] for e in episodes), "used_stage_frames": frames,
        "cameras": list(KEEP), "removed": drop, "top_crop": "0,0,560,480@640x480", "top_files_cropped": n_videos,
        "split": {"source": split_source, "train": train_ids, "validation": val_ids},
        "train_frames": norm["action"]["count"][0],
        "pick_tasks": count("pick_task"), "place_tasks": count("place_task"),
        "calibration_sha256": sha256(args.calibration) if args.calibration else None,
        "calibration_note": args.calibration_note,
    }
    (tmp / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = sorted(f for f in tmp.rglob("*") if f.is_file())
    (tmp / "SHA256SUMS.txt").write_text("".join(f"{sha256(f)}  {f.relative_to(tmp)}\n" for f in files))
    (tmp / "READY.json").write_text(json.dumps({"files": len(files), "episodes": len(episodes),
                                                "sha256_manifest": sha256(tmp / "SHA256SUMS.txt")}) + "\n")
    tmp.rename(out)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("split", "episodes")}, indent=2))
    print(f"{len(episodes)} episodes: train {len(train_ids)} / validation {len(val_ids)} ({split_source})")


if __name__ == "__main__":
    main()
