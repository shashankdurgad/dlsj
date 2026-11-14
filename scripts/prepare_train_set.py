"""Build a training copy of the teleop dataset: overhead camera cropped, plate labels corrected.

The raw recording is never modified: this copies a colour-only export of it, then edits the copy.

    # on the XPS, colour-only export of the raw recording (depth isn't used for ACT):
    python -m lerobot.scripts.lerobot_edit_dataset --repo_id local/duplo_teleop --root data/duplo_teleop \
        --new_repo_id local/duplo_teleop_rgb --new_root data/duplo_teleop_rgb \
        --operation.type remove_feature \
        --operation.feature_names '["observation.images.top_depth", "observation.images.inspect_depth"]'

    # then, anywhere with ffmpeg:
    uv run python scripts/prepare_train_set.py --src data/duplo_teleop_rgb --dst data/duplo_act_417 \
        --crop observation.images.top=0,0,560,480 \
        --corrections data/duplo_teleop/meta/dlsj/plate_corrections.json

Plate corrections: in episodes 0-311 a plate chosen between stopping and saving a take relabelled that take
(fixed in the recorder from episode 312). `plate_corrections.json` lists each affected episode with the
plate the brick actually went to; the place instruction of those frames is rewritten accordingly.

The crop must also be applied at deployment: export with
`python -m dlsj.deploy.act_export ... --crop observation.images.top=0,0,560,480@640x480`.
"""

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PLACE = "place the brick in the {} bowl"


def task_table(root: Path) -> dict[str, int]:
    t = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    return {str(task): int(i) for task, i in zip(t.index, t["task_index"], strict=True)}


def apply_corrections(root: Path, corrections: dict) -> dict[int, int]:
    """Rewrite the place-task index of corrected episodes. Returns {episode: frames changed}."""
    tasks = task_table(root)
    fix = {int(e): (tasks[PLACE.format(c["recorded"])], tasks[PLACE.format(c["actual"])])
           for e, c in corrections.items()}
    changed: dict[int, int] = {}
    for f in sorted((root / "data").glob("*/*.parquet")):
        t = pq.read_table(f)
        ep = t.column("episode_index").to_numpy()
        ti = t.column("task_index").to_numpy().copy()
        hit = False
        for e, (old, new) in fix.items():
            m = (ep == e) & (ti == old)
            if m.any():
                ti[m] = new
                changed[e] = changed.get(e, 0) + int(m.sum())
                hit = True
        if hit:
            col = t.schema.get_field_index("task_index")
            t = t.set_column(col, t.schema.field(col), pa.array(ti, type=t.schema.field(col).type))
            pq.write_table(t, f)
    missing = sorted(set(fix) - set(changed))
    if missing:
        raise RuntimeError(f"no place-task frames found to correct in episodes {missing}")
    # per-episode metadata: task list and task_index stats
    names = {i: s for s, i in tasks.items()}
    per_ep = {}
    for f in sorted((root / "data").glob("*/*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "task_index"]).to_pandas()
        for e, g in t[t.episode_index.isin(list(fix))].groupby("episode_index"):
            per_ep[int(e)] = g.task_index.to_numpy()
    for f in sorted((root / "meta/episodes").glob("*/*.parquet")):
        t = pq.read_table(f)
        eps = t.column("episode_index").to_pylist()
        if not any(e in fix for e in eps):
            continue
        cols = {name: t.column(name).to_pylist() for name in t.column_names}
        for r, e in enumerate(eps):
            if e not in fix:
                continue
            v = per_ep[e].astype(np.float64)
            cols["tasks"][r] = sorted({names[int(i)] for i in v})
            for k, val in (("min", v.min()), ("max", v.max()), ("mean", v.mean()), ("std", v.std()),
                           ("q01", np.quantile(v, .01)), ("q10", np.quantile(v, .10)), ("q50", np.quantile(v, .5)),
                           ("q90", np.quantile(v, .9)), ("q99", np.quantile(v, .99))):
                if f"stats/task_index/{k}" in cols:
                    cols[f"stats/task_index/{k}"][r] = [float(val)]
        pq.write_table(pa.table({n: pa.array(cols[n], type=t.schema.field(n).type) for n in t.column_names},
                                schema=t.schema), f)
    return changed


def crop_videos(root: Path, key: str, box: tuple[int, int, int, int], workers: int = 8) -> None:
    x, y, w, h = box
    files = sorted((root / "videos" / key).glob("*/*.mp4"))
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    feat = info["features"][key]
    g = feat["info"].get("video.g") or 2

    def one(f: Path) -> None:
        tmp = f.with_suffix(".crop.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(f),
                        "-vf", f"crop={w}:{h}:{x}:{y}", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                        "-g", str(g), "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-an", str(tmp)], check=True)
        tmp.replace(f)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, files))
    feat["shape"] = [h, w, feat["shape"][2]]
    feat["info"].update({"video.height": h, "video.width": w, "video.codec": "h264", "video.crf": 23})
    info_path.write_text(json.dumps(info, indent=4))


def frame_count(f: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(f)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="colour-only dataset (never modified)")
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--crop", action="append", default=[], help="key=x,y,w,h (repeatable)")
    ap.add_argument("--corrections", type=Path, help="plate_corrections.json")
    ap.add_argument("--force", action="store_true", help="replace --dst if it exists")
    args = ap.parse_args()
    if args.dst.exists():
        if not args.force:
            ap.error(f"{args.dst} exists (use --force to replace it)")
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst)

    if args.corrections:
        corr = json.loads(args.corrections.read_text())["corrections"]
        changed = apply_corrections(args.dst, corr)
        print(f"plate corrections: {len(changed)} of {len(corr)} episodes, {sum(changed.values())} frames relabelled")
    for spec in args.crop:
        key, box = spec.split("=", 1)
        vals = tuple(int(v) for v in box.split(","))
        before = {f.name: frame_count(f) for f in sorted((args.src / "videos" / key).glob("*/*.mp4"))}
        crop_videos(args.dst, key, vals)
        after = {f.name: frame_count(f) for f in sorted((args.dst / "videos" / key).glob("*/*.mp4"))}
        if before != after:
            raise RuntimeError(f"{key}: frame counts changed by cropping")
        print(f"cropped {key} to {vals[2]}x{vals[3]} at ({vals[0]},{vals[1]}); {len(after)} files, frame counts match")


if __name__ == "__main__":
    main()
