"""Build a two-instruction pick training set: grasp the brick, then bring it to the inspection camera.

Each take is split at the grasp, found from the commanded gripper trace (the last time it closes and stays
closed until the in-view mark):

* frames [0, grasp + --grasp-hold-s)  -> GRASP_TASK  "pick up a brick from the table"
* frames [.., mark + --hold-s)        -> CARRY_TASK  "bring the brick to the inspection camera"

Nothing after the inspection hold is kept (no place stage). Sources:

* `--full`: full pick -> inspect -> place demos, ranked and the best `--n-full` kept. Takes whose grasp comes
  sooner than `--min-approach-s` after the start (the arm began at the brick: no approach to learn) are dropped;
  then one clean grasp first (the gripper closes once before the mark), then the most direct approach: joint
  path length to the grasp over the straight-line joint distance (1 = no wandering or hesitation). Ranking by
  directness rather than speed keeps the spread of brick positions. `--exclude` drops episodes.
* `--pick`: pick-only takes (`dlsj.teleop_record --pick-only`), all used.

`--crop key=x,y,w,h` crops a camera while writing (deploy with the same crop). Use colour-only sources without
the cameras you don't want (e.g. `lerobot_edit_dataset remove_feature`). `meta/dlsj/provenance.jsonl` maps
each output episode to its source, grasp and split frames.

    uv run python scripts/build_pick_split_set.py --out data/duplo_pick2_400 \
        --full data/duplo_teleop_ws_517 --n-full 267 --exclude 464 --pick data/duplo_pick_ws \
        --crop observation.images.top=0,0,560,480
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

GRASP_TASK = "pick up a brick from the table"
CARRY_TASK = "bring the brick to the inspection camera"


def parse_crop(specs: list[str]) -> dict[str, tuple[int, int, int, int]]:
    out = {}
    for s in specs:
        key, _, box = s.partition("=")
        x, y, w, h = (int(v) for v in box.split(","))
        out[key] = (x, y, w, h)
    return out


def episode_frames(meta, ep: int, key: str, n: int) -> np.ndarray:
    """First `n` frames of episode `ep` for video `key`, uint8 [n, H, W, 3] RGB (PyAV, bundled FFmpeg)."""
    import av

    row = meta.episodes[ep]
    t0 = float(row[f"videos/{key}/from_timestamp"])
    want = t0 + np.arange(n) / meta.fps
    half = 0.5 / meta.fps
    out: list[np.ndarray] = []
    with av.open(str(meta.root / meta.get_video_file_path(ep, key))) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        c.seek(int(max(0.0, t0 - 1.0) / s.time_base), stream=s, backward=True)
        last = None
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            if t < want[len(out)] - half:
                continue
            img = fr.to_ndarray(format="rgb24")
            while len(out) < n and t >= want[len(out)] - half:
                out.append(img)
            last = img
            if len(out) >= n:
                break
        while len(out) < n and last is not None:
            out.append(last)
    if len(out) < n:
        raise RuntimeError(f"episode {ep} {key}: decoded {len(out)} of {n} frames")
    return np.stack(out)


def load_table(root: Path) -> pd.DataFrame:
    fs = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    return pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
                     for f in fs).sort_values(["episode_index", "frame_index"])



def find_grasp(gripper_cmd: np.ndarray, mark: int) -> tuple[int | None, int]:
    """(grasp frame, closings before the mark). The grasp is where the commanded gripper last drops to
    <= 35% of its pre-mark maximum and stays there until the mark."""
    g = np.asarray(gripper_cmd[:mark], dtype=float)
    if len(g) < 10 or g.max() <= 0:
        return None, 0
    closed = g <= 0.35 * g.max()
    if not closed[-1]:
        return None, 0
    stay = np.flip(np.cumprod(np.flip(closed)))
    return int(np.argmax(stay)), int(np.sum(np.diff(closed.astype(int)) == 1))


def directness(actions: np.ndarray, grasp: int) -> float:
    """Arm-joint path length from the start to the grasp over the straight-line distance (>= 1)."""
    arm = np.asarray(actions[: grasp + 1, :5], dtype=float)
    path = np.abs(np.diff(arm, axis=0)).sum()
    straight = np.abs(arm[-1] - arm[0]).sum()
    return float(path / max(straight, 1e-6))


def plan(meta, table: pd.DataFrame, labels: dict, ep: int, hold: int, grasp_hold: int):
    g = table[table.episode_index == ep]
    a = np.stack(g["action"].to_numpy())
    mark = labels[ep].get("inspection_frame")
    if mark is None:
        return None
    grasp, closings = find_grasp(a[:, 5], mark)
    if grasp is None:
        return None
    n = min(int(meta.episodes[ep]["length"]), mark + hold)
    split = min(grasp + grasp_hold, mark)
    return {"episode": ep, "n": n, "mark": mark, "grasp": grasp, "split": split, "closings": closings,
            "directness": round(directness(a, grasp), 3)}


def main() -> None:
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--repo-id", default=None)
    p.add_argument("--full", type=Path)
    p.add_argument("--n-full", type=int, default=None, help="keep the best N full demos (default all usable)")
    p.add_argument("--exclude", default="", help="comma-separated full-demo episodes to drop")
    p.add_argument("--min-approach-s", type=float, default=1.0,
                   help="drop full demos whose grasp comes sooner than this after the start")
    p.add_argument("--pick", type=Path, action="append", default=[])
    p.add_argument("--hold-s", type=float, default=0.3, help="kept after the in-view mark")
    p.add_argument("--grasp-hold-s", type=float, default=0.4, help="grasp instruction continues this long after the grasp")
    p.add_argument("--crop", action="append", default=[])
    p.add_argument("--vcodec", default="h264")
    p.add_argument("--limit", type=int, default=None, help="smoke test: this many episodes per source")
    p.add_argument("--dry-run", action="store_true", help="only print the selection")
    args = p.parse_args()
    crops = parse_crop(args.crop)
    exclude = {int(x) for x in args.exclude.split(",") if x.strip()}

    sources = []  # (meta, table, [plans], tag)
    for root, tag in ([(args.full, "full")] if args.full else []) + [(r, "pick") for r in args.pick]:
        meta = LeRobotDatasetMetadata(f"local/{root.name}", root=root)
        table = load_table(root)
        labels = {json.loads(l)["episode_index"]: json.loads(l)
                  for l in (root / "meta" / "dlsj" / "labels.jsonl").read_text().splitlines() if l.strip()}
        hold, gh = int(round(args.hold_s * meta.fps)), int(round(args.grasp_hold_s * meta.fps))
        plans, skipped = [], []
        for ep in range(meta.total_episodes):
            if tag == "full" and ep in exclude:
                continue
            pl = plan(meta, table, labels, ep, hold, gh)
            (plans if pl else skipped).append(pl or ep)
        if tag == "full":
            min_approach = int(round(args.min_approach_s * meta.fps))
            short = [r["episode"] for r in plans if r["grasp"] < min_approach]
            plans = [r for r in plans if r["grasp"] >= min_approach]
            print(f"  dropped {len(short)} full demos with no approach (grasp < {args.min_approach_s} s): {short}")
            plans.sort(key=lambda r: (r["closings"] != 1, r["directness"]))
            plans = plans[: args.n_full] if args.n_full else plans
        print(f"{tag} {root.name}: {len(plans)} episodes kept, {len(skipped)} without mark/grasp {skipped[:20]}"
              f" | one clean grasp: {sum(r['closings'] == 1 for r in plans)}", flush=True)
        sources.append((meta, table, plans[: args.limit] if args.limit else plans, root, tag))
    if args.dry_run:
        return

    meta0 = sources[0][0]
    cams = list(meta0.video_keys)
    feats = {k: dict(meta0.features[k]) for k in ("observation.state", "action")}
    for k in cams:
        h, w, c = meta0.features[k]["shape"]
        if k in crops:
            _, _, w, h = crops[k]
        feats[k] = {"dtype": "video", "shape": (h, w, c), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(args.repo_id or f"local/{args.out.name}", meta0.fps, features=feats, root=args.out,
                               robot_type=meta0.robot_type, use_videos=True, image_writer_threads=4 * len(cams),
                               rgb_encoder=RGBEncoderConfig(vcodec=args.vcodec), streaming_encoding=True)
    prov, i = [], 0
    total = sum(len(s[2]) for s in sources)
    for meta, table, plans, root, tag in sources:
        for pl in plans:
            ep, n, split = pl["episode"], pl["n"], pl["split"]
            g = table[table.episode_index == ep].iloc[:n]
            S, A = np.stack(g["observation.state"].to_numpy()), np.stack(g["action"].to_numpy())
            imgs = {}
            for k in cams:
                fr = episode_frames(meta, ep, k, n)
                if k in crops:
                    x, y, w, h = crops[k]
                    fr = fr[:, y: y + h, x: x + w]
                imgs[k] = fr
            for t in range(n):
                ds.add_frame({"observation.state": S[t].astype(np.float32), "action": A[t].astype(np.float32),
                              "task": GRASP_TASK if t < split else CARRY_TASK,
                              **{k: np.ascontiguousarray(imgs[k][t]) for k in cams}})
            ds.save_episode()
            prov.append({"episode_index": i, "source": str(root), "kind": tag, **pl, "crop": crops,
                         "tasks": {GRASP_TASK: [0, split], CARRY_TASK: [split, n]}})
            i += 1
            if i % 25 == 0:
                print(f"{i}/{total} episodes", flush=True)
    ds.finalize()
    d = args.out / "meta" / "dlsj"
    d.mkdir(parents=True, exist_ok=True)
    (d / "provenance.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prov))
    print(f"wrote {i} episodes to {args.out}")


if __name__ == "__main__":
    main()
