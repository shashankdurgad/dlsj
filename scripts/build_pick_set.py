"""Build a pick-only training set: pick a brick and bring it to the inspection camera, nothing after.

    uv run python scripts/build_pick_set.py --out data/duplo_pickset_300 \
        --full data/duplo_teleop_rgb_417 --full-episodes selected_200.json \
        --pick data/duplo_pick_rgb --crop observation.images.top=0,0,560,480

* `--full`: full pick -> inspect -> place demos. Each selected episode is cut at its in-view marker
  (`inspection_frame` in meta/dlsj/labels.jsonl) plus `--hold-s` (default 0.3 s, as the pick-only recorder does).
* `--pick`: pick-only takes from `dlsj.teleop_record --pick-only`, used whole (they already end that way).
* `--crop key=x,y,w,h`: crop applied to every frame of that camera while writing (the deployment applies the same
  crop), so no external ffmpeg is needed.
* `--pad-to W,H`: then pad any smaller camera with black on the right/bottom up to W x H. Keeps every camera's
  aspect ratio and resolution while giving LeRobot's Diffusion Policy the equal shapes it requires
  (e.g. the 560x480 cropped top view becomes 640x480 with an 80 px black band). Deploy with the same crop + pad.
* `--resize H,W`: alternatively, resize every camera to H x W (INTER_AREA). LeRobot's Diffusion Policy needs all cameras the
  same shape, and 240x320 is the Diffusion Policy paper's real-world input size. Deploy with the same crop + resize.

Every frame gets the pick instruction. `meta/dlsj/provenance.jsonl` maps each output episode to its source.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

PICK_TASK = "pick up the brick and hold it in front of the inspection camera"


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


def main() -> None:
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--repo-id", default=None, help="default local/<out name>")
    p.add_argument("--full", type=Path, help="full-demo dataset root")
    p.add_argument("--full-episodes", type=Path, help='json {"episodes": [...]} or a list; default all')
    p.add_argument("--pick", type=Path, action="append", default=[], help="pick-only dataset root (repeatable)")
    p.add_argument("--pick-episodes", default=None, help="a-b: only these episodes of each --pick dataset")
    p.add_argument("--hold-s", type=float, default=0.3)
    p.add_argument("--crop", action="append", default=[], help="key=x,y,w,h")
    p.add_argument("--pad-to", default=None, help="W,H: pad smaller cameras with black to this size, e.g. 640,480")
    p.add_argument("--resize", default=None, help="H,W applied to every camera after cropping, e.g. 240,320")
    p.add_argument("--vcodec", default="h264")
    p.add_argument("--limit", type=int, default=None, help="only this many episodes per source (smoke test)")
    args = p.parse_args()
    crops = parse_crop(args.crop)
    resize = tuple(int(v) for v in args.resize.split(",")) if args.resize else None
    pad_to = tuple(int(v) for v in args.pad_to.split(",")) if args.pad_to else None  # (W, H)

    jobs = []  # (meta, table, episode, n_frames, provenance)
    if args.full:
        meta = LeRobotDatasetMetadata(f"local/{args.full.name}", root=args.full)
        labels = {json.loads(l)["episode_index"]: json.loads(l)
                  for l in (args.full / "meta" / "dlsj" / "labels.jsonl").read_text().splitlines() if l.strip()}
        sel = json.loads(args.full_episodes.read_text()) if args.full_episodes else list(range(meta.total_episodes))
        sel = sel["episodes"] if isinstance(sel, dict) else sel
        table = load_table(args.full)
        hold = int(round(args.hold_s * meta.fps))
        for ep in sel[: args.limit]:
            split = labels[ep]["inspection_frame"]
            n = min(int(meta.episodes[ep]["length"]), split + hold)
            jobs.append((meta, table, ep, n, {"source": str(args.full), "episode": ep, "frames": [0, n],
                                              "cut": f"inspection_frame {split} + {hold}"}))
    for root in args.pick:
        meta = LeRobotDatasetMetadata(f"local/{root.name}", root=root)
        table = load_table(root)
        pick_eps = range(meta.total_episodes)
        if args.pick_episodes:
            a, b = (int(v) for v in args.pick_episodes.split("-"))
            pick_eps = range(a, min(b, meta.total_episodes - 1) + 1)
        for ep in list(pick_eps)[: args.limit]:
            n = int(meta.episodes[ep]["length"])
            jobs.append((meta, table, ep, n, {"source": str(root), "episode": ep, "frames": [0, n], "cut": "whole"}))

    meta0 = jobs[0][0]
    cams = [k for k in meta0.video_keys]
    feats = {k: dict(meta0.features[k]) for k in ("observation.state", "action")}
    for k in cams:
        h, w, c = meta0.features[k]["shape"]
        if k in crops:
            _, _, w, h = crops[k]
        if pad_to:
            if w > pad_to[0] or h > pad_to[1]:
                raise SystemExit(f"{k}: {w}x{h} is larger than --pad-to {pad_to[0]}x{pad_to[1]}")
            w, h = pad_to
        if resize:
            h, w = resize
        feats[k] = {"dtype": "video", "shape": (h, w, c), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(args.repo_id or f"local/{args.out.name}", meta0.fps, features=feats, root=args.out,
                               robot_type=meta0.robot_type, use_videos=True, image_writer_threads=4 * len(cams),
                               rgb_encoder=RGBEncoderConfig(vcodec=args.vcodec), streaming_encoding=True)
    prov = []
    for i, (meta, table, ep, n, info) in enumerate(jobs):
        g = table[table.episode_index == ep].iloc[:n]
        S, A = np.stack(g["observation.state"].to_numpy()), np.stack(g["action"].to_numpy())
        imgs = {}
        for k in cams:
            fr = episode_frames(meta, ep, k, n)
            if k in crops:
                x, y, w, h = crops[k]
                fr = fr[:, y: y + h, x: x + w]
            if pad_to and fr.shape[1:3] != (pad_to[1], pad_to[0]):
                padded = np.zeros((len(fr), pad_to[1], pad_to[0], 3), dtype=fr.dtype)
                padded[:, : fr.shape[1], : fr.shape[2]] = fr
                fr = padded
            if resize:
                import cv2

                fr = np.stack([cv2.resize(f, (resize[1], resize[0]), interpolation=cv2.INTER_AREA) for f in fr])
            imgs[k] = fr
        for t in range(n):
            ds.add_frame({"observation.state": S[t].astype(np.float32), "action": A[t].astype(np.float32),
                          "task": PICK_TASK, **{k: np.ascontiguousarray(imgs[k][t]) for k in cams}})
        ds.save_episode()
        prov.append({"episode_index": i, **info, "crop": crops, "pad_to": pad_to, "resize": resize})
        if i % 20 == 0:
            print(f"{i + 1}/{len(jobs)} episodes", flush=True)
    ds.finalize()
    d = args.out / "meta" / "dlsj"
    d.mkdir(parents=True, exist_ok=True)
    (d / "provenance.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prov))
    print(f"done: {len(jobs)} episodes, {ds.num_frames} frames -> {args.out}")


if __name__ == "__main__":
    main()
