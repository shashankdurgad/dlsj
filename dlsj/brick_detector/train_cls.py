"""Train the brick-in-view classifier (YOLO11n-cls) on inspection-camera captures and export it to OpenVINO.

    python -m dlsj.brick_detector.train_cls --frames ~/desmond/inspect_frames --out ~/desmond/brick_detector/runs

Input: the capture tool's sessions, `<frames>/<session>/{in_view,not_in_view}/*.jpg` plus `index.csv`.

The train/val split is by time block, not random. Holding a capture key saves near-identical consecutive
frames, so a random split would put copies of training images in validation and inflate the score.
Frames are sorted by capture time, cut into blocks of `--block` frames, and every `1/val_frac`-th block of
each class goes to validation.

Output:
* `<out>/split/{train,val}/<class>/*.jpg` (symlinks)
* the Ultralytics run, with `best.pt`
* the OpenVINO exports (FP16, and INT8 calibrated on the train split)
* `<out>/summary.json` with validation accuracy, per-class recall and export paths
"""

import argparse
import csv
import json
import os
from pathlib import Path

CLASSES = ("in_view", "not_in_view")


def collect(frames: Path) -> dict[str, list[tuple[float, Path]]]:
    """(time, path) per class across every session, sorted by time. Uses index.csv; falls back to mtime."""
    out: dict[str, list[tuple[float, Path]]] = {c: [] for c in CLASSES}
    for session in sorted(p for p in frames.iterdir() if p.is_dir()):
        times = {}
        index = session / "index.csv"
        if index.exists():
            with index.open(newline="") as f:
                for row in csv.DictReader(f):
                    times[row["file"]] = float(row["time"])
        for c in CLASSES:
            for img in sorted((session / c).glob("*.jpg")):
                rel = f"{c}/{img.name}"
                out[c].append((times.get(rel, img.stat().st_mtime), img))
    for c in CLASSES:
        out[c].sort()
    return out


def split(items: list, block: int, val_frac: float) -> tuple[list, list]:
    """Contiguous blocks of `block` items; every k-th block (k = round(1/val_frac)) goes to val."""
    k = max(2, round(1 / val_frac))
    train, val = [], []
    for b in range(0, len(items), block):
        (val if (b // block) % k == k - 1 else train).extend(items[b: b + block])
    if not val and items:  # tiny class: hold out its last block
        val, train = items[-block:], items[:-block]
    return train, val


def make_split(frames: Path, root: Path, block: int, val_frac: float) -> dict:
    data = collect(frames)
    counts = {}
    for part in ("train", "val"):
        for c in CLASSES:
            d = root / part / c
            d.mkdir(parents=True, exist_ok=True)
            for old in d.iterdir():
                old.unlink()
    for c in CLASSES:
        tr, va = split(data[c], block, val_frac)
        for part, items in (("train", tr), ("val", va)):
            for i, (_, src) in enumerate(items):
                os.symlink(src.resolve(), root / part / c / f"{i:05d}_{src.parent.parent.name}_{src.name}")
        counts[c] = {"train": len(tr), "val": len(va)}
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="yolo11n-cls.pt")
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--block", type=int, default=10, help="frames per contiguous block for the split")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--no-int8", action="store_true")
    args = p.parse_args()

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    split_root = args.out / "split"
    counts = make_split(args.frames.expanduser(), split_root, args.block, args.val_frac)
    print("split:", counts, flush=True)

    model = YOLO(args.model)
    model.train(
        data=str(split_root), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=0,
        project=str(args.out), name="cls", exist_ok=True, seed=0, patience=15,
        # held bricks appear at any angle and lighting, so augment hard; a vertical flip is still a brick
        fliplr=0.5, flipud=0.2, degrees=20, hsv_h=0.02, hsv_s=0.5, hsv_v=0.4, erasing=0.3, plots=True,
    )
    best = args.out / "cls" / "weights" / "best.pt"
    trained = YOLO(str(best))
    metrics = trained.val(data=str(split_root), imgsz=args.imgsz, split="val", device=0, plots=False)

    # per-class recall on the held-out blocks
    per_class = {}
    for c_idx, c in enumerate(trained.names.values()):
        imgs = sorted((split_root / "val" / c).iterdir())
        if not imgs:
            continue
        preds = trained.predict([str(i) for i in imgs], imgsz=args.imgsz, device=0, verbose=False)
        per_class[c] = round(sum(int(r.probs.top1) == c_idx for r in preds) / len(imgs), 4)

    exports = {"fp16": trained.export(format="openvino", imgsz=args.imgsz, half=True)}
    if not args.no_int8:
        exports["int8"] = trained.export(format="openvino", imgsz=args.imgsz, int8=True, data=str(split_root))
    summary = {
        "split": counts,
        "top1_val": round(float(metrics.top1), 4),
        "recall_per_class": per_class,
        "names": trained.names,
        "best_pt": str(best),
        "openvino": {k: str(v) for k, v in exports.items()},
        "imgsz": args.imgsz,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
