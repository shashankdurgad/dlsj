"""Build stage-2 training crops (`<out>/circle/*.png`, `<out>/cross/*.png`) for `train_classifier.py`.

Every crop uses the same geometry as at run time (`classify.crop_box`: mark box padded by `pad_frac`, square) and is
saved at the CNN input size, so iPhone crops, inspect-camera crops and live inference all look alike.

1. From `inspect`-camera frames (`capture.py` / synthetic layout: `bad/` = cross, `minor/` = circle), with the mark
   box from the stage-1 Anomalib model, exactly as the robot will see it:

       python -m dlsj.inspect.crops inspect --data ~/inference/inspect_data/real \\
           --manifest ~/inference/exports/inspect/padim/manifest.json --out data/mark_crops/camera

2. From iPhone photos (`<photos>/circle/*.jpg`, `<photos>/cross/*.jpg`; set the camera to "Most Compatible"/JPEG
   or convert HEIC first: `sips -s format jpeg *.HEIC --out jpg/`). The mark box comes from `--boxes`
   (JSON {"circle/IMG_0001.jpg": [x0, y0, x1, y1], ...} in photo pixels), from `--select` (drag a box per photo in
   an OpenCV window; needs opencv-python with GUI, not -headless), or from `--auto` (the shape-rule segmentation;
   check the contact sheet). The crop is then shrunk so the mark is `--mark-px` wide (the mark's typical width in the
   640x480 inspect frame; `inspect` mode prints it) and blown up again to the CNN input: the iPhone's detail is thrown
   away like the webcam would, which bridges most of the iPhone -> webcam gap:

       python -m dlsj.inspect.crops photos --photos ~/iphone_marks --auto --mark-px 70 --out data/mark_crops/iphone

`contact_sheet.jpg` in `--out` shows every crop for a quick visual check; `crops.json` records sources and boxes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from dlsj.inspect import classify, core

log = logging.getLogger("inspect.crops")

FOLDER_CLASS = {"bad": "cross", "minor": "circle", "cross": "cross", "circle": "circle"}


def save_crop(out: Path, cls: str, name: str, crop: np.ndarray, size: int) -> Path:
    d = out / cls
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.png"
    core.write_rgb(p, classify.to_input(crop, size))
    return p


def domain_match(crop: np.ndarray, mark_w: float, mark_px: float, size: int, jpeg: int | None = 90) -> np.ndarray:
    """Resample a high-resolution crop so the mark is `mark_px` wide (webcam scale), then back up to `size`."""
    scale = min(1.0, mark_px / max(mark_w, 1.0))
    small = cv2.resize(crop, (max(8, round(crop.shape[1] * scale)), max(8, round(crop.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
    if jpeg:
        ok, buf = cv2.imencode(".jpg", small[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)])
        if ok:
            small = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)


def auto_box(img: np.ndarray, work: int = 384) -> tuple[int, int, int, int] | None:
    """Mark box in a close-up photo of a brick face via the shape-rule segmentation (photo pixels)."""
    h, w = img.shape[:2]
    s = work / max(h, w)
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    mask, _ = classify.segment_mark(small)
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    k = max(small.shape[:2]) / classify.WORK / s  # mask is at WORK scale of `small`
    return int(xs.min() * k), int(ys.min() * k), int(np.ceil((xs.max() + 1) * k)), int(np.ceil((ys.max() + 1) * k))


def contact_sheet(paths: list[Path], out: Path, cols: int = 12, tile: int = 96) -> None:
    if not paths:
        return
    tiles = [cv2.resize(core.read_rgb(p), (tile, tile)) for p in paths]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    core.write_rgb(out, np.vstack(rows))


def from_inspect(args) -> list[dict]:
    from dlsj.inspect.detector import AnomalibInspector

    insp = AnomalibInspector.from_manifest(args.manifest, device=args.device, batch=args.batch, stage2="none",
                                           heatmap=False)
    data = args.data.expanduser()
    recs, widths = [], []
    for folder in ("bad", "minor", "cross", "circle"):
        cls = args.cls or FOLDER_CLASS[folder]
        for brick, paths in sorted(core.group_by_brick(core.list_images(data / folder)).items()):
            frames = [core.read_rgb(p) for p in paths]
            r = insp.inspect(frames)
            if r.region is None:
                log.warning("%s/%s: stage 1 found no region (label %s), skipped", folder, brick, r.label.value)
                continue
            widths.append(max(r.region[2] - r.region[0], r.region[3] - r.region[1]))
            box = classify.crop_box(r.region, frames[0].shape[:2], args.pad_frac)
            for p, f in zip(paths, frames):
                x0, y0, x1, y1 = box
                out = save_crop(args.out, cls, f"{args.prefix}{p.stem}", f[y0:y1, x0:x1], args.size)
                recs.append({"crop": str(out), "source": str(p), "class": cls, "region": list(r.region),
                             "box": list(box), "stage1": r.label.value, "score": round(r.score, 4)})
    if widths:
        print(f"stage-1 region size (max side, inspect px): median {np.median(widths):.0f}, "
              f"p10 {np.percentile(widths, 10):.0f}, p90 {np.percentile(widths, 90):.0f}  -> use as --mark-px for photos")
    return recs


def _select_box(img: np.ndarray, name: str):
    s = 1000 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=s, fy=s)
    x, y, w, h = cv2.selectROI(f"drag a box around the mark: {name} (Enter = ok, c = skip)", small[..., ::-1])
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        return None
    return int(x / s), int(y / s), int((x + w) / s), int((y + h) / s)


def from_photos(args) -> list[dict]:
    root = args.photos.expanduser()
    boxes = json.loads(args.boxes.read_text()) if args.boxes else {}
    recs = []
    folders = [args.cls] if args.cls else ["circle", "cross"]
    for cls in folders:
        d = root if args.cls else root / cls
        for p in sorted(x for x in d.rglob("*") if x.suffix.lower() in core.IMAGE_EXTS):
            rel = str(p.relative_to(root))
            img = core.read_rgb(p)  # cv2 applies the EXIF orientation
            box = boxes.get(rel) or boxes.get(p.name)
            how = "boxes"
            if box is None and args.select:
                box, how = _select_box(img, rel), "select"
            if box is None and args.auto:
                box, how = auto_box(img), "auto"
            if box is None:
                log.warning("%s: no mark box (use --boxes, --select or --auto), skipped", rel)
                continue
            box = tuple(int(v) for v in box)
            cb = classify.crop_box(box, img.shape[:2], args.pad_frac)
            x0, y0, x1, y1 = cb
            crop = domain_match(img[y0:y1, x0:x1], max(box[2] - box[0], box[3] - box[1]), args.mark_px, args.size,
                                args.jpeg)
            out = save_crop(args.out, cls, f"{args.prefix}{p.stem}", crop, args.size)
            recs.append({"crop": str(out), "source": rel, "class": cls, "box": list(box), "box_source": how,
                         "crop_box": list(cb)})
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("inspect", "photos"):
        s = sub.add_parser(name)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--size", type=int, default=128, help="saved crop size (= CNN input)")
        s.add_argument("--pad-frac", type=float, default=0.3, help="must match the classifier's pad_frac")
        s.add_argument("--class", dest="cls", choices=["circle", "cross"], help="force the class of every input")
        s.add_argument("--prefix", default="", help="prefix for crop names (e.g. the session or phone)")
    s = sub.choices["inspect"]
    s.add_argument("--data", type=Path, required=True, help="inspect frames: bad/ (cross), minor/ (circle)")
    s.add_argument("--manifest", type=Path, required=True, help="stage-1 manifest.json (gives the region)")
    s.add_argument("--device", default="CPU")
    s.add_argument("--batch", type=int, default=1)
    s = sub.choices["photos"]
    s.add_argument("--photos", type=Path, required=True, help="folder with circle/ and cross/ photos")
    s.add_argument("--boxes", type=Path, help='JSON {"circle/IMG_1.jpg": [x0,y0,x1,y1]} mark boxes in photo px')
    s.add_argument("--select", action="store_true", help="draw missing boxes by hand (OpenCV GUI)")
    s.add_argument("--auto", action="store_true", help="find missing boxes with the shape-rule segmentation")
    s.add_argument("--mark-px", type=float, default=70.0, help="mark width in inspect-camera px (domain match)")
    s.add_argument("--jpeg", type=int, default=90, help="JPEG quality applied at webcam scale (0 = off)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.out = args.out.expanduser()
    recs = from_inspect(args) if args.mode == "inspect" else from_photos(args)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = args.out / "crops.json"
    old = json.loads(meta.read_text()) if meta.exists() else []
    meta.write_text(json.dumps(old + recs, indent=1) + "\n")
    contact_sheet([Path(r["crop"]) for r in recs], args.out / f"contact_sheet_{args.mode}.jpg")
    counts = {c: sum(r["class"] == c for r in recs) for c in ("circle", "cross")}
    print(f"wrote {len(recs)} crops {counts} to {args.out}")


if __name__ == "__main__":
    main()
