"""Validate model-reviewed full-frame boxes and train a one-class brick detector."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import cv2


def prepare_annotations(annotation_path, destination):
    data = json.loads(Path(annotation_path).read_text())
    if data.get('class_names') != ['brick']:
        raise ValueError('Expected exactly one class: brick')
    rows = [r for r in data['images'] if r.get('review_status') == 'reviewed']
    groups, hashes, paths = {}, {}, set()
    checked = []
    for row in rows:
        source = Path(row['path']).resolve()
        if source in paths:
            raise ValueError(f'Duplicate image: {source}')
        paths.add(source)
        split = row['split']
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Unknown dataset split')
        capture = row['capture']
        if groups.setdefault(capture, split) != split:
            raise ValueError('Capture leaked between splits')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashes.setdefault(digest, split) != split:
            raise ValueError('Identical image leaked between splits')
        image = cv2.imread(str(source))
        if image is None or image.shape[:2] != (row['height'], row['width']):
            raise ValueError(f'Image dimensions do not match annotations: {source}')
        labels = []
        w, h = row['width'], row['height']
        for obj in row['boxes']:
            box = obj['xyxy']
            if obj.get('class_id') != 0 or len(box) != 4 or not all(math.isfinite(v) for v in box):
                raise ValueError('Invalid class or bounding box')
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError('Bounding box is outside original frame')
            labels.append(f'0 {(x0+x1)/2/w:.8f} {(y0+y1)/2/h:.8f} {(x1-x0)/w:.8f} {(y1-y0)/h:.8f}')
        checked.append((row, source, digest, labels))
    for split in ('train', 'validation', 'test'):
        if not any(r['split'] == split and labels for r, _, _, labels in checked):
            raise ValueError(f'Need reviewed positive examples in {split}')
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use an empty output directory to prevent stale labels')
    manifest = []
    for i, (row, source, digest, labels) in enumerate(checked):
        split = {'validation': 'val'}.get(row['split'], row['split'])
        name = f'{i:05d}_{source.name}'
        image_dir, label_dir = destination/'images'/split, destination/'labels'/split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, image_dir/name)
        (label_dir/Path(name).with_suffix('.txt')).write_text('\n'.join(labels)+'\n' if labels else '')
        manifest.append({**row, 'sha256': digest, 'dataset_image': str(image_dir/name)})
    (destination/'data.yaml').write_text(f'path: {json.dumps(str(destination))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: brick\n')
    (destination/'manifest.json').write_text(json.dumps({'images': manifest, 'annotation_sha256': hashlib.sha256(Path(annotation_path).read_bytes()).hexdigest()}, indent=2))
    return destination/'data.yaml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weights', default='yolo26n.pt')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--validation-only', action='store_true', help='Train/select on validation without test evaluation or export')
    args = parser.parse_args()
    dataset = prepare_annotations(args.annotations, args.out/'dataset')
    if args.prepare_only:
        print(dataset)
        return
    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.train(data=str(dataset), imgsz=640, epochs=args.epochs, batch=args.batch,
                device=args.device, workers=0, seed=2026, deterministic=True,
                patience=args.patience, project=str(args.out.resolve()), name='train',
                exist_ok=False, degrees=30, fliplr=.5, flipud=.5, mosaic=.5,
                close_mosaic=10, mixup=0., plots=True)
    if args.validation_only:
        print('Validation-only run complete; test and export skipped')
        return
    # Test is evaluated once after checkpoint selection using validation only.
    best = YOLO(args.out/'train/weights/best.pt')
    metrics = best.val(data=str(dataset), split='test', device=args.device, rect=False, plots=True)
    (args.out/'test_metrics.json').write_text(json.dumps(metrics.results_dict, indent=2))
    best.export(format='openvino', imgsz=640, batch=1, dynamic=False, half=False)


if __name__ == '__main__':
    main()
