"""Build matching detector crops for Anomalib/whole-brick classifier experiments.

Only model-reviewed, explicitly identified target bricks receive capture labels.
All three original views must be detected, or the whole capture is reported as
missed. Missing targets must count against end-to-end coverage in evaluations.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2

from dlsj.inspect.brick_detector import BrickDetector, inspection_crop, iou


def build_crops(annotations, source_manifest, detector, out, min_inspection_side=64):
    annotated = json.loads(Path(annotations).read_text())['images']
    sources = {r['capture']: r for r in json.loads(Path(source_manifest).read_text())['images']}
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty crop dataset directory')
    out.mkdir(parents=True, exist_ok=True)
    rows, misses, detector_only = [], [], []
    seen = set()
    for row in annotated:
        key = row['capture']
        if key in seen:
            raise ValueError('Expected one annotated representative per capture')
        seen.add(key)
        source = sources.get(key)
        if source is None:
            detector_only.append(key)
            continue
        if source['split'] != row['split'] or Path(source['path']).resolve() != Path(row['path']).resolve():
            raise ValueError('Annotations must match original source paths and splits')
        index = row.get('target_box_index')
        if row.get('review_status') != 'reviewed' or type(index) is not int or not 0 <= index < len(row['boxes']):
            misses.append({'capture': key, 'split': row['split'], 'reason': 'Target not explicitly reviewed'})
            continue
        target = row['boxes'][index]['xyxy']
        paths = [Path(str(row['path']).replace('_f0.jpg', f'_f{i}.jpg')) for i in range(3)]
        if not str(row['path']).endswith('_f0.jpg') or any(not p.exists() for p in paths):
            misses.append({'capture': key, 'split': row['split'], 'reason': 'Missing original three-view capture'})
            continue
        prepared = []
        for path in paths:
            image = cv2.imread(str(path))
            candidates = sorted(detector.detect(image), key=lambda d: iou(d.box, target), reverse=True)
            if not candidates or iou(candidates[0].box, target) < .3:
                break
            if len(candidates) > 1 and iou(candidates[1].box, target) >= .3:
                break
            chosen = candidates[0]
            x0, y0, x1, y1 = chosen.box
            if min(x1-x0, y1-y0) < min_inspection_side:
                break
            crop, bounds = inspection_crop(image, chosen.box)
            prepared.append((path, crop, chosen, bounds))
            target = chosen.box
        if len(prepared) != 3:
            misses.append({'capture': key, 'split': row['split'], 'reason': 'Missed, ambiguous or undersized target in three-view detection'})
            continue
        category = source['category']
        if category not in ('good', 'bad', 'circle'):
            raise ValueError('Unknown brick category')
        directory = out/'data'/category
        directory.mkdir(parents=True, exist_ok=True)
        for path, crop, chosen, bounds in prepared:
            saved = directory/path.name
            if not cv2.imwrite(str(saved), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f'Could not save {saved}')
            rows.append({'path': str(saved), 'capture': key, 'category': category, 'split': row['split'],
                         'sha256': hashlib.sha256(saved.read_bytes()).hexdigest(), 'source_path': str(path),
                         'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'detection_box': chosen.box, 'crop_box': bounds, 'detector_confidence': chosen.confidence})
    missing_annotations = set(sources)-seen
    misses.extend({'capture': key, 'split': sources[key]['split'], 'reason': 'No annotation'} for key in sorted(missing_annotations))
    meta = {'crop_policy': 'brick bbox + 5% padding, clipped; aspect-preserving 320 letterbox fill 127',
            'source_manifest': str(Path(source_manifest).resolve()),
            'minimum_source_box_side': min_inspection_side, 'images': rows}
    (out/'supervised_manifest.json').write_text(json.dumps(meta, indent=2))
    normal_rows = [r for r in rows if r['split'] != 'train' or r['category'] == 'good']
    (out/'anomaly_manifest.json').write_text(json.dumps({**meta, 'images': normal_rows}, indent=2))
    report = {'source_captures': len(sources), 'cropped_captures': len(rows)//3,
              'detector_only_captures_without_mark_labels': detector_only,
              'misses': misses, 'misses_by_split': dict(Counter(r['split'] for r in misses)),
              'note': 'Report misses in end-to-end coverage; do not report retained-only accuracy as complete pipeline accuracy.'}
    (out/'coverage.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--source-manifest', required=True, help='Original representative f0 manifest with category and split')
    parser.add_argument('--detector-weights', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', required=True)
    parser.add_argument('--min-inspection-side', type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(build_crops(args.annotations, args.source_manifest,
                                BrickDetector(args.detector_weights, args.device), args.out, args.min_inspection_side), indent=2))


if __name__ == '__main__':
    main()
