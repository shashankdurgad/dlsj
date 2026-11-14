"""Export a validation-selected brick detector and check FP32 CPU box parity.

Test evaluation is opt-in and refuses to overwrite an existing test report.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from dlsj.inspect.brick_detector import BrickDetector, iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True, help='Prepared detection dataset directory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.evaluate_test and (args.out/'test_metrics.json').exists():
        raise ValueError('Test was already evaluated; preserve its report')
    model = YOLO(str(args.weights))
    data = str((args.dataset/'data.yaml').resolve())
    validation = model.val(data=data, split='val', device='cpu', rect=False, plots=False,
                           project=str(args.out.resolve()), name='validation', exist_ok=False)
    (args.out/'validation_metrics.json').write_text(json.dumps(validation.results_dict, indent=2))
    if args.evaluate_test:
        test = model.val(data=data, split='test', device='cpu', rect=False, plots=True,
                         project=str(args.out.resolve()), name='test', exist_ok=False)
        (args.out/'test_metrics.json').write_text(json.dumps(test.results_dict, indent=2))
    exported = model.export(format='openvino', imgsz=640, batch=1, dynamic=False, half=False)
    pt, ov = BrickDetector(args.weights), BrickDetector(exported)
    rows = json.loads((args.dataset/'manifest.json').read_text())['images']
    records, count_mismatches, max_delta, max_confidence_delta = [], 0, 0, 0.
    for row in rows:
        if row['split'] != 'validation':
            continue
        image = cv2.imread(row['dataset_image'])
        a, b = pt.detect(image), ov.detect(image)
        count_mismatches += len(a) != len(b)
        remaining = list(b)
        for detection in a:
            if not remaining:
                break
            match = max(remaining, key=lambda d: iou(detection.box, d.box))
            remaining.remove(match)
            max_delta = max(max_delta, max(abs(x-y) for x, y in zip(detection.box, match.box)))
            max_confidence_delta = max(max_confidence_delta, abs(detection.confidence-match.confidence))
        records.append({'path': row['path'], 'pytorch': [asdict(d) for d in a],
                        'openvino': [asdict(d) for d in b]})
    if not records:
        raise ValueError('No validation images available for parity check')
    report = {'validation_images': len(records), 'count_mismatches': count_mismatches,
              'maximum_box_coordinate_difference_pixels': max_delta,
              'maximum_confidence_difference': max_confidence_delta,
              'passed': count_mismatches == 0 and max_delta <= 1 and max_confidence_delta <= .001,
              'device': 'CPU', 'precision': 'FP32', 'confidence_threshold': .5,
              'imgsz': 640, 'rect': False, 'export': str(exported), 'records': records}
    (args.out/'export_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
