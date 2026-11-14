"""Train a whole-brick YOLO classifier with an explicit capture-level split."""
import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import torch
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer

from dlsj.inspect.yolo_classifier import LABEL_MAP, capture_predictions


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def prepare(source_manifest, destination):
    """Copy explicit splits; never allow frames of one capture to cross splits."""
    source = json.loads(Path(source_manifest).read_text())
    destination = Path(destination)
    rows = source['images']
    groups = defaultdict(list)
    for row in rows:
        groups[row['capture']].append(row)
    for capture, frames in groups.items():
        if len(frames) != 3 or len({(r['split'], r['category']) for r in frames}) != 1:
            raise ValueError(f'Capture must have three consistently labelled frames in one split: {capture}')
    counts = Counter()
    for row in rows:
        split = {'train': 'train', 'validation': 'val', 'test': 'test'}[row['split']]
        label = LABEL_MAP[row['category']]
        source_path = Path(row['path'])
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Source image changed: {source_path}')
        target = destination / split / label / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        counts[(split, label)] += 1
    dump(destination / 'manifest.json', {'source_manifest': str(Path(source_manifest).resolve()),
                                       'images': rows, 'split_unit': 'capture',
                                       'image_counts': {f'{s}/{c}': n for (s, c), n in counts.items()}})
    return rows


class WholeFaceTransform:
    """PIL RGB -> float CHW. Training transforms retain the complete source crop."""
    def __init__(self, size=320, training=False):
        self.size = size
        self.training = training
        self.augment = A.Compose([
            A.SafeRotate(limit=(-15, 15), border_mode=cv2.BORDER_CONSTANT, fill=(127, 127, 127), p=.7),
            A.PadIfNeeded(min_height=352, min_width=352, border_mode=cv2.BORDER_CONSTANT, fill=(127, 127, 127)),
            A.Affine(scale=(.95, 1.05), translate_percent=(-.02, .02), keep_ratio=True,
                     border_mode=cv2.BORDER_CONSTANT, fill=(127, 127, 127), p=.7),
            A.RandomBrightnessContrast(brightness_limit=.1, contrast_limit=.1, p=.6),
            A.GaussNoise(std_range=(1/255, 3/255), mean_range=(0, 0), p=.2),
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(.1, .5), p=.1),
            A.ImageCompression(quality_range=(90, 100), p=.2),
        ], seed=2026)

    def __call__(self, image):
        image = np.asarray(image.convert('RGB'))
        if image.shape[:2] != (320, 320):
            raise ValueError('Expected the complete 320×320 brick crop')
        if self.training:
            image = self.augment(image=image)['image']
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0


class BrickDataset(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_transforms = WholeFaceTransform(kwargs['args'].imgsz, kwargs.get('augment', False))


class BrickTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return BrickDataset(root=img_path, args=self.args, augment=mode == 'train',
                            prefix=mode, names=self.data['names'])


def frame_predictions(model, rows, split, device):
    chosen = [r for r in rows if r['split'] == split]
    transform = WholeFaceTransform()
    model = model.to(device).eval()
    output = []
    with torch.inference_mode():
        for offset in range(0, len(chosen), 16):
            batch = chosen[offset:offset + 16]
            inputs = []
            for row in batch:
                with Image.open(row['path']) as image:
                    inputs.append(transform(image))
            prediction = model(torch.stack(inputs).to(device))
            probabilities = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
            output.extend([{**row, 'probabilities': p.tolist()}
                           for row, p in zip(batch, probabilities.cpu().numpy())])
    return output




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weights', default='yolo26n-cls.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--name', default='train')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = prepare(args.source_manifest, args.out / 'dataset')
    torch.set_num_threads(4)
    model = YOLO(args.weights)
    start = time.perf_counter()
    model.train(data=str(args.out / 'dataset'), trainer=BrickTrainer,
                epochs=args.epochs, imgsz=320, batch=16, device=args.device,
                workers=0, seed=2026, deterministic=True, optimizer='AdamW', lr0=.001,
                weight_decay=.0005, patience=15, project=str(args.out), name=args.name,
                exist_ok=False, plots=False, amp=False, save=True,
                auto_augment=None, erasing=0., cache=False)
    elapsed = time.perf_counter() - start
    dump(args.out / f'{args.name}_summary.json', {'seconds': elapsed, 'epochs_requested': args.epochs,
          'device': args.device, 'seed': 2026, 'versions': {n: version(n) for n in
          ('ultralytics', 'torch', 'torchvision', 'albumentations', 'openvino', 'numpy')}})
    if args.smoke:
        print(f'SMOKE COMPLETE in {elapsed:.1f}s', flush=True)
        return
    best_path = args.out / args.name / 'weights/best.pt'
    best = YOLO(best_path)
    classes = [best.names[i] for i in range(len(best.names))]
    validation = frame_predictions(best.model, rows, 'validation', args.device)
    captures = capture_predictions(validation, classes)
    # Validation-only abstention thresholds: retain as many reliable decisions as
    # possible while reaching 95% validation accuracy, with >=50% coverage.
    candidates = []
    for confidence in (.5, .6, .7, .8, .9, .95):
        for margin in (0., .1, .2, .3):
            kept = [r for r in captures if r['confidence'] >= confidence and r['margin'] >= margin]
            accuracy = sum(r['actual'] == r['predicted'] for r in kept) / max(1, len(kept))
            if accuracy >= .95 and len(kept) >= .5 * len(captures):
                candidates.append((len(kept), accuracy, -confidence, -margin))
    if candidates:
        _, _, confidence, margin = max(candidates)
        confidence, margin = -confidence, -margin
    else:
        confidence, margin = .8, .2
    calibration = {'min_confidence': confidence, 'min_margin': margin,
                   'target_met': bool(candidates), 'basis': 'validation only; mean probabilities across three frames'}
    dump(args.out / 'validation_frames.json', validation)
    dump(args.out / 'validation_captures.json', captures)
    test = frame_predictions(best.model, rows, 'test', args.device)
    test_captures = capture_predictions(test, classes)
    dump(args.out / 'test_frames.json', test)
    dump(args.out / 'test_captures.json', test_captures)
    actual = [r['actual'] for r in test_captures]
    predicted = [r['predicted'] for r in test_captures]
    kept = [r for r in test_captures if r['confidence'] >= confidence and r['margin'] >= margin]
    evaluation = {'classes': classes, 'training_seconds': elapsed, 'best_checkpoint': str(best_path),
        'test': classification_report(actual, predicted, labels=classes, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(actual, predicted, labels=classes).tolist(),
        'calibration': calibration, 'test_confident_captures': len(kept),
        'test_confident_accuracy': sum(r['actual'] == r['predicted'] for r in kept) / max(1, len(kept)),
        'test_captures': len(test_captures),
        'limitations': ['Capture grouping only; physical brick/session identities unknown.',
                         'Existing test split has been inspected in earlier experiments.',
                         'Confidence is not a calibrated probability of correctness.',
                         'Fresh camera-session validation is still needed.']}
    dump(args.out / 'evaluation.json', evaluation)
    # Export after evaluation; deployment is tested separately against these scores.
    best.model.cpu()
    exported = best.export(format='openvino', imgsz=320, batch=1, dynamic=False, half=False, device='cpu')
    export_dir = Path(exported)
    manifest = {'backend': 'yolo_classification', 'model': 'model.xml', 'classes': classes,
                'input_size': 320, 'input_format': 'RGB float32 NCHW /255', 'aggregation': 'mean of three probability vectors',
                **calibration}
    xmls = list(export_dir.glob('*.xml'))
    if len(xmls) != 1:
        raise ValueError('Expected one OpenVINO model')
    manifest['model'] = xmls[0].name
    dump(export_dir / 'classifier.json', manifest)
    dump(args.out / 'deployment.json', {'directory': str(export_dir), 'manifest': manifest})
    print(json.dumps(evaluation, indent=2), flush=True)


if __name__ == '__main__':
    main()
