"""Exploratory frozen-feature classifier; keep original test captures untouched."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import timm
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


LABELS = ['good', 'bad', 'circle']


def supervised_rows(rows, seed=2026):
    """Move only marked validation captures into supervised train/validation."""
    rng = np.random.default_rng(seed)
    assignments = {}
    for category in ('bad', 'circle'):
        captures = sorted({r['capture'] for r in rows
                           if r['category'] == category and r['split'] == 'validation'})
        rng.shuffle(captures)
        for i, capture in enumerate(captures):
            assignments[capture] = 'train' if i < int(.75 * len(captures)) else 'validation'
    return [{**r, 'original_split': r['split'],
             'split': assignments.get(r['capture'], r['split'])} for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = supervised_rows(json.loads(args.source_manifest.read_text())['images'])
    capture_splits = defaultdict(set)
    for row in rows:
        capture_splits[row['capture']].add(row['split'])
    if any(len(splits) != 1 for splits in capture_splits.values()):
        raise ValueError('A capture cannot occur in multiple splits')
    if any((r['original_split'] == 'test') != (r['split'] == 'test') for r in rows):
        raise ValueError('The original test split must remain untouched')
    (args.out / 'dataset.json').write_text(json.dumps({'images': rows}, indent=2))
    feature_file = args.out / 'features.npz'
    if feature_file.exists():
        saved = np.load(feature_file)
        if saved['paths'].tolist() != [r['path'] for r in rows]:
            raise ValueError('Cached feature paths differ from the source manifest')
        features = saved['features']
    else:
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        model = timm.create_model('wide_resnet50_2', pretrained=True,
                                 features_only=True, out_indices=(2, 3)).eval().to(device)
        transform = Compose([Resize((256, 256)), ToTensor(),
                             Normalize((.485, .456, .406), (.229, .224, .225))])
        features = []
        with torch.inference_mode():
            for offset in range(0, len(rows), 8):
                batch = []
                for row in rows[offset:offset + 8]:
                    with Image.open(row['path']) as image:
                        batch.append(transform(image.convert('RGB')))
                maps = model(torch.stack(batch).to(device))
                pooled = torch.cat([v for m in maps for v in
                                    (m.mean((2, 3)), m.amax((2, 3)))], dim=1)
                features.append(pooled.cpu().numpy())
                if offset % 80 == 0:
                    print('features', offset, '/', len(rows), 'device', device, flush=True)
        features = np.concatenate(features)
        np.savez_compressed(feature_file, features=features,
                            paths=np.array([r['path'] for r in rows]))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row['split'], row['category'], row['capture'])].append(i)
    keys = sorted(groups)
    if any(len(indices) != 3 for indices in groups.values()):
        raise ValueError('Expected three frames per capture')
    x = np.stack([features[groups[key]].mean(0) for key in keys])
    y = np.array([key[1] for key in keys])
    splits = np.array([key[0] for key in keys])
    train, validation, test = [splits == split for split in ('train', 'validation', 'test')]
    candidates = []
    for strength in (.001, .01, .1, 1.):
        classifier = make_pipeline(StandardScaler(), LogisticRegression(
            C=strength, class_weight='balanced', max_iter=2000, random_state=2026))
        classifier.fit(x[train], y[train])
        score = f1_score(y[validation], classifier.predict(x[validation]),
                         labels=LABELS, average='macro', zero_division=0)
        candidates.append((score, strength, classifier))
    score, strength, classifier = max(candidates, key=lambda item: item[0])
    prediction = classifier.predict(x[test])
    report = {
        'method': 'Frozen WideResNet50-2 layer2/layer3 mean+max features; mean of three views; scaled logistic regression',
        'weights': 'timm/wide_resnet50_2.racm_in1k',
        'seed': 2026, 'C': strength, 'validation_macro_f1': score,
        'validation_candidates': [{'C': c, 'macro_f1': s} for s, c, _ in candidates],
        'split_counts': {s: {c: sum(k[0] == s and k[1] == c for k in keys)
                             for c in LABELS} for s in ('train', 'validation', 'test')},
        'labels': LABELS,
        'test_confusion_matrix': confusion_matrix(y[test], prediction, labels=LABELS).tolist(),
        'test': classification_report(y[test], prediction, labels=LABELS, output_dict=True, zero_division=0),
        'limitations': ['Exploratory frozen-feature probe, not backbone fine-tuning.',
                         'Split by capture; physical brick/session identity is unknown.',
                         'No confidence threshold or joint Anomalib gating has been validated.',
                         'No live deployment or robot integration.'],
    }
    predictions = [{'capture': key[2], 'actual': key[1], 'predicted': str(p)}
                   for key, p in zip([k for k in keys if k[0] == 'test'], prediction)]
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2))
    (args.out / 'test_predictions.json').write_text(json.dumps(predictions, indent=2))
    joblib.dump(classifier, args.out / 'classifier.joblib')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
