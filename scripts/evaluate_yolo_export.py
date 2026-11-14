"""Check exported YOLO parity and evaluate the fixed Anomalib/YOLO gate."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dlsj.inspect.yolo_classifier import YoloClassifier, capture_predictions, combine_decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--anomaly-run', type=Path, required=True)
    args = parser.parse_args()
    deployment = json.loads((args.run_dir / 'deployment.json').read_text())
    directory = Path(deployment['directory'])
    classifier = YoloClassifier(directory)
    threshold = json.loads((args.anomaly_run / 'evaluation.json').read_text())['validation_selected_threshold']
    parity, joint = {}, {}
    for split in ('validation', 'test'):
        source = json.loads((args.run_dir / f'{split}_frames.json').read_text())
        predictions, differences, changes = [], [], 0
        for row in source:
            probabilities = classifier.predict(cv2.imread(row['path']))
            values = [probabilities[name] for name in classifier.classes]
            differences.append(float(np.max(np.abs(np.asarray(values) - row['probabilities']))))
            changes += int(np.argmax(values) != np.argmax(row['probabilities']))
            predictions.append({**row, 'probabilities': values})
        parity[split] = {'frames': len(source), 'maximum_probability_difference': max(differences),
                         'frame_label_changes': changes}
        (args.run_dir / f'{split}_openvino_frames.json').write_text(json.dumps(predictions, indent=2))
        if max(differences) > .001 or changes:
            raise ValueError(f'Export parity needs investigation: {parity[split]}')
        captures = capture_predictions(predictions, classifier.classes)
        anomaly_rows = json.loads((args.anomaly_run / f'{split}_predictions.json').read_text())
        groups = defaultdict(list)
        for row in anomaly_rows:
            groups[row['capture']].append(row['score'])
        capture_ids = {row['capture'] for row in captures}
        # Marked anomaly-validation captures were divided into supervised train
        # and validation. Score only the supervised validation subset here.
        if not capture_ids <= set(groups) or (split == 'test' and capture_ids != set(groups)):
            raise ValueError('Classifier and anomaly capture sets differ')
        decisions = []
        for row in captures:
            scores = groups[row['capture']]
            if len(scores) != 3:
                raise ValueError('Expected three anomaly scores per capture')
            anomaly_score = float(np.mean(scores))
            decision = combine_decision(anomaly_score >= threshold, row['probabilities'],
                                        classifier.config['min_confidence'], classifier.config['min_margin'])
            decisions.append({**row, 'anomaly_score': anomaly_score, 'decision': decision})
        decided = [r for r in decisions if r['decision'] != 'uncertain']
        correct = sum(r['decision'] == r['actual'] for r in decided)
        joint[split] = {'captures': len(decisions), 'decided': len(decided),
                        'uncertain': len(decisions) - len(decided), 'correct': correct,
                        'coverage': len(decided) / len(decisions),
                        'accuracy_among_decided': correct / max(1, len(decided)),
                        'correct_fraction_of_all_captures': correct / len(decisions),
                        'per_actual_class': {kind: dict(Counter(r['decision'] for r in decisions
                                             if r['actual'] == kind)) for kind in classifier.classes}}
        (args.run_dir / f'{split}_joint_predictions.json').write_text(json.dumps(decisions, indent=2))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in directory.iterdir() if p.suffix in ('.xml', '.bin')}
    parity['export_sha256'] = files
    parity['status'] = 'passed'
    (args.run_dir / 'export_validation.json').write_text(json.dumps(parity, indent=2))
    (args.run_dir / 'joint_evaluation.json').write_text(json.dumps(joint, indent=2))
    print(json.dumps({'parity': parity, 'joint': joint}, indent=2))


if __name__ == '__main__':
    main()
