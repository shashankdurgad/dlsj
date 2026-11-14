"""Standalone OpenVINO runtime for the whole-brick YOLO classification export."""
import json
from collections import defaultdict
from pathlib import Path
import threading

import cv2
import numpy as np


LABEL_MAP = {'good': 'plain', 'bad': 'cross', 'circle': 'circle'}


def capture_predictions(rows, classes):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['category'], row['capture'])].append(row['probabilities'])
    output = []
    for (category, capture), values in sorted(groups.items()):
        if len(values) != 3:
            raise ValueError(f'Incomplete prediction group: {capture}')
        mean = np.mean(values, axis=0)
        order = np.argsort(mean)
        output.append({'capture': capture, 'actual': LABEL_MAP[category],
                       'predicted': classes[int(order[-1])], 'confidence': float(mean[order[-1]]),
                       'margin': float(mean[order[-1]] - mean[order[-2]]),
                       'probabilities': dict(zip(classes, map(float, mean)))})
    return output


def combine_decision(anomalous, probabilities, min_confidence, min_margin):
    """Agreement gates the decision; unknown and weak evidence request reinspection."""
    if set(probabilities) != {'plain', 'cross', 'circle'}:
        raise ValueError('Expected probabilities for plain, cross and circle')
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    label, confidence = ranked[0]
    margin = confidence - ranked[1][1]
    if confidence < min_confidence or margin < min_margin:
        return 'uncertain'
    if anomalous:
        return label if label in ('cross', 'circle') else 'uncertain'
    return 'plain' if label == 'plain' else 'uncertain'


def calibrate_joint_gate(captures, anomaly_scores, anomaly_threshold):
    """Validation-only calibration of the complete gate, not classifier confidence alone.

    Zero observed validation errors is a selection rule, not a correctness guarantee.
    Among settings with >=50% correct coverage, maximize correct decisions, then
    prefer the higher confidence/margin thresholds for ties.
    """
    if not captures or any(row['capture'] not in anomaly_scores for row in captures):
        raise ValueError('Every validation capture needs a matching anomaly score')
    candidates = []
    for confidence in (.5, .6, .7, .8, .9, .95):
        for margin in (0., .1, .2, .3):
            decisions = [combine_decision(anomaly_scores[row['capture']] >= anomaly_threshold,
                                          row['probabilities'], confidence, margin) for row in captures]
            wrong = sum(d != 'uncertain' and d != row['actual'] for d, row in zip(decisions, captures))
            correct = sum(d == row['actual'] for d, row in zip(decisions, captures))
            if wrong == 0 and correct >= .5 * len(captures):
                candidates.append((correct, confidence, margin))
    correct, confidence, margin = max(candidates) if candidates else (None, .8, .2)
    return {'min_confidence': confidence, 'min_margin': margin, 'target_met': bool(candidates),
            'calibration_scope': 'joint Anomalib/YOLO gate',
            'validation_correct_decisions': correct, 'validation_captures': len(captures),
            'basis': 'Validation only: zero observed joint errors, >=50% correct coverage; maximize coverage. No generalization guarantee.'}


class YoloClassifier:
    def __init__(self, directory, device='CPU'):
        import openvino as ov

        directory = Path(directory)
        self.config = json.loads((directory / 'classifier.json').read_text())
        if set(self.config['classes']) != {'plain', 'cross', 'circle'}:
            raise ValueError('This live tester expects plain/cross/circle classes')
        self.classes = self.config['classes']
        self.size = self.config['input_size']
        self.lock = threading.Lock()
        self.compiled = ov.Core().compile_model(
            str(directory / self.config['model']), device,
            {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32',
             'CACHE_DIR': str(directory / 'runtime-cache-f32')},
        )
        self.device = device

    def health(self):
        return {'model': self.config.get('name', 'YOLO26n brick classifier'), 'device': self.device,
                'classes': self.classes, 'min_confidence': self.config['min_confidence'],
                'min_margin': self.config['min_margin'],
                'calibration_target_met': self.config.get('target_met')}

    def predict(self, bgr):
        if bgr.shape != (320, 320, 3):
            raise ValueError('Expected a complete 320×320 BGR brick crop')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        with self.lock:
            probabilities = np.asarray(self.compiled([tensor])[0]).reshape(-1).copy()
        if (len(probabilities) != 3 or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=.001)):
            raise ValueError('YOLO export did not return three class probabilities')
        return dict(zip(self.classes, map(float, probabilities)))


class CombinedModel:
    """Attach per-frame class probabilities; browser averages each three-frame window."""
    def __init__(self, anomaly_model, classifier):
        self.anomaly_model = anomaly_model
        self.classifier = classifier
        self.name = anomaly_model.name + ' + YOLO classification'

    def health(self):
        return {**self.anomaly_model.health(), 'classifier': self.classifier.health()}

    def predict(self, payload):
        import time
        from dlsj.inspect.live_web import decode_frame

        start = time.perf_counter()
        result = self.anomaly_model.predict(payload)
        _, image = decode_frame(payload)
        result['class_probabilities'] = self.classifier.predict(image)
        result['latency_ms'] = round((time.perf_counter() - start) * 1000)
        return result
