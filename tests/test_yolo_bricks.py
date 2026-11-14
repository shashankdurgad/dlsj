import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from dlsj.inspect.yolo_classifier import combine_decision
from dlsj.inspect.yolo_train import WholeFaceTransform, capture_predictions, prepare


def test_cross_split_capture_is_rejected_before_copying(tmp_path):
    rows = [{'capture': 'same', 'split': split, 'category': 'good'}
            for split in ('train', 'validation', 'test')]
    manifest = tmp_path / 'input.json'
    manifest.write_text(json.dumps({'images': rows}))
    with pytest.raises(ValueError, match='one split'):
        prepare(manifest, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_prepared_test_images_are_byte_identical_and_have_correct_labels(tmp_path):
    rows = []
    for index in range(3):
        path = tmp_path / f'capture_f{index}.jpg'
        cv2.imwrite(str(path), np.full((320, 320, 3), 100 + index, np.uint8))
        rows.append({'capture': 'capture', 'split': 'test', 'category': 'bad',
                     'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = tmp_path / 'input.json'
    manifest.write_text(json.dumps({'images': rows}))
    prepare(manifest, tmp_path / 'out')
    for row in rows:
        saved = tmp_path / 'out/test/cross' / Path(row['path']).name
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == row['sha256']


def test_validation_transform_preserves_edge_mark_and_rgb_order():
    rgb = np.zeros((320, 320, 3), np.uint8)
    rgb[:5, :5, 0] = 255
    tensor = WholeFaceTransform()(Image.fromarray(rgb)).numpy()
    assert tensor.shape == (3, 320, 320)
    assert np.all(tensor[0, :5, :5] == 1)
    assert np.all(tensor[1:] == 0)


@pytest.mark.parametrize('anomaly,probs,expected', [
    (False, {'plain': .95, 'cross': .03, 'circle': .02}, 'plain'),
    (True, {'plain': .03, 'cross': .95, 'circle': .02}, 'cross'),
    (True, {'plain': .03, 'cross': .02, 'circle': .95}, 'circle'),
    (True, {'plain': .95, 'cross': .03, 'circle': .02}, 'uncertain'),
    (False, {'plain': .03, 'cross': .95, 'circle': .02}, 'uncertain'),
    (True, {'plain': .1, 'cross': .46, 'circle': .44}, 'uncertain'),
])
def test_combined_gate_handles_agreement_disagreement_and_weak_evidence(anomaly, probs, expected):
    assert combine_decision(anomaly, probs, .8, .2) == expected


def test_capture_prediction_averages_three_frames_in_declared_class_order():
    rows = [{'category': 'bad', 'capture': 'capture', 'probabilities': p}
            for p in ([.1, .8, .1], [.2, .7, .1], [.1, .7, .2])]
    result = capture_predictions(rows, ['circle', 'cross', 'plain'])[0]
    assert result['actual'] == result['predicted'] == 'cross'
    assert result['confidence'] == pytest.approx(2.2 / 3)
