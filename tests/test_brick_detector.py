import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dlsj.inspect.brick_detector import Detection, inspection_crop, select_target
from dlsj.inspect.brick_detection_train import prepare_annotations
from dlsj.inspect.live_web import AutoCropService, decode_frame


def test_crop_clamps_bounds_and_preserves_aspect_and_rgb_channels():
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[0:40, 0:80] = (17, 90, 220)
    crop, bounds = inspection_crop(frame, (0, 0, 80, 40), padding=0)
    assert bounds == (0, 0, 80, 40)
    assert crop.shape == (320, 320, 3)
    assert np.all(crop[:80] == 127) and np.all(crop[240:] == 127)
    assert tuple(crop[160, 160]) == (17, 90, 220)
    _, bounds = inspection_crop(frame, (0, 0, 80, 40))
    assert bounds == (0, 0, 84, 42)


@pytest.mark.parametrize('box', [(20, 0, 10, 10), (300, 0, 400, 40), (0, 0, float('nan'), 30)])
def test_invalid_crops_fail(box):
    with pytest.raises(ValueError):
        inspection_crop(np.zeros((100, 200, 3), np.uint8), box)


def test_selection_does_not_switch_to_larger_brick_when_ambiguous():
    a, b = Detection((10, 10, 40, 40), .8), Detection((60, 10, 150, 80), .99)
    zone = (0, 0, 200, 100)
    assert select_target([a, b], zone) is None
    assert select_target([a, b], zone, previous=(12, 12, 42, 42)) == a
    assert select_target([a, b], (0, 0, 50, 50)) == a
    assert select_target([], zone, previous=a.box) is None


def service_frame():
    frame = np.full((480, 640, 3), (10, 90, 230), np.uint8)
    _, encoded = cv2.imencode('.jpg', frame)
    return {'frame': base64.b64encode(encoded).decode()}


def test_exact_detected_crop_reaches_anomaly_only_when_enabled():
    received = []
    def predict(payload):
        received.append(payload)
        _, image = decode_frame(payload)
        assert image.shape == (320, 320, 3)
        return {'score': .3}
    detector = SimpleNamespace(detect=lambda f: [Detection((50, 100, 250, 200), .9)])
    service = AutoCropService(detector, SimpleNamespace(predict=predict))
    preview = service.predict(service_frame())
    assert preview['selected']['box'] == (50, 100, 250, 200)
    assert 'score' not in preview and not received
    service.inspection_ready = True
    prediction = service.predict(service_frame())
    assert prediction['score'] == .3 and received[0]['frame'] == prediction['crop']


def test_no_target_never_calls_anomaly():
    detector = SimpleNamespace(detect=lambda f: [])
    def fail(*args):
        pytest.fail('No brick should never reach Anomalib')
    result = AutoCropService(detector, SimpleNamespace(predict=fail), True).predict(service_frame())
    assert result['selected'] is None and 'score' not in result


@pytest.mark.parametrize('extra', [{'zone': [0, 0, 900, 300]}, {'previous_box': [0, 0, float('nan'), 30]}])
def test_detection_rejects_invalid_client_coordinates(extra):
    service = AutoCropService(SimpleNamespace(detect=lambda f: []), None)
    with pytest.raises(ValueError):
        service.predict({**service_frame(), **extra})


def annotations(tmp_path):
    rows = []
    for i, split in enumerate(('train', 'validation', 'test')):
        path = tmp_path/f'{split}.jpg'
        cv2.imwrite(str(path), np.full((100, 200, 3), 30+i*60, np.uint8))
        rows.append({'path': str(path), 'capture': split, 'split': split, 'width': 200, 'height': 100,
                     'review_status': 'reviewed', 'boxes': [{'xyxy': [20, 10, 120, 60], 'class_id': 0}]})
    path = tmp_path/'annotations.json'
    path.write_text(json.dumps({'class_names': ['brick'], 'images': rows}))
    return path, rows


def test_annotation_export_preserves_full_frames_and_box_coordinates(tmp_path):
    path, rows = annotations(tmp_path)
    dataset = prepare_annotations(path, tmp_path/'dataset')
    assert dataset.exists()
    labels = next((tmp_path/'dataset/labels/train').glob('*.txt')).read_text().split()
    assert list(map(float, labels)) == pytest.approx([0, .35, .35, .5, .5])
    manifest = json.loads((tmp_path/'dataset/manifest.json').read_text())['images']
    for row in manifest:
        assert hashlib.sha256(Path(row['dataset_image']).read_bytes()).hexdigest() == row['sha256']
    with pytest.raises(ValueError, match='empty output'):
        prepare_annotations(path, tmp_path/'dataset')


@pytest.mark.parametrize('change,match', [
    (lambda rows: rows[0]['boxes'][0].update(xyxy=[0, 0, 201, 90]), 'outside'),
    (lambda rows: rows[1].update(capture='train'), 'Capture leaked'),
    (lambda rows: rows[0].update(review_status='needs_review'), 'positive examples in train'),
    (lambda rows: rows[1].update(path=rows[0]['path']), 'Duplicate image'),
])
def test_bad_annotations_rejected_before_writing_dataset(tmp_path, change, match):
    path, rows = annotations(tmp_path)
    change(rows)
    path.write_text(json.dumps({'class_names': ['brick'], 'images': rows}))
    with pytest.raises(ValueError, match=match):
        prepare_annotations(path, tmp_path/'dataset')
    assert not (tmp_path/'dataset').exists()


def test_crop_dataset_keeps_three_views_and_reports_missing_targets(tmp_path):
    from dlsj.inspect.brick_crop_dataset import build_crops
    rows = []
    for category in ('good', 'bad'):
        for i in range(3):
            image = np.full((100, 200, 3), 50+i, np.uint8)
            cv2.imwrite(str(tmp_path/f'{category}_f{i}.jpg'), image)
        rows.append(dict(path=str(tmp_path/f'{category}_f0.jpg'), capture=category, category=category, split='train',
                         target_box_index=0 if category == 'good' else None,
                         review_status='reviewed', boxes=[dict(xyxy=[20, 10, 120, 60], class_id=0)]))
    ann, source = tmp_path/'ann.json', tmp_path/'source.json'
    ann.write_text(json.dumps(dict(images=rows)))
    source.write_text(json.dumps(dict(images=rows)))
    detector = SimpleNamespace(detect=lambda f: [Detection((20, 10, 120, 60), .9)])
    report = build_crops(ann, source, detector, tmp_path/'crops', min_inspection_side=32)
    assert report['source_captures'] == 2 and report['cropped_captures'] == 1
    assert report['misses'][0]['capture'] == 'bad'
    manifest = json.loads((tmp_path/'crops/anomaly_manifest.json').read_text())
    assert len(manifest['images']) == 3
    assert all(row['category'] == 'good' for row in manifest['images'])
    assert all(cv2.imread(row['path']).shape == (320, 320, 3) for row in manifest['images'])


def test_crop_dataset_rejects_capture_with_one_missed_view(tmp_path):
    from dlsj.inspect.brick_crop_dataset import build_crops
    for i in range(3):
        cv2.imwrite(str(tmp_path/f'capture_f{i}.jpg'), np.full((100, 200, 3), 50+i, np.uint8))
    row = dict(path=str(tmp_path/'capture_f0.jpg'), capture='one', category='circle', split='test',
               target_box_index=0, review_status='reviewed', boxes=[dict(xyxy=[20, 10, 120, 60])])
    ann = tmp_path/'ann.json';ann.write_text(json.dumps(dict(images=[row])))
    results = iter([[Detection((20, 10, 120, 60), .9)], []])
    report = build_crops(ann, ann, SimpleNamespace(detect=lambda f: next(results)), tmp_path/'crops')
    assert report['cropped_captures'] == 0
    assert report['misses_by_split'] == {'test': 1}
    assert json.loads((tmp_path/'crops/supervised_manifest.json').read_text())['images'] == []



def test_small_bricks_are_previewed_but_never_sent_to_anomaly():
    detector = SimpleNamespace(detect=lambda f: [Detection((20, 20, 50, 45), .95)])
    def fail(*args):
        pytest.fail('An undersized brick must not reach Anomalib')
    result = AutoCropService(detector, SimpleNamespace(predict=fail), True).predict(service_frame())
    assert result['selected'] is not None and result['crop']
    assert not result['inspection_ready'] and 'closer' in result['inspection_reason']
    assert 'score' not in result
