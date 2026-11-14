import base64
import io
import json
from contextlib import closing

import cv2
import numpy as np
import pytest

from dlsj.inspect.live_web import StudioModel


def model_files(tmp_path):
    (tmp_path/'studio_model.json').write_text(json.dumps({'id':'model1','project_id':'project1','name':'PatchCore'}))
    (tmp_path/'evaluation.json').write_text(json.dumps({'model_id':'model1','validation_selected_threshold':.683024}))
    return StudioModel(tmp_path)


def crop(size=320):
    ok, image = cv2.imencode('.jpg', np.full((size,size,3),90,np.uint8))
    assert ok
    return image.tobytes()


def test_prediction_forwards_exact_crop_and_uses_trained_model(tmp_path, monkeypatch):
    model=model_files(tmp_path)
    data=crop()
    requests=[]
    def response(request, timeout):
        requests.append(request)
        return closing(io.BytesIO(json.dumps({'score':.72,'anomaly_map':'png'}).encode()))
    monkeypatch.setattr('urllib.request.urlopen',response)
    result=model.predict({'frame':base64.b64encode(data).decode()})
    assert requests[0].full_url.endswith('/projects/project1/models/model1:predict?device=CPU')
    assert data in requests[0].data
    assert result['score']==.72 and result['anomaly_map']=='png'
    assert model.threshold==.683024


@pytest.mark.parametrize('payload',[{}, {'frame':'invalid!'}, {'frame':base64.b64encode(crop(200)).decode()}])
def test_invalid_crop_is_rejected_before_prediction(tmp_path,payload,monkeypatch):
    model=model_files(tmp_path)
    def unexpected(*args,**kwargs):
        pytest.fail('Invalid crop reached Studio')
    monkeypatch.setattr('urllib.request.urlopen',unexpected)
    with pytest.raises(ValueError):model.predict(payload)


def test_calibration_must_match_model(tmp_path):
    model_files(tmp_path)
    (tmp_path/'evaluation.json').write_text(json.dumps({'model_id':'another-model','validation_selected_threshold':.5}))
    with pytest.raises(ValueError,match='do not match'):StudioModel(tmp_path)


def test_remote_camera_preserves_jpeg_and_bounds_download(monkeypatch):
    from dlsj.inspect.live_web import fetch_camera_frame
    frame = crop(640)
    requests = []
    def response(url, timeout):
        requests.append((url, timeout))
        return closing(io.BytesIO(frame))
    monkeypatch.setattr('urllib.request.urlopen', response)
    assert fetch_camera_frame('http://camera/snapshot/inspect.jpg') == frame
    assert requests == [('http://camera/snapshot/inspect.jpg', 3)]


@pytest.mark.parametrize('data', [b'not an image', b'\xff\xd8broken', b'\xff\xd8' + b'x' * (5 * 1024 * 1024)])
def test_remote_camera_rejects_invalid_or_oversized_frames(monkeypatch, data):
    from dlsj.inspect.live_web import fetch_camera_frame
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: closing(io.BytesIO(data)))
    with pytest.raises(ValueError):
        fetch_camera_frame('http://camera/snapshot/inspect.jpg')
