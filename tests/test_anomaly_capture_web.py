import json
import base64

import cv2
import numpy as np
import pytest

from dlsj.inspect.capture_web import CaptureStore
from dlsj.inspect import core


def jpeg(value):
    ok, data = cv2.imencode(".jpg", np.full((48, 64, 3), value, np.uint8))
    assert ok
    return data.tobytes()


def test_saved_views_are_a_single_training_group_and_undo_preserves_old_data(tmp_path):
    old = tmp_path / "good/old_f0.jpg"
    old.parent.mkdir()
    old.write_bytes(jpeg(1))
    frames = iter([jpeg(20), jpeg(40), jpeg(60)])
    store = CaptureStore("unused", tmp_path, fetch=lambda: next(frames))
    store.save("bad")
    groups = core.group_by_brick(core.list_images(tmp_path / "bad"))
    assert len(groups) == 1
    assert len(next(iter(groups.values()))) == 3
    assert json.loads((tmp_path / "dataset.json").read_text())["frame_size"] == [64, 48]
    assert store.snapshot()["counts"] == {"good": 1, "bad": 1, "circle": 0}
    store.undo()
    assert old.exists()
    assert not list((tmp_path / "bad").iterdir())


def test_invalid_feed_leaves_no_training_images(tmp_path):
    store = CaptureStore("unused", tmp_path, fetch=lambda: b"not an image")
    with pytest.raises(ValueError, match="valid image"):
        store.save("good")
    assert not list((tmp_path / "good").iterdir())


def test_frozen_feed_is_not_saved_as_multiple_views(tmp_path, monkeypatch):
    store = CaptureStore("unused", tmp_path, fetch=lambda: jpeg(20))
    times = iter([0, 0, 6])
    monkeypatch.setattr("dlsj.inspect.capture_web.time.monotonic", lambda: next(times))
    with pytest.raises(ValueError, match="frozen"):
        store.save("good")
    assert not list((tmp_path / "good").iterdir())


def test_browser_upload_records_selected_camera_and_three_views(tmp_path):
    store = CaptureStore("unused", tmp_path)
    store.save_upload("good", {"frames": [base64.b64encode(jpeg(v)).decode() for v in (10, 20, 30)],
                               "camera": "MacBook camera"})
    assert len(list((tmp_path / "good").glob("*.jpg"))) == 3
    meta = json.loads((tmp_path / "dataset.json").read_text())
    assert meta["camera"] == "MacBook camera"
    assert meta["source"] == "browser-camera"
    assert meta["source_url"] is None
    store.undo()
    assert not list((tmp_path / "good").iterdir())


def test_invalid_browser_batch_leaves_no_partial_images(tmp_path):
    store = CaptureStore("unused", tmp_path)
    with pytest.raises(ValueError):
        store.save_upload("bad", {"frames": [base64.b64encode(jpeg(10)).decode(), "invalid!", "invalid!"]})
    assert not list((tmp_path / "bad").iterdir())


def test_circle_capture_has_separate_images_count_and_undo(tmp_path):
    store = CaptureStore("unused", tmp_path)
    payload = {"frames": [base64.b64encode(jpeg(v)).decode() for v in (10, 20, 30)]}
    store.save_upload("good", payload)
    store.save_upload("bad", payload)
    assert "circle-marked" in store.save_upload("circle", payload)
    assert store.snapshot()["counts"] == {"good": 1, "bad": 1, "circle": 1}
    assert len(list((tmp_path / "circle").glob("*.jpg"))) == 3
    store.undo()
    assert store.snapshot()["counts"] == {"good": 1, "bad": 1, "circle": 0}
    assert not list((tmp_path / "circle").iterdir())


def test_unknown_category_is_rejected(tmp_path):
    store = CaptureStore("unused", tmp_path)
    with pytest.raises(ValueError, match="Choose"):
        store.save_upload("unknown", {})
    assert not (tmp_path / "unknown").exists()


def test_delete_last_photo_survives_restart_and_preserves_other_views(tmp_path):
    store = CaptureStore("unused", tmp_path)
    payload = {"frames": [base64.b64encode(jpeg(v)).decode() for v in (10, 20, 30)]}
    store.save_upload("circle", payload)
    images = sorted((tmp_path / "circle").glob("*.jpg"))
    store = CaptureStore("unused", tmp_path)
    assert store.snapshot()["can_delete_photo"]
    store.delete_photo()
    assert images[0].exists() and images[1].exists()
    assert not images[2].exists()
    meta = next((tmp_path / "circle").glob("*.json"))
    assert json.loads(meta.read_text())["views"] == 2
    store.delete_photo()
    store.delete_photo()
    assert not list((tmp_path / "circle").iterdir())
    assert not store.snapshot()["can_delete_photo"]
    assert store.snapshot()["counts"]["circle"] == 0
    assert store.delete_photo() == "No photos to delete"
