"""OpenVINO-only brick detector helpers: ultralytics-style letterbox and one-class YOLO decode + NMS."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from dlsj.inspect.brick_detector import decode_yolo, letterbox  # noqa: E402


def test_letterbox_keeps_aspect_and_centres_with_grey_padding():
    img, r, (left, top) = letterbox(np.zeros((480, 640, 3), np.uint8), 640)
    assert img.shape == (640, 640, 3) and r == pytest.approx(1.0) and (left, top) == (0, 80)
    assert (img[:80] == 114).all() and (img[80:560] == 0).all() and (img[560:] == 114).all()


def _head(boxes):
    """boxes: [(cx, cy, w, h, score)] in letterboxed pixels -> (1, 5, N)."""
    return np.asarray(boxes, np.float32).T[None]


def test_decode_maps_back_to_the_frame_and_suppresses_overlaps():
    # frame 480x640 -> scale 1, 80 px top padding
    out = _head([(320, 320, 100, 60, .9), (322, 321, 100, 60, .8),  # duplicate of the first
                 (100, 500, 40, 40, .7), (500, 200, 30, 30, .3)])     # a second brick; one below confidence
    dets = decode_yolo(out, 1.0, (0, 80), (480, 640), confidence=.5, iou_threshold=.7)
    assert [d.box for d in dets] == [(270, 210, 370, 270), (80, 400, 120, 440)]
    assert dets[0].confidence == pytest.approx(.9)


def test_decode_drops_tiny_boxes_and_handles_no_detections():
    assert decode_yolo(_head([(10, 10, 5, 5, .9)]), 1.0, (0, 0), (480, 640)) == []
    assert decode_yolo(_head([(10, 10, 50, 50, .1)]), 1.0, (0, 0), (480, 640)) == []
