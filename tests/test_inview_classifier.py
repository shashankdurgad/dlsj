import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import pytest

from dlsj.inspect_detect import BrickInViewDetector
from dlsj.inview_classifier import DEFAULT_MODEL, InViewClassifier, preprocess
from dlsj.teleop_record import make_detector

needs_model = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="trained model not in exports/")
FRAMES = Path(__file__).resolve().parent.parent / "data" / "inspect_frames"


def test_preprocess_matches_ultralytics_layout():
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, :, 2] = 255  # pure red in BGR
    x = preprocess(img, 224)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert x[0, 0].mean() == pytest.approx(1.0) and x[0, 2].mean() == pytest.approx(0.0)  # RGB order, 0-1


@needs_model
def test_stability_needs_consecutive_frames(monkeypatch):
    clf = InViewClassifier(DEFAULT_MODEL, "CPU", threshold=0.6, stable_frames=3)
    seq = iter([0.9, 0.9, 0.2, 0.9, 0.9, 0.9])
    monkeypatch.setattr(clf, "probability", lambda bgr: next(seq))
    img = np.zeros((480, 640, 3), np.uint8)
    got = [clf.update(img).in_view for _ in range(6)]
    assert got == [False, False, False, False, False, True]
    assert clf.debug_image(img).shape == img.shape


@needs_model
@pytest.mark.skipif(not FRAMES.exists(), reason="no local captures")
def test_real_captures_are_classified():
    clf = InViewClassifier(DEFAULT_MODEL, "CPU")
    for tag, want in (("in_view", True), ("not_in_view", False)):
        files = sorted(glob.glob(str(FRAMES / "*" / tag / "*.jpg")))[::5]
        if not files:
            pytest.skip("no captures for " + tag)
        hits = sum((clf.probability(cv2.imread(f)) >= 0.5) == want for f in files)
        assert hits / len(files) >= 0.9, (tag, hits, len(files))


def _args(**kw):
    d = dict(inspect_model="auto", inspect_device="CPU", inspect_threshold=0.6, inspect_stable_frames=None,
             inspect_roi="0.1,0.1,0.9,0.9", inspect_min_area=0.03, inspect_min_sat=100)
    d.update(kw)
    return argparse.Namespace(**d)


def test_make_detector_choice(tmp_path):
    assert isinstance(make_detector(_args(inspect_model="colour")), BrickInViewDetector)
    with pytest.raises(SystemExit):
        make_detector(_args(inspect_model=str(tmp_path / "missing.xml")))
    det = make_detector(_args())
    assert isinstance(det, InViewClassifier if DEFAULT_MODEL.exists() else BrickInViewDetector)
