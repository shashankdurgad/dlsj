import numpy as np
import pytest

from dlsj.inspect_detect import BrickInViewDetector, DetectorConfig

H, W = 240, 320


def scene(brick=None, colour=(0, 0, 220)):
    """Grey background (unsaturated) with an optional saturated BGR rectangle (x0,y0,x1,y1)."""
    img = np.full((H, W, 3), 120, np.uint8)
    img[200:, :] = (90, 95, 100)  # darker grey "table"
    if brick:
        x0, y0, x1, y1 = brick
        img[y0:y1, x0:x1] = colour
    return img


def det(**kw):
    return BrickInViewDetector(DetectorConfig(stable_frames=3, downscale=1, **kw))


def test_empty_scene_is_not_in_view():
    d = det()
    assert not d.update(scene()).in_view
    assert "no coloured blob" in d.last.reason


def test_held_brick_becomes_in_view_after_stable_frames():
    d = det()
    states = [d.update(scene((120, 80, 200, 160))) for _ in range(3)]
    assert [s.in_view for s in states] == [False, False, True]
    assert states[-1].candidate and 0.03 < states[-1].area < 0.7


def test_small_far_brick_ignored():
    d = det()
    for _ in range(5):
        r = d.update(scene((150, 120, 158, 128)))
    assert not r.in_view and "too small" in r.reason


def test_brick_cut_off_by_roi_edge_is_not_in_view():
    d = det(roi=(0.25, 0.25, 0.75, 0.75))
    for _ in range(5):
        r = d.update(scene((40, 80, 150, 160)))  # crosses the left ROI edge at x=80
    assert not r.in_view and "cut off" in r.reason


def test_moving_brick_resets_stability():
    d = det()
    d.update(scene((100, 80, 180, 160)))
    d.update(scene((100, 80, 180, 160)))
    r = d.update(scene((160, 60, 240, 140)))  # jumped
    assert not r.in_view and r.stable == 1


def test_works_for_several_duplo_colours():
    for bgr in [(0, 0, 220), (220, 60, 0), (0, 200, 0), (0, 220, 230)]:  # red, blue, green, yellow
        d = det()
        for _ in range(3):
            r = d.update(scene((120, 80, 200, 160), colour=bgr))
        assert r.in_view, bgr


def test_debug_image_and_roi_parsing():
    d = det()
    d.update(scene((120, 80, 200, 160)))
    out = d.debug_image(scene((120, 80, 200, 160)))
    assert out.shape == (H, W, 3)
    assert DetectorConfig.parse_roi("0.1,0.2,0.8,0.9") == (0.1, 0.2, 0.8, 0.9)
    with pytest.raises(ValueError):
        DetectorConfig.parse_roi("0.8,0.2,0.1,0.9")
