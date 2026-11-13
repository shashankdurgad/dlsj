"""Hardware-free tests of the `inspect` defect detector: aggregation, uncertainty band, region, heatmap,
calibration, batching and the `Inspector` contract. The OpenVINO test builds a tiny stand-in IR and runs it on
the CPU; it is skipped when OpenVINO isn't installed. Nothing here needs Anomalib.
"""

import json

import cv2
import numpy as np
import pytest

from dlsj.contracts import InspectionResult, Label
from dlsj.inspect import core
from dlsj.inspect.core import Manifest
from dlsj.inspect.detector import AnomalibInspector

H, W = 480, 640
ROI = (160, 80, 480, 400)  # 320x320
IN = (64, 64)


def manifest(**kw) -> Manifest:
    d = dict(model="model.xml", model_name="Fake", input_size=IN, crop=ROI, threshold=0.5, band_low=0.4,
             band_high=0.6, map_min=0.0)
    d.update(kw)
    return Manifest(**d)


def frame_with_cross(cross_box=None) -> np.ndarray:
    """Grey background, red brick, optional white cross inside `cross_box` (frame pixels)."""
    f = np.full((H, W, 3), 128, np.uint8)
    cv2.rectangle(f, (220, 180), (420, 320), (200, 30, 30), -1)
    if cross_box is not None:
        x0, y0, x1, y1 = cross_box
        cv2.line(f, (x0, y0), (x1, y1), (255, 255, 255), 6)
        cv2.line(f, (x0, y1), (x1, y0), (255, 255, 255), 6)
    return f


class WhitenessBackend:
    """Fake model: anomaly map = 1 where an 8x8 cell holds a near-white pixel; score = map max."""

    def __init__(self):
        self.calls = []

    def __call__(self, batch: np.ndarray):
        self.calls.append(batch.shape)
        white = (batch.min(axis=-1) > 240).astype(np.float32)  # (B,H,W)
        b, h, w = white.shape
        maps = white.reshape(b, h // 8, 8, w // 8, 8).max(axis=(2, 4))
        return maps.max(axis=(1, 2)), maps


class ConstBackend:
    """Fake model returning preset per-frame scores (cycled) and flat maps."""

    def __init__(self, scores):
        self.scores, self.i = list(scores), 0

    def __call__(self, batch):
        s = np.array([self.scores[(self.i + k) % len(self.scores)] for k in range(len(batch))], np.float32)
        self.i += len(batch)
        return s, np.repeat(s[:, None, None], 8, 1).repeat(8, 2)


# ----------------------------------------------------------------------------- decision
def test_aggregate_is_median():
    assert core.aggregate([0.1, 0.9, 0.2]) == pytest.approx(0.2)
    assert core.aggregate([3.0]) == 3.0
    with pytest.raises(ValueError):
        core.aggregate([])


@pytest.mark.parametrize("scores,label", [
    ((0.1, 0.2, 0.15), Label.PERFECT),
    ((0.8, 0.9, 0.7), Label.DEFECTIVE),
    ((0.45, 0.5, 0.55), Label.UNCERTAIN),   # median inside the band
    ((0.39, 0.39, 0.39), Label.PERFECT),    # just below band_low
    ((0.61, 0.61, 0.61), Label.DEFECTIVE),  # just above band_high
    ((0.1, 0.9, 0.95), Label.UNCERTAIN),    # median defective, but one frame clearly perfect: frames disagree
    ((0.1, 0.12, 0.9), Label.UNCERTAIN),    # median perfect, one frame clearly defective
    ((0.1, 0.5, 0.2), Label.PERFECT),       # one frame in the band is not a disagreement
])
def test_decide_band_and_disagreement(scores, label):
    assert core.decide(core.aggregate(scores), scores, 0.4, 0.6) is label


def test_single_frame_cannot_disagree():
    assert not core.frames_disagree([0.9], 0.4, 0.6)
    assert core.frames_disagree([0.1, 0.9], 0.4, 0.6)


# ----------------------------------------------------------------------------- region / heatmap
def test_region_scales_lowres_box_to_frame():
    amap = np.zeros((32, 32), np.float32)
    amap[8:12, 16:24] = 1.0  # rows 8..11, cols 16..23 of a 32x32 map over a 320x320 ROI → 10 px per cell
    assert core.region_from_map(amap, 0.5, ROI, (320, 320)) == (160 + 160, 80 + 80, 160 + 240, 80 + 120)
    assert core.region_from_map(amap, 0.5, None, (320, 320)) == (160, 80, 240, 120)


def test_region_picks_component_with_the_peak():
    amap = np.zeros((32, 32), np.float32)
    amap[0:2, 0:2] = 0.6
    amap[20:30, 20:30] = 0.7
    amap[25, 25] = 0.9
    assert core.region_from_map(amap, 0.5, None, (32, 32)) == (20, 20, 30, 30)


def test_region_fallback_below_threshold_and_flat_map():
    amap = np.zeros((16, 16), np.float32)
    amap[4:6, 4:6] = 0.3
    assert core.region_from_map(amap, 0.5, None, (16, 16)) == (4, 4, 6, 6)
    assert core.region_from_map(np.zeros((16, 16), np.float32), 0.5, None, (16, 16)) is None


def test_heatmap_aligned_to_frame():
    amap = np.full((32, 32), 0.5, np.float32)
    heat = core.to_heatmap(amap, map_min=0.0, threshold=0.5, frame_hw=(H, W), roi=ROI)
    assert heat.shape == (H, W) and heat.dtype == np.float32
    assert heat[0, 0] == 0.0 and heat[100, 200] == pytest.approx(0.5)  # outside ROI 0; threshold → 0.5
    assert core.to_heatmap(amap * 10, 0.0, 0.5, (H, W), ROI).max() == 1.0  # clipped to [0, 1]


def test_crop_and_prepare():
    f = np.arange(H * W * 3, dtype=np.uint32).reshape(H, W, 3).astype(np.uint8)
    assert core.crop_roi(f, ROI).shape == (320, 320, 3)
    assert core.crop_roi(f, (600, 400, 700, 500)).shape == (80, 40, 3)  # clamped
    with pytest.raises(ValueError):
        core.crop_roi(f, (700, 0, 800, 10))
    assert core.prepare(f, ROI, (256, 256)).shape == (256, 256, 3)


# ----------------------------------------------------------------------------- calibration
def test_calibrate_separable():
    c = core.calibrate([1.0, 2.0, 3.0], [7.0, 8.0])
    assert c["threshold"] == pytest.approx(5.0)
    assert (c["band_low"], c["band_high"]) == (pytest.approx(4.0), pytest.approx(6.0))
    assert c["stats"]["method"] == "separable-gap"


def test_calibrate_overlap_band_covers_overlap():
    good, bad = [1.0, 2.0, 3.0, 6.0], [4.0, 5.0, 7.0, 8.0]
    c = core.calibrate(good, bad)
    assert c["band_low"] <= c["threshold"] <= c["band_high"]
    assert c["band_low"] <= 4.0 and c["band_high"] >= 6.0
    assert c["stats"]["method"] == "overlap-balanced-acc"


def test_calibrate_good_only():
    good = np.random.default_rng(0).normal(10, 1, 200)
    c = core.calibrate(good)
    assert c["threshold"] >= good.max() and c["band_low"] < c["threshold"] < c["band_high"]


def test_manifest_roundtrip(tmp_path):
    m = manifest(calibration={"method": "x"})
    m.save(tmp_path / "manifest.json")
    m2 = Manifest.load(tmp_path / "manifest.json")
    assert m2 == m and m2.crop == ROI and m2.input_size == IN


def test_group_by_brick(tmp_path):
    names = ["b1_f0.png", "b1_f1.png", "b1_f2.png", "b2_f0.png", "single.png", "notes.txt"]
    for n in names:
        (tmp_path / n).write_bytes(b"")
    g = core.group_by_brick(core.list_images(tmp_path))
    assert {k: len(v) for k, v in g.items()} == {"b1": 3, "b2": 1, "single": 1}


# ----------------------------------------------------------------------------- inspector
def test_inspector_contract_and_localisation():
    insp = AnomalibInspector(manifest(), WhitenessBackend(), batch=3, device="FAKE")
    cross = (300, 200, 360, 260)
    r = insp.inspect([frame_with_cross(cross)] * 3)
    assert isinstance(r, InspectionResult)
    assert r.label is Label.DEFECTIVE and r.device == "FAKE" and len(r.frame_scores) == 3 and r.latency_ms > 0
    x0, y0, x1, y1 = r.region
    # The box is in frame pixels and covers the cross (8-px cells of a 64-px map over a 320-px ROI = 40 px).
    assert x0 <= cross[0] + 10 and y0 <= cross[1] + 10 and x1 >= cross[2] - 10 and y1 >= cross[3] - 10
    assert x0 >= cross[0] - 45 and y0 >= cross[1] - 45 and x1 <= cross[2] + 45 and y1 <= cross[3] + 45
    assert r.heatmap.shape == (H, W) and r.heatmap.max() <= 1.0 and r.heatmap[0, 0] == 0.0

    plain = insp.inspect([frame_with_cross(None)] * 3)
    assert plain.label is Label.PERFECT and plain.region is None and plain.score == 0.0


def test_inspector_batches_and_pads():
    be = WhitenessBackend()
    AnomalibInspector(manifest(), be, batch=3).inspect([frame_with_cross()] * 3)
    assert be.calls == [(3, *IN, 3)]  # one call for 3 frames
    be = WhitenessBackend()
    r = AnomalibInspector(manifest(), be, batch=2).inspect([frame_with_cross()] * 3)
    assert be.calls == [(2, *IN, 3), (2, *IN, 3)] and len(r.frame_scores) == 3  # last chunk padded


def test_inspector_uncertain_when_frames_disagree():
    insp = AnomalibInspector(manifest(), ConstBackend([0.1, 0.9, 0.95]), batch=1, heatmap=False)
    r = insp.inspect([frame_with_cross()] * 3)
    assert r.label is Label.UNCERTAIN and r.score == pytest.approx(0.9) and r.heatmap is None
    assert r.region is not None  # UNCERTAIN still reports where the most suspicious spot is
    assert r.frame_scores == pytest.approx((0.1, 0.9, 0.95))


def test_synthetic_generator_layout(tmp_path):
    from dlsj.inspect import synthetic

    synthetic.generate(tmp_path, n_good=2, n_bad=1, views=3, seed=5)
    meta = json.loads((tmp_path / "dataset.json").read_text())
    assert meta["crop"] == list(synthetic.ROI)
    assert len(core.list_images(tmp_path / "good")) == 6 and len(core.list_images(tmp_path / "bad")) == 3
    img = core.read_rgb(core.list_images(tmp_path / "bad")[0])
    assert img.shape == (H, W, 3)
    assert len(core.group_by_brick(core.list_images(tmp_path / "good"))) == 2


def test_overlay_draws():
    insp = AnomalibInspector(manifest(), WhitenessBackend(), batch=3)
    f = frame_with_cross((300, 200, 360, 260))
    out = core.overlay(f, insp.inspect([f] * 3))
    assert out.shape == f.shape and not np.array_equal(out, f)


# ----------------------------------------------------------------------------- OpenVINO back-end
def _tiny_ir(path, hw):
    """Stand-in for the exported IR: u8 NHWC input; map = mean channel /255 (B,1,H,W); score = map max."""
    ov = pytest.importorskip("openvino")
    import openvino.opset13 as ops

    x = ops.parameter([1, *hw, 3], ov.Type.u8, name="input")
    f = ops.convert(x, ov.Type.f32)
    m = ops.reduce_mean(f, np.int64(3), keep_dims=True)  # (B,H,W,1)
    m = ops.multiply(ops.transpose(m, np.array([0, 3, 1, 2], np.int64)), np.float32(1 / 255))
    s = ops.reduce_max(m, np.array([1, 2, 3], np.int64), keep_dims=False)
    s.output(0).get_tensor().set_names({"pred_score"})
    m.output(0).get_tensor().set_names({"anomaly_map"})
    model = ov.Model([s, m], [x], "tiny")
    ov.save_model(model, str(path))


def test_openvino_backend_static_batch_on_cpu(tmp_path):
    _tiny_ir(tmp_path / "model.xml", IN)
    m = manifest(threshold=0.9, band_low=0.85, band_high=0.95)
    m.save(tmp_path / "manifest.json")
    insp = AnomalibInspector.from_manifest(tmp_path / "manifest.json", device="CPU", batch=3, fallback=("CPU",))
    assert insp.device == "CPU"
    assert all(d.is_static for d in insp.backend.compiled.input(0).partial_shape)
    assert list(insp.backend.compiled.input(0).shape) == [3, *IN, 3]
    r = insp.inspect([frame_with_cross((300, 200, 360, 260))] * 3)
    assert r.label is Label.DEFECTIVE and r.score == pytest.approx(1.0, abs=1e-3) and r.region is not None
    dark = np.zeros((H, W, 3), np.uint8)
    assert insp.inspect([dark] * 3).label is Label.PERFECT


def test_openvino_backend_falls_back(tmp_path):
    _tiny_ir(tmp_path / "model.xml", IN)
    manifest().save(tmp_path / "manifest.json")
    insp = AnomalibInspector.from_manifest(tmp_path / "manifest.json", device="NOPE", batch=1, fallback=("CPU",))
    assert insp.device == "CPU"
