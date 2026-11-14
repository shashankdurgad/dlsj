"""Hardware-free tests for the ACT -> Physical AI Studio helpers (numpy only, no torch/LeRobot/Studio)."""

import numpy as np
import pytest

from dlsj.deploy.act_studio import (
    Crop,
    camera_name,
    check_studio_manifest,
    chunk_mae,
    diff_summary,
    sample_frames,
    studio_dataset_stats,
    studio_image_size,
)

CONFIG = {
    "input_features": {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
        "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 560]},
        "observation.images.inspect": {"type": "VISUAL", "shape": [3, 480, 640]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    "normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
}


def _norm():
    n = {"observation.state.mean": np.arange(6.0), "observation.state.std": np.full(6, 2.0),
         "action.mean": -np.arange(6.0), "action.std": np.full(6, 3.0), "action.q01": np.zeros(6)}
    for cam in ("wrist", "top", "inspect"):
        n[f"observation.images.{cam}.mean"] = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        n[f"observation.images.{cam}.std"] = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    return n


def test_camera_name():
    assert camera_name("observation.images.top") == "top"
    with pytest.raises(ValueError):
        camera_name("observation.state")
    with pytest.raises(ValueError):
        camera_name("observation.images.a.b")


def test_stats_conversion_renames_and_flattens():
    stats = studio_dataset_stats(CONFIG, _norm())
    assert set(stats) == {"state", "action", "wrist", "top", "inspect"}
    assert stats["state"]["type"] == "STATE" and stats["state"]["mean"] == list(np.arange(6.0))
    assert stats["action"]["std"] == [3.0] * 6
    assert stats["top"]["shape"] == (3, 480, 560)
    assert stats["top"]["mean"] == pytest.approx([0.485, 0.456, 0.406])
    assert all(s["name"] == k for k, s in stats.items())


def test_stats_conversion_refuses_bad_input():
    bad = _norm()
    del bad["observation.images.top.std"]
    with pytest.raises(ValueError, match="no mean/std"):
        studio_dataset_stats(CONFIG, bad)
    bad = _norm()
    bad["action.std"] = np.zeros(6)
    with pytest.raises(ValueError, match="non-positive"):
        studio_dataset_stats(CONFIG, bad)
    cfg = {**CONFIG, "normalization_mapping": {**CONFIG["normalization_mapping"], "ACTION": "MIN_MAX"}}
    with pytest.raises(ValueError, match="MEAN_STD"):
        studio_dataset_stats(cfg, _norm())


def test_crop_parse_metadata_and_apply():
    c = Crop.parse("0,0,560,480@640x480")
    assert c == Crop.from_metadata({"box_xywh": [0, 0, 560, 480], "source_hw": [480, 640]})
    assert c.cropped_hw == (480, 560)
    frame = np.arange(480 * 640 * 3, dtype=np.int64).reshape(480, 640, 3)
    out = c.apply_hwc(frame)
    assert out.shape == (480, 560, 3) and np.array_equal(out, frame[:, :560])
    chw = frame.transpose(2, 0, 1)[None]
    assert np.array_equal(c.apply_chw(chw), chw[..., :560])
    with pytest.raises(ValueError, match="expected a 480x640"):
        c.apply_hwc(frame[:, :560])  # already cropped frames are refused, not cropped twice
    with pytest.raises(ValueError, match="doesn't fit"):
        Crop.parse("100,0,560,480@640x480")
    with pytest.raises(ValueError, match="bad crop"):
        Crop.parse("0,0,560@640x480")


def test_studio_image_size_requires_one_full_frame_size():
    stats = studio_dataset_stats(CONFIG, _norm())
    top = Crop.parse("0,0,560,480@640x480")
    assert studio_image_size(stats, {"top": top}) == (480, 640)
    with pytest.raises(ValueError, match="different full-frame sizes"):
        studio_image_size(stats, {})  # uncropped: 480x560 and 480x640 can't share one letterbox size
    with pytest.raises(ValueError, match="trained on"):
        studio_image_size(stats, {"top": Crop.parse("0,0,600,480@640x480")})
    with pytest.raises(ValueError, match="doesn't use"):
        studio_image_size(stats, {"top": top, "side": top})


def _manifest(top_shape=(3, 480, 640), resize=(480, 640)):
    feats = [{"init_args": {"name": "state", "shape": [6]}}]
    for cam, shape in (("wrist", (3, 480, 640)), ("top", top_shape), ("inspect", (3, 480, 640))):
        feats.append({"init_args": {"name": f"images.{cam}", "shape": list(shape)}})
    return {"format": "policy_package", "policy": {"name": "act"},
            "model": {"artifacts": {"openvino": "act.xml"}, "input_features": feats,
                      "output_features": [{"init_args": {"name": "action", "shape": [100, 6]}}],
                      "preprocessors": [{"type": "to_float_tensor"},
                                        {"type": "resize", "image_resolution": list(resize), "mode": "letterbox"}]}}


def test_check_studio_manifest():
    cams = {"wrist": (480, 640), "top": (480, 640), "inspect": (480, 640)}
    assert check_studio_manifest(_manifest(), cams, 6, 100, 6) == []
    probs = check_studio_manifest(_manifest(top_shape=(3, 480, 560)), cams, 6, 100, 6)
    assert any("top" in p for p in probs)
    probs = check_studio_manifest(_manifest(resize=(480, 560)), cams, 6, 100, 6)
    assert any("resize" in p for p in probs)
    assert any("output" in p for p in check_studio_manifest(_manifest(), cams, 6, 50, 6))


def test_sample_frames_is_deterministic_and_respects_horizon():
    lengths = {0: 300, 1: 50, 2: 200, 3: 400}
    a = sample_frames(lengths, 20, seed=3, horizon=100)
    assert a == sample_frames(lengths, 20, seed=3, horizon=100)
    assert len(set(a)) == 20
    assert all(ep != 1 and 0 <= f < lengths[ep] - 100 for ep, f in a)
    assert {ep for ep, _ in a} == {0, 2, 3}  # spread over episodes
    with pytest.raises(ValueError):
        sample_frames({0: 50}, 1, horizon=100)


def test_chunk_mae_masks_steps_past_episode_end():
    pred = np.zeros((2, 4, 2))
    fut = np.ones((2, 4, 2))
    fut[1, 2:] = 100.0
    valid = np.ones((2, 4), bool)
    valid[1, 2:] = False
    assert np.allclose(chunk_mae(pred, fut, valid, 4), [1.0, 1.0])
    assert np.allclose(chunk_mae(pred, fut, valid, 1), [1.0, 1.0])
    with pytest.raises(ValueError):
        chunk_mae(pred, fut[:, :3], valid, 4)


def test_diff_summary():
    d = diff_summary(np.zeros((2, 3)), np.array([[0, 1, 0], [0, 0, -3.0]]))
    assert d["max_abs"] == 3.0 and d["mean_abs"] == pytest.approx(4 / 6)


def test_collapse_unstack_restack_is_exact():
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops

    from dlsj.deploy.act_studio import collapse_unstack_restack

    a = ops.parameter([4, 1, 3], np.float32, name="a")
    b = ops.parameter([1, 1, 3], np.float32, name="b")
    i0 = ops.constant(np.array([0], np.int64))
    rows = [ops.unsqueeze(ops.gather(a, ops.constant(np.int64(t)), ops.constant(np.int64(0))), i0) for t in range(4)]
    partial = [ops.unsqueeze(ops.gather(a, ops.constant(np.int64(t)), ops.constant(np.int64(0))), i0) for t in (1, 2)]
    out = ops.concat([b, *rows, *partial], 0)  # full run of a -> a; a partial run must be left alone
    model = ov.Model([out], [a, b])
    rng = np.random.default_rng(0)
    xa, xb = rng.random((4, 1, 3), np.float32), rng.random((1, 1, 3), np.float32)
    before = ov.Core().compile_model(model, "CPU")([xa, xb])[0]
    assert collapse_unstack_restack(model) == 1
    concat = next(o for o in model.get_ordered_ops() if o.get_type_name() == "Concat")
    assert concat.get_input_size() == 4  # b, a, a[1], a[2]
    after = ov.Core().compile_model(model, "CPU")([xa, xb])[0]
    assert after.shape == (7, 1, 3) and np.array_equal(before, after)
