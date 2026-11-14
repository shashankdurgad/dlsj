"""Tests for the pure helpers of the π0.5 OpenVINO optimisation study (deploy/pi05/opt/opt_lib.py)."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "pi05" / "opt"))
import opt_lib as L  # noqa: E402


def test_validation_episodes_match_training_split():
    ids = L.validation_episodes(212)
    assert len(ids) == 21
    assert ids == sorted(random.Random(42).sample(range(212), 21))
    assert all(0 <= i < 212 for i in ids)


def test_segments_split_on_task_change():
    tasks = ["pick"] * 3 + ["place"] * 2 + ["pick"]
    assert L.segments(tasks) == [(0, 3, "pick"), (3, 5, "place"), (5, 6, "pick")]
    assert L.segments([]) == []


def test_sample_frames_covers_every_segment_and_is_deterministic():
    eps = {0: ["pick"] * 100 + ["place"] * 50, 7: ["pick"] * 30 + ["place"] * 120}
    a = L.sample_frames(eps, 40, seed=3)
    assert a == L.sample_frames(eps, 40, seed=3)
    assert 35 <= len(a) <= 45
    seen = {(ep, task) for ep, _, task in a}
    assert seen == {(0, "pick"), (0, "place"), (7, "pick"), (7, "place")}
    for ep, f, task in a:
        assert eps[ep][f] == task


def test_sample_frames_gives_tiny_segments_one_frame():
    eps = {1: ["pick"] * 1000 + ["place"] * 2}
    picks = L.sample_frames(eps, 10)
    assert any(t == "place" for _, _, t in picks)


def test_future_actions_pads_and_masks():
    acts = np.arange(12, dtype=np.float32).reshape(6, 2)
    seg, mask = L.future_actions(acts, 4, chunk=5)
    assert seg.shape == (5, 2)
    assert mask.tolist() == [True, True, False, False, False]
    np.testing.assert_array_equal(seg[2:], np.repeat(acts[-1:], 3, axis=0))
    seg, mask = L.future_actions(acts, 0, chunk=3)
    assert mask.all()
    np.testing.assert_array_equal(seg, acts[:3])


def test_chunk_errors_masked_and_per_joint():
    target = np.zeros((2, 4, 6))
    pred = np.zeros((2, 4, 6))
    pred[:, :, 5] = 2.0  # gripper off by 2 everywhere
    mask = np.ones((2, 4), bool)
    mask[1, 2:] = False
    pred[1, 2:, :] = 100.0  # masked-out steps must not count
    e = L.chunk_errors(pred, target, mask, ks=(2, 4))
    assert e["k2"]["per_joint"]["gripper"]["mae"] == pytest.approx(2.0)
    assert e["k4"]["per_joint"]["gripper"]["mse"] == pytest.approx(4.0)
    assert e["k4"]["per_joint"]["shoulder_pan"]["mae"] == pytest.approx(0.0)
    assert e["k4"]["mae"] == pytest.approx(2.0 / 6)


def test_deviation_identical_and_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(3, 50, 6)) + 40
    d = L.deviation(ref, ref)
    assert d["mse"] == 0 and d["max_abs"] == 0
    assert d["cos_mean"] == pytest.approx(1.0) and d["cos_centered"] == pytest.approx(1.0)
    d = L.deviation(ref + 0.5, ref)
    assert d["mae"] == pytest.approx(0.5) and d["max_abs"] == pytest.approx(0.5)
    assert d["cos_mean"] > 0.99  # raw cosine hides the offset...
    assert d["cos_centered"] < d["cos_mean"]  # ...the centred one shows it


def test_latency_stats():
    s = L.latency_stats([10, 20, 30, 40, 1000])
    assert s["median_ms"] == 30 and s["min_ms"] == 10 and s["n"] == 5
    assert s["p99_ms"] > s["p90_ms"] > s["median_ms"]


def test_noise_is_fixed_per_index_and_distinct_across_indices():
    np.testing.assert_array_equal(L.noise_for(3), L.noise_for(3))
    assert not np.array_equal(L.noise_for(3), L.noise_for(4))
    assert L.noise_for(0).shape == (1, 50, 32) and L.noise_for(0).dtype == np.float32


def test_accept_gate():
    assert L.accept(1.02, 1.0, 0.3, 0.5)
    assert not L.accept(1.10, 1.0, 0.3, 0.5)
    assert not L.accept(1.00, 1.0, 0.6, 0.5)


def test_group_by():
    g = L.group_by([(1, 5, "a"), (2, 6, "b"), (1, 7, "a")], 0)
    assert g == {1: [(1, 5, "a"), (1, 7, "a")], 2: [(2, 6, "b")]}


@pytest.mark.parametrize("shape,label", [
    ([1, 50, 1024], "expert"), ([1, 8, 50, 1018], "expert"), ([1, 50, 32], "expert"),
    ([3, 256, 1152], "vision"), ([3, 256, 4304], "vision"),
    ([1, 968, 2048], "vlm"), ([1, 832, 16384], "vlm"), ([1, 8, 968, 256], "vlm"),
    ([1, 128, 1], "other"), ([7], "other"),
])
def test_classify_matmul(shape, label):
    assert L.classify_matmul(shape) == label


def test_per_sample_mae_masks_steps():
    pred = np.zeros((2, 4, 2))
    tgt = np.ones((2, 4, 2))
    mask = np.array([[1, 1, 1, 1], [1, 0, 0, 0]], bool)
    tgt[1, 1:] = 50  # masked
    np.testing.assert_allclose(L.per_sample_mae(pred, tgt, mask, k=4), [1.0, 1.0])


def test_paired_bootstrap_detects_shift_and_no_shift():
    rng = np.random.default_rng(1)
    b = rng.uniform(1, 10, 300)
    same = L.paired_bootstrap(b, b)
    assert same["rel"] == 0 and same["lo"] == 0 and same["hi"] == 0
    worse = L.paired_bootstrap(b * 1.1, b)
    assert worse["rel"] == pytest.approx(0.1) and worse["lo"] > 0.09


def test_jerk_zero_for_linear_and_masked():
    t = np.linspace(0, 1, 10)
    lin = np.stack([np.stack([t, 2 * t], axis=1)] * 2)  # [2,10,2]
    assert L.jerk(lin) == pytest.approx(0.0)
    zig = lin.copy()
    zig[:, ::2, :] += 1.0
    assert L.jerk(zig) == pytest.approx(2.0)
    mask = np.ones((2, 10), bool)
    mask[:, 3:] = False  # only steps 0..2 valid -> one second difference each
    assert L.jerk(zig, mask) == pytest.approx(2.0)


def test_range_ratio():
    tgt = np.zeros((2, 5, 1))
    tgt[:, -1] = 10
    pred = tgt * 0.5
    mask = np.ones((2, 5), bool)
    assert L.range_ratio(pred, tgt, mask) == pytest.approx(0.5)
    mask[:, -1] = False
    assert np.isnan(L.range_ratio(pred, tgt, mask))
