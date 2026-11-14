"""OpenVINO export of ACT must reproduce LeRobot's preprocessor -> ACT -> postprocessor path."""

import json

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

from dlsj.deploy import act_export

CAMS = {"observation.images.wrist": (3, 48, 64), "observation.images.top": (3, 48, 64)}
S = 6


def _fake_stats():
    rng = np.random.default_rng(0)
    stats = {
        "observation.state": {"mean": rng.normal(0, 30, S), "std": rng.uniform(5, 40, S),
                              "min": -np.full(S, 100.0), "max": np.full(S, 100.0)},
        "action": {"mean": rng.normal(0, 30, S), "std": rng.uniform(5, 40, S),
                   "min": -np.full(S, 100.0), "max": np.full(S, 100.0)},
    }
    for k in CAMS:
        stats[k] = {"mean": rng.uniform(0.3, 0.6, (3, 1, 1)), "std": rng.uniform(0.1, 0.3, (3, 1, 1)),
                    "min": np.zeros((3, 1, 1)), "max": np.ones((3, 1, 1))}
    return {k: {s: np.asarray(v, dtype=np.float32) for s, v in d.items()} for k, d in stats.items()}


@pytest.fixture(scope="module")
def tiny_act(tmp_path_factory):
    torch.manual_seed(0)
    cfg = ACTConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (S,)),
                        **{k: PolicyFeature(FeatureType.VISUAL, v) for k, v in CAMS.items()}},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (S,))},
        chunk_size=10, n_action_steps=10, dim_model=64, n_heads=4, dim_feedforward=128,
        n_encoder_layers=1, n_decoder_layers=1, pretrained_backbone_weights=None, device="cpu",
    )
    policy = ACTPolicy(cfg).eval()
    pre, post = make_act_pre_post_processors(cfg, dataset_stats=_fake_stats())
    d = tmp_path_factory.mktemp("act") / "pretrained_model"
    policy.save_pretrained(d)
    pre.save_pretrained(d)
    post.save_pretrained(d)
    return d, policy, pre, post


def _obs(seed):
    rng = np.random.default_rng(seed)
    state = rng.normal(0, 30, S).astype(np.float32)
    imgs = {k: rng.integers(0, 255, (v[1], v[2], 3), dtype=np.uint8) for k, v in CAMS.items()}
    return state, imgs


def _reference(policy, pre, post, state, imgs):
    obs = {"observation.state": state, **imgs}
    batch = prepare_observation_for_inference(obs, torch.device("cpu"))
    with torch.no_grad():
        actions = policy.predict_action_chunk(pre(batch))
    return post(actions).numpy()[0]


@pytest.mark.parametrize("precision,atol", [("fp32", 1e-3), ("fp16", 0.5), ("int8", 1.5)])
def test_openvino_matches_lerobot(tiny_act, tmp_path, precision, atol):
    d, policy, pre, post = tiny_act
    xml = act_export.export(d, tmp_path / precision, precision=precision)
    manifest = json.loads((xml.parent / "manifest.json").read_text())
    assert manifest["output"]["shape"] == [1, 10, S]
    assert [i["name"] for i in manifest["inputs"]["images"]] == list(CAMS)

    runner = act_export.OpenVINOACT(xml.parent, device="CPU", cache_dir=None, inference_precision="f32")
    for seed in range(3):
        state, imgs = _obs(seed)
        ref = _reference(policy, pre, post, state, imgs)
        got = runner(state, imgs)
        assert got.shape == ref.shape == (10, S)
        # actions are in robot units (degrees, std up to 40), so these tolerances are tight
        assert np.max(np.abs(got - ref)) < atol, f"{precision}: max diff {np.max(np.abs(got - ref))}"


def test_benchmark_runs(tiny_act, tmp_path):
    d, *_ = tiny_act
    act_export.export(d, tmp_path / "b", precision="fp16")
    res = act_export.benchmark(tmp_path / "b", "CPU", iters=3, warmup=1, cache_dir=None)
    assert res["median_ms"] > 0


def test_crop_from_manifest_is_applied_to_full_frames(tiny_act, tmp_path):
    d, policy, pre, post = tiny_act
    top = "observation.images.top"
    _, h, w = CAMS[top]
    crop = {"box": [0, 0, w, h], "source": [h, w + 16]}
    xml = act_export.export(d, tmp_path / "crop", precision="fp32", crops={top: crop})
    assert json.loads((xml.parent / "manifest.json").read_text())["crops"] == {top: crop}
    runner = act_export.OpenVINOACT(xml.parent, device="CPU", cache_dir=None, inference_precision="f32")
    state, imgs = _obs(0)
    full = dict(imgs)
    full[top] = np.concatenate([imgs[top], np.zeros((h, 16, 3), np.uint8)], axis=1)  # camera is wider
    assert np.allclose(runner(state, full), runner(state, imgs), atol=1e-5)  # cropped back to what it trained on
    with pytest.raises(ValueError, match="expects"):
        runner(state, {**imgs, top: np.zeros((h + 8, w + 16, 3), np.uint8)})  # neither full nor cropped size
    with pytest.raises(ValueError, match="trained on"):
        act_export.export(d, tmp_path / "bad", crops={top: {"box": [0, 0, w - 4, h], "source": [h, w + 16]}})
    assert act_export.parse_crop("observation.images.top=0,0,560,480@640x480") == (
        top, {"box": [0, 0, 560, 480], "source": [480, 640]})
    with pytest.raises(ValueError, match="doesn't fit"):
        act_export.parse_crop("observation.images.top=100,0,560,480@640x480")
