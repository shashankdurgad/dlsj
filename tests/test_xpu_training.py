"""Hardware-free tests for training/xpu (config generation for single-Intel-XPU Studio training)."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

XPU_DIR = Path(__file__).resolve().parents[1] / "training" / "xpu"
sys.path.insert(0, str(XPU_DIR))

import common  # noqa: E402
import train_pi05_xpu  # noqa: E402
import train_smolvla_xpu  # noqa: E402

TASKS = ["pick up the brick and hold it in front of the inspection camera", "place the brick in the yellow bowl"]


def fake_dataset(root: Path, n_episodes=10, frames=300, labels=False):
    (root / "meta").mkdir(parents=True)
    features = {f"observation.images.{name}": {"shape": [480, 640, 3], "dtype": "video"}
                for name in ("wrist", "top", "inspect")}
    features.update({key: {"shape": [6], "names": common.recipe.JOINTS} for key in ("action", "observation.state")})
    info = {"codebase_version": "v3.0", "fps": 30, "total_episodes": n_episodes,
            "total_frames": n_episodes * frames, "features": features}
    (root / "meta/info.json").write_text(json.dumps(info))
    if labels:
        (root / "meta/dlsj").mkdir()
        rows = [{"episode_index": i, "success": True, "n_frames": frames, "task": TASKS[i % 2]}
                for i in range(n_episodes)]
        (root / "meta/dlsj/labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return [{"episode_index": i, "length": frames, "tasks": [TASKS[i % 2]]} for i in range(n_episodes)]


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "ds"
    episodes = fake_dataset(root)
    with patch.object(common, "read_episodes", return_value=episodes):
        yield root


def run_plan(module, argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        assert module.main(argv) == 0
    text = out.getvalue()
    decoder = json.JSONDecoder()
    first, end = decoder.raw_decode(text)
    second, _ = decoder.raw_decode(text[end:].lstrip())
    return first, second["training_config"]


# ---------------------------------------------------------------- batch / trainer math

def test_accumulation_reaches_global_batch():
    assert common.global_batch_plan(2, 32, None) == (16, 32)
    assert common.global_batch_plan(3, 32, None) == (11, 33)  # rounds up, reports the real effective batch
    assert common.global_batch_plan(8, 4, None) == (1, 8)
    assert common.global_batch_plan(4, 32, 2) == (2, 8)  # explicit --accumulate wins
    with pytest.raises(ValueError):
        common.global_batch_plan(0, 32, None)


def test_bf16_uses_xpu_autocast_plugin_not_precision_flag(tmp_path):
    t = common.xpu_trainer({"precision": "bf16-mixed", "accelerator": "gpu", "devices": 4, "strategy": "auto"},
                           precision="bf16-mixed", accumulate=16, output=tmp_path, epochs=8, max_steps=None,
                           time_limit_min=None, smoke=False)
    assert (t["accelerator"], t["strategy"], t["devices"], t["num_nodes"]) == ("xpu", "xpu_single", 1, 1)
    assert "precision" not in t
    assert t["plugins"] == [{"class_path": common.AMP_PLUGIN,
                             "init_args": {"precision": "bf16-mixed", "device": "xpu"}}]
    assert t["accumulate_grad_batches"] == 16 and t["max_epochs"] == 8


def test_fp32_and_rejected_fp16(tmp_path):
    t = common.xpu_trainer({}, precision="32", accumulate=1, output=tmp_path, epochs=1, max_steps=None,
                           time_limit_min=None, smoke=False)
    assert t["precision"] == "32-true" and "plugins" not in t
    with pytest.raises(ValueError):
        common.xpu_trainer({}, precision="16-mixed", accumulate=1, output=tmp_path, epochs=1, max_steps=None,
                           time_limit_min=None, smoke=False)


def test_max_steps_time_limit_and_smoke(tmp_path):
    t = common.xpu_trainer({"callbacks": [{"init_args": {}}, {"init_args": {}}]}, precision="bf16-mixed",
                           accumulate=1, output=tmp_path, epochs=8, max_steps=500, time_limit_min=90, smoke=False)
    assert t["max_steps"] == 500 and t["max_epochs"] is None and t["max_time"] == "00:01:30:00"
    assert t["callbacks"][1]["init_args"]["dirpath"] == str(tmp_path / "checkpoints/resume")
    assert common.xpu_trainer({}, precision="32", accumulate=1, output=tmp_path, epochs=1, max_steps=None,
                              time_limit_min=1500, smoke=False)["max_time"] == "01:01:00:00"
    s = common.xpu_trainer({"callbacks": [{}]}, precision="32", accumulate=1, output=tmp_path, epochs=1,
                           max_steps=None, time_limit_min=30, smoke=True)
    assert s["fast_dev_run"] == 1 and s["callbacks"] == [] and s["logger"] is False
    assert s["enable_checkpointing"] is False and "max_time" not in s


def test_memory_estimate_expert_fits_full_does_not():
    args = train_pi05_xpu.build_parser().parse_args(["x"])
    assert train_pi05_xpu.estimate(args)["total_gib"] < 30
    full = train_pi05_xpu.build_parser().parse_args(["x", "--full-finetune"])
    assert train_pi05_xpu.estimate(full)["total_gib"] > 57


# ---------------------------------------------------------------- pi0.5 plan-only

def test_pi05_plan_only_without_labels_defaults(dataset, tmp_path):
    out = tmp_path / "run"
    plan, config = run_plan(train_pi05_xpu, [str(dataset), "--no-labels", "--plan-only", "--output", str(out)])
    assert not out.exists()  # plan-only writes nothing
    p = plan["plan"]
    assert p["episodes"] == 10 and p["validation_episode_ids"] == [1] and p["train_frames"] == 2700
    assert p["accumulate_grad_batches"] == 16 and p["effective_batch"] == 32
    assert p["optimizer_steps_per_epoch"] == 85  # 2700 // 2 = 1350 micro-batches / 16
    assert p["instructions"] == sorted(TASKS)
    m = config["model"]["init_args"]
    assert m["train_expert_only"] is True and m["freeze_vision_encoder"] is True
    assert m["lora_enabled"] is False and m["gradient_checkpointing"] is True and m["compile_model"] is False
    assert m["scheduler_warmup_steps"] == round(85 * 8 * 0.05)
    d = config["data"]["init_args"]
    assert d["root"] == str(dataset.resolve()) and d["train_batch_size"] == 2 and d["num_workers"] == 2
    assert d["video_backend"] == "pyav"
    t = config["trainer"]
    assert t["accelerator"] == "xpu" and t["strategy"] == "xpu_single" and t["devices"] == 1
    assert t["plugins"][0]["init_args"]["device"] == "xpu"


def test_pi05_lora_and_precision_32(dataset):
    _, config = run_plan(train_pi05_xpu, [str(dataset), "--no-labels", "--plan-only", "--lora", "--lora-rank", "16",
                                          "--precision", "32", "--batch", "4", "--global-batch", "16",
                                          "--max-steps", "40"])
    m = config["model"]["init_args"]
    assert m["lora_enabled"] is True and m["lora_rank"] == 16
    assert m["train_expert_only"] is False and m["freeze_vision_encoder"] is False  # Studio forbids the combo
    assert m["scheduler_warmup_steps"] == 2  # 5% of min(epoch plan, 40)
    t = config["trainer"]
    assert t["precision"] == "32-true" and t["accumulate_grad_batches"] == 4 and t["max_steps"] == 40


def test_labels_mode_reuses_cloud_validator(tmp_path):
    root = tmp_path / "ds"
    fake_dataset(root, labels=True)
    plan, _ = run_plan(train_pi05_xpu, [str(root), "--plan-only"])
    assert plan["plan"]["label_source"].startswith("meta/dlsj/labels.jsonl")
    lines = (root / "meta/dlsj/labels.jsonl").read_text().splitlines()
    bad = json.loads(lines[3]) | {"success": False}
    (root / "meta/dlsj/labels.jsonl").write_text("\n".join(lines[:3] + [json.dumps(bad)] + lines[4:]) + "\n")
    with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        train_pi05_xpu.main([str(root), "--plan-only"])


def test_labels_required_unless_no_labels(dataset):
    with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        train_pi05_xpu.main([str(dataset), "--plan-only"])


def test_dataset_still_recording_is_rejected(tmp_path):
    root = tmp_path / "ds"
    episodes = fake_dataset(root)
    with patch.object(common, "read_episodes", return_value=episodes[:-1]), \
            pytest.raises(ValueError, match="still being recorded"):
        common.dataset_plan(root, require_labels=False, batch=2, accumulate=1, epochs=1)
    with patch.object(common, "read_episodes", return_value=episodes[:-1] + [episodes[-1] | {"length": 5}]), \
            pytest.raises(ValueError, match="total_frames"):
        common.dataset_plan(root, require_labels=False, batch=2, accumulate=1, epochs=1)


def test_wrong_camera_set_is_rejected(tmp_path):
    root = tmp_path / "ds"
    episodes = fake_dataset(root)
    info = json.loads((root / "meta/info.json").read_text())
    del info["features"]["observation.images.inspect"]
    (root / "meta/info.json").write_text(json.dumps(info))
    with patch.object(common, "read_episodes", return_value=episodes), pytest.raises(ValueError, match="camera"):
        common.dataset_plan(root, require_labels=False, batch=2, accumulate=1, epochs=1)


# ---------------------------------------------------------------- launch path (mocked, no XPU)

def test_studio_commit_mismatch_stops_before_launch(dataset, tmp_path):
    with patch.object(common, "studio_commit", return_value="deadbeef"), \
            patch.object(common, "launch") as launch, pytest.raises(SystemExit), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        train_pi05_xpu.main([str(dataset), "--no-labels", "--output", str(tmp_path / "run")])
    launch.assert_not_called()
    assert common.expected_studio_commit() == "8f1324161ece2b24f2d38765b6e943ed1961258c"


def test_busy_machine_stops_before_launch(dataset, tmp_path):
    with patch.object(common, "check_studio", return_value=Path("/x/physicalai")), \
            patch.object(common, "busy_jobs", return_value=["123 python studio_export.py"]), \
            patch.object(common, "launch") as launch, pytest.raises(SystemExit), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        train_pi05_xpu.main([str(dataset), "--no-labels", "--output", str(tmp_path / "run")])
    launch.assert_not_called()


def test_resume_writes_config_and_uses_fit_namespace(dataset, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    with patch.object(common, "check_studio", return_value=Path("/x/physicalai")), \
            patch.object(common, "preflight", return_value=[]), \
            patch.object(common, "launch", return_value=0) as launch, contextlib.redirect_stdout(io.StringIO()):
        assert train_pi05_xpu.main([str(dataset), "--no-labels", "--output", str(out),
                                    "--resume", str(out / "last.ckpt")]) == 0
    command = launch.call_args.args[0]
    assert command[:3] == ["/x/physicalai", "fit", "--config"] and "--fit.ckpt_path" in command
    assert launch.call_args.kwargs["nice"] == 19 and launch.call_args.kwargs["min_free_gib"] == 8.0
    config = json.loads((out / "training-config.json").read_text())
    assert config["trainer"]["accelerator"] == "xpu"
    assert (out / "data-plan.json").exists()


# ---------------------------------------------------------------- SmolVLA

def test_smolvla_plan_only(dataset):
    plan, config = run_plan(train_smolvla_xpu, [str(dataset), "--no-labels", "--plan-only", "--smoke"])
    m = config["model"]["init_args"]
    assert config["model"]["class_path"] == "physicalai.policies.SmolVLA"
    assert m["num_cameras"] == 3 and m["train_expert_only"] is True and m["freeze_vision_encoder"] is True
    d = config["data"]["init_args"]
    assert d["val_gym"] is None and d["val_split"] == 0.1 and d["train_batch_size"] == 8
    t = config["trainer"]
    assert t["accelerator"] == "xpu" and t["strategy"] == "xpu_single" and t["accumulate_grad_batches"] == 8
    assert t["fast_dev_run"] == 1 and t["callbacks"] == []
    assert plan["memory_estimate"]["total_gib"] < 12
