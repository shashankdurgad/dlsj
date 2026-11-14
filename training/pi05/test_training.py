import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train


class TrainingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dataset"
        (self.root / "meta/dlsj").mkdir(parents=True)
        features = {f"observation.images.{name}": {"shape": [480, 640, 3], "dtype": "video"}
                    for name in ("wrist", "top", "inspect")}
        features.update({key: {"shape": [6], "names": train.JOINTS}
                         for key in ("action", "observation.state")})
        self.info = {"codebase_version": "v3.0", "fps": 30, "total_episodes": 10,
                     "total_frames": 3000, "features": features}
        self.labels = [{"episode_index": i, "success": True, "n_frames": 300,
                        "task": "place the brick on the blue plate"} for i in range(10)]
        self.write_dataset()

    def write_dataset(self):
        (self.root / "meta/info.json").write_text(json.dumps(self.info))
        (self.root / "meta/dlsj/labels.jsonl").write_text(
            "\n".join(json.dumps(label) for label in self.labels) + "\n")

    def test_plan_splits_episodes_and_preserves_global_batch(self):
        _, labels, _ = train.inspect_dataset(self.root)
        four = train.estimate(labels, 4, 8, 1, 8)
        eight = train.estimate(labels, 8, 4, 1, 8)
        self.assertEqual(four["validation_episode_ids"], [1])
        self.assertEqual(four["train_frames"], 2700)
        self.assertEqual(four["effective_batch"], eight["effective_batch"])
        self.assertEqual(four["optimizer_steps"], 672)
        self.assertEqual(four["optimizer_steps"], eight["optimizer_steps"])

    def test_failed_demonstrations_are_rejected(self):
        self.labels[3]["success"] = False
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "failure"):
            train.inspect_dataset(self.root)

    def test_joint_order_mismatch_is_rejected(self):
        self.info["features"]["action"]["names"] = list(reversed(train.JOINTS))
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "recorder order"):
            train.inspect_dataset(self.root)

    def test_incomplete_label_sidecar_is_rejected(self):
        self.labels.pop()
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "exactly one label"):
            train.inspect_dataset(self.root)

    def test_frame_mismatch_is_rejected(self):
        self.info["total_frames"] += 1
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "frame counts"):
            train.inspect_dataset(self.root)

    def test_dual_task_frames_preserve_inspection_boundary(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is required for frame-level validation")
        for label in self.labels:
            label.pop("task")
            label.update(pick_task="pick up the brick", place_task="place the brick",
                         inspection_frame=150)
        self.write_dataset()
        pq.write_table(pa.Table.from_pylist([
            {"task_index": 0, "task": "pick up the brick"},
            {"task_index": 1, "task": "place the brick"}]), self.root / "meta/tasks.parquet")
        (self.root / "data").mkdir()
        frames = [{"episode_index": i, "frame_index": j, "task_index": int(j >= 150)}
                  for i in range(10) for j in range(300)]
        path = self.root / "data/file.parquet"
        pq.write_table(pa.Table.from_pylist(frames), path)
        self.assertEqual(len(train.inspect_dataset(self.root)[1]), 10)
        frames[150]["task_index"] = 0
        pq.write_table(pa.Table.from_pylist(frames), path)
        with self.assertRaisesRegex(ValueError, "Frame task disagrees"):
            train.inspect_dataset(self.root)

    def test_three_stage_frames_preserve_grasp_and_inspection_boundaries(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is required for frame-level validation")
        for label in self.labels:
            label.pop("task", None)
            label.update(
                pick_task="pick up the brick",
                inspect_task="bring the brick to the inspection camera",
                place_task="place the brick in the orange bowl",
                grasp_frame=90,
                inspection_frame=150,
            )
        self.write_dataset()
        pq.write_table(pa.Table.from_pylist([
            {"task_index": 0, "task": "place the brick in the orange bowl"},
            {"task_index": 1, "task": "pick up the brick"},
            {"task_index": 4, "task": "bring the brick to the inspection camera"}]), self.root / "meta/tasks.parquet")
        (self.root / "data").mkdir(exist_ok=True)
        frames = [
            {"episode_index": i, "frame_index": j, "task_index": 1 if j < 90 else (4 if j < 150 else 0)}
            for i in range(10) for j in range(300)
        ]
        path = self.root / "data/file.parquet"
        pq.write_table(pa.Table.from_pylist(frames), path)
        self.assertEqual(len(train.inspect_dataset(self.root)[1]), 10)
        frames[90]["task_index"] = 1
        pq.write_table(pa.Table.from_pylist(frames), path)
        with self.assertRaisesRegex(ValueError, "Frame task disagrees"):
            train.inspect_dataset(self.root)

    def test_one_hour_run_freezes_backbone_and_limits_training(self):
        import one_hour
        output = Path(self.temp.name) / "bounded"
        revision = (train.HERE / "versions.env").read_text().split("STUDIO_COMMIT=", 1)[1].strip()
        argv = ["one_hour.py", str(self.root), "--pretrained-model", "/models/pi05_base",
                "--output", str(output)]
        with patch("sys.argv", argv), patch.object(one_hour.subprocess, "check_output", return_value=revision), \
             patch.object(one_hour.subprocess, "run"):
            one_hour.main()
        config = json.loads((output / "training-config.json").read_text())
        self.assertTrue(config["model"]["init_args"]["train_expert_only"])
        self.assertTrue(config["model"]["init_args"]["freeze_vision_encoder"])
        self.assertEqual(config["trainer"]["devices"], 4)
        timer = config["trainer"]["callbacks"][-2]
        self.assertEqual(timer["init_args"], {"duration": "00:00:50:00", "interval": "step"})

    def test_one_hour_full_run_unfreezes_backbone_and_saves_rarely(self):
        import one_hour
        output = Path(self.temp.name) / "bounded-full"
        revision = (train.HERE / "versions.env").read_text().split("STUDIO_COMMIT=", 1)[1].strip()
        argv = ["one_hour.py", str(self.root), "--pretrained-model", "/models/pi05_base",
                "--output", str(output), "--full"]
        with patch("sys.argv", argv), patch.object(one_hour.subprocess, "check_output", return_value=revision), \
             patch.object(one_hour.subprocess, "run"):
            one_hour.main()
        config = json.loads((output / "training-config.json").read_text())
        model = config["model"]["init_args"]
        self.assertFalse(model["train_expert_only"])
        self.assertFalse(model["freeze_vision_encoder"])
        self.assertEqual(model["scheduler_decay_steps"], 2200)
        self.assertEqual(config["trainer"]["max_steps"], 2200)
        best, resume = config["trainer"]["callbacks"][:2]
        self.assertTrue(best["init_args"]["save_weights_only"])
        self.assertEqual(resume["init_args"]["every_n_train_steps"], 500)
        plan = json.loads((output / "data-plan.json").read_text())
        self.assertEqual(plan["mode"], "pi05-full-finetune")

    def test_resume_uses_studio_fit_namespace_and_eight_gpu_topology(self):
        output = Path(self.temp.name) / "run"
        revision = (train.HERE / "versions.env").read_text().split("STUDIO_COMMIT=", 1)[1].strip()
        argv = ["train.py", str(self.root), "--gpus", "8", "--batch", "4",
                "--pretrained-model", "/models/pi05_base",
                "--output", str(output), "--resume", str(output / "last.ckpt")]
        with patch("sys.argv", argv), patch.object(train.subprocess, "check_output", return_value=revision), \
             patch.object(train.subprocess, "run") as run, patch.dict(train.os.environ, {}, clear=True), \
             contextlib.redirect_stdout(io.StringIO()):
            train.main()
        command = run.call_args.args[0]
        self.assertIn("--fit.ckpt_path", command)
        self.assertEqual(run.call_args.kwargs["env"]["NCCL_P2P_LEVEL"], "SYS")
        config = json.loads((output / "training-config.json").read_text())
        self.assertEqual(config["trainer"]["devices"], 8)
        self.assertEqual(config["model"]["init_args"]["pretrained_name_or_path"], "/models/pi05_base")
        callbacks = config["trainer"]["callbacks"]
        self.assertEqual(callbacks[1]["init_args"]["every_n_train_steps"], 250)
        self.assertNotEqual(callbacks[0]["init_args"]["dirpath"], callbacks[1]["init_args"]["dirpath"])
        self.assertEqual(callbacks[1]["init_args"]["dirpath"], str(output.resolve() / "checkpoints/resume"))
        self.assertEqual(config["data"]["init_args"]["root"], str(self.root.resolve()))
        self.assertEqual(config["model"]["init_args"]["scheduler_warmup_steps"], 34)

    def test_smoke_disables_checkpoint_callbacks_and_logging(self):
        output = Path(self.temp.name) / "smoke"
        revision = (train.HERE / "versions.env").read_text().split("STUDIO_COMMIT=", 1)[1].strip()
        argv = ["train.py", str(self.root), "--gpus", "1", "--smoke", "--output", str(output)]
        with patch("sys.argv", argv), patch.object(train.subprocess, "check_output", return_value=revision), \
             patch.object(train.subprocess, "run"), contextlib.redirect_stdout(io.StringIO()):
            train.main()
        config = json.loads((output / "training-config.json").read_text())
        self.assertEqual(config["trainer"]["fast_dev_run"], 1)
        self.assertFalse(config["trainer"]["enable_checkpointing"])
        self.assertFalse(config["trainer"]["logger"])
        self.assertEqual(config["trainer"]["callbacks"], [])

    def test_two_camera_check(self):
        with self.assertRaisesRegex(ValueError, "Expected wrist, top camera"):
            train.inspect_dataset(self.root, cameras=("wrist", "top"))
        del self.info["features"]["observation.images.inspect"]
        self.write_dataset()
        train.inspect_dataset(self.root, cameras=("wrist", "top"))
        with self.assertRaisesRegex(ValueError, "Expected wrist, top, inspect camera"):
            train.inspect_dataset(self.root)

    def two_camera_official_run(self, *extra, env=None):
        import full_official
        del self.info["features"]["observation.images.inspect"]
        self.write_dataset()
        (self.root / "CURATION.json").write_text(json.dumps({"included_episodes": list(range(9))}))
        output = Path(self.temp.name) / "official"
        revision = (train.HERE / "versions.env").read_text().split("STUDIO_COMMIT=", 1)[1].strip()
        argv = ["full_official.py", str(self.root), "--pretrained-model", "/models/pi05_base",
                "--output", str(output), *extra]
        with patch("sys.argv", argv), patch.object(full_official.subprocess, "check_output", return_value=revision), \
             patch.object(full_official.subprocess, "run") as run, \
             patch.dict(full_official.os.environ, {"STUDIO_DIR": "/studio", **(env or {})}, clear=True):
            full_official.main()
        return output, run

    def test_official_full_finetune_config(self):
        output, run = self.two_camera_official_run()
        config = json.loads((output / "training-config.json").read_text())
        model, trainer, data = (config[k] for k in ("model", "trainer", "data"))
        model = model["init_args"]
        self.assertFalse(model["train_expert_only"])
        self.assertFalse(model["freeze_vision_encoder"])
        self.assertFalse(model["lora_enabled"])
        self.assertEqual(model["optimizer_lr"], 2.5e-5)
        self.assertEqual(model["optimizer_betas"], [0.9, 0.95])
        self.assertEqual(model["optimizer_weight_decay"], 1e-10)
        self.assertEqual(model["optimizer_grad_clip_norm"], 1.0)
        self.assertEqual(model["scheduler_warmup_steps"], 1000)
        self.assertEqual(model["scheduler_decay_steps"], 30000)
        self.assertEqual(model["scheduler_decay_lr"], 2.5e-6)
        self.assertEqual(model["normalization_mode"], "QUANTILES")
        self.assertEqual(trainer["max_steps"], 20000)
        self.assertEqual(trainer["devices"] * data["init_args"]["train_batch_size"], 32)
        self.assertEqual(trainer["strategy"], "ddp")
        self.assertEqual(trainer["val_check_interval"], 1000)
        self.assertIsNone(trainer["check_val_every_n_epoch"])
        self.assertEqual(data["init_args"]["episodes"], list(range(9)))
        self.assertEqual(data["class_path"], "phase_data.PhaseDataModule")
        ckpt = trainer["callbacks"][0]
        self.assertEqual(ckpt["class_path"], "official_callbacks.OfficialCheckpoints")
        self.assertEqual(ckpt["init_args"]["resume_every"], 2000)
        source = (train.HERE / "official_callbacks.py").read_text()
        self.assertIn("SNAPSHOT_STEPS = (3000, 6000, 10000, 15000, 20000)", source)
        self.assertEqual(run.call_args.kwargs["env"]["NCCL_P2P_LEVEL"], "SYS")
        plan = json.loads((output / "data-plan.json").read_text())
        self.assertEqual(plan["cameras"], ["wrist", "top"])

    def test_official_four_gpus_keep_global_batch_and_resume_flag(self):
        output, run = self.two_camera_official_run("--gpus", "4", "--resume", "/ck/last.ckpt",
                                                   env={"RUN_DEADLINE": str(__import__("time").time() + 7200)})
        config = json.loads(next(output.glob("training-config-resume-*.json")).read_text())
        self.assertEqual(config["data"]["init_args"]["train_batch_size"], 8)
        timer = config["trainer"]["callbacks"][-1]
        self.assertEqual(timer["class_path"], "lightning.pytorch.callbacks.Timer")
        self.assertIn("--fit.ckpt_path", run.call_args.args[0])

    def test_official_smoke(self):
        import full_official
        cfg = full_official.build_config(dataset=self.root, pretrained_model="m", output=self.root / "o",
                                         episodes=[0], gpus=8, smoke=True)
        self.assertEqual(cfg["trainer"]["fast_dev_run"], 1)
        self.assertEqual(cfg["trainer"]["callbacks"], [])
        with self.assertRaises(ValueError):
            full_official.build_config(dataset=self.root, pretrained_model="m", output=self.root / "o",
                                       episodes=[0], gpus=3)


if __name__ == "__main__":
    unittest.main()
