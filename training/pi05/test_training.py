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


if __name__ == "__main__":
    unittest.main()
