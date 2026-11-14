"""Offline checks for the MolmoAct2 pipeline (no GPU, no Studio install needed).

    python -m unittest discover -s training/molmoact2 -p 'test_*.py'
Tests needing torch/pyarrow are skipped when those aren't installed.
"""
import importlib
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_data  # noqa: E402
import train  # noqa: E402


def has(module):
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


class Recipe(unittest.TestCase):
    def config(self, **kw):
        kw.setdefault("episodes", [0, 2, 5])
        kw.setdefault("validation_episodes", [2])
        return train.build_config(dataset="/data/snap", output="/out/run", gpus=4, **kw)

    @staticmethod
    def method_classes():
        """{class name: base names} from methods.py (it imports Studio, so read it with ast)."""
        import ast
        tree = ast.parse((HERE / "methods.py").read_text())
        return {n.name: [b.id for b in n.bases if isinstance(b, ast.Name)] for n in tree.body if isinstance(n, ast.ClassDef)}

    def test_methods_use_text128(self):
        classes = self.method_classes()
        def roots(name):
            return {b for base in classes[name] for b in (roots(base) if base in classes else {base})}
        for path in train.METHODS.values():
            self.assertEqual(roots(path.split(".", 1)[1]), {"MolmoAct2Text128"}, path)
        self.assertIn("from policy_text import MolmoAct2Text128", (HERE / "methods.py").read_text())

    def test_methods(self):
        a, b = self.config(method="ai2")["model"], self.config(method="studio")["model"]
        self.assertEqual(a["class_path"], "methods.MolmoAct2Ai2Split")
        self.assertEqual(b["class_path"], "methods.MolmoAct2StudioLoraFullExpert")
        self.assertEqual(b["init_args"]["lora_target_modules"], train.STUDIO_VLM_LORA_TARGETS)
        strip = lambda m: {k: v for k, v in m["init_args"].items() if k != "lora_target_modules"}
        self.assertEqual(strip(a), strip(b))
        with self.assertRaises(ValueError):
            self.config(method="full")

    def test_ai2_targets(self):
        import re
        r = re.compile(train.AI2_LORA_TARGETS)
        yes = ["backbone.model.transformer.blocks.0.self_attn.att_proj", "backbone.model.transformer.blocks.27.mlp.ff_out",
               "backbone.model.vision_backbone.image_vit.transformer.resblocks.3.attention.wo",
               "backbone.model.vision_backbone.image_vit.transformer.resblocks.3.feed_forward.w1"]
        no = ["backbone.model.vision_backbone.image_projector.w1", "backbone.model.vision_backbone.image_pooling_2d.wq",
              "backbone.model.action_expert.blocks.0.self_attn.qkv", "backbone.model.vision_backbone.image_vit.patch_embedding",
              "lm_head", "backbone.model.transformer.wte"]
        self.assertTrue(all(r.fullmatch(n) for n in yes))
        self.assertFalse(any(r.fullmatch(n) for n in no))

    def test_studio_vlm_targets(self):
        import re
        r = re.compile(train.STUDIO_VLM_LORA_TARGETS)
        yes = ["backbone.model.transformer.blocks.0.self_attn.att_proj", "backbone.model.vision_backbone.image_vit.patch_embedding",
               "backbone.model.vision_backbone.image_vit.transformer.resblocks.0.attention.wq",
               "backbone.model.vision_backbone.image_vit.transformer.resblocks.3.feed_forward.w2"]
        no = ["backbone.model.action_expert.blocks.0.self_attn.qkv", "backbone.model.action_expert.blocks.0.mlp.up_proj",
              "backbone.model.vision_backbone.image_projector.w3", "backbone.model.vision_backbone.image_pooling_2d.wq",
              "backbone.model.action_expert.final_layer.linear", "backbone.model.action_expert.context_k_proj",
              "lm_head", "backbone.model.transformer.wte"]
        self.assertTrue(all(r.fullmatch(n) for n in yes))
        self.assertFalse(any(r.fullmatch(n) for n in no))

    def test_shared_recipe(self):
        m = self.config(method="studio")["model"]["init_args"]
        self.assertEqual(m["pretrained_name_or_path"], "allenai/MolmoAct2-SO100_101")
        self.assertTrue(m["lora_enabled"])
        self.assertEqual(m["lora_rank"], 64)
        for key in ("optimizer_lr", "optimizer_vit_lr", "optimizer_connector_lr", "optimizer_action_expert_lr"):
            self.assertEqual(m[key], 5e-5, key)
        self.assertEqual((m["chunk_size"], m["n_action_steps"]), (30, 30))
        self.assertEqual(m["setup_type"], "single so100/so101 robotic arm in molmoact2")
        self.assertEqual(m["control_mode"], "absolute joint pose")
        self.assertTrue(m["adapt_to_so101"])
        self.assertNotIn("calibration", m)  # data is in degrees already: Studio scale must stay 1
        self.assertIsNone(m["scheduler_decay_steps"])  # cosine over the actual run

    def test_batch_data_and_augmentation(self):
        c = self.config()
        d = c["data"]["init_args"]
        self.assertEqual(c["data"]["class_path"], "phase_data.PhaseDataModule")
        # 128-token static text, trained in (policy_text.py); must match the export
        # 128-token static text: every method class must derive from policy_text.MolmoAct2Text128
        self.assertIn(c["model"]["class_path"], [f"methods.{k}" for k in self.method_classes()])
        self.assertNotIn("tokenizer_max_length", c["model"]["init_args"])  # not an init arg of Studio's class
        self.assertEqual(d["train_batch_size"] * c["trainer"]["devices"], 32)
        self.assertEqual((d["episodes"], d["validation_episodes"]), ([0, 2, 5], [2]))
        self.assertNotIn("val_split", d)  # explicit team split, applied by PhaseDataModule
        self.assertEqual(d["image_transforms"]["class_path"], "augment.Ai2Augment")
        self.assertEqual(c["trainer"]["max_steps"], 8000)
        paths = [cb["class_path"] for cb in c["trainer"]["callbacks"]]
        self.assertIn("progress.Progress", paths)

    def test_wandb_logger(self):
        self.assertNotIn("logger", self.config()["trainer"])
        t = self.config(wandb={"project": "p", "name": "m2a/run 1"})["trainer"]
        self.assertEqual(t["logger"]["class_path"], "lightning.pytorch.loggers.WandbLogger")
        self.assertEqual((t["logger"]["init_args"]["id"], t["logger"]["init_args"]["resume"]), ("m2a-run-1", "allow"))
        self.assertIs(self.config(smoke=True, wandb={"project": "p", "name": "x"})["trainer"]["logger"], False)

    def test_batch_must_divide(self):
        with self.assertRaises(ValueError):
            train.build_config(dataset="/d", output="/o", gpus=8, global_batch=20, episodes=[0, 1],
                              validation_episodes=[1])

    def test_smoke_has_no_checkpoints(self):
        t = self.config(smoke=True)["trainer"]
        self.assertEqual((t["fast_dev_run"], t["callbacks"], t["enable_checkpointing"]), (1, [], False))

    def test_versions_pinned(self):
        v = train.versions()
        self.assertRegex(v["STUDIO_COMMIT"], r"^[0-9a-f]{40}$")
        self.assertRegex(v["MOLMOACT2_REVISION"], r"^[0-9a-f]{40}$")
        self.assertEqual(v["MOLMOACT2_REPO"], train.RECIPE["pretrained_name_or_path"])
        pi05 = (HERE.parent / "pi05/versions.env").read_text()
        self.assertIn(v["STUDIO_COMMIT"], pi05)  # same Studio as the pi0.5 runs / XPS export env


class Data(unittest.TestCase):
    def test_studio_split(self):
        train_ids, val_ids = prepare_data.studio_split(100)
        expected = sorted(random.Random(42).sample(list(range(100)), 10))  # Studio datamodule, pinned commit
        self.assertEqual(val_ids, expected)
        self.assertEqual(sorted(train_ids + val_ids), list(range(100)))
        eps = [e for e in range(347) if e not in (77, 131, 284)]  # same rule as the team's CURATION.json split
        self.assertEqual(prepare_data.studio_split(eps)[1], sorted(random.Random(42).sample(eps, 34)))

    def labels(self, n=3):
        return [{"episode_index": i, "success": True, "pick_task": "pick up the red brick",
                 "inspect_task": prepare_data.INSPECT_TASK, "place_task": "place the brick in the purple bowl",
                 "grasp_frame": 30, "inspection_frame": 50, "n_frames": 100} for i in range(n)]

    def info(self, top=(480, 640, 3)):
        joint = {"shape": [6], "names": prepare_data.JOINTS}
        return {"codebase_version": "v3.0", "fps": 30, "total_episodes": 3, "total_frames": 300,
                "features": {"observation.state": joint, "action": joint,
                             "observation.images.wrist": {"shape": [480, 640, 3]},
                             "observation.images.top": {"shape": list(top)},
                             "observation.images.inspect": {"shape": [480, 640, 3]}}}

    def test_source_ok(self):
        prepare_data.check_source(self.info(), self.labels(), [0, 1, 2])
        labels = self.labels()
        labels[0]["pick_task"] = "pick up the brick"  # 517-set wording
        prepare_data.check_source(self.info(), labels, [0, 1, 2])

    def test_rejects_failed_take_and_bad_boundaries(self):
        labels = self.labels()
        labels[1]["success"] = False
        labels[2]["grasp_frame"] = 60  # after inspection
        with self.assertRaisesRegex(ValueError, r"\[1, 2\]"):
            prepare_data.check_source(self.info(), labels, [0, 1, 2])
        prepare_data.check_source(self.info(), labels, [0])  # unused episodes are not checked

    def test_rejects_already_cropped_top(self):
        with self.assertRaisesRegex(ValueError, "uncropped"):
            prepare_data.check_source(self.info(top=(480, 560, 3)), self.labels(), [0])

    def test_selection_drops_team_exclusions(self):
        labels = self.labels(5)
        labels[3].update(exclude_from_training=True, exclude_reason="already grasped at frame 0")
        self.assertEqual(prepare_data.select(labels), ([0, 1, 2, 4], {3: "already grasped at frame 0"}))
        self.assertEqual(prepare_data.select(labels, [1, 3])[0], [1])
        with self.assertRaises(ValueError):
            prepare_data.select(labels, [9])

    def test_team_split_restricted_to_selection(self):
        cur = {"train_episode_ids": [0, 1, 2, 4], "validation_episode_ids": [3, 5]}
        self.assertEqual(prepare_data.team_split(cur, [1, 3, 4]), ([1, 4], [3]))
        self.assertIsNone(prepare_data.team_split({"source_episodes": 3}, [0]))
        with self.assertRaisesRegex(ValueError, "neither"):
            prepare_data.team_split(cur, [1, 3, 7])

    def test_load_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "s.json"
            f.write_text(json.dumps({"total_shortlisted_episodes": 2, "episode_ids_sorted": [9, 4]}))
            self.assertEqual(prepare_data.load_selection(f), [4, 9])
            f.write_text("[3, 1]")
            self.assertEqual(prepare_data.load_selection(f), [1, 3])

    @unittest.skipUnless(has("pyarrow") and has("numpy"), "pyarrow/numpy not installed")
    def test_train_normalization_uses_train_episodes_only(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/chunk-000").mkdir(parents=True)
            ep = [0] * 4 + [1] * 4
            state = [[float(e * 100 + i)] * 6 for e, i in zip(ep, range(8))]
            pq.write_table(pa.table({"episode_index": ep, "observation.state": state, "action": state}),
                           root / "data/chunk-000/file-000.parquet")
            out = prepare_data.train_normalization(root, [0])
            self.assertEqual(out["observation.state"]["max"][0], 3.0)  # episode 1 (values >= 104) excluded
            self.assertEqual(out["action"]["count"], [4])
            self.assertAlmostEqual(out["action"]["q99"][0], float(np.quantile([0, 1, 2, 3], 0.99)))


@unittest.skipUnless(has("torch"), "torch not installed")
class Phase(unittest.TestCase):
    def setUp(self):
        # phase_data imports Studio at module level; stub just enough to import the pure function.
        stub = types.ModuleType("physicalai")
        for name in ("physicalai.data", "physicalai.data.lerobot", "physicalai.data.lerobot.dataset"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules.setdefault("physicalai", stub)
        sys.modules["physicalai.data.lerobot"].LeRobotDataModule = object
        sys.modules["physicalai.data.lerobot.dataset"]._LeRobotDatasetAdapter = object
        sys.modules["physicalai.data.lerobot"].datamodule = types.ModuleType("datamodule")
        self.phase = importlib.import_module("phase_data")

    def test_fixed_split(self):
        shim = self.phase.FixedSplit([7, 3])
        self.assertEqual(shim.Random(0).sample(list(range(10)), 2), [3, 7])
        with self.assertRaises(RuntimeError):
            shim.sample(list(range(10)), 3)
        for n_val, n in ((42, 428), (27, 250), (1, 5)):
            self.assertEqual(max(1, int(n * self.phase.val_fraction(n_val, n))), n_val)

    def test_chunks_stop_at_stage_boundaries(self):
        import torch
        action = torch.arange(30, dtype=torch.float32)[:, None].repeat(1, 6)
        hold = self.phase.hold_at_boundary
        held = hold(action, 55, (60, 100))  # pick stage, 5 steps before the grasp
        self.assertTrue(torch.equal(held[:5], action[:5]) and torch.all(held[5:] == action[4]))
        held = hold(action, 90, (60, 100))  # inspect stage, 10 steps before inspection
        self.assertTrue(torch.equal(held[:10], action[:10]) and torch.all(held[10:] == action[9]))
        self.assertTrue(torch.equal(hold(action, 100, (60, 100)), action))  # place stage
        self.assertTrue(torch.equal(hold(action, 10, (60, 100)), action))   # far from any boundary


@unittest.skipUnless(has("torchvision"), "torchvision not installed")
class Augment(unittest.TestCase):
    def test_shapes_and_range_preserved(self):
        import torch
        from augment import Ai2Augment
        aug = Ai2Augment()
        for h, w in ((480, 560), (480, 640)):
            out = aug(torch.rand(3, h, w))
            self.assertEqual(tuple(out.shape), (3, h, w))
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
