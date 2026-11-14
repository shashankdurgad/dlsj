"""Zero-shot OpenVINO export of allenai/MolmoAct2-SO100_101 for our SO-101 (wrist + top cameras), optional RTC.

Runtime units are our controller's LeRobot degrees (body joints) and 0-100 (gripper). Studio maps runtime joints
into the checkpoint's legacy frame as `checkpoint = sign * scale * runtime + offset`; with LeRobot degrees the scale
must be exactly 1. Studio only derives the scale from an SO-101 calibration (`range_width / 200 * 360 / 4095`), so a
synthetic calibration with `range_max - range_min = 2275` ticks (2275 * 360 / (200 * 4095) = 1) is passed: it yields
scale 1 on every body joint, and copying the checkpoint's state/action normalisation is then allowed.

    HF_HUB_OFFLINE=1 HF_HOME=~/molmobench/hf ~/studio-venv/bin/python export.py \
        --out ~/molmo-exports/molmoact2_so101_rtc_fp16 --rtc --int8 ~/molmo-exports/molmoact2_so101_rtc_int8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch

UNIT_SCALE_CALIBRATION = {
    j: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 2275}
    for i, j in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
} | {"gripper": {"id": 6, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}}


def build(pretrained: str, rtc: bool, cams: list[str], h: int, w: int, ckpt: str | None = None,
          steps: int | None = None, image_size: int | None = None, text_len: int | None = None):
    from physicalai.data.observation import Feature, FeatureType
    from physicalai.policies import MolmoAct2
    from physicalai.policies.molmoact2.so101 import so101_joint_scales

    scales = so101_joint_scales(UNIT_SCALE_CALIBRATION)
    assert all(abs(s - 1.0) < 1e-9 for s in scales), scales
    if ckpt:  # a Studio Lightning checkpoint (e.g. our LoRA fine-tune; Studio merges adapters at export)
        policy = MolmoAct2.load_from_checkpoint(ckpt, map_location="cpu", weights_only=False)
        _apply_variant(policy, steps, image_size, text_len)
        return policy.to(torch.float32).eval()
    policy = MolmoAct2(pretrained_name_or_path=pretrained, norm_tag="so100_so101_molmoact2", adapt_to_so101=True,
                       calibration=UNIT_SCALE_CALIBRATION, use_random_input_noise=True,
                       n_action_steps=30)  # RTC needs n_action_steps == chunk (the tag's action_horizon, 30)
    cfg = policy.config
    old_in = list(cfg.input_features or [])
    old_out = list(cfg.output_features or [])
    print("tag input features:", [(f.name, f.ftype, f.shape) for f in old_in])
    print("tag output features:", [(f.name, f.ftype, f.shape) for f in old_out])
    state = next(f for f in old_in if f.ftype == FeatureType.STATE)
    action = next(f for f in old_out if f.ftype == FeatureType.ACTION)
    inputs = [Feature(ftype=FeatureType.VISUAL, shape=(3, h, w), name=c) for c in cams]
    inputs.append(Feature(ftype=FeatureType.STATE, shape=state.shape or (6,), name=state.name))
    outputs = [Feature(ftype=FeatureType.ACTION, shape=action.shape or (6,), name=action.name)]
    policy.set_features(inputs, outputs, copy_state_normalization=True, copy_action_normalization=True)
    _apply_variant(policy, steps, image_size, text_len)
    return policy.to(torch.float32).eval()


def _apply_variant(policy, steps: int | None, image_size: int | None, text_len: int | None = None) -> None:
    """Flow steps baked into the unrolled graph; optionally a different square image size (needs a model
    fine-tuned at that size to keep quality — zero-shot it is an experiment only)."""
    import dataclasses

    backbone = policy.model.backbone.model
    if steps is not None:
        backbone.flow_matching_num_steps = int(steps)
        try:
            policy.config = dataclasses.replace(policy.config, flow_matching_num_steps=int(steps))
        except (TypeError, ValueError):
            pass
    changes = {}
    if image_size is not None:
        changes["image_processor_size"] = {"height": int(image_size), "width": int(image_size)}
    if text_len is not None:  # static text length (prompt + state + setup/control tokens); default 256 = ~158 pads
        changes["tokenizer_max_length"] = int(text_len)
    if changes:
        from physicalai.policies.molmoact2.processors.factory import make_molmoact2_preprocessors

        policy.config = dataclasses.replace(policy.config, **changes)
        policy._preprocessor, policy._postprocessor = make_molmoact2_preprocessors(policy.config)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained", default="allenai/MolmoAct2-SO100_101")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rtc", action="store_true")
    ap.add_argument("--int8", type=Path)
    ap.add_argument("--cams", default="wrist,top")
    ap.add_argument("--hw", default="480,640")
    ap.add_argument("--ckpt", help="Studio Lightning checkpoint (LoRA fine-tune); overrides --pretrained")
    ap.add_argument("--steps", type=int, help="flow-matching steps baked into the graph (default 10)")
    ap.add_argument("--image-size", type=int, help="square image size (default 378; zero-shot changes = experiment)")
    ap.add_argument("--text-len", type=int, help="static text token length (default 256; ~98 used -> 128 is plenty)")
    args = ap.parse_args()
    h, w = (int(v) for v in args.hw.split(","))

    t0 = time.perf_counter()
    policy = build(args.pretrained, args.rtc, args.cams.split(","), h, w, args.ckpt, args.steps, args.image_size,
                   args.text_len)
    cfg = policy.config
    print("config:", {k: getattr(cfg, k, None) for k in ("chunk_size", "n_action_steps", "flow_matching_num_steps",
                                                          "adapt_to_so101", "norm_tag")})
    t1 = time.perf_counter()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "tmp*.onnx.data")))
    sample = None
    if args.rtc:  # real RTC: patched flow loop + 4 extra graph inputs (Studio's MolmoAct2 has no RTC of its own)
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from rtc_patch import patch_policy

        rtc_sample = patch_policy(policy)
        sample = policy._get_default_export_input_sample() | rtc_sample
        print("RTC export: graph inputs", list(sample))
    try:
        policy.export(str(out), backend="openvino", input_sample=sample)
    finally:
        for p in set(glob.glob(os.path.join(tempfile.gettempdir(), "tmp*.onnx.data"))) - before:
            os.remove(p)
    if args.rtc:
        from rtc_patch import fix_rtc_input_names

        print("RTC input names fixed:", fix_rtc_input_names(next(p for p in out.glob("*.xml") if "tokenizer" not in p.name)))
    t2 = time.perf_counter()
    res = {"out": str(out), "rtc": args.rtc, "steps": args.steps, "image_size": args.image_size, "text_len": args.text_len,
           "ckpt": args.ckpt,
           "load_s": round(t1 - t0, 1), "export_s": round(t2 - t1, 1)}
    del policy
    if args.int8:
        from physicalai.export import compress_weights_openvino_int8_sym

        dst = args.int8.expanduser()
        shutil.copytree(out, dst, dirs_exist_ok=True)
        manifest = json.loads((dst / "manifest.json").read_text())
        compress_weights_openvino_int8_sym(str(dst / manifest["model"]["artifacts"]["openvino"]))
        res |= {"int8": str(dst), "int8_s": round(time.perf_counter() - t2, 1)}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
