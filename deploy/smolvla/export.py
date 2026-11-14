"""Export a Physical AI Studio SmolVLA Lightning checkpoint to OpenVINO (FP16, optional INT8, optional RTC).

Loads `physicalai.policies.SmolVLA.load_from_checkpoint(<.ckpt>)` (e.g. w3joe/smolvla-junction-so101-3stage-20261114,
2 cameras wrist + top), traces it in fp32 and writes Studio's OpenVINO package (`manifest.json` + IR + tokenizer) that
`physicalai.inference.InferenceModel` runs. `--rtc` adds the Real-Time Chunking inputs (prev_chunk_left_over,
inference_delay, max_guidance_weight, execution_horizon) like `deploy/pi05/studio_export.py --rtc`.

Run on the XPS in the Studio library env (commit 8f13241):

    cd ~/inference/physical-ai-studio/library && export TMPDIR=~/inference/tmp/onnx
    .venv/bin/python ~/inference/deploy/smolvla/export.py \
        --ckpt ~/inference/models/smolvla-junction-3stage/checkpoints/snapshots/step020000.ckpt \
        --out ~/inference/exports/smolvla_3stage_rtc_fp16 --rtc [--int8 ~/inference/exports/smolvla_3stage_rtc_int8]
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch


def load(ckpt: str, rtc: bool, steps: int | None):
    from physicalai.policies import SmolVLA

    policy = SmolVLA.load_from_checkpoint(ckpt, map_location="cpu")
    changes = {}
    if steps is not None:
        changes["num_steps"] = steps
    if rtc and policy.config.n_action_steps != policy.config.chunk_size:
        changes["n_action_steps"] = policy.config.chunk_size  # RTC executes a horizon out of the whole chunk
    if changes:
        policy.config = dataclasses.replace(policy.config, **changes)
        if "num_steps" in changes and hasattr(policy, "model") and hasattr(policy.model, "config"):
            try:
                policy.model.config = dataclasses.replace(policy.model.config, num_steps=steps)
            except Exception:  # noqa: BLE001  model config may be a different type; the policy config drives export
                pass
    if rtc:
        policy.rtc_enabled = True
    return policy.to(torch.float32).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rtc", action="store_true")
    ap.add_argument("--steps", type=int, help="flow-matching steps baked into the graph (default: checkpoint's, 10)")
    ap.add_argument("--int8", type=Path, help="also write an INT8 (weight-only SYM) copy here")
    args = ap.parse_args()

    t0 = time.perf_counter()
    policy = load(args.ckpt, args.rtc, args.steps)
    t1 = time.perf_counter()
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "tmp*.onnx.data")))
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    try:
        policy.export(str(out), backend="openvino")
    finally:
        for p in set(glob.glob(os.path.join(tempfile.gettempdir(), "tmp*.onnx.data"))) - before:
            os.remove(p)
    t2 = time.perf_counter()
    res = {"ckpt": args.ckpt, "out": str(out), "rtc": args.rtc, "load_s": round(t1 - t0, 1), "export_s": round(t2 - t1, 1),
           "config": {k: getattr(policy.config, k, None) for k in ("chunk_size", "n_action_steps", "num_steps",
                                                                     "resize_imgs_with_padding", "tokenizer_max_length")}}
    if args.int8:
        from physicalai.export import compress_weights_openvino_int8_sym

        dst = args.int8.expanduser()
        shutil.copytree(out, dst, dirs_exist_ok=True)
        manifest = json.loads((dst / "manifest.json").read_text())
        compress_weights_openvino_int8_sym(str(dst / manifest["model"]["artifacts"]["openvino"]))
        res["int8"] = str(dst)
        res["int8_s"] = round(time.perf_counter() - t2, 1)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
