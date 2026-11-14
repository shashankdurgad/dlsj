"""Export a fine-tuned Physical AI Studio π0.5 checkpoint (`.ckpt`) to an OpenVINO policy package.

Studio's `policy.export(backend="openvino")` writes `pi05.xml/.bin` (SigLIP on the 3 cameras, the Gemma
prefix and the unrolled denoising loop), `tokenizer.xml/.bin` (PaliGemma tokenizer as an OpenVINO model) and
`manifest.json` (normalisation with the checkpoint's dataset stats, image prep, noise, tokenizer). It is run by
`physicalai.inference.InferenceModel`; see `dlsj.deploy.pi05_runtime`. `--int8` then makes a weight-only
INT8_SYM copy with NNCF (Studio's `compress_weights_openvino_int8_sym` hook), about 2x smaller and faster.

Needs the Studio library env at the checkpoint's commit (8f1324161ece) with the `pi05` and `nncf` extras, and
the gated `google/paligemma-3b-pt-224` tokenizer in the HF cache. Peak memory is ~45-55 GB plus a ~13 GB ONNX
temp file, so run it on a big machine (we use the GB10) with TMPDIR on disk, ideally in a memory-capped scope
(`systemd-run --user --scope -p MemoryMax=56G ...`) so a shared box can't OOM other jobs, then copy the package
to the laptop:

    TMPDIR=~/desmond/tmp HF_HUB_OFFLINE=1 python -m dlsj.deploy.pi05_export \
        --ckpt best.ckpt --out exports/pi05_ft_fp16 --int8 exports/pi05_ft_int8 [--steps 5]
"""

import argparse
import glob
import json
import os
import shutil
import tempfile
import time
from pathlib import Path


def _mmap_checkpoint_loading() -> None:
    """Make Lightning memory-map the .ckpt: its 12 GB stays file-backed instead of a second copy in RAM."""
    import lightning.pytorch.core.saving as saving
    import torch

    def load(path, map_location=None, weights_only=None):
        return torch.load(path, map_location=map_location, weights_only=False, mmap=True)

    saving.pl_load = load


def load(ckpt: str, steps: int | None):
    import gc

    import torch
    from physicalai.policies import Pi05

    _mmap_checkpoint_loading()
    policy = Pi05.load_from_checkpoint(ckpt, map_location="cpu", pretrained_name_or_path=None, compile_model=False)
    gc.collect()
    if steps is not None:  # the export unrolls this many denoising steps into the graph
        policy.config.num_inference_steps = steps
        policy.inner_model._num_inference_steps = steps
    return policy.to(torch.float32).eval()


def compress_int8(src: Path, dst: Path) -> None:
    from physicalai.export import compress_weights_openvino_int8_sym

    shutil.copytree(src, dst, dirs_exist_ok=True)
    manifest = json.loads((dst / "manifest.json").read_text())
    compress_weights_openvino_int8_sym(str(dst / manifest["model"]["artifacts"]["openvino"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, type=Path, help="FP16 package directory")
    ap.add_argument("--int8", type=Path, help="also write an INT8 weight-compressed copy here")
    ap.add_argument("--steps", type=int, help="denoising steps baked into the graph (checkpoint default: 10)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    policy = load(args.ckpt, args.steps)
    t1 = time.perf_counter()
    leftovers = set(glob.glob(os.path.join(tempfile.gettempdir(), "tmp*.onnx.data")))
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        policy.export(str(args.out), backend="openvino")
    finally:  # Studio leaves its ~13 GB ONNX external-data file behind
        for p in set(glob.glob(os.path.join(tempfile.gettempdir(), "tmp*.onnx.data"))) - leftovers:
            os.remove(p)
    t2 = time.perf_counter()
    if args.int8:
        compress_int8(args.out, args.int8)
    print(json.dumps({"load_s": round(t1 - t0, 1), "export_s": round(t2 - t1, 1),
                      "int8_s": round(time.perf_counter() - t2, 1) if args.int8 else None,
                      "steps": policy.config.num_inference_steps,
                      "tokenizer_max_length": policy.config.tokenizer_max_length}))


if __name__ == "__main__":
    main()
