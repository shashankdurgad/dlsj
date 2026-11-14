"""Export π0.5 to OpenVINO with Physical AI Studio (`physicalai.policies.Pi05`).

This is the judged-stack export path. It loads either the base model from the Hugging Face hub
(`lerobot/pi05_base`, latency work) or a fine-tuned Studio Lightning checkpoint (`.ckpt`, the
real deployment), then calls `policy.export(..., backend="openvino")`. Studio converts via ONNX,
saves the IR in FP16, converts the PaliGemma tokenizer to `tokenizer.xml` and writes a
`manifest.json` that `physicalai.inference.InferenceModel` reads (normalise -> pi05 image prep ->
tokenizer -> noise -> model -> denormalise). `--precision int8` adds Studio's
`compress_weights_openvino_int8_sym` post-export hook (NNCF weight-only INT8_SYM).

Run on the XPS in the Studio library env. Use the venv's python directly: `uv run` would re-sync the env and
drop the `xpu` extra. Measured cost of one export: 8m46s wall, 41.7 GB peak RSS, plus a 13 GB ONNX temp file in
/tmp (tmpfs on the XPS, so it is RAM too). Studio deletes the temp .onnx but leaves `/tmp/tmp*.onnx.data`
behind; this script removes it. Set `TMPDIR=~/inference/tmp` to keep that file on disk instead of in RAM.
The `--ckpt` (fine-tuned) path follows Studio's documented `Pi05.load_from_checkpoint` API but has not been
run yet: no fine-tuned checkpoint existed when this was written.

    cd ~/inference/physical-ai-studio/library && export TMPDIR=~/inference/tmp
    .venv/bin/python ~/inference/deploy/pi05/studio_export.py \
        --out ~/inference/exports/pi05_studio_fp16 --precision fp16 [--steps 5] [--tokens 64]
    .venv/bin/python ~/inference/deploy/pi05/studio_export.py \
        --out ~/inference/exports/pi05_studio_int8 --precision int8

    # INT8 from an existing FP16 export (same hook, no second 42 GB trace):
    .venv/bin/python ~/inference/deploy/pi05/studio_export.py \
        --compress_from ~/inference/exports/pi05_studio_fp16 --out ~/inference/exports/pi05_studio_int8

    # fine-tuned checkpoint from Studio training (training/pi05/full.json):
    .venv/bin/python ~/inference/deploy/pi05/studio_export.py \
        --ckpt runs/pi05/.../checkpoints/last.ckpt --out ~/inference/exports/pi05_ft_int8 --precision int8
"""

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

DEFAULT_MODEL = "lerobot/pi05_base"


def load_policy(pretrained: str, ckpt: str | None, steps: int | None, tokens: int | None = None) -> torch.nn.Module:
    """Build a Pi05 policy on CPU in fp32; the IR is traced in fp32 and saved as FP16."""
    from physicalai.policies import Pi05

    if ckpt:
        policy = Pi05.load_from_checkpoint(ckpt, map_location="cpu")
        # Pi05Config is a frozen dataclass: swap in a modified copy instead of assigning fields.
        changes = {}
        if steps is not None:  # Pi05Model reads this at sample time; the export unrolls that many steps
            changes["num_inference_steps"] = steps
            policy.inner_model._num_inference_steps = steps
        if tokens is not None:
            # The prompt length also sets the tokenizer preprocessor, which builds the trace sample (so the IR's
            # static [1, tokens] input) and the exported tokenizer spec, so rebuild the preprocessors after it.
            changes["tokenizer_max_length"] = tokens
            policy.inner_model._tokenizer_max_length = tokens
        if changes:
            policy.config = dataclasses.replace(policy.config, **changes)
        if tokens is not None:
            policy._update_preprocessor_stats(policy._dataset_stats)
    else:
        kwargs = {"num_inference_steps": steps} if steps is not None else {}
        policy = Pi05(pretrained_name_or_path=pretrained, dtype="float32", gradient_checkpointing=False, **kwargs)
    return policy.to(torch.float32).eval()


def with_token_length(pretrained: str, tokens: int, work_dir: Path) -> str:
    """Local copy of a hub model (symlinked files) whose config.json sets `tokenizer_max_length`.

    Pi05's pretrained path takes the prompt length from the model's config.json (200 for pi05_base);
    the length is baked into the static IR, and the prefix is 3x256 image tokens + this many text tokens.
    """
    from huggingface_hub import snapshot_download

    src = Path(pretrained) if Path(pretrained).is_dir() else Path(snapshot_download(pretrained))
    dst = work_dir / f"{src.name}_tok{tokens}"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name != "config.json" and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f.resolve())
    cfg = json.loads((src / "config.json").read_text())
    cfg["tokenizer_max_length"] = tokens
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    return str(dst)


def export(policy: torch.nn.Module, out_dir: Path, precision: str) -> list[Path]:
    """Run Studio's OpenVINO export (+ optional INT8 hook) and return the files produced."""
    hooks = []
    if precision == "int8":
        from physicalai.export import compress_weights_openvino_int8_sym

        hooks.append(compress_weights_openvino_int8_sym)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.export(str(out_dir), backend="openvino", post_export_hooks=hooks)
    return sorted(p for p in out_dir.rglob("*") if p.is_file())


def compress_existing(src: Path, out_dir: Path) -> list[Path]:
    """Copy an FP16 Studio export and apply Studio's INT8 hook to it (what post_export_hooks does)."""
    from physicalai.export import compress_weights_openvino_int8_sym

    shutil.copytree(src, out_dir, dirs_exist_ok=True)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    compress_weights_openvino_int8_sym(str(out_dir / manifest["model"]["artifacts"]["openvino"]))
    return sorted(p for p in out_dir.rglob("*") if p.is_file())


def _remove_stale_onnx_data(before: set[str]) -> None:
    """Studio's via-ONNX export leaves its external-data file in the temp dir; delete ours."""
    for path in set(glob.glob(os.path.join(tempfile.gettempdir(), "tmp*.onnx.data"))) - before:
        os.remove(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained", default=DEFAULT_MODEL, help="HF repo id or local LeRobot-format dir")
    ap.add_argument("--ckpt", help="Studio Lightning checkpoint (.ckpt); overrides --pretrained")
    ap.add_argument("--out", required=True, help="export directory (keep under ~/inference/exports)")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "int8"])
    ap.add_argument("--steps", type=int, help="denoising steps baked into the graph (default: model's, 10)")
    ap.add_argument("--tokens", type=int, help="prompt length baked into the IR (pi05_base default 200)")
    ap.add_argument("--compress_from", type=Path, help="existing FP16 export dir to INT8-compress into --out")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    t0 = time.perf_counter()
    if args.compress_from:
        t1 = t0
        files = compress_existing(args.compress_from.expanduser(), out)
    else:
        pretrained = args.pretrained
        if args.tokens and not args.ckpt:
            pretrained = with_token_length(pretrained, args.tokens, Path(tempfile.gettempdir()) / "pi05_models")
        policy = load_policy(pretrained, args.ckpt, args.steps, args.tokens if args.ckpt else None)
        t1 = time.perf_counter()
        before = set(glob.glob(os.path.join(tempfile.gettempdir(), "tmp*.onnx.data")))
        try:
            files = export(policy, out, args.precision)
        finally:
            _remove_stale_onnx_data(before)
    t2 = time.perf_counter()
    print(json.dumps({"load_s": round(t1 - t0, 1), "export_s": round(t2 - t1, 1),
                      "files": {str(p): p.stat().st_size for p in files}}, indent=2))


if __name__ == "__main__":
    main()
