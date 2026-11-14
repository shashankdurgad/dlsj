"""Weight-compression variants of a MolmoAct2 FP16 OpenVINO export (no re-trace; seconds to a few minutes each).

Variants (NNCF data-free weight compression on the FP16 IR, written as a new export dir that copies the manifest and
tokenizer):
  int8_sym        all MatMul weights INT8 symmetric per-channel (what Studio's compress_weights_openvino_int8_sym does)
  int8_asym       INT8 asymmetric per-channel
  int8_keep_ae    INT8 sym, action expert left FP16 (ignored scope .*action_expert.*)
  int8_keep_ae_vit INT8 sym, action expert + vision encoder left FP16
  int4_lm         INT4 (sym, group 128) on the language model only, INT8 elsewhere — quality-risky experiment

    python quantize.py SRC_FP16_DIR DST_DIR --variant int8_keep_ae
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

VARIANTS = ("int8_sym", "int8_asym", "int8_keep_ae", "int8_keep_ae_vit", "int4_lm")


def compress(src: Path, dst: Path, variant: str) -> dict:
    import nncf
    import openvino as ov

    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant}; choose from {VARIANTS}")
    manifest = json.loads((src / "manifest.json").read_text())
    xml_name = manifest["model"]["artifacts"]["openvino"]
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix in (".json",) or f.name.startswith("tokenizer"):
            shutil.copy2(f, dst / f.name)
    t = time.perf_counter()
    model = ov.Core().read_model(str(src / xml_name))
    M = nncf.CompressWeightsMode
    if variant == "int8_sym":
        model = nncf.compress_weights(model, mode=M.INT8_SYM)
    elif variant == "int8_asym":
        model = nncf.compress_weights(model, mode=M.INT8_ASYM)
    elif variant == "int8_keep_ae":
        model = nncf.compress_weights(model, mode=M.INT8_SYM,
                                      ignored_scope=nncf.IgnoredScope(patterns=[".*action_expert.*"], validate=False))
    elif variant == "int8_keep_ae_vit":
        model = nncf.compress_weights(model, mode=M.INT8_SYM, ignored_scope=nncf.IgnoredScope(
            patterns=[".*action_expert.*", ".*vision_backbone.*"], validate=False))
    elif variant == "int4_lm":
        # INT8 everywhere except the LM, then INT4 on the LM (two passes: compress_weights skips already-compressed)
        model = nncf.compress_weights(model, mode=M.INT4_SYM, group_size=128, ratio=1.0, all_layers=False,
                                      ignored_scope=nncf.IgnoredScope(
                                          patterns=[".*action_expert.*", ".*vision_backbone.*"], validate=False))
        model = nncf.compress_weights(model, mode=M.INT8_SYM)
    ov.save_model(model, str(dst / xml_name), compress_to_fp16=True)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rtc_patch import fix_rtc_input_names

    fix_rtc_input_names(dst / xml_name)  # no-op for non-RTC exports
    return {"variant": variant, "src": str(src), "dst": str(dst), "seconds": round(time.perf_counter() - t, 1),
            "bin_gb": round(sum(p.stat().st_size for p in dst.glob("*.bin")) / 1e9, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--variant", default="int8_sym", choices=VARIANTS)
    a = ap.parse_args()
    print(json.dumps(compress(a.src.expanduser(), a.dst.expanduser(), a.variant), indent=1))


if __name__ == "__main__":
    main()
