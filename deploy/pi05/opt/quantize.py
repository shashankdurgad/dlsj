"""NNCF post-training quantisation variants of a Studio π0.5 FP16 OpenVINO export.

Writes a complete export directory (copies `manifest.json`, `tokenizer.*`, etc. from the source and
replaces the model IR), so `InferenceModel`, `eval_actions.py` and `studio_bench.py` work on it unchanged.

Modes:
- `w8a8`: `nncf.quantize` (static INT8 weights + activations) calibrated on real IR inputs captured by
  `eval_actions.py run --capture_ir` from training-split frames. `model_type=TRANSFORMER`, preset MIXED.
- `int4`: `nncf.compress_weights` INT4 (SYM/ASYM, grouped) on everything not matched by `--ignore`,
  then INT8_SYM on the ignored part (e.g. the action expert) with a second pass.
- `int8`: `nncf.compress_weights` INT8_SYM (Studio's hook), optionally with `--ignore`.
- `inspect`: print op/weight statistics of an IR grouped by subgraph (vision / VLM / expert / other).

    PY=~/inference/physical-ai-studio/library/.venv/bin/python
    nice -n 19 $PY quantize.py w8a8 ~/inference/exports/pi05_junction_fp16 --calib ~/inference/eval/calib_ir.npz \
        --out ~/inference/exports/pi05_junction_w8a8 [--ignore 'gemma_expert'] [--subset 256]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opt_lib as L  # noqa: E402


def model_xml(export_dir: Path) -> Path:
    manifest = json.loads((export_dir / "manifest.json").read_text())
    return export_dir / manifest["model"]["artifacts"]["openvino"]


def copy_export(src: Path, out: Path) -> Path:
    """Copy everything except the model IR (rewritten by the caller); return the IR path in `out`."""
    xml = model_xml(src)
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name not in (xml.name, xml.with_suffix(".bin").name):
            shutil.copy2(f, out / f.name)
    return out / xml.name


def matmul_groups(model) -> dict[str, list[str]]:
    """MatMul friendly names per subgraph (vision / vlm / expert / other), classified by activation shape."""
    groups: dict[str, list[str]] = defaultdict(list)
    for op in model.get_ordered_ops():
        if op.get_type_name() == "MatMul":
            shape = op.input_value(0).get_partial_shape()
            dims = [d.get_length() if d.is_static else -1 for d in shape] if shape.rank.is_static else []
            groups[L.classify_matmul(dims)].append(op.get_friendly_name())
    return dict(groups)


def _weight_kind(op) -> str:
    """Element type behind a MatMul's weight input: f16/f32 (plain), i8/u8/i4/u4 (compressed), act (activation)."""
    node = op.input_value(1).get_node()
    for _ in range(6):
        t = node.get_type_name()
        if t == "Constant":
            return node.get_output_element_type(0).get_type_name()
        if t == "FakeQuantize":
            return "fq_" + _weight_kind_from(node)
        if t in ("Convert", "Multiply", "Subtract", "Reshape", "Transpose") and node.get_input_size():
            node = node.input_value(0).get_node()
            continue
        return "act"
    return "act"


def _weight_kind_from(node) -> str:
    src = node.input_value(0).get_node()
    return src.get_output_element_type(0).get_type_name() if src.get_type_name() == "Constant" else "act"


def inspect_model(model) -> dict:
    """Per subgraph: MatMuls by weight kind, and MatMuls whose activation input passes a FakeQuantize."""
    stats: dict[str, Counter] = defaultdict(Counter)
    for op in model.get_ordered_ops():
        if op.get_type_name() != "MatMul":
            continue
        shape = op.input_value(0).get_partial_shape()
        dims = [d.get_length() if d.is_static else -1 for d in shape] if shape.rank.is_static else []
        g = L.classify_matmul(dims)
        stats[g]["matmul"] += 1
        stats[g][f"w_{_weight_kind(op)}"] += 1
        if op.input_value(0).get_node().get_type_name() == "FakeQuantize":
            stats[g]["act_fq"] += 1
    return {g: dict(c) for g, c in sorted(stats.items())}


def load_calibration(path: Path, input_names: list[str]) -> list[dict[str, np.ndarray]]:
    # Read each array once: NpzFile re-reads the whole array on every `data[k]`, and a slice keeps it alive.
    with np.load(path) as data:
        arrays = {k: data[k] for k in input_names}
    n = len(arrays[input_names[0]])
    return [{k: arrays[k][i] for k in input_names} for i in range(n)]  # stacked per call, exact IR shapes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["w8a8", "int4", "int8", "inspect"])
    ap.add_argument("src", type=Path, help="FP16 Studio export dir (or an export dir to inspect)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--calib", type=Path, help="IR-input npz from eval_actions.py run --capture_ir")
    ap.add_argument("--subset", type=int, default=256)
    ap.add_argument("--preset", default="MIXED", choices=["MIXED", "PERFORMANCE"])
    ap.add_argument("--no_smooth_quant", action="store_true", help="disable SmoothQuant (alpha -1)")
    ap.add_argument("--slow_bias", action="store_true", help="full BiasCorrection instead of FastBiasCorrection")
    ap.add_argument("--ignore", action="append", default=[], choices=["vision", "vlm", "expert", "other"],
                    help="subgraph to leave out of the main pass (repeatable); int4: that part gets INT8_SYM")
    ap.add_argument("--int4_mode", default="INT4_SYM", choices=["INT4_SYM", "INT4_ASYM"])
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--ratio", type=float, default=1.0)
    a = ap.parse_args()

    import nncf
    import openvino as ov

    core = ov.Core()
    src = a.src.expanduser()
    t0 = time.perf_counter()
    model = core.read_model(model_xml(src))
    if a.mode == "inspect":
        print(json.dumps(inspect_model(model), indent=1))
        return
    groups = matmul_groups(model)
    print("matmuls per subgraph:", {g: len(v) for g, v in groups.items()}, flush=True)
    ign_names = [n for g in a.ignore for n in groups.get(g, [])]
    ignored = nncf.IgnoredScope(names=ign_names, validate=False)
    if a.mode == "w8a8":
        names = [p.get_any_name() for p in model.inputs]
        calib = load_calibration(a.calib.expanduser(), names)
        print(f"calibration: {len(calib)} samples, inputs {names}", flush=True)
        adv = nncf.AdvancedQuantizationParameters(
            smooth_quant_alphas=nncf.AdvancedSmoothQuantParameters(matmul=-1) if a.no_smooth_quant else None)
        if adv.smooth_quant_alphas is None:
            adv = nncf.AdvancedQuantizationParameters()
        model = nncf.quantize(model, nncf.Dataset(calib), model_type=nncf.ModelType.TRANSFORMER,
                              preset=getattr(nncf.QuantizationPreset, a.preset), subset_size=min(a.subset, len(calib)),
                              fast_bias_correction=not a.slow_bias, ignored_scope=ignored, advanced_parameters=adv)
    elif a.mode == "int8":
        model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_SYM, ignored_scope=ignored)
    else:
        model = nncf.compress_weights(model, mode=getattr(nncf.CompressWeightsMode, a.int4_mode),
                                      group_size=a.group, ratio=a.ratio, ignored_scope=ignored,
                                      backup_mode=nncf.BackupMode.INT8_SYM)
        if a.ignore:  # second pass: INT8_SYM on what the first pass left in FP16 (the ignored part)
            rest = [n for g, v in groups.items() if g not in a.ignore for n in v]
            model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_SYM,
                                          ignored_scope=nncf.IgnoredScope(names=rest, validate=False))
    t1 = time.perf_counter()
    out_xml = copy_export(src, a.out.expanduser())
    ov.save_model(model, out_xml, compress_to_fp16=True)
    size = sum(f.stat().st_size for f in a.out.expanduser().iterdir() if f.is_file())
    print(json.dumps({"out": str(a.out), "mode": a.mode, "ignore": a.ignore, "quantize_s": round(t1 - t0, 1),
                      "export_bytes": size, "subgraphs": inspect_model(model)}, indent=1))


if __name__ == "__main__":
    main()
