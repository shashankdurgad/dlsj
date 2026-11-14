"""Hardware-free tests for deploy/pi05/kernel helpers (counter parsing, sub-graph attribution, sweep summary, patch)."""

import importlib.util
import sys
from pathlib import Path

import pytest

KDIR = Path(__file__).resolve().parents[1] / "deploy" / "pi05" / "kernel"
sys.path.insert(0, str(KDIR))
import kernel_lib as K  # noqa: E402

CSV = """layerName;execStatus;layerType;execType;realTime (ms);cpuTime (ms)
Convert_1;Status.OPTIMIZED_OUT;Convert;undef;0.000;0.000
fc_1;Status.EXECUTED;FullyConnected;jit:gemm:any__i8;1.500;0.010
sdpa_1;Status.EXECUTED;ScaledDotProductAttention;ocl::sdpa::opt__f16;0.700;0.000
rms_1;Status.EXECUTED;RMS;rms_gpu_bfyx_opt__f16;0.010;0.000
tokenized_prompt_mask;Status.EXECUTED;Parameter;wait_for_events__u8;0.000;0.000
Total;;;;2.210;0.010
"""


def test_parse_counters_csv():
    rows, total = K.parse_counters_csv(CSV)
    assert total == pytest.approx(2.21)
    assert [r["name"] for r in rows] == ["Convert_1", "fc_1", "sdpa_1", "rms_1", "tokenized_prompt_mask"]
    assert rows[1]["status"] == "EXECUTED" and rows[1]["real_ms"] == 1.5
    kern = [r for r in rows if K.is_kernel(r)]
    assert [r["name"] for r in kern] == ["fc_1", "sdpa_1", "rms_1"]


@pytest.mark.parametrize("et,backend", [
    ("jit:gemm:any__i8", "onednn"), ("jit:ir__f32", "onednn"), ("ocl:ref:any__f16", "onednn"),
    ("ocl::sdpa::opt__f16", "openvino-ocl"), ("fully_connected_gpu_bf_tiled__f16", "openvino-ocl"),
    ("rms_gpu_bfyx_opt", "openvino-ocl"), ("undef", "none"), ("wait_for_events__u8", "none"),
    ("reorder_data_fast_b1__f32", "openvino-ocl"),
])
def test_backend_of(et, backend):
    assert K.backend_of(et) == backend


def test_uses_xmx():
    assert K.uses_xmx("jit:gemm:any__i8")
    assert not K.uses_xmx("ocl:ref:any__f16")
    assert K.uses_xmx("sdpa_micro__f16")
    assert not K.uses_xmx("rms_gpu_bfyx_opt")


@pytest.mark.parametrize("shapes,part", [
    ([[1, 50, 1024]], "expert"),
    ([[1, 8, 50, 1018]], "expert"),
    ([[1, 968, 2048], [1, 50, 1024]], "expert"),  # expert checked first across all shapes
    ([[3, 256, 1152]], "vision"),
    ([[768, 1152]], "vision"),
    ([[3, 1152, 16, 16]], "vision"),
    ([[3, 1, 3, 224, 224]], "vision"),
    ([[1, 968, 2048]], "vlm"),
    ([[1, 8, 832, 832]], "vlm"),  # 64-token prompt: 768 + 64
    ([[1, 200, 2048]], "other"),
    ([[1, 32]], "other"),
    ([[1, 50, 8, 256]], "expert"),  # SDPA output layout [B, T, H, D]
    ([[1, 968, 8, 256]], "vlm"),
    ([[3, 256, 16, 72]], "vision"),
    ([[], [1, -1, 3]], "other"),
])
def test_subgraph_of(shapes, part):
    assert K.subgraph_of(shapes) == part


def test_prefix_rows_and_expert_kv_concat():
    shapes = [[[1, 968, 2048]], [[1, 968, 16384]], [[1, 200, 2048]], [[1, 50, 1024]], [[1, 968, 2048]]]
    pr = K.infer_prefix_rows(shapes)
    assert pr == 968
    assert K.subgraph_of([[1, 1, 1018, 256]]) == "vlm"
    assert K.subgraph_of([[1, 1, 1018, 256]], prefix_rows=pr) == "expert"
    assert K.infer_prefix_rows([[[1, 50, 1024]]]) is None


def test_op_family():
    assert K.op_family("FullyConnectedCompressed", "jit:gemm:any__i8") == "MatMul/FC"
    assert K.op_family("ScaledDotProductAttention", "ocl::sdpa::opt") == "SDPA"
    assert K.op_family("GatedMLP", "onednn::gated_mlp") == "GatedMLP"
    assert K.op_family("DynamicQuantize", "dynamic_quantize_gpu_opt") == "DynamicQuantize"
    assert K.op_family("RMS", "rms_gpu_bfyx_opt") == "RMSNorm"
    assert K.op_family("Reorder", "reorder_data_fast_b1") == "Reorder/Convert"


def test_aggregate_and_split():
    rows = [{"k": "a", "real_ms": 1.0}, {"k": "b", "real_ms": 3.0}, {"k": "a", "real_ms": 2.0}]
    agg = K.aggregate(rows, lambda r: r["k"])
    assert [a["key"] for a in agg] == ["a", "b"] or [a["key"] for a in agg] == ["b", "a"]
    a = next(x for x in agg if x["key"] == "a")
    assert a["count"] == 2 and a["ms"] == 3.0 and a["share"] == pytest.approx(0.5) and a["mean_us"] == 1500
    groups = K.split_steps(list(range(23)), 10)
    assert len(groups) == 10 and sum(map(len, groups)) == 23 and groups[0][0] == 0 and groups[-1][-1] == 22
    with pytest.raises(ValueError):
        K.split_steps([1], 0)


def test_dispatch_estimate():
    d = K.dispatch_estimate(200.0, 150.0, 5000, kernel_times_us=[5, 10, 50, 100])
    assert d["gap_ms"] == 50.0 and d["gap_share"] == 0.25 and d["gap_us_per_kernel"] == pytest.approx(10.0)
    assert d["n_small"] == 2 and d["small_ms"] == pytest.approx(0.015)
    assert K.dispatch_estimate(100.0, 120.0, 10)["gap_ms"] == 0.0


def test_stats_matches_numpy_percentiles():
    np = pytest.importorskip("numpy")
    xs = [5.0, 1.0, 3.0, 2.0, 4.0, 10.0, 7.5]
    s = K.stats(xs)
    assert s["median_ms"] == pytest.approx(np.median(xs))
    assert s["p90_ms"] == pytest.approx(np.percentile(xs, 90))
    assert s["n"] == 7
    with pytest.raises(ValueError):
        K.stats([])


def test_pooled_compare_paired_delta():
    def row(v, r, m, p=None):
        return {"variant": v, "round": r, "runtime_call": {"median_ms": m, "p90_ms": p or m + 2}}

    res = [row("default", 0, 180), row("x", 0, 170), row("default", 1, 190), row("x", 1, 181),
           {"variant": "broken", "round": 0, "error": "boom"}]
    out = K.pooled_compare(res)
    assert out[0]["variant"] == "default" and out[0]["delta_ms"] == 0
    x = next(o for o in out if o["variant"] == "x")
    assert x["median_ms"] == pytest.approx(175.5) and x["delta_ms"] == pytest.approx(-9.5)
    assert x["delta_pct"] == pytest.approx(-9.5 / 185 * 100)
    assert all(o["variant"] != "broken" for o in out)


def test_md_table():
    t = K.md_table([{"a": 1.234, "b": "x"}, {"a": None, "b": "y"}], [("a", "A"), ("b", "B")], {"a": ".1f"})
    assert t.splitlines()[0] == "| A | B |" and "| 1.2 | x |" in t and "|  | y |" in t


def _load_patch():
    spec = importlib.util.spec_from_file_location("patch_gm", KDIR / "patch_gated_mlp_gelu.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SRC = '''#include "openvino/op/swish.hpp"
    auto swish = wrap_type<ov::op::v4::Swish>({mm_gate});
        auto sw = ov::as_type_ptr<ov::op::v4::Swish>(pm.at(swish).get_node_shared_ptr());
        if (!down || !up || !gate || !sw || transformation_callback(down))
        if (sw->get_input_size() != 1 && sw->get_input_size() != 2)
        if (sw->get_input_size() == 2) {
          ov::op::internal::GLU::GluType::Swish,
          ov::op::internal::GLU::GluType::Swish,
          ov::op::internal::GLU::GluType::Swish,
'''


def test_patch_gated_mlp_gelu_idempotent():
    p = _load_patch()
    out = p.patch(SRC)
    assert "ov::op::v7::Gelu" in out and "OV_GPU_GATED_MLP_GELU" in out and "Gelu_Tanh" in out
    assert out.count("glu_type,\n") == 3 and "GluType::Swish,\n" not in out
    assert p.patch(out) == out
    with pytest.raises(SystemExit):
        p.patch("unrelated source")
