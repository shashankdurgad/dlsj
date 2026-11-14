"""Extend OpenVINO 2026.4.1's GPU `FuseGatedMLP` pass to Gemma's GeGLU MLP (gate -> Gelu(tanh) * up -> down).

The stock pass (PR #34427) only matches `MatMul -> Swish` (SwiGLU, Llama/Qwen). π0.5's PaliGemma and its action
expert are Gemma models whose MLP activation is `Gelu(approximation_mode=TANH)`, so the stock pass finds nothing in
a Studio π0.5 IR even with debug caps on. The oneDNN `gated_mlp` primitive the pass emits already supports
`dnnl_eltwise_gelu_tanh` / `gelu_erf` (`gated_mlp_onednn.cpp::to_dnnl_alg`), so only the pattern needs widening.

The Gelu branch is opt-in at runtime (`OV_GPU_GATED_MLP_GELU=1`), so the same binary can show the stock pass
behaviour too. Idempotent; run once on a fresh checkout:

    python patch_gated_mlp_gelu.py ~/inference/ovbuild/src
"""

import sys
from pathlib import Path

MARK = "// [dlsj] GeGLU extension"


def patch(text: str) -> str:
    if MARK in text:
        return text
    reps = [
        ('#include "openvino/op/swish.hpp"\n',
         '#include "openvino/op/swish.hpp"\n#include "openvino/op/gelu.hpp"\n#include <cstdlib>\n#include <string>\n'),
        ("    auto swish = wrap_type<ov::op::v4::Swish>({mm_gate});\n",
         f"    {MARK}: OV_GPU_GATED_MLP_GELU=1 also matches Gelu (Gemma GeGLU)\n"
         "    const char* gelu_env = std::getenv(\"OV_GPU_GATED_MLP_GELU\");\n"
         "    const bool match_gelu = gelu_env && std::string(gelu_env) == \"1\";\n"
         "    auto swish_only = wrap_type<ov::op::v4::Swish>({mm_gate});\n"
         "    auto gelu_act = wrap_type<ov::op::v7::Gelu>({mm_gate});\n"
         "    std::shared_ptr<ov::Node> swish = swish_only;\n"
         "    if (match_gelu)\n"
         "        swish = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{swish_only, gelu_act});\n"),
        ("        auto sw = ov::as_type_ptr<ov::op::v4::Swish>(pm.at(swish).get_node_shared_ptr());\n",
         "        auto act_node = pm.count(gelu_act) ? pm.at(gelu_act).get_node_shared_ptr()\n"
         "                                           : pm.at(swish_only).get_node_shared_ptr();\n"
         "        auto sw = ov::as_type_ptr<ov::op::v4::Swish>(act_node);\n"
         "        auto ge = ov::as_type_ptr<ov::op::v7::Gelu>(act_node);\n"
         "        auto glu_type = ov::op::internal::GLU::GluType::Swish;\n"
         "        if (ge)\n"
         "            glu_type = ge->get_approximation_mode() == ov::op::GeluApproximationMode::TANH\n"
         "                           ? ov::op::internal::GLU::GluType::Gelu_Tanh\n"
         "                           : ov::op::internal::GLU::GluType::Gelu;\n"),
        ("        if (!down || !up || !gate || !sw || transformation_callback(down))\n",
         "        if (!down || !up || !gate || (!sw && !ge) || transformation_callback(down))\n"),
        ("        if (sw->get_input_size() != 1 && sw->get_input_size() != 2)\n",
         "        if (sw && sw->get_input_size() != 1 && sw->get_input_size() != 2)\n"),
        ("        if (sw->get_input_size() == 2) {\n", "        if (sw && sw->get_input_size() == 2) {\n"),
    ]
    for old, new in reps:
        if text.count(old) != 1:
            raise SystemExit(f"patch anchor not found exactly once: {old!r}")
        text = text.replace(old, new)
    n = text.count("ov::op::internal::GLU::GluType::Swish,\n")
    if n != 3:
        raise SystemExit(f"expected 3 GatedMLP constructions, found {n}")
    return text.replace("ov::op::internal::GLU::GluType::Swish,\n", "glu_type,\n")


if __name__ == "__main__":
    f = Path(sys.argv[1]).expanduser() / "src/plugins/intel_gpu/src/plugin/transformations/fuse_gated_mlp.cpp"
    f.write_text(patch(f.read_text()))
    print("patched", f)
