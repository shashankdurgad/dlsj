# Further quantisation / compression of π0.5 on Core Ultra X9 388H (Arc B390, OpenVINO 2026.4, NNCF 3.4), as of Nov 2026

Scope: builds on `research_notes/XPS closed loop optimisation/pi05_openvino_panther_lake.md` and `research_notes/XPU kernels and Chinese VLAs/*.md`; facts already cited there are re-cited only where a recipe depends on them. "Our" numbers (243 → 200 ms, 69% `jit:gemm`, 891 DynamicQuantize kernels, +3.9% action error, 2.4× jerk, OOM, `InvalidGroupSizeError`) come from the brief and are not re-verified here.

Summary ranking by expected gain per hour (details and sources in the sections below; every latency expectation is an inference unless a source is given):

| # | Experiment | Cost | Expected latency | Expected accuracy / jitter | Verdict |
|---|---|---|---|---|---|
| 1 | A/B `DYNAMIC_QUANTIZATION_GROUP_SIZE` (0, default, 128, 64, 32) and measure jitter, not only latency | minutes, no re-export | 0 to +18% (0 = the known 18% slower case) | Isolates weight error from activation error; likely the main jitter source | Do first |
| 2 | Re-compress the existing FP16 IR with INT8 but keep SigLIP (+ projector) FP16 via `ignored_scope` | ~10 min per variant, data-free | about +4–8 ms | Removes both weight and dynamic-activation quantisation from the vision tower | Do second |
| 3 | Block-wise bisect with `ignored_scope` (vision / projector / Gemma first-last / Gemma attention vs MLP) | ~10 min per variant | +0–20 ms depending on what stays FP16 | Finds the sensitive block empirically | Do if 1–2 do not fix jitter |
| 4 | `INT8_ASYM` instead of `INT8_SYM` (GPU only) | ~10 min | ~0 | Small; LoopQ puts INT8 weight error at 0.15–0.2% | Cheap, low expectation |
| 5 | INT4 on Gemma backbone only (fix for the 4304 error) | 30–60 min | ~0, possibly negative | Risky on real robots (VLAQuantBench W4: 84 → 62%) | Memory only; not for speed |
| 6 | Static W8A8 (`nncf.quantize`) on a prefix-only IR, 32–64 samples, SmoothQuant off | hours, probably cloud box | ≤ ~6% beyond what dynamic quantisation already gives; can regress | Unknown; no data-aware INT8 alternative exists | Low priority |
| 7 | Vision encoder on NPU | days | No wall-time gain in a sequential pipeline | n/a | Not worth trying |

## 1. Making full W8A8 PTQ tractable, and what it would buy over weight-only INT8 + dynamic quantisation

### Takeaway
The failed run was slow because NNCF's OpenVINO backend hard-codes calibration inference to CPU at f32, and there is no supported switch to GPU. It can be made to fit (32–64 samples, SmoothQuant off, prefix-only scope, or a cloud CPU box), but the pay-off is small: the GPU plugin's dynamic quantisation already runs the prefix FullyConnected layers as W8A8 on XMX, so static W8A8 can at best recover the ~6% spent in DynamicQuantize kernels plus whatever the attention MatMuls give, and it risks breaking f16 SDPA fusion. Treat it as a low-priority cloud experiment.

### Cited Findings
- NNCF's OpenVINO calibration engine compiles with `ie.compile_model(model, device_name="CPU", config=config)` and by default sets `inference_precision: f32`. The constructor takes only `model` and `use_fp32_precision`; no parameter or environment variable selects another device — [nncf `src/nncf/openvino/engine.py`](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/src/nncf/openvino/engine.py)
- `nncf.quantize` defaults: `subset_size=300`; `preset=PERFORMANCE` (symmetric weights and activations; `MIXED` = symmetric weights, asymmetric activations); `fast_bias_correction=True`; `model_type=None` with `nncf.ModelType.Transformer` "the only supported special quantization scheme to preserve accuracy after quantization of Transformer models"; `target_device` one of `ANY, CPU, CPU_SPR, GPU, NPU` — [OpenVINO basic quantization flow](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.rst)
- `ignored_scope` accepts names, types, regex `patterns`, and `nncf.Subgraph(inputs=[...], outputs=[...])`, where "all nodes along all simple paths in the graph from input to output nodes will be excluded" — [OpenVINO basic quantization flow](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.rst)
- `AdvancedQuantizationParameters` defaults: `inplace_statistics=True`, `disable_bias_correction=False`, `quantize_outputs=False`, `overflow_fix=None`, `batchwise_statistics=None`, `smooth_quant_alphas=AdvancedSmoothQuantParameters(convolution=-1, matmul=0.95)`. A negative alpha disables SmoothQuant for that layer type (convolution is already -1 by default) — [nncf `advanced_parameters.py`](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/src/nncf/quantization/advanced_parameters.py)
- NNCF changelog: `quantize(...)` "can hang on models with transformer architecture when `fast_bias_correction` optional parameter is set to False"; in-place statistics collection was added to "reduce memory footprint during optimization" — [NNCF ReleaseNotes.md](https://github.com/openvinotoolkit/nncf/blob/develop/ReleaseNotes.md) (via search snippet)
- NNCF PTQ docs: batch sizes above 1 are not supported for transformers; `subset_size = subset_size_for_batch_size_1 // batch_size` — [NNCF PTQ Usage.md](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/docs/usage/post_training_compression/post_training_quantization/Usage.md)
- NNCF 3.0–3.4 release notes contain no `nncf.quantize` memory, transformer, SmoothQuant or GPU-calibration improvements; the only `quantize` items are a graph-structure migration (3.1) and a PyTorch FBC change (3.0). Newest version in the file is 3.4.0 — [NNCF ReleaseNotes.md](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/ReleaseNotes.md)
- Defensible small calibration sizes from the literature: QuantVLA uses "32 batches to estimate scales" (activation percentile 99.9) — [QuantVLA arXiv 2602.20309](https://arxiv.org/html/2602.20309v4). VLAQuantBench calibrates from two held-out episodes — [VLAQuantBench arXiv 2609.25376](https://arxiv.org/html/2609.25376). LoopQ's AWQ/GPTQ use 512 frames and "calibration brought no measurable benefit over plain RTN" on π0.5 — [LoopQ arXiv 2611.05093](https://arxiv.org/html/2611.05093). Hugging Face's OpenVINO VLM recipe uses 50 samples for static quantisation of a vision encoder — [HF blog openvino-vlm](https://huggingface.co/blog/openvino-vlm)
- Intel's roofline projection (3 cameras, 64 tokens, 10 steps; values read from Fig. 4 in the earlier notes): W8A16 VE/LE/AE = 18/68/40 ms (126), W8A8 = 13/35/40 ms (88). The action expert gains nothing from A8. The paper text says VE and LE are "compute-bound, GEMM-heavy" and that lower precision "shifts layers from bandwidth- to compute-bound"; it does not report any measured static W8A8 result, only "INT8 weight-only compression reduced the model size by half and improved inference speed by more than 30%, all without needing calibration data" — [Intel π0.5 paper](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf) (text extracted locally with pdftotext; figure values from prior notes)
- GPU-plugin dynamic quantisation "works when weights are already int4 or int8" and runs the multiplication in int8 instead of fp16; it is skipped for MatMul row counts of 64 or fewer — [OpenVINO blog: dynamic quantization on XMX](https://blog.openvino.ai/blog-posts/dynamic-quantization-support-from-gpu-with-xmx)
- VLAQuantBench π0.5 W8A8 (RTN, end-to-end, LIBERO): 98.5 / 99.0 / 97.0 / 93.5 against baseline 99.0 / 99.0 / 94.0 / 95.0. Real Franka (50 rollouts): Pick BF16 84%, W8 86%, W8A8 80%; pick-and-place BF16 80%, W8 76%, W8A8 82% — [VLAQuantBench arXiv 2609.25376](https://arxiv.org/html/2609.25376)
- Splitting is supported by existing artefacts: Nota's and Studio-style exports use separate `prefix` and `denoise` IRs with the per-layer KV as interface — [HF nota-pi05-libero-ae-npu-openvino](https://huggingface.co/wjwjwjwj/nota-pi05-libero-ae-npu-openvino)

### Recipe (untested; the cheapest configuration that should stay inside 40 GB / 30 min)
```python
import nncf, openvino as ov
from nncf.quantization.advanced_parameters import (
    AdvancedQuantizationParameters, AdvancedSmoothQuantParameters)

core = ov.Core()
m = core.read_model("prefix_fp16.xml")          # prefix-only IR (SigLIP + Gemma -> KV), NOT the 10-step monolith
q = nncf.quantize(
    m, nncf.Dataset(samples, lambda s: s),      # 32-64 real observation frames spread over episodes
    subset_size=32,
    model_type=nncf.ModelType.Transformer,
    preset=nncf.QuantizationPreset.PERFORMANCE, # symmetric acts; MIXED costs accuracy nothing but is not needed on XMX
    target_device=nncf.TargetDevice.GPU,
    fast_bias_correction=True,                  # never False on transformers (documented hang)
    ignored_scope=nncf.IgnoredScope(patterns=[".*vision_tower.*"], validate=False),  # check real node names first
    advanced_parameters=AdvancedQuantizationParameters(
        smooth_quant_alphas=AdvancedSmoothQuantParameters(matmul=-1)),  # removes one full statistics pass
)
ov.save_model(q, "prefix_w8a8.xml")
```
- If only the monolithic IR is available, restrict scope with `nncf.IgnoredScope(subgraphs=[nncf.Subgraph(inputs=[<first expert node>], outputs=[<actions output>])])` so that no quantiser statistics are collected in the ten unrolled expert copies.
- The OpenVINO IR is portable, so the same script can run on a high-RAM cloud CPU instance and the resulting `.xml/.bin` copied back. A cloud *GPU* does not help, because NNCF's OpenVINO backend only calibrates on CPU.

### Inferences
- Time: at the observed ~15 s per sample (CPU f32, 10-step monolith), 32 samples is ~8 min per statistics pass. SmoothQuant, min-max and fast bias correction each need statistics, so disabling SmoothQuant cuts roughly one third of the time; a prefix-only IR removes the ten expert steps from every calibration inference, which should cut per-sample time further (the expert is about 4% of compute per Intel, so most of the gain comes from fewer statistic points, not FLOPs). I found no source that states how many passes NNCF 3.4 actually makes, so the "three passes" figure is an assumption.
- Memory: the 33–59 GB peak most plausibly comes from the number of statistic points in a 10-step unrolled graph (10 copies of every expert MatMul) combined with SmoothQuant's per-channel statistics and 256 samples. `subset_size=32`, SmoothQuant off and a prefix-only scope all reduce it, but I found no source that quantifies NNCF peak memory for a model of this size. Unverified.
- Gain ceiling over the current state: dynamic quantisation is already active on the prefix FCs and is worth 18% (our measurement), while Intel's projection for full A16 → A8 is 30% (126 → 88). Static quantisation removes the DynamicQuantize kernels (6% of GPU time in our profile) and could quantise attention MatMuls, but the Xe3 SDPA micro-kernel path is f16 (prior notes), so FakeQuantize nodes inside attention may block that fusion and make things slower. Net expectation: 0–6% (0–12 ms of 200 ms), with regression risk. Not competitive with the experiments in sections 3 and 5.
- The expert's 50-row FCs are below the 64-row threshold, so they stay f16-activation today. Static W8A8 could force them to int8, but Intel's own roofline says the AE is memory-bound and does not benefit (40 → 40 ms).

### Gaps
- No published measurement of static W8A8 π0.5 on any Intel GPU. Intel's 35 ms language-model figure is a projection, not a measurement.
- No NNCF documentation of peak calibration memory, and no NNCF GitHub issue found on transformer PTQ OOM (two searches).
- Whether the GPU plugin fuses static FakeQuantize around Gemma's SDPA into int8 kernels on Xe3, or falls back to a slower path, is unverified.
- Whether lowering `OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD` (default 64) is honoured in release wheels is unverified; the blog describes it as an environment variable. If honoured, setting it to 32 would switch the 50-row expert FCs to int8 without any re-export, which is a five-minute test with expected gain near zero.

## 2. INT4 weight compression done right, and whether it is faster than INT8 on the Arc/Xe3 GPU plugin

### Takeaway
The error is arithmetic: 4304 = 16 × 269, so no group size above 16 divides SigLIP's MLP width, and NNCF's default fallback mode is `ERROR`. The clean fix is to exclude the vision tower with `ignored_scope` (every Gemma and expert width is a multiple of 128) or to set `group_size_fallback_mode=ADJUST`. But INT4 is a memory optimisation here, not a latency one: no source shows INT4 beating INT8 on an XMX iGPU for prefill-length inputs, the only cross-hardware VLA timing study found quantised π0.5 slower than F16 on two of three GPUs, and W4 lost 22–24 points on a real Franka.

### Cited Findings
- `validate_group_size()` raises `nncf.InvalidGroupSizeError` when `reduction_channel_size % group_size != 0`; weights with `group_size == -1` are skipped. The message suggests IGNORE or ADJUST — [nncf weight_compression `algorithm.py`](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/src/nncf/quantization/algorithms/weight_compression/algorithm.py)
- `AdvancedCompressionParameters` defaults: `group_size_fallback_mode=GroupSizeFallbackMode.ERROR`, `min_adjusted_group_size=32`, `lora_adapter_rank=256`, `statistics_path=None`. `IGNORE` leaves non-divisible nodes uncompressed; `ADJUST` picks the largest power of two that divides the channel size, is below the requested group size and is at least `min_adjusted_group_size`, otherwise the weight falls back to backup precision — [nncf `advanced_parameters.py`](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/src/nncf/quantization/advanced_parameters.py)
- Sub-algorithm defaults: AWQ `subset_size=32, percent_to_apply=0.002, alpha_min=0.0, alpha_max=1.0, steps=100, prefer_data_aware_scaling=True`; scale estimation `subset_size=64, initial_steps=5, scale_steps=5`; GPTQ `damp_percent=0.1, block_size=128, subset_size=128`; LoRA correction `adapter_rank=8, num_iterations=3, subset_size=128, use_int8_adapters=True` — [nncf `advanced_parameters.py`](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/src/nncf/quantization/advanced_parameters.py)
- "To disable grouped quantization and quantize weights per-channel, set `group_size = -1`"; `ratio=0.9` means 90% of layers at 4-bit and the rest at the backup mode; default backup is `INT8_ASYM` with `group_size=-1`; `backup_mode` may be `INT8_SYM`, `INT8_ASYM` or `NONE`; `scoped_configs` exist and "the last one in the list wins"; GPTQ "can be combined with AWQ and Scale Estimation in any combination" — [NNCF weight-compression Usage.md](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/docs/usage/post_training_compression/weights_compression/Usage.md)
- Layer selection: for non-INT8 modes the last MatMul is dropped from the ratio set unless `all_layers=True`; embeddings and the last layer are "always compressed to a backup_mode" otherwise; there is no first-layer special case. `sensitivity_metric` defaults to `WEIGHT_QUANTIZATION_ERROR` without a dataset and `MAX_ACTIVATION_VARIANCE` with one. Data-aware statistics are per-MatMul mean activations over `subset_size` samples — [nncf weight_compression `algorithm.py`](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/src/nncf/quantization/algorithms/weight_compression/algorithm.py)
- Data-aware mixed precision is "~30% slower" than data-free on a Xeon Gold 6430L; LoRA correction overheads are "negligible"; statistics caching via `AdvancedCompressionParameters(statistics_path=...)` "can significantly reduce compression time during repeated model compression iterations" — [NNCF weight-compression Usage.md](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/docs/usage/post_training_compression/weights_compression/Usage.md)
- NNCF 3.2 added ignoring specific weights by constant node name through `IgnoredScope` for `nncf.compress_weights`; 3.4 added GroupedMatMul to AWQ/scale estimation; 3.3 fixed the HAWQ sensitivity aggregator — [NNCF ReleaseNotes.md](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/ReleaseNotes.md)
- Nota's shipped NPU action expert uses INT4 symmetric **per-channel** on `gate_proj/up_proj/down_proj` (54 MatMuls) and INT8 symmetric per-channel elsewhere, via two `compress_weights` calls, with LIBERO 96.5–97.0% against 96.4% reference (800 episodes, not significant) — [HF nota-pi05-libero-ae-npu-openvino](https://huggingface.co/wjwjwjwj/nota-pi05-libero-ae-npu-openvino)
- VLAQuantBench π0.5: weight-only W4 on LIBERO is near baseline (97.5 / 99.0 / 96.5 / 94.0; W4 grid variants 97–98% on Spatial), but on a real Franka W4 drops Pick 84 → 62% and pick-and-place 80 → 56%, while W8 stays at 86% / 76%. W3 per-channel symmetric collapses to 0% where group-64/128 grids hold 96–99% — [VLAQuantBench arXiv 2609.25376](https://arxiv.org/html/2609.25376)
- LoopQ π0.5 timing: on an H100 "No quantized configuration is faster" (Q8_0 1.06–1.19×, Q4_0 1.24–1.33× slower than F16); on Jetson AGX Thor every W4 configuration was 1.13–1.93× slower; only on an L4 with int8 kernels did W4 beat F16, by 4–7%. "Quantization makes π0.5 smaller, not faster." Every method keeps the action expert at 16 bits — [LoopQ arXiv 2611.05093](https://arxiv.org/html/2611.05093)
- OpenVINO's dynamic-quantisation article says gains shrink "with short token lengths, which are dominated by weight I/O" and gives no INT4-vs-INT8 numbers — [OpenVINO blog](https://blog.openvino.ai/blog-posts/dynamic-quantization-support-from-gpu-with-xmx)
- GPU plugin 2026.4.0 accepts `u8, i8, u4, i4, u2` compressed FC weights. The dynamic-quantisation callback checks `is_wei_i8u8` to choose the group-128 path on arch ≥ Xe2; i4/u4 weights are not rejected but do not take that branch. Dynamic quantisation is disabled per-FC when the innermost size is < 32 or not a multiple of 32 — [intel_gpu `transformations_pipeline.cpp` @2026.4.0](https://raw.githubusercontent.com/openvinotoolkit/openvino/2026.4.0/src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp)
- OpenVINO 2026.4 known issue CVS-194267: initial compilation of an INT4-compressed model (granite-4.0-h-tiny) "may be slow on Core Ultra Series 3" — [OpenVINO release notes source](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/about-openvino/release-notes-openvino.rst)
- OpenVINO 2026.2 NNCF note: "Added INT4/INT8 weight compression support for Vision-Language-Action (VLA) with Pi0.5 model" (from prior notes) — [OpenVINO release notes](https://docs.openvino.ai/releasenotes)

### Recipe (untested)
```python
import nncf
from nncf.quantization.advanced_parameters import (
    AdvancedCompressionParameters, GroupSizeFallbackMode)

# Step A: INT4 only on the Gemma backbone MLPs; vision tower and action expert excluded.
m4 = nncf.compress_weights(
    fp16_model,
    mode=nncf.CompressWeightsMode.INT4_SYM,
    group_size=128,                 # Gemma 2048 / 16384 and expert 1024 / 4096 all divide by 128
    ratio=0.8,                      # 80% of eligible layers INT4, rest backup
    backup_mode=nncf.BackupMode.INT8_SYM,
    ignored_scope=nncf.IgnoredScope(
        patterns=[".*vision_tower.*", ".*gemma_expert.*"], validate=False),  # inspect real node names first
    # optional data-aware part (needs ~32-64 calibration samples run on CPU):
    # dataset=nncf.Dataset(samples), awq=True, scale_estimation=True,
    # sensitivity_metric=nncf.SensitivityMetric.MAX_ACTIVATION_VARIANCE,
    advanced_parameters=AdvancedCompressionParameters(
        group_size_fallback_mode=GroupSizeFallbackMode.ADJUST,  # belt and braces
        statistics_path="wc_stats"),                            # reuse statistics across variants
)
# Step B: second call compresses what is still FP16 (vision tower, expert) to INT8, as Nota did.
m48 = nncf.compress_weights(m4, mode=nncf.CompressWeightsMode.INT8_SYM)
```
- Alternatives for the 4304 width if the vision tower must be INT4: `group_size=-1` (per-channel; safe at 4 bits per VLAQuantBench, not at 3 bits), `group_size=16`, or `ADJUST` with `min_adjusted_group_size=16`. With the default `min_adjusted_group_size=32`, `ADJUST` sends the 4304-wide layer to the INT8 backup, which is the outcome one would want anyway.

### Inferences
- Local arithmetic check: 4304 / 16 = 269 (prime), 4304 mod 128 = 80, 4304 mod 32 = 16; 1152, 2048, 16384, 1024 and 4096 are all multiples of 128. So only SigLIP's `fc2` (input width 4304) triggers the error; the multimodal projector (1152 → 2048) does not.
- INT4 is unlikely to be faster than INT8 on the B390 for the prefix. The prefix GEMMs are compute-bound (Intel) and already run int8×int8 on XMX through dynamic quantisation; INT4 only halves weight traffic and adds unpacking. The benefit of INT4 in LLM practice is in bandwidth-bound decode steps, which π0.5 does not have in the prefix. LoopQ's cross-GPU timings point the same way. This is an inference: no Intel-GPU INT4-vs-INT8 π0.5 timing exists.
- Possible interplay to check before trusting an INT4 A/B: on Xe2+ the plugin routes INT8-weight FCs to a group-128 dynamic-quantisation path, while INT4-weight FCs appear to keep the configured default (per-token). An INT4 model could therefore get coarser activation quantisation than the INT8 model unless `DYNAMIC_QUANTIZATION_GROUP_SIZE` is set explicitly to the same value for both. This reading comes from a summary of one source file (the body of the gs128 decision is in another file) and is unverified.
- The data-aware options are cheap in memory (per-layer mean activations), unlike `nncf.quantize`. Cost is calibration inference on CPU: 32–64 samples × ~15 s ≈ 8–16 min plus the per-layer search. That fits the 30-minute budget; GPTQ and LoRA correction (subset 128) probably do not.
- With 64 GB of RAM and a model of a few GB, memory saving has no demo value. INT4 belongs in the "not worth trying for latency" list; the only reason to run it is to confirm the null result once.

### Gaps
- No measured INT4 π0.5 latency on any Arc/Xe GPU. No documentation of how the oneDNN compressed-FC kernel on Xe3 handles group-wise INT4 scales versus per-channel ones.
- The real node-name patterns in a Studio ONNX-route export are not known to me; patterns above are placeholders (`validate=False` hides a wrong pattern, so print the matched nodes).
- Whether `nncf.BackupMode.INT8_SYM` and `nncf.SensitivityMetric` enum spellings are unchanged in 3.4 was not checked against the 3.4 tag (taken from develop docs).

## 3. Reducing INT8-induced action jitter

### Takeaway
The published evidence says INT8 *weights* are nearly lossless on π0.5 (0.15–0.21% action error in LoopQ; W8 matches BF16 on a real Franka), so a +3.9% error and 2.4× jerk from an "INT8 weight-only" model most likely comes from the GPU plugin's dynamic activation quantisation that INT8 weights switch on in the prefix, not from the weights. The first experiment is therefore a runtime property sweep, and the second is keeping the vision tower FP16 (which disables both weight and activation quantisation there, since dynamic quantisation only applies to compressed-weight FCs). NNCF offers no data-aware INT8 weight compression.

### Cited Findings
- LoopQ offline action error on π0.5 (relative L2 against the unquantised reference): RTN Q8_0 0.15–0.21%; RTN Q4_0 about 2.5%; GPTQ W4 about 2.2%; AWQ W4 about 3.6%. LIBERO success is unchanged for all 4- and 8-bit configurations, and the paper's main finding is that offline error does not predict success. It does not measure smoothness or jerk — [LoopQ arXiv 2611.05093](https://arxiv.org/html/2611.05093)
- LoopQ's "A8-dyn" is int8 activations with a per-call scale computed per 32 values, and W8A8-dyn scores 96.7% against 96.5% F16 — [LoopQ arXiv 2611.05093](https://arxiv.org/html/2611.05093)
- OpenVINO GPU dynamic quantisation: default granularity is per-token (`dynamic_quantization_group_size = UINT64_MAX`); `0` removes the node; smaller groups (the article's example is 256) "can improve accuracy at some performance cost"; Intel warns that users "may see noticeable degradation depending on model and workload"; scales are per-group min/max along the innermost axis — [OpenVINO blog](https://blog.openvino.ai/blog-posts/dynamic-quantization-support-from-gpu-with-xmx)
- `OV_GPU_ASYM_DYNAMIC_QUANTIZATION` enables asymmetric dynamic quantisation with a zero-point — [OpenVINO blog](https://blog.openvino.ai/blog-posts/dynamic-quantization-support-from-gpu-with-xmx). In the 2026.4.0 source the asymmetric mode is flagged when combined with a finite group size (`asymmetric_dyn_quant && adj_group_size != UINT64_MAX`), and dynamic quantisation is skipped for FCs with any f32 input, rank > 3, or an innermost size that is < 32 or not a multiple of 32 — [intel_gpu `transformations_pipeline.cpp` @2026.4.0](https://raw.githubusercontent.com/openvinotoolkit/openvino/2026.4.0/src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp)
- NNCF rejects data-aware options for INT8 modes: the validator refuses `ratio != 1`, `all_layers`, `sensitivity_metric`, `dataset`, `awq`, `scale_estimation`, `gptq`, `lora_correction` and `backup_mode` when the mode is INT8_SYM/ASYM. INT8 is per-channel (`group_size=-1`) — [nncf weight_compression `algorithm.py`](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/src/nncf/quantization/algorithms/weight_compression/algorithm.py)
- NNCF's default mode is `INT8_ASYM`; Intel's robotics-ai-suite converter uses `INT8_ASYM` and Studio's hook uses `INT8_SYM` (prior notes) — [NNCF weight-compression Usage.md](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/docs/usage/post_training_compression/weights_compression/Usage.md)
- Component sensitivity on π0.5 (VLAQuantBench, Spatial, uncalibrated W4A4, a stress setting): vision-encoder-only 37.0%, LLM-only 0.0%, action-head-only 70.5%, baseline 99.0%. With two-episode calibration: VE 80.0%, LLM 70.5%, AH 98.5%. Layer counts: VE 162 linear layers (411.1M weights), LLM 126 (1981.8M), AH 167 (430.0M). Isolated losses "do not add up" (attention −1.0 pp and adaRMS −5.5 pp alone, −97.5 pp together) — [VLAQuantBench arXiv 2609.25376](https://arxiv.org/html/2609.25376)
- QuantVLA (π0.5, W4A8, LIBERO avg): LLM-only 96.5%, DiT-only 71.6%, LLM+DiT 76.3%, LLM + DiT-MLP-only 95.4% against FP16 97.1%; with its calibration, 97.6%. It keeps attention projections (Q, K, V, O) in floating point because they are "most sensitive to upstream distribution shifts", and attributes failures to attention-logit temperature drift and residual-energy drift. It does not state whether the vision encoder is quantised, and gives no latency — [QuantVLA arXiv 2602.20309](https://arxiv.org/html/2602.20309v4)
- QVLA (ICLR 2026) reports the cross-modal projector and action head as the most sensitive components and sensitivity as highly heterogeneous per channel — [QVLA arXiv 2602.03782](https://arxiv.org/html/2602.03782) (via search summary; not read in full)
- Real-robot data: π0.5 W8 weight-only 86% / 76% against BF16 84% / 80% (Franka, 50 rollouts); W4A8 falls to 22% / 32% — [VLAQuantBench arXiv 2609.25376](https://arxiv.org/html/2609.25376)
- NNCF 3.2 allows ignoring specific weights by constant node name via `IgnoredScope` in `compress_weights` — [NNCF ReleaseNotes.md](https://raw.githubusercontent.com/openvinotoolkit/nncf/develop/ReleaseNotes.md)

### Recipes (untested), in order
1. Runtime sweep on the existing INT8_SYM IR, measuring jerk and action error against FP16 on replayed frames:
```python
for gs in ("0", None, "256", "128", "64", "32"):          # None = plugin default
    cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f16", "CACHE_DIR": cache}
    if gs is not None: cfg["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = gs
    cm = core.compile_model(model, "GPU", cfg)
```
   Also one run with the environment variable `OV_GPU_ASYM_DYNAMIC_QUANTIZATION=1` at the default group size. If `gs=0` removes the jitter, the weights are exonerated and the remaining work is finding the cheapest group size or scope that keeps most of the 18%.
2. Vision tower (and projector) FP16, rest INT8, starting from the FP16 IR so no 54 GB export is needed:
```python
m = nncf.compress_weights(
    fp16_model, mode=nncf.CompressWeightsMode.INT8_SYM,
    ignored_scope=nncf.IgnoredScope(patterns=[".*vision_tower.*", ".*multi_modal_projector.*"], validate=False))
```
3. Bisect further with the same call: (a) Gemma first and last two blocks FP16; (b) Gemma attention projections `q_proj|k_proj|v_proj|o_proj` FP16 and only MLPs INT8 (QuantVLA's layout); (c) expert INT8 only (control). Each variant is one data-free `compress_weights` run plus a compile.
4. `mode=nncf.CompressWeightsMode.INT8_ASYM` as a GPU-only variant (do not reuse it on the NPU).

### Inferences
- Why activation quantisation is the prime suspect: our INT8 action error (+3.9%) is about 20 times LoopQ's INT8 weight-only figure and resembles its 4-bit numbers; our own finding that the jitter comes from the prefix matches the fact that dynamic quantisation only fires above 64 rows, i.e. in the prefix and not in the 50-row expert; and per-token scaling over a 2048-wide (Gemma) or 1152-wide (SigLIP) activation is far coarser than LoopQ's per-32 scaling that was lossless. This is a hypothesis until the `gs=0` run is measured.
- Cost of the FP16 vision tower: the encoder is 22 ms of ~200 ms in our profile and Intel's projection for VE is 24 ms (W16) against 18 ms (W8A16) or 13 ms (W8A8), so expect roughly +4–8 ms. FP16 Gemma attention projections would cost more, because they are a sizeable share of the 69% GEMM time; unmeasured.
- Smaller dynamic-quantisation groups cost throughput. If the default on Xe3 for INT8 weights is already the group-128 path (see section 5), the meaningful sweep values are 64 and 32, and the latency cost is unknown.
- Jitter is partly a decision-noise amplification effect: errors in the prefix KV are shared by all ten denoising steps and all 50 actions, so they shift whole chunks relative to each other. That is consistent with chunk-to-chunk jerk rising more (2.4×) than mean error (+3.9%). Reasoning only; no source measures this.
- The literature tests success rate, not smoothness. A 2.4× jerk increase may or may not matter for task success; LoopQ's result that offline error does not predict success argues for a closed-loop check before spending days on it.

### Gaps
- No paper measures action jerk or chunk-to-chunk discontinuity under quantisation for any VLA. HoloQ-VLA, FoldQuantVLA and BitVLA were not re-read in this pass; BitVLA did not surface in search at all, and the first two are cited in the earlier notes only for success rates.
- No published per-layer INT8 sensitivity map for π0.5; all published component ablations are at 4 bits.
- Whether QuantVLA, HoloQ-VLA or LoopQ quantise SigLIP is partly unclear: LoopQ says SigLIP "is also quantized"; QuantVLA does not say.

## 4. NPU: which sub-graphs run well, and can offloading the vision encoder shorten wall time

### Takeaway
The only sub-graph with published Panther Lake NPU results is the action expert (Nota: 144 → 78 ms for 10 steps after graph surgery and symmetric INT4/INT8), and that is still slower than the iGPU in a synchronous pipeline. Nobody has published SigLIP or Gemma on this NPU. Moving a 22 ms vision encoder to a device with under half the iGPU's INT8 throughput cannot shorten a sequential VE → LM → AE chain; it only frees iGPU time in a pipelined RTC schedule, which is outside quantisation.

### Cited Findings
- Nota's NPU action expert (Core Ultra X7 358H, NPU 5, driver 32.0.100.5540, OpenVINO 2026.4.1, NNCF 3.4.0), 10-step loop: FP16 144.2 ms (867 MB) → graph-optimised FP16 125.2 ms → mixed INT4/INT8 104.8 ms → both 77.8 ms (210 MB). "With asymmetric INT8 (the NNCF default), the NPU compiler stores weights in FP16 size and the model is not faster." The MQA `repeat_kv` rewrite into a 4-D Broadcast moved attention from SHAVE cores to DPU MatMuls. `NPU_TURBO=YES` avoided a 13–22 ms slowdown while the iGPU is busy. LIBERO 96.5–97.0% against 96.4% (800 episodes). The vision encoder and language model stayed FP16 on the iGPU; the card reports no NPU trials for them and no activation quantisation — [HF nota-pi05-libero-ae-npu-openvino](https://huggingface.co/wjwjwjwj/nota-pi05-libero-ae-npu-openvino)
- Intel's paper places VE and LE on the iGPU because they are "compute-bound, GEMM-heavy tasks that fit well on the iGPU, similar to LLM prefill", and the AE on the NPU "like an LLM decode step"; it says the split "adds minor overhead" and that the benefit is that it "saves power and frees up the iGPU for other tasks" and overlaps AE with the next VE+LE under RTC — [Intel π0.5 paper](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- Platform throughput: iGPU "up to 123 INT8-dense TOPS", NPU "up to 50 INT8 TOPS" — [Intel π0.5 paper](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- OpenVINO 2026.4 NPU changes: bounded dynamic shapes in preview for vision models (validated on ESPCN only), ONNX external weights in the weightless pipeline, compatibility strings for weightless-compiled models; known issue CVS-233822 (crashes with NPU drivers 32.0.100.4300/4509/4512 on Windows, fixed in 32.0.100.5540) — [OpenVINO release notes source](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/about-openvino/release-notes-openvino.rst)
- `nncf.quantize` has `target_device=nncf.TargetDevice.NPU` — [OpenVINO basic quantization flow](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.rst)
- Static INT8 on a vision encoder alone is an established OpenVINO VLM recipe (`OVPipelineQuantizationConfig` with `OVQuantizationConfig(bits=8)` for `vision_embeddings_model`, 50 samples), described as "a noticeable performance improvement without significant accuracy degradation", without numbers and on CPU — [HF blog openvino-vlm](https://huggingface.co/blog/openvino-vlm)

### Inferences
- In a single synchronous inference the vision encoder must finish before the language model starts, so there is nothing for the NPU to run in parallel with; wall time becomes VE(NPU) + LM + AE, and VE on the NPU is very unlikely to beat 22 ms given the 50-vs-123 TOPS ratio and Nota's evidence that NPU weights stream from DRAM. Expected effect: slower.
- The only arrangement in which an NPU vision encoder helps is cross-chunk pipelining (NPU encodes chunk n+1 images while the iGPU denoises chunk n). The upper bound is the 22 ms of iGPU time freed, minus NPU contention on shared memory bandwidth, and the images would be ~80 ms staler. That is a scheduling change with a small ceiling, not a quantisation win.
- If any NPU work is done, the transferable Nota findings are: symmetric weights only (`INT8_SYM`, `INT4_SYM` per-channel), static shapes, `NPU_TURBO=YES`, and checking with NPU profiling that attention lands on DPU rather than SHAVE.

### Gaps
- No published SigLIP-So400m or Gemma-2B latency on the Panther Lake NPU in FP16 or INT8 (one targeted search; found only a CNN benchmark and a Snapdragon study). Whether the NPU compiler handles the 4304-wide SigLIP MLP and 768-token batch efficiently is unknown.
- Whether static W8A8 from `nncf.quantize` (as opposed to weight-only) improves NPU latency for transformer blocks on NPU 5 is not documented in any source found.

## 5. Activation-format options on the GPU plugin and 2026.4 / 2026.5 changes

### Takeaway
Only `DYNAMIC_QUANTIZATION_GROUP_SIZE` matters for this model. `KV_CACHE_PRECISION` applies to stateful/PagedAttention LLM graphs and not to a stateless π0.5 IR; `ACTIVATIONS_SCALE_FACTOR` is an f16-overflow workaround, not a speed feature; f32 inference precision is slower and switches dynamic quantisation off. OpenVINO 2026.4 (16 Sep 2026) is the latest release in the repository's release-notes file and contains no GPU change aimed at this workload.

### Cited Findings
- The GPU plugin exposes `ov::hint::dynamic_quantization_group_size`, `ov::hint::kv_cache_precision`, `ov::hint::activations_scale_factor`, `ov::hint::inference_precision`, `ov::cache_dir`, `ov::cache_mode`, `ov::weights_path`, `ov::intel_gpu::hint::enable_sdpa_optimization` as read-write compile-time properties. Supported internal precisions are f32, f16, u8, i8, u1; the checklist says "Prefer FP16 inference precision over FP32" — [GPU device docs source](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.rst)
- Dynamic-quantisation semantics: default `UINT64_MAX` = per-token on XMX GPUs; `0` = "the dynamic_quantize node is not added to the graph at all"; other values are the group size along the innermost axis; applied only above 64 rows (`OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD`); expected gains "often by several tens of percent" for long inputs; verify by looking for `dynamic_quantize` in the execution graph — [OpenVINO blog](https://blog.openvino.ai/blog-posts/dynamic-quantization-support-from-gpu-with-xmx)
- 2026.4.0 source: the pass is registered only when `device_info.supports_immad && config.get_use_onednn()`; `use_gs128_for_int8_per_token` requires `arch >= cldnn::gpu_arch::xe2` and non-uniform work-group support and is keyed on i8/u8 weights; the pass is not registered when the model's permitted maximum group size is below the configured one; per-FC it is skipped for f32 inputs, rank > 3, innermost size < 32 or not a multiple of 32, and for grouped mode without non-uniform work-group support — [intel_gpu `transformations_pipeline.cpp` @2026.4.0](https://raw.githubusercontent.com/openvinotoolkit/openvino/2026.4.0/src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp)
- Activation scaling runs only `if (activations_scale_factor > 0.f && infer_precision == ov::element::f16)`, via `ScaleDownSingleLayer` — [intel_gpu `transformations_pipeline.cpp` @2026.4.0](https://raw.githubusercontent.com/openvinotoolkit/openvino/2026.4.0/src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp)
- KV-cache compression code is tied to KV-cache variables (`if (!has_shared_kv_cache_vars(func))`, key-cache group size 16 for BY_CHANNEL, forced i8 with XAttention) — [intel_gpu `transformations_pipeline.cpp` @2026.4.0](https://raw.githubusercontent.com/openvinotoolkit/openvino/2026.4.0/src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp)
- 2026.4 GPU plugin notes: faster TTFT/TPOT for Gemma 4, model caching for linear-attention models (Qwen3.5), lower TTFT for Qwen3-Omni / Qwen3-VL, remote tensors from file paths, OpenCL update. NNCF notes: GroupedMatMul in AWQ/scale estimation, an extended RoPE ignored pattern "for more accurate 4-bit compression", PyTorch 2.13. No bullets on dynamic quantisation, KV cache or memory. The file has no 2026.5 section — [OpenVINO release notes source](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/about-openvino/release-notes-openvino.rst)
- A real f16 failure mode exists for quantised π0.5 on another engine: LoopQ's AWQ int8-kernel path had "one FP16 overflow" that crashed its server on a LIBERO Goal frame — [LoopQ arXiv 2611.05093](https://arxiv.org/html/2611.05093)

### Inferences
- Our 891 DynamicQuantize kernels and the 18% penalty when disabled confirm the documented behaviour on Xe3. Per the earlier notes and the source, the Xe3 default for INT8 weights is probably the group-128 variant, not strict per-token; the execution graph or an OpenCL intercept dump would settle which is running. Unverified.
- `INFERENCE_PRECISION_HINT=f32` is useful only as a diagnostic (it disables dynamic quantisation through the f32-input rule and rules out f16 overflow); it should cost more than the 18% because the FCs also lose f16.
- `ACTIVATIONS_SCALE_FACTOR` is worth one run only if FP16 already shows saturated or NaN activations; it does not address int8 quantisation noise.
- Since 2026.4's GPU work targets GenAI LLM pipelines, an upgrade within 2026.4.x should not change π0.5 latency. I could not find evidence of a 2026.5 release as of 14 Nov 2026.

### Gaps
- The body of the gs128 decision (`ShouldUseGs128`) is in a file not read here, so the exact effective default group size for INT8 and INT4 weights on Xe3 is not confirmed.
- The public GPU docs page describes none of these three properties beyond listing them; the semantics above come from a 2025.2 blog post and the 2026.4.0 source.
- The release-notes page on docs.openvino.ai could not be parsed (navigation only); the repository `.rst` on master was used instead and may lag a just-published release.

## 6. Model size and load-time effects that matter for a demo

### Takeaway
Use `CACHE_DIR` with the default speed-optimised cache; because the IR is static, the GPU plugin caches the full compiled blob. INT8 weights halve the on-disk model, which is the only size effect of note; weightless caching (`CACHE_MODE=OPTIMIZE_SIZE`) trades load speed for disk space and is not what a demo wants.

### Cited Findings
- GPU model caching uses `ov::cache_dir`; static models are fully cached, dynamic models only get kernel caching (`.cl_cache` files) — [GPU device docs source](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.rst)
- Weightless caching creates "a cache file which doesn't contain the weights of the model", is selected with `CACHE_MODE=OPTIMIZE_SIZE`, and is "supported only by the GPU Plugin and IR model format"; some weights changed by compile-time transformations may still be stored. `OPTIMIZE_SPEED` is described as faster loading, `OPTIMIZE_SIZE` as a smaller cache — [OpenVINO model caching overview](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html) (via search snippet)
- `ov::enable_mmap(BOOL)` is a Core property; with on-demand mapping "there is no need to store all weights in RAM"; for models on removable or network drives "mapping may lead to latency increase" — [OpenVINO optimizing memory usage](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/optimizing-memory-usage.html) (via search snippet)
- A `WEIGHTS_PATH` hint gives the weight location for weightless models and a `COMPILED_BLOB` hint speeds compilation (2025.2 release notes) — [OpenVINO 2025.2 release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-2.html) (via search snippet)
- INT8 weight-only compression "reduced the model size by half" — [Intel π0.5 paper](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf). LoopQ: π0.5 file 6.71 GB (F16) → 2.40–2.70 GB at 4 bits, peak memory 7.04 → 3.0–3.5 GiB — [LoopQ arXiv 2611.05093](https://arxiv.org/html/2611.05093). Nota's NPU expert blob: 867 → 210 MB — [HF nota-pi05-libero-ae-npu-openvino](https://huggingface.co/wjwjwjwj/nota-pi05-libero-ae-npu-openvino)
- 2026.4 known issue: slow first compile of an INT4 model on Core Ultra Series 3 (CVS-194267) — [OpenVINO release notes source](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/about-openvino/release-notes-openvino.rst)

### Recipe (untested)
```python
core = ov.Core()
core.set_property({"CACHE_DIR": "ov_cache"})            # default CACHE_MODE is the speed-optimised one
cm = core.compile_model("pi05_int8.xml", "GPU",
        {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f16"})
```
Warm the cache once before the demo with exactly the same model file, device, properties and driver, since any change in them is expected to invalidate the blob.

### Inferences
- Each quantisation variant from sections 2–3 is a different model and compiles its own cache entry, so a sweep of N variants pays N cold compiles; keep the cache directory per variant to avoid confusion.
- All re-compression recipes above start from the saved FP16 IR with `core.read_model`, so they avoid the 54 GB export peak; `compress_weights` INT8 holds roughly one FP16 copy plus the compressed copy. This is reasoning, not a measured figure.
- With 64 GB of unified memory there is no demo-relevant reason to chase a smaller footprint; load time is governed by the cache hit, not by INT8 vs INT4.

### Gaps
- No measured cold or warm compile times for π0.5 on the B390 in any source; no statement of how `ENABLE_MMAP` interacts with GPU blob loading.
- The model-caching and memory pages were seen only as search snippets, not fetched in full.
