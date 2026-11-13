# Kernel- and compiler-level inference optimisation on Intel GPUs (XPU) in PyTorch, late 2026, for small robot policies (ACT) and VLA policies (pi0.5/SmolVLA-style)

Context: as of Nov 2026 the docs list PyTorch 2.11–2.15 for XPU ([2.15 cpp_extension docs](https://docs.pytorch.org/docs/2.15/cpp_extension.html); [2.13 XPU getting-started](https://docs.pytorch.org/docs/2.13/notes/get_start_xpu.html)), and 2.16 nightlies exist ([arc-a770 notes](https://github.com/BarzinL/arc-a770-pytorch-notes)). Install: `pip install torch torchao --index-url https://download.pytorch.org/whl/xpu` ([PyTorch 2.8+TorchAO blog](https://pytorch.org/blog/pytorch-2-8torchao-unlock-efficient-llm-inference-on-intel-ai-pcs/)). The workload is batch size 1, latency-bound and made of many small kernels, so host and launch overhead matter as much as GEMM throughput.

## 1. torch.compile / Inductor on XPU: status, modes, backends, speedups

### Takeaway
torch.compile on XPU is mature. Inductor generates Triton (intel-xpu-backend-for-triton) kernels and calls oneDNN for GEMM/conv/SDPA, on Linux and Windows. `max-autotune` is supported. **`mode="reduce-overhead"` silently does nothing on XPU as of PyTorch 2.15**, because cudagraph trees are CUDA-only, so graph capture must be done manually (section 3). Published speedups are 1.1–2.7x for LLM prefill/decode on Arc B580. There are per-release regressions of up to about 30% on some models, so pin a version and measure.

### Cited Findings
- Intel's 2026 H1 status post calls torch.compile on XPU "production-ready". On Windows XPU, torch.compile, AOT Inductor, the Kineto profiler and the SYCL C++ extension are enabled. Hardware coverage: Data Center GPU Max, Arc A/B-series, Core Ultra Series 3 (Panther Lake) — [dev-discuss 2026 H1 update](https://dev-discuss.pytorch.org/t/intel-gpu-cpu-enabling-status-and-feature-plan-2026-h1-update/3320)
- Inductor uses TorchDynamo for tracing and TorchInductor for Triton kernel generation on Intel GPUs. Linux support since 2.5, Windows since 2.7. The decode gains come from fused RoPE/RMSNorm kernels — [PyTorch 2.8+TorchAO blog](https://pytorch.org/blog/pytorch-2-8torchao-unlock-efficient-llm-inference-on-intel-ai-pcs/)
- Measured torch.compile speedups (fp16, Arc B580, Windows, PyTorch 2.8):

  | Model | Prefill | Decode |
  |---|---|---|
  | Llama-3.2-3B | 2.42x | 1.58x |
  | Phi-3-mini | 1.12x | 1.39x |
  | Qwen3-4B | 2.68x | 1.99x |

  Source: [PyTorch 2.8+TorchAO blog](https://pytorch.org/blog/pytorch-2-8torchao-unlock-efficient-llm-inference-on-intel-ai-pcs/)
- PyTorch 2.7 reported Arc B580 torch.compile-over-eager results on the Dynamo benchmark suite (HF, TIMM, TorchBench), and up to 3x Stable Diffusion fp16 inference over 2.6 — [PyTorch 2.7 Intel GPU blog](https://pytorch.org/blog/pytorch-2-7-intel-gpus/)
- `max-autotune` is supported on Intel GPUs (stated in the FlexAttention context). PyTorch 2.9 integrated Triton XPU and added FlexAttention forward and backward on Intel GPUs (API-unstable) — [FlexAttention on Intel GPUs blog](https://pytorch.org/blog/pytorch-2-9-flexattention-optimization-practice-on-intel-gpus/); [PyTorch 2.9 blog](https://pytorch.org/blog/pytorch-2-9/)
- PyTorch 2.11 added for XPU: Triton online-softmax kernels, AOTInductor standalone compile, Intel Triton kernel format switched from spv to zebin, and FlexAttention TMA (tensor-descriptor) path on by default — [v2.11.0 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.11.0)
- **reduce-overhead is a silent no-op on XPU.** Tested on 2.15.0a0 with triton-xpu 3.8.0:
  - Inductor's gate is `_CUDAGRAPH_SUPPORTED_DEVICE_TYPES = {"cuda"}`. The compiled function returns correct results but captures no graph, and logs a misleading "skipping cudagraphs due to multiple devices".
  - There is no `torch._inductor.config.triton.xpugraphs`.
  - Fix PR pytorch#197866 was still a draft, targeted at 2.16.
  - Workaround: eager `torch.xpu.XPUGraph`, `torch.xpu.graph`, `torch.xpu.make_graphed_callables`.
  - Source: [torch-xpu-ops #5338](https://github.com/intel/torch-xpu-ops/issues/5338)
  - A related PR series "[xpu cudagraph] 5/N: Enable cudagraph trees on XPU" exists, but its merge status was not verified — [pytorch#200279](https://github.com/pytorch/pytorch/pull/200279)
- Roadmap items, not yet available:
  - SYCL-TLA backend for torch.compile, targeting Linux discrete GPUs
  - XPUGraphTree integration into Inductor
  - Triton backend improvements
  - Source: [dev-discuss 2026 H1 update](https://dev-discuss.pytorch.org/t/intel-gpu-cpu-enabling-status-and-feature-plan-2026-h1-update/3320)
- Inductor-only regressions on Arc B580 (Windows):
  - up to about 32% worse latency on a pyhpc model in 2.13 — [torch-xpu-ops #4052](https://github.com/intel/torch-xpu-ops/issues/4052)
  - 8.3% on timm_nfnet amp-fp16 inference in 2.14 RC — [#5065](https://github.com/intel/torch-xpu-ops/issues/5065)
  - 9–11% on llama training in 2.14 — [#4925](https://github.com/intel/torch-xpu-ops/issues/4925)
- Arc A770 correctness bugs on 2.15/2.16 nightlies:
  - `sum(0)`, `mean(0)` and `amax(0)` over tens of thousands of rows return wrong values (torch-xpu-ops#6033).
  - `clip_grad_norm_` foreach path overcounts (#6035). Workaround: `foreach=False`.
  - These mostly affect training, but reductions also appear in inference.
  - Source: [arc-a770-pytorch-notes](https://github.com/BarzinL/arc-a770-pytorch-notes)

### Inferences
- For ACT (ResNet18 + about 80M transformer) at batch size 1, most of the time is launch and host overhead. torch.compile (default or max-autotune) fuses elementwise, norm and activation ops and reduces the kernel count. A realistic gain is about 1.3–2x, in line with the LLM decode numbers. Then wrap the compiled callable in a manual XPUGraph (section 3) to remove the remaining launch cost. Do not rely on `mode="reduce-overhead"`.
- Use `torch.compile(model, mode="max-autotune-no-cudagraphs")` or `mode="max-autotune"`. On XPU they behave the same, because the cudagraph part is gated off. Use `dynamic=False` and fixed shapes (fixed image size, fixed action chunk, static KV cache) so a VLA's denoising loop does not recompile.
- Pin one PyTorch version for the hackathon, given regressions of up to about 30% between releases.

### Gaps
- No published torch.compile speedups specifically for ViT/ResNet18 at batch size 1 on Lunar Lake or Panther Lake iGPUs.
- Merge status of the cudagraph-trees-on-XPU PRs (#197866, #200279) as of Nov 2026 is unconfirmed.

## 2. intel-xpu-backend-for-triton: maturity, custom kernels, block loads, DPAS, gaps vs CUDA

### Takeaway
Triton on XPU ships with the PyTorch XPU wheels (triton-xpu 3.8 alongside torch 2.15). It is mature enough for custom fused kernels. GEMM-like kernels must use `tl.make_tensor_descriptor` (2D, last stride 1) to reach 2D block IO and DPAS (XMX). Pointer-arithmetic kernels can be 2x to 30x slower. Block pointers have been removed as of v3.9.

### Cited Findings
- **Officially listed hardware:** Data Center GPU Max, Flex, Arc A770, Arc B580, Arc Pro B-series. Lunar Lake is not listed. Ubuntu 22.04/24.04; Windows is experimental. Toolchain: Intel Deep Learning Essentials 2026.1. Not compatible with IPEX — [intel-xpu-backend-for-triton README](https://github.com/intel/intel-xpu-backend-for-triton)
- **Performance guidance from the README:**
  - Use `tl.make_tensor_descriptor` (host- or device-side) for `tl.dot` operands and outputs. This gives ">2x" over tensor-of-pointers code.
  - Prefer 2D descriptors with last stride 1. 3D and higher descriptors are poorly optimized.
  - Write `acc = tl.dot(a, b, acc)`, not `acc += tl.dot(a, b)`. The first form maps to a single DPAS.
  - Put the tile index on grid `axis=0`. Getting this wrong costs 20% to 2x.
  - Autotune `num_warps`, `num_stages` and `grf_mode`. `grf_mode="256"` gives a large register file for register-heavy kernels.
  - Making strides `tl.constexpr` can silently make offset math int32. Cast with `.to(tl.int64)` for large offsets.
  - Source: [README](https://github.com/intel/intel-xpu-backend-for-triton)
- **Debug and perf env vars:**
  - `TRITON_PRINT_AUTOTUNING=1`
  - `TRITON_KERNEL_DUMP=1` + `TRITON_DUMP_DIR`
  - `TRITON_ALWAYS_COMPILE=1`
  - `TRITON_INTERPRET=1`
  - `MLIR_ENABLE_DUMP=1`
  - `USE_IR_LOC={ttir,ttgir}` (for profiling)
  - `TRITON_F32_DEFAULT` (ieee/tf32/tf32x3)
  - `DISABLE_LLVM_OPT` (loop strength reduction can swing results by up to 10%)
  - Source: [README](https://github.com/intel/intel-xpu-backend-for-triton)
- The v3.9.0 release notes list "Remove block pointer support" as a breaking change. Use tensor descriptors instead of `tl.make_block_ptr` — [v3.9.0 release](https://github.com/intel/intel-xpu-backend-for-triton/releases/tag/v3.9.0)
- **Open performance bugs (late 2026):**
  - A flattened persistent GEMM loses 2D block IO after loop fusion and is about 30x slower on BMG — [#8687](https://github.com/intel/intel-xpu-backend-for-triton/issues/8687)
  - The bf16 GEMM epilogue uses 8x16 block stores, leaving bmm about 14% slower than oneDNN — [#8723](https://github.com/intel/intel-xpu-backend-for-triton/issues/8723)
  - On Xe2, the tensor-of-pointers path rebuilds 2D descriptors every K iteration, with about 15.5% DPAS density versus 39% for the block path — [#7689](https://github.com/intel/intel-xpu-backend-for-triton/issues/7689)
  - Unexpected MOVs between 2D block load and DPAS in the FP8 scaled_mm kernel — [#7561](https://github.com/intel/intel-xpu-backend-for-triton/issues/7561)
- A 2026 paper (Xe-Forge) applies LLM-driven kernel optimisation to Intel GPU — [arXiv 2605.26118](https://arxiv.org/pdf/2605.26118) (not read in detail).

### Inferences
- For a hackathon, the best uses of custom Triton are fusing elementwise and norm chains (e.g. AdaRMSNorm/time-conditioning plus Euler update in the flow-matching loop) and small attention variants. Leave large GEMMs to oneDNN, which Inductor calls by default and which beats Triton bf16 bmm by about 14% per #8723.
- Always use `torch.library.triton_op` / `@triton.jit` with tensor descriptors. Never port CUDA-style pointer-arithmetic GEMMs as-is.

### Gaps
- No published head-to-head comparison of Triton-XPU and Triton-CUDA efficiency (% of peak) was found.
- `threads_per_warp` (sub-group size 16 vs 32) guidance was not found in current docs.

## 3. Graph capture on XPU (CUDA-graph equivalent)

### Takeaway
`torch.xpu.XPUGraph` / `torch.xpu.graph` / `torch.xpu.make_graphed_callables` shipped in **PyTorch 2.11**. They are absent in 2.10, where code that probes them crashes rather than falling back. They work in eager mode and give large launch-overhead cuts. On Arc A-series (Alchemist) the driver does not report record/replay support, so it must be forced with debug env vars (not validated by Intel). This is the biggest lever for a flow-matching denoising loop (e.g. 10 steps x N layers) at batch size 1.

### Cited Findings
- The API mirrors CUDA graphs:
  ```python
  g = torch.xpu.XPUGraph()
  with torch.xpu.graph(g):
      ...          # capture
  g.replay()
  ```
  - Helpers: `torch.xpu.is_current_stream_capturing()`, `torch.xpu.graph_pool_handle()`, `torch.xpu.make_graphed_callables()`.
  - Uses a private memory pool on the XPU caching allocator and RNG-state tracking. It is built on SYCL runtime graph APIs.
  - Milestone 2.11.0; the RFC was closed as completed on 2026-04-29.
  - Source: [RFC pytorch#162143](https://github.com/pytorch/pytorch/issues/162143)
- The PyTorch 2.11 blog's headline Intel feature is XPUGraph, which "reduces CPU overhead such as kernel launch and Python runtime overhead" — [PyTorch 2.11 blog](https://pytorch.org/blog/pytorch-2-11-release-blog/)
- On 2.10 wheels the bindings are missing and code crashes rather than falling back. Upstream stream capture arrived in 2.11 — [DeepSpeed PR #9007](https://github.com/deepspeedai/DeepSpeed/pull/9007); [YuE PR #275](https://github.com/multimodal-art-projection/YuE/pull/275)
- **Arc A770 needs forcing:**
  - Set `SYCL_UR_USE_LEVEL_ZERO_V2=1 NEOReadDebugKeys=1 OverrideRecordReplayGraphCapability=1`. The first variable selects the Level Zero v2 adapter; the other two make the driver report immutable-graph support.
  - Results: a 64-step GRU loop replays at 111 µs/step (width 1024) and 21 µs/step (width 512), versus about 173–176 µs eager, with output matching eager exactly. Training steps replayed 6–14x faster.
  - Limitations: embedding backward failed to capture with a Level Zero error; data-dependent output shapes (masked indexing) cannot be captured. Not validated by Intel.
  - To share one memory pool across several captured callables, pass them together in one `make_graphed_callables((f1, f2), ...)` call.
  - Source: [arc-a770-pytorch-notes](https://github.com/BarzinL/arc-a770-pytorch-notes)
- Known bug: the graph memory pool id becomes permanently unusable after release (`use_count > 0` INTERNAL ASSERT) — [pytorch#198794](https://github.com/pytorch/pytorch/issues/198794)
- XPU registered as a graph-safe RNG device (relevant to sampling noise inside a captured flow-matching loop) — [pytorch#203672](https://github.com/pytorch/pytorch/pull/203672)
- Roadmap: generalize graph record/replay through `torch.accelerator` and integrate XPUGraphTree into Inductor — [dev-discuss 2026 H1](https://dev-discuss.pytorch.org/t/intel-gpu-cpu-enabling-status-and-feature-plan-2026-h1-update/3320)

### Inferences
- **Recipe:**
  1. Compile the denoise step (or the whole N-step loop with a fixed step count) using `torch.compile(..., mode="max-autotune-no-cudagraphs", dynamic=False)`.
  2. Allocate static input buffers (obs embeddings, KV prefix cache, noise, timestep tensors).
  3. Warm up on a side stream (`torch.xpu.Stream`).
  4. Capture with `torch.xpu.graph`.
  5. At runtime, `copy_()` new observations into the static buffers and call `g.replay()`.
  - Generate noise outside the graph or rely on graph-safe RNG.
- On B-series (Xe2), Lunar Lake and Panther Lake, the driver presumably reports graph capability natively, since the A770 needed an override. This is unverified, so test with a tiny capture first.

### Gaps
- No official per-device support matrix for XPUGraph, and no published XPUGraph speedups for transformer inference.
- Whether Lunar Lake or Panther Lake iGPU drivers expose record/replay by default is unverified.

## 4. Scaled dot product attention on XPU

### Takeaway
`F.scaled_dot_product_attention` on XPU dispatches to oneDNN's fused SDPA ("overrideable" backend, the default since 2.7, with GQA since 2.8) or to a SYCL-TLA (cutlass-sycl) FlashAttention kernel. The flash kernels are AOT-built only for PVC and BMG (Xe2+); PVC is excluded for accuracy reasons. Use fp16/bf16. Head dims 64/96/128/192 (possibly 256) are supported by flash. Benchmarks on Xe2 show flash winning at d64 and oneDNN winning at d128 decode. On Arc A-series there is no fused attention for training, but inference uses oneDNN graph SDPA.

### Cited Findings
- oneDNN has been the default XPU SDPA backend since 2.7, optimized for prefill and decode. 2.8 added GQA. Works with HF Transformers static and dynamic KV caches — [PyTorch 2.8+TorchAO blog](https://pytorch.org/blog/pytorch-2-8torchao-unlock-efficient-llm-inference-on-intel-ai-pcs/)
- FlexAttention, SDPA and FlashAttention (SYCL-TLA) are described as fully enabled. "Improve library-based SDPA on oneDNN" and "use SYCL-TLA alongside oneDNN and Triton for GEMM, SDPA" are 2026 roadmap items — [dev-discuss 2026 H1](https://dev-discuss.pytorch.org/t/intel-gpu-cpu-enabling-status-and-feature-plan-2026-h1-update/3320)
- SYCL-TLA flash kernels are AOT-built only for `-device pvc,bmg`; other architectures get "does not support flash attention" — [pytorch#201409](https://github.com/pytorch/pytorch/pull/201409)
- PVC is excluded due to accuracy issues — [torch-xpu-ops #4094](https://github.com/intel/torch-xpu-ops/issues/4094)
- sycl-tla v0.9 moved flash attention to AOT compilation, requiring Xe2+. Forward pass on BMG regressed:
  - v0.9.2 was up to 8x slower than v0.8 at d64 on Arc Pro B60.
  - v0.9.3 fixed it.
  - Separately, a performance fix was needed for DPC++ 2026.1.
  - Source: [huggingface/kernels #921](https://github.com/huggingface/kernels/pull/921)
- Supported head dims: 64, 96, 128, 192 per a PyTorch PR, or 64/96/128/192/256 per SGLang docs (fp16/bf16). The sources conflict on 256 — [pytorch#201409](https://github.com/pytorch/pytorch/pull/201409); [SGLang attention backends](https://lmsysorg.mintlify.app/docs/sglang-diffusion/attention_backends)
- Arc Pro B70 (Xe2) bf16 benchmarks, median ms, A = oneDNN-overrideable-first, B = flash-first:

  | Case | A | B |
  |---|---|---|
  | micro d64 | 0.056 | 0.049 |
  | prefill s2048 d64 | 0.245 | 0.190 |
  | prefill s2048 d128 | 0.764 | 0.715 |
  | prefill s4096 d128 | 3.181 | 3.302 |
  | decode kv2048 d64 | 0.098 | 0.059 |
  | decode kv2048 d128 | 0.057 | 0.126 |
  | decode kv4096 d128 | 0.091 | 0.227 |

  - The PR that would prefer flash when d ≤ 64 was closed unmerged (stale, Oct 2026), so the default stays overrideable-first.
  - Source: [pytorch#191430](https://github.com/pytorch/pytorch/pull/191430)
- Arc A770: no fused attention for training (flash targets other architectures; the "efficient" backend equals math). Inference gets a fused oneDNN graph-API path that is selected automatically — [arc-a770-pytorch-notes](https://github.com/BarzinL/arc-a770-pytorch-notes)
- FlexAttention on XPU is supported in forward and backward, including max-autotune — [FlexAttention on Intel GPUs blog](https://pytorch.org/blog/pytorch-2-9-flexattention-optimization-practice-on-intel-gpus/)

### Inferences
- **For VLAs:**
  - Qwen/SmolLM/Gemma-style backbones have head_dim 64–128, which is fine.
  - PaliGemma/Gemma-2B (pi0-style) uses head_dim 256, which may fall off flash and onto oneDNN or math. Check this.
  - Prefix-LM masks (image+text bidirectional, action causal) should be passed as boolean `attn_mask` or via FlexAttention `block_mask`. Arbitrary masks may disable the flash path.
- **For ACT:** at short sequence lengths (a few hundred tokens), the default oneDNN path is adequate. Force or compare backends with `torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION])` vs `[SDPBackend.OVERRIDEABLE]` vs `[SDPBackend.MATH]` and time each.
- On Lunar Lake and Panther Lake iGPUs (Xe2-LPG/Xe3), the SYCL-TLA flash kernels are probably not built (AOT targets are pvc and bmg only), so oneDNN is the path.

### Gaps
- No paged or varlen SDPA in core PyTorch XPU was documented. vLLM/SGLang XPU backends have their own kernels, which were not researched here.
- Whether head_dim 256 is supported by SYCL-TLA flash is unresolved.

## 5. XMX (systolic) engines: dtypes, ensuring use, verification

### Takeaway
XMX accelerates fp16, bf16, int8 (and int4/int2, plus TF32 on Xe2/Xe3). fp32 matmuls do not use XMX unless TF32 is allowed. FP8 on Battlemage/Xe3 XMX is unconfirmed or likely unsupported natively: it is dequantized and used for storage or bandwidth, not native FP8 DPAS. Running the model in bf16/fp16 is the main way to hit XMX.

### Cited Findings
- Xe2 (Battlemage) XMX supports INT2, INT4, INT8, FP16, BF16 and TF32; INT8 is rated 4096 ops/clock per XMX engine — [TechPowerUp B580 preview](https://www.techpowerup.com/review/intel-arc-b580-battlemage-unboxing-preview/2.html); [Wccftech Xe2](https://wccftech.com/intel-xe2-gpus-50-percent-uplift-new-ray-tracing-cores-lunar-lake-arc-battlemage-discrete/)
- Xe3 (Panther Lake) XMX supports TF32, FP16, BF16, INT8, INT4 and INT2. Intel says Xe3 supports FP8 *dequantization*; the analysis judges that the matrix units probably do not natively support FP8 — [HWCooling Xe3 analysis](https://www.hwcooling.net/en/xe3-whats-new-in-panther-lakes-gpu-architecture-analysis/)
- oneDNN defines f8_e4m3/f8_e5m2 data types and targets Xe2-HPG and Xe3-LPG, but per-platform acceleration is not stated on the data-types page — [oneDNN data types](https://uxlfoundation.github.io/oneDNN/dev_guide_data_types.html)
- Roadmap: "optimize quant/dequant and GEMM kernels for FP8, MXFP8, MXFP4" — [dev-discuss 2026 H1](https://dev-discuss.pytorch.org/t/intel-gpu-cpu-enabling-status-and-feature-plan-2026-h1-update/3320)
- Triton issue #7561 shows FP8 scaled_mm Triton kernels with 2D block load to DPAS paths exist on XPU, with inefficiencies — [#7561](https://github.com/intel/intel-xpu-backend-for-triton/issues/7561)
- Faster `int_mm` with non-contiguous mat2 was added in 2.11 (int8 GEMM path) — [v2.11.0 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.11.0)

### Inferences
- **Practical steps:**
  - `model.to(torch.bfloat16)` (or fp16; Alchemist A-series is often faster in fp16) and inputs in the same dtype.
  - Or `torch.autocast("xpu", dtype=torch.bfloat16)`.
  - If fp32 must stay, `torch.backends.mkldnn.matmul.fp32_precision = "tf32"` / `torch.set_float32_matmul_precision("high")` to allow TF32 on XMX (Xe2+). The exact XPU knob was not verified in sources.
- Verify with `ONEDNN_VERBOSE=1` (or `DNNL_VERBOSE=1`) and look for `xmx`/`dpas` or the `gen9`/`xe_hp` implementation names in the impl string. This is the standard oneDNN mechanism; this research did not confirm the XPU-specific output format.
- Keep GEMM dims multiples of 16/32 so DPAS tiles are used fully. ACT's d_model=512 is fine.

### Gaps
- No authoritative Intel ISA table confirming native FP8 DPAS on Xe2 or Xe3 was found. Treat FP8 as a memory-saving format, not a compute speedup, unless benchmarked.
- XMX on Lunar Lake (Xe2-LPG, Arc 140V) is presumed the same as BMG's dtypes (same Xe2 architecture), but no source checked.

## 6. SYCL / ESIMD custom kernels and C++ extensions

### Takeaway
PyTorch has a first-class `torch.utils.cpp_extension.SyclExtension` (Linux ≥ 2.8, Windows ≥ 2.10) with an official tutorial. Use it only for ops Triton can't express well. For a hackathon, Triton is almost always the faster route to a fused kernel.

### Cited Findings
- **Official "Custom SYCL Operators" tutorial:**
  - `from torch.utils.cpp_extension import SyclExtension, BuildExtension`; `.sycl` source files.
  - Registered with `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL(ns, XPU, m)`.
  - Requires the Intel compiler from Intel Deep Learning Essentials.
  - PyTorch ≥ 2.8 on Linux, ≥ 2.10 on Windows. Covers inference ops only; for autograd and torch.compile compatibility it refers to the general custom-ops tutorial.
  - Source: [tutorial source](https://cocalc.com/github/pytorch/tutorials/blob/main/advanced_source/cpp_custom_ops_sycl.rst); [cpp_extension docs 2.15](https://docs.pytorch.org/docs/2.15/cpp_extension.html)
- Working example in pytorch/extension-cpp (SYCL, PyTorch ≥ 2.8) — [extension-cpp PR #114](https://github.com/pytorch/extension-cpp/pull/114)
- Windows XPU has the SYCL C++ extension enabled — [dev-discuss 2026 H1](https://dev-discuss.pytorch.org/t/intel-gpu-cpu-enabling-status-and-feature-plan-2026-h1-update/3320)
- Hugging Face `kernels` hub ships XPU flash-attention kernels (SYCL-TLA based) — [huggingface/kernels #921](https://github.com/huggingface/kernels/pull/921)

### Inferences
- SYCL is worth it when you need explicit sub-group, SLM or ESIMD control (e.g. a fused small-GEMM + epilogue for the action-expert MLPs) or must wrap a sycl-tla/CUTLASS template. Expect a day or more of build-toolchain friction, which is too slow for hours-long hackathon work unless prebuilt (e.g. HF `kernels` XPU builds).

### Gaps
- No current docs found on ESIMD use from PyTorch extensions.

## 7. IPEX (intel-extension-for-pytorch) status

### Takeaway
**IPEX is end-of-life and STALE.** Active development stopped after 2.8 and EOL was the end of March 2026. Most features are upstreamed into native PyTorch XPU. Do not base a late-2026 plan on `ipex.optimize` or `ipex.llm`; Triton-XPU is also incompatible with IPEX.

### Cited Findings
- Retirement was announced in the 2.8 release notes and community ticket #867. Development was discontinued after 2.8, EOL is end of March 2026, and the installation guide is labelled "(EOL)" — [IPEX docs](https://intel.github.io/intel-extension-for-pytorch); [install guide (EOL)](https://pytorch-extension.intel.com/); [issue #867](https://github.com/intel/intel-extension-for-pytorch/issues/867)
- DeepSpeed removed the IPEX dependency: "XPU device supported by PyTorch 2.8+" — [DeepSpeed commit](https://app.semanticdiff.com/gh/microsoft/DeepSpeed/commit/d8e15da43f9745712d9e935b9266243e29da9f80)
- Intel's Triton backend is "not compatible with Intel Extension for PyTorch" — [Triton XPU README](https://github.com/intel/intel-xpu-backend-for-triton)
- BigDL (the ipex-llm parent) development has ended (secondary source) — [Complete AI Training](https://completeaitraining.com/news/intel-ends-development-of-bigdl-open-source-ai-project/)

### Inferences
- The replacements are native `torch.xpu`, `torch.compile`, torchao for quantization, and oneDNN SDPA. ipex-llm low-bit kernels are legacy; avoid them unless a team already has a working pinned environment.

### Gaps
- Final archived state of the ipex-llm repo was not verified directly.

## 8. Quantisation on XPU (torchao) at batch size 1

### Takeaway
torchao int4 weight-only (A16W4 via oneDNN) is the proven batch-size-1 win on XPU: 1.56–2.14x decode speedup and 65%+ memory cut for 3–4B LLMs on Arc B580. The current API is `Int4WeightOnlyConfig(group_size=128, int4_packing_format="plain_int32")`; the older `Int4XPULayout` and string-config forms are deprecated or removed. Int8, FP8 and MXFP are on the roadmap but have no published batch-size-1 numbers.

### Cited Findings
- WOQ-INT4 decode speedup over fp16 on Arc B580 (PyTorch 2.8):

  | Model | Decode speedup | Memory |
  |---|---|---|
  | Llama-3.2-3B | 1.62x | 5.98 → 2.10 GB |
  | Phi-3-mini | 2.14x | 7.11 → 2.13 GB |
  | Qwen3-4B | 1.56x | 7.49 → 2.49 GB |

  - Prefill matches fp16, so there is no compute-bound gain.
  - Only the decoder Linear layers are quantized.
  - Call `unwrap_tensor_subclass_parameters(model)` (from `torch._functorch._aot_autograd.subclass_parametrization`) before `torch.compile` to reduce tensor-subclass host overhead.
  - Supports AWQ, RTN and GPTQ.
  - Test hardware also included the Core Ultra 9 288V (Arc 140V, Lunar Lake).
  - Source: [PyTorch 2.8+TorchAO blog](https://pytorch.org/blog/pytorch-2-8torchao-unlock-efficient-llm-inference-on-intel-ai-pcs/)
- The PyTorch 2.10 guide (Core Ultra Series 3) uses torchao 0.16 nightly from the XPU index with `Int4WeightOnlyConfig(group_size=128, int4_packing_format="plain_int32")` — [PyTorch 2.10 + torchao blog](https://pytorch.org/blog/pytorch-2-10torchao/)
- HF Transformers requires torchao ≥ 0.15 for the XPU int4 example and has removed the string API `TorchAoConfig("int4_weight_only")`; use AOBaseConfig objects — [HF torchao docs](https://huggingface.co/docs/transformers/en/quantization/torchao)
- Diffusers notes that XPU requires `int4_packing_format="plain_int32"` — [diffusers PR #13530](https://github.com/huggingface/diffusers/pull/13530)
- FP8, FP16, BF16, INT8 and INT4 (WOQ) are claimed supported across the stack. 2026 plans: good INT4/INT8/FP8/MXFP8/MXFP4 performance in torchao; MXFP4 production-ready with fast GPTQ — [dev-discuss 2026 H1](https://dev-discuss.pytorch.org/t/intel-gpu-cpu-enabling-status-and-feature-plan-2026-h1-update/3320)

### Inferences
- **For a VLA (0.5–3B backbone):**
  - Int4 WOQ on the VLM backbone's Linears helps the memory-bound parts: the prefix pass at batch size 1 with a short token count, and especially iGPUs with shared LPDDR5X.
  - The action expert (small, called N times per chunk) is likely launch-bound. Quantizing it gives little benefit; graph capture helps more.
- **For ACT (about 80M parameters):** quantization is unlikely to help latency. The weights fit in cache and memory budgets, and the dequant overhead may hurt. Use bf16/fp16 plus compile plus graph capture.
- Code sketch:
  ```python
  from torchao.quantization import quantize_, Int4WeightOnlyConfig
  quantize_(model.vlm, Int4WeightOnlyConfig(group_size=128, int4_packing_format="plain_int32"))
  ```
  Then call `unwrap_tensor_subclass_parameters`, then `torch.compile`.

### Gaps
- No published torchao int8 dynamic (W8A8) or FP8 latency results on XPU at batch size 1.
- No VLA- or ViT-specific quantization numbers on Intel GPUs.

## 9. Launch overhead, host bottlenecks, runtime env vars, profiling

### Takeaway
Launch overhead on Intel GPUs is significant: about 14 µs per launch by default on an A770 under i915. Enabling Level Zero direct submission cut this to about 3 µs and gave 1.5x on llama.cpp SYCL decode. Combined with XPUGraph, the xe kernel driver, and a pooled USM allocator, these are the cheap host-side levers.

### Cited Findings
- **Direct submission** (off by default on A770): `NEOReadDebugKeys=1 EnableDirectSubmission=1`; `PrintDebugSettings=1` confirms it was read.

  | Measure | Default | Enabled |
  |---|---|---|
  | Empty-kernel host launch | 14.1–14.6 µs | 2.5–3.3 µs |
  | Empty kernel on card | 2.3 µs | 0.6–0.7 µs |
  | 4,096-element add | 4.6 µs | 1.1 µs |
  | Recurrent eager training step | 2.64–2.71 s | 1.04–1.07 s |
  | llama.cpp SYCL generation | 26.9 tok/s | 40.2 tok/s |

  - Losses matched and idle power was unchanged.
  - Source: [arc-a770-pytorch-notes](https://github.com/BarzinL/arc-a770-pytorch-notes)
- **xe driver instead of i915 on A770:** kernel params `i915.force_probe=!56a0 xe.force_probe=56a0 xe.probe_display=0`. Host launch went from 4.0–4.1 µs to 2.8–3.1 µs, giving 5–10% on launch-bound work. Only a few hours of testing — [arc-a770-pytorch-notes](https://github.com/BarzinL/arc-a770-pytorch-notes)
- **Allocator:**
  - Each held device allocation adds about 1.1–1.6 µs to every eager step.
  - `PYTORCH_ALLOC_CONF=expandable_segments:True` did not help.
  - Fix: `SYCL_UR_USE_LEVEL_ZERO_V2=1 UR_L0_USM_ALLOCATOR="1;device:64M,4,256M"`. Format: `EnableBuffers;[MaxPoolSize;]device:MaxPoolableSize,Capacity,SlabMinSize`; `UR_L0_USM_ALLOCATOR_TRACE=1` prints the parsed values.
  - Recurrent training step went from 3.6 s to 1.1 s. Transformers gained only 1–3%.
  - Source: [arc-a770-pytorch-notes](https://github.com/BarzinL/arc-a770-pytorch-notes)
- **Profiling and memory tools:**
  - The Kineto profiler is enabled for XPU, including on Windows — [dev-discuss 2026 H1](https://dev-discuss.pytorch.org/t/intel-gpu-cpu-enabling-status-and-feature-plan-2026-h1-update/3320)
  - 2.11 added `torch.xpu.memory._record_memory_history`, `_dump_snapshot` and memory_snapshot — [v2.11.0 notes](https://github.com/pytorch/pytorch/releases/tag/v2.11.0)
- A late-2026 nightly fix vectorized fp32↔bf16 casts, raising bandwidth from about 200 to about 440 GB/s on the A770 (torch-xpu-ops#5806). Avoid strided-source copies, which run at half bandwidth — [arc-a770-pytorch-notes](https://github.com/BarzinL/arc-a770-pytorch-notes)

### Inferences
- **Priority for a batch-size-1 policy loop:**
  1. bf16/fp16 so matmuls run on XMX.
  2. torch.compile to cut the kernel count.
  3. XPUGraph capture of the whole inference step or denoising loop to cut per-launch cost.
  4. If not captured, direct submission and the xe driver.
  5. Pooled allocator.
  6. Pre-allocate and reuse static I/O tensors. Avoid `.item()`, `.cpu()` and host syncs inside the loop.
- Profile with `torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU])` and check `self_xpu_time_total` versus CPU time. If XPU time is much less than wall time, the loop is launch-bound and needs graph capture.
- These debug env vars are driver-internal (`NEOReadDebugKeys`) and unsupported. Use them for demos, not as a production guarantee.

### Gaps
- No source confirmed whether immediate command lists are the default per device (A-series, B-series, LNL/PTL, Max) in current releases. The legacy `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` was renamed to UR-era variables (`SYCL_UR_...` / `UR_L0_...`), but this research did not confirm the current names or defaults.
- unitrace and VTune usage with PyTorch XPU was not researched in sources here. Both are Intel's standard GPU kernel tracing tools.
- No sourced measurements of launch overhead on Battlemage, Lunar Lake or Panther Lake, or of whether direct submission is already on by default there.
