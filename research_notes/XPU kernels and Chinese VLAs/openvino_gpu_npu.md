# OpenVINO inference optimisation on Intel GPU/NPU for robot policies (ACT, VLA with VLM + flow-matching expert), as of Nov 2026

Scope note: these notes cover only OpenVINO. They do not deep-dive PyTorch-XPU/Triton. The most important primary source found is Intel's May-2026 paper *Optimizing π0.5 VLA on Intel Core Ultra Series 3 (Panther Lake)*. It is the only published, measured, end-to-end OpenVINO VLA latency with a per-optimisation breakdown. Its full text was extracted from the PDF and is summarised in the first section. All performance numbers come from vendors (Intel) unless stated otherwise.

## 1. Measured VLA numbers and the reference pipeline (π0.5 on Panther Lake, Intel robotics-ai-suite)

### Takeaway
On a Core Ultra X7 358H (Panther Lake, 12 Xe3 cores), Intel cut π0.5-DROID latency from 555 ms to 172 ms per 50-action chunk. The configuration was 3 cameras at 224², 64 language tokens, 10 denoising steps, FP16. The steps were OpenVINO IR, INT8 weight-only compression, batching all cameras into one SigLIP call, GEMM layout tuning, SDPA fusion and GatedMLP fusion. The model kept a 90% LIBERO success rate. Intel's reference split is VE+LLM on the iGPU and the action expert on the NPU, with the per-layer KV cache handed over zero-copy as USM-host tensors.

### Cited Findings
- Test system: Core Ultra X7 358H, 4P+8E+4LP-E cores, 32 GB LPDDR5-8533, 153–154 GB/s peak BW, iGPU 122–123 INT8-dense TOPS, NPU "up to 50 INT8 TOPS", 180 TOPS platform total. Software: Ubuntu 24.04.4, kernel 6.17, OpenCL compute runtime 26.01.36711.4. Tests ran in May 2026 — [Intel paper PDF](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- Workload: π0.5 DROID, 3 cameras at 224×224, 64 language tokens, chunk of 50 actions, 10 denoising steps, BF16/FP16 — [Intel paper](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- Baseline Intel-PyTorch (XPU, eager) latency was **555 ms/chunk** in FP16. torch.compile was explicitly out of scope — [Intel paper §4.1](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- The fully optimised OpenVINO version reached **172 ms/chunk**, a 3.2× speedup, with a 90% average LIBERO success rate (FP16 OV model) — [Intel paper §4, Fig 5](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf). The figure has a per-step bar chart, but the per-step ms values were not extractable from the text.
- INT8 weight-only compression (NNCF, data-free) "reduced the model size by half and improved inference speed by more than 30%" versus the FP16 IR — [Intel paper §4.2](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- Multi-camera batching: the per-camera SigLIP calls were refactored into a single batched SigLIP inference, because the weights are shared. This removes about 50% (2 cams) to 66.7% (3 cams) of redundant weight traffic — [Intel paper §4.3](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- GEMMs account for >99% of ops. The VLM uses 95% of compute and the action expert 4% — [Intel paper §4.4](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- SDPA fusion in the GPU plugin folds the broadcast/reshape ops that precede SDPA into one kernel and does output transposes in registers — [Intel paper §4.5](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- The `FuseGatedMLP` GPU-plugin transformation collapses gate/up/Swish/mul/down into a single oneDNN GatedMLP primitive. Unfused, each small kernel launch in the action expert cost about **104 µs** of dispatch latency. That makes the AE launch-bound, so fusion matters most there — [Intel paper §4.6](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- PERF_COUNT profiling on the Xe3 iGPU found the following. VLM attention/FFN are compute-bound FP16 GEMMs. SigLIP is compute-bound. The AE GatedMLP is memory-bandwidth-bound across denoise steps. AE attention projections are dispatch-overhead-bound because of the small token count (50 action tokens) — [Intel paper §4.7](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- PyTorch 2.12 numbers: PTL iGPU XPU at 40 W = 294 ms. Jetson Thor = 287 ms at max power and 408 ms at 40 W. AGX Orin = 1034 ms and 1530 ms. With OpenVINO vs TensorRT at max TDP, Intel reports 1.5× lower latency than Orin and 0.9× of Thor. At 40 W ISO-TDP it reports 2.6× better than Orin and 1.1× better than Thor (ratios only, not absolute ms) — [Intel paper §6](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf); [OEP article page](https://docs.openedgeplatform.intel.com/dev/OEP-articles/publications/optimizing-pi0.5-lva-model.html)
- Heterogeneous split: VE+LE run on the iGPU because they are prefill-like and compute-bound. The AE runs on the NPU because it is decode-like: small repeated matmuls over 10 steps. The per-layer KV cache is the only cross-device handoff. It is allocated once via the GPU plugin context as USM-host remote tensors, bound as the LM output (GPU) and the AE input (NPU), and neither plugin copies it. The paper says this works for any VLA with a separable action head conditioned on backbone KV (the π family), but not for autoregressive action-token VLAs (RT-2 style). No retraining is needed — [Intel paper §5](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- Under RTC, overlapping the AE on the NPU with VE+LE on the iGPU turns the per-refill bottleneck from T_VE+T_LE+T_AE into max(T_VE+T_LE, T_AE). The iGPU and NPU share one memory controller, so each slows down under concurrency, and max() is an upper bound on the savings. On LIBERO-10 (200 episodes), the serial GPU+NPU split matched the monolithic-iGPU success rate. Overlapping the VLM with an env step lowered per-chunk latency at "a small cost in success on the longest-horizon tasks". No ms values are given for the split — [Intel paper §5.1](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- Public code is `open-edge-platform/robotics-ai-suite` `src/pipelines/pi05-rtc-ov`:
  - `convert_ov.py` / `convert_ov_rtc.py`. Flags: `--torch_dir` (default `lerobot/pi05_base`), `--ov_output_dir`, `--dataset_path` (real inputs and stats), `--compress_int8` (NNCF), `--save_fp32` (default FP16 IR), `-c/--camera_num` (default 4; camera count is baked in, i.e. static shapes).
  - `benchmark_pi05_ov_rtc.py --device GPU --chunk_size 75 -n 10 [--run_torch] [--disable_rtc]`
  - `eval_aloha.py --use_ov --ov_device GPU.0` (default model dir `pi05_lerobot_ov_ir_INT8`) `--rtc_enabled --rtc_horizon 45 --fps 50`.
  - RTC models take the extra inputs `prev_chunk_left_over` and `prefix_weights`; pass zeros to disable RTC.
  - The README recommends binding the **xe** kernel driver instead of i915 for iGPU performance, and pinning to P-cores (`taskset -c 0-5`). It reports no latency numbers and does not mention the NPU.
  
  — [GitHub pi05-rtc-ov](https://github.com/open-edge-platform/robotics-ai-suite/tree/main/src/pipelines/pi05-rtc-ov); [OEP docs: Pi0.5 with RTC](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/embodied/sample_pipelines/pi05_with_rtc.html)
- Embodied SDK release notes: release 2026.0 enabled π0.5 on OpenVINO optimised for the iGPU. Known issue: CL_OUT_OF_RESOURCES on Core Ultra 2 with i915; the workaround is rebinding to xe — [OEP release notes](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-suites/robotics-ai-suite/embodied/release-notes.html) (search snippet)
- Intel `physicalai` runtime (openvinotoolkit/physicalai): `InferenceModel(path, backend="openvino", device="GPU")` with `select_action()`/`reset()` and built-in action chunking. Also `RobotRuntime`, `SyncExecution`/`AsyncExecution`, and `physicalai.benchmark.performance.InferenceLatencyBenchmark`. The README publishes no latency numbers — [GitHub physicalai](https://github.com/openvinotoolkit/physicalai)
- Physical AI Studio / OpenVINO Physical AI: a unified `policy.export(path, backend="openvino")`, `policy.to_openvino(...)` and `policy.to_executorch("model.pte", ...)` with an OpenVINO delegate. ACT, SmolVLA and PI0/PI05 sit behind one interface, and training uses `Trainer(accelerator="xpu")` — [PyTorch Conf EU 2026 slides (PDF)](https://hosted-files.sched.co/pytorchconferenceeu2026/fb/Full-Stack%20PyTorch%20Robotics%20VLA_PyTorch_Conf_EU2026.pdf); [Intel Physical AI page](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/physical-ai.html); [OpenVINO Physical AI docs](https://docs.openvino.ai/2026/physical-ai.html)
- A SmolVLA OpenVINO IR exists on HF (`modelapi/smolvla-libero-fp16-ov-catalog`, loaded via `physicalai.InferenceModel`). A community card reports a SmolVLA export size of FP32 1586 MB → FP16 800 MB → INT8 409 MB (NNCF), with no latency — [HF smolvla-libero-fp16-ov-catalog](https://huggingface.co/modelapi/smolvla-libero-fp16-ov-catalog) (via search snippet; community source)

### Inferences
- At 30 Hz with 50-action chunks, the execution window is about 1.67 s, so 172 ms fits easily on Panther Lake even without RTC. RTC plus a GPU/NPU split only matters for shorter execution horizons (e.g. executing about 5 actions = 167 ms) or bigger models.
- The optimisation order that paid off, in rough priority: (1) OpenVINO IR FP16 with static shapes, (2) INT8 weight-only, (3) batching cameras into one vision call, (4) letting GPU-plugin fusions (SDPA, GatedMLP) fire, which may need a recent OpenVINO (2026.x) and model patterns the matcher recognises.
- For an action expert that is launch-bound (about 100 µs per kernel), counting kernels per denoise step is the key metric. Check the fusions with `benchmark_app -pc` / `PERF_COUNT`.

### Gaps
- No per-stage (VE/LE/AE) ms on Panther Lake was found in text, nor any latency for the GPU+NPU split. The numbers exist only as figure bars.
- No published OpenVINO latencies were found for π0.5/SmolVLA/ACT/GR00T on Arc A-series/B-series dGPU or on Lunar Lake.
- I could not confirm how `convert_ov.py` splits the graph (single IR vs VE/LM/AE IRs). The README does not say.
- The paper is a vendor document. The Jetson comparisons are Intel-run, and no independent reproduction was found.

## 2. GPU plugin internals relevant to batch-1 latency

### Takeaway
Use static shapes, an f16 IR or `INFERENCE_PRECISION_HINT=f16`, `PERFORMANCE_HINT=LATENCY` (one stream), `CACHE_DIR` and remote/USM tensors for zero-copy. On XMX-capable GPUs (Alchemist, Battlemage, Lunar Lake Xe2, Panther Lake Xe3), GEMM/conv/GatedMLP go through oneDNN on the systolic (XMX) units automatically. Dynamic shapes are slower and lose fusions.

### Cited Findings
- Static vs dynamic shapes: dynamic shapes on the GPU are "preview", mostly validated on NLP models, with incomplete op coverage. They can be slower than static on dGPU because of runtime shape inference, kernel compilation and reduced fusion. Recommendation: use static shapes, or bounded ranges such as `{ov::Dimension(1,10), ov::Dimension(1,384)}` — [GPU device docs (rst)](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.rst)
- Precision: GPU op precision follows the IR. The docs checklist says FP32 is the default unless the IR is f16-compressed (which is the default output of `ovc`/`convert_model` save). f32, f16 and u8/i8 are supported. Check `ov::device::capabilities` — [GPU device docs](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.rst). Note: other OpenVINO docs state the GPU `inference_precision` hint defaults to f16. The fetched rst is ambiguous, so set `INFERENCE_PRECISION_HINT` explicitly.
- Model caching: `ov::cache_dir` gives full blob caching only for static models. Dynamic models get kernel caching only (`.cl_cache`). Plugin transformations still run on each `compile_model()` — [GPU device docs](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.rst)
- `ov::intel_gpu::hint::queue_throttle` reduces CPU spin-polling. Lower throttle saves CPU but **can increase latency**; it is meant for multi-stream/throughput. `queue_priority`, `host_task_priority` and `enable_loop_unrolling` are also listed. `ov::intel_gpu::enable_sdpa_optimization` controls SDPA. `ov::intel_gpu::disable_winograd_convolution` is also available — [GPU device docs](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.rst)
- Multi-stream: each stream has its own host thread and OpenCL queue. THROUGHPUT enables automatic batching when `optimal_batch_size>1`, which hurts per-request latency — [GPU device docs](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.rst)
- Remote tensor API: `ov::intel_gpu::ocl::ClContext::create_tensor` wraps cl_mem, USM pointers and VA/D3D surfaces. USM host and USM device allocation are supported. Sharing an existing OpenCL queue makes `start_async()` synchronous on submission and disables throughput mode. CPU pointers must be aligned to `ov::intel_gpu::cacheline_size`. The purpose is to "avoid any memory copy overhead when plugging OpenVINO inference into an existing GPU pipeline" — [Remote Tensor API GPU (rst)](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device/remote-tensor-api-gpu-plugin.rst)
- oneDNN: the GPU plugin offloads compute-heavy kernels (conv, matmul, some pooling) to oneDNN on GPUs with a systolic array. It checks hardware at compile time; look for `GPU_HW_MATMUL` in `OPTIMIZATION_CAPABILITIES` via `hello_query_device`. Verify with `ONEDNN_VERBOSE=1` or `benchmark_app -pc` (execType containing ":" such as `jit:ir` is oneDNN). Disabling it (`OV_GPU_DisableOnednn` debug config) is unsupported. For accurate small-layer timing, use `DevicePerformanceTiming` via opencl-intercept-layer, because oneDNN profiling overhead inflates small layers — [OpenVINO blog: oneDNN on dGPU](https://blog.openvino.ai/blog-posts/openvino-is-powered-by-onednn-for-the-best-performance-on-discrete-gpu) (the blog is older, pre-2025)
- Dynamic quantization of activations (for INT8/INT4-weight models on XMX GPUs: Alchemist, Battlemage, LNL, ARL) is on by default since 2025.2:
  - Property `dynamic_quantization_group_size`. The default `UINT64_MAX` means per-token; `0` disables it; 256 gives better accuracy.
  - Env vars: `OV_GPU_ASYM_DYNAMIC_QUANTIZATION`, and `OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD` (default 64 rows).
  - It is **skipped for ≤64 tokens**, and gains are small for short sequences, which are weight-I/O-bound.
  
  — [OpenVINO blog: dynamic quantization on XMX](https://blog.openvino.ai/blog-posts/dynamic-quantization-support-from-gpu-with-xmx)
- INT8-compressed KV cache is enabled by default for LLMs on all GPUs (2025.1 release notes) — [OpenVINO 2025.1 release notes](https://intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-1.html) (via search snippet)

### Inferences
- VLA prefix lengths of about 3×256 image tokens plus 64 text tokens are well above the 64-token threshold, so dynamic quantization applies to the VLM prefill when weights are INT8/INT4. It does **not** apply to a 50-token action-expert pass (≤64 rows), so the AE runs weight-only-dequant GEMMs. That is consistent with Intel finding the AE memory-bandwidth/dispatch-bound.
- For a 30 Hz loop, the recommended compile config is: `{"PERFORMANCE_HINT":"LATENCY","INFERENCE_PRECISION_HINT":"f16","CACHE_DIR":"./ovcache","NUM_STREAMS":"1"}`. Reshape every IR to static shapes before compile (`model.reshape({...})`). Pre-allocate input/output tensors once and reuse them with `set_tensor`. Use `infer()` or `start_async()+wait()` on one InferRequest per subgraph. Leave `GPU_QUEUE_THROTTLE` at default/high because low throttle adds latency.
- The ~100 µs/kernel dispatch figure from Intel suggests that on the iGPU, host-side overhead dominates small-shape subgraphs. Merging the 10 denoise steps into one IR (unrolled loop, or a `Loop` op with `enable_loop_unrolling`) could amortise per-infer overhead, but this is untested.

### Gaps
- No documentation was found on BF16 support in the GPU plugin, or on FP8 (f8e4m3) compute on Battlemage/Xe3. The fetched rst lists only f32/f16/i8/u8.
- No published measurement of per-`infer()` host overhead on Arc/LNL was found.
- The exact semantics of the LATENCY hint on GPU are not described in the fetched doc.

## 3. Custom ops / OpenCL kernel extensions for the GPU plugin

### Takeaway
It is feasible but a legacy path. You write an OpenCL C kernel plus a `CustomLayer` XML (`type="SimpleGPU"`, `version="1"`) and load it via the `CONFIG_FILE` property. It is only worth it for ops OpenVINO cannot express or fuse. For VLA workloads, Intel got its wins from built-in plugin fusions (SDPA, GatedMLP), not from custom kernels.

### Cited Findings
- XML schema: one `CustomLayer` per op (`name` = IR op type, `type="SimpleGPU"`, `version="1"`), with these children:
  - `Kernel` → `Source filename=...`, `Define name/param/type/default`
  - `Buffers` → `Data name arg-index`, `Tensor arg-index type port-index format` (BFYX/BYXF/YXFB/FYXB)
  - `CompilerOptions options=...`
  - `WorkSizes global/local/dim`
  
  Tensor types are float, half and char. Load the file via `core.set_property("GPU", {"CONFIG_FILE": "custom.xml"})` before `compile_model`, or add it to `cldnn_global_custom_kernels.xml` — [Custom GPU operations docs (rst)](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/documentation/openvino-extensibility/custom-gpu-operations.rst)
- The docs say nothing about performance or dynamic-shape support for custom kernels. The op also needs a frontend/opset extension (`ov::op::Op` subclass) so the IR contains it — [Custom GPU operations docs](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/documentation/openvino-extensibility/custom-gpu-operations.rst)

### Inferences
- Custom OpenCL kernels are opaque to the plugin's fusion passes and cannot use oneDNN/XMX unless you write XMX intrinsics yourself (`cl_intel_subgroup_matrix_multiply_accumulate`). They likely break fusion around the op and add a kernel launch. Use them only for things like a fused flow-matching Euler step plus AdaRMSNorm, or an unsupported op, and profile first.
- A more practical "kernel-level" route is to make the model match existing GPU-plugin patterns: SDPA instead of a hand-written softmax(QK^T)V, standard RMSNorm and SwiGLU forms. That way `FuseGatedMLP`, SDPA and RMS fusions fire.

### Gaps
- No 2025–2026 examples were found of custom GPU ops used for VLA/LLM workloads, and no benchmarks of custom-op overhead.

## 4. Stateful models / KV cache reuse across flow-matching denoising steps

### Takeaway
Yes, and Intel's own π0.5 pipeline does it. The backbone (VE+LM) runs once per chunk and emits a per-layer KV cache. The action expert consumes that KV as an **explicit input** in each of the 10 denoising steps; it is not recomputed. On a GPU+NPU split, the KV lives in USM-host memory shared by both plugins. This uses explicit KV tensors as model I/O, not an OpenVINO GenAI-style stateful (ReadValue/Assign) model.

### Cited Findings
- "LE generates a per-layer KV cache that AE utilizes along with its own noise input." Each KV tensor is "allocated once via the GPU plugin context and bound as LM's output on the iGPU and AE's input on the NPU" — [Intel paper §5](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- The AE does cross-attention of 50 action tokens (queries) over the VL KV cache. That cross-attention is memory-bound because of the large KV and small query count — [Intel paper §2.3](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- OpenVINO GenAI on NPU uses static shapes with separate prefill/generate models (`MAX_PROMPT_LEN` default 1024, `MIN_RESPONSE_LEN` default 128, `NPUW_LLM_PREFILL_CHUNK_SIZE`, `PREFILL_HINT` DYNAMIC/STATIC, `GENERATE_HINT` FAST_COMPILE/BEST_PERF) — [GenAI on NPU docs (rst)](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.rst)
- An Intel forum reply on NPU↔GPU memory sharing says no copy is needed if the GPU and NPU share an L0 context/driver with cross-device memory support. An older forum answer said Remote Tensor was GPU-only; that is now outdated — [Intel community thread](https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/OpenVINO-memory-sharing-for-NPU-on-LNL/m-p/1677429); [older thread](https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/How-to-achieve-memory-sharing-for-NPU-with-openvino/m-p/1566984) (via search snippets)

### Inferences
- Recommended export pattern for π0/π0.5/SmolVLA-style models: three static IRs.
  1. Vision: all cameras batched, so B = n_cams.
  2. Prefix/backbone: outputs `past_key_i` and `past_value_i` for each layer.
  3. Expert step: inputs are noisy actions, timestep, state and all KV tensors; output is velocity.
  
  Then run the Euler loop in Python/C++ over 10 steps, binding the same KV tensors once with `set_tensor` (USM/remote tensors on the GPU, so nothing is copied per step). An alternative is to unroll all 10 steps inside one expert IR. That removes 9 infer() round-trips but bakes in the step count.
- A stateful ReadValue/Assign model is a poor fit here. The expert needs read-only KV at every step and must not append to it, whereas the stateful KV in GenAI appends per token. With explicit I/O, the user controls when the KV is refreshed (once per chunk).
- SmolVLA differs: its expert interleaves cross-attention and self-attention with a reduced layer count, but the same "prefix KV as input" split applies.

### Gaps
- The `convert_ov.py` source was not inspected, so I could not confirm whether the public pi05-rtc-ov IR is split into a KV-producing backbone and a per-step expert, or is monolithic.
- No published code was found for the USM-host GPU→NPU KV binding. The paper references a guide, "Heterogeneous Computing with Zero-Copy Between NPU and GPU", which I could not locate.
- The OpenVINO stateful-models doc page returned 404 at the guessed raw path and was not fetched.

## 5. NNCF quantisation (INT8/INT4 weight-only, full INT8, FP8) for VLA components

### Takeaway
INT8 weight-only (`nncf.compress_weights`, default `INT8_ASYM`, data-free) is the safe default for VLA backbones on GPU. Intel measured >30% faster π0.5 at half the size while keeping 90% LIBERO success. INT4 (sym/asym, group size 64/128, AWQ plus scale estimation) mainly helps memory-bound decode and has more accuracy risk on models under ~1B. On the NPU, LLM-style parts should use INT4 symmetric, channel-wise (`group_size=-1`) or group 128. No FP8 or robot-policy accuracy studies for INT4 were found.

### Cited Findings
- Weight compression modes: default INT8 is `INT8_ASYM`. INT4 is mixed INT4 plus a backup INT8 (`ratio`). INT4_SYM has the highest latency gain and the highest accuracy loss; INT4_ASYM is medium. API calls:
  - `nncf.compress_weights(model, mode=CompressWeightsMode.INT4_ASYM, group_size=64, ratio=1.0, dataset=..., awq=True, scale_estimation=True)`
  - via Optimum: `OVWeightQuantizationConfig(bits=4, quant_method="awq", scale_estimation=True, group_size=64, ratio=1.0)`
  
  Models under 1B "may lose more accuracy" at low bit — [Weight compression docs (rst)](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/model-optimization-guide/weight-compression.rst)
- LLM perplexity examples: Llama-2-7b FP32 3.28 vs INT8_ASYM 3.29 vs INT4_ASYM g128 r0.8 3.41 (Lambada) — [Weight compression docs](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/model-optimization-guide/weight-compression.rst)
- π0.5: INT8 weight-only gave >30% speedup and half the size with no calibration data. The pipeline flag is `--compress_int8`, and the eval default model is `pi05_lerobot_ov_ir_INT8` — [Intel paper §4.2](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf); [pi05-rtc-ov README](https://github.com/open-edge-platform/robotics-ai-suite/tree/main/src/pipelines/pi05-rtc-ov)
- Intel's roofline projection considered W16A16, W8A16 and W8A8 for VE/LE/AE. Lower precision raises arithmetic intensity and shifts layers toward compute-bound. Only W8A16 (weight-only) was reported as measured — [Intel paper §3.2](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- Accuracy validation used MSE/MAE/cosine similarity of trajectories versus the PyTorch reference, plus LIBERO success rate — [Intel paper §4.8](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- NPU LLM requirements: `--sym --weight-format int4|nf4 --group-size -1|128 --ratio 1.0`. Group 128 is suggested for models up to about 4–5B. NF4 needs Lunar Lake or newer and channel-wise grouping — [GenAI on NPU docs](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.rst)
- Lunar Lake/Arc dynamic activation quantization makes INT8/INT4-weight GEMMs use XMX int8 for prompts over 64 tokens — [OpenVINO blog](https://blog.openvino.ai/blog-posts/dynamic-quantization-support-from-gpu-with-xmx)
- Third-party data point (ipex-llm/Ollama, not OpenVINO, Meteor Lake iGPU): for qwen2.5-7B decode, Q4_K_M was 12.08 tok/s, Q8_0 8.26 and FP16 4.74. This shows how strongly bandwidth limits decode on an iGPU — [nikolasent iGPU benchmark](https://nikolasent.github.io/hardware/deeplearning/2025/02/09/iGPU-Benchmark-VLM.html) (Feb 2025, different runtime)

### Inferences
- For batch-1 VLAs: the VLM prefix (several hundred tokens) is compute-heavy, so INT8 weights plus dynamic quantization help via XMX. The action expert (50 tokens × 10 steps) is weight-bandwidth-bound, so INT8 or INT4 weight-only gives roughly proportional savings. INT4 on the small (~300M) expert is risky for action precision; validate with trajectory MSE before using it.
- Full static INT8 PTQ (`nncf.quantize` with calibration) of SigLIP/DINOv2 is likely the best option for a vision encoder on the NPU (NPU computes in FP16/INT8). On the GPU, weight-only INT8 is lower-risk. No robot-policy accuracy data was found for full INT8 of the vision tower.

### Gaps
- No OpenVINO documentation was found on FP8 (f8e4m3/f8e5m2) execution on Battlemage/Xe3, and no NNCF FP8 speed results on GPU.
- No published INT4 results for any VLA (π0, SmolVLA, GR00T) on OpenVINO were found.
- No full-INT8 PTQ results for SigLIP/DINOv2 in a robot policy were found.

## 6. NPU mapping and heterogeneous GPU+NPU execution

### Takeaway
The NPU (NPU4 on Lunar Lake at about 48 TOPS; Panther Lake up to 50 TOPS) works best on static-shape, decode-like or small, repeated subgraphs. Intel puts the π0.5 **action expert** on the NPU and the vision encoder plus LLM on the iGPU. HETERO is only partially supported on the NPU, so use a manual split with separate compiled models and zero-copy tensors rather than HETERO/AUTO.

### Cited Findings
- NPU plugin facts:
  - Precisions: F32/F16/U8 (INT8 or mixed FP16-INT8). "Computation precision for the HW is FP16."
  - LATENCY is the default hint, with 1 optimal infer request.
  - Dynamic shapes are preview on NPU40XX+ (Lunar Lake) only. They must be bounded, need the Compiler-In-Plugin, and are validated only on vision models.
  - Compiler-In-Plugin is preferred from 2026.1 (`ov::intel_npu::compiler_type`).
  - `ov::intel_npu::turbo` (`NPU_TURBO`) raises clocks but is "not recommended for sustained workloads".
  - Caching: `ov::cache_dir` (bypasses UMD cache).
  - "HETERO is only partially supported, for certain models"; AUTO is supported.
  
  — [NPU device docs (rst)](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst)
- VLMs on the NPU via GenAI `VLMPipeline` (examples: Gemma-3-4B-it, Phi-4-multimodal). NPU properties go under `{"DEVICE_PROPERTIES":{"NPU":{"MAX_PROMPT_LEN":2048,"MIN_RESPONSE_LEN":512}}}`. AoT blob export uses `EXPORT_BLOB`/`BLOB_PATH`, with `CACHE_MODE` OPTIMIZE_SPEED vs OPTIMIZE_SIZE — [GenAI on NPU docs](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.rst)
- Intel's rationale: VE and LE are compute-bound and GEMM-heavy, like LLM prefill, which suits the iGPU. The AE is small matmuls repeated 10 times with a moderate bandwidth-to-compute ratio, which suits the NPU and is like LLM decode. Running the AE on the NPU "saves power and frees up the iGPU". Splitting adds minor overhead. Concurrent GPU+NPU share one memory controller — [Intel paper §5](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- A competing vendor (Quadric) argues that NPU-plus-CPU setups bounce non-matmul ops back to the CPU every layer. This is a marketing claim and not specific to OpenVINO — [Quadric press release](https://www.ai-infra-summit.com/press-releases/quadric-runs-pi-0-5-6-10-more-efficiently-than-nvidia-s-jetson-thor) (via search snippet)

### Inferences
- A practical split for 30 Hz with a static camera count: SigLIP/DINOv2 (static, batched cameras) could also go on the NPU, with full INT8 PTQ. That would overlap vision for frame t+1 with LLM prefill on the GPU, but it adds a serial hop to the critical path. Intel chose VE on the GPU. For the lowest single-chunk latency, everything on the iGPU is simplest and fastest. The split helps throughput/overlap under RTC and power, not single-shot latency.
- On Arc dGPU desktops there is no shared memory with the NPU (if present). The NPU handoff then crosses PCIe, so the split is unattractive there.

### Gaps
- No measured NPU-only latency was found for any VLA component (SigLIP, Gemma-300M expert) on Lunar Lake or Panther Lake.
- No source showed the exact C++/Python call for allocating a USM-host tensor from the GPU context and binding it to an NPU infer request.

## 7. SigLIP/DINOv2 vision encoder acceleration

### Takeaway
The main measured lever is batching all cameras into one encoder call with static B = n_cams. Beyond that, the GPU plugin's SDPA fusion handles attention if the exported graph uses `scaled_dot_product_attention`.

### Cited Findings
- Batching cameras into a single SigLIP call removes 50% (2 cams) or 66.7% (3 cams) of redundant weight traffic — [Intel paper §4.3](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- SigLIP is compute-bound on the Xe3 iGPU. Vision-token count scales linearly with cameras and resolution, and higher token counts shift the work toward compute-bound — [Intel paper §2, §4.7](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- SDPA fusion via the GPU plugin, plus the property `ov::intel_gpu::hint::enable_sdpa_optimization` — [Intel paper §4.5](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf); [GPU device docs](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.rst)

### Inferences
- Export the vision tower with `torch.nn.functional.scaled_dot_product_attention` (not eager attention) so the OpenVINO frontend emits the SDPA op. Confirm in `benchmark_app -pc` that an `sdpa` kernel appears instead of MatMul/Softmax/MatMul.

### Gaps
- No OpenVINO latency numbers for SigLIP-So400m or DINOv2 at batch 1–3 on Arc or Lunar Lake were found.

## 8. Benchmarks and benchmark_app usage

### Takeaway
The only solid OpenVINO VLA numbers are Intel's π0.5 results on Panther Lake (555 → 172 ms; PyTorch-XPU 294 ms at 40 W). I found no published OpenVINO latencies for SmolVLA, ACT, GR00T, Qwen2.5-VL-3B or InternVL on Arc or Lunar Lake.

### Cited Findings
- π0.5 numbers: see section 1 — [Intel paper](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- Profiling: `benchmark_app -pc` shows per-layer execType (oneDNN kernels contain ":"); `ONEDNN_VERBOSE=1`; `DevicePerformanceTiming` with opencl-intercept-layer — [OpenVINO oneDNN blog](https://blog.openvino.ai/blog-posts/openvino-is-powered-by-onednn-for-the-best-performance-on-discrete-gpu). Intel used the `PERF_COUNT` API for the per-layer breakdown — [Intel paper §4.7](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- Repo benchmark: `python benchmark_pi05_ov_rtc.py --device GPU --chunk_size 75 -n 10 [--run_torch]`, and physicalai's `InferenceLatencyBenchmark` — [pi05-rtc-ov](https://github.com/open-edge-platform/robotics-ai-suite/tree/main/src/pipelines/pi05-rtc-ov); [physicalai](https://github.com/openvinotoolkit/physicalai)
- Intel's Qwen3-8B demo on Lunar Lake/B580 reports TTFT of about 34 s with optimisations vs about 1 min 28 s without. This is a demo of a long-prompt case and not comparable to VLA prefill — [Intel Qwen3 article](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-qwen3-large-language-models.html) (via search snippet)
- HF blog SmolVLM2-256M on OpenVINO (Intel CPU): TTFT 0.420 s, TPOT 0.021 s — [HF blog openvino-vlm](https://huggingface.co/blog/openvino-vlm) (CPU only, via search snippet)

### Inferences
- Typical benchmark_app invocation for a static VLA subgraph: `benchmark_app -m expert.xml -d GPU -hint latency -shape "[...]" -data_shape ... -infer_precision f16 -niter 200 -pc -cdir ./ovcache`. For NPU, use `-d NPU`. Report median and p99, not the mean, for control loops.

### Gaps
- No OpenVINO latencies were found for SmolVLA, ACT, GR00T N1.x, pi0 (non-0.5), Qwen2.5-VL-3B or InternVL on Arc A770/B580 or Lunar Lake 258V. The physicalai/Physical AI Studio repos expose benchmarking tools but publish no numbers.

## 9. OpenVINO vs PyTorch-XPU eager vs torch.compile at batch 1

### Takeaway
On Panther Lake, for π0.5, OpenVINO (172 ms) was about 3.2× faster than Intel-PyTorch XPU eager (555 ms, FP16) in Intel's own pipeline. Intel's paper also reports PyTorch-XPU at 294 ms at 40 W in a different setup, PyTorch 2.12, so the PyTorch baselines are inconsistent. torch.compile was not measured.

### Cited Findings
- Baseline Intel-PyTorch FP16 = 555 ms; OpenVINO optimised = 172 ms (3.2×) — [Intel paper §4](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- PyTorch 2.12 XPU on PTL at 40 W = 294 ms (§6.1). This conflicts with the 555 ms "initial" PyTorch number in §4.1. They probably differ in PyTorch version, power setting or code path; the paper does not reconcile them — [Intel paper](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- "torch.compile() is out of scope of this work" — [Intel paper §4.1](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)

### Inferences
- Why OpenVINO wins at batch 1:
  1. Ahead-of-time static-shape graph compilation with pattern fusions (SDPA, GatedMLP, RMS), which cut kernel launches. Launches cost about 100 µs each on the iGPU, and this dominates the small-token action expert.
  2. INT8 weight-only plus dynamic quantization through oneDNN/XMX.
  3. A lower host overhead per infer than Python eager dispatch.
  
  Against 294 ms, the gain is about 1.7×; against 555 ms, about 3.2×.
- torch.compile on XPU (Inductor/Triton) could close part of the fusion gap. That is for the other researcher; no head-to-head comparison was found.

### Gaps
- No head-to-head OpenVINO vs torch.compile-XPU comparison for any VLA was found.
- No data was found for Arc dGPU, where PCIe and host overhead differ from the iGPU.
