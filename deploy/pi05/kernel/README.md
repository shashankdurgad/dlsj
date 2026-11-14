# π0.5 kernel and runtime study on the XPS (GPU)

This study makes the **same** Studio π0.5 graphs run faster on the Arc B390 iGPU (Xe3). It covers per-kernel
profiling, a GPU-plugin configuration sweep, host overhead, and the GatedMLP fusion through a debug-capable
OpenVINO build. Model-level variants (precision, prompt length, steps) live in `../opt/`. Machine: Dell XPS 16,
Core Ultra X9 388H, Arc B390 (`Intel(R) Graphics [0xb080]`), xe driver, OpenVINO **2026.4.1**
(`2026.4.1-22982-e213a147257`), Studio library env. Every number below was measured on that machine on
13–14 Nov 2026 (XPS clock, PST). Raw JSON and logs are in `~/inference/logs/pi05/kernel/` on the XPS.

| File | What it does |
|---|---|
| `kernel_lib.py` | Pure helpers (stdlib only): parse benchmark_app counters, map kernels to a library (oneDNN/XMX vs OpenVINO OpenCL), attribute kernels to π0.5 sub-graphs from tensor shapes, aggregate, estimate dispatch gaps, summarise an interleaved sweep. Tested by `tests/test_pi05_kernel.py` |
| `profile_kernels.py` | Compiles one IR on the GPU without and with `PERF_COUNT`. Joins `get_profiling_info()` with the compiled runtime graph (implementation, shapes, order) and writes one JSON with every primitive |
| `summarize_kernel.py` | JSON → the Markdown tables below (`profile`, `compare`, `sweep`, `counters`) |
| `sweep.py` | Interleaved config sweep through `../studio_bench.py` (`physicalai.InferenceModel`). Each variant runs in a fresh process with its own `CACHE_DIR`, baseline first in every round, the rest in shuffled order; reports the paired Δ against that round's baseline |
| `host_overhead.py` | `calls`: five ways of calling the same compiled model. `compile`: no cache vs cold vs warm `CACHE_DIR` |
| `build_ov_debugcaps.sh` | Builds OpenVINO 2026.4.1 from source with `ENABLE_DEBUG_CAPS=ON` (GPU+CPU, IR frontend, Python) into `~/inference/ovbuild/install` for `~/inference/.venv-ovdbg` |
| `build_governor.sh` | Runs the build at nice 19 / ionice idle and SIGSTOPs the whole build tree while load > 12 or available RAM < 15 GB |
| `patch_gated_mlp_gelu.py` | Extends the GPU `FuseGatedMLP` pass to Gemma's GeGLU MLP (see §3). Opt-in at runtime |
| `gated_mlp_micro.py` | One synthetic MLP block (expert/VLM shapes; Swish, GeLU-tanh, GeLU-erf; INT8 or f16 weights), fusion off vs on, debug build only |
| `run_kernel_queue.sh` | The queue used on the XPS. Stage A = release profile + sweep + host + accuracy, B = build, C = debug-build profiles + latency, D = GatedMLP accuracy + re-timing + micro. Each stage waits for an idle machine (≥25 GB available, no other benchmark and no live `pi05_control`) and runs under `flock ~/inference/tmp/heavy.lock` at nice 10 |

## 1. Kernel profile: `pi05_junction_int8` on the GPU (release 2026.4.1)

How it was measured: `profile_kernels.py` ran synthetic static-shape inputs (3×224² cameras, 200-token prompt,
chunk 50, 10 steps) with `PERFORMANCE_HINT=LATENCY` and `NUM_STREAMS=1`. Kernel times are GPU event times per
primitive (median of 20 iterations). It was run on a freshly rebooted, idle XPS (load average about 1). The
benchmark_app `-pc` run on the same IR is in `~/inference/logs/pi05/kernel/bapp_int8/`. It counts the same 3,940
kernels, with Σ 167.5 ms and the same ranking (`summarize_kernel.py counters`).

**Headline:** the raw `infer()` takes **168.8 ms** (median, p90 169.1) without profiling. The GPU kernels sum to
**158.1 ms** across **3,940 kernel launches**. Only **10.7 ms (6.4%) is not covered by kernel time**, which works out
to **2.7 µs per launch**. Turning profiling on (`PERF_COUNT`) adds 69 ms of its own overhead, so do not read
latency from a `-pc` run (benchmark_app `-pc` reported 232 ms).

**Per sub-graph** (from tensor shapes, `kernel_lib.subgraph_of`):

| Sub-graph | GPU ms / inference | Share | Kernels | µs / kernel |
|---|---|---|---|---|
| Gemma-2B VLM prefix (968 tokens) | 82.4 | 52.1% | 288 | 286 |
| Action expert, 10 steps × 18 layers | 53.4 | 33.8% | 3,303 | 16 |
| SigLIP vision, 3 cameras batched | 22.1 | 14.0% | 330 | 67 |
| Other (embedding, masks) | 0.2 | 0.1% | 19 | 9 |

Action expert per denoising step: 5.64, 5.42, 5.28, 5.32, 5.32, 5.28, 5.31, 5.29, 5.27, 5.26 ms. That is
**≈ 5.3 ms and 330 kernels per step**.

**Per op family / implementation:**

| Op | Implementation | ms | Share | Kernels | µs / kernel |
|---|---|---|---|---|---|
| FullyConnectedCompressed (INT8 weights) | oneDNN `jit:gemm:any` (XMX) | 109.2 | 69.1% | 1,564 | 69.8 |
| SDPA | OpenVINO `ocl::sdpa::opt` | 29.7 | 18.8% | 224 | 132 |
| DynamicQuantize (per-token INT8 activations) | `dynamic_quantize_gpu_opt` | 9.1 | 5.7% | 891 | 10.2 |
| Concat (expert KV = prefix cache + 50 tokens) | oneDNN `ocl:simple:reusable` | 3.0 | 1.9% | 363 | 8.2 |
| MVN (SigLIP LayerNorm) | `mvn_gpu_bfyx_opt` | 2.3 | 1.5% | 55 | 42 |
| RMS | `rms_gpu_bfyx_opt` | 2.0 | 1.3% | 405 | 5.0 |
| RoPE | `ocl::rope::opt` | 1.8 | 1.1% | 395 | 4.5 |
| everything else (31 kernels) | | 1.0 | 0.6% | | |

By library, oneDNN XMX GEMMs take 109.5 ms (69%), OpenVINO's own OpenCL kernels 45.7 ms (29%: SDPA, RMS, RoPE,
DQ, MVN), and oneDNN non-XMX kernels 3.0 ms (2%: the concats). Gelu, the GeGLU multiply and the residual adds
never appear as kernels: they are fused as post-ops into the oneDNN GEMMs. SDPA, RoPE and RMSNorm are fused
(224 / 395 / 405 instances).

**Sub-graph × op:**

| Sub-graph \| op | ms | Kernels | µs / kernel |
|---|---|---|---|
| VLM \| FC | 63.8 | 121 | 528 |
| expert \| FC | 29.9 | 1,280 | 23 |
| expert \| SDPA | 17.1 | 180 | 95 |
| vision \| FC | 15.6 | 163 | 95 |
| VLM \| SDPA | 9.5 | 17 | 558 |
| VLM \| DynamicQuantize | 6.4 | 69 | 92 |
| vision \| SDPA | 3.1 | 27 | 115 |
| expert \| Concat | 3.0 | 362 | 8 |
| expert \| DynamicQuantize | 1.8 | 740 | 2.5 |

What the profile says:
- **Dispatch overhead is not the bottleneck on this stack.** The ~104 µs per unfused launch in Intel's paper does
  not reproduce here. With the in-order queue and oneDNN on Xe3, the time not covered by kernels is 2.7 µs per
  launch, 10.7 ms in total. Fusing *every* kernel away would recover at most that much.
- **The expert is weight-bandwidth bound, not launch bound.** Each expert layer runs gate and up
  (`[1,50,4096]`, 39 µs each), down/o-proj (`[1,50,1024]`, 27 µs), q (21 µs) and k/v (5 µs each). Each gate or up
  projection reads a 4.19 MB INT8 weight in 39 µs, which is **≈107 GB/s**. That is about 80% of the LPDDR5X-8533
  theoretical peak (≈137 GB/s on a 128-bit bus). The MLP's three FCs (≈105 µs per layer) are therefore close to the
  memory roofline already. A fused GatedMLP still has to read the same 12.6 MB per layer, so on this stack its upside is
  mostly the 2 saved launches per layer (≈5 µs each, ≈2 ms per chunk). See §3 for the measurement.
- **Dynamic INT8 activation quantisation already runs on the 50-token expert.** 740 `DynamicQuantize` kernels
  execute on `[1,50,·]` activations (1.8 ms in total), so the expert FCs get INT8 activations too. This contradicts
  the playbook note that dynamic quantisation is skipped at ≤64 tokens. In 2026.4.1 the ≤64-row rule is a runtime
  *skip* (`src/graph/dynamic_quantize.cpp::should_skip_execution`, gated on `is_runtime_skippable()`), and it does
  not trigger in this static graph. Lowering the threshold to 16 (`OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD`) changed
  nothing (§2).
- The VLM's large GEMMs are compute bound (528 µs average, 968 rows). The levers there are fewer tokens (the
  64-token prompt, a model-level change) or a better GEMM. Neither comes from a runtime config.
- **FP16 export, same method** (`prof_fp16_release.json`): wall 234.7 ms, kernels 218.4 ms in 3,049 launches
  (no DynamicQuantize). VLM 111.7 ms, expert 81.5 ms (6.1 ms/step), vision 25.1 ms. FC is 82% of GPU time. INT8 cuts FC time
  from 179 to 109 ms.

Reproduce:
```bash
PY=~/inference/physical-ai-studio/library/.venv/bin/python; K=~/inference/deploy/pi05/kernel
$PY $K/profile_kernels.py ~/inference/exports/pi05_junction_int8/pi05.xml --out prof_int8.json
python3 $K/summarize_kernel.py profile prof_int8.json
~/inference/physical-ai-studio/library/.venv/bin/benchmark_app -m ~/inference/exports/pi05_junction_int8/pi05.xml \
  -d GPU -hint latency -niter 20 -pc -pcsort sort -report_type detailed_counters -report_folder bapp_int8
```

## 2. GPU plugin configuration sweep (`InferenceModel`, `pi05_junction_int8`)

Method: `sweep.py ~/inference/exports/pi05_junction_int8 --rounds 3 --iters 40`. Each variant runs as a fresh
`studio_bench.py` process (5 warmup calls, then 40 timed `predict_action_chunk` calls), on top of studio_bench's
default `PERFORMANCE_HINT=LATENCY`, `NUM_STREAMS=1` and a `CACHE_DIR` of its own. "Runtime" is the whole
`InferenceModel` call (tokenizer, resize, normalise, model, denormalise) from 640×480 uint8 frames. "OV" is the
OpenVINO call inside it.

**Only round 0 (13 Nov 23:00–23:04, load average 0.9–1.5, freshly rebooted XPS) was clean.** The team's YOLO
spin-and-inspect test started on the XPS during round 1 (load rose to 3–8). Every later run is slower by 20–230 ms in
ways unrelated to the setting: `default` went 174.1 → 174.0 → 201.3 ms across the rounds. The paired Δ that
`summarize_kernel.py sweep` prints is therefore confounded in rounds 1–2, so the table below uses round 0. Two variants
also ran clean at the start of round 1: `dq128` 173.7 ms and `pcores` 173.5 ms, against `default` 174.0 ms.

| Variant | Setting | Runtime median / p90 ms | OV-only median ms | Δ runtime vs default |
|---|---|---|---|---|
| **default** | `PERFORMANCE_HINT=LATENCY`, `NUM_STREAMS=1`, `CACHE_DIR` | **174.1 / 174.6** | 169.0 | — |
| f16 | `INFERENCE_PRECISION_HINT=f16` | 174.5 / 175.1 | 169.4 | +0.4 (none: f16 is already the Xe3 default) |
| dq128 | `DYNAMIC_QUANTIZATION_GROUP_SIZE=128` | 174.7 / 175.4 | 169.6 | +0.6 (none) |
| dq64 | `DYNAMIC_QUANTIZATION_GROUP_SIZE=64` | 178.2 / 178.9 | 173.2 | +4.1 |
| dq32 | `DYNAMIC_QUANTIZATION_GROUP_SIZE=32` | 208.4 / 210.3 | 203.3 | +34.3 |
| dq0 | `DYNAMIC_QUANTIZATION_GROUP_SIZE=0` (dynamic quantisation off) | 205.4 / 206.2 | 200.3 | +31.3 |
| dq_thr16 | env `OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD=16` | 174.5 / 175.4 | 169.4 | +0.4 (none) |
| sdpa_off | `GPU_ENABLE_SDPA_OPTIMIZATION=NO` | 213.1 / 214.2 | 208.1 | +39.0 |
| queue_prio_high | `GPU_QUEUE_PRIORITY=HIGH` | 174.4 / 175.0 | 169.3 | +0.3 (none) |
| host_prio_high | `GPU_HOST_TASK_PRIORITY=HIGH` | 174.6 / 175.4 | 169.5 | +0.5 (none) |
| throttle_high | `GPU_QUEUE_THROTTLE=HIGH` | 174.5 / 174.9 | 169.8 | +0.4 (none) |
| throttle_low | `GPU_QUEUE_THROTTLE=LOW` | 186.8 / 189.8 | 171.6 | +12.7 (mostly host side: runtime − OV is 15 ms, against 5 ms) |
| all_high | queue priority + host priority + throttle all HIGH | 174.3 / 174.8 | 169.7 | +0.2 (none) |
| tput_1stream | `PERFORMANCE_HINT=THROUGHPUT`, `NUM_STREAMS=1` | 175.3 / 176.0 | 170.2 | +1.2 |
| pcores | `taskset -c 0-3` (P-cores only; `ENABLE_CPU_PINNING` is a CPU-plugin property, so pinning the host process is the GPU equivalent) | 175.8 / 179.1 | 170.7 | +1.7 (173.5 in round 1) |

Rounds 1–2 (contended) agree on the direction of every large effect. dq0 measured +31/+82/+73 ms, dq32
+34/+73/+44, sdpa_off +39/+90/+203, and throttle_low +13 to +27. None of the "no change" rows ever beat
`default` by more than 0.3 ms when both ran under the same load.

**Conclusions:**
- **Nothing in the GPU plugin's config space beats the defaults** that `studio_bench.py` and `eval_actions.py`
  already use. The priority and throttle hints, f16, the latency hint and P-core pinning are all within ±2 ms when idle.
- **Keep dynamic quantisation at its default** (per-token, `UINT64_MAX` group size on Xe3,
  `execution_config.cpp`). Turning it off costs +31 ms (+18%), and group sizes ≤64 cost +4 to +34 ms. dq128 is
  bit-identical to the default.
- **Accuracy of the numerics-changing settings** (`eval_actions.py run --limit 100`, fixed noise, first 100 held-out
  frames; `acc_dq_*.json`). Every variant uses its own `CACHE_DIR`:

  | Variant | MAE k10 vs recorded | Δ k10 vs FP16 [95% CI] | dev vs FP16 MAE / max | dev vs INT8-default MAE / max |
  |---|---|---|---|---|
  | INT8 default (300 frames) | 6.665 | +3.9% [+3.2, +4.7] | 1.33 / 26.7 | — |
  | DQ group 0 (off) | 7.075 | +3.1% [+2.2, +4.2] | 1.23 / 23.2 | 0.47 / 8.1 |
  | DQ group 32 | 7.068 | +3.1% [+2.0, +4.3] | 1.29 / 21.6 | 0.46 / 8.4 |
  | DQ group 64 | 7.068 | +3.1% [+2.1, +4.2] | 1.30 / 26.5 | 0.52 / 7.3 |
  | DQ group 128 | 7.084 | +3.3% [+2.1, +4.5] | 1.44 / 26.7 | **0.00 / 0.00** (bit-identical) |
  | `OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD=16` | 7.084 | +3.3% [+2.1, +4.5] | 1.44 / 26.7 | **0.00 / 0.00** |

  (The k10 MAE of the 100-frame subset is higher than the 300-frame figure because the frames differ. Compare
  within the subset.) Dynamic quantisation moves actions by about 0.5° MAE, far below the 9.9° seed floor and below
  INT8-vs-FP16 (1.3°). The faster default is also no less accurate than DQ off.
- **Cache-key pitfall (affects `../opt/` results).** `~/inference/eval/preds/int8_nodq.npz` is **bit-identical** to
  `int8.npz` (max |Δ| = 0.0). Both runs used the shared `~/inference/ov_cache`, and the `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`
  run evidently loaded the default blob. The same setting with a fresh `CACHE_DIR` (`kernel/int8_dq0_fresh.npz`)
  differs by up to 8.1°. So `int8_nodq` and probably `int8_tok64_5step_nodq` did not measure "DQ off". Re-run them with
  a per-config `CACHE_DIR`. Any compile-property or `OV_*` env A/B on this stack needs its own cache directory.
- **Keep the fused SDPA on.** Disabling it costs +39 ms (+22%) and makes the latency noisy.
- **Host overhead:** runtime − OV-only is **5.1 ms** (174.1 − 169.0): the OV tokenizer on CPU, resizing three
  640×480 frames, normalisation and denormalisation. The OV-only call through physicalai's adapter (169.0 ms) is
  within 0.2 ms of a raw `InferRequest.infer()` on pre-set tensors (168.8 ms, `profile_kernels.py`). Pre-allocating
  and reusing input tensors can therefore save at most ~0.2 ms. `host_overhead.py calls` (raw IR, 4 interleaved blocks
  × 15 calls per style) ran during teleop with the source build running (load 15–30), so it is noisy. Medians:
  `compiled_model(dict)` 290.9, `req.infer(dict)` 290.2, `share_inputs/outputs` 289.8, pre-set tensors 284.5,
  `start_async+wait` 263.5 ms. The minima are 228–231 ms for all five, so no call style is reliably faster.
- **CACHE_DIR cold vs warm** (`load_compile_s` of the same processes, each variant with its own cache dir;
  `host_overhead.py compile` was dropped because a source build was running in parallel):
  the first compile into an empty cache took **8.7–12.3 s** (round 0). Loading the cached blob took **2.0–2.8 s** (rounds
  1–2, under load). The deployed runtime should always set `CACHE_DIR`.

## 3. GatedMLP fusion (debug-capable OpenVINO build)

**Result: on OpenVINO 2026.4.1 / Arc B390, GatedMLP makes π0.5 slower and cannot help.** The stock pass never
fires on π0.5. With our GeGLU extension it fires (197 GatedMLP primitives), but oneDNN runs them with its
**reference kernel** (`ocl:ref:any`), so `pi05_junction_int8` goes from **169 to 266 ms**. With INT8-compressed weights
the fused primitive cannot even be created, and the FP16 export fails to compile. Do not use it. Upstream deleted the
op in PR #39202 ("We decided to use binary-mul-inplace instead").

**Build (worked).** OpenVINO 2026.4.1 was built from source on the XPS with `-DENABLE_DEBUG_CAPS=ON` (GPU + CPU
plugins, IR frontend, Python): 2,236 + oneDNN-GPU steps, GCC 15.2, `rc=0`, about 52 min wall (23:13–00:05, including
restarts and governor pauses) at -j4/-j5. It reports `2026.4.1-1-e213a147257`, the same commit as the wheel. **The debug build is as fast as the release
wheel** with the fusion off: raw `infer()` 169.5 vs 168.8 ms, the same 3,940 kernels; through `InferenceModel` 173.4 vs
176.4 ms, measured back to back.

**Profiles** (`profile_kernels.py`, `pi05_junction_int8`, 14 Nov 00:06–00:08, load 3–6 with teleop running):

| Build / setting | Raw `infer()` median ms | Σ GPU kernel ms | Kernels | GatedMLP prims | GatedMLP ms | VLM ms | Expert ms |
|---|---|---|---|---|---|---|---|
| Release wheel | 168.8 | 158.1 | 3,940 | 0 | 0 | 82.4 | 53.4 |
| Debug build, fusion off (default) | 169.5 | 156.7 | 3,940 | 0 | 0 | 81.8 | 53.1 |
| Debug, `OV_GPU_DISABLE_GATED_MLP_FUSION=0` (stock pass) | 169.1 | 157.7 | 3,940 | **0** | 0 | 82.8 | 53.0 |
| Debug, + `OV_GPU_GATED_MLP_GELU=1` (our GeGLU patch) | **265.7** | 244.9 | 3,349 | **197** | 172.4 | 169.2 | 54.1 |

Where the time goes with the patch on:
- All 197 GatedMLP primitives run on oneDNN's **`ocl:ref:any`** (f16), the generic reference implementation: no
  XMX, no tiling.
- 17 are VLM MLPs at **8.8 ms each**. Unfused, those MLPs ran as XMX GEMMs (`jit:gemm:any`) at about 3.6 ms per layer
  (gate+up 2 × 0.90 ms, down ≈0.6 ms, plus dynamic quantisation). The VLM goes from 82 to 169 ms.
- 180 are expert MLPs at **125 µs each**, against ≈107 µs unfused (gate+up 2 × 39 µs, down ≈27 µs plus DQ). The
  expert is flat at 5.3 ms/step because the 2 saved launches per layer are smaller than the slower kernel.
- 180 small `Multiply` kernels (0.47 ms) newly appear: the expert's gating multiplies lost their post-op fusion.
- 591 fewer kernel launches did not help, which matches §1: launch gaps are only ≈2.7 µs each here.

**Synthetic check** (`gated_mlp_micro.py`, one MLP block at expert and VLM shapes, INT8-sym compressed weights like
NNCF's): with the fusion on, creating the primitive fails for GeLU-tanh and GeLU-erf:
`[GPU] Failed to create oneDNN gated_mlp primitive descriptor status=2 (dnnl_invalid_arguments) ... compressed_weights=1
... wg layout=i8:bfyx:1024x4096x1x1 md=s8:ab:4194304x1` (`gated_mlp_onednn.cpp:231`; the 2-D weight memory
descriptor is flattened to `4194304x1`). In the π0.5 graph the pass instead produced uncompressed f16 GatedMLPs,
which is why it compiled. The SwiGLU and f16-weight cases are queued (stage D); the first run had a model-builder bug
(f32 `beta` for an f16 Swish). Its other timings ran concurrently with the team's robot control and are not used.

**FP16 export + GeGLU patch does not compile:**
`[FuseGatedMLP] END: node: opset1::MatMul node_mul_232 (... f16[1,50,4096], opset1::Constant ...[0]:f16[1,1024,4096]) ...
CALLBACK HAS THROWN: Check 'w_down_ps.rank().compatible(2)' failed at plugins/intel_gpu/src/plugin/transformations/op/gated_mlp.cpp:93:
GatedMLP supports rank-2 w_down only.` Studio's FP16 IR keeps 3-D `[1, out, in]` weight constants. The obvious fix
(return `false` from the matcher for rank ≠ 2) would just leave FP16 unfused. Given the INT8 result it was not pursued.

**Latency through `InferenceModel`** (`sweep.py`, debug build except `release`, round 0 only, 14 Nov 00:08–00:10,
load 1.3–2.3; rounds 1–2 were terminated by a teammate taking the GPU):

| Variant | Runtime median / p90 ms | OV-only ms |
|---|---|---|
| debug build, fusion off | 173.4 / 174.0 | 168.8 |
| release wheel (same session) | 176.4 / 177.8 | 171.2 |
| debug, stock pass on | 208.4 / 224.5 (single noisy run; its profile is identical to "off", 169.1 ms raw) | 201.1 |
| debug, GeGLU GatedMLP on | **268.4 / 280.7** | 263.7 |

Re-timing (3 rounds) and the 300-frame action-accuracy check of GatedMLP on vs off are queued as stage D. Stage D
waits until the team's `pi05_control` has stopped using the GPU. See "What's left".

**The knob (2026.4.1 source):**
- `src/plugins/intel_gpu/include/intel_gpu/runtime/options.inl:116`:
  `OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, disable_gated_mlp_fusion, true, "Disable pass which fuses FC+SwiGLU to GatedMLP")`.
  The property name is `GPU_DISABLE_GATED_MLP_FUSION` (`internal_properties.hpp:169`).
- `transformations_pipeline.cpp:544`: `GPU_DEBUG_VALUE_OR(config.get_disable_gated_mlp_fusion(), true)`, so in a build
  without `ENABLE_DEBUG_CAPS` the value is hard-wired to `true`. Line 1543 registers `FuseGatedMLP` only
  `if (device_info.supports_immad && config.get_use_onednn() && !disable_gated_mlp_fusion)`.
- DEBUG-visibility options can be set **only through the environment or the debug config file**, not through
  `compile_model` properties (`plugin_config.hpp` `OptionVisibility::DEBUG`). Env names are `"OV_" + <property
  name>` (`PluginConfig::read_env`, `m_allowed_env_prefix = "OV_"`). Env options are applied at the end of every
  `finalize()`, in release builds too. So with the debug build: **`OV_GPU_DISABLE_GATED_MLP_FUSION=0`**.
- Not in OpenVINO's cache key: env options do not change the blob hash. Use a separate `CACHE_DIR` per setting
  (`sweep.py` does).

**The stock pass cannot fire on π0.5 even with the knob on.** `fuse_gated_mlp.cpp` matches only
`MatMul(x, W_gate) → Swish → Multiply(·, MatMul(x, W_up)) → MatMul(·, W_down)` (SwiGLU: Llama, Qwen). π0.5's
PaliGemma and its action expert are Gemma models with a **GeGLU** MLP. The IR has 224 `Gelu` ops with
`approximation_mode="TANH"` and 0 `Swish`. The oneDNN `gated_mlp` primitive the pass emits already supports GeLU
(`gated_mlp_onednn.cpp::to_dnnl_alg`: `Gelu_Tanh → dnnl_eltwise_gelu_tanh`, `Gelu → gelu_erf`). So
`patch_gated_mlp_gelu.py` widens the pattern to `Swish | Gelu(v7)` and passes the matching activation type. The new
branch is opt-in (`OV_GPU_GATED_MLP_GELU=1`), so the same binary also shows the stock behaviour. `build_ov_debugcaps.sh`
applies the patch before building. This is our own extension, not upstream code. The upstream pass does not match Gemma's
GeGLU, so the GatedMLP step in Intel's 555 → 172 ms progression presumably used a modified pass (not verifiable).

**Build commands** (≈52 min at -j4/-j5 on the XPS during teleop, with oneDNN-GPU built first as its own step):
```bash
cd ~/inference/deploy/pi05/kernel
JOBS=4 setsid nohup bash build_governor.sh > ~/inference/logs/pi05/kernel/build_governor.log 2>&1 < /dev/null &
# build_governor.sh: nice 19 + ionice idle; SIGSTOPs the whole build tree while load > 12 or RAM < 15 GB
# key cmake flags: -DENABLE_DEBUG_CAPS=ON -DENABLE_INTEL_GPU=ON -DENABLE_ONEDNN_FOR_GPU=ON -DENABLE_INTEL_CPU=ON
#   (the tokenizer IR runs on CPU) -DENABLE_INTEL_NPU=OFF, only the IR frontend, -DENABLE_PYTHON=ON -DENABLE_WHEEL=OFF,
#   -DENABLE_TESTS=OFF -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF (GCC 15.2), installs to ~/inference/ovbuild/install
source ~/inference/ovbuild/env.sh        # LD_LIBRARY_PATH + samples_bin
# .venv-ovdbg sees the build via 00_ov_debugcaps.pth, and physicalai/openvino_tokenizers via zz_studio_site.pth
# (read-only use of the Studio env's site-packages; its openvino is shadowed)
```
Then (stage C of `run_kernel_queue.sh`):
```bash
DPY=~/inference/.venv-ovdbg/bin/python; X=~/inference/exports/pi05_junction_int8/pi05.xml
$DPY profile_kernels.py $X --out prof_dbg_off.json                                              # fusion off
OV_GPU_DISABLE_GATED_MLP_FUSION=0 $DPY profile_kernels.py $X --out prof_dbg_stock.json          # stock pass: expect 0 GatedMLP
OV_GPU_DISABLE_GATED_MLP_FUSION=0 OV_GPU_GATED_MLP_GELU=1 $DPY profile_kernels.py $X --out prof_dbg_gelu.json
python3 summarize_kernel.py compare prof_*.json                                                  # "GatedMLP prims" column
SWEEP_CACHE_ROOT=~/inference/ovcache_kernel_dbg $DPY sweep.py ~/inference/exports/pi05_junction_int8 --python $DPY \
  --only default,gm_gelu,gm_stock,release --rounds 3 --iters 40 --out sweep_dbg_int8.jsonl       # 'release' = wheel
OV_GPU_DISABLE_GATED_MLP_FUSION=0 OV_GPU_GATED_MLP_GELU=1 $DPY ../opt/eval_actions.py run ~/inference/exports/pi05_junction_int8 \
  --cache ~/inference/eval/pi05_val300.npz --out ~/inference/eval/preds/kernel/int8_dbg_gelu.npz \
  --config CACHE_DIR=~/inference/ovcache_kernel_dbg/eval_gelu
~/inference/physical-ai-studio/library/.venv/bin/python ../opt/eval_actions.py report --cache ~/inference/eval/pi05_val300.npz \
  --ref ~/inference/eval/preds/int8.npz ~/inference/eval/preds/kernel/int8_dbg_*.npz
```

## 4. Splitting the graph (VLM prefix once, then the expert loop)

The Studio export does not expose this, and the latency case for it is weak:
- The runtime has a single runner, `physicalai.inference.runners.single_pass.SinglePass`, so the manifest always
  points at one `pi05.xml`.
- Inside that one static graph the prefix is already computed **once**. `Pi05Model.sample_actions` (Studio
  `policies/pi05/model.py`) runs the prefix with `use_cache=True`, then unrolls the 10 `denoise_step` calls against
  that KV cache. The profile confirms it: 288 VLM kernels per inference, and the expert's 360 KV concats (18 layers × 10
  steps × K/V) join the cached `[1,1,968,256]` prefix K/V with 50 new tokens.
- A split would therefore only help to (a) run the expert loop on another device (NPU; Nota measured that slower,
  330 vs 250 ms, see `../README.md`), (b) pipeline the next chunk's prefix with the current expert loop, or (c) reuse a
  prefix across chunks (invalid: the observation changes every chunk). Future work: export `prefix(images, prompt,
  state) → 18×(K,V)` and `expert(KV, noise) → actions` as two IRs, with the KV kept as GPU remote tensors.

## 5. Recommended runtime configuration

**Keep the configuration that `studio_bench.py` and `eval_actions.py` already pass to `InferenceModel`:**
```python
InferenceModel(export_dir, backend="openvino", device="GPU",
               PERFORMANCE_HINT="LATENCY", NUM_STREAMS="1", CACHE_DIR="/home/intel/inference/ov_cache")
```
- Do **not** set `DYNAMIC_QUANTIZATION_GROUP_SIZE` (the per-token default is the fastest; 0 or 32 cost 31–34 ms),
  `GPU_ENABLE_SDPA_OPTIMIZATION=NO` (+39 ms) or `GPU_QUEUE_THROTTLE=LOW` (+13 ms). `INFERENCE_PRECISION_HINT=f16` is
  harmless and documents intent, but it is already the default on Xe3. The priority and throttle-HIGH hints do nothing.
- `CACHE_DIR` is the one setting that matters at start-up: 8.7–12.3 s cold vs 2.0–2.8 s warm.
- **Measured gain over the 177–188 ms deploy figure from runtime configuration: none.** On an idle, freshly rebooted
  XPS this config measured **174.1 ms median / 174.6 ms p90** through `InferenceModel` (169.0 ms OV-only, 168.8 ms raw
  `infer()`). That is 3–14 ms below the earlier 177–188 ms figure, but the difference comes from the idle machine,
  not a setting. The 188 ms run had a recorder running in parallel. The same config measured 201 ms once the YOLO
  test started. The biggest runtime lever is contention, so keep other GPU/CPU work off the laptop while π0.5 runs.
- Remaining runtime-side levers, in order of measured size: (1) the 5.1 ms of host pre/post work (tokenizer and
  image resize; could move into the graph with `ov.preprocess.PrePostProcessor` and a cached tokenisation of the
  fixed instruction strings); (2) the 10.7 ms of launch gaps (only fusion/graph-level reductions can remove them);
  (3) GatedMLP via the debug build: measured **+96 ms slower** (§3). Do not use it.

## What failed / caveats

- **The XPS was powered off twice during this work** (boots ended 22:49:01 and 22:54:53 PST with a clean
  `systemd-poweroff`). The coordinator says the team rebooted it. Killed: the other agent's `bench_variants`, my
  first queue (still waiting) and the FP16 profile (re-run afterwards). After that, everything waits for ≥25 GB
  available and runs under `flock`.
- **Paused by the coordinator** for the YOLO test after the config sweep (13 Nov 23:10), then resumed **during
  teleop/recording**. Everything from 23:13 onwards (host-call modes, DQ accuracy, the build, all debug-build numbers)
  was taken on a shared machine: load 3–30, `pi05_control` and the recorder running at times. Treat those latencies
  as noisy. The §1 profile and the §2 round-0 sweep are from the quiet window.
- **Stage C was terminated** at 00:10 (`Terminated flock ... heavy.lock`) when a teammate started the live
  `dlsj.deploy.pi05_control` on the GPU. Lost: debug sweep rounds 1–2 and the 300-frame GatedMLP accuracy runs. They
  are re-queued as stage D, which waits for `pi05_control` to exit:
  `setsid nohup bash run_kernel_queue.sh D > ~/inference/logs/pi05/kernel/queue_D.log 2>&1 < /dev/null &`.
- **First build attempt:** oneDNN-for-GPU is an ExternalProject whose inner `cmake --build` ran ~11 compilers. That
  ignored `-j`, and because ninja gives every job its own process group, a group SIGSTOP did not reach it either
  (load 30). Fixed: `build_ov_debugcaps.sh` builds `onednn_gpu_build` first with `CMAKE_BUILD_PARALLEL_LEVEL=$JOBS`, and
  `build_governor.sh` signals the whole process tree.
- `host_overhead.py compile` was dropped (a parallel build would distort compile times); cold/warm numbers come from the
  sweep's `load_compile_s`. `benchmark_app` (C++) was not built (samples are not compiled by the install); per-kernel
  data for the debug build comes from `get_profiling_info()` via `profile_kernels.py`, the same counters `-pc` prints.
- Sweep rounds 1–2 were contaminated by the YOLO test (see §2), so only round 0 is quoted. Each round-0 cell is one
  40-iteration run. Re-run `sweep.py --rounds 3` on an idle machine before quoting fine (<2 ms) differences.
- First run of benchmark_app: `FileNotFoundError: .../bapp_int8/benchmark_report.csv` (benchmark_app does not create
  `-report_folder`). Fixed in the queue with `mkdir -p`. The Python benchmark_app's `-pc` latency (232 ms) includes
  the profiling overhead, so take latency from non-profiled runs.
- Sub-graph attribution is from tensor shapes (the ONNX-exported op names are anonymous). Spot-checked: all 180 expert
  SDPA (`[1,50,8,256]`), 27 SigLIP SDPA (`[3,256,16,72]`), 17 Gemma SDPA (`[1,968,8,256]`; the 18th prefix layer only
  produces K/V) and 360 expert KV concats (`[1,1,1018,256]`) are attributed correctly. 19 tiny kernels (0.2 ms) remain
  "other".
