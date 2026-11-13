# Shrink the prefix, then pin everything else

The biggest win on the XPS is the π0.5 vision-plus-language **prefix**, not the denoising loop, the NPU split or kernel tricks. The team's own baseline shows this. At 10 steps PyTorch-XPU takes 467 ms per chunk, and about 296 ms of that is prefix, leaving about 17 ms per denoising step. The plan follows from that. First, move to a fresh Physical AI Studio OpenVINO export on **OpenVINO 2026.4.x**, avoiding 2026.3, which has a known π0.5 regression. Third-party measurements put that export at about **250 ms at 10 steps on the same Arc B390 class GPU**. Then cut `tokenizer_max_length` from Studio's default of **200 to about 64**, add **INT8_SYM** weights, and distil to **one step with Studio's SnapFlow recipe**. Together these should reach roughly **90–130 ms per 50-action chunk** with all three cameras. That figure is an inference and must be measured. Real-Time Chunking then hides what remains. Leave Anomalib on the NPU, where it costs about a millisecond. Do not chase the action-expert-on-NPU split for latency, because a third party measured it **slower than iGPU-only in synchronous mode (330 vs 250 ms)**. Most of the remaining risk is in the system layer. Kernel 7.0 does not keep latency-critical threads on P-cores, so the servo loop needs its own pinned SCHED_FIFO process, and the cameras need MJPEG and a sensible USB layout. For judging, the 20 optimisation points reward evidence more than raw speed. That means a measured device-mapping table, a per-step "staircase" from 467 ms to the final number with p50 and p99, energy per chunk and per task from RAPL, and a contention test showing what happens when NPU, GPU and cameras run together.

## The 296 ms prefix is the target, and four fixes remove most of it

The baseline decomposes cleanly. A 10-step chunk takes 467 ms and a 1-step chunk 313 ms, so each denoising step costs about 17 ms and the vision encoder plus Gemma prefix costs about 296 ms. Reducing steps alone caps out near 1.5×. Intel's profile of π0.5 explains why. **GEMMs are over 99% of operations and the VLM is 95% of compute**, while the action expert is small but dispatch-bound ([Intel π0.5 paper](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)).

**Fix 1: get onto the right exporter and runtime.** Studio exports π0.5 as **one monolithic static-shape IR** (torch → ONNX → OV, FP16 weights, a separate `tokenizer.xml`, runner `SinglePass`). The camera count, `tokenizer_max_length` and the unrolled `num_inference_steps` are all baked in at export time ([Studio `mixin_policy.py`](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/src/physicalai/export/mixin_policy.py); [`pi05/config.py`](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/src/physicalai/policies/pi05/config.py)). All cameras already go through **one batched SigLIP call** at inference ([`pi05/model.py`](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/src/physicalai/policies/pi05/model.py)), so the playbook's "batch the cameras" step comes for free.

The PLAYBOOK's alternative, robotics-ai-suite `convert_ov_rtc.py`, is a poor fit for this machine. It is a patch set on **LeRobot 0.4.3, Python <3.11, `openvino==2026.0`**. It emits the same kind of monolithic IR, with no split and no compile hints, and its INT8 option is `INT8_ASYM` ([pi05-rtc-ov](https://github.com/open-edge-platform/robotics-ai-suite/tree/main/src/pipelines/pi05-rtc-ov)). The XPS environments are Python 3.12 with OV 2026.4, and Studio is the judged hub, so `physicalai export` is the native path. PLAYBOOK §4.3/§4.5 should be updated to say so.

Two version traps matter. First, the physicalai project moved its pin to 2026.4 "because of perf regression in pi05, which is fixed in .4. .3 is excluded" ([physicalai PR #283](https://github.com/openvinotoolkit/physicalai/pull/283)). Second, **in-graph noise is broken on the GPU plugin**. OpenVINO's GPU plugin returned the same random values on every call, so every chunk started from identical noise. On an iGPU task that gave 5–18/20 success, against 16–18/20 once noise became a graph input ([physicalai PR #442](https://github.com/openvinotoolkit/physicalai/pull/442)). Any export made before that change, or made with `use_random_input_noise=False`, must be redone.

What this buys is an estimate, not a measurement on this machine. Nota AI ran the same 968-token, 3-camera FP16 π0.5 through OV 2026.4 on a Core Ultra X7 358H with the same Arc B390 class GPU and got a **250 ms median at 10 steps** ([Nota VTP/Drift card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)). Their 10-step and 1-step means imply about **9 ms per step in OpenVINO**, against our 17 ms in PyTorch, and an OV prefix of roughly 150–165 ms. Both of those are derived figures.

**Fix 2: shrink the padded language tokens.** Studio's default `tokenizer_max_length` is **200**, and the tokenizer is exported with `use_max_padding=True`. Our prefix is therefore 3×256 image tokens plus 200 language tokens, or **968 tokens**. The actual prompt is `"Task: {task}, State: {6 discretised numbers};\nAction: "` ([`pi05/preprocessor.py`](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/src/physicalai/policies/pi05/preprocessor.py)), which is very likely under 48 tokens. Intel's paper used 64. Setting it to about 64 before export removes roughly **14% of prefix tokens**.

This is the cheapest lever on the list, and it is unverified. Padding tokens are masked, so outputs should match as long as nothing is truncated. Tokenise the longest instruction (`"pick up the brick and show it to the inspection camera"` plus state) to confirm it fits. Check trajectory MSE against PyTorch before trusting it. This setting is probably also part of why our PyTorch baseline (467 ms) is slower than Intel's 294 ms PyTorch 2.12 figure ([OEP article](https://docs.openedgeplatform.intel.com/dev/OEP-articles/publications/optimizing-pi0.5-lva-model.html)). Expect judges who know Intel's paper to ask about that gap.

**Fix 3: INT8_SYM weights.** Studio's hook `compress_weights_openvino_int8_sym` applies `nncf.compress_weights(mode=INT8_SYM)` and is Python-only, passed as `post_export_hooks=[...]` ([Studio `hooks.py`](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/src/physicalai/export/hooks.py)). Intel measured INT8 weight-only at **">30%" faster**. Its roofline figure projects a 3-camera VE+LM falling from 92 ms (W16A16) to 86 ms (W8A16) and to 48 ms (W8A8), with the action expert going from 66 to 40 ms ([Intel paper Fig. 4](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)).

The W8A8 case happens automatically. On Xe3 the GPU plugin applies per-token dynamic INT8 activation quantisation only to FCs that have compressed weights and **more than 64 rows**. The roughly 900-token prefix therefore runs W8A8 on the XMX units, while the 50-token expert stays weight-dequant FP16 ([OV `execution_config.cpp` @2026.4.0](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/plugins/intel_gpu/src/runtime/execution_config.cpp)). Leave dynamic quantisation at its default. INT4 should stay off the full stack, because generic full-stack INT4 collapsed π0.5 to 11–16% in earlier evidence ([HoloQ-VLA](https://arxiv.org/pdf/2605.28803)).

**Fix 4: SnapFlow one-step distillation.** Studio's SO-101 recipe `configs/physicalai/pi05/so101/snapflow.yaml` runs two phases. Epochs 0–4 are a normal flow-matching fine-tune. Epochs 5–7 are SnapFlow with `num_inference_steps=1`, which freezes the VLM and trains about 10% of parameters. The exported artifact carries the 1-step path ([snapflow.yaml](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/configs/physicalai/pi05/so101/snapflow.yaml)). On LIBERO the student scored **97.55% against 97.05%** for the 10-step teacher, which is not a significant difference ([snapflow-libero.md](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/docs/explanation/benchmark/snapflow-libero.md)). Nota's 1-step "Drift" lost 1.5 points on LIBERO-10 on the same GPU ([Nota card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)).

At about 9 ms per OV step, distillation saves roughly 80 ms per chunk. Because it only adds epochs to the same Modal run, it costs little extra if planned from the start. Nobody has published SO-101 real-robot results for 1-step π0.5, so keep the 10-step export as a fallback.

| Step (cumulative) | Expected chunk latency, 3 cams | Basis |
|---|---|---|
| PyTorch-XPU bf16, 10 steps (baseline) | **467 ms** (measured) | PLAYBOOK §8.1 |
| Studio OV FP16 export on 2026.4, 10 steps | ~250 ms | Nota, measured on X7 358H / B390 |
| + `tokenizer_max_length` 64 | ~225–235 ms | inferred, −14% prefix tokens |
| + INT8_SYM weights | ~165–190 ms | inferred from Intel ">30%" and Fig. 4; Intel's own 64-token INT8 10-step figure is 172 ms |
| + SnapFlow 1 step | **~90–130 ms** | inferred; Nota's 1-step FP16 is ~175 ms mean at 968 tokens without INT8 |
| Optional: drop `inspect` from policy input | another −20–25% | Intel Fig. 4 projection; needs retraining |
| Optional: VTP 128 tokens/camera | 105–117 ms in Nota's run | costs ~2 LIBERO points; needs a trained sampler |

Two levers are deliberately parked. Dropping a camera or pruning visual tokens changes the policy's inputs, so it means retraining and re-evaluating on our task. Lowering resolution below 224² has no π0.5 evidence at all.

## Three published optimisations do not apply to this setup

The earlier report leaned on Intel's 555 → 172 ms progression. Part of that progression cannot be reproduced from `pip install openvino`. The **FuseGatedMLP pass was debug-only in 2026.2–2026.4**, behind `disable_gated_mlp_fusion=true`, which release builds cannot flip. Master removed it on 2026-10-28 in favour of "binary-mul-inplace", and that replacement has not shipped in a release ([PR #34427](https://github.com/openvinotoolkit/openvino/pull/34427); [PR #39202](https://github.com/openvinotoolkit/openvino/pull/39202)). The playbook's step 6, which says to confirm GatedMLP kernels appear in `-pc`, should be dropped. Intel's 172 ms probably used a custom build.

SDPA fusion is still available. On XMX it stays fused for head_dim ≤ 256, which covers Gemma, and it comes from OV's MatMul–Softmax–MatMul pattern matching, because Studio traces with eager attention ([`transformations_pipeline.cpp`](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp)). Curiously, Nota compiled its Panther Lake prefix with **`GPU_ENABLE_SDPA_OPTIMIZATION=NO`** ([Nota card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)). That is worth one A/B run.

Precision needs care. **bf16 is not a selectable GPU inference precision**; the plugin accepts f16, f32 or dynamic ([`options.inl` @2026.4.0](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/plugins/intel_gpu/include/intel_gpu/runtime/options.inl)). Our bf16 PyTorch baseline will therefore become f16 in OpenVINO. Validate the actions numerically, since Gemma activations can overflow in f16. The plugin keeps RMS inputs and gated residuals in fp32, which should help.

The **GPU+NPU split is the second thing to skip** for latency. Nota built it on Panther Lake with OV 2026.4, putting the prefix on the iGPU and the denoise step on the NPU. The median at 10 steps was **330 ms against 250 ms iGPU-only**, and 129 against 105 ms at one step. The NPU ran the 10-step expert in 144 ms, against roughly 9 ms per step on the iGPU ([Nota card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)). Heavy NPU-specific work brought the expert to **77.8 ms**: symmetric INT4/INT8, a Broadcast rewrite of `repeat_kv`, and precomputed adaRMS tables. `NPU_TURBO=YES` was needed because the NPU downclocks while the iGPU runs ([Nota AE-NPU card](https://huggingface.co/wjwjwjwj/nota-pi05-libero-ae-npu-openvino)).

Intel's own claims for the split are power and freeing the iGPU, not milliseconds. It notes that both devices share one memory controller, so max(T_VE+T_LE, T_AE) is only an upper bound ([Intel paper §5](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)). With SnapFlow there is a single expert step left to hide. The CLAUDE.md stretch goal should be reframed accordingly. If the team builds the split, present it as a **power and iGPU-headroom experiment with a measured, honest result**, not as a latency gain.

The third non-lever is **zero-copy and remote tensors**. These only matter when a KV cache is handed between devices. The monolithic IR takes about 3×150 KB of images per call, which is negligible next to 100+ ms of compute.

The compile configuration should come in through the runtime. physicalai's `OpenVINOAdapter` forwards extra kwargs straight into `compile_model` and sets no LATENCY hint or `CACHE_DIR` by default ([physicalai 0.4.0](https://pypi.org/project/physicalai/)). So pass `InferenceModel(export_dir, device="GPU", PERFORMANCE_HINT="LATENCY", INFERENCE_PRECISION_HINT="f16", CACHE_DIR="/abs/ovcache")`. This is unverified and based on reading the code. Print `EXECUTION_DEVICES` after compiling to prove the model is on the GPU, because the playbook notes that `InferenceModel` auto-picks GPU → NPU → CPU.

RTC is the last model-side setting, and latency feeds straight into it. `RTCExecution` defaults are `execution_horizon=15`, `max_guidance_weight=5.0` and `fps=30`, "validated on π0.5 at 30 fps". It sets `inference_delay = ceil(max_latency × fps)` over a sliding window ([physicalai 0.4.0 `rtc.py`](https://pypi.org/project/physicalai/)). At 467 ms that freezes **15 actions per chunk**, the entire execution horizon. At about 120 ms it freezes 4. Faster inference therefore shows up directly as fresher, smoother motion, and that is the link between model latency and task behaviour to show judges. PR #410 fixed stale-observation replay in RTC ([PR #410](https://github.com/openvinotoolkit/physicalai/pull/410)), so use a runtime that includes it.

## Anomalib already fits on the NPU; the work is shapes, thresholds and copies

The detector is not on the critical path. ResNet18 at 256² takes 0.88 ms on the NPU, against 0.59 ms on the GPU and 12.6 ms on the CPU. The point of putting it on the NPU is placement, not raw speed: keeping it there leaves the iGPU entirely to π0.5. Intel's own heterogeneous guide gives exactly that reasoning for offloading to the NPU when the GPU is busy ([Intel DL Streamer heterogeneous guide](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-libraries/dlstreamer/dev_guide/heterogeneous_systems_guide.html)).

**PatchCore with a resnet18 backbone** is the recommended model. Memory-bank methods were the most robust to viewpoint and illumination in the RAD robustness benchmark, and "light spots caused by uneven illuminations" were the main false-positive source for other families ([arXiv 2406.07176](https://arxiv.org/html/2406.07176v2)). PaDiM fits one Gaussian per patch, so a set of many brick colours inflates its covariance. Use one PaDiM per colour, or prefer PatchCore.

The export path in Anomalib 2.7.0 is `engine.export(model, ExportType.OPENVINO, input_size=(256,256), compression_type=CompressionType.FP16)`. Even with `input_size` set, the batch axis stays dynamic ([`engine.py`](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/engine/engine.py); [export utils](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/components/base/export_mixin/utils.py)). The NPU needs a static shape, so reshape before compiling. This matches the playbook's "Upper bounds are not specified" failure.

FP16 is the right precision, because NPU hardware computes in FP16 anyway ([NPU device doc](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst)). INT8_PTQ warns below 300 images and is untested on distance heads. Apply it to the backbone only, if at all. Use `num_neighbors=1` at inference so the head uses `min` rather than `topk`, whose NPU compile support is unverified.

Three cheap pipeline changes are worth making:

- **Fold the u8/NHWC/BGR conversion into the graph** with `PrePostProcessor`. Then the NumPy cast and transpose that `OpenVINOInferencer` does on every call disappear ([`openvino_inferencer.py`](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/deploy/inferencers/openvino_inferencer.py)).
- **Take the box from the low-resolution map**, not from a full-resolution upsample plus a 33×33 blur ([`patchcore/anomaly_map.py`](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/image/patchcore/anomaly_map.py)).
- **Batch three frames or pipeline them.** The NPU plugin runs a batch as batch-1 command lists, so batching only saves call overhead ([NPU batching doc](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device/batching-on-npu-plugin.rst)). `AsyncInferQueue(cm, 3)` with batch 1 can instead overlap inference with capture.

Compile with the explicit `"NPU"` device, not AUTO. Set an **absolute** `CACHE_DIR` and warm it before the robot loop, because `OpenVINOInferencer` caches to a relative `./openvino_cache`. Intel says precompiled NPU blobs are for development only.

Threshold calibration is the reliability lever. With no anomalous validation images, Anomalib's F1-adaptive threshold falls back to "the highest anomaly score observed in the normal validation images" ([`f1_adaptive_threshold.py`](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/metrics/threshold/f1_adaptive_threshold.py)). That is a poor threshold. Collect 10–30 crossed bricks across all colours. Set the threshold between the 99th-percentile normal score and the minimum defect score. Report the min–max spread of the three frames' scores as the uncertainty band that drives the playbook's re-inspect rule.

Leave **`NPU_TURBO` off** for Anomalib. It raises NPU frequency and memory throughput within the shared package budget ([OV NPU docs](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)). Here that would take power and bandwidth from the iGPU running π0.5. Anomalib only runs while the arm holds still, so it is not latency-critical.

## Kernel 7.0 will not protect the servo loop unless you pin it

The system layer decides p99, not p50. On a sibling Panther Lake part, Linux numbers CPUs 0–3 as P-cores, 4–11 as E-cores and 12–15 as LP E-cores ([LinuxLinks p1](https://www.linuxlinks.com/minisforum-m2-does-linux-schedule-panther-lake-correctly/)). Under load, a foreground thread spent only **29% of its time on P-cores and migrated 827 times**, and one placed on an idle LP E-core ran about 24% slower ([LinuxLinks p9](https://www.linuxlinks.com/minisforum-m2-does-linux-schedule-panther-lake-correctly/9/)). Intel's cluster-scheduling fix for hybrid P/E/LPE parts was queued for **Linux 7.3** ([hwbusters](https://hwbusters.com/news/intel-says-linux-cluster-scheduling-fails-on-its-hybrid-cpus-the-fix-is-queued-for-7-3/)), so Ubuntu 26.04's kernel 7.0 does not have it.

The architecture has to compensate. Put the **servo loop in its own process**, because SCHED_FIFO on a Python process only helps whichever thread holds the GIL. Pin it to one P-core at SCHED_FIFO priority 80. Run cameras and OpenVINO in a second process. Connect the two through a `multiprocessing.shared_memory` action buffer guarded by a sequence counter. Keep IRQs and housekeeping on the LP E-cores. Verify the 388H's numbering with `lscpu` before trusting the 0–3/4–11/12–15 map, which comes from the sibling part.

Power settings matter because CPU, iGPU and NPU share one package budget, and the xe driver reports PL1/PL2/thermal throttle reasons for the GPU ([kernel.org xe GT freq](https://docs.kernel.org/7.0/gpu/xe/xe_gt_freq.html)). Run on AC with the platform profile set to performance; the `dell-pc` driver maps that to Dell's thermal modes when the firmware supports it ([Phoronix](https://www.phoronix.com/news/Dell-PC-Driver-For-Linux-6.11)). Raise the xe GT `min_freq` to `rpa_freq` so each chunk skips the frequency ramp from idle. "If max <= min, then freq_min becomes a fixed frequency request" ([kernel.org](https://docs.kernel.org/7.0/gpu/xe/xe_gt_freq.html)). Log `freq0/throttle/reasons` during every run, because reviewers found the 2026 XPS 16 runs warmer than its predecessor and throttles under some sustained loads ([Notebookcheck](https://www.notebookcheck.net/2026-Dell-XPS-16-runs-noticeably-warmer-than-last-generation-model.1255741.0.html); [HotHardware](https://hothardware.com/reviews/new-dell-xps-16-da16260-review?page=3)). Prefer `rpa` over `rp0` as the floor, and A/B test both, since `rp0` under combined load is likely to hit PL1.

Cameras are where three-device rigs usually break first. Three YUYV 640×480@30 streams need about 147 Mbit/s each, and a USB 2 hub will refuse them with `-28`/ENOSPC ([Arch forum](https://bbs.archlinux.org/viewtopic.php?id=278537)). Several UVC cameras on one root hub need MJPEG ([OpenPnP thread](https://groups.google.com/g/openpnp/c/gSSFehc1cwI)). Put the D415 alone on a USB 3 port, since its colour stream is likely uncompressed. Run the two UVC cameras in MJPEG through the kit hub, and force the 2K webcam to 640×480. Use one capture thread per camera that keeps only the newest frame, with `CAP_PROP_BUFFERSIZE=1`. Stamp each frame with `time.monotonic_ns()` at `grab()`. libjpeg-turbo decode at 640×480 is cheap, so VA-API decode is not worth the complexity.

The servo bus deserves one 1000-iteration timing test. The Dynamixel-derived Feetech SDK lineage assumes a **16 ms default USB latency** ([Robotis FAQ](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/faq/)), and that alone would consume the whole 60 Hz budget. If the adapter is FTDI, set `latency_timer` to 1. If it is CH340/CH343 (`cdc_acm`), that knob does not exist; check Return_Delay_Time instead. The target is a p99 below 3 ms for `sync_read` and below 1 ms for `sync_write`, which is protocol arithmetic rather than a sourced benchmark.

One gap remains unresolved. **No published measurement covers iGPU+NPU+camera contention on Panther Lake.** The only evidence is indirect: the shared memory controller noted in Intel §5, and Nota's NPU downclocking while the iGPU runs. The team has to measure it, and that measurement doubles as presentation material.

## Twenty points reward attribution and evidence, not one fast number

Rubric expectation #4 names five things: conversion, precision, graph optimisation, pre/post-processing efficiency, and device selection. Expectation #5 asks for NPU/GPU/CPU used "according to workload characteristics and operator support" while keeping control responsive. Each needs one artefact. The innovation and clarity points (5) go to the explanation, so attribute every millisecond.

**Measure in two layers.** The first layer is isolated model benchmarks, which give the "kernel ceiling".

For the per-chunk latency claim, run `benchmark_app -m model.xml -d GPU -hint latency -niter 500 -latency_percentile 99 -report_type detailed_counters -report_folder out/pi05 -json_stats -exec_graph_path out/pi05_exec.xml -dump_config out/pi05_cfg.json`. Run the same command on NPU, GPU and CPU for the Anomalib IR to fill a 3×N device table ([benchmark_app docs](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/get-started/learn-openvino/openvino-samples/benchmark-tool.rst)). Also run physicalai's `InferenceLatencyBenchmark(max_iters=..., warmup_iters=..., max_duration=...)` on the whole exported policy ([physicalai README](https://github.com/openvinotoolkit/physicalai)). Use at least 300 iterations after discarding 10–20 warmup runs, prefer 1000 for p99 claims, and repeat each configuration three times.

The second layer is in-loop stage timestamps, recorded with `time.perf_counter_ns()` into a ring buffer and never printed in the loop. Record capture, pre-processing, Anomalib done, decision, VLA done, action sent and servo write return. From these derive photon-to-motion latency, control-period jitter, chunk staleness, task time, tasks/min, and success k/N with a Wilson interval. At N=20, 18/20 has a 95% interval of roughly 70–97%, so state N honestly. These figures are standard practice, not sourced.

**Measure utilisation and power with tools that work on xe.** `intel_gpu_top` **now rejects xe devices** and points users to `gputop` (igt-gpu-tools ≥2.4) ([igt NEWS](https://drm.pages.freedesktop.org/igt-gpu-tools/NEWS/)). GPU busy % comes from the xe PMU's `engine-active-ticks`/`engine-total-ticks` via `perf stat` ([intel-xe patch](https://lists.freedesktop.org/archives/intel-xe/2025-February/068357.html)). NPU busy % comes from `npu_busy_time_us` in intel_vpu sysfs, sampled at about 1 Hz because faster reads perturb job submission. NPU frequency files need **kernel 7.2**, so they are absent on 7.0 ([sysfs-driver-ivpu](https://kernel.org/doc/Documentation/ABI/testing/sysfs-driver-ivpu)).

Energy comes from RAPL `energy_uj` deltas, since there is no instantaneous power file ([powercap docs](https://docs.kernel.org/power/powercap/powercap.html)), or from `turbostat`. No per-IP NPU power counter was found, so the defensible metric is **SoC energy per chunk and per task**. Measure idle package power for 30 s. Report ΔE/N and the incremental (ΔE − P_idle·T)/N. For device attribution, run the same model alone on each device.

**Run the contention matrix.** Measure π0.5 chunk latency alone, with Anomalib looping on the NPU, with the three cameras streaming, and with all of them together. Log `act_freq`, throttle reasons and package watts throughout. No published Panther Lake data exists for this, so a measured table is original evidence. It also justifies the device split better than any assertion. Run the final configuration for at least five minutes and plot latency against time. A flat line answers the thermal-throttling question.

**Present five artefacts.**

1. **Device-mapping diagram.** Draw it in Intel's paper style, annotate each box with measured p50 ms and busy %, and add the 3×N Anomalib device table, which already shows NPU 0.88 / GPU 0.59 / CPU 12.6 ms for the backbone.
2. **Staircase bar chart.** Go from 467 ms through each step in the first table, with p50 bars, p99 whiskers and the Δms labelled on each step. Intel's paper shows the stages but does not attribute the gains to them, so this is a differentiator.
3. **Latency CDF and over-time plot.** Compare before and after.
4. **Efficiency tile row.** Show chunks/s, tasks/min, J/chunk, J/task, mean package W, and success k/N with its confidence interval. Use Intel's ISO-TDP and perf/W vocabulary.
5. **Live overlay on the demo feed.** Show per-stage ms, device labels and busy %, next to a recorded clip of the 467 ms arm.

Keep backup slides with the benchmark_app JSON, the top-10 `-pc` hotspots, and the measurement protocol: AC power, fixed profile, start temperature, versions printed. That way any number can be reproduced on request.

## Ordered plan with commands

The order is by expected gain per hour, with the measurement scaffolding first so every later step lands on the staircase. Items marked ⚠️ are unverified on this machine.

| # | Action | Expected gain | Level |
|---|---|---|---|
| 1 | Build the stage logger, RAPL/xe/NPU telemetry logger and benchmark_app scripts; re-measure the 467 ms baseline under the protocol | enables every claim | measurement |
| 2 | AC, performance profile, EPP performance, xe `min_freq=rpa_freq`, throttle logging | lower p99, fewer GPU ramp stalls | system |
| 3 | Studio `physicalai export --backend openvino` on OV **2026.4.x**, noise as graph input, 3 cameras, compile with LATENCY / f16 / absolute CACHE_DIR | 467 → ~250 ms | OpenVINO |
| 4 | `tokenizer_max_length` ≈ 64 before export (check no truncation, MSE vs PyTorch) ⚠️ | ~−14% prefix tokens | model |
| 5 | `post_export_hooks=[compress_weights_openvino_int8_sym]` | ~−20–30% prefix | model/OpenVINO |
| 6 | Train with `pi05/so101/snapflow.yaml` (phase 2 adds 3 epochs); export the 1-step artifact; keep 10-step as fallback | ~−80 ms; target ~90–130 ms | model |
| 7 | `RTCExecution` defaults (horizon 15, guidance 5.0, 30 fps), runtime with PR #410/#442 | frozen actions 15 → ~4 | concurrency |
| 8 | Servo process pinned to a P-core, SCHED_FIFO 80, gc frozen; inference process off that core; shared-memory action buffer ⚠️ | control jitter, p99 | system |
| 9 | Cameras: D415 on its own USB 3 port, UVC in MJPEG 640×480, buffer 1, latest-frame threads, exposure/WB locked | no ENOSPC, fresher frames | system |
| 10 | Feetech: identify chip, `latency_timer=1` if FTDI, Return_Delay_Time 0, time sync_read/write over 1000 iterations | up to −16 ms per bus round trip | system |
| 11 | Anomalib PatchCore-r18 FP16, static reshape, PPP u8 input, low-res box, explicit NPU, warm cache, calibrated threshold with 10–30 crosses | reliability; ~1 ms detector | model/NPU |
| 12 | A/B `GPU_ENABLE_SDPA_OPTIMIZATION=NO`, `min_freq` rpa vs rp0 | small, measured | OpenVINO/system |
| 13 | Contention matrix + 5-minute thermal run + energy per chunk/task | evidence for 20 + 5 pts | measurement |
| 14 | Stretch only: drop `inspect` from the policy (retrain), or the AE-on-NPU split presented as a power experiment | −20–25% / power, not latency | model |

```bash
# ⚠️ All commands below are from docs/source reading; verify paths and names on the XPS.
# Topology and pinning
lscpu -e=CPU,CORE,CLUSTER,MAXMHZ
for c in /sys/devices/system/cpu/cpu[0-9]*; do echo "$(basename $c) cap=$(cat $c/cpu_capacity) max=$(cat $c/cpufreq/cpuinfo_max_freq)"; done
sudo chrt -f 80 taskset -c 3 python control_loop.py          # servo process alone on a P-core
sudo systemctl set-property --runtime user.slice AllowedCPUs=0-2,4-15
sudo cyclictest -m -p80 -a3 -i1000 -D60 --histogram=200      # jitter check on the control core

# Power and GPU frequency
cat /sys/firmware/acpi/platform_profile_choices; powerprofilesctl set performance
G=/sys/class/drm/card0/device/tile0/gt0                       # find the xe card first
cat $G/freq0/{rpn_freq,rpe_freq,rpa_freq,rp0_freq}
cat $G/freq0/rpa_freq | sudo tee $G/freq0/min_freq
watch -n0.5 "cat $G/freq0/act_freq $G/freq0/throttle/reasons"

# Cameras and serial
lsusb -t; v4l2-ctl -d /dev/videoX --list-formats-ext
echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend
lsusb | grep -Ei 'ftdi|1a86|wch'; echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer   # FTDI only

# Telemetry during every run
sudo turbostat --quiet --interval 1 --show PkgWatt,CorWatt,GFXWatt,PkgTmp,Busy%,Bzy_MHz
cat /sys/class/power_cap/intel-rapl:0/energy_uj               # before/after each window
cat /sys/bus/pci/drivers/intel_vpu/*/npu_busy_time_us         # 1 Hz
sudo perf list | grep xe_                                     # then perf stat engine-active/total-ticks
```

```python
# ⚠️ physicalai kwargs pass-through is read from source, not tested
model = InferenceModel(export_dir, device="GPU", PERFORMANCE_HINT="LATENCY",
                       INFERENCE_PRECISION_HINT="f16", CACHE_DIR="/home/intel/ovcache")
# Anomalib on NPU
m = core.read_model(xml); m.reshape({m.input(0): ov.PartialShape([1, 3, 256, 256])})
core.set_property({"CACHE_DIR": "/home/intel/ovcache_npu"})
cm = core.compile_model(m, "NPU", {"PERFORMANCE_HINT": "LATENCY"})
print(cm.get_property("EXECUTION_DEVICES"))
```

## Conclusion

The research changes the optimisation story from "copy Intel's 555 → 172 ms" to something more defensible and more interesting to judges. Part of Intel's progression (GatedMLP) is not available in release builds. The showcase NPU split loses latency in synchronous mode. One default, 200 padded language tokens, quietly inflates every chunk. The team's advantage is that its baseline already isolates prefix from denoising cost. That lets it attribute each step of the staircase to a mechanism, which Intel's own paper does not do. The negative results are worth presenting too: the measured GPU+NPU split, and the contention matrix. They show device placement chosen from evidence, which is what rubric expectation #5 actually asks for.

Two uncertainties should shape the schedule. The ~90–130 ms target rests on third-party numbers from an X7 358H under Windows, so step 3 is the first real checkpoint. If the fresh Studio export is not near 250 ms, stop and profile with `-pc` before stacking further changes. Second, SnapFlow has no SO-101 evidence. Plan its three extra epochs into the Modal run from the start, but demo whichever export (1-step or 10-step) actually sorts bricks reliably. With RTC hiding the latency, reliability is worth more points than the last 50 ms.
