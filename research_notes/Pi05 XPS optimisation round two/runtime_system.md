# Runtime-, graph- and system-level levers for π0.5 on Panther Lake (OpenVINO 2026.4.1, xe, Ubuntu 26.04)

Reading guide for the report writer:

- "Cited Findings" are things I read in a source during this session.
- "Inferences" hold the levers, commands and expected gains. Commands that follow directly from a cited sysfs or API definition are marked (derived). Commands from my prior knowledge that I did not re-verify are marked (unverified).
- No expected-gain figure below is a measurement. I found no published measurement of any of these levers on π0.5, or on any bursty iGPU inference workload, on Panther Lake. Every gain is an estimate to be tested on the device.
- Several primary pages could not be read (Phoronix returned 403; docs.openvino.ai pages were too long to reach the body; some raw GitHub and kernel-doc URLs returned 404). The affected points are listed under Gaps.

## 1. GPU frequency and power (xe sysfs, SLPC, platform profile, RAPL, thermal)

### Takeaway
The xe driver exposes three writable knobs per GT (`min_freq`, `max_freq`, `power_profile`) plus a throttle-reason readout, and it has no waitboost, so a bursty workload depends entirely on GuC SLPC's ramp. Pinning `min_freq` to `rp0_freq` (sudo) is the one cheap, untested lever for the gap between the back-to-back benchmark and a one-burst-per-second live loop; nobody has published its effect on inference.

### Cited Findings
- xe enables GuC SLPC; GuC requests a frequency between the sysfs min and max, and PCODE makes the final decision from thermal and other conditions — [kernel docs 7.0, Xe GT Frequency Management](https://docs.kernel.org/7.0/gpu/xe/xe_gt_freq.html)
- The sysfs directory is `<device>/tile#/gt#/freq0/`. Read-only: `act_freq` (actual, resolved by PCODE), `cur_freq` (requested by GuC), `rpn_freq` (minimum), `rpa_freq` (achievable, runtime-computed), `rpe_freq` (efficient, runtime-computed), `rp0_freq` (maximum). Read-write: `min_freq`, `max_freq`. "If max <= min, then freq_min becomes a fixed frequency request" — [kernel docs 7.0](https://docs.kernel.org/7.0/gpu/xe/xe_gt_freq.html)
- In the v7.0 source, `freq0/` also has a read-write `power_profile` attribute, which the DOC comment does not describe — [xe_gt_freq.c v7.0](https://raw.githubusercontent.com/torvalds/linux/v7.0/drivers/gpu/drm/xe/xe_gt_freq.c)
- `power_profile` accepts `base` and `power_saving`; reading prints the active one in brackets. With power saving, "SLPC will use conservative thresholds to ramp frequency" — [xe_guc_pc.c v7.0](https://raw.githubusercontent.com/torvalds/linux/v7.0/drivers/gpu/drm/xe/xe_guc_pc.c)
- The default SLPC profile is base (0); power_saving (1) has conservative up/down thresholds (i915 patch describing the same GuC feature) — [dri-devel, SLPC power profiles patch](https://lists.freedesktop.org/archives/dri-devel/2025-January/486434.html)
- Writing `min_freq` in xe: rejected outside `rpn_freq`..`rp0_freq`; returns `-EAGAIN` during a reset; sets `SLPC_PARAM_IGNORE_EFFICIENT_FREQUENCY` to true when the value is below RPe; then sets `SLPC_PARAM_GLOBAL_MIN_GT_UNSLICE_FREQ_MHZ` — [xe_guc_pc.c v7.0](https://raw.githubusercontent.com/torvalds/linux/v7.0/drivers/gpu/drm/xe/xe_guc_pc.c)
- xe has no waitboost logic in `xe_guc_pc.c`; the only raise is a one-time one for GuC load — [xe_guc_pc.c v7.0](https://raw.githubusercontent.com/torvalds/linux/v7.0/drivers/gpu/drm/xe/xe_guc_pc.c)
- xe enables `SLPC_OPTIMIZED_STRATEGY_COMPUTE` at GuC PC start, and `pc_modify_defaults()` disables DCC (duty-cycle control) only on Panther Lake — [xe_guc_pc.c v7.0](https://raw.githubusercontent.com/torvalds/linux/v7.0/drivers/gpu/drm/xe/xe_guc_pc.c)
- "Render-C states is also a GuC PC feature that is now enabled in Xe for all platforms" — [xe_guc_pc.c v7.0](https://raw.githubusercontent.com/torvalds/linux/v7.0/drivers/gpu/drm/xe/xe_guc_pc.c)
- Throttling is exposed under `freq0/throttle/`: `status` (0/1), `reasons` (list, or "none"), and per-reason files `reason_pl1`, `reason_pl2`, `reason_pl4`, `reason_thermal`, `reason_prochot`, `reason_ratl`, `reason_vr_thermalert`, `reason_vr_tdc`. The docs recommend reading `reasons` because checking `status` then individual reasons is racy — [kernel docs 7.0](https://docs.kernel.org/7.0/gpu/xe/xe_gt_freq.html)
- Linux 7.2 made the Arc B390 faster across OpenGL and Vulkan workloads on a Core Ultra X7 358H laptop, attributed to xe driver improvements, with little change in power draw — [Phoronix, Linux 7.2 Panther Lake](https://www.phoronix.com/review/linux-72-panther-lake/2) (read via search snippet only; page returned 403)
- A secondary summary puts that gain at up to 12% at unchanged power — [PCGH citing Phoronix](https://www.pcgameshardware.de/Linux-Software-26761/News/Kernel-7-2-Intel-Arc-B390-bis-zu-12-Prozent-schneller-1549010/)
- A kernel series adds a power slider for Panther Lake through the Linux platform-profile API; Ubuntu's SRU verifies it by checking `/sys/class/platform-profile/platform-profile-<>`, and lists a possible performance/thermal regression as the risk — [Ubuntu kernel-team SRU](https://lists.snapcraft.io/archives/kernel-team/2025-October/163671.html); [int340x power slider series](https://patchew.org/linux/20250825132315.75521-1-srinivas.pandruvada@linux.intel.com)
- On Dell laptops, changing the platform profile updates a BIOS thermal option and thermald is not affected (2024 thread about older Dell hardware, not Panther Lake) — [LKML, dell-laptop platform_profile](https://lkml.iu.edu/hypermail/linux/kernel/2405.1/03694.html)
- Intel's own π0.5 benchmark distinguishes a Max-TDP point (65 W on the X7 358H) from an ISO-TDP 40 W point, and reports different latency at each, so package power limit is a first-order variable for this model — [Intel OEP π0.5 article](https://docs.openedgeplatform.intel.com/dev/_sources/OEP-articles/publications/optimizing-pi0.5-lva-model.md.txt)

### Inferences
- Lever F1, pin GT frequency (sudo, reversible, low risk on AC). (derived)
  ```
  G=/sys/class/drm/card0/device/tile0/gt0/freq0     # card index may differ; check gt1 too
  cat $G/{rpn,rpe,rpa,rp0,min,max,cur,act}_freq $G/power_profile $G/throttle/reasons
  cat $G/rp0_freq | sudo tee $G/min_freq            # floor at max; restore with rpn_freq value
  ```
  Because there is no waitboost and Render-C states are on, a 170 ms burst arriving after 0.3–1.3 s of idle starts from whatever SLPC chose and from a possibly power-gated GT. The 169 ms figure was presumably measured back-to-back, when SLPC is already ramped. So this lever targets the live-loop penalty, not the 169 ms. Expected gain: unknown; zero if `act_freq` already sits at RP0 for the whole burst. Test first by sampling `act_freq` at 1 ms intervals across a live burst.
- Whether a frequency floor also prevents RC6 entry between bursts is not established by anything I read. If the GT still power-gates, the fallback is a keep-alive: submit a trivial GPU job (for example a tiny infer request) every few milliseconds between chunks. This costs power and heat, which matters because PCODE can lower frequency for thermal reasons.
- Lever F2, confirm `power_profile` is `[base]` (derived). It should be by default; a power-management daemon could have changed it. `echo base | sudo tee $G/power_profile`. Expected gain: none unless it was wrong.
- Lever F3, read `throttle/reasons` during a run (no sudo to read). This says directly whether PL1, PL2, PL4, thermal or VR limits are clipping the GPU. It is a diagnostic, and it decides whether F4 is worth doing. The 15–30% contention penalty could be partly a package-power effect (CPU work consuming shared budget) rather than GPU time-slicing; `reason_pl1`/`reason_pl2` appearing only under contention would show that.
- Lever F4, platform profile to performance (no sudo for `powerprofilesctl`). (unverified commands) `cat /sys/firmware/acpi/platform_profile_choices; powerprofilesctl set performance`, and check `/sys/class/platform-profile/`. Then read RAPL limits: `grep . /sys/class/powercap/intel-rapl:0/constraint_*_power_limit_uw` (sudo may be needed). Expected gain: only if F3 shows power throttling. Intel's 40 W vs 65 W split implies the compute-bound prefix (82 ms) responds to power budget more than the bandwidth-bound expert does.
- Lever F5, mainline kernel 7.2 or later (sudo, reboot, moderate risk the day before a demo). The cited gain is for graphics; I found no compute or OpenVINO measurement, and the mechanism was not readable. Treat as unproven for this workload.
- Will not help: `max_freq` (already RP0 by default per `pc_adjust_freq_bounds`), and the `power_saving` profile (strictly slower ramp).

### Gaps
- No measured data anywhere on SLPC ramp time on Xe3, or on iGPU inference latency for bursty versus continuous load. This must be measured locally.
- The `gtidle` documentation page returned 404. From prior knowledge the directory is `tile#/gt#/gtidle/` with `name`, `idle_status`, `idle_residency_ms`, and xe offers no userspace switch to disable GT-C6; both are unverified this session.
- Dell XPS 16 (Panther Lake) PL1/PL2 values, default platform profile, and thermald behaviour: no source found. The Arch Wiki page that exists covers the Meteor Lake XPS 16 9640.
- Phoronix said it would publish a balanced/performance/power-saver comparison for Panther Lake; I could not find or read it.
- Which xe change in 7.2 produced the graphics gain, and whether Ubuntu 26.04's 7.0 kernel has it backported: not determined.
- Whether Panther Lake exposes a separate media GT (`gt1`) with its own `freq0`: not verified.

## 2. Memory bandwidth

### Takeaway
I found one hard number: Intel quotes 153–154 GB/s peak for LPDDR5 at 8533 MT/s on the X7 358H, which makes the measured 107 GB/s about 70% of that peak rather than 80%, if the XPS memory runs at the same speed. I found no sourced runtime or system lever that raises bandwidth available to the iGPU; the remaining options are to stop other consumers and to read fewer bytes, and the latter is mostly model-level work outside this brief.

### Cited Findings
- Intel's π0.5 article lists the Core Ultra X7 358H test system as 32 GB LPDDR5 at 8533 MT/s with 153 GB/s peak bandwidth (the abstract says "up to 154 GB/s") — [Intel OEP π0.5 article](https://docs.openedgeplatform.intel.com/dev/_sources/OEP-articles/publications/optimizing-pi0.5-lva-model.md.txt)
- The same table lists Jetson AGX Orin 64 GB at 204 GB/s and Jetson Thor at 273 GB/s, and Intel reports only 1.1× better latency than Thor at 40 W, with stock PyTorch within 2.4% of Thor at full power (294 ms vs 287 ms) — [Intel OEP π0.5 article](https://docs.openedgeplatform.intel.com/dev/_sources/OEP-articles/publications/optimizing-pi0.5-lva-model.md.txt)
- In Intel's heterogeneous design the KV cache crosses devices "via shared DDR with zero-copy USM-host tensors", so iGPU, NPU and CPU all draw on the same memory — [Intel OEP π0.5 article](https://docs.openedgeplatform.intel.com/dev/OEP-articles/publications/optimizing-pi0.5-lva-model.html)
- OpenVINO 2026.2 added INT4 KV-cache compression on GPUs and by-channel INT8 KV quantisation in a preview feature — [OpenVINO GitHub releases](https://github.com/openvinotoolkit/openvino/releases)
- OpenVINO 2026.0 added int4 data-aware compression for 3D MatMuls — [OpenVINO GitHub releases](https://github.com/openvinotoolkit/openvino/releases)

### Inferences
- The "80% of peak" figure in the profile implies an assumed peak near 134 GB/s. Against Intel's 153 GB/s the expert MLP is at about 70%. The X9 388H XPS may run a different memory speed; check with `sudo dmidecode -t memory | grep -i -E "speed|type"` (sudo). Either way the headroom from system tuning is at most the distance to peak, and real workloads rarely exceed 80–85% of theoretical, so expect single-digit percent at best from this whole category.
- Since CPU, iGPU and NPU share one memory bus, any bandwidth-heavy co-tenant slows the expert directly. This is a plausible cause of part of the 15–30% contention penalty, and it means moving the expert to the NPU (Intel's layout) does not escape the bandwidth limit. Practical lever: keep CPU-side memory traffic low during the expert phase (no image resize/colour conversion, video encode, or large memcpy concurrent with denoising; schedule camera preprocessing into the prefix phase, which is compute-bound). No sudo. Expected gain: recovers contention loss only.
- INT4 weights for the expert MLP FCs would halve the bytes read, which is the only large lever against a bandwidth bound, but it is a quantisation recipe and belongs to the model-level researchers. The runtime caveat is that an INT4 FullyConnected path must not decompress to a wider format in memory before the multiply, or the saving disappears; verify with the per-kernel profile before trusting it.
- KV-cache INT4/INT8 compression (2026.2) reduces bytes read by SDPA, which is 19% of time. It is designed for stateful LLM pipelines; whether it applies to explicit KV tensors in a static IR is not established. The property is `ov::hint::kv_cache_precision` (unverified name).
- Transparent huge pages: I found no source showing that THP or shmem huge-page settings change iGPU USM allocation or GPU TLB behaviour on xe. A cheap experiment is `cat /sys/kernel/mm/transparent_hugepage/{enabled,shmem_enabled}` and trying `advise`/`always` (sudo). Expectation: no measurable change; a sequential weight stream is not TLB-limited.
- Weight layout: nothing found. Weight sharing across the 10 denoising steps is already implicit (same weights re-read each step); the bound is the re-read, which only fewer bytes or fewer steps removes.

### Gaps
- LPDDR5X speed and gear mode of the XPS 16 with the X9 388H: no source found.
- No published measurement of iGPU/CPU bandwidth contention on Panther Lake.
- No documentation found on compute-runtime USM allocation flags that affect bandwidth or page size on xe.

## 3. Newer software (OpenVINO, compute-runtime, Level Zero, oneDNN)

### Takeaway
OpenVINO 2026.4.1 (1 Oct 2026) is the newest release on GitHub; I found no 2026.5 or 2027.0 release. The one concrete upgrade available is intel-compute-runtime 26.40.40196.8 (3 Nov 2026), which targets Ubuntu 26.04 / kernel 7.0, but its notes make no performance claims.

### Cited Findings
- OpenVINO GitHub releases, newest first: 2026.4.1 (1 Oct, marked Latest), 2026.4.0 (16 Sep), 2026.3.1 (26 Aug), 2026.3.0 (4 Aug), 2026.2.1 (17 Jun), 2026.2.0 (28 May) — [OpenVINO GitHub releases](https://github.com/openvinotoolkit/openvino/releases)
- 2026.4.0 includes Xe3 GenAI optimisations and visual-token support on Core Ultra Series 3; the documented GPU item is improved TTFT/TPOT for Gemma 4 models — [OpenVINO GitHub releases](https://github.com/openvinotoolkit/openvino/releases); [OpenVINO release notes](https://docs.openvino.ai/releasenotes)
- 2026.4.1 GPU fixes: wrong or corrupted results with zero-copy remote output tensors that have a transpose at the output under dynamic shapes; blocked-format reorder offsets; layout propagation through Reshape — [OpenVINO GitHub releases](https://github.com/openvinotoolkit/openvino/releases)
- 2026.4.1 also lists sliding-window KV-cache fixes and GroupQueryAttention causal support — [OpenVINO GitHub releases](https://github.com/openvinotoolkit/openvino/releases)
- 2026.2.1 fixed the NPU shared Level Zero command queue violating app/inference priorities — [OpenVINO GitHub releases](https://github.com/openvinotoolkit/openvino/releases)
- The release notes on that page do not mention FullyConnected, dynamic quantisation, gated MLP, VLA or π0 — [OpenVINO GitHub releases](https://github.com/openvinotoolkit/openvino/releases)
- compute-runtime releases in 2026: 26.40.40196.8 (3 Nov, Latest; Level Zero 1.34.0, IGC 2.42.6; targets Ubuntu 26.04 with kernel 7.0.0-14), 26.35.39758.10 (17 Sep), 26.31.39395.13 (21 Aug), 26.27.39122.11 (21 Jul; also validated on Ubuntu 26.04 kernel 7.0.0-14), 26.22.38646.4 (16 Jun; introduces LEO, "Level Zero Executing OpenCL"), back to 26.01.36711.4 (14 Jan). Panther Lake is Production in all of them. The notes do not mention performance or USM changes — [intel/compute-runtime releases](https://github.com/intel/compute-runtime/releases)
- LEO was enabled by default for Nova Lake in 26.35 — [intel/compute-runtime releases](https://github.com/intel/compute-runtime/releases)
- Intel's π0.5 benchmark used OpenCL Compute Runtime 26.01.36711.4 on Ubuntu 24.04.4, kernel 6.17 — [Intel OEP π0.5 article](https://docs.openedgeplatform.intel.com/dev/_sources/OEP-articles/publications/optimizing-pi0.5-lva-model.md.txt)
- oneDNN's Graph API Gated-MLP pattern lists "ReLU, GELU, Sigmoid, and so on" as activations, supports f32/bf16/f16 (int8 is not mentioned), and on GPU is "implemented as the reference implementation", typically three matmul primitives with binary or eltwise post-ops — [oneDNN Gated-MLP guide](https://uxlfoundation.github.io/oneDNN/dev_guide_graph_gated_mlp.html)
- A oneDNN draft PR adding a benchdnn GatedMLP driver notes a known GPU bug traced to `src/gpu/intel/gated_mlp/ref.hpp` — [oneDNN PR #4951](https://ddprojects.chickenkiller.com/uxlfoundation/oneDNN/pull/4951) (mirror; read via search snippet only)
- An OpenVINO PR titled "[GPU] Adding GeGLU" (#24970, June 2024) appeared in a PR listing; its content and merge state were not confirmed — [OpenVINO pulls listing](https://fr.github.com/openvinotoolkit/openvino/pulls)

### Inferences
- Lever S1, upgrade compute-runtime to 26.40.40196.8 (sudo; low-to-moderate risk; keep the old .debs to roll back). Check the installed version first with `apt list --installed 2>/dev/null | grep -E "intel-opencl-icd|libze|libigc"` or `clinfo | grep -i "driver version"`. If Ubuntu 26.04 already ships something near 26.27 or later, expect no change. Kernel compile time (IGC) and 3,940-kernel programs may change; clear the OpenVINO `cache_dir` after upgrading. Expected latency gain: unknown, probably within noise, since the hot kernels are oneDNN gemm and an OpenVINO OpenCL SDPA kernel whose speed depends mostly on generated code and hardware.
- LEO changes how OpenCL is executed (on top of Level Zero). On Panther Lake it is not stated to be default. If an environment switch exists to enable it, it is an experiment with unknown sign; I did not find the switch.
- Lever S2, OpenVINO nightly (no sudo, pip wheel in a separate venv). No evidence found of post-2026.4 SDPA, XMX INT8 or `gated_mlp` GELU work. Low expected value against the cost of re-validating; the patched debug-caps build already underway covers the one known item.
- GatedMLP fusion, expectation management. The profile says the expert FCs are bound by weight reads. Fusing up-proj, gate, activation and multiply removes intermediate activation traffic and kernel launches, not weight bytes. Launch overhead is already ruled out (2.7 µs per kernel). So even with GeGLU matching patched in, the gain on the expert should be small; the larger effect, if any, would be on the compute-bound prefix. oneDNN's own description of its Gated-MLP as a reference implementation built from three matmuls supports a low expectation, though OpenVINO's GPU plugin has its own implementation that I could not read.
- The 2026.4.1 remote-tensor fix matters if the pipeline is split (section 4): zero-copy output tensors with a trailing transpose were producing wrong results under dynamic shapes before 2026.4.1. Static shapes and 2026.4.1 avoid it.

### Gaps
- The docs.openvino.ai release-notes body could not be read (page too long), so per-plugin detail for 2026.4.0 is from the GitHub summary only.
- The OpenVINO `gated_mlp` fusion source could not be fetched (404 on the guessed path), so I could not confirm the SwiGLU-only matcher, the debug gating, or precision constraints independently.
- oneDNN release notes after mid-2026 were not reviewed; no evidence either way on new Xe3 gemm or SDPA micro-kernel work.
- No published before/after latency for any compute-runtime version on Panther Lake.

## 4. Graph-level restructuring within OpenVINO

### Takeaway
Intel's published π0.5 design is exactly a three-way split (vision encoder and language model on the iGPU, action expert on the NPU, per-layer KV cache handed over as zero-copy USM-host tensors), but Intel publishes no per-stage latency, so there is no evidence that it is faster than all-GPU. For a single robot loop, splitting helps only if two stages can overlap on different devices, and all devices share one memory bus.

### Cited Findings
- "the Vision Encoder (VE) and Language Model (LM) run on the iGPU, while the Action Expert (AE) runs on the NPU"; "The per-layer KV cache is the only cross-device handoff, carried via shared DDR with zero-copy USM-host tensors" — [Intel OEP π0.5 article](https://docs.openedgeplatform.intel.com/dev/OEP-articles/publications/optimizing-pi0.5-lva-model.html)
- Intel's benchmark configuration: DROID variant, 3 cameras at 224×224, 64-token language context, 10 denoising steps, BF16/FP16 — [Intel OEP π0.5 article](https://docs.openedgeplatform.intel.com/dev/_sources/OEP-articles/publications/optimizing-pi0.5-lva-model.md.txt)
- Intel's optimised results are given only as ratios: up to 1.5× lower latency than AGX Orin at max TDP; at 40 W, up to 2.6× lower latency than Orin and 1.1× better than Thor. No absolute OpenVINO latency and no per-stage or per-device breakdown is published — [Intel OEP π0.5 article](https://docs.openedgeplatform.intel.com/dev/_sources/OEP-articles/publications/optimizing-pi0.5-lva-model.md.txt)
- Intel's quick-start runs the exported model through `InferenceModel` with the OpenVINO backend, `PolicyRuntime` at 30 fps, and `SyncExecution` in chunk mode — [Intel OEP π0.5 article](https://docs.openedgeplatform.intel.com/dev/_sources/OEP-articles/publications/optimizing-pi0.5-lva-model.md.txt)
- Intel also ships a π0.5 + Real-Time Chunking sample on OpenVINO, and an Embodied Intelligence SDK pipeline "optimized on Intel integrated GPU" with an INT8 weight-compression tutorial — [Pi0.5 with RTC](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/embodied/sample_pipelines/pi05_with_rtc.html); [OpenVINO optimisation of Pi0.5](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/embodied/openvino_optimization.html)
- A known issue in that SDK: `CL_OUT_OF_RESOURCES` running π0.5 with the i915 driver on a Core Ultra Series 2 platform, worked around by rebinding to xe — [Embodied Intelligence SDK release notes](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-suites/robotics-ai-suite/embodied/release-notes.html)
- Nota AI: a LinkedIn post says NetsPresso-optimised VLA models, including π0.5, were shown at Humanoids Summit Seoul; a separate demo on Qualcomm Dragonwing IQ-9075 reports action-head time cut from 218 ms to 31 ms by action-head-only optimisation. Neither involves Intel hardware — [LinkedIn post](https://kr.linkedin.com/in/kimdowha); [Edge AI and Vision Alliance](https://www.edge-ai-vision.com/2026/07/nota-ai-demonstration-of-vla-robotics-on-dragonwing-enhcanced-by-netspresso/)

### Inferences
- Lever G1, split the IR into prefix (vision + LM) and expert, and pipeline across chunks. With one GPU, running chunk N+1's prefix while chunk N's expert runs does not shorten either: the GPU time-slices, and the measured 15–30% contention penalty suggests overlap costs more than it saves. Pipelining on a single device changes observation staleness, not throughput. It only pays if the two stages are on different devices.
- Lever G2, expert on NPU, prefix on GPU (Intel's layout), with overlap. If the NPU ran the 10-step expert in about the same 87 ms as the GPU, a pipelined loop could approach max(82, 87) ms per chunk instead of 169 ms, at the cost of one chunk of extra observation latency. Three unknowns block this estimate: NPU expert latency is unpublished; the expert is bandwidth-bound and the NPU shares the bus, so overlapping it with the prefix may slow both; and Intel's framing is performance per watt, which suggests the NPU placement may be an efficiency choice. Measure the expert alone on NPU before investing: `benchmark_app -m expert.xml -d NPU -hint latency -niter 50` (unverified flags). No sudo. Risk: NPU compile time, static-shape requirement (already met), INT8 handling differences.
- Lever G3, vision encoder on NPU concurrently. Only worthwhile if something else useful runs on the GPU at the same moment; within one chunk the LM cannot start before vision finishes. It helps only in combination with G2-style pipelining or if SigLIP is a large share of the 82 ms. The share is not in the profile as given.
- Lever G4, explicit KV tensors as USM-host or remote tensors between sub-models. This is needed for any split and matches Intel's design. On GPU-to-GPU handoff use remote tensors from the same `ov::intel_gpu::ocl::ClContext` so nothing is copied (unverified API name); for GPU-to-NPU use USM-host as Intel does. Use 2026.4.1 and static shapes because of the remote-output-tensor bug fixed there (section 3). Expected gain by itself: none versus one static IR, which already keeps KV on-device; it is an enabler.
- Stateful models, multiple infer requests, `ov::hint::model_priority`: no benefit expected for one model run sequentially by one client. Multiple infer requests on the same GPU share the same queue. `model_priority` arbitrates between models under AUTO/multi-device scheduling and is unrelated to single-model latency (unverified semantics; consistent with the measured null result for queue priority).
- Batching three cameras is already done. Nothing further found.

### Gaps
- No per-stage, per-device latency from Intel; no NPU numbers for the action expert; no measured heterogeneous trade-off from Intel or Nota AI. The assignment expected such measurements; I could not find them.
- No Nota AI model card or blog for π0.5 on Intel hardware was found.
- The Intel `pi05-optimization` and `openvino_optimization` source pages returned 404 when fetched, so their benchmark_app flags and split details are unread.
- docs.openvino.ai GPU-device page body could not be read; all OpenVINO property names in this note are from prior knowledge.

## 5. SDPA alternatives and prefix attention cost

### Takeaway
I found no source describing an alternative SDPA implementation for Xe3 that beats the default path, and the profile already shows that turning the SDPA optimisation off is 18–22% slower. The remaining SDPA ideas are structural (KV reuse, compressed KV) and none has published numbers for this model.

### Cited Findings
- OpenVINO 2026.2 added an SDPA path for LFM2 models, INT4 KV-cache compression on GPUs, and by-channel INT8 KV quantisation in the XAttention preview — [OpenVINO GitHub releases](https://github.com/openvinotoolkit/openvino/releases)
- An earlier GPU-plugin note says the ScaledDotProductAttention operator was enhanced for GenAI workloads including long prompts — [OpenVINO release notes 2025](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino.html)
- OpenVINO 2026.4.1 added GroupQueryAttention causal and partial-rotary support (ONNX-frontend operator) — [OpenVINO GitHub releases](https://github.com/openvinotoolkit/openvino/releases)
- In Intel's design the expert consumes the prefix's per-layer KV cache, so prefix KV is computed once per chunk and reused across the 10 denoising steps — [Intel OEP π0.5 article](https://docs.openedgeplatform.intel.com/dev/OEP-articles/publications/optimizing-pi0.5-lva-model.html)

### Inferences
- KV reuse between prefix and expert is the published baseline design. If the static IR already computes prefix KV once and feeds all ten steps, there is nothing left here; confirm in the IR that prefix K/V projections are not recomputed per step.
- Causal or block masks do not reduce SDPA work in a dense kernel; a mask is an added tensor, and the kernel still computes the full QKᵀ. Savings need a smaller sequence, which is token pruning and out of scope.
- KV precision: if `ov::hint::kv_cache_precision` (unverified name) applies only to stateful models, the equivalent for explicit KV tensors is to store them as f16 or i8 in the graph. For the expert's cross-attention over the prefix, that halves or quarters the KV bytes read in each of 10 steps. This is the only SDPA idea here that addresses a bandwidth cost. Size of gain unknown; SDPA is 19% in total and only part of that is KV reads.
- head_dim 256: nothing found on how `ocl::sdpa::opt` or a micro-kernel variant handles it, or on a oneDNN SDPA primitive being selectable in OpenVINO 2026.4.

### Gaps
- No source found comparing a oneDNN SDPA primitive with `ocl::sdpa::opt` on Xe3, or documenting micro-kernel variants and their head-dimension limits.
- No flash-attention-style GPU kernel work found in the release notes read.
- Source code of the SDPA kernel selector was not read.

## 6. Alternative runtimes on the same hardware

### Takeaway
The only published number for another runtime is Intel's own: stock PyTorch 2.12 on XPU takes 294 ms for π0.5 on the X7 358H at 40 W (BF16/FP16), which is well behind the 169 ms already measured with OpenVINO INT8. I found no head-to-head showing any runtime beating OpenVINO GPU for this model, and the other candidates are wrappers around OpenVINO.

### Cited Findings
- "Using stock PyTorch 2.12, Intel Core Ultra Series 3 X7 358H processor at 40 W records 294 ms", against 287 ms on Jetson Thor at full power — [Intel OEP π0.5 article](https://docs.openedgeplatform.intel.com/dev/OEP-articles/publications/optimizing-pi0.5-lva-model.html)
- Intel's optimised comparison uses OpenVINO on Intel and TensorRT on NVIDIA, i.e. Intel itself moves from PyTorch to OpenVINO for best latency — [Intel OEP π0.5 article](https://docs.openedgeplatform.intel.com/dev/_sources/OEP-articles/publications/optimizing-pi0.5-lva-model.md.txt)
- Intel's PyTorch 2026 H1 plan: continue improving the Triton backend for Intel GPUs and introduce a SYCL-TLA backend for `torch.compile` — [dev-discuss.pytorch.org](https://dev-discuss.pytorch.org/t/intel-gpu-cpu-enabling-status-and-feature-plan-2026-h1-update/3320.md)
- PyTorch 2.12 XPU wheels install from `https://download.pytorch.org/whl/xpu`; Intel's prerequisites page lists Panther Lake as verified on Windows 11 and warns that Intel Deep Learning Essentials 2026 is incompatible with PyTorch 2.12 — [PyTorch XPU getting started](https://docs.pytorch.org/docs/2.12/notes/get_start_xpu.html); [Intel prerequisites for PyTorch 2.12](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-12.html)
- OpenVINO 2026.1 introduced a preview llama.cpp OpenVINO backend, and Intel published an ExecuTorch-with-OpenVINO-backend update in 2026 — [OpenVINO GitHub releases](https://github.com/openvinotoolkit/openvino/releases); [Intel community blog](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/ExecuTorch-with-OpenVINO-Backend-in-2026-New-Capabilities-and/post/1743166) (title seen in search results; not read)

### Inferences
- Will not help: switching runtime. PyTorch XPU at 294 ms (40 W, FP16) versus 169 ms measured here is not a like-for-like comparison (different SKU, power cap and precision), but the direction matches Intel's own choice of OpenVINO for its best numbers. ExecuTorch's OpenVINO backend, ONNX Runtime's OpenVINO EP and llama.cpp's OpenVINO backend all execute through the same GPU plugin, so they cannot beat a tuned direct OpenVINO pipeline and add a layer.
- `torch.compile` on XPU is still on Intel's roadmap as a thing being improved, which suggests it is not yet a mature fast path. No numbers found.
- A hand-written SYCL path is not justified: the profile shows 69% of time in oneDNN gemm that is already near the memory-bandwidth limit, which custom code cannot exceed.

### Gaps
- No published head-to-head (OpenVINO vs PyTorch XPU vs ONNX Runtime OpenVINO EP vs ExecuTorch) for π0.5 or any VLA on Panther Lake.
- No absolute OpenVINO latency from Intel to compare against the 169 ms.
- XPUGraph: nothing found under that name.

## 7. Process-level isolation for a live demo

### Takeaway
The xe driver has per-engine-class scheduler knobs (timeslice and preemption timeouts) and gates high exec-queue priority behind CAP_SYS_NICE, which may explain why queue priority made no measurable difference. I found no published contention measurements; the most reliable isolation is keeping every other client off the iGPU.

### Cited Findings
- xe exposes per-engine-class sysfs scheduler properties `timeslice_duration_us`, `preempt_timeout_us` and `job_timeout_ms`, with min/max caps and a `.defaults/` directory under each engine class (`gtN/engines/<class>/`) — [intel-xe patch series](https://lists.freedesktop.org/archives/intel-xe/2023-August/009171.html); [xe Kconfig.profile](https://git.zx2c4.com/linux-rng/plain/drivers/gpu/drm/xe/Kconfig.profile)
- The same bounds apply to the exec-queue property UAPI; when limits are not enforced, the CAP_SYS_NICE check decides access to elevated values — [xe Kconfig.profile](https://git.zx2c4.com/linux-rng/plain/drivers/gpu/drm/xe/Kconfig.profile)
- OpenVINO 2026.2.1 fixed the NPU's shared Level Zero command queue violating app/inference priorities — [OpenVINO GitHub releases](https://github.com/openvinotoolkit/openvino/releases)
- Measured contention in the project profile (15–30%) has no published counterpart; Intel's π0.5 article reports no multi-tenant numbers — [Intel OEP π0.5 article](https://docs.openedgeplatform.intel.com/dev/_sources/OEP-articles/publications/optimizing-pi0.5-lva-model.md.txt)

### Inferences
- Lever P1, keep other clients off the iGPU (no sudo). Run the demo browser with software rendering (`chromium --disable-gpu`, or `LIBGL_ALWAYS_SOFTWARE=1` for GL apps) (unverified flags), decode camera streams on the CPU (raw YUYV or CPU MJPEG decode, no VA-API), and avoid video overlays or animated dashboards. The desktop compositor will still use the iGPU; a static screen minimises it. Expected gain: recovers most of the GPU-side share of the 15–30% penalty. See which clients hold GPU time with DRM fdinfo: `grep -H drm- /proc/*/fdinfo/* 2>/dev/null` (sudo for other users' processes) (unverified for xe field names).
- Lever P2, give the inference process CAP_SYS_NICE so an elevated exec-queue priority is granted (sudo once). (derived from the UAPI gate; whether compute-runtime requests a higher xe priority when OpenVINO's `queue_priority` is HIGH is unverified.) Options: run the inference process as root for one A/B test, or `sudo setcap cap_sys_nice+ep $(readlink -f $(which python3))` (affects every use of that interpreter; revert with `setcap -r`). Expected gain: none in isolation; possibly a smaller contention penalty.
- Lever P3, xe timeslice tuning (sudo). (derived) Inspect `ls /sys/class/drm/card0/device/tile0/gt0/engines/*/` and read `timeslice_duration_us` and `preempt_timeout_us` for the render and compute classes. Shortening the timeslice makes a competing context yield sooner but raises switching overhead. Unmeasured; try only if P1 is impossible.
- Lever P4, CPU-side isolation (no sudo for user scopes). (unverified commands) `systemd-run --user --scope -p CPUWeight=1000 -p IOWeight=1000 python infer.py`, and put noisy helpers in a scope with `CPUWeight=20` and `AllowedCPUs=` restricted to E-cores. P-core pinning alone already showed no effect in isolation, and host gap is only 2.7 µs per kernel, so this protects against contention and gives nothing when idle. Remember that CPU work also competes for package power and memory bandwidth (sections 1 and 2); weights do not address those, only doing less work does.
- Lever P5, separate the contention causes before fixing them: run the co-tenant load three ways (CPU-only spin, memory-bandwidth load such as a large memcpy loop, GPU-only load) and read `throttle/reasons` and `act_freq` during each. That shows whether the 15–30% is GPU time-slicing, package power, or bandwidth.

### Gaps
- No published measurement of GPU contention effects or of xe priority/timeslice tuning on inference latency.
- Whether intel-compute-runtime maps OpenCL/OpenVINO queue priority to an xe exec-queue priority, and what it does without CAP_SYS_NICE: not verified in source.
- Default timeslice and preempt timeout values on kernel 7.0 for Panther Lake: not read.
- cgroup-level GPU controllers (a DRM cgroup for xe): nothing found; assume unavailable.
