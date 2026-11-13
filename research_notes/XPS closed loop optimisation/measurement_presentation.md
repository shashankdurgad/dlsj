# Measuring, Profiling and Presenting Latency / Throughput / Power for an On-Device Robotics AI Pipeline on Panther Lake (Core Ultra X9 388H, Ubuntu 26.04, OpenVINO 2026.4)

Scope: measurement and presentation only (not model optimisation or OS tuning). Research date: 14 Nov 2026. Each fact cites a source. Anything without a fetched source is under Inferences (marked "unverified" where useful) or Gaps. About 18 tool calls; some pages (docs.openvino.ai HTML, kernel.org xe_pmu page) could not be fetched and are listed as gaps.

Pipeline under test: cameras → Anomalib detector (NPU) → rule-based decision → pi0.5 VLA action chunks (iGPU, OpenVINO) → SO-101 arm. Baseline: pi0.5 PyTorch-XPU at 467 ms per chunk.

---

## Q1. OpenVINO tooling: benchmark_app, profiling counters, compiled-model properties, physicalai InferenceLatencyBenchmark

### Takeaway
Use `benchmark_app` to get repeatable, model-level numbers per device: `-hint latency` for the per-chunk latency claim, `-hint throughput` for throughput, and `-pc`/`-report_type detailed_counters` to show per-layer hotspots. Save JSON reports with `-report_folder` + `-json_stats` so the before/after evidence can be reproduced. For the whole policy, physicalai's `InferenceLatencyBenchmark` (warmup + max iters + max duration) is the in-framework equivalent. Neither tool measures the closed loop; that needs your own per-stage timestamps (Q3).

### Cited Findings
- `-hint` accepts `throughput` (default), `latency`, `cumulative_throughput` and `none`. Use `none` when you set `nstreams` or other device-specific knobs yourself. — [OpenVINO benchmark_app docs (source rst)](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/get-started/learn-openvino/openvino-samples/benchmark-tool.rst)
- `-pc` reports performance counters. `-pcsort` takes `no_sort|sort|simple_sort`, where `sort` orders by hotspot and `simple_sort` shows only executed ops. — [same](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/get-started/learn-openvino/openvino-samples/benchmark-tool.rst)
- `-report_type` options: `no_counters` gives config, FPS and latency; `average_counters` adds average per-layer counters; `detailed_counters` adds per-layer counters and latency per infer request. Reports go to `-report_folder` as `benchmark_<type>_report.csv`, or JSON with `-json_stats`. — [same](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/get-started/learn-openvino/openvino-samples/benchmark-tool.rst)
- `-latency_percentile` (1–100, default 50 = median) sets which percentile is reported as latency. Run it at 90 or 99 to report tails. — [same](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/get-started/learn-openvino/openvino-samples/benchmark-tool.rst)
- Other flags: `-niter`, `-t` (seconds; if both are set the run continues until both are met), `-nireq`, `-nstreams` (set 1 for the best latency estimate), `-api sync|async` (sync is the default under the latency hint), `-inference_only` (default for static models), `-exec_graph_path` (dumps the runtime graph with the ops actually executed), `-dump_config` / `-load_config` (JSON device config, so the exact config can be shipped with the results), `-data_shape` for dynamic models, and `-max_irate` (caps the inference rate, "may improve accuracy and reduce power consumption"). — [same](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/get-started/learn-openvino/openvino-samples/benchmark-tool.rst)
- Output: Count, Duration, Latency (Median, Average, Min, Max), Throughput (FPS), and first-inference time reported separately. C++ has `-no_warmup`, which is "not recommended" except in simulators. Default run time is 60 s on CPU and "depends on device" otherwise. The docs warn that power settings, overclocking and thermal throttling change results. — [same](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/get-started/learn-openvino/openvino-samples/benchmark-tool.rst)
- physicalai runtime: `from physicalai.benchmark.performance import InferenceLatencyBenchmark`. The README example is `InferenceLatencyBenchmark(max_iters=100, warmup_iters=2, max_duration=10000).run(InferenceModel("./exports/act_policy"))` and prints a JSON dict of metrics. The backend can be forced with `backend="openvino", device="GPU"`. The README does not document which metrics are returned. — [openvinotoolkit/physicalai README](https://github.com/openvinotoolkit/physicalai)
- The OpenVINO docs list `InferenceLatencyBenchmark` defaults as `max_iters=1000`, `warmup_iters=1`, `max_duration=60000` (units not shown in the snippet; the README example implies ms). — [OpenVINO Physical AI Benchmark API Reference (search snippet)](https://docs.openvino.ai/2026/physical-ai/reference/benchmark-api.html)

### Inferences
- Commands per model (paths illustrative):
  - Latency, VLA on iGPU, with p90 and per-layer counters:
    `benchmark_app -m pi05_vlm.xml -d GPU -hint latency -niter 500 -latency_percentile 90 -report_type detailed_counters -report_folder out/pi05_gpu_lat -json_stats -exec_graph_path out/pi05_gpu_exec.xml -dump_config out/pi05_gpu_cfg.json`
  - Anomalib on NPU: `benchmark_app -m anomalib.xml -d NPU -hint latency -t 30 -report_folder out/anom_npu -json_stats`. Repeat with `-d CPU` and `-d GPU` to justify the device choice with a 3×N device-mapping table, which directly addresses "appropriate use of NPU/GPU/CPU".
  - Throughput for comparison: the same with `-hint throughput` (and `-d GPU -hint cumulative_throughput` / `-d AUTO` only if relevant).
  - Run once with `-pc -pcsort sort`. The top-10 op table (e.g. attention / MatMul / Convolution, with the exec type and precision actually used, such as FP16 vs f32 fallbacks) makes a good backup slide showing where time goes.
- In-application profiling: set `ov::enable_profiling(true)` (property `PERF_COUNT=YES`) at `compile_model`, then call `infer_request.get_profiling_info()` (Python: `request.profiling_info` / `get_profiling_info()`), which returns per-node status, real_time, cpu_time, exec_type and node_type. This is the documented OpenVINO API, but the docs page was not fetched; verify against the 2026.4 Python API.
- Compiled-model properties to print into the evidence log for each model, via `compiled_model.get_property(...)`: `EXECUTION_DEVICES`, `PERFORMANCE_HINT`, `NUM_STREAMS`, `OPTIMAL_NUMBER_OF_INFER_REQUESTS`, `INFERENCE_PRECISION_HINT`, `KV_CACHE_PRECISION` (GPU LLM-style parts), `CACHE_DIR` / model-cache status, plus `ov.get_version()` and `core.get_property("GPU"/"NPU", "FULL_DEVICE_NAME")` and the NPU driver/compiler version (`NPU_DRIVER_VERSION`, `NPU_COMPILER_VERSION`). These are standard properties but were unverified against 2026.4 docs. Printing `EXECUTION_DEVICES` is the simplest proof that each model really ran on its intended device and did not silently fall back to CPU.
- Report first-inference / compile time separately ("cold start" vs "steady state"). Model caching (`CACHE_DIR`) changes it a lot, and judges care about deployment efficiency.
- benchmark_app uses random or synthetic inputs and isolated runs. It does not see camera I/O, preprocessing, Python overhead, device contention between NPU and GPU, or shared-memory bandwidth contention. Present it as the "kernel ceiling", next to the in-loop numbers.

### Gaps
- The full metric schema returned by `InferenceLatencyBenchmark.run()` (mean, p50, p95, std?) was not visible in fetched pages. Print it once and document it.
- docs.openvino.ai HTML pages (profiling / performance hints) were too large to fetch; the `ov::profiling_info` field names come from prior knowledge and are unverified for 2026.4.
- I found no OpenVINO 2026.4-specific changes to benchmark_app.

---

## Q2. GPU / NPU / CPU utilisation and power on Linux for Panther Lake (kernel 7.x, xe driver)

### Takeaway
On the xe driver, `intel_gpu_top` no longer works. Use igt `gputop`, qmassa, or `perf stat` on the xe PMU (engine-active/total ticks) for iGPU busy %. NPU busy % comes from `npu_busy_time_us` in intel_vpu sysfs (kernel 6.11+). NPU frequency attributes need kernel 7.2, which is newer than a stock 7.0 kernel. Energy comes from RAPL `energy_uj` deltas (package, and psys if present) or `turbostat`. Energy per inference = ΔJ / N inferences over a fixed window, with idle power subtracted if reporting incremental energy. Use unitrace / VTune for iGPU kernel timelines.

### Cited Findings
- igt-gpu-tools 2.4 (2026-03-31): "intel_gpu_top now rejects Xe devices and points users to gputop", and GPU frequency reporting was added to gputop. 1.29 added Xe support to gputop. 1.30 added PantherLake support and "a new power tool for power and energy measurement", renamed `igt_power` in 2.00. The latest release is 2.6 (2026-09-28). — [igt-gpu-tools NEWS](https://drm.pages.freedesktop.org/igt-gpu-tools/NEWS/)
- Users already hit "Failed to detect engines!" from intel_gpu_top on xe systems. — [Arch forums thread (via search)](https://bbs.archlinux.org/viewtopic.php?id=295732)
- xe PMU engine activity: two counters, `engine-active-ticks` and `engine-total-ticks`, with busy % = active/total × 100. Engine events need `gt`, `engine_class` and `engine_instance`, e.g. `perf stat -e xe_0000_00_02.0/engine-active-ticks,gt=0,engine_class=0,engine_instance=0/ -I 1000`. They are only advertised when GuC supports engine-activity reporting. — [intel-xe patch "Add PMU support for engine activity" v6, Feb 2025](https://lists.freedesktop.org/archives/intel-xe/2025-February/068357.html)
- xe PMU GT C6 residency: `perf stat -e xe_0000_00_02.0/gt-c6-residency,gt=0/ -I1000`, in ms. — [intel-xe patch "Add GT C6 events" v13](https://lists.freedesktop.org/archives/intel-xe/2025-January/064939.html); merged commit mirror [897286f2](https://tangled.org/tjh.dev/kernel/commit/897286f2948cefe5f9d37fc3148fc19d0c7b160c)
- qmassa / its exporter qmmd report, for xe devices: per-engine utilisation (bcs/ccs/rcs/vcs/vecs), actual and max GT frequencies, power for the `gpu` and `package` domains in watts, smem/vram memory and temperatures. The README example is a discrete Arc Pro B50. The README does not say whether iGPU power or Panther Lake is supported. — [qmassa GitHub](https://github.com/ulissesf/qmassa)
- Intel's xe-driver GPU table lists Arc B390 (Panther Lake) with initial support in kernel 6.13 and full support in Ubuntu 25.10+ / kernel 6.17. — [Intel dGPU docs: Xe KMD driver GPUs (via search)](https://dgpu-docs.intel.com/overview/supported-hardware/xe-driver-gpus.html)
- NPU sysfs (`/sys/bus/pci/drivers/intel_vpu/<bdf>/`):
  - `npu_busy_time_us` (kernel 6.11) counts time while jobs are submitted to firmware. Utilisation = Δbusy/Δwall, sampled about once per second; reading it more often "can affect job submission performance".
  - `npu_memory_utilization` (6.15).
  - `freq/current_freq`, `hw_min/max/efficient_freq`, `set_min/max_freq` (all kernel 7.2, April 2026; `current_freq` reads 0 when idle).
  — [kernel ABI sysfs-driver-ivpu](https://kernel.org/doc/Documentation/ABI/testing/sysfs-driver-ivpu); [original busy-time patch](https://lists.freedesktop.org/archives/dri-devel/2024-May/452990.html)
- RAPL powercap: zones under `/sys/class/power_cap/intel-rapl/intel-rapl:0` (`name` = package-0) with subzones `:0:0` core and `:0:1` uncore. `energy_uj` is a µJ counter with range `max_energy_range_uj`. There is no instantaneous `power_uw`, so power must be computed from energy deltas. — [kernel powercap docs](https://docs.kernel.org/power/powercap/powercap.html)
- Phoronix measures CPU power per test via PowerCap/RAPL on Linux. On the Panther Lake X7 358H it averaged about 15 W across about 300 workloads, peaking at 64 W. — [Phoronix X7 358H review](https://www.phoronix.com/review/intel-core-ultra-x7-358h-linux)
- Phoronix has tested the X9 388H on Ubuntu 26.04 with Linux 7.2 vs 7.3 Git and found no change in CPU power. X9 388H power envelope: 15 W min assured, 25 W base, 80 W max turbo. — [search summary of Phoronix/Notebookcheck coverage](https://www.phoronix.com/review/intel-core-ultra-x7-358h-linux) (the 7.3 article URL was not retrieved directly; treat as unverified)
- unitrace (intel/pti-gpu): `--device-timing` gives per-kernel device time. `--chrome-kernel-logging` / `--chrome-device-logging` produce timelines, and older flags are deprecated in favour of `--chrome-kernel-logging`. It supports Level Zero and OpenCL; OpenVINO's GPU plugin uses OpenCL by default. Sources conflict on the viewer: Intel's optimization guide says chrome://tracing, while the unitrace README says to use Perfetto rather than chrome://tracing. — [ALCF Aurora unitrace docs](https://docs.alcf.anl.gov/aurora/performance-tools/unitrace/); [intel/pti-gpu](https://github.com/intel/pti-gpu/blob/master/chapters/device_activity_tracing/LevelZero.md); [Intel oneAPI GPU optimization guide (PTI)](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/pti-gpu.html)

### Inferences
- Concrete monitoring recipe, run as root during each benchmark window and logged to CSV at 1 Hz:
  - iGPU busy %: `sudo perf list | grep xe_` to discover the PMU name and engines, then `sudo perf stat -I 1000 -x, -e xe_0000_00_02.0/engine-active-ticks,gt=0,engine_class=4,engine_instance=0/,xe_0000_00_02.0/engine-total-ticks,gt=0,engine_class=4,engine_instance=0/`. OpenVINO GPU inference normally runs on the compute (CCS, class 4 in xe uAPI) and/or render (RCS, class 0) engine, so check both. The PCI BDF and engine class numbers are unverified; confirm with `ls /sys/bus/event_source/devices/` and the PMU `format/` directory. Alternatives: `gputop` (igt ≥2.4) or `qmassa` for a live TUI on the demo screen.
  - iGPU frequency: `/sys/class/drm/card*/device/tile0/gt0/freq0/act_freq` (xe sysfs layout; unverified path).
  - NPU busy %: `B=/sys/bus/pci/drivers/intel_vpu/0000:00:0b.0; while :; do echo $(date +%s.%N),$(cat $B/npu_busy_time_us); sleep 1; done` (BDF unverified; `ls /sys/bus/pci/drivers/intel_vpu/`). On kernel 7.0, expect `freq/` to be missing, since it needs 7.2.
  - Package and platform energy: read `/sys/class/power_cap/intel-rapl:0/energy_uj` (package), plus `intel-rapl:1` if it is named `psys`, before and after the window. Handle wrap with `max_energy_range_uj`. Since the 2020 PLATYPUS mitigation, `energy_uj` is root-only on most distros (from memory; unverified here), so run the logger with sudo or `chmod` once.
  - `sudo turbostat --quiet --interval 1 --show PkgWatt,CorWatt,GFXWatt,RAMWatt,PkgTmp,Busy%,Bzy_MHz` gives per-second package, core and graphics watts plus temperature in one stream (column names from memory; verify on the box). Whether GFXWatt (PP1) is populated on Panther Lake is unknown. If it is absent, report package power only and say so.
  - CPU per-process or per-thread usage: `pidstat -t -u 1 -p <pid>`, or `perf stat -e task-clock,context-switches,cpu-migrations -p <pid>` for jitter diagnosis. Use `perf sched timehist` if control-loop jitter needs explaining.
- No reliable per-IP NPU power counter was found. RAPL package energy includes CPU, GPU, NPU and the SoC uncore together. The defensible metric is therefore system-on-chip energy per inference/task:
  - Measure idle package W over 30 s (P_idle).
  - Run N inferences over T seconds and measure ΔE.
  - Report total energy per inference = ΔE/N and, optionally, incremental energy = (ΔE − P_idle·T)/N.
  - Also report the task-level number: J per pick (ΔE over one full task episode).
  - For NPU-vs-GPU attribution, run the same model alone on each device and compare package ΔE per inference.
- Battery as a cross-check: `/sys/class/power_supply/BAT0/power_now` (or `current_now × voltage_now`) on battery gives whole-laptop draw including the screen. It is useful for a "runs X min on battery" claim but noisy. RAPL psys, where exposed, is the cleanest platform number.
- VTune: VTune's GPU Compute/Media Hotspots analysis supports Intel iGPUs and is the heavyweight option. For a hackathon, unitrace `--device-timing` + `--chrome-kernel-logging` on a short pi0.5 run (around 20 chunks) is faster to set up, and the Perfetto screenshot of one chunk (kernels laid out on the GPU timeline, with gaps showing host-side stalls) makes a strong "where the 467 ms went" visual. Both are unverified for Xe3 / Panther Lake on kernel 7.0.

### Gaps
- Not confirmed: whether qmassa or gputop reports iGPU power (hwmon energy) for Panther Lake integrated graphics. On xe, iGPU hwmon energy is generally only exposed for discrete cards; unverified.
- No source confirms that turbostat GFXWatt (PP1) or RAPL psys works on Panther Lake. Check on hardware.
- No Phoronix article specifically on Panther Lake Linux power measurement methodology for the iGPU or NPU was found, only RAPL CPU power.
- `nvtop` xe support status as of late 2026 is unverified (no source fetched).
- The kernel.org rendered xe_pmu docs page returned 404, so event names come from patch mails and a merged commit; confirm with `perf list` on 7.0.
- Version mismatch: the user specifies kernel 7.0, while NPU freq sysfs needs 7.2 and Phoronix's Ubuntu 26.04 X9 388H testing used 7.2/7.3. If the team runs a newer HWE/mainline kernel, more counters are available.

---

## Q3. Methodology: warmup, statistics, iteration counts, power/thermal control, per-stage closed-loop timing, task metrics

### Takeaway
Report two layers:
- (a) Isolated model benchmarks (benchmark_app / InferenceLatencyBenchmark) with warmup, at least a few hundred iterations, and median plus p90/p99, all on AC power, in a fixed power profile and at a recorded temperature.
- (b) In-loop per-stage timings from monotonic timestamps on every control cycle, giving p50/p90/p99 per stage, end-to-end latency, control-loop period jitter, task time, tasks/min, and success rate with a confidence interval over N trials.

Intel's own pi0.5 paper fixes workload parameters and TDP but does not publish its timing methodology. Beating it on transparency is cheap.

### Cited Findings
- benchmark_app warns that results depend on power settings, overclocking and thermal throttling, and recommends an environment "optimized for maximum performance". Warmup is on by default; skipping it is "not recommended". — [benchmark_app docs](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/get-started/learn-openvino/openvino-samples/benchmark-tool.rst)
- First inference time is reported separately from steady-state Median/Average/Min/Max. — [same](https://raw.githubusercontent.com/openvinotoolkit/openvino/master/docs/articles_en/get-started/learn-openvino/openvino-samples/benchmark-tool.rst)
- Intel's pi0.5 paper fixes the workload as DROID variant, 3 cameras at 224×224, 64-token language context, 10 denoising steps, BF16/FP16. It reports at Max TDP (65 W for X7 358H) and at an ISO-TDP of 40 W. It gives no warmup or iteration counts and no power-measurement tool. — [Intel OEP: Optimizing π0.5 VLA on Core Ultra Series 3](https://docs.openedgeplatform.intel.com/dev/_sources/OEP-articles/publications/optimizing-pi0.5-lva-model.md.txt)
- NPU busy-time sampling should be about 1 s to avoid perturbing job submission. The observer effect is real for the NPU counter. — [sysfs-driver-ivpu](https://kernel.org/doc/Documentation/ABI/testing/sysfs-driver-ivpu)
- physicalai benchmark defaults: 1 warmup iteration, up to 1000 iterations, max duration 60000. The README example uses 2 warmups and 100 iterations. — [OpenVINO Benchmark API ref (snippet)](https://docs.openvino.ai/2026/physical-ai/reference/benchmark-api.html); [physicalai README](https://github.com/openvinotoolkit/physicalai)

### Inferences
- Protocol checklist (put it on a backup slide):
  1. AC power connected, battery above 80%.
  2. Same power profile for before and after (`powerprofilesctl get` logged).
  3. Lid open on a hard surface, fan state noted.
  4. Record package temperature at start (turbostat PkgTmp); start each run below a fixed threshold, e.g. ≤50 °C, with a cooldown between runs.
  5. Close the browser and other GPU clients.
  6. Log `ov.get_version()`, the kernel (`uname -r`), the xe/NPU firmware and driver versions, and the compiled-model properties.
  7. Same inputs (recorded camera frames for the detector and VLA) for before and after.
- Iterations and warmup: discard the first 10–20 inferences per model (GPU kernel JIT, NPU firmware load, cache warm). Then:
  - Isolated: at least 300 iterations per configuration (enough for a stable p99 estimate: 99th percentile of 300 ≈ 3rd worst sample). For p99 claims prefer 1000. Also run the "after" config for at least 5 minutes to expose thermal throttling (plot latency vs time; a flat line is itself evidence).
  - Repeat each configuration 3 times on separate runs and report the median of medians with min–max across runs.
- Statistics: median is the headline. p90/p99 matter more for control because a late chunk stalls the arm. Also report max and the fraction of cycles that miss the deadline (e.g. chunk compute > chunk execution horizon). Avoid the mean alone, since the latency distributions are right-skewed.
- Per-stage instrumentation inside the closed loop:
  - Use `time.perf_counter_ns()` (CLOCK_MONOTONIC) at stage boundaries:
    - `t_cam_capture` (use the driver frame timestamp if available, e.g. V4L2 buffer timestamp)
    - `t_preproc_done`
    - `t_anomaly_done` (NPU)
    - `t_decision_done`
    - `t_vla_vision_done` / `t_vla_prefill_done` / `t_vla_denoise_done` (if the VLA is split into several compiled models)
    - `t_action_sent`
    - `t_arm_ack` (servo bus write return)
  - Write one row per cycle to a ring buffer and flush to CSV/Parquet at the end. Never print in the loop.
  - For async OpenVINO requests, record the callback time too, so queueing is separated from compute.
  - Derived metrics:
    - per-stage p50/p90/p99
    - photon-to-motion latency = t_action_sent − t_cam_capture
    - control-loop jitter = std and p99 of (period_i − nominal period) for the arm command loop (e.g. 30 Hz SO-101 command rate)
    - chunk staleness = time between observation capture and execution of the first action in that chunk (relevant if using Real-Time-Chunking-style async execution, which Intel ships as a sample: [Pi0.5 with RTC](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-suites/robotics-ai-suite/embodied/sample_pipelines/pi05_with_rtc.html))
- Optional trace-level view: emit the same stage spans as Chrome-trace JSON (`{"name","ph":"X","ts","dur","tid"}`) and load it in Perfetto next to a unitrace GPU trace. The host spans and GPU kernels then line up on one timeline.
- End-to-end task metrics:
  - task completion time (trigger to object placed, median of N)
  - tasks per minute
  - success rate k/N with a 95% Wilson interval. With N=20 and 18/20 successes the interval is roughly 70–97%, so state N honestly; N ≥ 20 per configuration is a reasonable hackathon floor.
  - detector precision/recall on a held-out defect set if the anomaly gate drives decisions
  - energy per task (RAPL ΔJ per episode)
- Separate "model latency" from "system latency". A 467 → X ms VLA improvement only matters if it shows up in photon-to-motion latency and task time. Show both, so judges can see the optimisation was not just a micro-benchmark.
- Contention test: run NPU anomaly + GPU VLA concurrently and compare each to its solo latency. On a shared-memory SoC (153 GB/s LPDDR5 class bandwidth per the Intel paper's X7 table) concurrency inflation is likely, and it is a nice "we measured it" point.

### Gaps
- No primary source was found specifying recommended iteration counts or percentile practice for robotics VLA benchmarking; the numbers above are standard practice, not sourced.
- Intel's pi0.5 paper does not disclose its power-measurement tool or timing procedure, so there is no Intel reference methodology to copy.

---

## Q4. Presentation: what judges value, slide/dashboard formats, Intel's pi0.5 paper as a model

### Takeaway
Map evidence one-to-one onto the rubric:
- (1) a device-mapping diagram with a measured 3-device table justifying NPU / iGPU / CPU placement
- (2) a before/after per-stage latency waterfall (467 ms PyTorch-XPU → OpenVINO), with p50 and p99
- (3) throughput / tasks-per-minute and energy per task
- (4) a live overlay of per-stage ms and device utilisation on the demo video

Intel's own pi0.5 paper presents results as an architecture diagram showing which sub-model is on which device, a hardware spec table, and bar charts of latency and perf/W at fixed TDP points. Copy that vocabulary: ISO-TDP, perf/W, device mapping.

### Cited Findings
- Intel's pi0.5 paper structure:
  - Figure 1 is an architecture diagram: Vision Encoder + Language Model on the iGPU, Action Expert on the NPU, with the per-layer KV cache as the only cross-device handoff via zero-copy USM-host tensors in shared DDR.
  - A hardware spec table.
  - Figure 2: "π0.5 model latency using PyTorch on Core Ultra X7 358H, Jetson AGX Orin, and Jetson Thor".
  - Figures 3–4: "Competitive analysis … at Max TDP" and "… at 40 W TDP".
  — [Intel OEP pi0.5 paper](https://docs.openedgeplatform.intel.com/dev/_sources/OEP-articles/publications/optimizing-pi0.5-lva-model.md.txt); [PDF](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf)
- Intel's headline numbers:
  - PyTorch on X7 358H at 40 W: 294 ms, vs Thor at full power 287 ms.
  - OpenVINO vs TensorRT at Max TDP: up to 1.5× lower latency than Orin and up to 1.3× better perf/W than both NVIDIA platforms.
  - At 40 W: up to 2.6× lower latency and 2.6× perf/W vs Orin.
  - The abstract and ISO-TDP section disagree on the Thor perf/W factor (1.3× vs 1.1×).
  — [same](https://docs.openedgeplatform.intel.com/dev/_sources/OEP-articles/publications/optimizing-pi0.5-lva-model.md.txt)
- Intel marketing framing: "50% faster performance on medium sized VLA versus Jetson Orin AGX", with a footnote on the 3-camera DROID setup at FP16. — [Intel robotics page (via search)](https://www.intel.com/content/www/us/en/products/details/robotics.html)
- Intel shipped OpenVINO Physical AI / Embodied Intelligence SDK pi0.5 pipelines: v26.0 OpenVINO pi0.5 on iGPU, v26.1 Pi0.5 with Real-Time Chunking. Reusing Intel's sample terminology signals alignment with the sponsor stack. — [Embodied Intelligence SDK release notes](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-suites/robotics-ai-suite/embodied/release-notes.html); [SiliconANGLE on the OpenVINO Physical AI launch](https://siliconangle.com/2026/05/31/intel-touts-130-plus-edge-design-wins-series-3-launches-openvino-physical-ai-framework/)

### Inferences
- Context for the team's baseline: Intel reports 294 ms for stock PyTorch at 40 W on the X7 358H, against the team's 467 ms PyTorch-XPU on the X9 388H. Judges who know Intel's paper may ask why. Be ready to explain the differences: camera count, resolution, denoise steps, precision, power mode, and the PyTorch version (Intel used PyTorch 2.12). Match Intel's workload parameters (3 cams at 224², 64 tokens, 10 denoise steps) when quoting the headline number, or state clearly how they differ.
- Recommended slides and visuals for a short demo:
  1. **Device-mapping diagram**: cameras → [NPU] Anomalib → [CPU] rules → [iGPU] pi0.5 (vision / prefill / denoise) → [CPU] arm driver. Annotate each box with its measured p50 ms and busy %, and add a small 3×N table showing why each model sits on its device (latency and J/inference on CPU/GPU/NPU). Intel's paper uses the same diagram style (Figure 1).
  2. **Before/after waterfall or stacked bar** of per-stage latency in the closed loop: baseline PyTorch-XPU 467 ms, then each optimisation step as a separate bar (e.g. OpenVINO FP16 → INT8/compressed weights → static shapes/caching → async/pipelining). Each step should be labelled with Δms, using p50 bars with p99 whiskers and the same y-axis throughout. An incremental "staircase" attributes the gains, which Intel's paper does not do. That is a differentiator.
  3. **Latency-over-time / CDF plot**: 5-minute run, before vs after, showing tails and thermal stability. A CDF with p50/p99 markers is a compact way to show jitter reduction.
  4. **Efficiency tile row**: chunks/s, tasks/min, J per chunk, J per task, package W (mean), success rate k/N with CI. Quote "ISO-power" comparisons (e.g. same power profile or PL1) in Intel's vocabulary.
  5. **Live overlay on the demo video**: draw per-stage ms (rolling p50 and last value), the device label, NPU/GPU busy % and package W onto the camera feed (OpenCV `putText`), or run a side panel with gputop/qmassa + a small matplotlib/Grafana strip chart. qmmd exports Prometheus metrics ([qmassa](https://github.com/ulissesf/qmassa)), which could feed a Grafana panel. That is overkill for a hackathon unless already set up.
  6. **Backup slides**: benchmark_app commands + JSON reports, the top-10 op hotspot table from `-pc`, the Perfetto/unitrace timeline of one chunk, and the measurement protocol checklist (Q3), so any number can be reproduced on request.
- For the live demo, keep one pre-recorded clip of the "before" (467 ms) behaviour side-by-side with the live "after" run. A visibly slower arm response is more persuasive than numbers alone.
- Present honest caveats (shared-RAPL attribution, single machine, N trials). Judges scoring "clarity of optimisation decisions" tend to reward clear reasoning about trade-offs, though no published judging write-up confirming this was found.

### Gaps
- No published judging write-ups or rubric commentary from Intel robotics/OpenVINO hackathons were found, and no source states what such judges specifically value beyond the rubric text supplied by the user.
- No public example of an Intel "good optimisation slide" or dashboard template was retrieved. The recommendations above are inferred from Intel's paper format and general practice.
- The PDF version of the pi0.5 paper was not opened, so figure visuals (bar style, units) are known only from captions.
