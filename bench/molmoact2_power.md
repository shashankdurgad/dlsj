# MolmoAct2 power and energy per action chunk: baseline vs now

Dell XPS 16 (Core Ultra X9 388H, Arc B390 iGPU), on AC, 14 Nov 2026 19:27-19:50. Energy from the Intel RAPL counters
(package-0 = CPU cores + iGPU + uncore; psys = whole platform), read by a root-only logger every 100 ms
(`/tmp/rapl_log.sh`, read-only; log in `bench/molmoact2_power/rapl.log.gz`) and matched to the wall-clock window of
each benchmark's timed calls (`baseline_bench.py` now records it). Same 60 cal2 frames as `bench/molmoact2_after.md`.

| | baseline: stock PyTorch, CPU, bf16 | now: fine-tuned 8k, OpenVINO, Arc GPU, INT8 + RTC | change |
|---|---|---|---|
| latency per 30-action chunk (median) | 13,746 ms | 313 ms | 44x faster |
| package power while running | 31.8 W (cores 27.6 W) | 38.7 W (iGPU/uncore 25.7 W, cores 6.5 W) | +22% |
| platform power (psys) while running | 42.2 W | 51.8 W | +23% |
| **energy per chunk, package** | **436.9 J** | **12.1 J** | **36x less** |
| energy per chunk, platform | 580.2 J | 16.2 J | 36x less |
| energy per chunk above idle, package | 397.2 J | 11.2 J | 35x less |
| energy per action (30 per chunk), package | 14.6 J | 0.40 J | |

Idle (60 s, nothing running, arm controller off): package 2.9 W, psys 11.5 W.

**Reading it**: the GPU path draws slightly more power while it runs (the iGPU is busy instead of 14 CPU cores), but
it finishes each chunk 44x sooner, so every action chunk costs 36x less energy. Running the policy continuously at the
deployed RTC rate (one chunk every 500 ms, each taking 313 ms) the GPU is busy ~63% of the time.

**Caveats**
- The baseline ran at 13.7 s per chunk here vs 10.8 s in the original baseline run (`bench/molmoact2_baseline.md`);
  this run came after other heavy work and its load guard waited 4 x 30 s for the machine to settle. Power is computed
  over only the stretch where the timed calls ran (the last n x median seconds of the window; the waits come before it),
  `power_busy.json`. `power.json` is the first pass over the whole window (diluted by the waits) and is superseded.
- Package power includes the iGPU (uncore) and the CPU cores; it does not include the NPU separately, the screen or the
  SSD. psys covers the whole platform.
- The package power limit is 200 W short-term / 65 W sustained (RAPL constraints); neither row hit it.
