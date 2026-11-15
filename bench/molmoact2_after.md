# MolmoAct2 on the XPS: baseline vs now (after optimisation and fine-tuning)

Dell XPS 16 (Core Ultra X9 388H, Arc B390 iGPU, NPU, 64 GB), 14 Nov 2026. Same 60 recorded cal2 frames (grasp /
present / place, wrist + top cameras) for every row, `deploy/molmoact2/baseline_bench.py`, 50 timed warm calls per
row, load average < 3.5 before each timed block (no blocks redone). Raw: `bench/molmoact2_after/`, baseline raw:
`bench/molmoact2_baseline/`.

## Headline

| | baseline | now (deployed) | change |
|---|---|---|---|
| what | stock Physical AI Studio MolmoAct2, PyTorch, CPU, bf16, zero-shot | fine-tuned MolmoAct2 (LoRA, 8k steps), OpenVINO, Arc GPU, INT8 + RTC | |
| latency per 30-action chunk, median | **10,770 ms** | **313 ms** | **34.4x faster** |
| latency p99 | 11,679 ms | 317 ms | 36.8x |
| actions per second (30-action chunk) | 2.8 | 96 | |
| peak RAM | 26.8 GB | 5.9 GB | 4.5x less |
| error vs the human demos (joints, mean abs) | 8.34 deg | **2.80 deg** | 3.0x lower |
| control loop (30 Hz, RTC horizon 15) | a chunk takes 10.8 s: the arm waits | every chunk arrives before the action queue runs dry (p99 317 ms < 500 ms) | never idle |

## All rows

| row | median ms | p99 ms | speed-up | peak RAM | error vs demos |
|---|---|---|---|---|---|
| 0 baseline: stock PyTorch CPU bf16, zero-shot | 10,770 | 11,679 | 1.0x | 26.8 GB | 8.34 deg |
| 1 OpenVINO GPU FP16, zero-shot | 362 | 391 | 29.8x | 20.0 GB | 8.47 deg |
| 2 OpenVINO GPU INT8, zero-shot | 285 | 291 | 37.8x | 5.9 GB | 8.48 deg |
| 3 OpenVINO GPU INT8 + RTC, zero-shot | 312 | 318 | 34.6x | 5.9 GB | 8.43 deg |
| 4 **fine-tuned 8k, OpenVINO GPU INT8 + RTC (deployed)** | **313** | **317** | **34.4x** | 5.9 GB | **2.80 deg** |
| 5 fine-tuned 5.5k, OpenVINO GPU INT8 + RTC | 314 | 318 | 34.3x | 5.9 GB | 3.52 deg |
| 6 fine-tuned 8k wrist-edit (experimental), INT8 + RTC | 351 | 359 | 30.7x | 5.9 GB | 2.84 deg |

Rows 0-3: `bench/molmoact2_baseline.md` (same frames). Rows 4-6: this run (`bench/molmoact2_after/results.json`).

## What we did, and what each step bought

**Policy inference (MolmoAct2)**
1. **PyTorch -> OpenVINO on the Arc GPU** (`deploy/molmoact2/export.py`): 10,770 -> 362 ms. PyTorch on the GPU (XPU)
   crashed the device (`UR_RESULT_ERROR_DEVICE_LOST`) and froze the XPS, so the stock baseline is PyTorch on the CPU.
2. **INT8 weight-only quantisation** (`quantize.py`, symmetric): 362 -> 285 ms and 20 -> 5.9 GB; 0.24 deg from FP16,
   same error vs the demos.
3. **Real-time chunking (RTC)** (`rtc_patch.py`, in-graph guidance on the previous chunk): +27 ms per call, but the
   next chunk is computed while the arm executes the current one, so the 30 Hz loop never waits, and the jump at each
   chunk hand-off drops 2.62 -> 1.23 deg (24 real hand-offs).
4. **Fine-tuning (LoRA, 8k steps on our 409 cal2 demos)**: same graph and cost (313 vs 312 ms), error vs the demos
   8.4 -> 2.8 deg. 5.5k -> 8k steps: 3.52 -> 2.80 deg.
5. **Checked and already optimal** (no gain, documented): graph level (Studio hoists the KV context; OpenVINO constant-
   folds the step-invariant time conditioning, `deploy/molmoact2/graph_analysis/`), kernel level (all 720 expert
   attention calls use the fused `ocl::sdpa::opt` kernel), runtime settings (every alternative equal or slower).
6. **Not applied yet**: 5 instead of 10 denoising steps and 128 instead of 256 text tokens (the 650-token prefix has
   158 padding tokens). The zero-shot model shifted 1.5 / 1.07 deg with these, which is within its own noise spread,
   so they need checking on the fine-tuned model (`optimize.py` variants `steps5_int8`, `text128_int8`); the 8k
   model was not trained at 128 tokens.

**The rest of the closed loop (device split: VLA on the GPU, perception on the NPU, control on the CPU)**
- **Inspection on the NPU**: YOLO v2 classifier + one-class brick detector, ~6 ms per frame (1.1 ms model time); the
  GPU stays free for the policy.
- **Top-camera detector on the NPU and only when needed** (surveying / bringing): running it in the 30 Hz control
  process starved the Feetech servo reads (3 bus errors in 2 picks: timeout / checksum / parsing); none since.
- **Continuous spin**: 61 frames in 8.2 s (6.6 s roll + 1.6 s return) vs 12 frames in 11.2 s stepped; 5x the
  evidence in less time (the wrist servo lags its command, so the effective roll is ~23 deg/s, not the commanded 90).
- **No trip home between bricks**: the next brick is picked from above the plate.
- **End-to-end**: 20 sorted bricks today, cycle time median **35.3 s** (p10 26.5 s, p90 47.9 s) per brick
  (survey, pick, grasp check, bring, spin + YOLO, place).

## Caveats (stated, not hidden)
- The baseline row is the zero-shot model with its zero-shot prompts; the fine-tuned rows use the training prompts
  and top crop on the SAME 60 frames and the same recorded demo actions. Fine-tuning doesn't change the graph, so the
  latency comparison is like for like; the error comparison is each model with the inputs it is deployed with.
- The "vs FP16 reference" columns in `results.json` are meaningless for this run (placeholder reference: the
  zero-shot reference predictions were deleted in a disk clean-up on 14 Nov).
- Row 6 (wrist-edit) is 38 ms slower with an identical graph; it loads from the external SSD (exFAT). Not explained
  yet; re-check before relying on it.
- Thermals: rows run back to back after a quiet machine (load < 3.5). Throttled runs are slower (zero-shot INT8 measured
  538 ms straight after the 20-minute CPU baseline).
- Power: package energy counters need root on the XPS; not measured.
