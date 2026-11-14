# MolmoAct2-SO100_101: baseline vs optimised inference on the XPS

Zero-shot `allenai/MolmoAct2-SO100_101` (no fine-tune yet), Dell XPS 16 (Core Ultra X9 388H, Arc B390 iGPU, 64 GB),
14 Nov 2026. Same 60 recorded cal2 frames (3 phases, wrist + top cameras) for every row; `deploy/molmoact2/baseline_bench.py`.
Speed-up is relative to row 0. Quality: mean |difference| from the OpenVINO FP16 reference over a 30-step chunk, and
error against the recorded demonstrations (reference 8.47 deg).

| row | load s | median ms / chunk | p90 | p99 | speed-up | actions/s (30-step chunk) | peak RAM | vs FP16 ref (joints / gripper) | error vs demos |
|---|---|---|---|---|---|---|---|---|---|
| 0 **baseline**: stock Studio MolmoAct2, PyTorch CPU bf16 (256-token text, 10 steps) | 17.7 | **10,770** | 11,154 | 11,679 | 1.0x | 3 | 26.8 GB | 1.62 / 0.83 | 8.34 |
| 1 OpenVINO GPU FP16 | 8.4 | **362** | 388 | 391 | **29.8x** | 83 | 20.0 GB | 0 / 0 (reference) | 8.47 |
| 2 OpenVINO GPU INT8 (weight-only, sym) | 8.2 | **285** | 287 | 291 | **37.8x** | 105 | 5.9 GB | 0.24 / 0.11 | 8.48 |
| 3 OpenVINO GPU INT8 + RTC | 6.4 | **312** | 313 | 318 | **34.6x** | 96 | 5.9 GB | 1.99 / 0.64 * | 8.43 |

\* The RTC build draws different flow noise from the reference; noise alone moves outputs by 2.0-2.6 deg, so this is
within the model's own sampling spread (its error against the demos is unchanged).

**Control loop (RTC, 30 Hz, 30-step chunk, horizon 15):** a new chunk is requested every 15 ticks (500 ms) and arrives
after 312 ms (p99 318 ms): 100% of requests land before the action queue runs dry; the arm is never idle.

## Notes
- **Baseline device:** stock PyTorch MolmoAct2 on the Arc GPU (XPU, bf16) crashed the GPU (`UR_RESULT_ERROR_DEVICE_LOST`)
  on the third call and hung on retry; the XPS froze afterwards. The CPU is the only PyTorch configuration that runs, so it
  is the baseline. OpenVINO on the same iGPU is stable.
- **Thermals:** the GPU rows were re-timed after a 10-min cool-down, quiet-machine guard on (load < 3). Run straight after
  the CPU row (20 min at ~15 cores), the shared CPU/iGPU package was throttled and INT8 measured 538 ms; that run is in
  `molmoact2_baseline/results_cpu_and_first_gpu.json` for reference.
- **Power:** package energy counters need root on this machine; not measured.
- **Profile (INT8, 286 ms):** action expert 110 ms (10 flow steps x 36 layers), LLM prefill 106 ms, vision 53 ms, other
  20 ms; the 650-token prefix has 392 image tokens and 158 padding tokens (fixed by training at 128 text tokens, see
  `training/molmoact2/policy_text.py`).
- **Next rows after the LoRA fine-tune:** 128-token text and fewer flow steps (re-test on the fine-tuned model), with
  `deploy/molmoact2/optimize.py`.

Raw: `bench/molmoact2_baseline/` (row_torch-cpu-bf16.json, results_gpu_cooled.json, results_cpu_and_first_gpu.json).
