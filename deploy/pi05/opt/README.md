# π0.5 OpenVINO optimisation on the XPS: latency and accuracy

Optimisation study for the team's fine-tuned π0.5 (`~/inference/models/pi05-junction-20261114/best.ckpt`,
action-expert-only fine-tune of `lerobot/pi05_base`) on the deploy laptop (Dell XPS 16, Core Ultra X9 388H,
Arc B390 iGPU, OpenVINO 2026.4.1, NNCF 3.4.0, Studio `8f13241`, runtime `physicalai 0.4.1.dev5+g8381d43`).
Every variant was measured for latency and for accuracy on held-out data. All numbers below were measured on
the XPS on 13 Nov 2026 (XPS clock, PST). Exports stay on the XPS under `~/inference/exports/`; logs, raw
JSON and predictions are under `~/inference/logs/pi05/opt/` and `~/inference/eval/`.

| File | What it does |
|---|---|
| `opt_lib.py` | Pure helpers (numpy only): validation split, frame sampling, targets, metrics, bootstrap, IR subgraph classifier. Tested in `tests/test_pi05_opt.py` |
| `eval_actions.py` | Accuracy harness: `cache` (decode held-out frames once), `run` (one export through `InferenceModel` with fixed noise; optional capture of the IR's input tensors for calibration), `report` |
| `quantize.py` | NNCF variants of the FP16 IR: `w8a8` (`nncf.quantize`), `int4` (VLM/vision INT4 + expert INT8), `int8`, `inspect` (what got quantised, per subgraph) |
| `bench_variants.py` | Interleaved latency benchmark of several exports (runtime call and OV-only), compile-config A/B, cold vs cached compile |
| `run_queue.sh` | The job queue used on the XPS (every heavy step under `flock ~/inference/tmp/heavy.lock`, exports wait for ≥45 GB free RAM) |
| `summarize_opt.py` | Joins the accuracy JSON and latency JSONL into the table below |

`../studio_export.py` gained working `--tokens` and `--steps` for fine-tuned checkpoints (`--ckpt`). Studio's
`Pi05Config` is a frozen dataclass, so the script swaps in a `dataclasses.replace` copy and rebuilds the
preprocessors, which set the trace sample's `[1, tokens]` prompt shape and the exported tokenizer spec.

## How accuracy is measured

- **Data:** Desmond's LeRobot v3 dataset `~/desmond/dlsj/data/duplo_teleop_rgb`, episodes 0–211 only (the frozen
  training snapshot: the first 212 episodes add up to its 47,079 frames). Held-out set = the 21 validation
  episodes of the training split, `sorted(random.Random(42).sample(range(212), 21))` =
  6 7 8 22 23 26 28 35 55 57 59 62 70 108 129 139 151 163 173 188 189 (identical to the run's
  `data-plan.json`). 301 frames sampled across every episode and every task segment in proportion to segment
  length, each with its own frame-level task string: 195 pick, 106 place (52 yellow, 34 purple, 20 orange).
  Video decoded once with `video_backend="pyav"` into `~/inference/eval/pi05_val300.npz`, so every variant sees
  identical inputs (640×480 uint8, resized by the runtime exactly as at deploy time).
- **Inference:** `physicalai.InferenceModel` (the deployment runtime) on `GPU`, with a fixed start noise per frame
  (`noise` passed in the observation, `default_rng(1234 + i)`), so variants differ only by the model.
- **Error vs recorded actions:** MAE/MSE between the predicted 50-step chunk and the recorded actions
  `[t, t+50)` in the dataset's joint units (degrees for the five joints, 0–100 for the gripper), over the first
  k=10 steps and all 50, masked past the episode end, per joint. The ΔMAE column is the paired change in k=10
  MAE against FP16 (same frames, same noise) with a 95% bootstrap interval (2,000 resamples).
- **Deviation vs the FP16 10-step reference** on the same inputs and noise: MAE, max abs error and the mean
  per-sample cosine of the mean-centred chunk (the raw cosine is ≈1 for any plausible chunk, because joint
  positions are far from zero).
- **Seed floor:** FP16 run again with different noise seeds. Its deviation from FP16 is the policy's own
  sampling spread, the yardstick for every other deviation.

**Acceptance threshold used:** a variant is accepted if (1) the upper end of the 95% interval of its paired
k=10 MAE change vs FP16 is at most **+5%**, and (2) its deviation MAE vs FP16 is below the seed floor.

## Results

Latency: `InferenceModel.predict_action_chunk` on the GPU (runtime call = tokenizer, normalisation, image prep,
model and denormalisation, from 640×480 uint8 frames), 40 timed calls after 5 warmup calls, with
`PERFORMANCE_HINT=LATENCY`, `NUM_STREAMS=1` and `CACHE_DIR`. **Main latency column: one round at 22:31–22:34**
(load average 1.5–2.9, Desmond's recorder stopped, only the Studio container and another agent's light jobs
running). The 3-step rows from that round are not usable (see the caveats below). "Eval" is the median over
the 300 calls of the accuracy run for the same export, a second, independent timing taken under different load.
The accuracy columns use 301 held-out frames, FP16 10-step / 200-token as the reference, and the same noise for
every variant. Error is in joint units (degrees; gripper 0–100). Jerk is the mean |second difference| of the chunk
(recorded demonstrations: 0.34). Range is the predicted range of motion over the recorded one.

| Variant (export in `~/inference/exports/`) | Runtime median / p90 ms | OV-only ms | Eval median ms | vs 467 ms | vs 188 ms | MAE k10 | ΔMAE k10 vs FP16 [95% CI] | MAE k50 | Dev vs FP16 MAE / max | cos (centred) | Jerk | Range | Size GB | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pi05_junction_fp16` (reference: 10 steps, 200 tokens) | 243 / 245 | 238 | 319 ¹ | 1.92× | 0.77× | 6.41 | 0 | 13.51 | 0 / 0 | 1.000 | 1.18 | 0.90 | 6.28 | ref |
| `pi05_junction_int8` (INT8_SYM weights, 10 steps) | 200 / 201 | 193 | 204 | 2.34× | 0.94× | 6.67 | +3.9% [+3.2, +4.7] | 13.61 | 1.33 / 26.7 | 0.979 | 2.85 | 0.98 | 3.16 | pass |
| `pi05_junction_fp16_tok64` | 211 / 211 | 205 | 244 | 2.22× | 0.89× | 6.41 | −0.0% (identical) | 13.51 | 0.012 / 0.22 | 1.000 | 1.18 | 0.90 | 6.28 | pass |
| `pi05_junction_int8_tok64` | 162 / 164 | 157 | 163 | 2.88× | 1.16× | 6.67 | +3.9% [+3.2, +4.7] | 13.61 | 1.33 / 27.1 | 0.979 | 2.85 | 0.98 | 3.16 | pass |
| `pi05_junction_fp16_tok64_5step` | 166 / 167 | 161 | 207 | 2.82× | 1.13× | 6.04 | −5.8% [−6.5, −5.1] | 12.98 | 1.22 / 17.6 | 0.980 | 0.98 | 0.86 | 6.28 | pass |
| **`pi05_junction_int8_tok64_5step`** (recommended) | **140 / 141** | 135 | 135 | **3.34×** | **1.34×** | 6.33 | −1.2% [−2.2, −0.3] | 13.13 | 1.85 / 33.7 | 0.959 | 3.16 | 0.96 | 3.16 | pass |
| `pi05_junction_fp16_tok64_3step` | (172) ² | (167) | n/a | | | 5.75 | −10.3% [−11.6, −8.9] | 12.47 | 2.58 / 38.0 | 0.924 | 0.84 | 0.82 | 6.28 | pass |
| `pi05_junction_int8_tok64_3step` | (200) ² | (195) | 123 | 3.80× ³ | 1.53× ³ | 6.04 | −5.6% [−7.0, −4.2] | 12.65 | 2.86 / 46.2 | 0.913 | 3.33 | 0.93 | 3.15 | pass |
| `pi05_junction_int8vlm` (INT8 vision+VLM, FP16 expert) | 229 / 231 | 224 | 228 | 2.04× | 0.82× | 6.66 | +3.8% [+3.1, +4.6] | 13.60 | 1.32 / 26.6 | 0.980 | 2.83 | 0.98 | 3.47 | pass (slower) |
| FP16 with other noise seeds (the seed floor) | | | | | | 6.30 | −1.6% [−5.9, +2.9] | 12.95 | **9.93 / 147** | 0.491 | 1.20 | 0.88 | | |

¹ First accuracy run (20:09), while another agent's PyTorch job used about 11 cores.
² From the round at 22:33, which ran into the iGPU memory exhaustion described below. They are slower than the 5-step rows, which
contradicts every other measurement, so treat them as invalid. Re-timing them is still to do (cancelled to free the XPS).
³ From the eval-time median (123 ms), not the benchmark round.

Other latency measurements of the same exports, all of the runtime call:
- 20:35, quiet machine, two rounds: FP16 249 and 257 ms, INT8 177 and 186 ms. The OV-only medians were 244/251 and 172/181 ms.
- The table above has INT8 at 200 ms. These two runs and the 188 ms starting point show the usual ±10% run-to-run spread on this
  laptop, because the iGPU shares memory bandwidth with the CPU.
- Speed-ups are computed from the main column and are indicative to about ±10%.

Per-segment k=10 MAE, FP16 → recommended: pick 5.69 → 5.64, place 7.85 → 7.71. Per joint (FP16 → recommended):
shoulder_pan 6.11 → 6.07, shoulder_lift 8.69 → 8.40, elbow_flex 8.36 → 8.26, wrist_flex 6.04 → 6.04,
wrist_roll 5.90 → 5.84, gripper 3.34 → 3.39. All per-joint values are in `~/inference/logs/pi05/opt/acc_v1.json`.

### Findings

1. **A 64-token prompt is free.** The outputs match the 200-token export to 0.012 MAE (padding tokens are masked out). It saves 33–38 ms
   at FP16/INT8 in the same round, and every variant should use it. The longest prompt measured with the exported tokenizer is
   48 tokens, over all 4 instructions × 2,008 state vectors including the extreme bins, so 64 leaves a 16-token margin.
2. **Fewer denoising steps do not hurt this checkpoint offline.** Its k=10 error against the recorded actions goes *down* with fewer steps:
   −5.8% at 5 steps and −10.3% at 3 steps in FP16, and the deviation from the 10-step chunks stays far below the seed floor.
   The range of motion also shrinks (0.90 → 0.86 → 0.82 of recorded). Fewer steps pull the samples towards the conditional
   mean, which scores well against a single recorded trajectory but may undershoot on the robot. That is why 5 steps is
   recommended over 3 until a closed-loop A/B is run.
3. **INT8 weight-only costs +3.9% k=10 MAE (95% CI +3.2…+4.7%) and makes chunks 2.4× jerkier.** The jerk goes from 1.18 to 2.85.
   Deviation from FP16 stays small (MAE 1.3 vs the 9.9 seed floor), and range of motion is kept (0.98). The jitter comes from
   quantising the **prefix**, not the expert: INT8 on vision+VLM with an FP16 expert has the same jerk and error, and is 30 ms slower.
   Disabling the GPU plugin's dynamic activation quantisation (`DYNAMIC_QUANTIZATION_GROUP_SIZE=0`) gave bit-identical
   outputs, so that is not the cause, or the property has no effect here. The robot's position controller low-pass filters
   step-to-step jitter of about 3 joint units. If it shows on the arm, `pi05_junction_fp16_tok64_5step` (166 ms, jerk 0.98, −5.8% error)
   is the smooth alternative at +26 ms.
4. **The variance is large:** the same export ranges from 177 to 204 ms across runs, depending on what else uses the CPU/memory.

### Recommendation

**`~/inference/exports/pi05_junction_int8_tok64_5step`**: INT8_SYM weights, 64-token prompt, 5 denoising steps.
- Latency: **140 ms median / 141 ms p90** for the runtime call (135 ms in the 300-call accuracy run). That is **3.3×** faster than the 467 ms
  PyTorch-XPU baseline and **1.34×** faster than the 188 ms starting point, or about 4.2 actions of delay at 30 fps.
- Accuracy: k=10 MAE −1.2% vs the FP16 10-step reference (CI −2.2…−0.3%). Deviation MAE 1.85 vs a 9.9 seed floor. Range of motion 0.96.
  It passes the gate (ΔMAE upper CI ≤ +5% and deviation below the seed floor).
- Load it exactly like the current export: `InferenceModel(path, backend="openvino", device="GPU", PERFORMANCE_HINT="LATENCY",
  NUM_STREAMS="1", CACHE_DIR=...)`. The manifest, tokenizer and normalisation stats come from the same checkpoint.
- Next options, which need a robot A/B, not just offline numbers: `pi05_junction_int8_tok64_3step` (≈123 ms, −5.6% error, smaller range 0.93) for speed,
  `pi05_junction_fp16_tok64_5step` (166 ms) if the INT8 jitter shows.
- These are offline open-loop metrics on held-out frames. **No robot rollout has been run with any variant.**

### Not done, failed, pending

- **W8A8 (`nncf.quantize`, static INT8 activations): abandoned, too slow and memory-hungry to calibrate on this machine.**
  The prefix-only run (`--ignore expert`, 256 training-split samples, preset MIXED, SmoothQuant on) needed **65 min** for the first
  statistics pass (about 15 s per sample on the CPU at nice 19, 21–45 GB RSS). The kernel OOM killer then killed it at 75 min,
  during quantiser setup, when a concurrent GPU benchmark pushed memory over (anon RSS 33.5 GB). An earlier attempt was
  OOM-killed in 1 minute at 59.5 GB by my calibration loader (`np.load(...)[k][i]` re-reads the whole array on every access, and the slices kept 420
  copies alive). That is fixed in `quantize.py`. Not run: whole-graph W8A8. Note that the GPU plugin already applies
  dynamic per-token INT8 activations to compressed-weight FCs with more than 64 rows (the prefix), per `research_notes/.../pi05_openvino_panther_lake.md`.
- **INT4 on the frozen VLM: not measured.** The first attempt (INT4_SYM g128 on vision+VLM) failed with
  `nncf.errors.InvalidGroupSizeError: Failed to apply group-wise quantization with group size value 128.`, because SigLIP's MLP is
  4304 wide, which is not a multiple of 128. The retry (INT4 on the Gemma VLM only, SigLIP + expert INT8: `run_queue.sh int4`) was queued, but it was
  lost when the laptop was power-cycled after the OOM freeze. Not retried.
- **Runtime/compile settings (f): not re-measured for this checkpoint.** On pi05_base, `INFERENCE_PRECISION_HINT=f16` and
  `GPU_ENABLE_SDPA_OPTIMIZATION=NO` made no difference (`../README.md`, interleaved A/B). Measured here: compile with a warm
  `CACHE_DIR` takes 2–17 s (INT8 2–10 s, FP16 7–17 s). The first inference after compile takes 0.4–17 s (16.6 s once, right after the INT8
  export was new to the cache). `bench_variants.py --compile_test` (no cache / cold / warm) is written but was not run.
- **The 3-step latency** from the benchmark round is invalid (see ²). Not re-timed: the re-run (one process per variant) was
  queued, killed by a reboot at 22:54, re-queued, and then cancelled to free the XPS for another test. A second benchmark round for
  every variant is also still to do. To finish: `bench_variants.py <exports> --rounds 2` (behind the heavy lock).
- Exports: the studio export's peak RSS is **53.6 GB** for the fine-tuned checkpoint (`/usr/bin/time`), not the 42 GB measured on pi05_base.
  The 10-step export took 10 min, the 5- and 3-step exports about 4 min each, and the INT8 hook 21 s.

### Incident: the laptop ran out of memory (13 Nov, 22:35–22:50)

`bench_variants.py` first loaded the variants one after another **in one process**. The compiled models' iGPU buffers were
not returned between loads, and they do not show in RSS: the coordinator saw about 40 GB held by the process. At the 10th load the
system went into an OOM storm, froze, and had to be power-cycled. All exports survived. `bench_variants.py` now runs every
measurement in its own subprocess, and waits for ≥25 GB available before each one. Anyone timing several exports should do the same:
one model per process.

## System-level settings (need sudo; NOT applied, manual steps)

None of these were changed for the measurements above. From `research_notes/XPS closed loop optimisation/system_tuning.md`;
measure p50/p99 before and after, and watch `freq0/throttle/reasons` for PL1/thermal throttling.

```bash
# Power profile (also drives Dell's fan/thermal mode); run on AC
powerprofilesctl set performance            # or: echo performance | sudo tee /sys/firmware/acpi/platform_profile
# CPU EPP / governor (power-profiles-daemon may override manual EPP writes)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
# Xe GPU floor frequency: raise min_freq to RPa (not RP0) to skip the GuC ramp at the start of each chunk
G=$(ls -d /sys/class/drm/card*/device/tile0/gt0 | head -1)
cat $G/freq0/{min_freq,max_freq,rpe_freq,rpa_freq,rp0_freq}
echo $(cat $G/freq0/rpa_freq) | sudo tee $G/freq0/min_freq
cat $G/freq0/throttle/reasons               # during a run
```
