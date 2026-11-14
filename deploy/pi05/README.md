# π0.5 → OpenVINO on the XPS

Export and benchmark scripts for π0.5 (3 cameras at 224², chunk 50) on the deploy laptop: Dell XPS 16,
Core Ultra X9 388H, Arc B390 iGPU on the `xe` kernel driver, 64 GB, Ubuntu 26.04, kernel 7.0.
Everything below was measured on that machine on 13 Nov 2026 (XPS clock) with `lerobot/pi05_base`
at revision `22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0`. Large artifacts stay on the XPS under
`~/inference/exports/`, and logs go to `~/inference/logs/pi05/`.

| File | Env | What it does |
|---|---|---|
| `studio_export.py` | Studio library `.venv` | `physicalai.policies.Pi05` → `policy.export(backend="openvino")`, plus Studio's INT8 hook, `--steps`, `--tokens` |
| `studio_bench.py` | Studio library `.venv` | Times `InferenceModel.predict_action_chunk` (the deployment runtime) and the OpenVINO call inside it |
| `ov_bench.py` | any env with `openvino` | Times one raw `model.xml` call (Studio or Intel IR) with synthetic static-shape inputs |
| `run_benchmarks.sh` | (calls the above) | The full latency matrix → `~/inference/logs/pi05/final_*.jsonl` |
| `summarize.py` | any python | JSONL → the Markdown table below |

## Optimised deployment of the fine-tuned checkpoint (14 Nov run)

See **`opt/README.md`**: latency *and* held-out accuracy for every variant. Recommended:
`~/inference/exports/pi05_junction_int8_tok64_5step` (INT8_SYM weights, 64-token prompt, 5 steps): 140 ms median runtime
call, k=10 action error within −1.2% of the FP16 10-step export. `studio_export.py --ckpt` now supports `--tokens` and `--steps`.

## Recommended path (for the fine-tuned checkpoint)

**Studio export + Studio INT8 hook, run with `InferenceModel` on `GPU`.** It is the judged stack, it bundles
the tokenizer, normalisation and noise into the export (`manifest.json`), it uses a newer OpenVINO, and on the
GPU it compiles to the same fused kernels as Intel's hand-written export (below). Then:

1. Train with `tokenizer_max_length: 64` in `training/pi05/full.json` (`model.init_args`). With our 6-d state,
   the longest instruction is a 42-token prompt (measured with the exported tokenizer). The pi05_base default of
   200 pads every prompt to 200 tokens. Setting the length at training time avoids a train/deploy mismatch.
2. Export it (Studio library env on the XPS; ~42 GB RAM peak, so nothing else heavy at the same time):
   ```bash
   cd ~/inference/physical-ai-studio/library && export TMPDIR=~/inference/tmp
   .venv/bin/python ~/inference/deploy/pi05/studio_export.py --ckpt <path>/last.ckpt \
       --out ~/inference/exports/pi05_ft_fp16
   .venv/bin/python ~/inference/deploy/pi05/studio_export.py \
       --compress_from ~/inference/exports/pi05_ft_fp16 --out ~/inference/exports/pi05_ft_int8
   .venv/bin/python ~/inference/deploy/pi05/studio_bench.py ~/inference/exports/pi05_ft_int8 --device GPU
   ```
   `--ckpt` follows Studio's documented `Pi05.load_from_checkpoint` API but is **untested**: no fine-tuned
   checkpoint existed yet. The equivalent CLI is `physicalai export --policy physicalai.policies.Pi05
   --ckpt_path <ckpt> --backend openvino --output_dir <dir>` (FP16 only; INT8 needs the Python hook).
3. Check INT8 accuracy on held-out episodes before using it on the robot (trajectory MSE or cosine against the
   FP16 export). The only parity number so far is Intel's script on pi05_base with dummy inputs (below).

## Environments on the XPS

| Env | OpenVINO | Used for |
|---|---|---|
| `~/inference/physical-ai-studio/library/.venv` (Studio @ `8f13241`, runtime `physicalai 0.4.1.dev5+g8381d43`) | 2026.4.1 | Studio export, `InferenceModel`, all headline numbers. I added `nncf==3.4.0` (= the `physicalai-train[nncf]` extra; additive only) |
| `~/inference/.venv-pi05ov` (new) | 2026.0 (Intel's pin) | Intel's `pi05-rtc-ov` (robotics-ai-suite `4544ddb`, LeRobot submodule `0b067df` + 9 patches), torch 2.7.1+cpu, transformers `fix/lerobot_openpi` (4.53.3), nncf 3.4.0 |
| `~/inference/.venv` | (none) | LeRobot PyTorch-XPU baseline, `bench/pi05_torch_xpu.py` |

Run Studio scripts with `.venv/bin/python`, not `uv run`: `uv run` syncs the project env first, which risks
dropping the `xpu` extra the env was built with (not tested, avoided). Recreate the Intel env:
```bash
cd ~/inference && git clone --depth 1 https://github.com/open-edge-platform/robotics-ai-suite.git
cd robotics-ai-suite/src/pipelines/pi05-rtc-ov && git submodule update --init lerobot && cd lerobot
git -c user.name=x -c user.email=x@x am --whitespace=fix ../patches/*.patch
UV_PROJECT_ENVIRONMENT=~/inference/.venv-pi05ov uv sync --python 3.12 --extra pi-ov
uv pip install --python ~/inference/.venv-pi05ov/bin/python nncf
```

## What each export produces

**Studio** (`studio_export.py`, `~/inference/exports/pi05_studio_{fp16,int8,int8_5step,int8_1step,int8_tok64}`):
`pi05.xml/.bin` (one static graph: SigLIP on all 3 cameras as one batch, Gemma-2B prefix, and the N-step
denoising loop unrolled with the 300M expert), `tokenizer.xml/.bin` (OpenVINO tokenizer, padded to the
prompt length) and `manifest.json` (runtime pipeline: quantile normalise → pi05 image prep → fresh noise →
tokenizer → model → denormalise). Inputs: `images [3,1,3,224,224] f32`, `image_masks [3,1]`,
`tokenized_prompt [1,200] i64`, `tokenized_prompt_mask [1,200]`, `state [1,32]`, `noise [1,50,32]`.
Output: `action [1,50,32]`. FP16 6.25 GB; INT8_SYM 3.13 GB (NNCF: 414/416 layers int8 per-channel).
`noise` is a graph input and the runtime draws fresh noise each call (avoids the GPU in-graph RNG repeat).
pi05_base has no dataset stats, so the manifest's normaliser is identity; a fine-tuned checkpoint carries its own.

**Intel** (`convert_ov_rtc.py -c 3 --compress_int8`, `~/inference/exports/intel_pi05/pi05_rtc_3c_{FP16,INT8}`):
`model.xml/.bin` only. Inputs: image (unnamed port `292`) `[3,1,3,224,224]`, `img_masks [3,1]`,
`lang_tokens [1,200]`, `lang_masks [1,200]`, `noise [1,50,32]`, `prev_chunk_left_over [1,50,32]`,
`prefix_weights [1,50,1]` (RTC guidance; zeros = off). Output `actions_out [1,50,32]`. INT8_ASYM. Tokenizer,
normalisation and state discretisation stay in LeRobot Python. Their script's own parity check on CPU
(pi05_base, dummy inputs, normalised action space): FP16 vs PyTorch max abs 6.0e-4 (cos 1.0); **INT8 vs PyTorch
max abs 0.108, MAE 0.0044, cos 0.99899**.

| Export cost | Wall | Peak RSS |
|---|---|---|
| Studio, 10 steps, FP16 | 8m46s | 41.7 GB (+13 GB ONNX temp file) |
| Studio, 1 step, FP16 | 3m05s | 41.7 GB |
| Studio, 10 steps, 64 tokens, FP16 | 9m58s (niced, under load) | 41.7 GB |
| Studio INT8 hook on an FP16 export | 21 s | 10.7 GB |
| Intel `convert_ov_rtc.py` (FP16 + INT8 + CPU parity run) | 13m39s | 42.7 GB |

## Kernel fusion check (`benchmark_app -pc`, GPU, INT8)

The two IRs differ before compilation: Studio's has 43 `ScaledDotProductAttention` + 181 bare `Softmax`
(the expert's attention inside the denoising loop is exported as matmul-softmax-matmul), Intel's has 222 SDPA +
2 Softmax. **After GPU compilation they are the same**: both run 224 `ocl::sdpa::opt` kernels (the GPU plugin
fuses Studio's pattern too), plus fused `rope::opt` (395) and `rms_gpu_bfyx_opt` (405). Time split (Studio INT8,
per-layer counters): INT8 GEMM `jit:gemm:any__i8` ~61%, SDPA 20%, dynamic activation quantisation 6%, the rest
under 4% each. No GatedMLP kernel appears; it is a debug-only fusion in 2026.x release builds. Commands:
```bash
B=~/inference/physical-ai-studio/library/.venv/bin/benchmark_app
$B -m ~/inference/exports/pi05_studio_int8/pi05.xml -d GPU -hint latency -niter 5 \
   -report_type detailed_counters -report_folder ~/inference/logs/pi05/pc_studio
```

## Failures and caveats

- **64-token prompt + pi05_base through the runtime fails.** π0.5 puts the discretised state in the prompt;
  pi05_base's 32-d state makes it 130–135 tokens (depends on the state values) and the exported tokenizer does not truncate:
  `RuntimeError: [GPU] The tensor size is not equal to model, can't set input tensor with index: 2, because model
  input (shape=[1,64]) and tensor (shape=[1,130]) are incompatible`. With our 6-d state the longest prompt is 42
  tokens, so 64 works for the fine-tuned model. The 64-token latency row is the raw IR only.
- **Studio leaks its ONNX external-data file** (`/tmp/tmp*.onnx.data`, 13 GB) after every export. `/tmp` is
  tmpfs on the XPS, so it holds RAM until deleted. `studio_export.py` deletes it; `TMPDIR=~/inference/tmp` keeps it on disk.
- **GPU+NPU split: not attempted.** Intel's `pi05-rtc-ov` has no split code (no "NPU" anywhere in its patches)
  and Studio exports one graph. External report (Nota, X7 358H, OV 2026.4): split 330 ms vs iGPU-only 250 ms.
- **Intel's own `benchmark_pi05_ov_rtc.py`** compiles without a latency hint or cache on OpenVINO 2026.0. It is
  listed in the table as measured, but `ov_bench.py` on the same IR is the fair comparison.
- Benign warnings: Studio `Missing keys ... target_time_mlp_{in,out}` (SnapFlow-only layers, unused when
  SnapFlow is off); LeRobot `Missing key ... embed_tokens.weight` (tied to `lm_head`).
- The 5-step and 1-step variants are speed-only. Without SnapFlow distillation their accuracy is unknown.
- INT8 accuracy has only been checked via Intel's dummy-input parity. Validate on real episodes.
- Every INT8 artifact here was made with `studio_export.py --compress_from`, which applies Studio's
  `compress_weights_openvino_int8_sym` to a saved FP16 export (the same thing the post-export hook does).
  The single-pass `--precision int8` option calls the hook inside `policy.export` and has not been run.

## Results (GPU = Arc B390 iGPU, 3 cameras 224², chunk 50, batch 1)

**Main table: run 3, 13 Nov 15:26–15:31 (XPS clock).** It was taken back to back with the PyTorch baseline
in one script (`BASELINE=1 run_benchmarks.sh 30`), at load average 2–4. The baseline reproduced at 469 ms,
against 467 ms measured on a quiet machine earlier, so the conditions are comparable. The only other job was
Desmond's `teleop_record` (idle preview, about 0.5 core), plus the Studio container. Latency is per 50-action
chunk, 30 iterations after 3 warmup calls. "Runtime" is
`InferenceModel.predict_action_chunk`: tokenizer, normalisation, image prep and fresh noise, plus the model, plus
denormalisation. It starts from 640×480 uint8 frames. "OV" is the OpenVINO call inside it.

| π0.5 config | Path | Precision | Median ms | p90 ms | Speed-up | Memory |
|---|---|---|---|---|---|---|
| PyTorch-XPU, 10 steps (baseline) | LeRobot `bench/pi05_torch_xpu.py` | bf16 | **469** | 471 | 1.0× | 8.9 GiB XPU peak |
| Studio export, 10 steps | runtime (OV 278) | FP16 | **283** | 285 | 1.66× | 6.2 GiB proc RSS |
| Studio export + INT8 hook, 10 steps | runtime (OV 208) | INT8_SYM w / FP16 act | **212** | 215 | 2.21× | 3.4 GiB proc RSS |
| Studio INT8, 5 steps ⚠ untested accuracy | runtime (OV 171) | INT8 | **175** | 177 | 2.68× | 3.3 GiB |
| Studio INT8, 1 step ⚠ needs SnapFlow distillation | runtime (OV 142) | INT8 | **147** | 149 | 3.19× | 3.3 GiB |
| Studio INT8, 10 steps, 64-token prompt | raw IR only (see failures) | INT8 | 194 (vs 208 at 200 tokens) | 210 | ≈ −7% | |
| Intel `pi05-rtc-ov`, 10 steps, OV 2026.4.1 | raw IR | INT8_ASYM | 212 | 239 | ≈ Studio INT8 | |
| Intel `pi05-rtc-ov`, 10 steps, OV 2026.0 (their pin) | raw IR | INT8_ASYM | 247 | 273 | | |
| Intel `pi05-rtc-ov`, their `benchmark_pi05_ov_rtc.py`, OV 2026.0 | LeRobot pre/post + OV | INT8_ASYM | 215 (mean) | n/a | | |
| Intel `pi05-rtc-ov`, 10 steps, OV 2026.4.1 | raw IR | FP16 | 312 | 328 | | |

Peak memory: the PyTorch figure is `torch.xpu.max_memory_allocated`. The OpenVINO figures are the benchmark
process's peak RSS. On the iGPU, OpenVINO's device allocations may not all appear in RSS, so treat them as
indicative. Compile time with the cache warm: 3.5–7.7 s. The first call after compile takes 0.4–2.6 s.

**Interleaved A/B (round 1 at load 2.7, raw IR, 30 iterations each, same 3 minutes):**

| Variant | Median ms | p90 ms |
|---|---|---|
| Studio INT8, default compile config (`PERFORMANCE_HINT=LATENCY`, `NUM_STREAMS=1`, `CACHE_DIR`) | 195.2 | 196.1 |
| + `INFERENCE_PRECISION_HINT=f16` | 195.3 | 197.4 |
| + `GPU_ENABLE_SDPA_OPTIMIZATION=NO` | 196.1 | 212.5 |
| Intel INT8 IR (same OV 2026.4.1) | 199.9 | 209.6 |
| Studio INT8, 64-token prompt | 179.0 | 180.7 |
| Studio FP16 | 277.0 | 281.8 |
| Studio INT8 through `InferenceModel` (OV-only 212.7) | 217.6 | 223.6 |

Conclusions from the paired runs:
- INT8 weight compression is the big OpenVINO win against FP16: −25 to −30% in the three lightly loaded runs
  (242→177, 283→212, 277→195 ms), but only −6 to −8% under heavy CPU load.
- Studio's and Intel's graphs are equally fast on the same OpenVINO. They compile to the same kernels (see the fusion check).
- OpenVINO 2026.4.1 beat 2026.0 on Intel's IR in all four paired runs, by 10–25%.
- The f16 precision hint and the SDPA-optimisation toggle make no measurable difference.
- A 64-token prompt saves 6–8% in the two clean runs.
- The runtime's pre/post-processing costs about 5 ms. Its OV call matched `ov_bench`'s raw
  `infer_request.infer` in run 3, and measured 18 ms slower in interleaved round 1.

**Variance warning.** The iGPU shares memory bandwidth with the CPU, so other work on the laptop moves
these numbers a lot. Studio INT8 through the runtime measured 177 ms (14:17, Anomalib training running), 276 and
285 ms (14:58 and 15:14, `teleop_record` encoding video at about 3 cores, load 7–17; the baseline measured 599 ms in the 15:14 run), 212 ms
(run 3) and 218/235 ms (interleaved rounds 1/2). Quote the run-3 table, and rerun `BASELINE=1 run_benchmarks.sh`
on an idle machine before the final OPTIMIZE slide. All raw JSON is in `~/inference/logs/pi05/`
(`results.jsonl`, `final_*.jsonl`, `interleaved.jsonl`).

Reference: Intel reports 172 ms per chunk for π0.5 on a Core Ultra X7 358H (3 cameras, 10 steps, OpenVINO + INT8).
