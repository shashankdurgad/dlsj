# MolmoAct2 on the XPS: export, profile, optimise, RTC

All scripts run on the XPS in the Studio library venv (commit 8f13241) with the cached weights:

```sh
cd ~/inference/physical-ai-studio/library
export HF_HOME=~/inference/hf-molmo HF_HUB_OFFLINE=1 TMPDIR=~/inference/tmp/onnx
PY=.venv/bin/python; D=~/inference/deploy/molmoact2
```

| file | what |
|---|---|
| `export.py` | Trace a MolmoAct2 (HF id or `--ckpt` Lightning checkpoint, LoRA merged by Studio) to OpenVINO. Options: `--steps K` (flow steps baked into the graph), `--text-len N` (static text length; default 256 of which ~98 used), `--image-size S` (needs a model trained at S), `--rtc` (real RTC, see below), `--int8 DIR` |
| `quantize.py` | Weight-compression variants of an FP16 export without re-tracing: `int8_sym`, `int8_asym`, `int8_keep_ae` (action expert FP16), `int8_keep_ae_vit`, `int4_lm` |
| `harness.py` | Shared evaluation: recorded cal2 frames from all 3 phases (cached), predictions, latency, agreement vs a reference, PASS criterion |
| `profile_ov.py` | Token accounting, wall split (pre/graph/post), per-op GPU profile grouped into vision / LM prefill / action expert |
| `optimize.py` | **One command after training**: builds the listed variants, tests them against the FP16/10-step reference of the same checkpoint, writes `report.md` and the recommended launch line |
| `rtc_patch.py` | RTC for MolmoAct2: export-time patch of the flow loop (+4 graph inputs) and the runtime wrapper the controller uses |
| `rtc_verify.py` | RTC checks: guidance ~0 equals the plain export; hand-off jumps with vs without guidance |
| `precision_compare.py` | The first INT8-vs-FP16 study (zero-shot) |

## After the LoRA fine-tune

```sh
$PY $D/optimize.py --name lora_v1 --ckpt /path/to/last.ckpt --prompts three_phase \
    --variants fp16_ref,int8_sym,int8_keep_ae,text128_int8,s5t128_int8,s5t128_rtc_int8
# -> ~/inference/exports/molmo_opt/lora_v1/report.md, recommended variant + launch line, e.g.
EXPORT=~/inference/exports/molmo_opt/lora_v1/s5t128_rtc_int8 ~/inference/deploy/smolvla/run_molmo.sh
$PY $D/rtc_verify.py ~/inference/exports/molmo_opt/lora_v1/s5t128_rtc_int8 \
    --plain ~/inference/exports/molmo_opt/lora_v1/s5t128_int8
```

`--prompts three_phase` uses the relabelled strings ("grasp the brick" / "lift the brick and hold it still in front
of the inspection camera" / "place the brick in the {bowl} bowl and release it"); pass a JSON for others. Run the
fine-tuned model's launch with the same strings (`run_molmo.sh` `--pick-prompt/--bring-prompt/--place-prompt`).

PASS criterion (harness.py): mean |variant - reference| over the first 10 actions <= 0.5 deg (joints) and <= 0.3
(gripper) — the INT8-vs-FP16 noise floor measured on 14 Nov was 0.26 / 0.08 — and the error vs the human demo not
worse than the reference by more than 0.2 deg. Text-length trimming should be exact (padding is masked); step
reductions and quantisation are measured.

## RTC

Studio's MolmoAct2 has no RTC (`rtc_enabled` is ignored at export). `export.py --rtc` patches the flow loop
(`rtc_patch.patch_policy`) and adds graph inputs `prev_chunk_left_over` (1, 30, 6, runtime units), `inference_delay`,
`execution_horizon` (int64) and `max_guidance_weight` (float32). The previous chunk is mapped into the model space
inside the graph (checkpoint frame + quantile normalisation). MolmoAct2 integrates t = 0 -> 1, so the guidance is
Studio's training-free RTC with x1 = x_t + (1 - t) v and v += w(t) W (prev - x1), w(t) Studio's schedule with
tau = t. The runtime preprocessors drop unknown keys, so `rtc_patch.MolmoRTCPolicy` adds the RTC inputs after
preprocessing; `deploy/smolvla/smolvla_control.py` picks it automatically for MolmoAct2 exports (RTC if the export
has the inputs, else crossfade). Chunk is 30, so `--rtc-horizon` <= 15.

## Notes

- One big model at a time; re-traces peak ~44 GB RAM: `optimize.py` runs them under `~/inference/tmp/heavy.lock`
  with a watchdog (kills below 6 GB free) and waits while a robot controller is running.
- The exported graph starts the flow from zeros (deterministic per input).
- First GPU compile of a new IR takes ~4 min (FP16) / ~12 s (INT8); later loads hit `~/inference/ov_cache`.
