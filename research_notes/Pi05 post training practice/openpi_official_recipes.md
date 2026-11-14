# Official openpi fine-tuning recipes for π0.5 (pi05)

Snapshot: openpi `main` at commit `215abfb` (2026-08-24, "docs(droid): fix config search instruction (#1023)"). That was still the latest commit when cloned on 2026-11-14. Code facts below come from reading that clone directly. Issue threads were read through the GitHub API on 2026-11-14.
Key files:
- Configs: [config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)
- Optimizer and LR schedule: [optimizer.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/optimizer.py)
- Model config: [pi0_config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py)
- Flow-matching model: [pi0.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)
- Prompt and state tokenization: [tokenizer.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py)
- Normalize and delta transforms: [transforms.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py)
- Norm stats: [normalize.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/shared/normalize.py) and [compute_norm_stats.py](https://github.com/Physical-Intelligence/openpi/blob/main/scripts/compute_norm_stats.py)
- LoRA variants: [gemma.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py)

## 1. Which pi05 configs exist and what are their hyperparameters?

### Takeaway
openpi ships six real pi05 `TrainConfig`s plus `debug_pi05`:
- `pi05_aloha` and `pi05_droid`, inference only
- `pi05_libero`
- `pi05_aloha_pen_uncap`
- `pi05_full_droid_finetune`
- `pi05_droid_finetune`

There is no official pi05 LoRA config. All pi05 configs inherit these global defaults:
- AdamW with b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, gradient clip 1.0
- Warmup plus cosine LR
- bf16 mixed precision
- 224×224 images
- 32-dim padded actions
- max_token_len 200
- Training noise time drawn from Beta(1.5, 1)
- 10 Euler denoising steps at inference

`pi05_libero` is the reference recipe: batch size 256, peak LR 5e-5 held almost constant, 10k warmup steps, EMA 0.999, 30k steps.

### Cited Findings
**Global `TrainConfig` defaults** ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)):
- `batch_size=32` (global batch)
- `num_train_steps=30_000`
- `ema_decay=0.99`
- `lr_schedule=CosineDecaySchedule()`, `optimizer=AdamW()`
- `freeze_filter=nnx.Nothing`, so everything is trained by default
- `seed=42`, `save_interval=1000`, `keep_period=5000`, `log_interval=100`, `num_workers=2`
- `pytorch_training_precision="bfloat16"`

**`CosineDecaySchedule` defaults** ([optimizer.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/optimizer.py)):
- `warmup_steps=1_000`, `peak_lr=2.5e-5`, `decay_steps=30_000`, `decay_lr=2.5e-6`
- Implemented with `optax.warmup_cosine_decay_schedule`, starting from `init_value=peak_lr/(warmup_steps+1)`.
- An `RsqrtDecaySchedule` also exists, with `peak_lr=5e-5` and `timescale=10_000`.

**`AdamW` defaults** ([optimizer.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/optimizer.py)):
- `b1=0.9`, `b2=0.95`, `eps=1e-8`, `clip_gradient_norm=1.0`, chained as `clip_by_global_norm` → `adamw`
- `weight_decay=1e-10`, with this code comment: "Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value."
- An SGD option also exists.

**`Pi0Config` defaults** ([pi0_config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py)):
- `dtype="bfloat16"`, `paligemma_variant="gemma_2b"`, `action_expert_variant="gemma_300m"`
- `action_dim=32`, `action_horizon=50`
- `max_token_len` is 200 when `pi05=True` (48 for pi0).
- `discrete_state_input` defaults to the value of `pi05`.
- The code lists two pi05 differences from pi0:
  - "the state input is part of the discrete language tokens rather than a continuous input"
  - "the action expert uses adaRMSNorm to inject the flow matching timestep"

**Image resolution:** `IMAGE_RESOLUTION = (224, 224)`, and every model transform applies `ResizeImages(224, 224)`. There are three camera slots: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` ([model.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/model.py), [config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)).

**Flow matching** ([pi0.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)):
- Training time is sampled as `time = beta(1.5, 1) * 0.999 + 0.001`.
- `x_t = t*noise + (1-t)*actions` and the target is `u_t = noise - actions`, using the convention that t=1 is noise.
- `sample_actions(..., num_steps=10)` uses Euler steps with `dt = -1/num_steps`.
- The PyTorch port also defaults to `num_steps=10` ([pi0_pytorch.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)).

**Precision** ([README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md)):
- JAX training defaults to mixed precision: weights and gradients in fp32, most activations in bf16. Setting `dtype` to float32 gives full fp32.
- PyTorch trains in full bf16 (the default) or full fp32. The README says bf16 "exhibits higher losses compared to float32".

**`pi05_libero`** ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)):
- Model: `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)`
- Data: `LeRobotLiberoDataConfig(repo_id="physical-intelligence/libero", prompt_from_task=True, extra_delta_transform=False)`
- `batch_size=256`
- `CosineDecaySchedule(warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5)`. Because `decay_lr` equals `peak_lr`, this is effectively warmup followed by a constant 5e-5.
- `AdamW(clip_gradient_norm=1.0)`, `ema_decay=0.999`, `num_train_steps=30_000`
- Initialized from `gs://openpi-assets/checkpoints/pi05_base/params`

**Released `pi05_libero` checkpoint at 30k steps** ([LIBERO README](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)):

| Suite | Success rate (%) |
|---|---|
| Spatial | 98.8 |
| Object | 98.2 |
| Goal | 98.0 |
| LIBERO-10 | 92.4 |
| Average | 96.85 |

**`pi05_aloha_pen_uncap`** ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)):
- `Pi0Config(pi05=True)`, so action_horizon stays at the default 50.
- Norm stats are reused from `gs://openpi-assets/checkpoints/pi05_base/assets` with asset_id `trossen`.
- `default_prompt="uncap the pen"`
- `num_train_steps=20_000`, `batch_size=64`
- Everything else uses the defaults: LR 2.5e-5 peak decaying to 2.5e-6 over 30k steps, EMA 0.99.
- ALOHA joints use delta actions by default (`use_delta_joint_actions=True`).

**`pi05_full_droid_finetune`** ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)):
- `Pi0Config(pi05=True, action_dim=32, action_horizon=16)`
- RLDS DROID data with `DroidActionSpace.JOINT_POSITION`, and norm stats from `pi05_base/assets/` with asset_id `droid`
- LR: warmup 1k steps, peak 5e-5, constant (decay_lr=5e-5, decay_steps=1M)
- `num_train_steps=100_000`, `batch_size=256`, `save_interval=5000`, `keep_period=10_000`
- `num_workers=0`, because the RLDS loader handles its own multiprocessing

**`pi05_droid_finetune`**, for a small custom DROID dataset ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)):
- `Pi0Config(pi05=True, action_dim=32, action_horizon=16)`, with the comment "pi05 is trained with 32-dim actions".
- Initialized from `pi05_droid/params`, not from base.
- The norm stats are explicitly reused from `pi05_droid/assets` (asset_id `droid`), with the comment "Important: reuse the original DROID norm stats during fine-tuning!"
- `num_train_steps=20_000`, `batch_size=32`, default LR and EMA.

**Inference-only configs** ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)):
- `pi05_droid`: `Pi0Config(action_horizon=15, pi05=True)`
- `pi05_aloha`: `Pi0Config(pi05=True)` with asset_id `trossen`
- `debug_pi05` uses dummy model variants.

**pi0-FAST configs only:** `max_token_len` rule of thumb is about 180 for single-arm robots and about 250 for bimanual. The pi05 default of 200 covers prompt plus discretized state, with no action tokens ([config.py comments](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)).

**Released checkpoints** ([README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md)):

| Checkpoint | Description |
|---|---|
| `pi05_base` | For fine-tuning |
| `pi05_libero` | For inference |
| `pi05_droid` | Inference or fine-tuning, "fine-tuned on DROID with knowledge insulation" |

The README update log says pi05 and PyTorch support were released in Sept 2025.

### Inferences
- PI's own pi05 recipes at batch size 256 (LIBERO, full DROID) use a constant LR of 5e-5 after warmup, not real cosine decay. The cosine decay to 2.5e-6 only applies to configs that keep the defaults (`pi05_aloha_pen_uncap`, `pi05_droid_finetune`).
- The action horizon is set per robot or dataset: 10 for LIBERO, 15 to 16 for DROID, 50 (the default) for ALOHA. The model was pretrained with 50 per the π0.5 paper ([arXiv 2504.16054](https://arxiv.org/pdf/2504.16054)).

### Gaps
- There is no official per-config statement of wall-clock time for `pi05_libero`.
- PI has not said which settings beyond the config produced the released `pi05_libero` checkpoint.

## 2. Full fine-tune vs LoRA vs freezing: what is supported, memory needs, quality

### Takeaway
openpi supports full fine-tuning (the default) and LoRA through the `gemma_2b_lora` and `gemma_300m_lora` variants plus `get_freeze_filter()`. LoRA is JAX only, not PyTorch.

Official memory guidance:
- More than 22.5 GB for LoRA, e.g. an RTX 4090
- More than 70 GB for full fine-tuning, e.g. an A100 80GB or H100
- More than 8 GB for inference

There is no official pi05 LoRA config, and PI has published no LoRA-vs-full quality comparison. Community reports since 2025 repeatedly find that default-rank LoRA on pi05 underperforms in closed loop even when loss matches.

### Cited Findings
**Memory and multi-GPU**
- The README memory table: "Inference > 8 GB (RTX 4090); Fine-Tuning (LoRA) > 22.5 GB (RTX 4090); Fine-Tuning (Full) > 70 GB (A100 80GB / H100)". `fsdp_devices` reduces per-GPU memory. JAX training is single node only, while PyTorch supports multi-node through torchrun ([README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md)).
- If training runs out of memory, the README says to set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`, use `--fsdp-devices <n>`, and "consider disabling EMA" ([README troubleshooting](https://github.com/Physical-Intelligence/openpi/blob/main/README.md)).

**LoRA support**
- The PyTorch port does not support FSDP, mixed precision, LoRA, or EMA ([README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md)).
- LoRA ranks ([gemma.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py)):
  - `gemma_2b_lora` has rank 16 and alpha 16 on both attention and FFN.
  - `gemma_300m_lora` has rank 32 and alpha 32 on both attention and FFN.
- How `get_freeze_filter()` works ([pi0_config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py)):
  - With a LoRA PaliGemma variant, it freezes `.*llm.*` params. If the action expert is not LoRA, it excludes the action expert (`.*llm.*_1.*`) from freezing, so the expert stays fully trainable.
  - All `.*lora.*` params are always trainable.
  - The SigLIP vision encoder is not matched by `.*llm.*`, so it stays trainable under this filter.
- The only official LoRA examples are pi0 configs, `pi0_libero_low_mem_finetune` and `pi0_fast_libero_low_mem_finetune`. Both set `ema_decay=None` with the comment "Turn off EMA for LoRA finetuning", keep 30k steps, and keep the default LR ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)).
- Community users confirm there is no pi05 LoRA config and build one by copying the pi0 pattern: `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")` plus a matching `freeze_filter` ([issue #672](https://github.com/Physical-Intelligence/openpi/issues/672), [#974](https://github.com/Physical-Intelligence/openpi/issues/974)). This is a community pattern, not an official one.

**Freezing parts of the model**
- No action-expert-only config exists. `freeze_filter` can freeze arbitrary path regexes.
- Asked about KI-style freezing, maintainer kpertsch said fine-tuning "likely will work a lot worse if you freeze the full VLM backbone and only finetune the action expert (you can try though)" ([issue #649](https://github.com/Physical-Intelligence/openpi/issues/649)).
- Known inefficiency: when `freeze_filter` is set and EMA is left on, `train.py` keeps an EMA copy of the whole parameter tree, frozen weights included. That wastes memory, and the fix was still an open issue as of 2026-09 ([issue #1054](https://github.com/Physical-Intelligence/openpi/issues/1054)).

**Community quality reports (not official)**
- [Issue #1060](https://github.com/Physical-Intelligence/openpi/issues/1060), 2026-10-04, openpi `215abfb`, simulated UR3 scooping, 100 expert episodes, 10k steps, batch size 32:
  - pi05 LoRA with default ranks reached the same training loss (about 0.0024) and the same open-loop error as full fine-tuning.
  - In closed loop, LoRA scored 0/150 in two separate runs while full fine-tuning scored 60/150.
  - Confound flagged in replies: the LoRA run used peak LR 2.5e-5 with no EMA, while the full run used 5e-5 with EMA 0.999.
- Another commenter on #1060 (PrismaX) reports that r=128 LoRA with a frozen vision encoder nearly matched full fine-tuning on a simulated YAM pick-and-place: 84% vs 89% pickup, 59% vs 53% dropoff. The LoRA model was more robust to a 300 ms delay between chunks and to clutter ([issue #1060](https://github.com/Physical-Intelligence/openpi/issues/1060)).
- [Issue #711](https://github.com/Physical-Intelligence/openpi/issues/711): pi05 LoRA on LIBERO with batch size 64 and LR 1.25e-5 gave about 1% success on Spatial. Another user said they "am not getting good performance" with pi05_base LoRA on LIBERO.
- [Issue #906](https://github.com/Physical-Intelligence/openpi/issues/906): pi05 LoRA on a real Franka reached low loss but failed to pick. Replies blame norm-stat mismatch, chunk timing, and latency; one user found several hundred ms of camera-pipe latency.
- [Issue #850](https://github.com/Physical-Intelligence/openpi/issues/850): pi05 LoRA ran at about 16 s/iteration at batch size 32 on an RTX 5880 Ada (48 GB). At batch size 16, 60k steps took about 44 hours on one GPU.

### Inferences
- If you have at least 80 GB of GPU memory, full fine-tuning is the officially exercised path: every pi05 config PI ships is a full fine-tune. Default-rank LoRA on pi05 is an unvalidated community recipe with several failure reports.
- When comparing LoRA to full fine-tuning, match LR and EMA, and judge in closed loop, since loss and open-loop MSE did not separate them in #1060.

### Gaps
- PI has published no measured GPU-GB figures specific to pi05 (the README table is generic across models) and no pi05 LoRA vs full benchmark.

## 3. Knowledge insulation and FAST-token co-training in post-training

### Takeaway
The pi05 checkpoints were pretrained with KI in a single stage. openpi does not implement KI or FAST plus flow-matching co-training for fine-tuning: fine-tuning uses only the flow-matching loss, with gradients going into both the action expert and the VLM backbone. PI says it has not seen conclusive evidence that KI helps during fine-tuning.

### Cited Findings
- The README says pi05 is "trained with knowledge insulation". It also says: "in this repository, we currently only support the flow matching head for both π0.5 training and inference" ([README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md)).
- kpertsch (PI) answered in [issue #649](https://github.com/Physical-Intelligence/openpi/issues/649) (2025-09):
  - "openpi doesn't currently implement knowledge insulation for finetuning (or combined FAST + action expert flow matching losses for that matter)... gradients get propagated through *both* the action expert and the VLM backbone."
  - "we use KI during pre-training for the pi05 checkpoints."
  - KI in fine-tuning "will likely work much better if you still update the VLM backbone using FAST token targets."
  - "openpi doesn't implement this atm, and we haven't seen conclusive evidence that it actually helps during finetuning."
- kvablack (PI) in [issue #635](https://github.com/Physical-Intelligence/openpi/issues/635): "we currently only support the flow matching head for both training and inference". Subtask prediction and next-token prediction are not released.
- kpertsch in [issue #650](https://github.com/Physical-Intelligence/openpi/issues/650): "The released pi05 checkpoints are actually trained with knowledge insulation in a single stage training process."
- How KI works, per the KI paper (Driess et al., arXiv 2505.23705, 29 May 2025) ([arXiv HTML](https://arxiv.org/html/2505.23705)):
  - The VLM backbone is trained with next-token prediction on FAST-discretized actions plus VLM data (captioning, VQA, localization) and planning data.
  - The flow-matching action expert, about 300M parameters, is trained on continuous actions.
  - The method has to "stop the gradient flow from the action expert to the pre-trained weights."
  - Training speed: it trains "as quickly as" π0-FAST, while π0 needs "7.5 times as many training steps" for similar performance on table bussing. Joint training adds about 20% compute per step.
  - Action horizon H=50.
- KI paper LIBERO results for the model "finetuned on LIBERO from the generalist" ([arXiv HTML](https://arxiv.org/html/2505.23705)):

  | Suite | Success rate (%) |
  |---|---|
  | Spatial | 98.0 |
  | Object | 97.8 |
  | Goal | 95.6 |
  | LIBERO-10 | 85.8 |
  | LIBERO-90 | 96.0 |

  The paper does not say whether the KI losses were active during that fine-tune.
- The README describes `pi05_droid` as fine-tuned on DROID "with knowledge insulation". That run was internal, not the openpi `pi05_full_droid_finetune` recipe, which uses flow matching only ([README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md)).

### Inferences
- For openpi users, KI is irrelevant at fine-tune time unless they implement it themselves. The practical consequence is that full fine-tuning updates the VLM through the flow-matching gradient, which is the very thing KI was designed to avoid in pretraining. This is a plausible source of language-following degradation on small datasets, but that is unverified.

### Gaps
- No public ablation of KI vs non-KI fine-tuning exists.
- No step counts or hyperparameters for PI's internal KI fine-tunes have been published.

## 4. Normalization: quantile vs mean/std, compute_norm_stats, pitfalls

### Takeaway
pi05 (and pi0-FAST) use quantile normalization to [-1, 1] based on the 1st and 99th percentiles. pi0 uses z-score. Stats are computed per dimension for `state` and `actions` after the repack and data transforms, so they are already in delta space if `DeltaActions` is used. Training refuses to start without them.

Official advice:
- Try both reusing PI's pretraining stats for a known robot and computing fresh stats.
- Always reuse the DROID stats when fine-tuning `pi05_droid`.
- Watch for tiny q01/q99 ranges on rarely used dimensions.

### Cited Findings
- `create_base_config` sets `use_quantile_norm = model_type != PI0`, so pi05 and pi0-FAST use quantiles ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)).
- Quantile normalization is `(x - q01) / (q99 - q01 + 1e-6) * 2 - 1`, and z-score is `(x - mean)/(std + 1e-6)`. Stats are sliced to the data's dimension before padding ([transforms.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py)).
- `RunningStats` computes mean, std, q01 and q99 using 5000-bin histograms built on the fly ([normalize.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/shared/normalize.py)).
- How `compute_norm_stats.py` works ([compute_norm_stats.py](https://github.com/Physical-Intelligence/openpi/blob/main/scripts/compute_norm_stats.py)):
  - It applies `repack_transforms.inputs` and `data_transforms.inputs` (which include DeltaActions when enabled), then accumulates stats over the keys `state` and `actions`.
  - Output is written to `assets/<config>/<repo_id>`.
  - `--max-frames` subsamples the data; the full-DROID instructions use 10M frames ([DROID training README](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md)).
- The data loader raises "Normalization stats not found. Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`" when stats are missing ([data_loader.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/data_loader.py)). At inference, `create_trained_policy` applies the same Normalize and Unnormalize with `use_quantiles` ([policy_config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/policy_config.py)).
- Reloading pretraining stats ([docs/norm_stats.md](https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md)):
  - Reuse is done with `AssetsConfig(assets_dir=".../pi0_base/assets", asset_id="trossen")`.
  - Available asset IDs: `trossen`, `trossen_mobile`, `droid`, `franka`, `ur5e`, `ur5e_dual`, `arx`, `arx_mobile`, `fibocom_mobile`.
  - The doc says reuse "can lead to better performance" if your robot follows PI's action-space conventions, and recommends "always try both".
  - PI action conventions:
    - Joints in radians.
    - Gripper in [0, 1], where 0 is open and 1 is closed.
    - Control at 20 Hz for UR5e and Franka, 50 Hz for ARX and Trossen.
    - DROID uses joint velocity at 15 Hz.
    - 7-DoF arms use dimensions 0–6 for joints and dimension 7 for the gripper.
- The doc's table only names `pi0_base` and `pi0_fast_base` asset directories, but the pi05 configs load from `gs://openpi-assets/checkpoints/pi05_base/assets` (`trossen`, `droid`) and `pi05_droid/assets` ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)).
- README troubleshooting for "Diverging training loss": "Check the q01, q99, and std values in norm_stats.json... rarely used [dimensions] can end up with very small q01, q99, or std values, leading to huge states and actions after normalization. You can manually adjust the norm stats as a workaround" ([README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md)).
- Community advice (not official): print which norm-stats file is loaded at eval, since a mismatch "fails quietly and only shows up as lower success" ([#711](https://github.com/Physical-Intelligence/openpi/issues/711)). Also compare raw observations at inference to the training distribution ([#906](https://github.com/Physical-Intelligence/openpi/issues/906)).
- Community report: gripper values encoded as 0/0.3 predicted poorly. A reply suggests matching the 0/1 pretraining convention ([#850](https://github.com/Physical-Intelligence/openpi/issues/850)).

### Inferences
- Since pi05 discretizes the normalized state into 256 bins over [-1, 1] (see section 5), quantile stats directly set the state-token resolution. Values outside [q01, q99] saturate into the edge bins. Bad stats therefore damage both actions and state tokens.

### Gaps
- No official guidance on per-timestep normalization ([#594](https://github.com/Physical-Intelligence/openpi/issues/594) unanswered).
- PI has not published a full list of the asset IDs available under `pi05_base/assets`.

## 5. Delta vs absolute actions, state input, prompt format

### Takeaway
- **Delta actions.** `DeltaActions(mask)` subtracts the current state from the masked action dimensions over the whole chunk, so each action is relative to the chunk's first state. Grippers stay absolute.
  - ALOHA mask: `make_bool_mask(6, -1, 6, -1)`, meaning joints are delta and both grippers absolute.
  - LIBERO `extra_delta_transform` mask: `make_bool_mask(6, -1)`.
  - DROID: no delta, because the actions are velocities, or joint positions in the full-DROID RLDS recipe.
- **State input.** pi05 feeds the state as discretized text tokens in the VLM prefix. The prompt reads `"Task: {text}, State: {256-bin ints};\nAction: "`.
- **The pi05_libero exception.** `pi05_libero` sets `discrete_state_input=False`, so that model gets no state at all. PI has not explained why.

### Cited Findings
- `DeltaActions`: `actions[..., :dims] -= expand_dims(where(mask, state[..., :dims], 0), axis=-2)`. `AbsoluteActions` is the inverse and is applied to outputs at inference ([transforms.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py)).
- The LIBERO config comment ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)) says:
  - "pi0 models are trained on delta actions (relative to the first state in each action chunk). IF your data has absolute actions... convert... The only exception is for the gripper actions which are always absolute."
  - LIBERO raw actions are already deltas, so `extra_delta_transform=False` for pi05. Older pi0 checkpoints used the extra transform.
- The ALOHA config sets `use_delta_joint_actions=True` by default with "Gripper dimensions will remain in absolute values", and `adapt_to_pi=True` converts standard ALOHA data to PI's internal runtime space ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)).
- The DROID config comment: "We assume joint *velocity* actions, so we should *not* apply an additional delta transform." `pi05_full_droid_finetune` uses `DroidActionSpace.JOINT_POSITION` ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)).
- Tokenization ([tokenizer.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py)):
  - The prompt is cleaned with `strip()`, `_` replaced by a space, and `\n` replaced by a space.
  - The state is discretized as `np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1`, giving 256 bins.
  - Full prompt: `f"Task: {cleaned_text}, State: {state_str};\nAction: "`.
  - When there is no discrete state, the format is pi0-style: BOS + text + `\n`.
- Padding: `PadStatesAndActions(model_config.action_dim)` pads state and actions to 32. `prompt_from_task=True` reads the LeRobot `task` field as the prompt and is "the recommended setting". `default_prompt` is injected when no prompt is given ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)).
- On state placement, kpertsch confirmed in [#650](https://github.com/Physical-Intelligence/openpi/issues/650) that the state goes through the VLM as text, not into the action expert. In `pi0.py`, the continuous state token in the suffix is only added `if not self.pi05` ([pi0.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)).
- [Issue #687](https://github.com/Physical-Intelligence/openpi/issues/687), "pi05-libero discrete_state_input = False", has 18 comments asking why. As of Nov 2026 no maintainer has answered. A commenter warns that flipping the flag relative to the checkpoint "shows up only as lower success, with no error."

### Inferences
- With `pi05=True` and `discrete_state_input=False`, the model is effectively state-blind: there is no continuous state path in pi05. For custom robots, keep the default `discrete_state_input=True` unless you deliberately want that.
- Discretized state takes about 8 to 14 numbers plus separators in the token budget for a single- or dual-arm robot. That fits within 200 tokens alongside short prompts.

### Gaps
- PI has given no rationale for LIBERO's state-free setting and no ablation of discrete vs no state in fine-tuning.

## 6. Demonstrations, steps, and training time

### Takeaway
Official data points:
- About 100 demonstrations is a starting point for a low-variation task (kpertsch, about pi0-FAST LoRA).
- An example custom DROID dataset of 30 demonstrations.
- 20k to 30k steps for small-dataset fine-tunes.
- 100k steps at batch size 256, about 2 days on 8×H100 (about 1 epoch), for full DROID.

### Cited Findings
- kpertsch in [#491](https://github.com/Physical-Intelligence/openpi/issues/491) (2025-05): "100 demonstrations for a task with little variation could be a good place to start" (asked for pi0-FAST LoRA on a new robot).
- From the DROID training README ([examples/droid/README_train.md](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md)):
  - Full DROID: "approximately 2 days on 8x H100 GPUs for convergence (100k iterations, bs256, approx. 1 epoch)". Starting from PaliGemma instead takes "~5 days on 8x H100s (240k iterations, i.e. 3 epochs)".
  - The custom DROID example uses "a subset of just 30 demonstrations". Datasets under "10s of hours" should use LeRobot format; larger ones should use RLDS.
  - An idle-timestep filter "significantly improve[s] policy performance" on DROID.
- The pi0-FAST full-DROID config comment says "100k steps should be sufficient, takes ~2 days on 8x H100s" ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)).
- Official step counts: `pi05_libero` 30k steps at batch size 256; `pi05_aloha_pen_uncap` 20k at 64; `pi05_droid_finetune` 20k at 32. The README serving example uses the step-20,000 checkpoint ([config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py), [README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md)).
- Community data points:
  - 105 episodes of about 40 s each (about 93k frames): pi05 LoRA, 60k steps at batch size 16, about 44 h on one RTX 5880 Ada ([#850](https://github.com/Physical-Intelligence/openpi/issues/850)).
  - 100 scripted episodes (105k frames): 10k steps at batch size 32, full fine-tune on one H100 ([#1060](https://github.com/Physical-Intelligence/openpi/issues/1060)).

### Gaps
- PI has not published a demos-vs-success curve for pi05 or a measured wall-clock time for `pi05_libero` on a given GPU.
- The π0.5 paper's internal post-training step counts were not found in sources I could access.

## 7. Real-time chunking, inference steps, and their interaction with fine-tuning

### Takeaway
openpi's default inference is synchronous, with 10 Euler denoising steps. The repo has no real-time chunking (RTC) implementation; asynchronous RTC is an open feature request. The RTC paper says RTC is inference-time only, needs no retraining, and works with any flow or diffusion VLA. Action horizon and execution horizon are set by the fine-tune config and the client.

### Cited Findings
- `sample_actions(num_steps=10)` is the default in both the JAX and PyTorch ports ([pi0.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)). I found no RTC or inference-delay code anywhere in `src/openpi` at commit `215abfb` (searched in the clone).
- [Issue #636](https://github.com/Physical-Intelligence/openpi/issues/636), "Are there any plans to add asynchronous Realtime chunking inference": no maintainer reply.
- The RTC paper, "Real-Time Execution of Action Chunking Flow Policies" (arXiv 2506.07339, June 2025, v2 Dec 2025, NeurIPS 2025) ([arXiv](https://arxiv.org/abs/2506.07339)):
  - It is an inference-time method: it "freezes" the actions that are guaranteed to execute and inpaints the rest while the current chunk runs.
  - It needs no retraining and applies to any diffusion or flow VLA.
  - It was evaluated on 12 Kinetix sim tasks and 6 real bimanual tasks.
- A third-party RTC port for openpi pi0.5 exists ([HF Daniel-F/rtc-pi05](https://huggingface.co/Daniel-F/rtc-pi05)). It is unofficial and I did not verify it.
- Community latency findings:
  - Camera-pipeline latency of several hundred ms significantly hurt π0.5 rollouts ([#906](https://github.com/Physical-Intelligence/openpi/issues/906), [PrismaX write-up](https://blog.prismax.ai/article.html?slug=latency-vs-pi05-rollout-performance)).
  - The #1060 setup executes 5 of the 10 predicted steps before re-inferring ([#1060](https://github.com/Physical-Intelligence/openpi/issues/1060)).
- From the π0.5 paper ([arXiv 2504.16054](https://arxiv.org/pdf/2504.16054)): the action expert produces a 50-step chunk. Secondary sources report 10 denoising steps; I did not read this in the paper text myself.

### Inferences
- Because RTC is inference-only, a fine-tuned action horizon (for example 10 or 16) limits how much inference delay RTC can absorb. A short horizon leaves little room for in-flight inpainting. That is a design consideration, not a documented PI recommendation.

### Gaps
- There is no official openpi guidance on choosing the execution horizon, on changing `num_steps` after fine-tuning, or on RTC for pi05. I could not fetch the PI RTC and KI blog pages (HTTP 429), so blog-only details are missing.
