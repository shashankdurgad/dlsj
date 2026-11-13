# Deploying open VLAs on an SO-101 arm with Intel-only compute (XPU / OpenVINO), late 2026

Research date: 2026-11-13. About 18 tool calls. Primary sources: the LeRobot GitHub repo and docs, Physical AI Studio, and Intel Open Edge Platform docs. Models not covered by a primary source are listed under Gaps. They have not been verified and should not be read as negative findings.

## Q1. Which models are integrated into LeRobot or Physical AI Studio (late 2026)?

### Takeaway
LeRobot `main` now ships native policy types for many of the target models: `xvla`, `eo1`, `evo1`, `wall_oss_05`/`wall_x`, `groot`, `lingbot_vla_v2`, `pi05`, `smolvla`, `molmoact2` and `cosmos3`. Physical AI Studio (Intel) has a much shorter native list: ACT, Pi0.5, SmolVLA, MolmoAct2, RLDX-1, XR0 and Cosmos3. None of the Chinese VLAs are on it.

### Cited Findings
- The `huggingface/lerobot` `src/lerobot/policies/` directory has these subdirectories: act, common, cosmos3, diffusion, dm05, eo1, evo1, fastwam, fineart_vla, flux3, g05, gaussian_actor, groot, lawam, lingbot_va, lingbot_vla_v2, longwam, molmoact2, multi_task_dit, pi0, pi05, pi0_fast, praxis, rtc, smolvla, tdmpc, vla_jepa, vqbet, wall_oss_05, wall_x, xvla. — [LeRobot policies dir](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies)
- X-VLA is native (`policy.type=xvla`) and installed with `pip install 'lerobot[xvla]'`. The base checkpoint is `lerobot/xvla-base` (0.9B), and there are fine-tuned checkpoints for LIBERO (93%), WidowX, folding, AgiBot and Google Robot. — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/xvla)
- EVO1 is native (`policy.type=evo1`) and built on InternVL3-1B-hf with a flow-matching head. It is installed with `pip install -e ".[training,evo1]"`. — [LeRobot EVO1 docs](https://huggingface.co/docs/lerobot/evo1)
- The LeRobot hardware guide covers `xvla` and `wall_x` (in the "Large VLA" group with pi0/pi05) and `groot` and `eo1` (in the "Multimodal" group). — [LeRobot hardware guide](https://huggingface.co/docs/lerobot/main/hardware_guide)
- Physical AI Studio says: "Native policy families include ACT, Pi0.5, SmolVLA, MolmoAct2, RLDX-1, XR0, and Cosmos3". It supports `--xpu` / `physicalai-train[xpu]` and can export to OpenVINO with `policy.export(..., backend="openvino")` or `physicalai export --backend openvino`. — [physical-ai-studio](https://github.com/open-edge-platform/physical-ai-studio), [library README](https://github.com/open-edge-platform/physical-ai-studio/tree/main/library)
- Physical AI Studio training is built on PyTorch Lightning. It installs with `uv sync --extra xpu --extra all` and needs Python 3.12+. Its docs have no per-policy XPU/OpenVINO support matrix. — [library README](https://github.com/open-edge-platform/physical-ai-studio/tree/main/library)

### Inferences
- Integration tiers based on the findings above:
  - **Native LeRobot plus Physical AI Studio (Intel tooling):** pi0.5 and SmolVLA. Both have a supported XPU training path and OpenVINO export.
  - **Native in LeRobot only:** X-VLA, EVO1, EO-1, WALL-OSS (`wall_oss_05`, `wall_x`), GR00T (`groot`) and LingBot (`lingbot_vla_v2`).
  - **Possibly native, unconfirmed:** `g05` may be Galaxea G0.5. I am inferring that from the name and have not checked it.
  - **Not in LeRobot's policies directory:** RDT-2, InternVLA-M1/A1, GO-1, Being-H0, VLA-Adapter and GR-3. Running any of them would mean using the authors' own repos plus a conversion step.
- The directory listing shows `main` as of late 2026. Some of these policies may not be in the latest PyPI release, so pin a commit.

### Gaps
- I did not open the LeRobot docs for `eo1`, `wall_oss_05`, `lingbot_vla_v2`, `g05` or `groot`, so I have no recipe or latency figures for them.
- I could not confirm whether RDT-2, InternVLA-M1/A1, GO-1, Being-H0, VLA-Adapter or GR-3 have third-party LeRobot plugins. A combined search returned nothing for any of them.
- Background, not verified in this session: GR-3 (ByteDance) may not have public weights.

## Q2. Which models have SO-100/SO-101 fine-tunes, and how well do they work?

### Takeaway
SmolVLA has the best-documented real SO-101 results: 60–80% pick-and-place from 75 demos. X-VLA has a built-in `so101_bimanual` action mode and several community SO-101 checkpoints, but none publish evaluations. GR00T N1.5/N1.6 have Seeed SO-101 tutorials, but they target NVIDIA Jetson. EVO1 has an SO-100 base checkpoint.

### Cited Findings
- **SmolVLA:**
  - Fine-tuned from `lerobot/smolvla_base` on SO-101 with wrist and overhead cameras.
  - Data: 75 teleoperated episodes (10 s each, 30 fps).
  - Result: 60–80% pick-and-place success.
  - Compute: about 10 h for 20k steps at batch 64 on an RTX 3090.
  - Source: [ggand0/vla-so101](https://github.com/ggand0/vla-so101)
- **X-VLA, LeRobot docs:** The docs give an SO-101 bimanual example: `action_mode=so101_bimanual`, 3,000 steps on a `bimanual-so100-handover-cube` dataset. `action_mode=auto` handles single 6-DoF arms by padding to 20 dimensions and computing loss only on the real dimensions. — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/xvla)
- **X-VLA, community checkpoints:**
  - [fabono/xvla_so101_black](https://huggingface.co/fabono/xvla_so101_black): SO-101 follower with wrist and top cameras.
  - [sachinkum0009/so101_xvla_toothbrush](https://huggingface.co/sachinkum0009/so101_xvla_toothbrush)
  - [wsagi/X-VLA-PickOrange](https://huggingface.co/wsagi/X-VLA-PickOrange): LeIsaac SO-101 sim pick-orange, reports 3/3 completion on a single episode.
  - None of these has a rigorous published evaluation.
- **EVO1:** [kevinqz/EVO1-SO100-CoreAI](https://huggingface.co/kevinqz/EVO1-SO100-CoreAI) converts `Beilinghamburger/evo1_so100_vla` to Apple Core AI. Its card says it is a base checkpoint that must be fine-tuned on your robot. The paper's real-robot tests used an xArm6, not an SO-100. — [Evo-1 arXiv](https://arxiv.org/html/2511.04555v2)
- **GR00T N1.5/N1.6:** Seeed has SO-101 fine-tune tutorials that deploy on AGX Orin or Jetson Thor. The default fine-tuning settings need about 25 GB VRAM, and the guides pin `flash-attn==2.8.2.post1` (N1.6). — [Seeed GR00T N1.6 SO-101](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.6_for_lerobot_so_arm_and_deploy_on_agx_orin/), [Seeed GR00T N1.5](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.5_for_lerobot_so_arm_and_deploy_on_jetson_thor/)
- **VLA-REPLICA (arXiv 2605.20774):** A low-cost real-world VLA benchmark built from off-the-shelf parts. The abstract does not name the robot or the models, and I could not retrieve the PDF (over the size limit). — [arXiv abs](https://arxiv.org/abs/2605.20774)

### Inferences
- For a single SO-101 with a wrist camera, SmolVLA and X-VLA (`action_mode=auto`) are the most likely to work without extra glue code.

### Gaps
- I found no published SO-101 success rates for X-VLA, EVO1, EO-1, WALL-OSS, LingBot, pi0.5 or any of the non-integrated Chinese models.
- The VLA-REPLICA paper may contain SO-101 comparisons, but I could not read its full text.

## Q3. What blocks these models on Intel XPU, and what are the workarounds?

### Takeaway
Upstream LeRobot officially supports XPU (`--policy.device=xpu`) and bf16 mixed precision through accelerate. For LeRobot-native policies, flash-attn is optional: EVO1 explicitly falls back to standard attention, and its LIBERO eval profile runs with `use_flash_attn=false`. GR00T (flash-attn pinned in its guides) and the out-of-tree Chinese repos are the main dependency risks.

### Cited Findings
- **LeRobot XPU support:** LeRobot lists "XPU: acceleration with Intel integrated and discrete GPUs" as a supported accelerator for training and inference. You install the PyTorch XPU build and select the device with `--policy.device`. Mixed precision is set with `--accelerator.mixed_precision=bf16|fp16`. The docs list no XPU-specific known issues. — [LeRobot torch_accelerators](https://huggingface.co/docs/lerobot/torch_accelerators)
- **Origin of XPU support:** LeRobot PR #1277, "Support Intel XPU" (2025), made the device utilities recognize XPU and added an XPU Dockerfile. — [PR #1277 mirror](https://mygithub.libinneed.workers.dev/huggingface/lerobot/pull/1277)
- **EVO1:** "Install a `flash-attn` wheel only if it is compatible … EVO1 falls back to standard attention when `flash_attn` is not available." There is a `policy.use_flash_attn` flag, and the documented eval profile sets it to `false`. — [LeRobot EVO1 docs](https://huggingface.co/docs/lerobot/evo1)
- **X-VLA:** The docs list only the `lerobot[xvla]` extra and mention no flash-attn. The examples use `--policy.device=cuda`, which you change to `xpu`. — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/xvla)
- **GR00T:** Community SO-101 guides require `flash-attn` (2.7.1.post4 or 2.8.2.post1) and say CUDA version matching matters. — [Seeed GR00T N1.6](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.6_for_lerobot_so_arm_and_deploy_on_agx_orin/)

### Inferences
- **Workaround:** set `attn_implementation="sdpa"` or `"eager"` in the transformers backbones, or the policy's flash-attn flag to false. PyTorch SDPA runs on XPU.
- **Packages to strip:** remove `flash-attn`, `xformers`, `bitsandbytes` and `deepspeed` from out-of-tree repos' requirements. In those repos, check for `.cuda()` calls and `torch.cuda.amp` and replace them with device-agnostic code.
- **GR00T's other CUDA dependency:** GR00T N1.5/N1.6 probably also relies on a TensorRT deploy path. Treat GR00T as high risk on Intel.

### Gaps
- I found no public reports of X-VLA, EO-1, WALL-OSS, LingBot or EVO1 being trained or run on XPU. Compatibility is inferred, not demonstrated.
- I did not inspect `requirements.txt` for RDT-2, InternVLA-M1/A1, GO-1, Being-H0 or VLA-Adapter.

## Q4. Can these be fine-tuned on one 16–24 GB Intel GPU in a few hours?

### Takeaway
SmolVLA (10–16 GB) is the only VLA that fits comfortably. X-VLA, pi0.5, wall_x, GR00T and EO-1 need 24–40 GB, so 24 GB is tight even at batch 1. EVO1 (0.77B, about 2.3 GB at inference) is a promising middle option, with a frozen-VLM Stage 1. Expect Arc to be slower than the NVIDIA figures below, so plan for small step counts.

### Cited Findings
- **Peak VRAM at batch 8 with AdamW** ([LeRobot hardware guide](https://huggingface.co/docs/lerobot/main/hardware_guide)):

  | Group | Policies | Peak VRAM |
  | --- | --- | --- |
  | Small VLA | smolvla | ~10–16 GB |
  | Large VLA | pi0/pi05/xvla/wall_x | ~24–40 GB ("24 GB tight at BS 1") |
  | Multimodal | groot/eo1 | ~24–40 GB |

  For SmolVLA, the guide's memory tip is `freeze_vision_encoder=True`.
- **Wall-clock (same guide):**
  - 5 epochs on about 50 episodes (45k frames): SmolVLA takes about 3–6 h on an L4/A10G at batch 4. pi0/pi05 take about 4–8 h on an A100 40 GB.
  - On 4×H100 for 5k steps: SmolVLA takes 27m49s (0.312 s/step) and pi05 takes 3h41m (2.548 s/step, with gradient checkpointing and bf16).
- **X-VLA:** Use `policy.dtype=bfloat16` to avoid OOM. For OOM, the docs suggest reducing `chunk_size` (32 to 16), enabling gradient checkpointing, lowering batch size and freezing more components. The docs recommend full fine-tuning with the VLM at 1/10 learning rate. The paper notes that tuning only soft prompts (9M parameters, about 1%) gets close to pi0 on LIBERO. — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/xvla)
- **EVO1:** Stage 1 freezes the VLM and trains the action head (5k steps). Stage 2 fine-tunes the VLM. Gradient checkpointing and bf16 are on by default, and `vlm_num_layers=14`. The paper reports 0.77B parameters, 2.3 GB GPU memory and 16.4 Hz on an RTX 4090d. — [LeRobot EVO1 docs](https://huggingface.co/docs/lerobot/evo1), [Evo-1 arXiv](https://arxiv.org/html/2511.04555v2)
- **GR00T:** Default fine-tuning settings need about 25 GB VRAM. — [Seeed GR00T N1.6](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.6_for_lerobot_so_arm_and_deploy_on_agx_orin/)

### Inferences
- **On a 24 GB B60:**
  - SmolVLA (bf16, batch 4–8) fits.
  - EVO1 Stage 1 (frozen VLM) should fit easily.
  - X-VLA might fit if you train only soft prompts plus the action transformer with a frozen VLM, bf16, chunk 16 and batch 1–2.
  - pi0.5 needs gradient checkpointing, batch 1 and likely a frozen vision encoder.
- **On a 16 GB A770:** use SmolVLA or EVO1 Stage 1.
- **Training steps:** Arc throughput is likely well below an A100's, so budget about 5k–10k steps for SmolVLA in a hackathon. Shrink `scheduler_decay_steps` to match, as the hardware guide advises.
- **LoRA:** I saw no first-class LoRA option for these LeRobot policies. The cheap paths are frozen backbones (X-VLA soft prompts, EVO1 Stage 1, SmolVLA `train_expert_only`).

### Gaps
- I found no measured LeRobot training throughput for any VLA on Arc or B-series GPUs.
- I did not check whether LeRobot has a PEFT/LoRA flag in late 2026.

## Q5. Are there Intel or OpenVINO deployments of these models?

### Takeaway
Intel has first-party optimized pi0.5: 294 ms per chunk with stock PyTorch XPU on a Core Ultra X7 358H iGPU, and an OpenVINO plus Real-Time Chunking pipeline that converts LeRobot pi0.5 checkpoints. Physical AI Studio adds XPU training and OpenVINO export for SmolVLA, ACT and pi0.5. I found no Intel or OpenVINO deployments of X-VLA, EVO1 or the other Chinese VLAs.

### Cited Findings
- **pi0.5 (DROID variant) on Panther Lake** ([Intel OEP article](https://docs.openedgeplatform.intel.com/dev/OEP-articles/publications/optimizing-pi0.5-lva-model.html)):
  - Setup: 3×224² cameras, 10 denoising steps, BF16/FP16, Core Ultra X7 358H at 40 W.
  - Stock PyTorch 2.12 XPU latency is 294 ms, against 287 ms on Jetson Thor. The article says this is 3.5× faster than AGX Orin at 60 W.
  - The OpenVINO version splits the model: the vision encoder and LLM run on the iGPU, and the action expert runs on the NPU, with the KV cache shared zero-copy.
  - The OpenVINO version claims up to 1.5× lower latency than Orin at max TDP. No absolute OpenVINO millisecond figure is given.
  - Training quick start: `pip install physicalai-train`, then `physicalai fit` with a LeRobot datamodule, then export to OpenVINO.
- **Intel Robotics AI Suite "Pi0.5 with Real-Time Chunking"** ([Intel Edge AI Suites](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-suites/robotics-ai-suite/embodied/sample_pipelines/pi05_with_rtc.html)):
  - `convert_ov.py` / `convert_ov_rtc.py` convert a LeRobot checkpoint (default `lerobot/pi05_base`) to OpenVINO IR, FP16 by default, with optional INT8 via NNCF (`--compress_int8`).
  - The examples target ALOHA (real and MuJoCo), aiming for 50 Hz control with RTC. The page has no measured latency.
  - PaliGemma is gated on Hugging Face, so you need an HF login.
- **SmolVLA / ACT:** Physical AI Studio exports SmolVLA, ACT and pi0.5 to OpenVINO. — [physical-ai-studio](https://github.com/open-edge-platform/physical-ai-studio)
- A PyTorch Conference EU 2026 talk ("Full-Stack PyTorch Robotics VLA") reportedly shows `physicalai` training with `accelerator="xpu"` and lists SmolVLA. I could not extract the PDF text, so treat this as unverified. — [slides](https://hosted-files.sched.co/pytorchconferenceeu2026/fb/Full-Stack%20PyTorch%20Robotics%20VLA_PyTorch_Conf_EU2026.pdf)

### Inferences
- **pi0.5 at 294 ms per chunk on an iGPU** is usable with action chunking plus RTC on an SO-101. A discrete B580/B60 should be faster.
- **SmolVLA (about 450M parameters)** should be much faster still, though I found no Intel number.
- **Exporting X-VLA or EVO1 to OpenVINO** would mean writing your own `torch.export`/`ov.convert_model` script. Flow-matching loops are exportable, as the pi0.5 work shows, but this takes real engineering time.

### Gaps
- I found no absolute OpenVINO latency for pi0.5 and no Intel latency for SmolVLA.
- The memory spec on the Intel article is internally inconsistent (32 GB vs 128 GB).

## Per-model scorecard

| Model | LeRobot | SO-100/101 evidence | Fit on 16–24 GB | CUDA-only risk | Intel/OV support |
|---|---|---|---|---|---|
| SmolVLA | native | 60–80% pick-place, 75 demos ([ggand0](https://github.com/ggand0/vla-so101)) | yes, 10–16 GB | low | Physical AI Studio XPU train + OV export |
| pi0.5 | native | not found (stale or none) | 24 GB tight at BS1 | low (PaliGemma gated) | Intel-optimized, 294 ms iGPU, OV + RTC |
| X-VLA 0.9B | native `xvla` | community ckpts, no evals; `so101_bimanual`/`auto` modes | 24–40 GB full; soft-prompt only should fit | low (no flash-attn in docs) | none found |
| EVO1 0.77B | native `evo1` | SO-100 base ckpt | likely yes (Stage 1 frozen VLM) | flash-attn optional, falls back | none found |
| EO-1 | native `eo1` | not found | 24–40 GB | unknown | none |
| WALL-OSS | native `wall_oss_05`/`wall_x` | not found | 24–40 GB | unknown | none |
| LingBot-VLA | native `lingbot_vla_v2` | not found | unknown | unknown | none |
| GR00T N1.5/1.6 | native `groot` | Seeed SO-101 tutorials | ~25 GB | high (flash-attn pinned, Jetson/TensorRT oriented) | none |
| Galaxea G0 | maybe `g05` (unverified) | not found | unknown | unknown | none |
| RDT-2, InternVLA-M1/A1, GO-1, Being-H0, VLA-Adapter, GR-3 | not native | not found | unknown | unknown (not inspected) | none |

## Recommendation (12–24 h hackathon, Intel only)

1. **SmolVLA (primary).** It fits 16 GB, has an XPU training path in both upstream LeRobot (`--policy.device=xpu`) and Physical AI Studio, can be exported to OpenVINO, and has proven SO-101 results (60–80% from about 75 demos). Collect 50–80 episodes with wrist plus fixed camera, train 5k–10k steps in bf16, and freeze the vision encoder if memory is tight.
2. **pi0.5 (stretch, if a 24 GB B60 or a strong iGPU with lots of shared RAM is available).** Intel has optimized it end to end: XPU inference at 294 ms, OpenVINO conversion scripts and RTC. Fine-tuning will be slow and memory-tight, so use gradient checkpointing, batch 1 and bf16, or train through Physical AI Studio.
3. **X-VLA or EVO1 (the "Chinese VLA" pick).** Both are native in LeRobot, with no hard flash-attn dependency and a cheap frozen-backbone mode: X-VLA soft-prompt or policy-only training, EVO1 Stage 1. X-VLA has SO-101 action modes and community SO-101 checkpoints. EVO1 is smaller (0.77B, 2.3 GB at inference, 16.4 Hz on a 4090d). Run inference in PyTorch XPU rather than OpenVINO. Neither has been shown working on Intel yet, so do a 30-minute smoke test (load, one train step, one inference on `xpu`) before committing.
- **Avoid in a hackathon:** GR00T (flash-attn and Jetson/TensorRT tooling), and every model without a LeRobot policy (RDT-2, InternVLA, GO-1, Being-H0, VLA-Adapter, GR-3). Porting them and stripping their CUDA dependencies would take most of the time available.
