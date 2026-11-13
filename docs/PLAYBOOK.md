# SO-101 Robot Learning Hackathon Playbook (Intel XPU)

*Compiled 13 Nov 2026 from the official links plus four research threads: anomaly detection, robustness, speed, and force sensing. Updated the same day with XPU inference optimisation and a VLA comparison (Sections 2.1, 4.5; full report in `reports/XPU kernels and Chinese VLAs.md`).*

**Hardware:** SO-101 leader/follower arms, wrist-mounted Intel RealSense depth camera (likely D405, ~7–50 cm range), Intel GPU ("XPU"), optional force sensor from a friend.
**Tasks:** pick-and-place, sorting, anomaly detection.
**Judged on (assumed):** success, robustness to lighting/texture/colour, speed.

> Items marked ⚠️ **Unverified** come from docs/source reading or from extrapolation and were not tested on hardware. Smoke-test them first.

---

## 0. TL;DR: the plan

| Component | Choice | Why |
|---|---|---|
| Motor policy | **ACT**, trained in **bf16 on XPU**, 20–40k steps | Smallest, fastest to train, recommended first by LeRobot |
| Deployment | **OpenVINO export** + `physicalai` runtime with **AsyncExecution**, chunk 50–100, **no temporal ensembling** | Low latency, smooth, hides inference time |
| Data | ~150 brisk demos over **6–10 lighting/background conditions**, idle frames trimmed | Diversity beats quantity; fast demos give a fast policy |
| Robustness | LeRobot image augmentation (brightness/contrast/shadow/white-balance), **hue jitter ≈ 0**; depth-based background replacement if time allows | Lighting robustness without destroying colour info |
| Sorting | **HSV colour selector (or tiny classifier) → per-bin ACT policy** (or language-conditioned SmolVLA) | All policies score 0–10% on selective colour sorting end-to-end |
| Anomaly detection | **Anomalib PaDiM/PatchCore** at a **fixed inspection pose**, OpenVINO export; **AnomalyDINO** if few images; plus **depth height-map check** | Intel-native, millisecond inference, works with ~100 good images |
| Force | **Gripper `Present_Current`** in `observation.state` + rule-based grasp verification; friend's sensor as a stretch | One SO-101 project reported 77.5% → 95% success |
| VLA track (parallel) | **π0.5** (SnapFlow 1-step distil → OpenVINO INT8 + RTC) if ≥24 GB; else **SmolVLA**. X-VLA only as a 30-min smoke test | Only VLA with a measured Intel path (172 ms/chunk on Panther Lake); no Chinese VLA beats π0.5 in independent tests (Section 2.1) |
| Stretch | DINOv2 encoder, green-screen-style augmentation, NPU offload of the action expert | Only with spare hours |

### Suggested timeline (adapt to remaining hours)
1. **Hour 0–1:** environment setup (Section 1) and verify the arm and camera; one person starts collecting demos.
2. **Hour 1–3:** collect diverse demos (Section 3); in parallel, capture good and bad images for anomaly detection (Section 6).
3. **Hour 3–5:** trim data (Section 4.4), train ACT on XPU (Section 4); train Anomalib (Section 6).
4. **Hour 5–7:** export to OpenVINO, deploy with async runtime, add the HSV selector and grasp check.
5. **Hour 7+:** test on held-out lighting/background, then iterate on failures (more demos of the failure condition beats more training).
6. **Final:** demo video with the anomaly heatmap overlay, live score readout, and a lighting-change test on camera.

---

## 1. Toolchain & environment setup

### 1.1 Provided Intel tools and what each is for

| Tool | Role | Notes |
|---|---|---|
| [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops) | PyTorch operators for Intel GPUs (the `xpu` device) | Built into stock PyTorch, not installed separately |
| [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio) | **Train / benchmark / export** policies (ACT, Pi0.5, SmolVLA, MolmoAct2, RLDX-1, XR0, Cosmos3) | UI + `physicalai fit` CLI; OpenVINO export; no anomaly detection, augmentation not documented |
| [physicalai runtime](https://github.com/openvinotoolkit/physicalai) | **Deploy** on the robot: SO-101 + RealSense support, `InferenceModel`, leader-follower teleop source, async execution | `pip install physicalai[so101,realsense]` |
| [Robotics AI Suite](https://github.com/open-edge-platform/robotics-ai-suite) | Reference pipelines: ACT, Diffusion Policy, iDP3, Pi0.5 with Real-Time Chunking, OpenVINO-optimised | Borrow conversion scripts; no SO-101/LeRobot mention |
| [LeRobot](https://github.com/huggingface/lerobot) | Alternative training stack with XPU support and **image augmentation** | Use if you need augmentation or custom observations |
| [Anomalib](https://github.com/open-edge-platform/anomalib) | Anomaly detection (Intel) | `pip install "anomalib[xpu,openvino]"` |

### 1.2 PyTorch on XPU
```bash
# Install an Intel GPU driver + Intel Deep Learning Essentials first (see Intel's PyTorch prerequisites page)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
```
- FP32, BF16, FP16 and AMP are supported on XPU. **Use bf16.** fp16 `GradScaler` needs hardware FP64, which Arc A-series cards lack.
- `torch.compile` is supported on XPU, **but `mode="reduce-overhead"` is a silent no-op** (Inductor only graph-captures on CUDA, as of PyTorch 2.15). Use `mode="max-autotune-no-cudagraphs", dynamic=False` and capture graphs manually with `torch.xpu.graph` (PyTorch ≥2.11). See Section 4.5.
- **Do not install IPEX** (intel-extension-for-pytorch): end-of-life since March 2026 and incompatible with Triton-XPU.
- Pin one PyTorch version for the whole event; 2.13/2.14 had Inductor regressions of 8–32% on some models.

### 1.3 Physical AI Studio
**Docker (recommended):**
```bash
git clone https://github.com/open-edge-platform/physical-ai-studio && cd physical-ai-studio/application/docker
cp .env.example .env
./setup-devices.sh --xpu
docker compose up -d        # UI at http://localhost:7860
```
**Python library only:**
```bash
pip install 'physicalai-train[xpu]'          # Python 3.12–3.14
# or, from the repo:  cd library && uv sync --extra xpu --extra all
```
- Set `HF_TOKEN` for Hugging Face-backed policies (SmolVLA, Pi0.5).

### 1.4 physicalai runtime (deployment)
```bash
pip install "physicalai[so101,realsense]"
```
- Connect to the SO-101 over its serial port. `verify_robot` runs an interactive joint-by-joint check.
- Full walkthrough: `examples/tutorials/collect_train_deploy.ipynb` in the runtime repo. **Start here.**

### 1.5 RealSense tips
- Use `pyrealsense2`, or the runtime's `[realsense]` extra. You need the camera's serial number in configs.
- Use **640×480 @ 30 fps**, and plug **directly into the laptop, not through a hub** (USB bandwidth causes frame drops).
- **Lock exposure and white balance** (turn auto-exposure/auto-WB off) for anomaly detection, and ideally for policies too. Auto-exposure changes images between frames and between runs.

---

## 2. Choosing a policy

| Policy | Training cost (50 eps) | Inference | Strengths | Weaknesses |
|---|---|---|---|---|
| **ACT** | ~30–60 min on RTX 4090, ~1–2 h on L4; expect similar or slower on Intel GPU | Fast, single forward pass per chunk | Simple, reliable, LeRobot's recommended first policy | No language; needs diverse data for robustness |
| **SmolVLA** | ~3–6 h on L4 | Flow matching; supports Real-Time Chunking | Language conditioning, multi-task | Pretrained state projection hasn't seen extra dims; slower |
| **π0.5** | Too slow to fine-tune in hours on one Intel GPU | Heavy; SO-101 SnapFlow recipe distils to 1 step | Strongest on SO-101 benchmark | Compute cost |

**SO-101 benchmark** ([arXiv 2606.08881](https://arxiv.org/html/2606.08881)), average success: **π0.5 56.25%**, ACT 33.75%, SmolVLA 32.5%. Selective colour sorting was **0–10% for all three**, which is why sorting needs a separate perception step (Section 5.6).

**Decision:** ACT for everything you need working (guaranteed baseline). Run a VLA as a parallel track (Section 2.1); it earns the demo only if it beats ACT on held-out lighting.

### 2.1 Which VLA (updated after research, see `reports/XPU kernels and Chinese VLAs.md`)

**Chinese VLAs vs π0.5:** none beats π0.5 in independent testing. Each lab wins on a benchmark it chose (saturated LIBERO, its own RoboTwin runs, RoboChallenge Table30). On the independent **RoboDojo** real-robot benchmark ([arXiv 2607.04434](https://arxiv.org/html/2607.04434v1)): π0.5 12.8%, InternVLA-A1 7.2%, Galaxea G0 4.4%, Xiaomi-R0 3.9%, X-VLA 3.3%, DM0 0% (DM0 topped Table30 at 62%). Several do beat SmolVLA: X-VLA 6.52% vs SmolVLA 0.85% in RoboDojo simulation; WALL-X reportedly 51.25% vs 32.5% on SO-101 (blog figure, unchecked).

| Model | Params | LeRobot | Intel path | Verdict |
|---|---|---|---|---|
| **π0.5** | 3B | native | Intel OpenVINO + RTC pipeline; Physical AI Studio `pi05/so101/snapflow.yaml` | **Primary** if a 24 GB card (Arc Pro B60) or iGPU with ≥32 GB RAM |
| **SmolVLA** | ~0.45B | native | Physical AI Studio XPU train + OpenVINO export | **Fallback** on 16 GB cards (A770/B580). Buys language, not success rate, over ACT |
| **X-VLA** (Tsinghua) | 0.9B | native, SO-101 action modes | PyTorch XPU only, untested on Intel | Best Chinese gamble: 30-min smoke test only |
| **EVO1** | 0.77B | native, flash-attn optional | PyTorch XPU only, untested on Intel | Smallest Chinese option |
| WALL-OSS / WALL-X | 4B | native | none | Too big |
| GR00T, Xiaomi-R0, Spirit, RDT2, InternVLA-A1, Galaxea G0.5 | 3–7B | partial/none | none; GR00T pins flash-attn; G0.5 non-commercial | Avoid |

**Rule:** training memory decides, not accuracy (π0.5/X-VLA/WALL-X need 24–40 GB, "24 GB tight at batch 1"). Time the first 200 training steps; if the projected run exceeds the budget, drop to SmolVLA within the first hour. Porting an out-of-tree repo to XPU (`attn_implementation="sdpa"`, remove flash-attn/xformers/bitsandbytes, replace `.cuda()`) costs a day, not an hour.

---

## 3. Data collection

### 3.1 Diversity beats quantity
- A data-scaling study found environment and object diversity matters far more than the absolute demo count; returns flatten around ~50 demos per environment–object pair ([arXiv 2410.18647](https://arxiv.org/html/2410.18647v3)).
- **Plan for ~150 demos per task** across **6–10 conditions × 15–25 demos**:
  - Lighting: desk lamp vs overhead, warm vs cool, hard shadows, dim.
  - Backgrounds: 3–4 tablecloths, posters, plain table.
  - Objects: different shades and colours (for pick-and-place), different starting positions and orientations.
  - Distractors: 1–2 irrelevant objects in some episodes.
- **Record the most likely judging conditions first.**
- **Hold out one lighting setup and one background** that you never train on, as your own robustness test.

### 3.2 Demo quality = policy speed
- The policy copies your speed and hesitation. **Demo at the speed you want the robot to move**, decisively, with no fumbling. (Lesson from [Flash ALOHA](https://flash-aloha.github.io), which reached >80% success at near-human speed with 50 demos per task.)
- **Keep your hands and arms out of the camera view.** Flash ALOHA masks the operator because operator appearance hurts visual alignment. The wrist camera mostly avoids this, but watch any static cameras.
- Discard failed or retried demos, or record them separately; don't train on them by default.
- Keep a consistent strategy: always approach from the same side and grasp the same way. Multimodal demos confuse ACT.

### 3.3 Data collection tooling
- **LeRobot:** `lerobot-record` with leader → follower teleop.
- **physicalai runtime:** `TeleopSource` (leader-follower) inside the runtime loop, or the Studio UI.
- Keep fps consistent (30 Hz) across all recording.

---

## 4. Training & speed on Intel XPU

### 4.1 LeRobot on XPU
LeRobot detects `torch.xpu` automatically and accepts `--policy.device=xpu`; AMP is allowed on xpu ([device_utils.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/utils/device_utils.py)).
```bash
lerobot-train \
  --dataset.repo_id=$HF_USER/so101_pick \
  --policy.type=act \
  --policy.device=xpu \
  --policy.use_amp=true \
  --batch_size=8 \
  --steps=20000 \
  --dataset.image_transforms.enable=true \
  --policy.repo_id=$HF_USER/act_pick
```
- It warns that some Intel GPUs have no FP64; that's expected.
- ⚠️ **Unverified:** `--accelerator.mixed_precision=bf16` appears in the train script docstring but was not tested on XPU.
- **Time your first ~200 steps** (`update_s` in logs) and extrapolate before committing to a step count.

### 4.2 Physical AI Studio on XPU
A ready-made XPU config ships at `library/configs/physicalai/act/pusht/xpu.yaml` (`accelerator: xpu`, `strategy: xpu_single`, `precision: 32`, `max_steps: 70000`).
```bash
cd library
# Smoke test first
physicalai fit --config configs/physicalai/act/pusht/xpu.yaml \
  --data.repo_id $HF_USER/so101_pick --data.val_gym null \
  --trainer.max_steps 20000 --trainer.precision bf16-mixed \
  --trainer.fast_dev_run=true
# Then drop --trainer.fast_dev_run for the real run
```
- ⚠️ **Unverified:** `--data.val_gym null` parsing and `bf16-mixed` with `xpu_single`. If either fails, fall back to the shipped `precision: 32`.
- **Gotcha:** the other configs (SmolVLA, `lerobot/act.yaml`) default to `accelerator: gpu`. Add `--trainer.accelerator xpu --trainer.strategy xpu_single`.
- **SmolVLA:** `physicalai fit --config configs/physicalai/smolvla/pusht/default.yaml --trainer.accelerator xpu --trainer.strategy xpu_single --data.repo_id ...`. Change the default `num_cameras: 0` and the PushT `val_gym` for SO-101.
- **π0.5:** an SO-101 recipe exists at `configs/physicalai/pi05/so101/snapflow.yaml` (bf16, `compile_model: true`, distilled to 1 denoising step; ~2.98× faster chunks reported on H100, not XPU).

### 4.3 Export & inference
**Export to OpenVINO:**
```bash
physicalai export --policy physicalai.policies.ACT \
  --ckpt_path experiments/lightning_logs/version_0/checkpoints/last.ckpt \
  --backend openvino --output_dir exports/act
```
**INT8 weight compression (Python only, no CLI flag):**
```python
# pip install physicalai-train[nncf]
policy.export("./exports", backend="openvino",
              post_export_hooks=[compress_weights_openvino_int8_sym])
```
**Run on the robot with async execution:**
```python
from physicalai import InferenceModel
model = InferenceModel("./exports/act")          # auto-picks OpenVINO device: GPU → NPU → CPU
# In the runtime config / Python API:
# PolicySource(model=model, execution=AsyncExecution(request_threshold=0.5))
# → inference runs in a background thread; the next chunk is requested when the
#   action queue drops below 50% of the chunk.
```
```bash
physicalai run --config runtime.yaml --run.duration_s=60
```
- **Benchmark:** the runtime has `InferenceLatencyBenchmark` (iterations, warmup, max duration → JSON). The only published reference is Intel's **π0.5 at 172 ms per 50-action chunk** on a Core Ultra X7 358H (Panther Lake iGPU, 3 cameras, 10 steps, OpenVINO + INT8), down from 555 ms PyTorch eager ([Intel paper](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf), vendor-reported). Nothing is published for ACT/SmolVLA, or for any policy on Arc or Lunar Lake; **measure your own** (median and p99).
- **π0.5 via Robotics AI Suite:** `uv run --extra pi-ov --with nncf scripts/convert_ov_rtc.py --torch_dir <ckpt> --dataset_path <ds> --ov_output_dir out --compress_int8` (in `src/pipelines/pi05-rtc-ov`).

### 4.4 Execution speed (task throughput)

**Chunking choices:**
- **Temporal ensembling (ACT):** forces `n_action_steps=1`, so the policy runs on every tick. Smooth but the most compute. **Avoid on XPU**; use plain chunking + async.
- **Real-Time Chunking (RTC):** only for flow-matching policies (SmolVLA, π0.5). Set `policy.rtc_enabled=True` **before export**. Validated defaults: `execution_horizon=15`, `max_guidance_weight=5.0` at 30 fps; `execution_horizon ≤ chunk_size/2`.
  - LeRobot: `lerobot-rollout --strategy.type=base --inference.type=rtc --inference.rtc.execution_horizon=10 ...`
- LeRobot's old async client/server is now `--inference.type=remote` with `lerobot-policy-server`.
- If chunk boundaries cause jerks, add light smoothing (Flash ALOHA used a lightweight smoothing step for action-chunk tracking instability).

**Faster-than-demo tricks:**
- `--interpolation_multiplier` only adds intermediate commands; it does **not** speed up playback.
- ⚠️ **Untested idea:** run rollout at a higher `--fps` than recorded (e.g. record at 30, run at 40). Breaks if a dataset is attached, since `--dataset.fps` must equal `--fps`. Test carefully, since it can cause overshoot.

**WARP-RM / WARP-BC** ([arXiv 2606.28320](https://arxiv.org/pdf/2606.28320), [code](https://github.com/uynitsuj/WARP-RM), [project](https://uynitsuj.github.io/warp-rm/)):
- "Faster" means **task throughput**, not training speed.
- Method: frozen DINOv3 ViT-B/16 + bidirectional transformer predicts task progress, trained self-supervised by replaying successful demos at random speeds and in reverse. Differencing progress gives a "progress velocity" per frame (~1 = reference pace, ~0 = stalling, <0 = regressing). WARP-BC drops 1-second action chunks with velocity ≤ τ=1.0 and weights the rest in a flow-matching BC loss (π₀ backbone).
- Results: T-shirt folding 31.6 → 56.3 folds/hr (1.78×) on the cleanest dataset; 1.5 → 27.4/hr on a noisier one; bottle-in-bin 147.8 → 237.8/hr (1.6×).
- **Too heavy for tonight** (thousands of episodes, OpenPI-based). Use the approximation below.

**Cheap WARP-BC approximation (~30 min):**
1. For each frame compute joint speed `v[t] = ||q[t+1] − q[t]|| × fps`, excluding the gripper.
2. Drop runs of **more than 5 consecutive frames** with `v < 2%` of the 95th-percentile speed (idle/hesitation). **Keep frames around gripper open/close events**, since those pauses are deliberate.
3. Drop episodes longer than **~1.3× the median duration**, and any with retries.
4. Optional (DemoSpeedup-lite): downsample fast "transit" segments 2×. Related work: [DemoSpeedup](https://arxiv.org/abs/2506.05064) (up to ~3× faster with no success loss), [SAIL](https://arxiv.org/abs/2506.11948) (up to 3.2× on real robots).
5. Rebuild the dataset (`lerobot-edit-dataset`, or write a new LeRobotDataset). **Check that action chunks don't jump across cut gaps.**

```python
import numpy as np
def idle_mask(q, gripper, fps=30, min_run=5, frac=0.02, keep_win=10):
    """q: (T, J) joint positions without gripper; gripper: (T,) gripper position."""
    v = np.linalg.norm(np.diff(q, axis=0), axis=1) * fps
    v = np.append(v, v[-1])
    idle = v < frac * np.percentile(v, 95)
    # protect windows around gripper open/close events
    g_events = np.where(np.abs(np.diff(gripper)) > np.percentile(np.abs(np.diff(gripper)), 90))[0]
    for e in g_events:
        idle[max(0, e - keep_win): e + keep_win] = False
    # only drop runs longer than min_run
    drop = np.zeros_like(idle)
    t = 0
    while t < len(idle):
        if idle[t]:
            s = t
            while t < len(idle) and idle[t]:
                t += 1
            if t - s > min_run:
                drop[s:t] = True
        else:
            t += 1
    return ~drop   # True = keep frame
```

### 4.5 Kernel / compiler-level inference optimisation on XPU

Full evidence in `reports/XPU kernels and Chinese VLAs.md`. **Key lesson:** on Intel, "kernel-level" mostly means shaping the model so OpenVINO's built-in fusions fire, then graph-capturing what's left. Model-level cuts (1 denoising step, fewer cameras, INT8) beat hand-written kernels; even a fully optimised π0 needs 20–37 ms on an RTX 4090, and an iGPU has ~1/6 of its bandwidth.

**Where π0.5 latency goes** (Intel profile): VLM = 95% of compute (compute-bound); action expert = 4% but **dispatch-bound at ~104 µs per unfused kernel launch**.

**Ordered checklist (VLA; ✓ = also applies to ACT):**

| # | Step | How | Gain |
|---|---|---|---|
| 0 ✓ | Pin the stack | PyTorch ≥2.11 XPU, OpenVINO 2026.1+, `xe` kernel driver (not i915) on iGPUs, no IPEX, pin P-cores with `taskset` | Avoids regressions, `CL_OUT_OF_RESOURCES` |
| 1 ✓ | Shrink the input | 1–2 cameras (wrist + one fixed) at 224², fixed 64 language tokens, chunk 50 | ~7–9 ms per view on a 4090; more on iGPU |
| 2 | Distil to 1 step | `configs/physicalai/pi05/so101/snapflow.yaml`; SmolVLA: test 3–5 Euler steps | Expert cost ~10× less; LIBERO 98.75% vs 97.75% (10-step) |
| 3 ✓ | Static-shape OpenVINO IR (+RTC baked in) | π0.5: `convert_ov_rtc.py` in robotics-ai-suite `pi05-rtc-ov`, **set `-c/--camera_num`** (defaults to 4, baked in). ACT/SmolVLA: `physicalai export --backend openvino` | Base of 555 → 172 ms |
| 4 | INT8 weight-only | `--compress_int8` / `compress_weights_openvino_int8_sym`; validate trajectory MSE/cosine vs PyTorch | >30% faster, half size. **Skip for ACT** (launch-bound, keep FP16) |
| 5 | One batched vision call | Exported vision IR has batch = n_cams, not n calls | −50–67% vision weight traffic |
| 6 ✓ | Make fusions fire | Export attention via `F.scaled_dot_product_attention`, standard RMSNorm/SwiGLU. Check `benchmark_app -m model.xml -d GPU -hint latency -pc` shows `sdpa` + GatedMLP kernels and oneDNN GEMMs | Fixes the ~104 µs/launch expert cost |
| 7 ✓ | Latency config | `{"PERFORMANCE_HINT":"LATENCY","INFERENCE_PRECISION_HINT":"f16","NUM_STREAMS":"1","CACHE_DIR":"./ovcache"}`; pre-allocate and reuse tensors; leave queue throttle default | Full blob caching only for static shapes |
| 8 ✓ | Async (+ RTC for VLA) | `AsyncExecution(request_threshold=0.5)`, `execution_horizon ≤ chunk/2` | Hides remaining latency (~30% faster tasks for SmolVLA) |
| 9 | iGPU only: action expert on NPU | Separate compiled models, KV cache shared zero-copy as USM-host tensors; not HETERO. Useless on discrete Arc (PCIe) | Refill ≈ max(VE+LE, AE); power/overlap |
| 10 | INT4 on VLM only | NNCF `INT4_ASYM`, group 64–128, AWQ + scale estimation; keep expert INT8 | Risky: generic full-stack INT4 collapses π0.5 to 11–16% |

**OpenVINO gotchas:**
- Dynamic activation quantization is skipped at ≤64 tokens, so the 50-token action expert doesn't get INT8 compute.
- Custom OpenCL kernels (`CustomLayer` XML) bypass fusions and XMX. Match built-in patterns instead.
- Set `INFERENCE_PRECISION_HINT` explicitly; the GPU default is ambiguous in the docs.
- Do not reuse the VLM KV cache across *environment* steps (only across denoising steps within one chunk).

**PyTorch-XPU fallback (if OpenVINO export breaks):**
- bf16 (or fp16 on Arc A-series) so matmuls hit XMX; `torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=False)`.
- **Capture the fixed-step denoising loop (or ACT's single forward) with `torch.xpu.graph`** around static buffers. On Arc A770 it must be forced with `SYCL_UR_USE_LEVEL_ZERO_V2=1 NEOReadDebugKeys=1 OverrideRecordReplayGraphCapability=1` ⚠️ not validated by Intel.
- Cheap launch-overhead fix without graphs: `NEOReadDebugKeys=1 EnableDirectSubmission=1` (A770 launch ~14 µs → ~3 µs).
- VLM backbone memory: torchao `Int4WeightOnlyConfig(group_size=128, int4_packing_format="plain_int32")` (1.56–2.14× decode on B580 for 3–4B LLMs); call `unwrap_tensor_subclass_parameters` before compiling. Useless for ACT.
- Attention: SYCL-TLA FlashAttention is built only for Battlemage and PVC; others use oneDNN SDPA. π0.5's Gemma uses head_dim 256, which may miss the flash path, so time `sdpa_kernel([SDPBackend.FLASH_ATTENTION])` vs `[SDPBackend.OVERRIDEABLE]` on the real card.
- Custom Triton kernels only for elementwise chains (e.g. the Euler update); use `tl.make_tensor_descriptor` (pointer-style kernels are 2–30× slower; block pointers removed in Triton-XPU v3.9). Leave GEMMs to oneDNN (~14% faster than Triton bf16).
- Custom SYCL kernels (`torch.utils.cpp_extension.SyclExtension`) are supported but too slow to build for a hackathon.

---

## 5. Robustness to lighting, textures & colours

Ranked by impact per hour.

### 5.1 Diverse demos (highest impact, zero code)
See Section 3.1. This matters more than any augmentation.

### 5.2 LeRobot image augmentation (5 minutes)
Training-time only; no re-recording needed.
```bash
lerobot-train ... --dataset.image_transforms.enable=true
```
- **Defaults:** `max_num_transforms=3`, brightness (0.8, 1.2), contrast (0.8, 1.2), saturation (0.5, 1.5), hue (±0.05), sharpness jitter ([docs](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3)).
- **Newer LeRobot** adds `RandomAffine`, `RandomRotation`, `GaussianNoise`, `MotionBlur`, `GaussianPatchBrightness`, `RandomShadow`, `CoarseDropout`, `GammaCorrection`, `PlanckianJitter`, plus any torchvision v2 class. A `backend=gpu` option exists but refuses `--policy.device=cpu`.
- **Older installs** only support `ColorJitter`, `SharpnessJitter`, `Identity`; anything else raises `ValueError`. **Check with `pip show lerobot`.**

**Custom set (Python API):**
```python
from lerobot.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig as T
cfg = ImageTransformsConfig(enable=True, max_num_transforms=3, random_order=True, tfs={
  "bright": T(weight=2,   type="ColorJitter",            kwargs={"brightness": (0.5, 1.5), "contrast": (0.6, 1.4)}),
  "white":  T(weight=1,   type="PlanckianJitter",        kwargs={}),   # light colour temperature
  "shadow": T(weight=1,   type="RandomShadow",           kwargs={}),
  "patch":  T(weight=1,   type="GaussianPatchBrightness",kwargs={}),   # light spots
  "shift":  T(weight=1,   type="RandomAffine",           kwargs={"degrees": 3, "translate": (0.05, 0.05)}),
  "sharp":  T(weight=0.5, type="SharpnessJitter",        kwargs={"sharpness": (0.5, 1.5)}),
})
```
- ⚠️ CLI form for `tfs` (`--dataset.image_transforms.tfs='{...}'`) is undocumented; verify it parses.
- ⚠️ **Colour sorting:** keep **hue jitter at ~0** and saturation narrow, otherwise the policy learns to ignore the colour it needs. Brightness, white-balance and shadow jitter are safe.
- **Wrist camera:** keep affine shifts and crops small; large crops break image-to-action alignment.
- Physical AI Studio's README doesn't document augmentation, so **train in LeRobot if you need it.**

### 5.3 Background replacement (1–3 h, big gain on background/texture shift)
- **GreenAug** ([arXiv 2407.07868](https://arxiv.org/html/2407.07868v2)): demos on a green screen with random textures keyed in. **91% average success vs 55% (no aug) and 70% (standard aug)**. Texture variety matters: no texture 48%, solid colours 65%, real textures 87%. Costs ~0.009 s per frame.
- **Depth as a green screen (untested idea):** with the RealSense, mark pixels beyond the table depth as background and replace them with random texture photos, or zero them. Needs aligned depth recorded with the dataset.
- **Cheaper:** random overlay `img' = (1−α)·img + α·random_photo` with α ≈ 0.2–0.3 on part of each batch ([arXiv 2011.13389](https://arxiv.org/pdf/2011.13389) used α=0.5 in RL).

```python
import torch, random
class DepthBackgroundSwap:
    """Replace far-away pixels with a random texture. Needs depth aligned to RGB."""
    def __init__(self, textures, max_depth_m=0.45, p=0.5):
        self.textures, self.max_depth, self.p = textures, max_depth_m, p
    def __call__(self, rgb, depth):            # rgb: (3,H,W) float [0,1]; depth: (H,W) metres
        if random.random() > self.p: return rgb
        tex = random.choice(self.textures)      # (3,H,W) float [0,1]
        bg = (depth > self.max_depth) | (depth == 0)
        return torch.where(bg.unsqueeze(0), tex, rgb)
```

### 5.4 Encoders & policy choice
- Pretrained ViTs are not robust by default. Among them, **DINO-style emergent segmentation predicted OOD robustness better than ImageNet accuracy** ([source](https://paperswithcode.com/paper/what-makes-pre-trained-visual-representations)).
- Swapping ACT's ResNet18 for DINOv2 is ~2+ hours of code. **Skip unless spare time.**

### 5.5 Depth / point clouds (high payoff, high risk)
- **DP3** drops colour from point clouds and generalised across cube colours with 40 real demos ([project](https://3d-diffusion-policy.github.io)). Colour-blind by design, so it can't sort by colour, and full integration is too much for a hackathon.
- Cheaper: feed **depth (or depth-masked RGB) as a second image** into ACT.

### 5.6 Colour sorting: keep the decision outside the policy
- VLAs often grasp fine but pick the wrong object. SmolVLA dropped from 44% to 0% on unseen object–region pairings, and more data barely helped ([source](https://www.alphaxiv.org/abs/2602.24143.md)).
- **Recipe:**
  1. **Perception:** HSV threshold or a tiny classifier picks the target object and bin. Calibrate HSV ranges per lighting condition, or use a small CNN trained on crops under varied lighting.
  2. **Control:** a conditioned policy, either one ACT policy per bin or SmolVLA/π0.5 with a prompt like "place the red block in the left bin".
  3. Now the motor policy can use **strong colour augmentation**, because colour is decided outside it.

```python
import cv2, numpy as np
HSV = {  # calibrate on site, per lighting condition if needed
  "red":   [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
  "green": [((40, 70, 50), (85, 255, 255))],
  "blue":  [((95, 100, 50), (130, 255, 255))],
}
def dominant_colour(bgr, roi=None, min_px=400):
    img = bgr if roi is None else bgr[roi[1]:roi[3], roi[0]:roi[2]]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    counts = {}
    for name, ranges in HSV.items():
        m = sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) for lo, hi in ranges)
        counts[name] = int((m > 0).sum())
    best = max(counts, key=counts.get)
    return best if counts[best] >= min_px else None
```

### 5.7 Skip tonight
- Generative augmentation (NICE inpainting: π0 53% → 64%, no runtime reported; Cosmos-Transfer). Too slow on Intel hardware in hours.

---

## 6. Anomaly detection

### 6.1 Recommended approach
**Anomalib PaDiM or PatchCore**, trained on **50–200 "good" crops from one fixed inspection pose**, exported to OpenVINO, gating a reject/sort action. Runs **separately from the policy**.

- Intel's own robot demo: a Dobot arm with **PaDiM, resnet18, 256×256, 1 epoch, F1AdaptiveThreshold** ([Intel article](https://www.intel.com/content/www/us/en/developer/articles/training/defect-detection-with-anomalib.html)), which is almost exactly this setup.
- Anomalib v2.7.0 has 33 image models, including PatchCore, PaDiM, EfficientAd, Dinomaly, SuperSimpleNet, UniNet, and few/zero-shot **AnomalyDINO, WinCLIP, FoundAD, AnomalyVFM, VLM-AD** ([models](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/index.html)).
- XPU: single-GPU training, tested on Arc 750/770 ([XPU guide](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/training_on_intel_gpus/index.html)).
- Low-latency option: **EfficientAD ~2 ms/image** on GPU ([arXiv 2303.14535](https://arxiv.org/abs/2303.14535)).

### 6.2 Code
```bash
pip install "anomalib[xpu,openvino]"
```
Dataset layout:
```
data/
  good/   # 50–200 images of normal objects at the inspection pose
  bad/    # optional: a few defective examples (for threshold calibration/eval)
```
```python
from anomalib.data import Folder
from anomalib.models import Patchcore            # or Padim()
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.deploy import ExportType, OpenVINOInferencer

dm = Folder(name="parts", root="data", normal_dir="good", abnormal_dir="bad",  # abnormal_dir=None if none
            normal_split_ratio=0.2, train_batch_size=32, eval_batch_size=32)
model = Patchcore()
eng = Engine(max_epochs=1, strategy=SingleXPUStrategy(), accelerator=XPUAccelerator())
eng.fit(datamodule=dm, model=model)
eng.export(model=model, export_type=ExportType.OPENVINO)

inf = OpenVINOInferencer(path="results/.../weights/openvino/model.bin")  # use the path export prints
p = inf.predict(image=frame)   # p.pred_score, p.pred_label, p.anomaly_map
```
CLI equivalents:
```bash
anomalib train --model Padim --data anomalib.data.Folder --data.root data ... \
  --trainer.accelerator xpu --trainer.strategy xpu_single
anomalib export --model Patchcore --ckpt_path <ckpt> --export_type OPENVINO
```
⚠️ The script is assembled from several docs pages; expect small API tweaks.

### 6.3 Few-shot / zero-shot alternatives

| Situation | Use |
|---|---|
| ≥50 good images of a fixed object type | **PatchCore / PaDiM** |
| 1–10 good images, or object type changes | **AnomalyDINO** (96.6% AUROC 1-shot, training-free, DINOv2 patch kNN; [arXiv](https://arxiv.org/abs/2405.14529v3)) or **WinCLIP+** (93.1 vs PatchCore 83.4 AUROC at 1-shot; 95.2 vs 88.8 at 4-shot; [arXiv](https://arxiv.org/pdf/2303.14814)) |
| Explaining a defect in the demo | A VLM (e.g. SmolVLM/Qwen-VL) **only as narrator**, not detector. GPT-4o averaged 74.9% on MMAD, "far short of industrial requirements" ([arXiv](https://arxiv.org/abs/2410.09453)) |

Zero-shot CLIP methods vary a lot by benchmark (AnomalyCLIP 0.916 vs WinCLIP 0.612 image AUROC in one re-evaluation), and gains flatten after 1–2 shots ([arXiv 2601.13440](https://arxiv.org/html/2601.13440v1)).

**Hand-rolled AnomalyDINO-style detector (if Anomalib is fighting you):**
```python
import torch, torch.nn.functional as F
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to("xpu").eval()
@torch.no_grad()
def patch_feats(img):                       # img: (1,3,H,W) normalised, H,W multiples of 14
    f = dino.forward_features(img.to("xpu"))["x_norm_patchtokens"][0]   # (N, D)
    return F.normalize(f, dim=-1)
memory = torch.cat([patch_feats(x) for x in good_images])            # (M, D)
@torch.no_grad()
def score(img, k=1):
    f = patch_feats(img)                                               # (N, D)
    d = 1 - f @ memory.T                                               # cosine distance
    nn = d.topk(k, largest=False).values.mean(-1)                      # per-patch anomaly
    return nn.max().item(), nn                                         # image score, patch map
```

### 6.4 RGB-D: depth height-map check
- On MVTec 3D-AD, colour-only methods beat 3D deep methods; FPFH (classical 3D) alone also did well but misses colour/texture defects; **fusing colour + 3D (BTF) was best at 99.3% pixel AUROC** ([arXiv 2203.05550](https://arxiv.org/abs/2203.05550v3)).
- **Practical version:**
```python
import numpy as np
def height_features(depth_m, table_depth_m, mask):
    h = np.clip(table_depth_m - depth_m, 0, None) * mask       # height above table, metres
    return np.array([h.max(), h.sum(), mask.sum()])            # max height, ~volume, footprint
# Fit mean/std on good samples; flag if any |z| > 3. Final decision = RGB_flag OR depth_flag.
```
Catches wrong size, missing parts, stacked or tipped objects.

### 6.5 Robustness of the detector
- In the RAD benchmark, **memory-bank methods (PatchCore) were the most robust to lighting and viewpoint changes**; flow/distillation methods were hurt by light spots. **Viewpoint misalignment caused false positives for every method** ([arXiv 2406.07176](https://arxiv.org/pdf/2406.07176)).
- **Mitigations:**
  - Fixed inspection pose + crop around the object.
  - Lock RealSense exposure and white balance.
  - Collect good samples under the real venue lighting (and different times of day).
  - Add brightness/colour-jitter copies of the good images.
  - Recalibrate the threshold on site.

### 6.6 Wiring into the robot loop
1. Pick the object → place at a fixed inspection spot (or hover at a fixed pose) → **wait until the arm is still** (servo `Moving` register = 0).
2. Grab **3 frames**, crop the ROI, take the **median score**.
3. **Threshold:** max score over held-out good images + a margin, or the F1-adaptive threshold if you have defects.
4. Route: good bin vs reject bin. Detector inference takes milliseconds, so it doesn't slow the loop.
5. **Demo:** overlay `anomaly_map` as a heatmap on the camera feed.

```python
def inspect(get_frame, inferencer, threshold, roi):
    scores, maps = [], []
    for _ in range(3):
        f = get_frame()[roi[1]:roi[3], roi[0]:roi[2]]
        p = inferencer.predict(image=f)
        scores.append(float(p.pred_score)); maps.append(p.anomaly_map)
    s = float(np.median(scores))
    return s > threshold, s, maps[int(np.argsort(scores)[1])]
```

---

## 7. Force & proprioceptive sensing

### 7.1 Free force proxy: servo current (do this first)
- One SO-101 project put **gripper `Present_Current`** (30 Hz) into ACT's state: **77.5% → 95% success** vs a zeroed-current baseline, and current separated soft TPU from rigid PLA objects ([source](https://vibhusiddha.vercel.app/so101-research)).
- In LeRobot, any `float` entry in the robot's `observation_features` becomes a dimension of `observation.state` automatically (`hw_to_dataset_features`).

**STS3215 registers** (LeRobot Feetech table / [Waveshare memory map](https://docs.waveshare.com/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation)):

| Register | Address | Size / unit |
|---|---|---|
| `Present_Velocity` | 58 | 2 bytes |
| `Present_Load` | 60 | 2 bytes, 0.1% |
| `Present_Temperature` | 63 | 1 byte |
| `Moving` | 66 | 1 while moving |
| `Present_Current` | 69 | 2 bytes, ~6.5 mA/count ⚠️ |
| Protection current | 28 | ~6.5 mA/count |

```python
from functools import cached_property
class SOFollowerForce(SOFollower):
    @cached_property
    def observation_features(self):
        return {**self._motors_ft, "gripper.cur": float, "gripper.load": float, **self._cameras_ft}
    def get_observation(self):
        obs = super().get_observation()
        obs["gripper.cur"] = float(self.bus.read("Present_Current", "gripper", normalize=False))
        v = self.bus.read("Present_Load", "gripper", normalize=False)
        obs["gripper.load"] = float((v & 0x3FF) * (-1 if v & 0x400 else 1))   # sign-magnitude ⚠️
        return obs
# action_features stays = _motors_ft (never add force to actions)
```
- ⚠️ **Unverified:** `Present_Load` bit layout (bit 10 = direction) and the 6.5 mA unit; whether your pinned LeRobot uses `tables.py` or the newer `SerialMotorsBus`. Newer main also has `MotorCurrentProcessorStep` (`bus.sync_read("Present_Current")`).
- Each extra serial read costs a few ms. Read only the gripper, or `sync_read` all joints once.
- **Gripper is capped by `configure()`:** `Max_Torque_Limit=500`, `Protection_Current=250`, `Overload_Torque=25`. Current saturates near the protection limit, so it's good for on/off contact but useless for fine force.

### 7.2 Grasp verification (no retraining)
```python
def grasp_state(grip_pos, grip_cur, closed_pos, pos_tol, cur_thresh):
    if grip_pos <= closed_pos + pos_tol and grip_cur < cur_thresh:
        return "EMPTY"     # closed all the way with no resistance → re-open and retry
    if grip_cur >= cur_thresh:
        return "HOLDING"
    return "UNKNOWN"
# During lift: sudden current drop while HOLDING → "SLIP"
```
The same "fully closed = nothing grasped" override improved disturbance recovery in [arXiv 2503.23835](https://arxiv.org/html/2503.23835v1).

### 7.3 Other uses
1. **Sorting by softness/material:** holding current at a fixed close command separates soft from rigid. **Weight shows up in `shoulder_lift`/`elbow` load during the lift**, not at the gripper.
2. **Anomaly/collision detection:** log current and load across good episodes, then fit per-timestep mean and std and flag |z| > 3. Collisions show as spikes on arm joints.
3. **Contact-rich insertion:** add force to state and include demos that recover from jamming. A wrist F/T sensor helps most here.

### 7.4 Literature for adding force to policies
- **FACTR** (RSS'25): joint torques → MLP → a single "force token" next to the image tokens in an ACT-style transformer. Images are heavily blurred early in training (curriculum) so the policy doesn't ignore force. ~40–43% gains on unseen objects ([arXiv 2502.17432](https://arxiv.org/html/2502.17432v2)).
- **Feel the Force:** the policy predicts a target gripper force; a PD loop closes until reached. 77% success on force-sensitive tasks ([arXiv 2506.01944](https://arxiv.org/abs/2506.01944)).
- **ACT** supports `observation.environment_state` as a separate token, which is a FACTR-like way to keep force from being drowned out. **Diffusion Policy** concatenates state + env state. **SmolVLA** pads state to `max_state_dim=32` (fits, but its pretrained projection hasn't seen the extra dims).

### 7.5 Friend's sensor & pitfalls
- **Identify it first:** wrist F/T sensor, fingertip FSR/tactile, or load cell.
- **Sync:** sample inside `get_observation()` so it shares the frame timestamp. For 100 Hz–1 kHz sensors take the latest value or a short window mean per 30 Hz frame. Don't record a separate async stream.
- **Arduino/serial sensors:** read in a background thread and keep the latest value; **tare at the start of every episode** (zero drift).
- **Noise:** servo current is quantised, spiky and motion-dependent. Use a median of 3–5 or EMA (α≈0.3), **identically at record and inference time.**
- **Normalisation:** recompute dataset stats after adding dims. A near-constant channel gets a tiny std and its normalised values blow up, so clip or use min-max. A Paxini/LeRobot guide warns that without normalisation "the model will learn to ignore the tactile signal"; start with a scalar total force before any pressure map ([guide](https://roboticscenter.ai/learn/paths/paxini-gen3/unit-5-train-deploy)).
- **Heat/saturation:** stalled grippers sit at protection current and heat up; current drifts with temperature. Keep grasps short.

---

## 8. Putting it together: system architecture

```
             ┌─────────────────────────────── perception (outside policy) ───────────────────────────────┐
RealSense ──►│ HSV/classifier → target colour & bin     Anomalib (OpenVINO) + depth height check → OK/NG │
 (RGB+D)     └──────────────┬───────────────────────────────────────────────┬────────────────────────────┘
                            │ task id / prompt                              │ route decision
                            ▼                                               ▼
 joint state + ──► ACT policy (OpenVINO, AsyncExecution, chunk 50–100) ──► SO-101 follower
 gripper current             ▲                                               │
                             └──── grasp check (current + position) ◄───────┘  → retry if EMPTY / SLIP
```

**Task state machine (sorting + anomaly):**
1. `DETECT`: find the next object and its colour (HSV) → choose policy/prompt.
2. `PICK`: run the pick policy until the gripper closes.
3. `VERIFY_GRASP`: current check; on EMPTY, retry (max 2).
4. `INSPECT`: move to the inspection pose, wait for `Moving=0`, run 3-frame anomaly scoring + depth check.
5. `PLACE`: route to the colour bin, or the reject bin if anomalous.
6. Loop.

---

## 9. Evaluation checklist
- [ ] `torch.xpu.is_available()` is True; Studio/LeRobot runs a `fast_dev_run` on XPU.
- [ ] RealSense streams at 640×480@30 with **auto-exposure/WB locked**.
- [ ] ≥150 demos across ≥6 conditions; one lighting + one background **held out**.
- [ ] Idle frames trimmed; slow/retry episodes dropped.
- [ ] Augmentation on, **hue ≈ 0** if colour matters.
- [ ] ACT trained in bf16; `update_s` timed; checkpoint exported to OpenVINO; latency benchmarked.
- [ ] Async execution working, with no jerks at chunk boundaries.
- [ ] OpenVINO IR is static-shape; `benchmark_app -pc` shows SDPA/GatedMLP fusions; median + p99 latency logged.
- [ ] VLA track: first 200 training steps timed; go/no-go on π0.5 vs SmolVLA within the first hour; VLA beats ACT on held-out lighting before it's used in the demo.
- [ ] Anomaly threshold calibrated **on site**; false-positive rate checked on 20+ good samples.
- [ ] Grasp check triggers retries correctly.
- [ ] **10+ trials per condition** for any claim (WARP authors: 1/20 differences are noise).
- [ ] Demo video: heatmap overlay, live lighting change, throughput (tasks/min).

## 10. Risks & fallbacks

| Risk | Fallback |
|---|---|
| `bf16-mixed` / XPU strategy errors in Studio | `precision: 32`; or train with LeRobot `--policy.device=xpu` |
| LeRobot version lacks new augmentations | Use `ColorJitter` + `SharpnessJitter` only, or pass torchvision v2 transforms |
| OpenVINO export fails | PyTorch on XPU with `torch.compile(mode="max-autotune-no-cudagraphs")` + manual `torch.xpu.graph` (not `reduce-overhead`, a no-op on XPU); or ONNX backend in `InferenceModel` |
| π0.5 doesn't fit / trains too slowly | SmolVLA (`train_expert_only` / frozen vision encoder); ACT stays the baseline |
| INT8/INT4 hurts policy accuracy | Check trajectory MSE vs PyTorch; drop to INT8 weight-only or FP16; never generic INT4 on the action expert |
| A Chinese VLA won't run on XPU | Don't port during the event; stick to LeRobot-native X-VLA/EVO1 smoke tests only |
| Training too slow | Fewer steps (10–15k), smaller batch, fewer cameras, lower resolution |
| Policy fails under new lighting | Record 15–20 demos in that condition and fine-tune; don't just train longer |
| Anomaly false positives | Tighten pose/crop, lock exposure, add jittered good images, raise threshold |
| Servo register names differ | Check `lerobot/motors/feetech/tables.py` in your install |
| Friend's sensor is noisy/laggy | Use only gripper current |

---

## 11. Sources

**Provided links**
- [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)
- [open-edge-platform/robotics-ai-suite](https://github.com/open-edge-platform/robotics-ai-suite)
- [openvinotoolkit/physicalai](https://github.com/openvinotoolkit/physicalai)
- [open-edge-platform/physical-ai-studio](https://github.com/open-edge-platform/physical-ai-studio)
- [WARP-RM paper (arXiv 2606.28320)](https://arxiv.org/pdf/2606.28320), [code](https://github.com/uynitsuj/WARP-RM), [project page](https://uynitsuj.github.io/warp-rm/)
- [Flash ALOHA](https://flash-aloha.github.io)

**XPU inference optimisation & VLAs** (details and full citations in `reports/XPU kernels and Chinese VLAs.md`)
- [Intel: Optimizing π0.5 on Panther Lake (paper)](https://docs.openedgeplatform.intel.com/shared_media/publication-optimize-pi0.5-paper.pdf) · [pi05-rtc-ov pipeline](https://github.com/open-edge-platform/robotics-ai-suite/tree/main/src/pipelines/pi05-rtc-ov)
- [Running VLAs at Real-time Speed](https://arxiv.org/html/2510.26742v1) · [SnapFlow](https://arxiv.org/pdf/2604.05656) · [QuantVLA](https://arxiv.org/pdf/2602.20309) · [HoloQ-VLA](https://arxiv.org/pdf/2605.28803)
- [intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton) · [torch-xpu-ops #5338 (reduce-overhead on XPU)](https://github.com/intel/torch-xpu-ops/issues/5338) · [PyTorch 2.8 + TorchAO on Intel AI PCs](https://pytorch.org/blog/pytorch-2-8torchao-unlock-efficient-llm-inference-on-intel-ai-pcs/)
- [RoboDojo benchmark](https://arxiv.org/html/2607.04434v1) · [X-VLA](https://arxiv.org/html/2510.10274v1) · [LeRobot hardware guide](https://huggingface.co/docs/lerobot/main/hardware_guide)

**Training / speed**
- [LeRobot device_utils.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/utils/device_utils.py)
- [LeRobot hardware guide](https://github.com/huggingface/lerobot/blob/main/docs/source/hardware_guide.mdx)
- [PyTorch XPU getting started](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.md)
- [DemoSpeedup](https://arxiv.org/abs/2506.05064) · [SAIL](https://arxiv.org/abs/2506.11948)

**Robustness**
- [Data Scaling Laws in Imitation Learning](https://arxiv.org/html/2410.18647v3)
- [LeRobot dataset / image transforms docs](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3)
- [GreenAug](https://arxiv.org/html/2407.07868v2) · [Random overlay](https://arxiv.org/pdf/2011.13389)
- [What makes pre-trained visual representations robust](https://paperswithcode.com/paper/what-makes-pre-trained-visual-representations)
- [SO-101 VLA benchmark](https://arxiv.org/html/2606.08881) · [VLA object–region generalisation](https://www.alphaxiv.org/abs/2602.24143.md)
- [DP3](https://3d-diffusion-policy.github.io) · [NICE](https://arxiv.org/html/2511.22777v1) · [NVIDIA Cosmos SO-101](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/14-strategy3-cosmos.html)

**Anomaly detection**
- [Anomalib](https://github.com/open-edge-platform/anomalib) · [model list](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/index.html) · [XPU guide](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/training_on_intel_gpus/index.html) · [Folder datamodule](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/data/image/folder.html)
- [Intel: Defect detection with Anomalib (robot arm demo)](https://www.intel.com/content/www/us/en/developer/articles/training/defect-detection-with-anomalib.html)
- [EfficientAD](https://arxiv.org/abs/2303.14535) · [WinCLIP](https://arxiv.org/pdf/2303.14814) · [AnomalyDINO](https://arxiv.org/abs/2405.14529v3) · [VLM AD analysis](https://arxiv.org/html/2601.13440v1) · [MMAD](https://arxiv.org/abs/2410.09453)
- [BTF / MVTec 3D-AD](https://arxiv.org/abs/2203.05550v3) · [RAD robustness benchmark](https://arxiv.org/pdf/2406.07176)

**Force**
- [SO-101 gripper-current study](https://vibhusiddha.vercel.app/so101-research)
- [Waveshare ST servo memory map](https://docs.waveshare.com/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation)
- [FACTR](https://arxiv.org/html/2502.17432v2) · [Feel the Force](https://arxiv.org/abs/2506.01944) · [Gripper-state ambiguity](https://arxiv.org/html/2503.23835v1)
- [Paxini tactile + LeRobot guide](https://roboticscenter.ai/learn/paths/paxini-gen3/unit-5-train-deploy)
- [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101)
