# SO-101 Robot Learning Hackathon Playbook (Intel XPU)

*Compiled 13 Nov 2026 from the official links plus research threads on anomaly detection, robustness, speed, force sensing, XPU inference optimisation and VLAs (full report in `reports/XPU kernels and Chinese VLAs.md`). Updated after kickoff with Intel's official challenge brief and the real hardware.*

## Challenge brief (Intel @ Junction, from the kickoff slide)

- Build an **end-to-end Physical AI solution for robotic defect detection and intelligent manipulation** using **Intel Physical AI Studio, Anomalib and OpenVINO**.
- Combine **VLA capabilities through Physical AI Studio** with **Anomalib-based visual anomaly and defect detection**.
- The robot must **perceive objects, identify defects, reason about the required action, and execute the manipulation**.
- **Optimise and deploy on Intel Core Ultra Series 3 using OpenVINO**, leveraging **CPU, GPU and NPU** to improve inference performance, latency and deployment efficiency.
- **Primary task:** distinguish acceptable ("perfect") from anomalous ("defective") objects, then physically sort or respond according to the **rules revealed at kickoff**. ⚠️ Fill in the rules here: _TBD_.
- **Closed loop:** **PERCEIVE → DETECT → REASON → ACT → OPTIMIZE**, demonstrated end to end.

**End goal: Intel's reference pipeline** ("End Goal" slide):

```
 brick on the table ──ACT──► picked and shown to the inspection camera ──Anomalib──► perfect / defective ──SmolVLA──► defective → one plate
                                                                                                                   perfect   → the other plate
```

- The demo setup has **two plates (orange/red and blue)**. In the slide's photos the perfect brick ends on the blue plate. ⚠️ Confirm the plate-to-label mapping in the kickoff rules.
- Intel's own reference split: **ACT does the pick-and-present skill, SmolVLA does the language-conditioned place** ("place the brick on the … plate").
- Plate mapping in the slide photos: **perfect → blue plate**, **defective → pink/red plate**. ⚠️ Confirm in the kickoff rules.

**Our concrete task (team decision): "defective" = a brick marked with a cross.**
- We draw a **cross (✗) on some bricks** and leave the others **plain**.
- **Anomalib is trained only on plain bricks**, so a cross is the anomaly. Anomalib decides; π0.5 only moves:
  - **plain → "perfect" → blue plate**: `"place the brick on the blue plate"`;
  - **cross → "defective" → pink plate**: `"place the brick on the pink plate"`.
- The anomaly map's hot region is the cross itself, which gives the **localization signal** for free (show it as the heatmap in the demo).
- Further extension: numbered bricks (Section 8.2).
- **Our choice (team decision): no ACT. One VLA, π0.5, does every movement**, switched by instruction:
  - `"pick up the brick and show it to the inspection camera"`;
  - `"place the brick on the blue plate"`;
  - `"place the brick on the pink plate"`.

  Anomalib decides between the two place instructions. This makes the VLA the whole "ACT" stage, which is what the 20-point VLA criterion rewards. SmolVLA (Intel's reference VLA) is the fallback, doing the same job.

**Judging criteria (100 points, deck slide 7):**

| Points | Criterion | What it means for us |
|---|---|---|
| **25** | End-to-end Physical AI solution | One **integrated closed loop**, perception → defect detection → decision → SO-101 action, "rather than independent component demonstrations". Run it as one program/state machine (Section 8) |
| **20** | Defect detection with Anomalib | Detection quality, model integration, normal vs defective handling, **how defect outputs are exposed to and consumed by the downstream workflow**: the `{label, score, region}` signal feeding the instruction |
| **20** | VLA & Physical AI Studio integration | Studio + VLA reasoning turn perception results and task instructions into robot actions; show integration across **perception, reasoning, planning and manipulation** |
| **20** | OpenVINO & Core Ultra 3 optimization | **Latency, throughput, deployment efficiency**, and appropriate use of **NPU, GPU and CPU**: the Section 8.1 table |
| **10** | Robotic execution & reliability | Object localization, manipulation accuracy, repeatability, stable task completion, and **recovery from incorrect or uncertain defect detections** (Section 8, uncertainty band) |
| **5** | Innovation & technical demonstration | Originality, use of Intel tech, completeness, clarity of the live demo, architecture and optimization explanation |

**What teams must deliver** (same slide):
1. **Working Physical AI prototype:** a closed-loop SO-101 solution from perception to action.
2. **VLA + Anomalib integration:** a VLA robotics workflow with visual anomaly/defect detection.
3. **OpenVINO-optimized deployment:** an optimized inference pipeline on Intel Core Ultra Series 3.
4. **Live demonstration:** execute the **Challenge-Day task** and explain the architecture and optimization choices.

**Technical implementation expectations** (kickoff slide; treat as the judging rubric):

| # | Expectation | Slide wording (condensed) | Where we cover it |
|---|---|---|---|
| 1 | Perception & defect detection | Capture workspace imagery, apply Anomalib; defect output available to the policy as a **structured signal: label, score, or localized region** | Section 8 DETECT: `{label, score, box, table_xyz}` |
| 2 | VLA reasoning & task planning | **Use Physical AI Studio to construct the VLA-driven application flow.** The VLA interprets the **task instruction and perception results**, determines the response, and produces the **action intent** for the robot controller | Section 8 REASON → π0.5 instruction; Studio flow (4.0) |
| 3 | SO-101 execution | Turn the action into robot behaviour: motion, grasping, sorting or other challenge-specific manipulation | π0.5 + grasp check |
| 4 | OpenVINO optimization | Model conversion, **precision, graph optimization, preprocessing/postprocessing efficiency, device selection** → lower **end-to-end** latency | Sections 4.5, 8.1 (incl. pre/post-processing rows) |
| 5 | Core Ultra Series 3 deployment | Use **NPU, GPU and CPU according to workload characteristics and operator support**, while keeping robot control and the app responsive | Section 8 device split |
| 6 | Closed-loop demonstration | Full sequence from image acquisition and defect detection through VLA reasoning to robot action, **then a concise explanation of the architecture and optimization decisions** | Section 9 demo + pitch |

**Provided hardware/software** (HW/SW slide):
- **Compute:** HP OmniBook Ultra (C2NM8AV) or **Dell XPS 16 (DA16260)**, both Core Ultra X9 388H. We have the XPS.
- **Arm:** LeRobot SO-ARM101, assembled: 1 leader + 1 follower + **1 wrist camera**.
- **Cameras:** **one RealSense D415** and **one Pro QHD 2K/60 fps webcam**, plus the arm's wrist camera = **three cameras**: `wrist`, `top`, `inspect` (Section 1.5).
- **Extras:** camera desk mounts, a **USB hub**, **Lego/Duplo bricks and plates** (the objects and the destinations), printed robot safety guidelines.
- **Software to install as part of the challenge:** Physical AI Studio and **Anomalib v2.7.0** (both done on our XPS, Section 1.0).
- **Pre-installed:** Ubuntu 26.04 and NPU/GPU drivers. On our XPS the GPU *compute* runtime and the NPU user-space driver were still missing, and we installed them.
- A **GPU cloud instance via [Modal](https://modal.com)** is provided for training. ⚠️ Modal is serverless, not a plain SSH box, so Studio's SSH remote-trainer flow may not apply directly. Plan to run Studio's training library (`physicalai fit`) or LeRobot in a Modal job instead (Section 4.0).

**Required technology stack** (second kickoff slide):

| # | Component | Role per the slide | Our use |
|---|---|---|---|
| 01 | **SO-101 arm** | Physical platform; "closes the loop from AI perception and reasoning to real-world action" | Follower + leader for demos |
| 02 | **Intel Physical AI Studio** | "Design and integrate the VLA-based robotics workflow, connecting perception, task reasoning, robot control, and application orchestration" | **Studio is the hub:** environment setup, recording, remote training on RTX, model export, running the policy (Section 4.0) |
| 03 | **Anomalib** | "Visual anomaly and defect detection, producing **defect decisions and localization signals** that can be **consumed by the VLA / robot workflow**" | Score → perfect/defective, **and** the anomaly map → defect location → fed into REASON (Section 8) |
| 04 | **OpenVINO + Core Ultra Series 3** | "Optimize and deploy supported AI workloads with OpenVINO. **Map inference across the integrated NPU, GPU, and CPU** to improve latency and on-device efficiency" | Anomalib on NPU, π0.5 on GPU, control on CPU; OPTIMIZE table |

**What this means for us:** the judged stack is Studio + Anomalib + OpenVINO. Plain LeRobot is fine for data and baselines, but the final policy is trained via Physical AI Studio and deployed with the `physicalai` runtime on OpenVINO. The NPU must visibly do real work, and we must show before/after optimisation numbers.

**Team ontology (follow `dlsj`, Desmond's code):**
- **Cameras:** `wrist`, `top`, `inspect`.
- **Objects:** bricks.
- **Destinations:** plates.
- **Workflow:** **pick a brick → present it to the `inspect` camera → Anomalib decides plain (perfect) vs cross-marked (defective) → choose the plate → place it**.
- **Tools:**
  - recording: `dlsj.teleop_record` (leader/follower) and `dlsj.kinesthetic_web` (hand-guided);
  - calibration: in `calibration/` via `dlsj.arms`;
  - `dlsj.deploy.act_export` (ACT → OpenVINO) exists but is **not used**: the team dropped ACT.

**Hardware:**
- **Robot:** SO-101 leader/follower arms.
- **Cameras (team naming, from `dlsj`):** `wrist` (on the gripper), `top` (fixed top-down view of the table) and `inspect` (fixed **side** camera the arm presents a picked brick to). Kit devices: RealSense D415 + Pro QHD 2K webcam + the arm's wrist camera. ⚠️ Confirm which device is `top` and which is `inspect`.
- **Deploy machine:** **Dell XPS 16 DA16260**: Core Ultra **X9 388H** (Panther Lake, Core Ultra Series 3), **Arc B390** iGPU, **NPU** ("Intel AI Boost"), **64 GB** RAM, Ubuntu 26.04. Reach it with `ssh xps` over Tailscale (Section 1.0).
- **Training:** **cloud / team NVIDIA RTX GPUs** (CUDA).
- Optional force sensor from a friend.

> Items marked ⚠️ **Unverified** come from docs/source reading or from extrapolation and were not tested on hardware. Smoke-test them first.

---

## 0. TL;DR: the plan

| Loop step | Component | Choice | Device |
|---|---|---|---|
| **PERCEIVE** | Cameras | `wrist`, `top`, `inspect` at 640×480@30, exposure/WB locked; the arm picks a brick and **presents it to `inspect`** | CPU |
| **DETECT** | Defect detection | **Anomalib PaDiM/PatchCore** on the **`inspect` view of the presented brick**, OpenVINO, **static shapes**; anomaly map → **defect localization** (region on the brick) | **NPU** |
| **REASON** | Decision | Kickoff rules + defect result (+ brick colour/type) → **which plate** → a **language instruction**, e.g. "place the brick on the reject plate" | CPU |
| **ACT** | Manipulation | **π0.5 does every movement**, by instruction: pick and show to `inspect`, then place on the blue or pink plate. Fine-tuned via Physical AI Studio on cloud GPUs (Modal), exported to OpenVINO **INT8**, `physicalai` runtime with RTC/async. No ACT | **GPU** |
| **OPTIMIZE** | Evidence | Latency table: PyTorch-XPU vs OpenVINO FP16 vs INT8 vs NPU; per-stage timings; tasks/min (Section 8.1) | all |
| *Phase 2* | Numbered sorting | Number reader on `top` (NPU) → planner orders bricks → π0.5 picks the **marked** brick → number + defect check on `inspect` → place in order (Section 8.2) | NPU/CPU/GPU |

| Component | Choice | Why |
|---|---|---|
| Policy (all movement) | **π0.5** via Physical AI Studio, one model, three instructions. **No ACT.** Fallback **SmolVLA** (Intel's reference VLA) | Required "VLA through Studio" (20 pts); language selects the skill and plate; best independent results; Intel-measured OpenVINO path (172 ms/chunk on Panther Lake). Section 2.1 |
| Training | **Cloud RTX** (CUDA, bf16), Studio CLI or Studio trainer container | π0.5 needs ≥24 GB to fine-tune; XPU training is fallback only |
| Deployment | **OpenVINO export** + `physicalai` runtime on the XPS, **AsyncExecution**, **no temporal ensembling** | Required stack; low latency |
| Data | ~150 brisk demos over **6–10 lighting/background conditions**, idle frames trimmed; demos **with language instructions** per bin | Diversity beats quantity; π0.5 learns the instruction → bin mapping |
| Robustness | Image augmentation (brightness/contrast/shadow/white-balance), **hue jitter ≈ 0** | Lighting robustness without destroying colour info |
| Force | **Gripper `Present_Current`** + rule-based grasp verification | One SO-101 project reported 77.5% → 95% success |
| Stretch | π0.5 action expert on the NPU (Intel's split), SnapFlow 1-step distillation, depth background swap | Only with spare hours |

### Suggested timeline (adapt to remaining hours)
1. **Hour 0–1:** XPS is already set up (Section 1.0). Verify the arm and all three cameras (lock exposure/WB, fix the `top` and `inspect` mounts). Write the kickoff rules into the brief above. One person starts collecting demos.
2. **Hour 1–3:** collect diverse demos with language instructions (Section 3); in parallel, capture good and defective images of bricks **held at the `inspect` pose** (Section 6).
3. **Hour 3–5:** push the dataset to the Hub; **fine-tune π0.5 on the Modal GPU instance**; **train Anomalib on the XPS** and export to the NPU.
4. **Hour 5–7:** export π0.5 to OpenVINO INT8, deploy with the `physicalai` runtime; wire the REASON step and the grasp check.
5. **Hour 7+:** test on held-out lighting/background; iterate on failures (more demos of the failure condition beats more training); fill in the OPTIMIZE table.
6. **Final:** demo of the full loop with the anomaly heatmap, live per-stage latency readout, and the OPTIMIZE table.

---

## 1. Toolchain & environment setup

### 1.0 The XPS deploy machine (already set up)

**Access:** the XPS is on a teammate's Tailscale network (`iuredce@gmail.com`). Switch to it with `tailscale switch`, then `ssh xps`, which logs in as user `intel` at 100.77.191.42. Only `intel` is allowed by that network's SSH policy.

**System (installed 13 Nov):**
- Intel GPU compute runtime (`libze1`, `libze-intel-gpu1`, `intel-opencl-icd`).
- **NPU driver v1.38.0** (`intel-level-zero-npu`, `intel-driver-compiler-npu`, `intel-fw-npu`).
- Docker, plus `build-essential`.
- `intel` is in the `render`, `video`, `dialout` and `docker` groups, and has linger enabled.
- OpenVINO sees **CPU, GPU, NPU**.

**Python environments** (in `~/inference`, all Python 3.12 via `uv`). They are separate because the packages pin conflicting versions:

| Env | Contents | Use |
|---|---|---|
| `physical-ai-studio/library/.venv` | Physical AI Studio library from the repo lock (`uv sync --extra xpu`): `physicalai` CLI, torch 2.11+xpu, OpenVINO 2026.4 | Studio export and the `physicalai` runtime on the XPS |
| `.venv-anomalib` | Anomalib 2.7.0, OpenVINO 2026.4.1, torch 2.15+xpu | Defect detection (train and export to NPU) |
| `.venv` | LeRobot 0.7.0 (π0.5, SmolVLA, Feetech, RealSense, policy server), torch 2.15+xpu | Recording, dataset tools, the **unoptimised PyTorch baseline** for the OPTIMIZE table |

**Studio UI:** runs in Docker on the XPS (`~/inference/physical-ai-studio/application/docker`, profile `xpu`) at **http://100.77.191.42:7860**.

**Install gotchas we hit:**
- `pip install physicalai-train[xpu]==0.3.0` is unresolvable: it pins `lerobot==0.6.0` (torchvision <0.27) but `xpu` needs torchvision 0.29. **Install from the repo lock** instead.
- Anomalib 2.7 and physicalai-train 0.3 conflict on `jsonargparse`.
- physicalai 0.4 and LeRobot 0.7 `[pi]` conflict on `transformers`.
- LeRobot 0.7's policy-server extra is `remote` (`async` no longer exists). Its `hardware` extra needs a C compiler (`evdev`).
- π0.5 needs a Hugging Face token with access to the gated `google/paligemma-3b-pt-224` (tokenizer). Set `HF_TOKEN`.

**Measured on the XPS** (ResNet-18, 256², batch 1, `bench/device_smoke.py`): CPU 12.6 ms, **GPU FP16 0.59 ms, NPU 0.88 ms**, PyTorch-XPU eager fp32 2.17 ms. **The NPU only accepts static shapes:** export with a fixed `input=[1,3,H,W]`, or it fails with "Upper bounds are not specified".

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

### 1.5 Camera setup (`wrist`, `top`, `inspect`)
`dlsj` tools take cameras as `--cameras wrist=<idx>,top=<idx>,inspect=<idx>` (OpenCV indices, 640×480@30 by default). They read every camera through OpenCV, so the **D415 gives RGB only** there. Use `pyrealsense2` only if you add depth.

| Name | Mount | Job |
|---|---|---|
| `wrist` | On the gripper (comes with the arm) | Close-range view for grasping and placing |
| `top` | Fixed, top-down over the table | Where bricks and plates are; policy context |
| `inspect` | Fixed, **side-on**, at a set "present" spot | Anomalib defect detection on the brick the arm holds up |

**RealSense D415 (whichever of `top`/`inspect` it is):**
- Use `pyrealsense2`, or the runtime's `[realsense]` extra. You need the camera's serial number in configs.
- Use **640×480 @ 30 fps**. The kit includes a USB hub, but plug the **RealSense directly into a laptop USB-C/3.x port**, and put the arm and webcam(s) on the hub. RealSense through a shared hub causes frame drops.
- **It's a D415**, not a D405. ⚠️ The D415's minimum depth distance is roughly **0.3–0.45 m depending on resolution** (from memory of the spec, not verified), and accuracy improves with distance. Mount it so the **table is at least ~0.45 m away**, or check the depth at your height in `realsense-viewer`. Closer than its minimum, depth goes to 0/holes.
- **Coverage at 0.45 m:** depth FOV ~65°×40° ⇒ ≈ 0.57 m × 0.33 m. As `top`, make sure all plates and the pick area fit in view.
- The minimum depth distance only matters if you use depth. For RGB-only use, any distance works.
- D415 uses a rolling shutter: hold the brick still at the `inspect` pose before grabbing frames.
- **Mount `top` and `inspect` rigidly and never move them** between recording and deployment. A bumped camera hurts the policy and the anomaly detector more than any lighting change. Mark positions with tape.
- Optional depth (via `pyrealsense2`, aligned to colour with `rs.align(rs.stream.color)`): needed for the tricks in Sections 5.3, 5.5 and 6.4.

**`wrist` camera and the 2K webcam (RGB only):**
- The 2K/60 fps webcam: run it at **640×480@30** (or 1280×720 if there's bandwidth) to match the policy input. 2K is wasted on a 224² VLA input and costs USB bandwidth and resize time.
- Plain UVC camera, so lock exposure/WB with `v4l2-ctl` (e.g. `v4l2-ctl -d /dev/videoN -c auto_exposure=1 -c white_balance_automatic=0`; control names vary by camera, list them with `v4l2-ctl -d /dev/videoN -l`). **Re-check after every replug**; cheap webcams reset.
- Expect motion blur, rolling shutter and colour drift: it's the noisiest stream, so keep augmentation on for it (Section 5.2).

**Both:**
- **Lock exposure and white balance** (turn auto-exposure/auto-WB off) on all three. Auto-exposure changes images between frames and between runs. This matters most for `inspect` (Anomalib).
- **`inspect` lighting:** give it its own steady light (desk lamp) and a plain background behind the present spot, so defects, not shadows, drive the anomaly score.
- Same fps (30) on both; read the latest frame from each inside the same `get_observation()` and check their timestamps stay close.
- For VLA export, the camera count is **3** (Section 4.5).

---

## 2. Choosing a policy

| Policy | Training cost (50 eps) | Inference | Strengths | Weaknesses |
|---|---|---|---|---|
| **ACT** | ~30–60 min on RTX 4090, ~1–2 h on L4; expect similar or slower on Intel GPU | Fast, single forward pass per chunk | Simple, reliable, LeRobot's recommended first policy | No language; needs diverse data for robustness |
| **SmolVLA** | ~3–6 h on L4 | Flow matching; supports Real-Time Chunking | Language conditioning, multi-task | Pretrained state projection hasn't seen extra dims; slower |
| **π0.5** | Too slow to fine-tune in hours on one Intel GPU | Heavy; SO-101 SnapFlow recipe distils to 1 step | Strongest on SO-101 benchmark | Compute cost |

**SO-101 benchmark** ([arXiv 2606.08881](https://arxiv.org/html/2606.08881)), average success: **π0.5 56.25%**, ACT 33.75%, SmolVLA 32.5%. Selective colour sorting was **0–10% for all three**, which is why sorting needs a separate perception step (Section 5.6).

**Decision (after kickoff, updated): no ACT.** **π0.5 does every movement** (pick and show to `inspect`, place on the blue or pink plate), selected by instruction; **SmolVLA** (Intel's reference VLA) is the fallback. Train on the Modal GPU instance. The ACT rows above are background only.

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

### 3.1b Language instructions (for π0.5)
- One π0.5 model learns **three skills**, recorded as separate episodes, each with its own **task string**:
  1. `"pick up the brick and show it to the inspection camera"`: from a brick on the table to holding it still at the `inspect` spot;
  2. `"place the brick on the blue plate"`: from the `inspect` spot to the blue plate;
  3. `"place the brick on the pink plate"`: from the `inspect` spot to the pink plate. Use **one fixed string per destination** from the kickoff rules, and the exact same strings at run time (REASON step, Section 8).
- **Balance the demos across instructions**, and vary which object goes where, so the policy listens to the instruction instead of guessing from the object's look.
- With `dlsj.teleop_record` / `dlsj.kinesthetic_web` this is `--task "..."`. Record each instruction as its own `--repo-id`, or as a `--resume` batch, and merge.
- **Mix plain and cross-marked bricks in every skill's demos.** The policy must not learn "cross → go pink" by itself: the plate comes only from the instruction, which Anomalib's result chooses. Place demos for each plate should use both kinds of brick.
- Splitting at the `inspect` spot is what lets Anomalib + REASON choose skill 2 or 3. Record roughly **as many pick demos as place demos combined**, and the two place skills equally.
- Optional fourth skill for recovery: `"turn the brick and show it to the inspection camera again"` (Section 8, uncertainty band).

### 3.2 Demo quality = policy speed
- The policy copies your speed and hesitation. **Demo at the speed you want the robot to move**, decisively, with no fumbling. (Lesson from [Flash ALOHA](https://flash-aloha.github.io), which reached >80% success at near-human speed with 50 demos per task.)
- **Keep your hands and arms out of the camera view.** Flash ALOHA masks the operator because operator appearance hurts visual alignment. The `wrist` camera mostly avoids this, but **`top` and `inspect` will see you**: teleoperate from outside their frames.
- Discard failed or retried demos, or record them separately; don't train on them by default.
- Keep a consistent strategy: always approach from the same side and grasp the same way. Multimodal demos confuse ACT.

### 3.3 Data collection tooling
- **LeRobot:** `lerobot-record` with leader → follower teleop.
- **physicalai runtime:** `TeleopSource` (leader-follower) inside the runtime loop, or the Studio UI.
- Keep fps consistent (30 Hz) across all recording.

---

## 4. Training & speed on Intel XPU

### 4.0 Train on the cloud GPU (Modal), deploy on the XPS (Studio-orchestrated)
⚠️ The provided GPU instance is on **Modal**. Studio's remote trainer expects an **SSH host** (or AWS). Check with the Intel mentors whether their Modal setup exposes SSH or a Studio trainer. If not, run training as a Modal job: `physicalai fit` from the same Studio commit, or `lerobot-train --policy.type=pi05`. Then import the checkpoint into Studio and export from there.

**Preferred flow: everything driven from the Studio UI on the XPS** (http://100.77.191.42:7860). This is what the "Studio integrates the workflow" requirement wants to see:
1. **Environment:** add the SO-101 robot and the three cameras (`wrist`, `top`, `inspect`), then create an environment (`application/docs/04-environment-setup.md`).
2. **Record:** create a dataset and record episodes with the task (instruction) string (`05-recording-datasets.md`), or record with `dlsj.teleop_record` / `dlsj.kinesthetic_web` and **import** the LeRobot v3 dataset into Studio.
3. **Train remotely:** register the cloud RTX box as a **remote SSH trainer** (`07-remote-training.md`) and pick it when creating the model. Studio uploads a dataset snapshot and downloads the artifacts when done.
   - The trainer's SSH user needs **passwordless sudo** for Studio's automatic Docker+GPU setup (`/etc/sudoers.d/physicalai-trainer`: `<user> ALL=(ALL) NOPASSWD: ALL`, mode 0440). Otherwise, prepare Docker + NVIDIA support on it by hand.
   - AWS EC2 / AWS Batch trainers are also supported.
4. **Export:** when training finishes Studio exports to **PyTorch, OpenVINO, ONNX and ExecuTorch** automatically.
5. **Run:** on the Models screen use **Run model**, choose backend **OpenVINO** and device **GPU**. Studio's built-in inference supports only **GPU or CPU**, so NPU work (Anomalib, or the stretch action-expert split) runs through our own OpenVINO / `physicalai` runtime code.
6. **HF token:** π0.5 needs `HF_TOKEN` (read-only, with PaliGemma access) in `application/docker/.env`, then `docker compose up -d --force-recreate`.

**Manual fallback flow:** record on the XPS → push the dataset to the HF Hub → **train on cloud/team RTX** (CUDA, bf16) → download the checkpoint → `physicalai export --backend openvino` (INT8) → run on the XPS.
- **Studio on CUDA:** the Studio configs default to `accelerator: gpu`, which is what you want on RTX. The XPU flags in 4.2 are only needed when training on Intel. Studio also ships a trainer container (`application/docker/docker-compose.trainer.yaml`).
- **π0.5 GPU size:** ≥24 GB to fine-tune; 48–80 GB (RTX 6000 Ada / L40S / A100 / H100) is comfortable. Plain fine-tune first; SnapFlow 1-step distillation is optional (~12 GPU-h).
- **Pin versions:** use the **same Physical AI Studio commit** on the cloud box and the XPS (the XPS has commit `8f13241`), so exports load in the runtime.
- Sections 4.1–4.2 (training on XPU) are now the **fallback** if cloud GPUs are unavailable.

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
| 1 ✓ | Shrink the input | 3 cameras (`wrist`, `top`, `inspect`) at 224², fixed 64 language tokens, chunk 50. If latency is tight, test dropping `inspect` from the **policy** input (it's mainly for Anomalib). Don't add depth images | ~7–9 ms per view on a 4090; more on iGPU. 3 views ≈ 37 ms vs 27 ms for 2 on a 4090 |
| 2 | Distil to 1 step | `configs/physicalai/pi05/so101/snapflow.yaml`; SmolVLA: test 3–5 Euler steps | Expert cost ~10× less; LIBERO 98.75% vs 97.75% (10-step) |
| 3 ✓ | Static-shape OpenVINO IR (+RTC baked in); for ACT use `python -m dlsj.deploy.act_export --precision fp16` (normalisation in-graph, static) | π0.5: `convert_ov_rtc.py` in robotics-ai-suite `pi05-rtc-ov`, **set `-c 3`** (`--camera_num` defaults to 4 and is baked into the IR). ACT/SmolVLA: `physicalai export --backend openvino` | Base of 555 → 172 ms |
| 4 | INT8 weight-only | `--compress_int8` / `compress_weights_openvino_int8_sym`; validate trajectory MSE/cosine vs PyTorch | >30% faster, half size. **Skip for ACT** (launch-bound, keep FP16) |
| 5 | One batched vision call | Exported vision IR has batch = 3 (all cameras), not 3 calls | −67% vision weight traffic with 3 cameras |
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
- **`wrist`:** keep affine shifts and crops small; large crops break image-to-action alignment. Add `MotionBlur`/`GaussianNoise` here if available, since it's the noisiest stream.
- **`top` / `inspect`:** fixed views, so small `RandomAffine` shifts (a few px) help tolerate a slightly bumped mount; keep them small.
- Physical AI Studio's README doesn't document augmentation, so **train in LeRobot if you need it.**

### 5.3 Background replacement (1–3 h, big gain on background/texture shift)
- **GreenAug** ([arXiv 2407.07868](https://arxiv.org/html/2407.07868v2)): demos on a green screen with random textures keyed in. **91% average success vs 55% (no aug) and 70% (standard aug)**. Texture variety matters: no texture 48%, solid colours 65%, real textures 87%. Costs ~0.009 s per frame.
- **Depth as a green screen (untested idea):** if the D415 is `top` and depth is recorded, "background" is everything at or below the table plane, or outside the workspace. Replace those pixels in the **`top` image only** with random texture photos (the wrist webcam has no depth). Needs aligned depth recorded with the dataset.
- **Cheaper:** random overlay `img' = (1−α)·img + α·random_photo` with α ≈ 0.2–0.3 on part of each batch ([arXiv 2011.13389](https://arxiv.org/pdf/2011.13389) used α=0.5 in RL).

```python
import torch, random
class DepthBackgroundSwap:
    """`top` camera only (needs depth): replace table/background pixels with a random texture.
    Needs depth aligned to RGB. Set max_depth_m just above the table surface depth
    (e.g. table_depth - 0.005) so anything standing on the table is kept."""
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
- Cheaper: feed **`top` depth (or depth-masked `top` RGB) as an extra image** into ACT. This adds a fourth image stream (~one extra view of latency for a VLA), so try RGB-only first.

### 5.6 Colour sorting: keep the decision outside the policy
This is the **REASON** step of the loop (Section 8): defect result + colour + kickoff rules → instruction → π0.5.
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
**Run it on the NPU** (the brief requires using the NPU): export to OpenVINO with a **fixed input shape** and compile on `"NPU"` (`OpenVINOInferencer(path=..., device="NPU")`). PaDiM/PatchCore with a ResNet-18 backbone is a natural NPU fit (~0.9 ms for the backbone on this NPU). Train in `~/inference/.venv-anomalib` on the XPS: the model is tiny and needs on-site images.

**Our anomaly = a hand-drawn cross on the brick** (Challenge brief). What that means for data and setup:
- **Train only on plain bricks**: 50–200 `inspect` crops, covering **every brick colour and size** in the kit, several grasp orientations, and the lighting you'll demo under. Anything not in the plain set (a new colour, a new orientation) will also look anomalous, so cover the normal variation well.
- **Make the cross visible to `inspect`:** decide which face gets the cross, and record pick demos so that face ends up facing the camera. Simpler: **draw the cross on every face** of a marked brick (or at least the top and two long sides).
- **Cross style:** a thick, high-contrast marker (black on light bricks, white paint pen on dark bricks), at least a third of the face. Thin or tiny crosses are what PaDiM/PatchCore miss.
- **Test set:** 20+ crossed bricks of mixed colours, plus 20+ held-out plain bricks. Calibrate the threshold so all plain bricks pass with margin. False "defective" calls are the usual failure (Section 6.5).
- **Don't put crosses in the training set**, not even by accident. One crossed brick in `good/` teaches Anomalib that crosses are normal.

**Anomalib PaDiM or PatchCore**, trained on **50–200 "good" crops from the fixed `inspect` camera, of plain bricks held at the present pose**, exported to OpenVINO, gating which plate the brick goes to. Runs **separately from the policy**. The policy brings the brick to roughly the same spot in front of `inspect` every time, which is exactly the fixed viewpoint memory-bank methods need. Crop tightly around the brick and keep the gripper fingers' position consistent.

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
  good/   # 50–200 `inspect` crops of PLAIN bricks (no cross) at the present pose, all colours/orientations
  bad/    # crossed bricks: used only for threshold calibration and evaluation, never for training
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
- Applies only if the D415 is `top` and depth is read via `pyrealsense2` (optional; the main detector is Anomalib on `inspect`). `table_depth_m` is a per-pixel depth map of the empty table captured once at setup (median of ~30 frames). The object mask is simply `table_depth_m - depth_m > 5 mm`. Recapture it if the camera or table moves.

### 6.5 Robustness of the detector
- In the RAD benchmark, **memory-bank methods (PatchCore) were the most robust to lighting and viewpoint changes**; flow/distillation methods were hurt by light spots. **Viewpoint misalignment caused false positives for every method** ([arXiv 2406.07176](https://arxiv.org/pdf/2406.07176)).
- **Mitigations:**
  - Fixed `inspect` camera + a consistent present pose + a tight crop around the brick. Mask out or keep fixed the gripper fingers, which appear in every image.
  - Lock RealSense exposure and white balance.
  - Collect good samples under the real venue lighting (and different times of day).
  - Add brightness/colour-jitter copies of the good images.
  - Recalibrate the threshold on site.

### 6.6 Wiring into the robot loop
1. The policy picks the brick and **presents it to `inspect`** → **wait until the arm is still** (servo `Moving` = 0).
2. Grab **3 frames**, crop the ROI, take the **median score**.
3. **Threshold:** max score over held-out good images + a margin, or the F1-adaptive threshold if you have defects.
4. Route: which plate (REASON, Section 8). Detector inference takes milliseconds, so it doesn't slow the loop.
5. **Demo:** overlay `anomaly_map` as a heatmap on the `inspect` feed.

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

## 8. Putting it together: the closed loop on the XPS

```
 PERCEIVE (CPU)              ACT-1 (GPU)                DETECT (NPU)                 REASON (CPU)               ACT-2 (GPU)
 wrist + top + inspect ──►  π0.5 "pick up the     ──►  Anomalib on the `inspect` ──► kickoff rules + label  ──► π0.5: "place the brick
 joint state + gripper       brick and show it ..."     crop → {label, score,        (+ colour/type) →           on the <X> plate"
 current                     (hold still, Moving=0)     region}                      which plate → instruction        │
                                                                                                                       ▼
                                          grasp check (gripper current) after the pick ◄──────────────────────── SO-101
 OPTIMIZE: per-stage timers on every loop → live latency readout + Section 8.1 table
```

**Device split (the brief asks for CPU + GPU + NPU):**

| Device | Work | Why |
|---|---|---|
| **NPU** | Anomalib PaDiM/PatchCore on the `inspect` crop | Small, static-shape CNN; frees the GPU for the VLA; low power |
| **GPU** (Arc B390) | π0.5 for every movement (pick & show, place) | Compute-bound VLM needs XMX; INT8 weights |
| **CPU** | Camera capture, crop/pre-processing, rules → instruction, arm control, grasp check | Light, latency-tolerant logic |
| Stretch | π0.5 **action expert on the NPU**, KV cache shared zero-copy (Intel's split, Section 4.5 step 9) | Overlaps VLM and expert under RTC |

**Task state machine:**
1. `PICK`: run **π0.5** with `"pick up the brick and show it to the inspection camera"` until the brick is held at the `inspect` spot. → `VERIFY_GRASP` (gripper current; on EMPTY, retry up to 2).
2. `DETECT`: wait for `Moving` = 0, grab 3 `inspect` frames, crop the brick, run Anomalib on the **NPU** (median score). Output a **structured signal** `{label: perfect|defective, score, region}`:
   - `region` = the anomaly map thresholded → connected components → bounding box on the brick: the **localization signal** the brief requires;
   - also log it and draw it as a heatmap overlay.
3. `REASON`: apply the **kickoff rules** to (label, score, region, brick colour/type via HSV on the `inspect` crop, anything else the rules use) → **which plate** (our task: **plain → blue, cross → pink**).
   - **Uncertainty band (reliability criterion, 10 pts):** if the score is within a margin of the threshold, or the 3 frames disagree, don't guess. Re-inspect: wait, re-grab frames, or run the optional `"turn the brick and show it ... again"` skill. If it's still uncertain after 2 tries, treat it as **defective** (the safe default) and log it.
   - Then pick the instruction → one of a fixed set of instruction strings, e.g. `"place the brick on the reject plate"`. Keep the strings **identical** to the training demos. A rule table is enough; an LLM is optional narration only.
4. `PLACE`: run **π0.5** with that instruction until the brick is released on the plate.
5. Loop. Log per-stage timings every iteration.

If π0.5 isn't ready: **SmolVLA** with the same three instructions (Intel's reference VLA). No ACT.

### 8.1 OPTIMIZE: the numbers to show

Measure on the XPS (median and p99 over ≥100 runs, after warmup, on mains power):

| Stage | Baseline | Optimised | Gain |
|---|---|---|---|
| Preprocessing | Python/NumPy resize + normalise on CPU per camera | Capture at the policy's resolution; fold normalisation into the OpenVINO graph (`ov.preprocess.PrePostProcessor`); batch cameras | |
| Anomalib | PyTorch-XPU eager | OpenVINO on **NPU** (and GPU FP16 for comparison) | |
| Postprocessing | Full-res anomaly map → NumPy threshold | Threshold/box on the downsampled map; only upsample the box | |
| π0.5 chunk (pick and place) | LeRobot PyTorch-XPU (`~/inference/.venv`, policy server) | OpenVINO FP16 → **INT8** → (stretch) 1-step distilled / expert on NPU | |
| End-to-end loop | sum of baselines | sum of optimised + async overlap | |
| Throughput | tasks/min, baseline | tasks/min, optimised | |

**Measured on our XPS** (X9 388H, Arc B390; `bench/pi05_torch_xpu.py`, `lerobot/pi05_base`, LeRobot 0.7.0, PyTorch 2.15 XPU, bf16, 3 cameras at 224², chunk 50, median of 20 after warmup, 13 Nov):

| π0.5 config | Median / chunk | Notes |
|---|---|---|
| PyTorch-XPU, 10 denoising steps | **467 ms** | Baseline row. Peak GPU memory 8.9 GiB |
| PyTorch-XPU, 5 steps | 382 ms | No distillation, so accuracy is not checked |
| PyTorch-XPU, 1 step | 313 ms | Same caveat |

⇒ ~17 ms per denoising step, and **~296 ms is the vision encoder + VLM prefix**. Step reduction alone caps out at ~1.5×. The VLM prefix is where OpenVINO INT8, fused SDPA and batched cameras must win.

Reference points:
- Intel's π0.5 on a Core Ultra X7 358H: 555 → 172 ms per chunk (3 cameras, 10 steps).
- Our ResNet-18 smoke test on this XPS: GPU FP16 0.59 ms, NPU 0.88 ms, PyTorch-XPU 2.17 ms, CPU 12.6 ms.

### 8.2 Extension (phase 2): sort numbered bricks in order

**Goal:** once pick → inspect → place is reliable, the bricks carry numbers (1, 2, 3, …) and the robot handles them **in numeric order**. It still rejects defective ones. ⚠️ The exact target needs confirming: placed in a row of numbered slots, stacked, or each dropped onto a plate in order. The design below works for all three.

**Who does what:** Anomalib cannot read numbers. It only learns "normal" vs "not normal". So phase 2 adds one perception module and a planner:

| Job | Module | Device | Notes |
|---|---|---|---|
| Find bricks and **read their numbers** (whole table) | **Number reader on the `top` image**: brick detection (depth mask or HSV blobs) → crop each brick → digit classifier | **NPU** | A tiny CNN trained on brick-digit crops (fast, static shapes), or OpenVINO's PaddleOCR. Outputs `[{number, confidence, box, table_xy}]` |
| **Plan the order** | Planner (CPU): sort detections by number → a task queue; re-plan after every brick | CPU | Trivial and deterministic. This is the "planning" in the brief's "perception, reasoning, planning and manipulation" |
| Pick the **right** brick | **π0.5 with a grounded instruction** (options below) | GPU | The hard part |
| **Verify** before placing | Read the number again on the `inspect` crop + Anomalib defect check | NPU | Catches picking the wrong brick: if the number ≠ the expected one, put it back and re-plan (reliability points) |
| Place in order | π0.5: `"place the brick in slot <k>"` / on the plate | GPU | Slot k is chosen by the planner |

**How π0.5 picks a specific brick (choose one; the order is our recommendation):**
1. **Visual marker (recommended):** the planner draws a marker (e.g. a coloured circle) on the target brick **in the `top` image fed to the policy**. π0.5 is trained to "pick up the marked brick and show it to the inspection camera". It doesn't need to read digits, and it works for any number of bricks.
   - Demos: record picks with the marker drawn on the brick being picked (overlay applied at recording time from the brick detector, or added afterwards).
2. **Position words:** the planner turns the brick's location into words: `"pick up the brick on the far left"` / `"… second from the left"`. This works with a small fixed layout. It needs demos for each position word.
3. **Pure language grounding:** `"pick up brick number 3"`, relying on π0.5's PaliGemma backbone to read the digit. This is the most "VLA reasoning", but also the most data-hungry and least reliable. Selective picking among look-alike objects is where VLAs fail most: 0–10% on the SO-101 colour-sorting benchmark (Section 2). Treat it as a stretch to show off, with 1 or 2 as the fallback.

**Optional "reasoning" layer:** a small VLM on the GPU or NPU (e.g. Qwen2.5-VL-3B via OpenVINO GenAI) can turn a spoken or typed goal ("sort the bricks in ascending order, rejects to the pink plate") plus the detections into the task queue, and narrate decisions in the demo. Keep the deterministic planner as the source of truth and use the VLM only to produce or check the plan.

**Phase-2 loop:**
```
 top image ──► number reader (NPU) ──► planner: queue sorted by number ──► next target k
      ▲                                                                        │
      │                                    π0.5 "pick up the marked brick and show it ..." (GPU)
      │                                                                        ▼
      │                         inspect: number check + Anomalib (NPU) ──► wrong number → put back, re-plan
      │                                                                        │ defective → pink plate
      └──────────── re-plan ◄── π0.5 "place the brick in slot k" (GPU) ◄───────┘ perfect
```

**Building phase 1 so phase 2 slots in (do this now):**
- **Skill library, not one script:** keep π0.5 skills as named instruction strings in one table (`PICK_AND_SHOW`, `PLACE_BLUE`, `PLACE_PINK`, later `PICK_MARKED`, `PLACE_SLOT_k`). The state machine only calls skills by name.
- **Perception modules return structured signals** with the same shape: `{label, score, region}` from Anomalib, `[{number, confidence, box, table_xy}]` from the number reader. Log them all.
- **A planner/task queue from day one**, even if phase 1's queue is "whatever brick is there". Phase 2 then just changes how the queue is ordered.
- **Record phase-1 demos with bricks in varied positions** (and some with several bricks on the table), so the pick skill already copes with clutter.
- **Collect number crops early:** while recording phase 1, save `top` and `inspect` crops of numbered bricks, so the digit classifier can be trained with no extra session.
- **OPTIMIZE:** the number reader is a second NPU workload next to Anomalib. Report both in the Section 8.1 table (and try batching all brick crops into one NPU call).

---

## 9. Evaluation checklist
- [ ] Kickoff rules written into the Challenge brief; instruction strings fixed and used identically in demos and the REASON step.
- [ ] XPS: OpenVINO lists CPU, GPU, NPU; `HF_TOKEN` set (PaliGemma access); Studio UI reachable on :7860.
- [ ] π0.5 fine-tune on the Modal GPU uses the same Studio/LeRobot version as the XPS; first 200 steps timed; all three instructions covered in the data.
- [ ] `wrist`, `top`, `inspect` all streaming at 640×480@30 with **auto-exposure/WB locked**; `top` and `inspect` mounts fixed and marked; `inspect` has steady light and a plain background.
- [ ] ≥150 demos across ≥6 conditions; one lighting + one background **held out**.
- [ ] Idle frames trimmed; slow/retry episodes dropped.
- [ ] Augmentation on, **hue ≈ 0** if colour matters.
- [ ] π0.5 checkpoint exported to OpenVINO INT8 (`-c 3`); latency benchmarked; trajectories checked against PyTorch.
- [ ] Async execution working, with no jerks at chunk boundaries.
- [ ] OpenVINO IR is static-shape; `benchmark_app -pc` shows SDPA/GatedMLP fusions; median + p99 latency logged.
- [ ] Go/no-go on π0.5 vs SmolVLA within the first hour of training.
- [ ] Uncertain detections trigger re-inspection, and fall back to the defective plate after 2 tries.
- [ ] Anomalib trained only on plain bricks (all colours/orientations); threshold calibrated on 20+ crossed vs 20+ held-out plain bricks; crosses visible to `inspect` from the presented face.
- [ ] Anomaly threshold calibrated **on site**; false-positive rate checked on 20+ good samples; model **runs on the NPU**.
- [ ] Defect output is a structured signal `{label, score, box, table_xyz}` and is logged per object.
- [ ] Pitch ready: **concise architecture + optimization decisions** explanation (loop diagram, device split and why, OPTIMIZE table), per expectation 6.
- [ ] OPTIMIZE table (Section 8.1) filled with measured baseline vs optimised numbers; per-stage latency shown live in the demo.
- [ ] Grasp check triggers retries correctly.
- [ ] **10+ trials per condition** for any claim (WARP authors: 1/20 differences are noise).
- [ ] Demo video: heatmap overlay, live lighting change, throughput (tasks/min).

## 10. Risks & fallbacks

| Risk | Fallback |
|---|---|
| Modal GPU unavailable or slow | Train SmolVLA on the XPS GPU (Section 4.1–4.2); 64 GB shared RAM may even fit π0.5, but slowly |
| π0.5 fails on gated tokenizer (401 on `google/paligemma-3b-pt-224`) | Set `HF_TOKEN` from an account that accepted PaliGemma's terms |
| Anomalib export fails on NPU ("Upper bounds are not specified") | Export with a static input shape; else run it on GPU and use the NPU for the π0.5 action expert instead |
| Studio export not loadable by the runtime | Use the same Studio commit for training and deploy; fall back to the LeRobot policy server on PyTorch-XPU (baseline env) |
| `bf16-mixed` / XPU strategy errors in Studio | `precision: 32`; or train with LeRobot `--policy.device=xpu` |
| LeRobot version lacks new augmentations | Use `ColorJitter` + `SharpnessJitter` only, or pass torchvision v2 transforms |
| OpenVINO export fails | PyTorch on XPU with `torch.compile(mode="max-autotune-no-cudagraphs")` + manual `torch.xpu.graph` (not `reduce-overhead`, a no-op on XPU); or ONNX backend in `InferenceModel` |
| π0.5 doesn't fit / trains too slowly | SmolVLA with the same three instructions (`train_expert_only` / frozen vision encoder) |
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
