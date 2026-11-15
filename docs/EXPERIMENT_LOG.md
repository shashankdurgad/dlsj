# Experiment log: research, experiments and decisions (13–14 Nov 2026)

One place for everything we researched and measured, with numbers, the decision each result led to, and links to the
scripts and detailed write-ups. Open items are marked **(open)** and get filled in when they finish.

Machines: **XPS** = Dell XPS 16 (Core Ultra X9 388H, Arc B390 iGPU, NPU, 64 GB, Ubuntu 26.04), the deploy box.
**Spark** = Joe's NVIDIA GB10 (121 GB unified memory), used for Isaac Sim and exports that need more RAM than the XPS.

Contents
1. Research reports
2. Inspection (spin and inspect)
3. Policies: π0.5, SmolVLA, MolmoAct2 (models, sim and robot tests)
4. Inference optimisation (π0.5 and MolmoAct2): profiles, precision, RTC, graph/kernel level, baseline benchmark
5. Datasets: 3-phase relabel, optimisation-rule audit
6. Simulation realism and the residual-RL environment
7. Infrastructure findings (XPS stability, Wi-Fi, calibration)
8. Decisions log
9. Open items

---

## 1. Research reports
| Report | Question | Key conclusion |
|---|---|---|
| `reports/XPU kernels and Chinese VLAs.md` | Intel GPU kernels; alternative VLAs | baseline knowledge for the XPS work |
| `reports/XPS closed loop optimisation.md` | How to optimise the whole loop on the XPS | prefix (vision+language) dominates; OpenVINO + async/RTC; Anomalib fits the NPU |
| `reports/Pi05 XPS optimisation round two.md` | Further π0.5 options, ranked | 1 async/RTC, 2 quiet machine + GPU min freq, 3 longer fine-tune, 4 drop inspect view, 5 SnapFlow |
| `reports/Pi05 post training practice.md` | How people post-train π0.5 | full fine-tune 20–30k steps beats action-expert-only; fix gripper/data/deploy issues before training longer; RTC is the cheapest real gain; RECAP > online RL at hackathon scale |

Other research done in conversation (not separate reports):
- **Long-WAM (NVIDIA)**: long visual history + future-video prediction; strong for dynamic/long-horizon tasks, NVIDIA-only kernels (NVFP4, CUDA graphs) → not a fit for our static, controller-sequenced task on Intel hardware.
- **MolmoAct2 (AI2)**: 5B (Molmo2-ER LLM 36 layers/2560, SigLIP ViT 27 layers at 378 px, flow action expert 36 layers/768, 10 steps, chunk 30); `allenai/MolmoAct2-SO100_101` is already trained on SO-100/101 data in absolute joint pose; all official checkpoints are FP32; Studio supports training (LoRA) and OpenVINO export but had no RTC.
- **Prompt design**: canonical per-phase prompts; no colour/location (not labelled in our data); only the inspection decision (bowl) is a parameter; training and run-time strings must be identical.

## 2. Inspection (spin and inspect)
- **Classifier**: YOLO26n brick classifier. v1 (`brick-yolo-classifier-v1`): 90.7% test accuracy. v2 = the detected-brick classifier from `brick-auto-ensemble-v1`: 97.4% (113/116). Both 3/3 on the three saved real sweeps (cross, plain, circle); v2 is confident on every plain frame (0.92–1.00) where v1 was unsure on 2 of 8. **v2 is the default.**
- **Brick detector** (one-class YOLO26n from the ensemble release) run with plain OpenVINO (letterbox + NMS, no ultralytics): NPU 4 ms, GPU 3 ms, CPU 19 ms per frame; boxes identical across devices. Finds the held brick in 4/14 real frames (fingers occlude), falls back to the fixed ROI otherwise.
- **Decision rule** (`yolo_topk`, marks are on all sides of a marked brick): marked needs >= 3 confident mark frames; plain needs >= 50% confident plain frames (one glare outlier allowed); otherwise re-inspect → purple. **Early stop**: 3 agreeing mark frames or 6 plain frames. Sim (Isaac, 12 angles, marks on all sides): plain/cross/circle all correct; cross/circle decided in 3 stops, plain in 8.
- **Sweep page** (`dlsj.inspect.sweep_live`, port 8012): leader teleop, hand posing (torque off / hold), save inspect pose, gripper open / close-on-brick, early stop; YOLO + detector on the NPU.
- **Anomalib**: our pipeline supports PaDiM and PatchCore; PaDiM/PatchCore were trained on synthetic data on the XPS (pipeline tests). The real model is Tan's **PatchCore** (`brick-patchcore-geometry-v1`, WideResNet50-2, trained in Anomalib Studio): 15/16 plain accepted, 54/58 crosses and 50/55 circles flagged. Ensemble (detector + PatchCore + YOLO): 88 correct / 1 wrong / 27 uncertain of 116. **Decision: YOLO only in the live loop; PatchCore kept off (available as an option).**
- Docs: `docs/sweep-live.md`, `docs/yolo-brick-classification.md`, `docs/brick-object-detection.md`.
- **Automatic inspection in the VLA controller** (`deploy/smolvla/auto_control.py` + `auto_inspect.py`, 14 Nov): a
  second controller next to the manual `smolvla_control.py` (unchanged: M + 1/2/3). Space = pick, M = bring, M at the
  camera = policy paused, wrist spun by the same `Sweeper` (calibrated roll limits, grasp check, wrist returned),
  YOLO v2 + detector on the NPU with early stop, then it places in the decided bowl by itself; uncertain → one
  re-spin → purple; Q aborts and holds; dropped brick holds; 1/2/3 still override. On the XPS: both models on the
  NPU, 1.5 s load, ~6 ms per frame. 10 unit tests (`tests/test_auto_control.py`, `tests/test_auto_inspect.py`).
  The detector needs ultralytics, installed as an overlay (`~/inference/pydeps/ultra`, `--no-deps`) so Desmond's
  `pi05rt` venv is unchanged. Arm test pending (supervised).

- **Top-camera brick finding + end-of-loop YOLO verification (15 Nov).** The loop's top survey ran the released
  detector (`brick-detector-v1` = GitHub release `brick-auto-ensemble-v1` `detector_openvino_model`, hashes match) on
  the full 640 px frame at its default conf 0.5. It was trained on inspection-camera close-ups; the ~40 px top-view
  bricks scored **0.13–0.43** (all missed) while the yellow plate scored 0.61. On the same frame at 960 px they scored
  0.53–0.65. Fix (`perception.detect_tiled`): 2×2 overlapping tiles of the 560×420 work area, each letterboxed to 640
  (≈2× zoom), cross-tile NMS; boxes cut by a tile border only fill in where no whole box exists. Its own detector
  instance (`--top-detector`, OpenVINO on the NPU when the dlsj checkout has `make_brick_detector`, else ultralytics on
  CPU) at conf 0.25; survey counts bricks at ≥ 0.3. **25 live top frames from a running loop (15 Nov):** old pipeline
  0–2 bricks per frame, tiled 1–5, every brick on the table found whenever the arm is home (0.42–0.75), no false
  boxes at home; one false box on the arm while it is out over the table (not counted: survey/verify only count with
  the arm home). New **verify** step: when the survey sees an empty table from home, the loop does not stop; it takes 8
  top frames, and a brick counts only if it is seen at the same spot (±20 px) in ≥ half of them. Clear → done
  ("Verified clear"). Bricks → survey again; after 2 such re-surveys → pause with a message (never a false "done").
  Dashboard: "Verify" chip and verify status. 4 new loop tests + 3 perception tests; 24 tests pass. **(open)**: an
  empty-table test on the real camera (false positives there would block "done"); the gripper fingertips scored
  0.23–0.35 on older-setup frames when just outside the home ignore box.

- **Held-brick detector v1 (release `held-brick-detector-v1-20261115`) as stage 1 of held-crop** (`--inspector
  held_crop`, confidence 0.2): identical rig result to the v4-box version, **31 / 2 / 5** of 38 (3 circles passed as
  plain by the v2 classifier); the backbone alone gives exactly v4 epoch 4's answers (33 / 4 / 1). The detector is
  fine; the v2 classifier is the limit on circles. v4 alone stays live. `bench/held_brick_v1/`.
- **Two-stage held-crop tried (15 Nov)**: held-cube v4 draws the box, the YOLO v2 classifier classifies only that crop
  (`--inspector held_crop`). Real decision rule, 38 rig spins: v2 on a centre/detector crop 27 right / 3 uncertain /
  8 wrong (7 circles passed as plain); held-crop 31 / 2 / 5 (3 circles as plain); **v4 alone 33 / 4 / 1** (15/15
  circles; the 4 uncertain are plain bricks below the v2-tuned plain threshold, the 1 wrong is the disputed red brick).
  Cropping fixes v2's background confusion but v2 itself misses circles, so v4 alone stays live. `bench/held_crop/`.
- **Held-cube YOLO26n v4 epoch 4 (release `held-cube-yolo26n-v4-20261115`, training still running) replaced v2 as the
  live spin classifier**: rig eval **37 right / 0 uncertain / 1 wrong** of 38 (14/14 cross, 15/15 circle), 4.1 ms/frame
  on the NPU. The one miss is again red brick `232c447e14` (labelled plain, called cross on every frame by both v2 and
  v4, which was trained on corrected red-cross labels): likely our label. `bench/held_cube_v4/`.
- Earlier the same day: **held-cube YOLO26n v2 (release `held-cube-yolo26n-v2-20261115`) as the live spin classifier**
  (`auto_inspect.HeldCubeInspector`, `--inspector yolo_v2` falls back). One detector finds the cube held in the
  gripper and its visible symbol on the full inspection frame (table cubes ignored by design). On our 38 labelled rig
  spins (12 saved frames each): **36 right / 1 uncertain / 1 wrong** vs the v2 classifier's live decisions **29 / 5 / 4**
  (v2 called 4 circles plain; the new model got 15/15 circles); 429/455 confident frames correct; **4.1 ms per frame on
  the NPU**. The one wrong spin (a red brick labelled plain, called cross on 12/12 frames) is unresolved: label or a
  red-on-red weakness. Caveats: unknown whether these spins were in its training data; its own validation had no cross
  images. `bench/held_cube_v2/`.

## 3. Policies
### π0.5
- Fine-tune `w3joe/pi05-junction-so101-20261114` (action-expert-only, ~2300 updates), later `ft9060`.
- Isaac Sim (Joe's scene, 3 episodes): π0.5 base 0/3 lifted, closest 11–13 cm; fine-tune 0/3, closest 6–11 cm, never closes. 0.38 s/chunk (PyTorch bf16, GB10).
- Exports (XPS GPU): `pi05_ft9060` FP16 244.5 ms, INT8 178.3 ms; old `junction` INT8 64-token 5-step 135.4 ms.
### SmolVLA (`w3joe/smolvla-junction-so101-3stage-20261114`)
- 3-stage prompts, wrist + top, 20k steps, val loss 0.2588.
- Isaac Sim, 5 episodes: 0/5 lifted, closest **4.0–5.8 cm**, opens and closes (closes ~5 cm short). 0.16 s/chunk (PyTorch, GB10).
- XPS: OpenVINO RTC INT8 **74 ms/chunk**; switch jump 0.4–2.7 deg. Robot controller `deploy/smolvla/smolvla_control.py` (joint offset `recal_2026-11-14`, M advances stages).
### MolmoAct2-SO100_101 (zero-shot)
- Isaac Sim: generic prompt closest 7.0 cm; 12 attempts with green-brick prompts closest **3.9 cm** (best: "pick up the green brick and lift it up"); 10 long (40 s) episodes with "pick up a green brick and lift it up": closest 4.5 cm; never closes on the brick. 0.42 s/chunk (PyTorch bf16, GB10).
- On the real arm (zero-shot, INT8): drives to the bricks and hovers with the gripper open just above a brick; attempts to close but does not commit. ~340 ms/chunk.
- **Decision: MolmoAct2 SFT (LoRA) is the main track**; residual RL considered after SFT quality is known.

## 4. Inference optimisation
### π0.5 (XPS)
- Kernel profile: 3,940 kernels; INT8 matmuls 69%, SDPA 19%; launch overhead negligible.
- Runtime settings sweep: all equal or slower than defaults (SDPA off +39 ms, dynamic quantisation off +31 ms, throttle LOW +13 ms).
- GatedMLP fusion (custom OpenVINO build + GeLU patch): fires but runs on a reference kernel, **+96 ms** → abandoned. `deploy/pi05/kernel/README.md`.
- **RTC**: `studio_export.py --rtc` (+ memory-mapped checkpoint load, export peak 53 → 42 GB); controller wrapper `deploy/pi05/rtc_control.py`. Offline on 17 recorded hand-offs: switch jump **16.1 → 1.3–3.7 deg** (guidance 5 / 2), gripper 10 → 0.6–1.5; latency ~150 ms (INT8, tok64, 5 steps).
### MolmoAct2 (XPS)
- **Profile (INT8, 286 ms)**: action expert 110 ms (38%; 28,496 kernels = 10 steps x 36 layers), LLM prefill 106 ms (37%), vision 53 ms (18%), other 20 ms. FP16: expert 183, prefill 220, vision 69 ms. Prefix = 650 tokens: 392 image + **158 padding**.
- **INT8 vs FP16** (60 cal2 frames): INT8 0.24 deg joints / 0.11 gripper from FP16, same error vs demos (8.48 vs 8.47 deg); outputs deterministic per input.
- **No-retrain variants** (vs FP16/10-step ref): asym INT8, expert-in-FP16 and DQ group 64 pass with no gain; removing setup/control text FAILS (0.60 deg, behaviour changes); text 256 → 128 (1.07 deg) and 5 steps (1.50 deg) are not exact zero-shot — but a different noise draw alone moves outputs 2.0–2.6 deg, so they must be judged on the fine-tuned model. **Decision: train at tokenizer_max_length 128** (`training/molmoact2/policy_text.py`; verified the saved config carries 128).
- **RTC for MolmoAct2** (`deploy/molmoact2/rtc_patch.py`): patched flow loop with in-graph prev-chunk transform; PyTorch equivalence at guidance ~0 exact; on 24 real hand-offs jump **2.62 → 1.23 deg** (p90 5.79 → 3.68), gripper 2.10 → 0.87; INT8 RTC 312 ms.
- **Graph level**: Studio already hoists the per-layer KV context, masks and RoPE out of the flow loop; the remaining step-invariant time conditioning (391 MatMuls, ~1.9 GB weight reads if executed) is constant-folded by the OpenVINO GPU plugin (0 ms). Nothing to hoist. `deploy/molmoact2/graph_analysis/`.
- **Kernel level**: all 720 expert attention calls use the fused `ocl::sdpa::opt` FP16 kernel, 2–4x faster than a decomposed rewrite; layout changes do nothing. Defaults are optimal.
- **Baseline vs optimised** (`bench/molmoact2_baseline.md`, same 60 frames):

| row | median ms/chunk | p99 | speed-up | peak RAM |
|---|---|---|---|---|
| 0 baseline: stock PyTorch CPU bf16 | 10,770 | 11,679 | 1.0x | 26.8 GB |
| OpenVINO GPU FP16 | 362 | 391 | 29.8x | 20.0 GB |
| OpenVINO GPU INT8 | 285 | 291 | 37.8x | 5.9 GB |
| OpenVINO GPU INT8 + RTC | 312 | 318 | 34.6x | 5.9 GB |

  PyTorch on the Arc GPU (XPU) crashed the device (`UR_RESULT_ERROR_DEVICE_LOST`) and froze the XPS → CPU is the PyTorch baseline. Thermals: run straight after the CPU row, INT8 measured 538 ms (throttled); 285 ms after a 10-min cool-down. With RTC every chunk lands before the 30 Hz queue runs dry.
- **After the LoRA (14 Nov, `bench/molmoact2_after.md`)**: deployed fine-tuned 8k, OpenVINO GPU INT8 + RTC: **313 ms
  median (p99 317) vs 10,770 ms baseline = 34.4x**, 5.9 vs 26.8 GB, error vs demos **2.80 vs 8.34 deg**; 5.5k: 314 ms,
  3.52 deg. Steps 5 / text 128 still not tried on the fine-tuned model (`optimize.py` variants; the 8k model was not
  trained at 128 tokens). Loop: inspection ~6 ms/frame on the NPU, continuous spin 61 frames in 8.2 s (vs 12 in
  11.2 s), cycle time median 35.3 s per brick over 20 bricks.
- **Power (`bench/molmoact2_power.md`)**: energy per action chunk 436.9 J (stock PyTorch CPU, 31.8 W x 13.7 s) -> **12.1 J** (deployed OV GPU INT8 + RTC, 38.7 W x 0.31 s) = **36x less** (package, RAPL via a root read-only logger); platform 580 -> 16 J; idle 2.9 W package / 11.5 W platform.
- Recipe note: AI2's LoRA (connector + action expert fully trained) vs Studio's default (LoRA everywhere) have identical inference cost; A's fully trained expert may be more INT8-sensitive (fallback: expert in FP16, 380 ms).

## 5. Datasets
- **3-phase relabel** (`dlsj/data_tools/relabel_3phase.py`): grasp_frame from the gripper (fires ~0.8 s before the operator's "pickup done" mark, i.e. at the close; IQR -32..-21 frames vs 123 operator marks), inspection mark, plate corrections. duplo_teleop 516/517 episodes, cal2 224/225.
- **Optimisation-rule audit of the prepared MolmoAct2 sets** (on the Spark): `molmoact2-517top250-3stage` (old calibration, colourless prompts) and `molmoact2-cal2-431-3stage` (current calibration, colour pick prompts): both wrist+top, top 560x480, 30 fps, 6-d state/action, prompts well within 128 tokens; cal2 calibration identical to the robot's (`8a3e5e04…`). Issues: the pick prompt differs between sets (colourless vs coloured); the old-calibration set needs `--joint-offset recal_2026-11-14` at deploy; mixing sets needs a calibration conversion.
- Calibration: follower recalibrated 2026-11-14 01:00 (wrist roll and gripper horns reseated); offsets old → new in `configs/joint_offsets.json` (pan +20.35, lift -7.96 deg).

## 6. Simulation realism and the residual-RL environment
- **A1 servo/gripper system ID** (741 episodes, `sim/realism/servo_sysid.md`): first-order model with a 1-tick lag, K 0.20–0.30 per tick (tau ~0.1 s; gripper 0.15 s), held-out RMSE 2–4x better than naive; gravity bias (elbow +1.9 deg, lift up to 6 deg); gripper holds at 5.75 (command ~2.5 lower); controllers hit the LeRobot clamp often.
- **A2 scene layout** (`sim/realism/scene_layout.md`): grasp positions, plate layout (re-randomise per episode), inspection pose, phase durations; top-camera check median 38–50 px; a data-only camera refinement was rejected (did not generalise).
- **Isaac Lab env** (`sim/residual_env/`): parallel cloned arms, 120 Hz physics / 30 Hz control, fitted servo model, clamp, noise, box finger pads, wrist+top tiled cameras, residual action on gripper/wrist, phase rewards, SAC. Fixes: Fabric scene delegate off (black cameras on GB10), physics replication off, box finger pads, gripper offset.
- **Joint-offset fit** (`sim/realism/joint_offset_fit.md`): replay grasp 15% → 30%; recovers the 2026-11-14 recalibration from images alone.
- **Merge of Joe's cal2 calibration** (`joe_cal2`, default; `sim/realism/joe_cal2_merge.md`): replay of 20 cal2 demos **80% grasp/lift** (vs 20% with our fit), 75% brick in the inspect view; remaining gaps: sim gripper holds at 4.3 vs 5.8 real; Joe's own sim lifts more of his episodes.
- RL was put out of scope for deployment; an **RL demo** (frozen base + SAC residual, video) was built instead
  (`sim/residual_env/demo_*.py`, README "Residual-RL demo"). Task: grasp, lift >= 5 cm, hold 1 s; residual on the
  gripper (+-25) and joints (+-4..8 deg).
  - **Camera bug found and fixed (14 Nov):** the old GB10 workaround (Fabric scene delegate off) made every sim camera
    show the initial scene, because GPU PhysX never writes poses to USD. Wrist, top and overview images were frozen
    (any earlier camera-based sim run was fed frozen images). Under Isaac Lab 3.0 the delegate is back on; all three
    cameras checked to follow the arm.
  - **Scripted fallback (Spark, 64 envs, 75 min, ~1.1M transitions, SAC 16 updates/step):** base = a scripted
    VLA-failure stand-in (IK reach with a 2.5-5 cm biased grasp error and mistimed close; NOT a learned VLA). Held-out
    evaluation on 256 unseen episodes: **base alone 7.4% -> base + residual 14.8%** (19 -> 38 successes; mean max
    lift 0.9 -> 2.0 cm). The trainer's 64-episode evals were noisy (6-20%); a first run with 4 updates/step did not learn.
    Per-episode effect in one 64-episode batch: 6 fixed, 4 broken. Video: before/after, first 2 episodes unselected +
    2 labelled SELECTED (fixed) episodes, held-out numbers on the cards. Modest, honestly reported gain.
  - **MolmoAct2 as the base (bf16, GB10, zero-shot weights, rendered cameras, shared GPU):** load 27 s, 16.9 GB;
    env throughput N=4 / 8 / 16: **18.0 / 23.0 / 26.4 env ticks/s** (~260 / 280 / 320 episodes/h); worker latency
    1.38 / 2.47 / 4.76 s per batch (~0.3 s per env): the policy, not the sim, is the bottleneck. Zero-shot MolmoAct2
    in the sim: 0% success, ~0 cm lift (as expected before the LoRA).
  - **Modal (RTX PRO 6000):** `sim/residual_env/modal_demo.py`. Rendering works in a Modal container (Vulkan and the
    NVIDIA graphics libraries are present): smoke verdict OK, 8 envs: launch 111 s, physics 247 env ticks/s, wrist +
    top + overview render 0.13 s. Isaac Sim 6.2.0 + Isaac Lab 3.0.0 pinned to the Spark's working set
    (`modal_isaac_overrides.txt`; isaacsim and isaaclab conflict on urllib3, resolved by overrides). Base MolmoAct2
    @152569fe is in our `rl-demo-hf` volume; the robot USDs are in `rl-demo-assets`. MolmoAct2 throughput on Modal was
    not measured (the run was stopped when the plan changed).
  - **Real-policy RL run: NOT done.** The fine-tuned checkpoint (step 5500, LoRA r64; slim version 3.1 GB) exists only
    on the XPS, and it could not be moved in time: XPS uplink ~2.5 MB/s, no XPS→Spark route, XPS↔Mac 140 KB/s. The
    user called it off (15 Nov). Everything for it is ready: `molmo_policy.load_ckpt` loads full or slim checkpoints
    (strict=False, checks that no trained tensor is missing), and `modal_demo.py::train` / `run_real_demo.sh` run
    worker + trainer + video in one command. The only demo video is the scripted fallback.

## 7. Infrastructure findings
- **XPS freezes**: three times under heavy GPU+RAM load (PyTorch XPU MolmoAct2, 20 GB FP16 loads, concurrent jobs). Rules: one model on the GPU, memory watchdog, nothing heavy while the arm runs; deploy INT8 (5.9 GB vs 20 GB).
- **XPS Wi-Fi**: 10 reconnects in 6 h (AP roaming every ~5 min) + power save → power save off and BSSID pinned; venue Wi-Fi still congested (5–750 ms). After reboots, check Wi-Fi and `tailscale status` (the pin can block reconnection).
- **Thermals**: CPU and iGPU share the package; a hot package made INT8 ~2x slower → cool down and keep the machine quiet before demos.
- **Transfers**: no direct XPS ↔ Spark route; relays via a laptop run at 0.4–0.7 MB/s → pull large artefacts from HF/cloud directly.
- **Exports**: 5B MolmoAct2 export peaks ~44 GB (Spark) — on the XPS only with nothing else running; π0.5 fine-tune export 53 → 42 GB with a memory-mapped checkpoint.

## 8. Decisions log
- Policy cameras: wrist + top only (inspect camera dropped as noisy); top cropped 560x480 in training and deployment.
- Inspection: YOLO v2 + detector on the NPU; PatchCore off.
- **Run-time loop = `loop_control.py`** (closed loop + perception dashboard, 15 Nov; wraps `auto_control.py`). Earlier: run-time loop = `auto_control.py` (automatic spin-and-inspect + place); `smolvla_control.py` = manual VLA testing only (also in CLAUDE.md).
- Main policy track: MolmoAct2 LoRA SFT; prompts: 3-phase canonical, bowl as the only parameter; text length 128 trained in.
- Deployment: OpenVINO on the Arc GPU, INT8 preferred on data (FP16 if accuracy concerns after fine-tune), RTC on.
- RL: out of scope for deployment; demo video only, real policy as base, scripted fallback (done). Real-policy run cancelled (checkpoint transfer too slow).
- **Physical AI Studio is the model registry (15 Nov)**: the cal2 3-stage training set (456 episodes, exact training prompts) and the MolmoAct2 8k INT8+RTC export are in the Studio project (dataset `db961a58…`, model `22fb0973…`, Studio copy sha256-identical to the deployed files); `run_loop.sh` resolves the policy by Studio model ID (`dlsj.deploy.studio_registry`), the dashboard shows it. `deploy/studio/README.md`.

## 9. Open items
- Supervised arm test of `auto_control.py` (spin, decision, automatic place).
- Empty-table check of the loop's YOLO verification on the real top camera; deploy the newer
  `dlsj/inspect/brick_detector.py` (`make_brick_detector`) to `~/inference/dlsj-smolvla` so the top detector runs on
  the NPU, not ultralytics on the CPU.
- Post-LoRA levers: steps 5 and text 128 on the fine-tuned model (`optimize.py`); baseline-vs-now table done.
- RL demo with the real policy: not done (checkpoint stuck on the XPS); code ready (`modal_demo.py::train`).
- Pick-prompt mismatch between the two prepared datasets.
