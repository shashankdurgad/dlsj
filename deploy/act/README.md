# ACT → OpenVINO → Physical AI Studio on the XPS

Deploys the team's LeRobot ACT checkpoint
[`w3joe/act-warp-duplo-417-20261114`](https://huggingface.co/w3joe/act-warp-duplo-417-20261114) @ `1a7f83ec7bd7e58acdd76a6b99a738c4705a4f0f`
(LeRobot 0.7.0, ResNet-18, 100-step chunks of 6 joint targets, cameras `wrist`/`top`/`inspect`, `top` trained on the
crop x 0–560 of the 640×480 frame). Everything below was measured on the XPS (Core Ultra X9 388H, Arc B390, NPU)
on 14 Nov 2026, with Desmond's `teleop_record` running throughout (load average 3–16), so latencies vary by up to ~2×
between runs. Logs are in `~/inference/logs/act/`.

**Studio model ID: `1c994313-8a00-4323-a771-b7d511385f9c`**, "ACT WARP duplo 417 (20261114, OpenVINO FP32, top crop
in graph)", policy `act`, backend `openvino`, project `0fad5636-…`, dataset `2b424ecf-…`.

| File | Env | What it does |
|---|---|---|
| `torch_reference.py` | `~/inference/.venv` (LeRobot 0.7) | 50 recorded frames → LeRobot pre → ACT → post; saves frames + chunks + recorded future actions (`act_warp_417_ref.npz`) |
| `torch_bench.py` | `~/inference/.venv` | PyTorch baseline latency (XPU, CPU) on the same frames |
| `ov_parity.py` | `~/inference/.venv-actov` | `dlsj.deploy.act_export` export vs reference: parity, crop check, latency per device |
| `studio_convert.py` | Studio library `.venv` | LeRobot checkpoint → Studio `model.ckpt` + Studio `exports/openvino` (crop in graph) |
| `studio_parity.py` | Studio library `.venv` | Studio's `InferenceModel` on the export vs reference, latency per device |
| `import_to_studio.sh` | XPS host | `docker cp` + `physicalai-studio models import-dir` + API check |
| `backend_check.py` | inside the Studio container | Studio backend's `model_inputs` + model-worker `Inference.infer` on the imported model |
| `../../dlsj/deploy/act_studio.py` | numpy (+ openvino) | Helpers: stats conversion, crop, manifest check, frame sampling, MAE, IR re-stack fix. Tests: `tests/test_act_deploy.py` |

## Commands (as run)

```bash
# code: dlsj/ + deploy/act/ rsynced to ~/inference/dlsj-act on the XPS
cd ~/inference/models && ~/inference/.venv/bin/hf download w3joe/act-warp-duplo-417-20261114 \
    --revision 1a7f83ec7bd7e58acdd76a6b99a738c4705a4f0f --local-dir act-warp-duplo-417-20261114
cd act-warp-duplo-417-20261114 && sha256sum -c SHA256SUMS.txt          # all 7 files OK
M=~/inference/models/act-warp-duplo-417-20261114; R=~/inference/exports/act_warp_417_ref.npz; cd ~/inference/dlsj-act

# 1. PyTorch reference (50 frames, seed 0, one per episode, each with 100 recorded future actions)
nice -n 5 ~/inference/.venv/bin/python deploy/act/torch_reference.py --policy_path $M \
    --dataset ~/desmond/dlsj/data/duplo_teleop_rgb --out $R
nice -n 5 ~/inference/.venv/bin/python deploy/act/torch_bench.py $M $R --device xpu cpu

# 2. Our OpenVINO export (single graph, normalisation inside, crop applied by OpenVINOACT)
#    .venv-actov = new venv layered on ~/inference/.venv via a .pth file, plus openvino 2026.4.1, nncf 3.4.0, pytest
for p in fp32 fp16 int8; do nice -n 5 ~/inference/.venv-actov/bin/python -m dlsj.deploy.act_export --policy_path $M \
    --out ~/inference/exports/act_warp_417_$p --precision $p --crop observation.images.top=0,0,560,480@640x480; done
nice -n 5 ~/inference/.venv-actov/bin/python deploy/act/ov_parity.py ~/inference/exports/act_warp_417_fp16 $R --device CPU GPU NPU

# 3. Studio model folder + Studio runtime parity
cd ~/inference/physical-ai-studio/library
nice -n 5 .venv/bin/python ~/inference/dlsj-act/deploy/act/studio_convert.py --policy_path $M \
    --out ~/inference/exports/act_warp_417_studio
nice -n 5 .venv/bin/python ~/inference/dlsj-act/deploy/act/studio_parity.py \
    ~/inference/exports/act_warp_417_studio/exports/openvino $R --device CPU GPU NPU

# 4. Import (what import_to_studio.sh does) and verify
deploy/act/import_to_studio.sh ~/inference/exports/act_warp_417_studio "ACT WARP duplo 417 (20261114, OpenVINO FP32, top crop in graph)"
curl -s localhost:7860/api/projects/0fad5636-5f89-4d15-b83c-bf069a7b23bf/models   # lists it, policy "act"
curl -s localhost:7860/api/models/1c994313-8a00-4323-a771-b7d511385f9c          # model + export io_spec
```

## Results

### Download and reference
- `SHA256SUMS.txt`: all 7 files OK, on the Mac and on the XPS.
- Reference frames: `duplo_teleop_rgb` (LeRobot v3, 417 episodes / 85,647 frames, the same counts as the training
  set), pyav decode, 50 frames from 50 different episodes. Pixels are exactly uint8/255, so the uint8 frames given
  to every runtime are bit-identical to what LeRobot decoded.
- **Sanity check against recorded actions** (mean abs error, degrees, per joint
  pan/lift/elbow/wrist_flex/wrist_roll/gripper), next to a "hold the current position" baseline:

| | first 10 steps | first 100 steps |
|---|---|---|
| ACT prediction | 3.8 / 9.5 / 6.1 / 5.0 / 3.9 / 2.9 | 6.6 / 8.4 / 7.5 / 5.9 / 6.8 / 3.8 |
| hold position | 3.8 / 13.5 / 9.2 / 4.5 / 4.0 / 2.8 | 19.1 / 35.3 / 30.3 / 17.9 / 18.6 / 16.8 |

  Over the full chunk the policy is 3–4× closer to what the operator did than standing still, so weights,
  normalisation and the top crop are wired correctly. Over the first 10 steps it is no better than holding for
  4 of 6 joints (the recorded action leads the state by a few degrees, and these are frames the model trained on,
  so this is not a generalisation measure).

### Parity (actions in degrees, 50 recorded frames, vs the PyTorch CPU reference)

| Runtime | Device | max abs | mean abs | p99 abs |
|---|---|---|---|---|
| PyTorch-XPU (LeRobot) | XPU | 0.0002 | 0.0000 | 0.0001 |
| `act_export` FP32, f32 hint | CPU / GPU | 0.0003 | 0.0000 | 0.0001 |
| `act_export` FP16 | CPU | 0.15 | 0.017 | 0.08 |
| `act_export` FP16 | GPU | 0.38 | 0.037 | 0.18 |
| `act_export` FP16 | NPU | 2.03 | 0.097 | 0.57 |
| `act_export` INT8 (NNCF weight-only) | CPU / GPU / NPU | 11.3–12.1 | 0.46–0.48 | 2.2–2.4 |
| **Studio `InferenceModel`, imported export** | CPU | 0.0003 | 0.0000 | 0.0001 |
| **Studio `InferenceModel`** | GPU | 0.44 | 0.038 | 0.18 |
| **Studio `InferenceModel`** | NPU | 2.31 | 0.095 | 0.59 |
| **Studio backend (`model_inputs` + model-worker `Inference.infer`) in the container** | CPU / GPU | 0.0003 / 0.44 | 0.00002 / 0.038 | |

Action std is 15–52° per joint, so FP16 on GPU is ~0.1% of a std. **INT8 weight-only is not usable**: 12° worst case
and no speed-up on GPU (19.5 vs 19.7 ms). Use FP16/FP32.

### Latency (one 100-step chunk, batch 1, static shapes, median / p90 ms)

| Path | CPU | GPU (Arc B390) | NPU |
|---|---|---|---|
| PyTorch LeRobot fp32 (baseline) | 3762 / 3814 | **35.1 / 35.8** (XPU) | – |
| `act_export` FP32 (f32 hint) | 237 / 242 | 47.8 / 49.1 | – |
| `act_export` FP16 | 229–338 | **19.7–20.6** / 20.2–21.7 | **11.2–12.4** / 14.9–15.0 |
| `act_export` INT8 | 183 / 294 | 19.5–21.5 | 13.1 / 16.7 |
| Studio `InferenceModel` (FP32 IR; GPU runs f16 by default) | 293–310 / 317–329 | **12.2–20.2** / 12.6–21.2 | 16.9–21.4 / 17.8–23.6 |
| Studio backend `Inference.infer` (container) | 265 | 17.9 | – |

The ranges are repeat runs minutes apart under the recorder's load. Studio's pre-processing (uint8→float, a no-op
letterbox) costs about 6 ms per call (cProfile). The CPU PyTorch number is with the recorder running and is
only there for completeness; the baseline to quote is PyTorch-XPU 35 ms → OpenVINO GPU FP16 ~20 ms (1.7×) or NPU FP16
~12 ms (2.9×). Compile time with a warm cache: 0.1–0.7 s; cold GPU 13–23 s, NPU 5–6 s.

## The top-camera crop

The policy was trained on `top` cropped to x 0–560 (480×560). Studio's environment `so101 brick station` delivers
`top` as a 640×480 RGB frame (RealSense D415, `ColorMode.RGB`), like `wrist` and `inspect`.

- **Studio has no per-camera crop or ROI** (nothing in the backend, the library or the camera config).
  Studio's ACT inference pipeline is `to_float_tensor` → `resize(image_resolution=<one size>, mode="letterbox")` for
  every camera. `ACT._from_hf` / `image_size` take the size of the first camera only.
- What would happen without a fix: with `image_size` 480×640 (the size of wrist/inspect), the export's `top` input
  becomes 480×640 and Studio feeds the full uncropped frame, so the network sees 80 extra columns it never trained
  on (I confirmed Studio's export letterbox-pads the trained 480×560 sample to 480×640 when tracing). With
  `image_size` 480×560 every camera would be squeezed and padded.
- **Fix used: bake the crop into the exported graph.** `studio_convert.py` sets `image_size` to 480×640 (a no-op
  resize for all three cameras), wraps Studio's ACT model so `images.top` takes the full 480×640 frame and slices
  x 0–560 as the first op, and exports with Studio's own `policy.export(backend="openvino")`. Studio's manifest,
  pre/post-processors and runtime are unchanged. `crop.json` in the export records the crop.
  Checks: the wrapper equals cropped input in torch (max diff 0.0); our runtime gives identical actions for a full
  and a pre-cropped top frame (0.0 CPU, ≤2e-5 GPU); Studio parity above uses full 640×480 frames.
- Trade-offs: (+) works with Studio's stock runtime, UI and import; nothing to install in the container; the crop
  cannot be forgotten. (−) Only this export carries the crop: `model.ckpt` is the plain Studio ACT with `top`
  480×560, so if Studio re-exports or fine-tunes from it, Studio would feed full frames again (letterboxed), and a
  Studio dataset would need cropped `top` frames. Re-run `studio_convert.py` instead of exporting from the UI.
  (−) If the `top` camera is ever set to another resolution, the graph rejects it (static shape) rather than
  rescaling. Alternatives considered: a custom manifest preprocessor (needs our code inside the container), a
  Studio camera ROI (doesn't exist), retraining on uncropped frames (changes the model).

## How the Studio model was made (and two IR fixes)

- Studio's backend runs policy `act` with `physicalai.policies.ACT`, its own port of LeRobot's ACT (same module
  names). `ACT(pretrained_name_or_path=<dir>)` would load the LeRobot weights but **not the normalisation**: LeRobot
  0.7 stores stats in the processor safetensors, and Studio only logs a warning and keeps mean 0 / std 1.
  `studio_convert.py` builds `ACT(dataset_stats=…)` from the saved normaliser instead (types as plain strings like
  Studio's datasets, so the checkpoint loads with `weights_only=True`), loads the weights with Studio's own
  `_remap_lerobot_act_state_dict` (234 tensors, 51.67M values, no unexpected keys; the only "missing" keys are
  normaliser buffers and `explain_attention_layer`, an alias of the decoder cross-attention, which the script checks
  is the same module), writes `model.ckpt` like Studio's `to_torch`, and reloads it with `ACT.load_from_checkpoint`
  (max diff 0.0).
- **Studio's stock ACT export does not run on the Arc B390 GPU (OpenVINO 2026.4.1)**:
  1. compile: `[GPU] ProgramBuilder build failed! … primitive_onednn_base.h:559: could not execute a primitive`,
     bisected to `decoder.layers.0/aten::add/Add_1` (the decoder's first self-attention runs on a zeros tensor and
     was left unfolded). Fixed with OpenVINO's `ConstantFolding` pass (also folds the sinusoidal position
     embeddings);
  2. inference, same message, bisected to an 872-input `Concat`: Studio's ACT builds the encoder tokens with
     `tokens.extend(list(cam_features))` + `torch.stack`, i.e. 2 + 300 + 270 + 300 single-row inputs. Splitting it into
     64-input concats did not help. `act_studio.collapse_unstack_restack` replaces each camera's run of
     `X[0], …, X[N-1]` by `X` (exact; CPU output identical, 0.0).
  Both are applied in place to the saved IR by `studio_convert.py`; CPU parity is unchanged (0.0003) and GPU/NPU
  then work. Both look like Studio/OpenVINO issues worth reporting upstream. (Our `act_export` graph, traced from
  LeRobot with torch 2.15, has neither problem.)

## Known limitations

- **Not run on the robot.** "Run model" was not pressed. The backend check replaces the camera store with recorded
  frames; it does not prove the live Studio cameras look like the recordings. The training data was recorded with
  `dlsj.teleop_record` (auto-exposure/WB locked); Studio's camera config shows `exposure: null`. Compare a live
  Studio `top`/`inspect` frame with a recorded one before trusting rollouts.
- The UI was not inspected (no screenshot). The model is visible through the same API the UI uses
  (`/api/projects/<p>/models`, `/api/models/<id>`), with `available_backends: ["openvino"]` and the export io_spec.
- Studio's environment-fit check (`model_inputs`) verifies camera names and state size only, not resolution; it
  passes for `so101 brick station` (wrist/top/inspect, 6 joints).
- The imported export is FP32 IR (Studio's ACT default). On GPU, OpenVINO runs it in f16 anyway (parity above).
- NPU FP16 has a 2° worst-case error over 50 frames (mean 0.1°). Acceptable for 100-step chunks in our view, but
  check on held-out episodes before using NPU for control.
- 50 reference frames come from training episodes. The MAE table is a wiring check, not an accuracy estimate.
- Studio leaves `ACT.forward` behaviour unchanged, so chunk execution (how many of the 100 actions are played
  before re-querying) is Studio's `ModelControl` default. Not checked here.
- `.venv-actov` is a new venv layered on `~/inference/.venv` (LeRobot 0.7, torch XPU) via a `.pth` file. Nothing
  was installed into the existing envs.

## Closed-loop question for the team (ACT has no language input)

ACT takes images and joint state only, so the loop cannot tell it "place in the orange bowl". In `duplo_teleop_rgb`
every one of the 417 episodes is pick-and-present (task 1) followed by one place task in the same episode:
purple 148, yellow 135, orange 134 episodes. ACT never sees those strings. Either the choice of bowl comes from what
the cameras see, or ACT averages the three placements. That breaks our design, where
Anomalib + `dlsj/loop/reasoner.py` decide the bowl and condition the policy with an instruction.

**Question:** in the 417 demos, is the bowl determined visually? For example, does the operator always put a brick
with a given mark in a given bowl, with the mark visible on `inspect`/`wrist` during the place phase? If so, ACT is
doing its own defect classification end to end, and Anomalib's decision can only gate or veto the motion; it can't
steer it. If not, the options are: (a) one ACT per bowl (or per phase: pick-and-present, then place-yellow /
place-orange / place-purple), switched by the reasoner; (b) add a task/bowl input to ACT (e.g. a one-hot token in
the state vector) and retrain; (c) keep π0.5/SmolVLA for the language-conditioned place step. We should measure
which bowl the policy actually chooses on held-out episodes per brick class before deciding.
