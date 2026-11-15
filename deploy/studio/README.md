# Physical AI Studio: the project, its datasets and models, and how the loop uses them

Physical AI Studio runs on the XPS in Docker (`physical-ai-studio-xpu`, http://100.77.191.42:7860, commit `8f13241`).
One project holds the whole VLA lifecycle of the brick-sorting robot, and the run-time loop takes its policy from it.

| Studio object | What it is | ID |
|---|---|---|
| Project | **dlsj brick sorting** | `0fad5636-5f89-4d15-b83c-bf069a7b23bf` |
| Robots | SO101 Follower, SO101 Leader (calibration bind-mounted from `~/.cache/huggingface/lerobot/calibration`) | |
| Cameras | `wrist`, `top`, `inspect` | |
| Environment | **so101 brick station** (follower + leader + the three cameras) | `246feb8d-2d73-451a-bdd8-3edfa066c748` |
| Dataset | **cal2 teleop, 3-stage prompts (456 episodes, MolmoAct2 training set)**: the current-calibration demos with the per-frame stage prompts the policy was trained on | `db961a58-3f56-4498-a597-7b948ae32a91` |
| Model (deployed) | **MolmoAct2 SO-101 cal2 3-stage LoRA, step 8000 (OpenVINO INT8 + RTC)**, policy `molmoact2` | `22fb0973-b9df-4c10-bca7-8af2a8f9739d` |
| Earlier models | `pi05 junction 20261114 (OpenVINO INT8)`, `ACT WARP duplo 417` (ACT dropped by the team) | |

## The VLA lifecycle

1. **Robot + cameras + environment** are set up in Studio (above).
2. **Demos** are recorded with `dlsj.teleop_record` against the same arm and cameras (LeRobot v3), then labelled
   into three stages ("pick up the {colour} brick", "bring the brick to the inspection camera", "place the brick in
   the {colour} bowl") and imported into Studio as a dataset.
3. **Fine-tuning** uses Studio's training library (`physicalai`, the same commit `8f13241`): Studio's `MolmoAct2`
   policy, LightningCLI config (`training/molmoact2/`), LoRA r64, 4 cloud GPUs (Modal), bf16. Base model
   `allenai/MolmoAct2-SO100_101`.
4. **Export** with Studio's OpenVINO exporter (`deploy/molmoact2/export.py`, `optimize.py`): RTC inputs patched in
   (`rtc_patch.py`), INT8 symmetric weights. Measured on the Arc B390 (same architecture, `bench/molmoact2_baseline.md`):
   312 ms/chunk median with RTC (285 ms without) vs 10.8 s for the PyTorch CPU baseline.
5. **Registration** in Studio: `import_model.sh` imports the export into the project, linked to its training
   dataset, and checks that Studio's copy is byte-identical (sha256) to the export the loop runs.
6. **Run**: `deploy/smolvla/run_loop.sh` selects the policy by Studio model ID (`STUDIO_MODEL`). The model is loaded
   with Studio's runtime (`physicalai.inference.InferenceModel`, OpenVINO, GPU); the dashboard shows the Studio
   model, and the language instruction for each stage comes from the perception / inspection decision.

## Commands (run on the XPS)

```bash
# code: dlsj/deploy/studio_registry.py -> ~/inference/dlsj-smolvla/dlsj/deploy/, deploy/studio/ -> ~/inference/deploy/studio/
S=~/inference/tmp/studio_stage

# dataset: the training snapshot's labels (meta/ + data/ only, from the Spark) on the XPS recording;
# videos are hard-linked, top kept full-frame (the 560x480 crop is the model's preprocessing)
~/inference/.venv/bin/python ~/inference/deploy/studio/build_dataset.py $S/labels_cal2_456 \
    ~/desmond/build/dlsj-main/data/duplo_teleop_cal2 $S/cal2_456_3stage_studio --full-frame observation.images.top
~/inference/deploy/studio/import_dataset.sh $S/cal2_456_3stage_studio \
    "cal2 teleop, 3-stage prompts (456 episodes, MolmoAct2 training set)" "pick up the blue brick"

# model: Studio copy + sha256 check + the loop's local copy (hard links) + lineage record
REPO=~/inference/dlsj-smolvla ~/inference/deploy/studio/import_model.sh \
    ~/inference/exports/molmo_opt/ft_cal2_456_8k/rtc_int8 \
    "MolmoAct2 SO-101 cal2 3-stage LoRA, step 8000 (OpenVINO INT8 + RTC)" db961a58-3f56-4498-a597-7b948ae32a91 \
    $S/molmo_8k_lineage.json

# what the loop will load
cd ~/inference/dlsj-smolvla && PYTHONPATH=$PWD ~/desmond/venvs/pi05rt/bin/python -m dlsj.deploy.studio_registry \
    show 22fb0973-b9df-4c10-bca7-8af2a8f9739d
~/inference/deploy/smolvla/run_loop.sh --check-only      # resolves via Studio, validates prompts/crop, no arm
STUDIO_MODEL=<id> ~/inference/deploy/smolvla/run_loop.sh  # another Studio model; EXPORT=<dir> for a plain folder
```

`dlsj.deploy.studio_registry` keeps the loop's copy in `~/inference/studio_models/<id>/openvino/` with a
`studio_model.json` record. On every launch it checks the files against Studio's record (count and bytes); if Studio
is down it says so and uses the saved record; a missing or changed copy is re-downloaded through Studio's
`/api/models/<id>/exports/openvino/download`. Tests: `tests/test_studio_registry.py`.

## Known limits

- Studio's built-in **Run model** page feeds RTC inputs only to SmolVLA exports, so it cannot run this RTC
  MolmoAct2 export as is; the loop runs it with the same runtime plus our RTC wrapper. To have it on the Run page too,
  register a plain INT8 export of the same checkpoint (`step008000.ckpt`, export peak ~44 GB RAM: only with the loop
  stopped).
- Studio's Run page uses GPU/CPU; the NPU models (inspection) run in our own OpenVINO code.
- Imported models have no training curves in Studio (the CSV log stayed on the training volume).
