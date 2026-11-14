# SmolVLA fine-tune on GCP (official LeRobot recipe, wrist + top cameras)

A fallback / comparison VLA for the π0.5 run, trained on the **same curated dataset and the same
train/validation episodes**, but without the `inspect` camera. It runs on its own 1-GPU Spot VM, so it does not
compete with the π0.5 GPUs.

## Recipe

The official one from the [LeRobot SmolVLA docs](https://huggingface.co/docs/lerobot/smolvla), unchanged:

```bash
lerobot-train --policy.path=lerobot/smolvla_base --batch_size=64 --steps=20000 \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' ...
```

Everything else comes from the pinned `smolvla_base` config, which `train.py` checks (`RECIPE`) but never
overrides. In short: only the action expert and the state projection train (100M of 450M parameters; the VLM is
frozen); AdamW at lr 1e-4, betas (0.9, 0.95), weight decay 1e-10, grad clip 10; cosine schedule, warmup 1,000
and decay over 30,000 steps to 2.5e-6; chunk 50, all 50 actions executed; 10 denoising steps; images padded to
512×512; MEAN_STD state/action normalisation.

- **LeRobot rescales the schedule.** With 20k steps (fewer than the 30k decay steps), LeRobot 0.7 scales warmup
  to about 667 and decay to 20k ("Auto-scaling LR scheduler" in the log). That is LeRobot's own behaviour for
  this recipe.
- **20k steps × batch 64 is about 13 passes** over the 96,621 training frames.

## Pinned versions (`versions.env`)

The latest versions as of 2026-11-14: LeRobot **0.7.0** (the newest release, and the recorder's version),
`lerobot/smolvla_base` **5e8d12a6** and `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` **4cde9d20**.
`cache_base.sh` packs both into a Hugging Face hub cache in GCS. The VM restores it and trains with
`HF_HUB_OFFLINE=1`, so the run cannot silently pick up a newer upstream commit, and the checkpoint still names
the canonical repos, so it loads on the XPS. `train.py` refuses to start if `main` doesn't resolve to the
pinned revisions.

## Data (`prepare_data.py`)

Input: `gs://onflow-30k-pi05-artifacts/datasets/duplo517-3stage-20261114` (the 3-stage dataset built on Joe
Spark 2: 517 source episodes, `top` already cropped to 560×480; `CURATION.json` selects 514 episodes, excluding
1 and 10 (brick already grasped at frame 0) and 464 (incomplete placement), split 464 train / 50 validation).
The old `pi05-517-curated-20261114` dataset is no longer used.

- Keeps only `wrist` and `top`. Drops `observation.images.inspect`, `observation.images.inspect_depth` and
  `observation.images.top_depth` using LeRobot's `remove_feature`. LeRobot 0.7.0 leaves the removed cameras'
  per-episode `videos/…` and `stats/…` columns behind, which breaks `split_dataset`, so the script removes those
  columns. Nothing else changes.
- Splits into train / validation with the exact episode ids in `CURATION.json` (464 / 50; 96,621 train frames).
- Recomputes state/action stats **from train only**, and gives validation the same stats.
- Task strings in the data (three stages): `pick up the brick` (until `grasp_frame`), `bring the brick to the
  inspection camera` (until `inspection_frame`), and `place the brick in the {orange,purple,yellow} bowl`. Use
  these exact strings at run time.
- Not yet re-run on the new dataset: the counts above come from its `CURATION.json`, not from this script.

## Run it

```bash
# 0. once: pin + mirror the base model (refuses to overwrite)
bash training/smolvla/cache_base.sh /tmp/smolvla-base --upload

# 1. dry run, then create the VM (stage "gate" by default; powers off when done)
bash training/smolvla/launch_vm.sh smolvla-517-2cam-<date>
bash training/smolvla/launch_vm.sh smolvla-517-2cam-<date> --create
```

**Stage `gate`** (about 30–45 min): environment setup, data prep, a 4-step smoke run, then a 200-step timing run.
It uploads `gate/000200/pretrained_model` and `gate/timing.json` (median s/step and the projected 20k-step
hours). **Before the full run, check the deploy path with this checkpoint:** import it into Studio, or export it
to OpenVINO and load it on the XPS. If neither works, stop: the judging needs a Studio/OpenVINO path.

**Stage `full`:**
```bash
gcloud compute instances add-metadata <name> --zone=europe-west4-b --metadata=stage=full,max-uptime-s=21600
gcloud compute instances start <name> --zone=europe-west4-b
```
Training stops at 20k steps or the uptime budget (6 h by default), whichever comes first. Checkpoints (1.2 GB
each, every 2,000 steps) go to `gs://onflow-30k-pi05-artifacts/runs/<name>/full/checkpoints/<step>/` as each
one completes. After a Spot preemption, just start the VM again: it restores the newest checkpoint from GCS
and resumes with LeRobot's `--resume`, keeping the data order.

Expected time: about 2–4 h of training on 1× RTX PRO 6000. This extrapolates LeRobot's figure of about 4 h for
20k steps on one A100; the gate's `timing.json` gives the measured number.

## Deploying the checkpoint

- Feed exactly **two** cameras: `top` (cropped `0,0,560,480@640x480`) as `observation.images.camera1` and
  `wrist` as `observation.images.camera2`. Pass the same `--rename_map` you trained with. Don't send a third
  camera. `config.json` still lists `camera3` from the base model; SmolVLA skips missing cameras, in training
  and at inference alike.
- The checkpoint carries its own (train) normalisation in `policy_preprocessor*.safetensors`.
- The `inspect` camera is still used by Anomalib, just not by the policy.

## Verified locally (2026-11-14, CPU, no GPU)

- `prepare_data.py` ran on the old curated dataset (checksums verified); it has not yet been re-run on the 3-stage dataset.
- `train.py --smoke` ran 4 real `lerobot-train` steps offline from the pinned cache. The loss was finite, and
  the checkpoint recorded the train stats and the rename map.
- `--resume` resumed the data order at the right sample.
- `run.sh`'s checkpoint sync was tested with a stub `gcloud`. `launch_vm.sh` was dry-run only.
- `python3 -m unittest discover -s training/smolvla -p 'test_*.py'` passes.

**Not yet run:** anything on a GCP VM or GPU (stages `gate`/`full`), the Studio/OpenVINO export of a SmolVLA
checkpoint, and robot evaluation. Validation loss says nothing about task success. Compare it with π0.5 on the
real arm from the same start positions: Isaac Sim's calibration error makes every policy miss there.

## Known differences from the π0.5 recipe

- π0.5's `phase_data.py` stops pickup-instruction action chunks at the inspection frame. Plain LeRobot does not,
  so chunks near that boundary pair the pickup instruction with placement motion. If SmolVLA hesitates right
  after presenting the brick, look here first.
- SmolVLA's official recipe trains only the action expert. The π0.5 official recipe fine-tunes every weight.
