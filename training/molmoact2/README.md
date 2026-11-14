# MolmoAct2-SO100_101 fine-tune in Physical AI Studio (Ai2's recipe)

Fine-tunes [`allenai/MolmoAct2-SO100_101`](https://huggingface.co/allenai/MolmoAct2-SO100_101) (Ai2, about 5.4B
parameters, already trained on SO-100/101 data with absolute joint control) on our teleop demos. It follows
**Ai2's recommended small-dataset recipe** and trains through **Physical AI Studio's own MolmoAct2 policy**
at the pinned commit, so the checkpoint exports to OpenVINO for the XPS.

## Recipe and where each setting comes from

| Setting | Value | Source |
|---|---|---|
| Start checkpoint | `allenai/MolmoAct2-SO100_101` @ `152569fe` | Ai2 `experiments/README.md`: "SO100/SO101 fine-tuning" |
| Method | **Currently Studio's default LoRA, NOT yet Ai2's split**: rank 64 / alpha 16 adapters on the LLM, ViT, connector *and* action expert, with every base weight frozen | Ai2 instead puts adapters only on LLM + ViT linears and trains the connector and action expert in full (`experiments/README.md`, "LoRA Fine-Tuning"). See PLAN in the chat; not implemented yet |
| Learning rates | **5e-5 for every group** (LLM, ViT, connector, action expert) | Ai2 LoRA command; LeRobot MolmoAct2 docs |
| Optimizer | AdamW (0.9, 0.95), eps 1e-6, weight decay 0, per-group grad clip 1.0 | Studio MolmoAct2 defaults = Ai2 optimizer |
| Schedule | warmup 200, cosine to 1e-6 over the run | Studio/Ai2 |
| Global batch | **32** (8 per GPU on 4 GPUs) | LeRobot docs (Ai2 port): "< 200 demonstrations: global batch 16–32" |
| Steps | **8,000** (about 5.6 passes over the 199 training episodes) | Our choice: Ai2's 50k-step command is for large datasets. Snapshots every 2,000 steps, so earlier ones can be tested on the arm |
| Prompt metadata | setup `single so100/so101 robotic arm in molmoact2`, control `absolute joint pose` | The checkpoint's `so100_so101_molmoact2` norm tag / Ai2 `so100_so101` mixture |
| Action chunk | 30 predicted, 30 executed (1 s at 30 fps) | Same norm tag (`action_horizon` 30) |
| Joint frame | `adapt_to_so101: true`, no `calibration` (our joints are already LeRobot degrees, so the scale is 1) | Studio docs: use it when fine-tuning MolmoAct2-SO100_101 ("generally converges faster") |
| Image augmentation | 95% crop, ±5° rotation, colour jitter, blur p=0.2 (`augment.Ai2Augment`) | Ai2 original recipe (LeRobot docs list the first three; Ai2 adds the blur) |
| Cameras | wrist + top (top cropped to 560×480) | Team decision for this round; matches the 517-episode runs |

Two deliberate differences from Ai2's own trainer, both inherited from the pinned Studio trainer:
- Studio has no sequence packing. Ai2 also starts custom fine-tunes with `--packing=false`.
- The phase-aware data module (`phase_data.py`) stops pick-instruction action chunks at the inspection frame,
  as in the π0.5 runs.

## Data: `duplo_teleop_cal2` (221 demos on Spark 2)

`/home/joe-spark-2/desmond/datasets/duplo-cal2-221-20261114/duplo_teleop_cal2` contains:
- 221/221 successful teleop takes, 50,831 frames at 30 fps (episodes 0–99 are identical to the earlier
  100-demo import);
- the same task strings as the 517 set (bowls: 83 purple / 72 yellow / 66 orange);
- no calibration file. We reuse the cal2 `follower.json` captured with the 100-demo import (sha256 `8a3e5e04…`),
  because the recorder and arm were not changed between the two imports;
- recorded after the cal2 recalibration. **Don't merge it with the 517 set without checking the joint
  calibration.**

```bash
python prepare_data.py <...>/duplo_teleop_cal2 <out>/molmoact2-cal2-221 \
  --calibration <...>/calibration-observed-at-transfer/robots/so_follower/follower.json
```

This removes `inspect` and both depth streams, crops `top` to 560×480, copies the labels and calibration,
and writes `TRAIN-NORMALIZATION.json`. The normalization is computed over **exactly the 199 training
episodes** of Studio's seed-42 split, so the 22 validation episodes stay unseen. The script finishes with
`SHA256SUMS.txt` and `READY.json`. Upload the folder to `DATASET_URI` (in `versions.env`), then never
change it. Add later recordings as a new snapshot and a new run.

## Run on GCP

```bash
bash launch_vm.sh molmoact2-cal2-221-<date>            # dry run
bash launch_vm.sh molmoact2-cal2-221-<date> --create   # 4x RTX PRO 6000 Spot VM, stage "gate"
```

**Stage `gate`** (about 1 h). It sets up Studio with the `molmoact2` extra, downloads the model, and checks
that it is still the pinned revision before going offline. It then verifies dataset checksums and runs a
smoke run (1 train + 1 val batch). Next comes a **200-step timing run**: `timing.json` records the measured
s/step, peak memory and the projected hours for 8k steps. Finally it does an **OpenVINO export of the 200-step
checkpoint**, uploaded to `gate/openvino-step200`.

**Load that export on the XPS GPU and time it before the full run.** A 5B model on an Arc iGPU is the
biggest risk of this plan.

**Stage `full`:**
```bash
gcloud compute instances add-metadata <name> --zone=europe-west4-b --metadata=stage=full,max-uptime-s=25200
gcloud compute instances start <name> --zone=europe-west4-b
```
Checkpoints go to `runs/<name>/full/checkpoints/`:
- `snapshots/` every 2,000 steps;
- `best/`, chosen by validation loss;
- `resume/last.ckpt`, the full training state.

Each one uploads when complete. After a Spot preemption, start the VM again: it restores `last.ckpt` and
resumes.

**Time:** not measured yet. My estimate is about 3.5–5.5 h for 8k steps on 4 GPUs (MolmoAct2 at about 1.5–2× π0.5's cost per
sample); the gate's `timing.json` gives the measured number. Checkpoints may be large, because Lightning
saves the full ~5.4B weights. Watch disk space and GCS uploads.

## Verified so far

- Unit tests pass: `python -m unittest discover -s training/molmoact2 -p 'test_*.py'`. They cover the recipe
  values, the Studio split replication, the pick-chunk hold, augmentation shapes, the source checks and the
  train-only stats.
- `run.sh` checkpoint sync was tested with a stub `gcloud`. `launch_vm.sh` was dry-run only.
- Real data: the 221-demo snapshot was prepared and checked independently: 99/99 checksums; state, action,
  task and wrist frames identical across episodes 0/104/160/220; top crop within 0.9%; the instruction switches at
  the labelled inspection frame; train stats cover 46,025 frames.
- **Not yet run:** the Studio MolmoAct2 environment, any training step, the OpenVINO export, and anything
  on a GPU or the XPS. Ai2 has published no SO-101 fine-tuning results for this checkpoint, so judge it on
  the real arm against π0.5 and SmolVLA from the same start positions.
