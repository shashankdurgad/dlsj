#!/usr/bin/env bash
# Ablation (c) on a local 8-GPU box (2 procs / GPU), the same settings as the Modal job abl_c_8g.yaml.
# WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT from the environment (W&B off when no key).
set -euo pipefail
cd "$(dirname "$0")/../../../.."
GPUS=${GPUS:-8}
WB=()
if [ -n "${WANDB_API_KEY:-}" ]; then
  WB=(--wandb "--wandb_project ${WANDB_PROJECT:-dlsj-twin-dicerl} --wandb_entity ${WANDB_ENTITY:?set WANDB_ENTITY}")
fi
NAME=${NAME:-dlsj-twin-ablc} MEMORY=${MEMORY:-256g} sim/twin/run.sh sim/twin/rl/multi_image.py \
  --gpus "$GPUS" --procs_per_gpu 2 --run abl_c --minutes "${MINUTES:-90}" --stagger_s 8 "${WB[@]}" -- \
  --detector yolo --latent 0 --top_target 0 --veto 0 --start ready --K 1 --bc_weight 2 \
  --dr ramp --dr_scale 0.5 --spot_scale 1.0 --dr_ramp_eps 0,500 --phys_dr 1 --play_dr 1 \
  --clutter abl_c --p_near 0.5 --eval_every_min 20 --eval_eps 40 --eval_dr_eps 8 \
  --expert sim/twin/out/rl/expert_img8k_prior.pt --drift_joint_sd 3,3,3,1,0,0
