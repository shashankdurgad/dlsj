#!/usr/bin/env bash
# RLPD expert rows: the 8k prior's chunk on the REAL post-train v3.1 frames, residual label = clip(demo - prior).
# 8 shards on 8 GPUs (~5 min), merged into sim/twin/out/rl/expert_img8k_prior.pt.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
NAME=${NAME:-dlsj-twin-expert} sim/twin/run.sh --bash-c 'for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i /opt/isaaclab-venv/bin/python sim/twin/rl/expert.py --latent_snap sim/twin/data/models/molmoact2_ft_8k/step008000.ckpt --prior_chunk 1 --pre all --shard $i/8 --out sim/twin/out/rl/expert_img8k_prior_s$i.pt > sim/twin/out/rl/expert_img8k_prior_s$i.log 2>&1 & sleep 20; done; wait; /opt/isaaclab-venv/bin/python sim/twin/rl/expert.py --merge "sim/twin/out/rl/expert_img8k_prior_s*.pt" --out sim/twin/out/rl/expert_img8k_prior.pt'
