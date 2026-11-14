#!/bin/bash
# Fine-tuned MolmoAct2 (cal2-456 3-stage LoRA) -> tested OpenVINO exports on the XPS.
#
#   ~/inference/deploy/molmoact2/export_finetuned.sh /path/to/checkpoint.ckpt [name] [variants]
#
# Builds the FP16 / 10-step reference, INT8 and INT8+RTC from the Studio Lightning checkpoint (LoRA merged by
# Studio), checks them on recorded cal2 frames with the TRAINING prompts and top crop (finetune_cal2_456.json), and
# writes ~/inference/exports/molmo_opt/<name>/report.md. Re-traces peak ~44 GB RAM: nothing else heavy may run, and
# optimize.py waits while a robot controller is running. Then:
#   EXPORT=~/inference/exports/molmo_opt/<name>/rtc_int8 BRICK=red ~/inference/deploy/molmoact2/run_finetuned.sh
set -euo pipefail
CKPT=${1:?usage: export_finetuned.sh CHECKPOINT.ckpt [name] [variants]}
NAME=${2:-ft_cal2_456}
VARIANTS=${3:-fp16_ref,int8_sym,rtc_int8}
D=~/inference/deploy/molmoact2
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
free -g | awk '/Mem:/ { if ($7 < 46) print "WARNING: only " $7 " GB available; the trace needs ~44 GB" }'
cd ~/inference/physical-ai-studio/library
export HF_HOME=~/inference/hf-molmo HF_HUB_OFFLINE=1 TMPDIR=~/inference/tmp/onnx
mkdir -p "$TMPDIR"
exec .venv/bin/python $D/optimize.py --name "$NAME" --ckpt "$CKPT" \
  --prompts $D/prompts_cal2_456.json --crop "top=0,0,560,480" --brick red \
  --variants "$VARIANTS"
