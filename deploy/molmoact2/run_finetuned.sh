#!/bin/bash
# Fine-tuned MolmoAct2 (cal2-456 3-stage LoRA) on the follower, web console http://<xps>:${PORT:-8020}.
#
#   EXPORT=~/inference/exports/molmo_opt/ft_cal2_456/rtc_int8 BRICK=red ~/inference/deploy/molmoact2/run_finetuned.sh
#   CHECK=1 ... run_finetuned.sh     validate prompts / crop / export only (no arm, no cameras)
#   AUTO=1  ... run_finetuned.sh     the run-time loop (auto_control.py: automatic inspect + bowl decision)
#
# Exactly what the model was trained on (deploy/molmoact2/finetune_cal2_456.json, read from the dataset):
#   Space = "pick up the <BRICK> brick" (blue|brown|red|white|yellow) -> M = "bring the brick to the inspection
#   camera" -> choose the bowl (1 yellow / 2 purple / 3 orange) -> M = "place the brick in the <bowl> bowl";
#   wrist + top cameras, top cropped to 0,0,560,480; current calibration (8a3e5e04), no joint offset;
#   Space first eases the arm to the median start pose of the training episodes.
set -euo pipefail
CFG=~/inference/deploy/molmoact2/finetune_cal2_456.json
EXPORT=${EXPORT:?set EXPORT=<OpenVINO export dir of the fine-tuned model>}
BRICK=${BRICK:-red}
READY=$(python3 -c "import json; print(','.join(str(v) for v in json.load(open('$(eval echo $CFG)'))['ready_pose']))")
cd ~/inference/dlsj-smolvla
export PYTHONPATH=$PWD
CAL=$(sha256sum calibration/robots/so_follower/follower.json | cut -c1-16)
[ "$CAL" = "8a3e5e04c5977ff7" ] || { echo "follower calibration $CAL is not the training one (8a3e5e04)"; exit 1; }
PROG=smolvla_control.py; [ "${AUTO:-0}" = 1 ] && PROG=auto_control.py
EXTRA=(); [ "${CHECK:-0}" = 1 ] && EXTRA=(--check-only)
exec ~/desmond/venvs/pi05rt/bin/python ~/inference/deploy/smolvla/$PROG "${EXTRA[@]}" \
  --export "$EXPORT" --device GPU --cache-dir ~/inference/ov_cache \
  --training-tasks "$CFG" --brick "$BRICK" \
  --pick-prompt "pick up the {brick} brick" --bring-prompt "bring the brick to the inspection camera" \
  --place-prompt "place the brick in the {colour} bowl" --bowls yellow,purple,orange \
  --crop top=0,0,560,480 --ready=$READY \
  --rtc-horizon ${H:-15} --rtc-guidance ${G:-3} --max-step 4 \
  --cameras wrist=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0 \
  --realsense top=346422061366,inspect=243722062690 --port ${PORT:-8020} "$@"
