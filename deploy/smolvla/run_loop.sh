#!/bin/bash
# THE RUN-TIME LOOP: closed-loop brick sorting + perception dashboard (loop_control.py) on http://<xps>:${PORT:-8020}/
# EXPORT defaults to the newest fine-tuned MolmoAct2 (8k steps). Extra args go to loop_control.py / smolvla_control.py.
cd ~/inference/dlsj-smolvla
export PYTHONPATH=$PWD
mkdir -p ~/inference/logs/loop
exec ~/desmond/venvs/pi05rt/bin/python ~/inference/deploy/smolvla/loop_control.py \
  --export ${EXPORT:-$HOME/inference/exports/molmo_opt/ft_cal2_456_8k/rtc_int8} --device GPU --cache-dir ~/inference/ov_cache \
  --training-tasks ~/inference/deploy/molmoact2/finetune_cal2_456.json --brick blue \
  --pick-prompt "pick up the {brick} brick" --bring-prompt "bring the brick to the inspection camera" \
  --place-prompt "place the brick in the {colour} bowl" --bowls yellow,purple,orange --crop top=0,0,560,480 \
  --ready=26.3,-51.8,30.1,73.1,15.9,37.7 --rtc-horizon ${H:-15} --rtc-guidance ${G:-3} --max-step ${MAX_STEP:-4} \
  --cameras wrist=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0 \
  --realsense top=346422061366,inspect=243722062690 --port ${PORT:-8020} "$@"
