#!/bin/bash
# MANUAL VLA TESTING (M + 1/2/3 to choose the bowl). For the real loop run the same args with
# AUTO=1 (-> deploy/smolvla/auto_control.py: automatic spin-and-inspect + place).
# MolmoAct2-SO100_101 zero-shot on the follower via the generic RTC controller (console :8010).
# No --joint-offset: the export maps LeRobot degrees (current calibration) into the checkpoint frame itself.
# Start pose = median demo start pose in the current calibration (extended, near the workspace, per AI2).
cd ~/inference/dlsj-smolvla
export PYTHONPATH=$PWD
exec ~/desmond/venvs/pi05rt/bin/python ~/inference/deploy/smolvla/$([ "${AUTO:-0}" = 1 ] && echo auto_control.py || echo smolvla_control.py) \
  --export ${EXPORT:-$HOME/inference/exports/molmoact2_so101_rtc_int8} --device GPU --cache-dir ~/inference/ov_cache \
  --rtc-horizon ${H:-15} --rtc-guidance ${G:-3} --max-step 4 \
  --pick-prompt "pick up the brick and lift it up" --bring-prompt "" --place-prompt "put the brick in the {colour} bowl" \
  --ready=18.3,-67.0,55.8,60.3,17.5,28.9 \
  --cameras wrist=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0 \
  --realsense top=346422061366,inspect=243722062690 "$@"
