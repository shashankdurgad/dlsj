#!/bin/bash
# MANUAL VLA TESTING (M + 1/2/3 to choose the bowl). For the real loop run the same args with
# AUTO=1 (-> deploy/smolvla/auto_control.py: automatic spin-and-inspect + place).
cd ~/inference/dlsj-smolvla
export PYTHONPATH=$PWD
exec ~/desmond/venvs/pi05rt/bin/python ~/inference/deploy/smolvla/$([ "${AUTO:-0}" = 1 ] && echo auto_control.py || echo smolvla_control.py) \
  --export ~/inference/exports/smolvla_3stage_rtc_int8 --device GPU --cache-dir ~/inference/ov_cache \
  --joint-offset recal_2026-11-14 --rtc-horizon 15 --rtc-guidance 3 --max-step 4 \
  --cameras wrist=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0 \
  --realsense top=346422061366,inspect=243722062690 "$@"
