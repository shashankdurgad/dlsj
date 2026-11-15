#!/bin/bash
# THE RUN-TIME LOOP: closed-loop brick sorting + perception dashboard (loop_control.py) on http://<xps>:${PORT:-8020}/
# The policy is chosen by its Physical AI Studio model ID (STUDIO_MODEL, default: the fine-tuned MolmoAct2 8k,
# registered in the "dlsj brick sorting" project): dlsj.deploy.studio_registry checks it against Studio and gives
# the export dir. EXPORT=<dir> overrides with a plain export folder. Extra args go to loop_control.py /
# smolvla_control.py.
# Grasp residual (pick only): append --residual shadow|on --residual-weights <deploy/residual_actor.npz>
# (deploy/smolvla/residual.py; off by default). POST /ctl/residual_off fades it out live.
# loop_control.py is the one next to this script (~/inference/deploy/smolvla/ on the XPS, or a test checkout).
HERE=$(cd "$(dirname "$0")" && pwd)
cd ~/inference/dlsj-smolvla
export PYTHONPATH=$PWD
PY=~/desmond/venvs/pi05rt/bin/python
mkdir -p ~/inference/logs/loop
STUDIO_MODEL=${STUDIO_MODEL-22fb0973-b9df-4c10-bca7-8af2a8f9739d}
if [ -z "${EXPORT:-}" ] && [ -n "$STUDIO_MODEL" ]; then
  EXPORT=$($PY -m dlsj.deploy.studio_registry resolve "$STUDIO_MODEL") || {
    echo "Physical AI Studio model $STUDIO_MODEL could not be resolved (see above); EXPORT=<dir> runs a plain export" >&2
    exit 1
  }
fi
exec $PY "$HERE/loop_control.py" \
  --export ${EXPORT:-$HOME/inference/exports/molmo_opt/ft_cal2_456_8k/rtc_int8} --device GPU --cache-dir ~/inference/ov_cache \
  --training-tasks ~/inference/deploy/molmoact2/finetune_cal2_456.json --brick blue \
  --pick-prompt "pick up the {brick} brick" --bring-prompt "bring the brick to the inspection camera" \
  --place-prompt "place the brick in the {colour} bowl" --bowls yellow,purple,orange --crop top=0,0,560,480 \
  --ready=26.3,-51.8,30.1,73.1,15.9,37.7 --rtc-horizon ${H:-15} --rtc-guidance ${G:-3} --max-step ${MAX_STEP:-4} \
  --cameras wrist=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0 \
  --realsense top=346422061366,inspect=243722062690 --port ${PORT:-8020} "$@"
