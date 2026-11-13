#!/usr/bin/env bash
# Run the dlsj station tools on the XPS, reachable from any tailnet machine at http://100.77.191.42:8000
#
#   ssh -t intel@100.77.191.42 '~/desmond/dlsj/scripts/xps.sh teleop'    # -t so Ctrl+C reaches it
#   ssh -t intel@100.77.191.42 '~/desmond/dlsj/scripts/xps.sh record --repo-id local/duplo_teleop --task "..." --operator me'
#   ssh -t intel@100.77.191.42 '~/desmond/dlsj/scripts/xps.sh cameras'     # camera check only, arms untouched
#
# Extra arguments are passed through (e.g. --calibrate leader, --invert wrist_roll).
# Cameras use stable device paths so a re-plug can't swap them. Stop Physical AI Studio's own teleop first:
# only one program may drive the arm.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-$HOME/inference/.venv/bin/python}   # LeRobot 0.7 + OpenCV env already on the XPS
WRIST=${WRIST:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}
TOP=${TOP:-/dev/v4l/by-path/pci-0000:00:0d.0-usb-0:3.4.1:1.3-video-index0}   # RealSense D415 colour
CAMS=${CAMS:-wrist=$WRIST,top=$TOP}

if [ ! -t 0 ]; then
  echo "WARNING: no terminal. Ctrl+C won't reach this program; run it as: ssh -t intel@... '$0 ...'" >&2
fi

mode=${1:-teleop}
shift || true
case "$mode" in
  teleop)  exec "$PY" -m dlsj.teleop_web --host 0.0.0.0 --cameras "$CAMS" "$@" ;;
  record)  exec "$PY" -m dlsj.teleop_record --host 0.0.0.0 --cameras "$CAMS" "$@" ;;
  cameras) exec "$PY" -m dlsj.teleop_web --no-robot --host 0.0.0.0 --cameras "$CAMS" "$@" ;;
  *) echo "usage: $0 teleop|record|cameras [args...]" >&2; exit 2 ;;
esac
