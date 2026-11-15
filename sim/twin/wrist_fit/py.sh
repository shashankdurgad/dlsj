#!/usr/bin/env bash
# Plain python (no Isaac) in the twin image for the wrist-fit loop: sim/twin/wrist_fit/py.sh <script> [args]
# GPU unless GPU=0; throwaway container named dlsj-twin-wristfit-x-* (counts toward the box's GPU-container cap).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
gpu=(); [ "${GPU:-1}" = 1 ] && gpu=(--device nvidia.com/gpu=all)
exec timeout -k 30 "${T:-3000}" docker run --rm --init --name "dlsj-twin-wristfit-x-$(date +%s)" "${gpu[@]}" \
  --network host --ipc host -v "$REPO:/repo" -w /repo -e PYTHONPATH=/repo -e PYTHONUNBUFFERED=1 \
  --entrypoint /opt/isaaclab-venv/bin/python dlsj/twin:latest "$@"
