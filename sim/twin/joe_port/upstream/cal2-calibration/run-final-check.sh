#!/usr/bin/env bash
set -euo pipefail
cd /home/joe-spark-2/pi05-isaac-so101
exec > cal2-calibration/final-check.log 2>&1
SEND=/home/joe-spark-2/isaacsim/skills/isaac-sim-remote/scripts/isaacsim_send.py
for script in cal2-calibration/configure_reference.py synthetic/setup_physics.py cal2-calibration/replay.py; do
 python3 "$SEND" --context cal2 --timeout 600 --execution-timeout 0 --file "$script"
done
bash cal2-act-step7000/run-act.sh
echo CAL2_FINAL_CHECK_COMPLETE
