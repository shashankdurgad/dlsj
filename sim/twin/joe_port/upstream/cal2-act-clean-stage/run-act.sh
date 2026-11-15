#!/usr/bin/env bash
set -euo pipefail
cd /home/joe-spark-2/pi05-isaac-so101
mkdir -p cal2-act-clean-stage/outputs cal2-act-clean-stage/ipc
exec > cal2-act-clean-stage/run.log 2>&1
SEND=/home/joe-spark-2/isaacsim/skills/isaac-sim-remote/scripts/isaacsim_send.py
# Configure the red2x4 scene and corrected camera estimates before evaluating ACT.
for script in cal2-calibration/configure_reference.py synthetic/setup_physics.py cal2-calibration/clean-background.py; do
 python3 "$SEND" --context cal2 --timeout 600 --execution-timeout 0 --file "$script"
done
docker exec -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e HF_HUB_CACHE=/work/hf-cache/hub -e HF_XET_CACHE=/work/hf-cache/xet pi05-isaac-runtime /work/venv/bin/python /work/cal2-act-clean-stage/policy_worker_act.py > cal2-act-clean-stage/policy.log 2>&1 &
worker_pid=$!
for ((attempt=0;attempt<180;attempt++)); do
 if [[ -s cal2-act-clean-stage/outputs/policy-ready.json ]]; then break; fi
 if ! kill -0 "$worker_pid" 2>/dev/null; then tail -25 cal2-act-clean-stage/policy.log; exit 1; fi
 sleep 2
done
python3 "$SEND" --context cal2 --timeout 1800 --execution-timeout 0 --file cal2-act-clean-stage/rollout_act.py > cal2-act-clean-stage/rollout.log 2>&1
wait "$worker_pid"
echo COMPARISON_COMPLETE
