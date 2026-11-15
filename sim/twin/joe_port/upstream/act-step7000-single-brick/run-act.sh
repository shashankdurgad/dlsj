#!/usr/bin/env bash
set -euo pipefail
cd /home/joe-spark-2/pi05-isaac-so101
mkdir -p act-step7000-single-brick/outputs act-step7000-single-brick/ipc
exec > act-step7000-single-brick/run.log 2>&1
SEND=/home/joe-spark-2/isaacsim/skills/isaac-sim-remote/scripts/isaacsim_send.py
# Reuse the idle act7000 scene; all controller and camera settings are retained.
docker exec -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e HF_HUB_CACHE=/work/hf-cache/hub -e HF_XET_CACHE=/work/hf-cache/xet pi05-isaac-runtime /work/venv/bin/python /work/act-step7000-single-brick/policy_worker_act.py > act-step7000-single-brick/policy.log 2>&1 &
worker_pid=$!
for ((attempt=0;attempt<180;attempt++)); do
 if [[ -s act-step7000-single-brick/outputs/policy-ready.json ]]; then break; fi
 if ! kill -0 "$worker_pid" 2>/dev/null; then tail -25 act-step7000-single-brick/policy.log; exit 1; fi
 sleep 2
done
python3 "$SEND" --context act7000 --timeout 1800 --execution-timeout 0 --file act-step7000-single-brick/rollout_act.py > act-step7000-single-brick/rollout.log 2>&1
wait "$worker_pid"
echo COMPARISON_COMPLETE
