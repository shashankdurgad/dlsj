#!/usr/bin/env bash
set -euo pipefail
cd /home/joe-spark-2/pi05-isaac-so101
mkdir -p smooth-control-comparison/outputs smooth-control-comparison/ipc
exec > smooth-control-comparison/run.log 2>&1
SEND=/home/joe-spark-2/isaacsim/skills/isaac-sim-remote/scripts/isaacsim_send.py
for script in synthetic/setup_scene.py synthetic/configure_reference.py synthetic/setup_physics.py; do
 python3 "$SEND" --context smooth --timeout 300 --execution-timeout 0 --file "$script"
done
docker exec -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e HF_HUB_CACHE=/work/hf-cache/hub -e HF_XET_CACHE=/work/hf-cache/xet pi05-isaac-runtime /work/venv/bin/python /work/policy_worker_smooth.py > smooth-control-comparison/policy.log 2>&1 &
worker_pid=$!
for ((attempt=0;attempt<180;attempt++)); do
 if [[ -s smooth-control-comparison/outputs/policy-ready.json ]]; then break; fi
 if ! kill -0 "$worker_pid" 2>/dev/null; then tail -25 smooth-control-comparison/policy.log; exit 1; fi
 sleep 2
done
python3 "$SEND" --context smooth --timeout 1800 --execution-timeout 0 --file rollout_smooth.py > smooth-control-comparison/rollout.log 2>&1
wait "$worker_pid"
echo COMPARISON_COMPLETE
