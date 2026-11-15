#!/usr/bin/env bash
set -euo pipefail
cd /home/joe-spark-2/pi05-isaac-so101
mkdir -p act-step7000/outputs act-step7000/ipc
exec > act-step7000/run.log 2>&1
SEND=/home/joe-spark-2/isaacsim/skills/isaac-sim-remote/scripts/isaacsim_send.py
for script in synthetic/setup_scene.py synthetic/configure_reference.py synthetic/setup_physics.py; do
 python3 "$SEND" --context act7000 --timeout 300 --execution-timeout 0 --file "$script"
done
docker exec -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e HF_HUB_CACHE=/work/hf-cache/hub -e HF_XET_CACHE=/work/hf-cache/xet pi05-isaac-runtime /work/venv/bin/python /work/act-step7000/policy_worker_act.py > act-step7000/policy.log 2>&1 &
worker_pid=$!
for ((attempt=0;attempt<180;attempt++)); do
 if [[ -s act-step7000/outputs/policy-ready.json ]]; then break; fi
 if ! kill -0 "$worker_pid" 2>/dev/null; then tail -25 act-step7000/policy.log; exit 1; fi
 sleep 2
done
python3 "$SEND" --context act7000 --timeout 1800 --execution-timeout 0 --file act-step7000/rollout_act.py > act-step7000/rollout.log 2>&1
wait "$worker_pid"
echo COMPARISON_COMPLETE
