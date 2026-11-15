#!/usr/bin/env bash
set -euo pipefail
cd /home/joe-spark-2/pi05-isaac-so101
exec > cal2-act-real-frame-check/run.log 2>&1
docker exec -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 pi05-isaac-runtime /work/venv/bin/python /work/cal2-act-real-frame-check/policy_worker_act.py > cal2-act-real-frame-check/policy.log 2>&1 &
pid=$!
for ((i=0;i<120;i++)); do
 if [[ -s cal2-act-real-frame-check/ipc/actions-003.npz ]]; then echo done > cal2-act-real-frame-check/ipc/done.flag; break; fi
 if ! kill -0 "$pid" 2>/dev/null; then tail -20 cal2-act-real-frame-check/policy.log; exit 1; fi
 sleep 1
done
wait "$pid"
echo REAL_FRAME_CHECK_COMPLETE
