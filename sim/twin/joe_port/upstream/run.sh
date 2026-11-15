#!/usr/bin/env bash
# Re-run the pinned SO-101 policy in the already-configured Spark environment.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-$HOME/isaacsim}"
SEND="$ISAAC_SIM_DIR/skills/isaac-sim-remote/scripts/isaacsim_send.py"
cd "$TASK_ROOT"
python3 "$SEND" --timeout 15 --file "$ISAAC_SIM_DIR/skills/isaac-sim-remote/scripts/health_check.py"
if [[ -f outputs/rollout-result.json ]]; then
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p runs
    mv outputs "runs/outputs-$stamp"
    mv ipc "runs/ipc-$stamp"
    mkdir -p outputs ipc
    python3 "$SEND" --context pi05 --timeout 300 --execution-timeout 0 --file setup_scene.py
    python3 "$SEND" --context pi05 --timeout 120 --file adjust_scene.py
fi
if ! docker exec pi05-isaac-runtime pgrep -f '^/work/venv/bin/python /work/policy_worker.py$' >/dev/null; then
    docker exec -e PYTHONUNBUFFERED=1 -e HF_HUB_CACHE=/work/hf-cache/hub -e HF_XET_CACHE=/work/hf-cache/xet \
        pi05-isaac-runtime /work/venv/bin/python /work/policy_worker.py > logs/policy.log 2>&1 &
fi
deadline=$((SECONDS + 900))
until [[ -f outputs/policy-ready.json ]]; do
    if (( SECONDS >= deadline )); then
        tail -30 logs/policy.log
        exit 1
    fi
    if ! docker exec pi05-isaac-runtime pgrep -f '^/work/venv/bin/python /work/policy_worker.py$' >/dev/null; then
        tail -30 logs/policy.log
        exit 1
    fi
    sleep 2
done
python3 "$SEND" --context pi05 --timeout 3600 --execution-timeout 0 --file rollout_guarded.py | tee logs/rollout.log
"$HOME/.local/bin/ffmpeg" -hide_banner -loglevel error -y -framerate 30 \
    -i outputs/frames/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 20 outputs/rollout.mp4
cat outputs/rollout-result.json
