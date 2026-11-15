#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-$HOME/isaacsim}"
SEND="$ISAAC_SIM_DIR/skills/isaac-sim-remote/scripts/isaacsim_send.py"
cd "$TASK_ROOT"
python3 "$SEND" --timeout 15 --file "$ISAAC_SIM_DIR/skills/isaac-sim-remote/scripts/health_check.py"
if docker exec pi05-isaac-runtime pgrep -f '^/work/venv/bin/python /work/policy_worker.py$' >/dev/null; then
    echo 'An existing policy worker is active; wait for it to finish.' >&2
    exit 1
fi
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p runs
if [[ -d outputs ]]; then mv outputs "runs/outputs-$stamp"; fi
if [[ -d ipc ]]; then mv ipc "runs/ipc-$stamp"; fi
mkdir -p outputs ipc logs
if [[ "${1:-}" == --reuse-scene ]]; then
    cp "runs/outputs-$stamp"/reference-*.png outputs/
    cp "runs/outputs-$stamp"/reference-configuration.json "runs/outputs-$stamp"/scene-reference.usda outputs/
else
    python3 "$SEND" --context pi05 --timeout 300 --execution-timeout 0 --file setup_scene.py
    python3 "$SEND" --context pi05 --timeout 120 --file configure_reference.py
fi
docker exec -e PYTHONUNBUFFERED=1 -e HF_HUB_CACHE=/work/hf-cache/hub -e HF_XET_CACHE=/work/hf-cache/xet \
    pi05-isaac-runtime /work/venv/bin/python /work/policy_worker.py > logs/policy-reference.log 2>&1 &
worker_pid=$!
deadline=$((SECONDS + 900))
until [[ -f outputs/policy-ready.json ]]; do
    if (( SECONDS >= deadline )) || ! kill -0 "$worker_pid" 2>/dev/null; then
        tail -30 logs/policy-reference.log
        exit 1
    fi
    sleep 2
done
python3 "$SEND" --context pi05 --timeout 3600 --execution-timeout 0 --file rollout_reference.py | tee logs/rollout-reference.log
wait "$worker_pid"
"$HOME/.local/bin/ffmpeg" -hide_banner -loglevel error -y -framerate 30 -i outputs/frames/%05d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 20 outputs/rollout.mp4
cat outputs/rollout-result.json
