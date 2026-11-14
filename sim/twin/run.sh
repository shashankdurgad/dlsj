#!/usr/bin/env bash
# Run one Python script (repo-relative path) in a throwaway twin container:
#   sim/twin/run.sh sim/twin/scripts/smoke.py [args...]      sim/twin/run.sh --shell
# --rm removes the container on exit, --init reaps child processes (no zombies), and the container is named
# dlsj-twin-* so `docker ps --filter name=dlsj-twin` shows (and `docker stop` ends) anything still running.
# The repo is mounted at /repo; Kit / shader caches live in named volumes dlsj-twin-* so restarts are fast.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
NAME="${NAME:-dlsj-twin}-$(date +%s)"
ARGS=(--rm --init --name "$NAME" --device nvidia.com/gpu=all --network host --ipc host --memory "${MEMORY:-64g}"
      -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PYTHONPATH=/repo
      -v "$REPO:/repo" -w /repo
      -v dlsj-twin-kitcache:/isaac-sim/kit/cache -v dlsj-twin-ovcache:/root/.cache/ov
      -v dlsj-twin-glcache:/root/.cache/nvidia/GLCache -v dlsj-twin-computecache:/root/.nv/ComputeCache
      -v dlsj-twin-ovdata:/root/.local/share/ov/data)
if [ "${1:-}" = "--shell" ]; then
  exec docker run -it "${ARGS[@]}" --entrypoint bash "${IMAGE:-dlsj/twin:latest}"
fi
exec docker run "${ARGS[@]}" --entrypoint /workspace/isaaclab/isaaclab.sh "${IMAGE:-dlsj/twin:latest}" -p "$@"
