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
      -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=N -e OMNI_KIT_ACCEPT_EULA=YES -e PYTHONPATH=/repo
      -v "$REPO:/repo" -w /repo
      -v dlsj-twin-kitcache:/isaac-sim/kit/cache -v dlsj-twin-ovcache:/root/.cache/ov
      -v dlsj-twin-glcache:/root/.cache/nvidia/GLCache -v dlsj-twin-computecache:/root/.nv/ComputeCache
      -v dlsj-twin-ovdata:/root/.local/share/ov/data)
# pass secrets by NAME only (the value never appears on a command line)
# the twin's render path (Fabric scene delegate off + VisualSync); sim/residual_env/app.py reads DLSJ_KIT_ARGS
ARGS+=(-e "DLSJ_KIT_ARGS=${DLSJ_KIT_ARGS---/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false}")
for v in WANDB_API_KEY HF_TOKEN; do [ -n "${!v:-}" ] && ARGS+=(-e "$v"); done
if [ "${1:-}" = "--bash-c" ]; then  # sim/twin/run.sh --bash-c "cmd; cmd"  (a shell command in the container)
  exec docker run "${ARGS[@]}" --entrypoint bash "${IMAGE:-dlsj/twin:latest}" -c "$2"
fi
if [ "${1:-}" = "--shell" ]; then
  exec docker run -it "${ARGS[@]}" --entrypoint bash "${IMAGE:-dlsj/twin:latest}"
fi
exec docker run "${ARGS[@]}" --entrypoint /workspace/isaaclab/isaaclab.sh "${IMAGE:-dlsj/twin:latest}" -p "$@"
