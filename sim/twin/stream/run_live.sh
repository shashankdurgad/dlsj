#!/usr/bin/env bash
# Start (or restart) the live twin mirror as ONE detached container named dlsj-twin-live-<timestamp>:
#   sim/twin/stream/run_live.sh [live.py args...]       (default: --pose_url http://127.0.0.1:8040/api/pose) e.g. --pose_url http://127.0.0.1:8041/api/pose --look minimal
#   sim/twin/stream/run_live.sh --stop                  stop + remove every dlsj-twin-live* container
#   docker logs -f $(docker ps -q --filter name=dlsj-twin-live)
# Any old dlsj-twin-live* container is removed first. No --rm: the container uses --restart unless-stopped instead
# (the two conflict), so a crash / watchdog exit (live.py exits on a dead HTTP child or 60 s of black frames) restarts
# it; --stop removes it for good. The stream binds 127.0.0.1:8030 by default (the Mac forwards it over SSH); --bind auto
# = this host's Tailscale IPv4; it is never 0.0.0.0 unless you pass --bind 0.0.0.0 yourself.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
old=$(docker ps -aq --filter name=dlsj-twin-live)
if [ -n "$old" ]; then
  echo "stopping old live container(s): $(docker ps -a --filter name=dlsj-twin-live --format '{{.Names}}' | tr '\n' ' ')"
  docker rm -f $old >/dev/null
fi
[ "${1:-}" = "--stop" ] && exit 0
n=$(docker ps --format '{{.Names}}' | grep -c '^dlsj-twin' || true)
[ "$n" -ge 4 ] && echo "WARNING: $n Isaac (dlsj-twin*) containers already running on this box (keep <= 4)"
NAME="dlsj-twin-live-$(date +%s)"
docker run -d --init --restart unless-stopped --name "$NAME" --device nvidia.com/gpu=all --network host --ipc host \
  --memory "${MEMORY:-32g}" -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=N -e OMNI_KIT_ACCEPT_EULA=YES -e PYTHONPATH=/repo \
  -v "$REPO:/repo" -w /repo \
  -v dlsj-twin-kitcache:/isaac-sim/kit/cache -v dlsj-twin-ovcache:/root/.cache/ov \
  -v dlsj-twin-glcache:/root/.cache/nvidia/GLCache -v dlsj-twin-computecache:/root/.nv/ComputeCache \
  -v dlsj-twin-ovdata:/root/.local/share/ov/data \
  --entrypoint /workspace/isaaclab/isaaclab.sh "${IMAGE:-dlsj/twin:latest}" -p sim/twin/stream/live.py "$@"
echo "started $NAME ($*); logs: docker logs -f $NAME"
