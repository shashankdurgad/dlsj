#!/usr/bin/env bash
# Plain python (no Isaac / Kit) in a throwaway twin container: sim/twin/dynamics/py.sh sim/twin/dynamics/ident.py [args]
# CPU only, 8 GB cap; the container is named dlsj-twin-dyn-x-* and removed on exit.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
exec docker run --rm --init --name "dlsj-twin-dyn-x-$(date +%s)" --memory "${MEMORY:-8g}" -e PYTHONPATH=/repo \
  -e MPLBACKEND=Agg -v "$REPO:/repo" -w /repo --entrypoint /opt/isaaclab-venv/bin/python "${IMAGE:-dlsj/twin:latest}" "$@"
