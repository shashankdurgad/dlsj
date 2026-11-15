#!/usr/bin/env bash
# Register an OpenVINO policy export (a physicalai policy package: manifest.json + IR) as a model of the dlsj
# project in the running Physical AI Studio, then seed the loop's local copy (dlsj.deploy.studio_registry) with
# sha256 checks against Studio's copy. Run on the XPS. Prints the Studio model ID last.
#
#   deploy/studio/import_model.sh EXPORT_DIR "Display name" DATASET_ID [lineage.json]
#
# Studio stores its own copy (docker volume); the host export stays where it is. Light on the machine: one copy
# of the files plus two sha256 passes, all niced, nothing on the GPU.
set -euo pipefail
SRC=$(realpath "${1:?export dir with manifest.json}")
NAME=${2:?display name}
DATASET=${3:?Studio dataset id the model was trained on}
LINEAGE=${4:+$(realpath "$4")}
CONTAINER=${CONTAINER:-physical-ai-studio-xpu}
PROJECT=${PROJECT:-0fad5636-5f89-4d15-b83c-bf069a7b23bf}
PY=${PY:-$HOME/desmond/venvs/pi05rt/bin/python}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
test -f "$SRC/manifest.json" || { echo "no manifest.json in $SRC" >&2; exit 1; }

STAGE=/app/storage/import_$(basename "$SRC")_$(date +%s)
trap 'docker exec -u 0 "$CONTAINER" rm -rf "$STAGE" 2>/dev/null || true' EXIT
# Studio's exports-only layout: <dir>/openvino/manifest.json
docker exec "$CONTAINER" mkdir -p "$STAGE"
nice -n 10 docker cp "$SRC" "$CONTAINER:$STAGE/openvino"
# the backend runs as the owner of /app/storage/models; docker cp leaves host uid/gid
docker exec -u 0 "$CONTAINER" sh -c "chown -R \$(stat -c %u:%g /app/storage/models) $STAGE"
OUT=$(docker exec "$CONTAINER" physicalai-studio models import-dir --source-dir "$STAGE" \
      --project-id "$PROJECT" --dataset-id "$DATASET" --model-name "$NAME" --move)
echo "$OUT"
ID=$(echo "$OUT" | sed -n 's/^Model ID: //p')
[ -n "$ID" ] || { echo "import failed" >&2; exit 1; }
MODEL_PATH=$(echo "$OUT" | sed -n 's/^Model path: //p')

SUMS=$(mktemp)
trap 'rm -f "$SUMS"; docker exec -u 0 "$CONTAINER" rm -rf "$STAGE" 2>/dev/null || true' EXIT
docker exec "$CONTAINER" sh -c "cd '$MODEL_PATH/exports/openvino' && nice -n 10 sha256sum *" > "$SUMS"
cd "$REPO"
PYTHONPATH=$REPO nice -n 10 "$PY" -m dlsj.deploy.studio_registry seed "$ID" "$SRC" --studio-sums "$SUMS" \
    ${LINEAGE:+--lineage "$LINEAGE"}
echo "$ID"
