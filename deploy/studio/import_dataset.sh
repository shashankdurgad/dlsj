#!/usr/bin/env bash
# Import a LeRobot v3 dataset folder into the dlsj project of the running Physical AI Studio, attached to the
# project's SO-101 environment (robot + wrist/top/inspect cameras). Run on the XPS. Prints the Studio dataset ID last.
#
#   deploy/studio/import_dataset.sh DATASET_DIR "Display name" ["default task"]
#
# Studio keeps its own copy in its storage volume; DATASET_DIR is left as it is.
set -euo pipefail
SRC=$(realpath "${1:?LeRobot v3 dataset dir (meta/, data/, videos/)}")
NAME=${2:?display name}
TASK=${3:-}
CONTAINER=${CONTAINER:-physical-ai-studio-xpu}
PROJECT=${PROJECT:-0fad5636-5f89-4d15-b83c-bf069a7b23bf}
ENVIRONMENT=${ENVIRONMENT:-246feb8d-2d73-451a-bdd8-3edfa066c748}  # "so101 brick station"
test -f "$SRC/meta/info.json" || { echo "no meta/info.json in $SRC" >&2; exit 1; }

STAGE=/app/storage/import_ds_$(basename "$SRC")_$(date +%s)
trap 'docker exec -u 0 "$CONTAINER" rm -rf "$STAGE" 2>/dev/null || true' EXIT
nice -n 10 docker cp "$SRC" "$CONTAINER:$STAGE"
docker exec -u 0 "$CONTAINER" sh -c "chown -R \$(stat -c %u:%g /app/storage/datasets) $STAGE"
OUT=$(docker exec "$CONTAINER" physicalai-studio datasets import-dir --source-dir "$STAGE" \
      --project-id "$PROJECT" --environment-id "$ENVIRONMENT" --dataset-name "$NAME" --default-task "$TASK" --move)
echo "$OUT"
ID=$(echo "$OUT" | grep -oiE 'dataset id: *[0-9a-f-]{36}' | grep -oE '[0-9a-f-]{36}' | tail -1)
[ -n "$ID" ] || { echo "import failed" >&2; exit 1; }
echo "$ID"
