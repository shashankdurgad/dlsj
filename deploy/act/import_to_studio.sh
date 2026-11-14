#!/usr/bin/env bash
# Import a Studio-format ACT model folder (model.ckpt + exports/openvino) into the running Studio container and
# print the resulting model record. Run on the XPS. Usage:
#   deploy/act/import_to_studio.sh ~/inference/exports/act_warp_417_studio "ACT WARP duplo 417 (...)"
set -euo pipefail
SRC=${1:?model folder with model.ckpt and exports/openvino}
NAME=${2:?display name}
CONTAINER=${CONTAINER:-physical-ai-studio-xpu}
PROJECT=${PROJECT:-0fad5636-5f89-4d15-b83c-bf069a7b23bf}
DATASET=${DATASET:-2b424ecf-3f75-4c72-a77d-8277ca66da2a}
STAGE=import_$(basename "$SRC")_$(date +%s)

# stage a copy (import-dir --move consumes the source inside the container, the original stays on the host)
TMP=$(mktemp -d ~/inference/tmp/act_import.XXXX)
cp -r "$SRC/model.ckpt" "$SRC/exports" "$TMP/"
docker cp "$TMP" "$CONTAINER:/app/storage/$STAGE"
rm -rf "$TMP"
# the backend runs as the owner of /app/storage/models; docker cp leaves host uid/gid
docker exec -u 0 "$CONTAINER" sh -c "chown -R \$(stat -c %u:%g /app/storage/models) /app/storage/$STAGE"
OUT=$(docker exec "$CONTAINER" physicalai-studio models import-dir --source-dir "/app/storage/$STAGE" \
      --project-id "$PROJECT" --dataset-id "$DATASET" --model-name "$NAME" --move)
echo "$OUT"
ID=$(echo "$OUT" | sed -n 's/^Model ID: //p')
curl -s "http://localhost:7860/api/models/$ID" | python3 -m json.tool | head -40
