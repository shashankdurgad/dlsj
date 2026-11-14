#!/usr/bin/env bash
# Download the pinned SmolVLA base + SmolVLM2 backbone into a Hugging Face hub cache, check that `main`
# resolves to the pinned revisions, pack it, and (with --upload) mirror it to GCS for the training VM.
#
#   bash cache_base.sh <work dir>            # download + pack + verify only
#   bash cache_base.sh <work dir> --upload   # also upload to $BASE_CACHE_URI (refuses to overwrite)
#
# The VM restores the tarball into $HF_HOME/hub and trains with HF_HUB_OFFLINE=1, so LeRobot loads
# exactly these revisions under their canonical names (checkpoint configs stay portable to the XPS).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$here/versions.env"
work="${1:?usage: cache_base.sh <work dir> [--upload]}"
upload="${2:-}"
export HF_HOME="$work/hf-home" HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME"
python3 - <<PY
from huggingface_hub import snapshot_download
for repo, rev, ignore in (("$SMOLVLA_REPO", "$SMOLVLA_REVISION", ["*.gif", "*.ipynb"]),
                          ("$SMOLVLM_REPO", "$SMOLVLM_REVISION", ["onnx/*"])):
    path = snapshot_download(repo, revision="main", ignore_patterns=ignore)
    if not path.endswith(rev):
        raise SystemExit(f"{repo} main is now {path.rsplit('/', 1)[-1]}, pinned {rev}: update versions.env deliberately")
    print("OK", repo, rev)
PY
tar_name="hf-hub-cache.tar"
tar -C "$HF_HOME" -cf "$work/$tar_name" hub
(cd "$work" && shasum -a 256 "$tar_name" > "$tar_name.sha256" && cat "$tar_name.sha256")
du -h "$work/$tar_name"
if [[ "$upload" == --upload ]]; then
  if gcloud storage ls "$BASE_CACHE_URI/$tar_name" >/dev/null 2>&1; then
    echo "$BASE_CACHE_URI/$tar_name already exists; not overwriting" >&2; exit 1
  fi
  gcloud storage cp "$work/$tar_name" "$work/$tar_name.sha256" "$BASE_CACHE_URI/"
  echo "uploaded to $BASE_CACHE_URI/"
fi
