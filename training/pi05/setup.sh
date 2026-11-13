#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$here/versions.env"
studio="${STUDIO_DIR:-$here/.studio}"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo 'Run setup on the Linux x86_64 NVIDIA training VM.' >&2; exit 1;
}
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'Install the NVIDIA driver first.' >&2; exit 1; }
if [[ ! -d "$studio/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout \
    https://github.com/open-edge-platform/physical-ai-studio.git "$studio"
else
  [[ -z "$(git -C "$studio" status --porcelain)" ]] || {
    echo "Refusing to change a modified Studio checkout: $studio" >&2; exit 1;
  }
fi
git -C "$studio" fetch --depth 1 origin "$STUDIO_COMMIT"
GIT_LFS_SKIP_SMUDGE=1 git -C "$studio" checkout --detach "$STUDIO_COMMIT"
uv sync --project "$studio/library" --locked --python 3.12 \
  --extra cu128 --extra pi05 --extra loggers
"$studio/library/.venv/bin/python" - <<'PY'
import torch
from triton.runtime import driver

assert torch.cuda.is_available()
print(torch.__version__)
print([(torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1))
       for i in range(torch.cuda.device_count())])
print("Triton CUDA device:", driver.active.get_current_device())
PY
"$studio/library/.venv/bin/physicalai" fit --help >/dev/null
echo 'Environment ready. Authenticate to Hugging Face with the environment'
echo "CLI: $studio/library/.venv/bin/hf auth login"
echo 'The account needs access to google/paligemma-3b-pt-224.'
