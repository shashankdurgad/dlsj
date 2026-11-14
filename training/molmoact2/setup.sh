#!/usr/bin/env bash
# Studio checkout at the pinned commit with the MolmoAct2 extra (separate from training/pi05/.studio, whose
# venv carries the pi05 extra). Run on the Linux x86_64 NVIDIA training VM.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$here/versions.env"
studio="${STUDIO_DIR:-$here/.studio}"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Run on the Linux x86_64 GPU VM.' >&2; exit 1; }
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'Install the NVIDIA driver first.' >&2; exit 1; }
if [[ ! -d "$studio/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout \
    https://github.com/open-edge-platform/physical-ai-studio.git "$studio"
else
  [[ -z "$(git -C "$studio" status --porcelain)" ]] || { echo "Modified Studio checkout: $studio" >&2; exit 1; }
fi
git -C "$studio" fetch --depth 1 origin "$STUDIO_COMMIT"
GIT_LFS_SKIP_SMUDGE=1 git -C "$studio" checkout --detach "$STUDIO_COMMIT"
uv sync --project "$studio/library" --locked --python 3.12 --extra cu128 --extra molmoact2 --extra loggers
"$studio/library/.venv/bin/python" - <<'PY'
import torch, peft
from physicalai.policies import MolmoAct2  # noqa: F401
assert torch.cuda.is_available()
print("torch", torch.__version__, "peft", peft.__version__,
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
echo "Studio ready at $studio (commit $STUDIO_COMMIT)"
