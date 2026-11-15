#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
studio_root="$repo_root/third_party/physical-ai-studio"

if [[ ! -x "$studio_root/application/backend/.venv/bin/physicalai-studio" || ! -f "$studio_root/application/ui/dist/index.html" ]]; then
    echo "Studio needs setup first. See docs/PHYSICAL_AI_STUDIO.md." >&2
    exit 1
fi

cd "$studio_root/application/backend"
export STATIC_FILES_DIR="$studio_root/application/ui/dist"
exec uv run --no-sync physicalai-studio serve --host 127.0.0.1 --port "${PORT:-7860}"
