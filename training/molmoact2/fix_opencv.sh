#!/usr/bin/env bash
# Studio's uv.lock (pinned commit) installs BOTH opencv-python 5.0.0.93 and opencv-python-headless 4.13.0.92 into
# the same cv2/ package; the mixed files make `import cv2` fail ("partially initialized module 'cv2'"), which breaks
# `from physicalai.policies import MolmoAct2`. Keep exactly one: the headless build. Idempotent.
#   bash fix_opencv.sh <venv python>
set -euo pipefail
py="${1:?usage: fix_opencv.sh <venv python>}"
if "$py" -c "import cv2; assert cv2.__version__.startswith('4.13')" >/dev/null 2>&1 \
   && [[ -z "$(uv pip list --python "$py" 2>/dev/null | grep -E '^opencv-python ')" ]]; then
  echo "opencv ok: $("$py" -c 'import cv2; print(cv2.__version__)')"; exit 0
fi
uv pip uninstall --python "$py" opencv-python opencv-python-headless
uv pip install --python "$py" --no-deps 'opencv-python-headless==4.13.0.92'
"$py" -c "import cv2; print('opencv fixed:', cv2.__version__)"
