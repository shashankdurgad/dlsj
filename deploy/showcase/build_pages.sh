#!/bin/bash
# Static (GitHub Pages) build of the showcase: the page + recorded data instead of the live robot API.
#   deploy/showcase/build_pages.sh [outdir]   (default deploy/showcase/dist)
set -e
cd "$(dirname "$0")"
OUT=${1:-dist}
rm -rf "$OUT"; mkdir -p "$OUT/api"
cp -R static/. "$OUT/"
cp -R media "$OUT/media"
echo '{"offline": true, "public": true, "error": ""}' > "$OUT/api/state"
cp replay.json "$OUT/api/replay"
python3 - "$OUT/api/story" <<'PY'
import json, sys
s = json.load(open("story.json"))
s["gallery"] = [
  {"src": "media/twin.jpg", "caption": "The live 3D digital twin: the SO-101 follows the real joint angles; plates and bricks are placed from the top camera."},
  {"src": "media/mission.jpg", "caption": "Mission control during a run: top-camera detections, the inspection and wrist feeds (blurred here: bystanders), device split, cycle log and event feed."},
  {"src": "media/how.jpg", "caption": "How it works: perceive, act, detect, reason and place, each on its device (GPU, NPU or CPU)."},
  {"src": "media/detection.jpg", "caption": "Defect detection on the NPU: rig evaluation of every inspector we tried."},
  {"src": "media/posttraining.jpg", "caption": "Post-training: LoRA fine-tune, the Isaac Lab digital twin and residual RL."},
  {"src": "media/runtime.jpg", "caption": "Runtime optimisation: 34x faster, 36x less energy, 4.5x less memory, 3x more accurate than the stock baseline."}
]
json.dump(s, open(sys.argv[1], "w"))
PY
touch "$OUT/.nojekyll"
echo "built $OUT ($(du -sh "$OUT" | cut -f1))"
