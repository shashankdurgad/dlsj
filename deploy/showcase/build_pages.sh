#!/bin/bash
# Static GitHub Pages build of the public research page (deploy/showcase/research/):
#   / -> research/ ; research/ reads ../media, ../static (three.js, twin, robot model, fonts), ../story.json, ../replay.json
#   deploy/showcase/build_pages.sh [outdir]   (default deploy/showcase/dist)
set -e
cd "$(dirname "$0")"
OUT=${1:-dist}
rm -rf "$OUT"; mkdir -p "$OUT"
cp -R research "$OUT/research"
cp -R media "$OUT/media"
mkdir -p "$OUT/static"
cp -R static/vendor static/robot static/fonts static/twin.js "$OUT/static/"
cp story.json replay.json "$OUT/"
cat > "$OUT/index.html" <<'HTML'
<!doctype html><html lang="en"><head><meta charset="utf-8"><title>DLSJ</title>
<meta http-equiv="refresh" content="0; url=research/"><link rel="canonical" href="research/"></head>
<body><a href="research/">DLSJ: closed-loop defect sorting</a></body></html>
HTML
touch "$OUT/.nojekyll"
echo "built $OUT ($(du -sh "$OUT" | cut -f1))"
