#!/bin/bash
# Static GitHub Pages build of the project page (https://shashankdurgad.github.io/dlsj/).
#   deploy/showcase/build_pages.sh [outdir] [media_dir]   (default outdir deploy/showcase/dist)
#
# Sources: deploy/showcase/site/ (index.html, style.css, site.js, fonts/, logos/). No build step.
# Team names: the TEAM array at the top of site/site.js.
# Media (videos, posters, images) are not kept on main to keep the repo light. They are taken from
# media_dir if given, else from the published gh-pages branch (git archive origin/gh-pages media).
# Videos: H.264 high, yuv420p, -movflags +faststart, no audio; hero = team demo at 5x (setpts=PTS/5).
#
# Publish: build into a gh-pages worktree, commit, push origin HEAD:gh-pages.
set -eo pipefail
cd "$(dirname "$0")"
OUT=${1:-dist}
MEDIA=${2:-}
rm -rf "$OUT"; mkdir -p "$OUT"
cp -R site/. "$OUT/"
# Cache-bust the stylesheet and script (Pages serves them with max-age=600).
V=$(cat site/style.css site/site.js | shasum | cut -c1-10)
sed -i.bak "s/?v=dev/?v=$V/g" "$OUT/index.html" && rm "$OUT/index.html.bak"
if [ -n "$MEDIA" ]; then
  mkdir -p "$OUT/media"; cp -R "$MEDIA/." "$OUT/media/"
else
  git fetch -q origin gh-pages
  ROOT=$(git rev-parse --show-toplevel)
  git -C "$ROOT" archive origin/gh-pages media | tar -x -C "$OUT"
fi
mkdir -p "$OUT/research"
cat > "$OUT/research/index.html" <<'HTML'
<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Adaptive Inspection</title>
<meta http-equiv="refresh" content="0; url=../"><link rel="canonical" href="../"></head>
<body><a href="../">Adaptive Inspection</a></body></html>
HTML
touch "$OUT/.nojekyll"
echo "built $OUT ($(du -sh "$OUT" | cut -f1))"
