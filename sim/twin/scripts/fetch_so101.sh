#!/usr/bin/env bash
# Fetch the SO-101 description (URDF + meshes) the team's sim uses, pinned, into sim/twin/assets/so101/urdf
# (git-ignored). Same model as Joe's so101_new_calib USD: TheRobotStudio/SO-ARM100 Simulation/SO101.
set -euo pipefail
REF=${SO_ARM100_REF:-a758567c3978dfeefe282ede0500085a48fe8f78}
DST=$(cd "$(dirname "$0")/.." && pwd)/assets/so101/urdf
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
curl -sfL "https://codeload.github.com/TheRobotStudio/SO-ARM100/tar.gz/$REF" | tar -xz -C "$TMP" --wildcards '*/Simulation/SO101/*'
rm -rf "$DST"; mkdir -p "$DST"; cp -r "$TMP"/*/Simulation/SO101/. "$DST"/
echo "$REF" > "$DST/REF"; ls "$DST" | head; echo "meshes: $(ls "$DST/assets" | wc -l)"
