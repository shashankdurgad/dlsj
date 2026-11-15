#!/usr/bin/env bash
# Isaac Lab checkout for the Modal image (same place and ref as sim/twin/docker/build.sh; git-ignored).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD="$HERE/../.build"
REF=${ISAACLAB_REF:-v3.0.0}
mkdir -p "$BUILD"
[ -d "$BUILD/IsaacLab/.git" ] || git clone -q --depth 1 --branch "$REF" https://github.com/isaac-sim/IsaacLab.git "$BUILD/IsaacLab"
git -C "$BUILD/IsaacLab" describe --tags --always
