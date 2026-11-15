#!/usr/bin/env bash
# Build the twin image in two layers:
#   dlsj/isaaclab-base:3.0.0  Isaac Lab v3.0.0's own docker/Dockerfile.base on nvcr.io/nvidia/isaac-sim:6.2.0
#   dlsj/twin:latest          + Physical AI Studio's library (docker/Dockerfile)
# The Isaac Lab checkout lives in sim/twin/.build (git-ignored), so nothing is written outside the repo.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD="$HERE/../.build"
REF=${ISAACLAB_REF:-v3.0.0}
mkdir -p "$BUILD"
[ -d "$BUILD/IsaacLab/.git" ] || git clone -q --depth 1 --branch "$REF" https://github.com/isaac-sim/IsaacLab.git "$BUILD/IsaacLab"
docker build -f "$BUILD/IsaacLab/docker/Dockerfile.base" \
  --build-arg ISAACSIM_VERSION_ARG=6.2.0 --build-arg ISAACSIM_ROOT_PATH_ARG=/isaac-sim \
  --build-arg ISAACLAB_PATH_ARG=/workspace/isaaclab --build-arg DOCKER_USER_HOME_ARG=/root \
  -t dlsj/isaaclab-base:3.0.0 "$BUILD/IsaacLab"
docker build -f "$HERE/Dockerfile" -t dlsj/twin:latest "$HERE"
docker image ls --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep '^dlsj/'
