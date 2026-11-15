#!/usr/bin/env bash
# Lint (and optionally fix) the twin code with ruff in a throwaway container: sim/twin/lint.sh [--fix] [paths...]
# Standard: sim/twin/ruff.toml. New or changed files must pass before they are committed.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
FIX=(); [ "${1:-}" = "--fix" ] && { FIX=(--fix); shift; }
PATHS=("${@:-sim/twin}")
exec docker run --rm --init -v "$REPO:/repo" -w /repo ghcr.io/astral-sh/ruff:0.8.6 \
  check --config sim/twin/ruff.toml "${FIX[@]}" "${PATHS[@]}"
