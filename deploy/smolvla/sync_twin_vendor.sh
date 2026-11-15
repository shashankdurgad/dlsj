#!/bin/bash
# Refresh deploy/smolvla/twin_vendor/ (used by residual.py) from this repo's sim/twin at a git ref, unchanged.
# The deploy machine runs a flat copy of deploy/smolvla/ without sim/, so residual.py reads these copies.
#   deploy/smolvla/sync_twin_vendor.sh            # from HEAD (REF=origin/main for the pushed twin)
#   HOST=watson deploy/smolvla/sync_twin_vendor.sh   # from the twin's working tree on a GB10 instead
# Then re-run: pytest tests/test_residual.py
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(git rev-parse --show-toplevel)
V=twin_vendor
FILES="sim/twin/perception/wrist_target.py sim/twin/rl/gate.py sim/twin/rl/test_gate.py sim/twin/rl/export_residual.py
sim/twin/rl/kin.py sim/twin/rl/DEPLOY.md sim/twin/robot/wrist_mount.json sim/twin/gripper/assets/finray_so101.npz
sim/twin/align/joint_fit/result.json sim/twin/calib/cameras_fit.json"
URDF=sim/twin/assets/so101/urdf/so101_new_calib.urdf  # git-ignored in sim/twin (built from the copy below)
URDF_GIT=sim/twin/joe_port/upstream/synthetic/robot-assets/so101_new_calib.urdf
if [ -n "${HOST:-}" ]; then
  perl -e 'alarm 120; exec @ARGV' ssh -F ~/.brev/ssh_config "$HOST" \
    "cd ~/Documents/dlsj && git log -1 --format='%h %ci' >&2 && git status --short sim/twin >&2 && tar czf - $(echo $FILES $URDF)" \
    | tar xzf - -C "$V"
else
  REF=${REF:-HEAD}
  echo "twin files from $(git -C "$REPO" log -1 --format='%h %ci' "$REF")" >&2
  for f in $FILES; do
    mkdir -p "$V/$(dirname "$f")"
    git -C "$REPO" show "$REF:$f" > "$V/$f"
  done
  mkdir -p "$V/$(dirname "$URDF")"
  git -C "$REPO" show "$REF:$URDF_GIT" > "$V/$URDF"
fi
chmod 644 "$V/sim/twin/align/joint_fit/result.json"
git status --short "$V"
