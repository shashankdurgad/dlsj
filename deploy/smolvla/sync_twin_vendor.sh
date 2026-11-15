#!/bin/bash
# Refresh deploy/smolvla/twin_vendor/ (used by residual.py) from the twin on watson, unchanged.
# Then re-run: pytest tests/test_residual.py
set -euo pipefail
cd "$(dirname "$0")/twin_vendor"
FILES="sim/twin/perception/wrist_target.py sim/twin/rl/gate.py sim/twin/rl/test_gate.py sim/twin/rl/export_residual.py
sim/twin/rl/kin.py sim/twin/rl/DEPLOY.md sim/twin/assets/so101/urdf/so101_new_calib.urdf sim/twin/robot/wrist_mount.json
sim/twin/gripper/assets/finray_so101.npz sim/twin/align/joint_fit/result.json sim/twin/calib/cameras_fit.json"
perl -e 'alarm 120; exec @ARGV' ssh -F ~/.brev/ssh_config "${HOST:-watson}" \
  "cd ~/Documents/dlsj && git log -1 --format='%h %ci' >&2 && tar czf - $(echo $FILES)" | tar xzf -
chmod 644 sim/twin/align/joint_fit/result.json
git status --short .
