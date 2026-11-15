#!/usr/bin/env bash
# Periodic audits while the twin changes: sim/twin/audit/loop.sh [RUNS=5] [INTERVAL_S=3000]
# Sleeps INTERVAL_S, runs one audit, RUNS times, then exits (bounded: nothing is left running). Start detached:
#   setsid nohup sim/twin/audit/loop.sh 5 3000 > sim/twin/out/audit/loop.out 2>&1 < /dev/null &
# Stop early: touch sim/twin/out/audit/STOP (checked every minute), or kill the PID in sim/twin/out/audit/loop.pid.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"
O=sim/twin/out/audit
N=${1:-5}
INTERVAL=${2:-3000}
echo $$ > "$O/loop.pid"
trap 'rm -f "$O/loop.pid"' EXIT
for i in $(seq 1 "$N"); do
  end=$(( $(date +%s) + INTERVAL ))
  while [ "$(date +%s)" -lt "$end" ]; do
    [ -f "$O/STOP" ] && { echo "STOP file found, exiting"; exit 0; }
    sleep 60
  done
  echo "== loop run $i / $N at $(date)"
  sim/twin/audit/run_audit.sh
done
echo "== loop finished at $(date)"
