#!/usr/bin/env bash
# Run build_ov_debugcaps.sh politely on a shared XPS: nice 19, ionice idle, -j6, and SIGSTOP its whole process
# tree (ninja puts every job in its own process group, so a group signal is not enough) while the 1-min load average is above MAX_LOAD or available RAM is below MIN_AVAIL_GB
# (resume below RESUME_LOAD / above RESUME_AVAIL_GB). Detach it:
#   setsid nohup bash build_governor.sh > ~/inference/logs/pi05/kernel/build_governor.log 2>&1 < /dev/null &
set -uo pipefail
K=$(cd "$(dirname "$0")" && pwd)
LOG=${LOG:-$HOME/inference/logs/pi05/kernel/build.log}
MAX_LOAD=${MAX_LOAD:-12} RESUME_LOAD=${RESUME_LOAD:-9}
MIN_AVAIL_GB=${MIN_AVAIL_GB:-15} RESUME_AVAIL_GB=${RESUME_AVAIL_GB:-18}
export JOBS=${JOBS:-6}

setsid nice -n 19 ionice -c3 bash "$K/build_ov_debugcaps.sh" > "$LOG" 2>&1 < /dev/null &
PGID=$!  # setsid execs in place, so this is the session/group leader
echo "$(date +%T) build started, pgid $PGID, JOBS=$JOBS"
desc() { local p; for p in $(pgrep -P "$1"); do echo "$p"; desc "$p"; done; }
tree_signal() { kill "-$1" "$PGID" $(desc "$PGID") 2>/dev/null; }
stopped=0
while kill -0 "$PGID" 2>/dev/null; do
  load=$(cut -d' ' -f1 /proc/loadavg)
  avail=$(free -g | awk '/^Mem:/{print $7}')
  hi=$(awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN{print (l > m)}')
  lo=$(awk -v l="$load" -v m="$RESUME_LOAD" 'BEGIN{print (l < m)}')
  if [[ $stopped == 0 && ( $hi == 1 || $avail -lt $MIN_AVAIL_GB ) ]]; then
    tree_signal STOP; stopped=1; echo "$(date +%T) STOP  load=$load avail=${avail}G"
  elif [[ $stopped == 1 && $lo == 1 && $avail -ge $RESUME_AVAIL_GB ]]; then
    tree_signal CONT; stopped=0; echo "$(date +%T) CONT  load=$load avail=${avail}G"
  elif [[ $stopped == 1 ]]; then
    tree_signal STOP  # catch children spawned in the instant before the stop
  fi
  sleep 20
done
wait "$PGID"; rc=$?
echo "$(date +%T) build exited rc=$rc"; tail -3 "$LOG"
