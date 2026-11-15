#!/usr/bin/env bash
# Detached training run with W&B, the API key read from STDIN (never on disk, argv or in a log):
#   ssh box 'cd ~/Documents/dlsj && sim/twin/rl/launch_wandb.sh RUN [train.py args...]' <<< "$KEY"
# The key goes through a mode-600 FIFO under sim/twin/out/rl that train.py (--wandb_key_fifo) reads once at start;
# the FIFO is removed right after. Logs: sim/twin/out/rl/RUN.log.
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN=$1
shift
IFS= read -r KEY
FIFO="sim/twin/out/rl/.wbkey_${RUN}_$$"
mkdir -p sim/twin/out/rl
rm -f "$FIFO"
mkfifo -m 600 "$FIFO"
NAME=dlsj-twin-rl setsid nohup sim/twin/run.sh sim/twin/rl/train.py --run "$RUN" \
  --wandb_project "${WANDB_PROJECT:-dlsj-twin-dicerl}" --wandb_entity "${WANDB_ENTITY:-sfb5s44zhz-university-of-cambridge}" \
  --wandb_key_fifo "$FIFO" "$@" > "sim/twin/out/rl/${RUN}.log" 2>&1 < /dev/null &
# writer: a forked subshell (the key stays in memory, not in any argv); holds the FIFO open until train.py has
# read it (train.py unlinks the FIFO after reading), at most 10 min
(
  trap '' HUP
  exec 3<>"$FIFO"
  printf '%s\n' "$KEY" >&3
  for _ in $(seq 600); do [ -p "$FIFO" ] || break; sleep 1; done
  rm -f "$FIFO"
) > /dev/null 2>&1 < /dev/null &
disown
echo "launched $RUN"
