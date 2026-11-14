#!/bin/bash
# usage: run_model.sh <ft|base> [episodes] [chunks]
set -u
M=$1; E=${2:-3}; C=${3:-10}
cd ~/pi05bench
rm -rf ipc_$M out_$M; mkdir -p ipc_$M out_$M
BASE=$(ls -d ~/pi05bench/hf/hub/models--lerobot--pi05_base/snapshots/* 2>/dev/null | head -1)
HF_HOME=~/pi05bench/hf HF_HUB_OFFLINE=1 ~/isaacsim-venv/bin/python policy_worker.py --model $M --ipc ipc_$M --base-path "$BASE" > out_$M/worker.log 2>&1 &
W=$!
until [ -f ipc_$M/ready ]; do kill -0 $W 2>/dev/null || { echo "worker died"; tail -20 out_$M/worker.log; exit 1; }; sleep 3; done
echo "worker ready: $(cat ipc_$M/ready)"
OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 ~/isaacsim-venv/bin/python rollout.py --ipc ipc_$M --out out_$M --episodes $E --chunks $C > out_$M/rollout.log 2>&1
touch ipc_$M/done; wait $W
grep -E "EPISODE_RESULT|ROLLOUT_DONE|Traceback" out_$M/rollout.log | cut -c1-400
echo RUN_FINISHED
