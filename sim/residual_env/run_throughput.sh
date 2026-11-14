#!/bin/bash
# Residual-env throughput with the real MolmoAct2 base (zero-shot weights; same compute as the LoRA) and rendered
# wrist + top cameras, for N parallel envs. Writes out/throughput/N<k>/ (train.log has env_ticks_per_s, episodes_per_h).
#   ./run_throughput.sh "4 8 16" [minutes]
cd ~/residual_env
export OMNI_KIT_ACCEPT_EULA=YES
NS=${1:-"4 8 16"} MIN=${2:-4}
for N in $NS; do
  OUT=~/residual_env/out/throughput/N$N IPC=~/residual_env/ipc_tp$N
  mkdir -p "$OUT" "$IPC"
  HF_HUB_OFFLINE=1 HF_HUB_CACHE=/home/yan/molmobench/hf ~/molmo-cuda-venv/bin/python molmo_batch_worker.py --ipc "$IPC" \
    > "$OUT/worker.log" 2>&1 &
  WPID=$!
  for _ in $(seq 600); do [ -f "$IPC/ready" ] && break; sleep 1; done
  LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 timeout $(( MIN * 60 + 900 )) ~/isaaclab-venv/bin/python demo_train.py \
    --base molmo --num_envs $N --minutes $MIN --eval_every_min 1000 --warmup_steps 1000000 --latency 0.31 --ipc "$IPC" \
    --out "$OUT" > "$OUT/train.log" 2>&1
  touch "$IPC/done"; sleep 3; kill $WPID 2>/dev/null; wait $WPID 2>/dev/null
  echo "N=$N $(grep '^TRAIN' "$OUT/train.log" | tail -1)"
  grep -E '^REQ|WORKER_DONE' "$OUT/worker.log" | tail -2
done
echo THROUGHPUT_DONE
