#!/bin/bash
# usage: run_molmo.sh [episodes] [chunks] [exec_per_chunk] [cams]
set -u
E=${1:-1}; C=${2:-10}; X=${3:-30}; CAMS=${4:-top,wrist}
cd ~/molmobench
SNAP=$(ls -d ~/molmobench/hf/models--allenai--MolmoAct2-SO100_101/snapshots/* | head -1)
rm -rf ipc out; mkdir -p ipc out
HF_HUB_OFFLINE=1 ~/isaacsim-venv/bin/python molmo_worker.py --ipc ipc --snap "$SNAP" --cams "$CAMS" > out/worker.log 2>&1 &
W=$!
until [ -f ipc/ready ]; do kill -0 $W 2>/dev/null || { echo "worker died"; tail -30 out/worker.log; exit 1; }; sleep 3; done
echo "worker ready: $(cat ipc/ready)"
OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 ~/isaacsim-venv/bin/python rollout.py --ipc ipc --out out \
  --episodes $E --chunks $C --exec $X > out/rollout.log 2>&1
touch ipc/done; wait $W
grep -E "EPISODE_RESULT|ROLLOUT_DONE|Traceback" out/rollout.log | cut -c1-400
F=/home/joe-spark-2/.local/bin/ffmpeg; [ -x $F ] || F=ffmpeg
for d in out/ep*/; do $F -hide_banner -loglevel error -y -framerate 15 -i $d/frames/%05d.jpg -c:v libx264 -pix_fmt yuv420p -crf 22 $d/overview.mp4; done
echo RUN_FINISHED
