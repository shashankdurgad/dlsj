#!/bin/bash
# usage: run_smol.sh [episodes] [chunks]   (on the Spark as yan)
set -u
E=${1:-5}; C=${2:-24}
cd ~/smolbench
rm -rf ipc out; mkdir -p ipc out
HF_HOME=~/smolbench/hf ~/isaacsim-venv/bin/python smol_worker.py --ckpt ckpt/checkpoints/snapshots/step020000.ckpt --ipc ipc > out/worker.log 2>&1 &
W=$!
until [ -f ipc/ready ]; do kill -0 $W 2>/dev/null || { echo "worker died"; tail -30 out/worker.log; exit 1; }; sleep 3; done
echo "worker ready: $(cat ipc/ready)"; grep LOAD out/worker.log
OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 ~/isaacsim-venv/bin/python smol_rollout.py --ipc ipc --out out --episodes $E --chunks $C > out/rollout.log 2>&1
touch ipc/done; wait $W
for d in out/ep*; do
  ffmpeg=/home/joe-spark-2/.local/bin/ffmpeg
  $ffmpeg -hide_banner -loglevel error -y -framerate 15 -i $d/frames/%05d.jpg -c:v libx264 -pix_fmt yuv420p -crf 23 $d/overview.mp4
  $ffmpeg -hide_banner -loglevel error -y -framerate 15 -i $d/wrist/%05d.jpg -c:v libx264 -pix_fmt yuv420p -crf 23 $d/wrist.mp4
done
grep -E "EPISODE_RESULT|ROLLOUT_DONE|Traceback" out/rollout.log | cut -c1-500
echo RUN_FINISHED
