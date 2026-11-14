#!/usr/bin/env bash
# MolmoAct2-SO100_101 fine-tune on a GCP GPU VM through Physical AI Studio. Idempotent: rerun after a Spot
# preemption and it skips finished stages and resumes from the newest checkpoint.
#
#   RUN_NAME=<name> METHOD=ai2 DATASET_URI=gs://.../molmoact2-cal2-431-... STAGE=gate bash run.sh  # smoke, timing, export
#   RUN_NAME=<name> METHOD=ai2 DATASET_URI=gs://.../molmoact2-cal2-431-... STAGE=full bash run.sh  # the full run
#
# METHOD: ai2 (method A, Ai2 LoRA split) | studio (method B). DATASET_URI overrides versions.env; DATASET_URI=wait
# (launch_vm.sh without --dataset) sets up Studio + the model, then polls the VM's `dataset` metadata key and that
# snapshot's READY.json for up to DATASET_WAIT_S (default 6 h) before continuing (the VM powers off after that; restarting it resumes the wait).
# Optional env: GPUS (default: all visible), RUN_ARTIFACTS (gs://onflow-30k-pi05-artifacts/runs/$RUN_NAME),
# WORK_DIR (/opt/$RUN_NAME), MAX_UPTIME_S (default 25200 = 7 h), AUTO_SHUTDOWN=1, STEPS (8000).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dataset_override="${DATASET_URI:-}"
source "$here/versions.env"
DATASET_URI="${dataset_override:-$DATASET_URI}"
METHOD="${METHOD:-ai2}"
[[ "$METHOD" == ai2 || "$METHOD" == studio ]] || { echo "METHOD must be ai2 or studio" >&2; exit 2; }
: "${RUN_NAME:?set RUN_NAME}"
STAGE="${STAGE:-gate}"
[[ "$STAGE" == gate || "$STAGE" == full ]] || { echo "STAGE must be gate or full" >&2; exit 2; }
export RUN_ARTIFACTS="${RUN_ARTIFACTS:-gs://onflow-30k-pi05-artifacts/runs/$RUN_NAME}"
work="${WORK_DIR:-/opt/$RUN_NAME}"
MAX_UPTIME_S="${MAX_UPTIME_S:-25200}"
STEPS="${STEPS:-8000}"
deadline=$(python3 -c "import time; print(int(time.time() + $MAX_UPTIME_S - float(open('/proc/uptime').read().split()[0])))")
mkdir -p "$work" && cd "$work"
exec > >(tee -a "$work/run-$STAGE.log") 2>&1

sync_checkpoints() {  # upload each .ckpt once it has been unchanged for 2 min; re-upload when it changes
  local run="$1" f rel marker
  [[ -d "$run/checkpoints" ]] || return 0
  while IFS= read -r f; do
    rel="${f#"$run"/}"
    [[ "$rel" == checkpoints/resume/* && "$rel" != checkpoints/resume/last.ckpt ]] && continue
    marker="$f.uploaded"
    [[ -e "$marker" && ! "$f" -nt "$marker" ]] && continue
    gcloud storage cp "$f" "$RUN_ARTIFACTS/full/$rel" && touch -r "$f" "$marker"
  done < <(find -L "$run/checkpoints" -name '*.ckpt' -type f -mmin +2 | sort)
  [[ -f "$run/progress.json" ]] && gcloud storage cp "$run/progress.json" "$RUN_ARTIFACTS/full/progress.json" || true
}
finish() {
  local code=$?
  trap - EXIT
  [[ "$STAGE" == full && -d runs/full ]] && sync_checkpoints runs/full || true
  printf '{"stage":"%s","exit_code":%s,"finished_at":"%s"}\n' "$STAGE" "$code" "$(date -u +%FT%TZ)" > "status-$STAGE.json"
  gcloud storage cp "run-$STAGE.log" "status-$STAGE.json" "$RUN_ARTIFACTS/" || true
  sync
  [[ "${AUTO_SHUTDOWN:-0}" == 1 ]] && shutdown -h now
  exit "$code"
}
trap finish EXIT
echo "STAGE_STARTED $STAGE $(date -u +%FT%TZ) method=$METHOD steps=$STEPS dataset=$DATASET_URI artifacts=$RUN_ARTIFACTS"

# --- environment: Studio at the pinned commit with the molmoact2 extra -------------------------------------
for _ in $(seq 1 36); do nvidia-smi -L && break; sleep 5; done
GPUS="${GPUS:-$(nvidia-smi -L | grep -c '^GPU ')}"
command -v ffmpeg >/dev/null || { apt-get update -q && apt-get install -yq ffmpeg libgl1; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH" STUDIO_DIR="$work/studio"
[[ -x "$STUDIO_DIR/library/.venv/bin/python" ]] || bash "$here/setup.sh"
py="$STUDIO_DIR/library/.venv/bin/python"
bash "$here/fix_opencv.sh" "$py"  # idempotent; also repairs a venv made before the fix existed

# --- pinned base model: download, check `main` is the pinned revision, then go offline ----------------------
export HF_HOME="$work/hf-home" HF_HUB_DISABLE_TELEMETRY=1
"$py" - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download("$MOLMOACT2_REPO")
if not path.rstrip("/").endswith("$MOLMOACT2_REVISION"):
    raise SystemExit(f"$MOLMOACT2_REPO main moved to {path.rsplit('/', 1)[-1]}; update versions.env deliberately")
print("MODEL_PINNED", path)
PY
export HF_HUB_OFFLINE=1

# --- data: wait until a prepared snapshot is named and uploaded (launch without --dataset) --------------------
metadata() { curl -sf -H 'Metadata-Flavor: Google' \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true; }
waited=0
until [[ "$DATASET_URI" == gs://* ]] && gcloud storage ls "$DATASET_URI/READY.json" >/dev/null 2>&1; do
  (( waited < ${DATASET_WAIT_S:-21600} )) || { echo "no dataset after $waited s" >&2; exit 1; }
  (( waited % 600 == 0 )) && echo "WAITING_FOR_DATASET $(date -u +%FT%TZ) dataset=$DATASET_URI (set metadata dataset=gs://.../<snapshot>)"
  sleep 60; waited=$(( waited + 60 ))
  v="$(metadata dataset)"; [[ -n "$v" ]] && DATASET_URI="$v"
  v="$(metadata method)"; [[ -n "$v" ]] && METHOD="$v"
  v="$(metadata steps)"; [[ -n "$v" ]] && STEPS="$v"
done
[[ "$METHOD" == ai2 || "$METHOD" == studio ]] || { echo "METHOD must be ai2 or studio" >&2; exit 2; }
[[ "$STEPS" =~ ^[0-9]+$ ]] || { echo "STEPS must be a number" >&2; exit 2; }
echo "DATASET_READY $DATASET_URI method=$METHOD steps=$STEPS"
echo "$DATASET_URI method=$METHOD steps=$STEPS" | gcloud storage cp - "$RUN_ARTIFACTS/dataset-$STAGE.txt" || true

# --- data: prepared snapshot from GCS, checksum-verified ----------------------------------------------------
if [[ ! -f data/.verified ]]; then
  mkdir -p data
  gcloud storage cp "$DATASET_URI/READY.json" data/READY.json
  gcloud storage rsync --recursive "$DATASET_URI/" data/
  (cd data && sha256sum --quiet -c SHA256SUMS.txt) && touch data/.verified
fi
dataset="$work/data"

if [[ "$STAGE" == gate ]]; then
  [[ -f runs/smoke.done ]] || { "$py" "$here/train.py" "$dataset" --output runs/smoke --gpus "$GPUS" --method "$METHOD" --smoke \
    && touch runs/smoke.done; }
  if [[ ! -f runs/timing/checkpoints/snapshots/step000200.ckpt ]]; then
    rm -rf runs/timing
    "$py" "$here/train.py" "$dataset" --output runs/timing --gpus "$GPUS" --method "$METHOD" \
      --steps 200 --snapshot-every 200
  fi
  "$py" - "$STEPS" <<'PY'
import json, sys
p = json.load(open("runs/timing/progress.json"))
p["projected_hours_for_steps"] = {"steps": int(sys.argv[1]), "hours": round(int(sys.argv[1]) * p["median_s_per_step"] / 3600, 2)}
json.dump(p, open("runs/timing/timing.json", "w"), indent=2)
print("TIMING", json.dumps(p))
PY
  if [[ ! -f runs/openvino/.done ]]; then
    rm -rf runs/openvino
    "$STUDIO_DIR/library/.venv/bin/physicalai" export --policy physicalai.policies.MolmoAct2 \
      --ckpt_path runs/timing/checkpoints/snapshots/step000200.ckpt --backend openvino --output_dir runs/openvino
    touch runs/openvino/.done
  fi
  du -sh runs/openvino
  gcloud storage cp runs/timing/timing.json runs/timing/training-config.json "$RUN_ARTIFACTS/gate/"
  gcloud storage rsync --recursive runs/openvino "$RUN_ARTIFACTS/gate/openvino-step200"
  echo "GATE_READY: load $RUN_ARTIFACTS/gate/openvino-step200 on the XPS (GPU) and time it before STAGE=full"
  exit 0
fi

# --- STAGE=full: resume-safe ---------------------------------------------------------------------------------
run=runs/full
resume=()
if [[ ! -f "$run/checkpoints/resume/last.ckpt" ]] && gcloud storage ls "$RUN_ARTIFACTS/full/checkpoints/resume/last.ckpt" >/dev/null 2>&1; then
  mkdir -p "$run/checkpoints/resume"
  gcloud storage cp "$RUN_ARTIFACTS/full/checkpoints/resume/last.ckpt" "$run/checkpoints/resume/last.ckpt"
  echo "RESTORED resume checkpoint from GCS"
fi
if [[ -f "$run/checkpoints/resume/last.ckpt" ]]; then
  resume=(--resume "$run/checkpoints/resume/last.ckpt")
else
  rm -rf "$run"  # preempted before the first resume checkpoint: start clean (train.py refuses an existing output)
fi
( while sleep 600; do sync_checkpoints "$run"; done ) &
syncer=$!
remaining=$(( deadline - $(date +%s) - 900 ))  # leave 15 min for the final checkpoint and upload
(( remaining > 900 )) || { echo "not enough VM time left ($remaining s)"; exit 1; }
set +e
"$py" "$here/train.py" "$dataset" --output "$run" --gpus "$GPUS" --method "$METHOD" --steps "$STEPS" \
  --deadline-seconds "$remaining" "${resume[@]}"
code=$?
set -e
kill "$syncer" 2>/dev/null || true
sleep 125 && sync_checkpoints "$run"   # let the last files age past the 2-minute rule
echo "TRAINING_FINISHED exit=$code $(date -u +%FT%TZ)"
exit "$code"
