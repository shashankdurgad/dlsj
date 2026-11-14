#!/usr/bin/env bash
# Create the 4-GPU Spot VM for the MolmoAct2 run. Prints what it would do unless --create is given.
#
#   bash launch_vm.sh <run name> --method ai2|studio [--dataset gs://.../datasets/<snapshot>] \
#        [--steps 8000] [--max-uptime-s 25200] [--create]
# Without --dataset the VM sets up Studio + the model and then waits (up to 6 h) for you to name the snapshot:
#   gcloud compute instances add-metadata <name> --zone=$ZONE --metadata=dataset=gs://.../datasets/<snapshot>
# (method= and steps= metadata are re-read while it waits, so they can be changed then too.)
# Without --create it only prints the bundle, metadata and gcloud command (dry run).
#
#   bash launch_vm.sh m2-a-cal2-431-20261114 --method ai2 \
#        --dataset gs://onflow-30k-pi05-artifacts/datasets/molmoact2-cal2-431-3stage-20261114 --steps 12000
#
# The VM runs run.sh from its startup script on every boot (so a preempted Spot VM resumes when restarted),
# with STAGE from instance metadata (default "gate"), then powers itself off. After the gate checkpoint
# passes the export check:
#   gcloud compute instances add-metadata <name> --zone=$ZONE --metadata=stage=full,max-uptime-s=<s>
#   gcloud compute instances start <name> --zone=$ZONE
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
usage="usage: launch_vm.sh <run name> --method ai2|studio [--dataset gs://...] [--steps N] [--max-uptime-s S] [--create]"
name="${1:?$usage}"; shift
method="" dataset=wait steps=8000 uptime=25200 create=""
while (( $# )); do
  case "$1" in
    --method) method="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --max-uptime-s) uptime="$2"; shift 2 ;;
    --create) create=--create; shift ;;
    *) echo "$usage" >&2; exit 2 ;;
  esac
done
[[ "$method" == ai2 || "$method" == studio ]] || { echo "$usage" >&2; exit 2; }
[[ "$dataset" == gs://* || "$dataset" == wait ]] || { echo "$usage" >&2; exit 2; }
[[ "$steps" =~ ^[0-9]+$ && "$uptime" =~ ^[0-9]+$ ]] || { echo "$usage" >&2; exit 2; }
PROJECT="${PROJECT:-onflow-30k}" ZONE="${ZONE:-europe-west4-b}" MACHINE="${MACHINE:-g4-standard-192}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2404-nvidia-580}"  # same image as the pi0.5/ACT VMs
artifacts="gs://onflow-30k-pi05-artifacts/runs/$name"
tmp="$(mktemp -d)"
bundle="$tmp/recipe.tar.gz"
tar -C "$here/.." --exclude='molmoact2/.studio' --exclude='molmoact2/runs' --exclude='__pycache__' \
  -czf "$bundle" molmoact2

startup=$(cat <<EOF
#!/bin/bash
set -euo pipefail
md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/\$1" || echo "\$2"; }
export RUN_NAME='$name' RUN_ARTIFACTS='$artifacts' AUTO_SHUTDOWN=1 HOME=/root
export STAGE="\$(md stage gate)" MAX_UPTIME_S="\$(md max-uptime-s $uptime)"
export METHOD="\$(md method $method)" DATASET_URI="\$(md dataset $dataset)" STEPS="\$(md steps $steps)"
mkdir -p /opt/\$RUN_NAME && cd /opt/\$RUN_NAME
gcloud storage cp "\$RUN_ARTIFACTS/recipe.tar.gz" . && tar xzf recipe.tar.gz
nohup bash molmoact2/run.sh > /opt/\$RUN_NAME/startup-\$STAGE.out 2>&1 &
EOF
)
printf '%s\n' "$startup" > "$tmp/startup.sh"
ZONES="${ZONES:-$ZONE}"  # space-separated zones to try in order (RTX PRO 6000 Spot stock-outs are common)
cmd=(gcloud compute instances create "$name" --project="$PROJECT" --machine-type="$MACHINE"
     --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE
     --image-family="$IMAGE_FAMILY" --image-project=deeplearning-platform-release --boot-disk-size=500GB
     --scopes=cloud-platform --metadata=stage=gate,max-uptime-s=$uptime,method=$method,steps=$steps
     --metadata-from-file=startup-script="$tmp/startup.sh")

echo "bundle: $bundle ($(du -h "$bundle" | cut -f1)) -> $artifacts/recipe.tar.gz"
printf 'vm: '; printf '%q ' "${cmd[@]}"; echo "--zone=<first of: $ZONES>"
if [[ "$create" != --create ]]; then
  [[ "$dataset" == wait ]] || gcloud storage ls "$dataset/READY.json" >/dev/null 2>&1 || echo "NOTE: $dataset/READY.json not uploaded yet"
  echo "--- startup script ---"; echo "$startup"
  echo "(dry run; pass --create to upload the bundle and create the VM)"
  exit 0
fi
if [[ "${REUSE_BUNDLE:-0}" != 1 ]] && gcloud storage ls "$artifacts/recipe.tar.gz" >/dev/null 2>&1; then
  echo "$artifacts/recipe.tar.gz exists; use a new run name (or REUSE_BUNDLE=1 to retry this run elsewhere)" >&2; exit 1
fi
[[ "$dataset" == wait ]] || gcloud storage ls "$dataset/READY.json" >/dev/null || { echo "dataset not uploaded: $dataset" >&2; exit 1; }
[[ "${REUSE_BUNDLE:-0}" == 1 ]] || gcloud storage cp "$bundle" "$artifacts/recipe.tar.gz"
for zone in $ZONES; do
  echo "trying $zone"
  "${cmd[@]}" --zone="$zone" --async || continue
  status="" waited=0
  while (( waited < ${PLACE_S:-300} )); do  # a pending create can sit in STAGING for many minutes
    sleep 15; waited=$(( waited + 15 ))
    status="$(gcloud compute instances describe "$name" --project="$PROJECT" --zone="$zone" --format='value(status)' 2>/dev/null || echo GONE)"
    [[ "$status" == STAGING || "$status" == PROVISIONING ]] || break
  done
  if [[ "$status" == RUNNING ]]; then
    sleep "${SURVIVE_S:-120}"  # Spot capacity can be reclaimed seconds after a successful create
    status="$(gcloud compute instances describe "$name" --project="$PROJECT" --zone="$zone" --format='value(status)' 2>/dev/null || echo GONE)"
  fi
  if [[ "$status" == RUNNING ]]; then
    echo "created $name in $zone (still RUNNING after ${SURVIVE_S:-120}s); follow with:"
    echo "  gcloud compute ssh $name --zone=$zone -- sudo tail -f /opt/$name/run-gate.log"
    exit 0
  fi
  echo "$zone: $status (no capacity, still pending, or preempted); deleting and moving on"
  [[ "$status" == GONE ]] || gcloud compute instances delete "$name" --project="$PROJECT" --zone="$zone" --quiet || true
done
echo "no capacity in: $ZONES (bundle stays at $artifacts/recipe.tar.gz; remove it before retrying with this name)" >&2
exit 1
