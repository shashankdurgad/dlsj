#!/usr/bin/env bash
# Create the 1-GPU Spot VM for the SmolVLA run. Prints what it would do unless --create is given.
#
#   bash launch_vm.sh smolvla-517-2cam-20261114            # dry run: show the bundle + gcloud commands
#   bash launch_vm.sh smolvla-517-2cam-20261114 --create   # upload the recipe bundle and create the VM
#
# The VM runs run.sh from its startup script on every boot (so a preempted Spot VM resumes when restarted),
# with STAGE from instance metadata (default "gate"), then powers itself off. After the gate checkpoint
# passes the export check:
#   gcloud compute instances add-metadata <name> --zone=$ZONE --metadata=stage=full,max-uptime-s=21600
#   gcloud compute instances start <name> --zone=$ZONE
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
name="${1:?usage: launch_vm.sh <run name> [--create]}"
create="${2:-}"
PROJECT="${PROJECT:-onflow-30k}" ZONE="${ZONE:-europe-west4-b}" MACHINE="${MACHINE:-g4-standard-48}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2404-nvidia-580}"  # same image as the pi0.5/ACT VMs
artifacts="gs://onflow-30k-pi05-artifacts/runs/$name"
tmp="$(mktemp -d)"
bundle="$tmp/recipe.tar.gz"
tar -C "$here/.." --exclude='smolvla/.venv' --exclude='smolvla/runs' --exclude='__pycache__' \
  -czf "$bundle" smolvla

startup=$(cat <<EOF
#!/bin/bash
set -euo pipefail
md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/\$1" || echo "\$2"; }
export RUN_NAME='$name' RUN_ARTIFACTS='$artifacts' AUTO_SHUTDOWN=1 HOME=/root
export STAGE="\$(md stage gate)" MAX_UPTIME_S="\$(md max-uptime-s 21600)"
mkdir -p /opt/\$RUN_NAME && cd /opt/\$RUN_NAME
gcloud storage cp "\$RUN_ARTIFACTS/recipe.tar.gz" . && tar xzf recipe.tar.gz
nohup bash smolvla/run.sh > /opt/\$RUN_NAME/startup-\$STAGE.out 2>&1 &
EOF
)
printf '%s\n' "$startup" > "$tmp/startup.sh"
cmd=(gcloud compute instances create "$name" --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE"
     --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE
     --image-family="$IMAGE_FAMILY" --image-project=deeplearning-platform-release --boot-disk-size=200GB
     --scopes=cloud-platform --metadata=stage=gate,max-uptime-s=21600
     --metadata-from-file=startup-script="$tmp/startup.sh")

echo "bundle: $bundle ($(du -h "$bundle" | cut -f1)) -> $artifacts/recipe.tar.gz"
printf 'vm: '; printf '%q ' "${cmd[@]}"; echo
if [[ "$create" != --create ]]; then
  echo "--- startup script ---"; echo "$startup"
  echo "(dry run; pass --create to upload the bundle and create the VM)"
  exit 0
fi
if gcloud storage ls "$artifacts/recipe.tar.gz" >/dev/null 2>&1; then
  echo "$artifacts/recipe.tar.gz exists; use a new run name" >&2; exit 1
fi
gcloud storage cp "$bundle" "$artifacts/recipe.tar.gz"
"${cmd[@]}"
echo "created $name; follow with: gcloud compute ssh $name --zone=$ZONE -- sudo tail -f /opt/$name/run-gate.log"
