#!/usr/bin/env bash
# Create the GPU trainer VM (run on any machine with gcloud). COSTS MONEY until delete_vm.sh / stop_vm.sh.
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env
test -f "$XPS_PUBKEY_FILE" || { echo "Missing $XPS_PUBKEY_FILE: run ./xps_tunnel.sh setup on the XPS first and copy the .pub here"; exit 1; }
args=(--project "$GCP_PROJECT" --zone "$GCP_ZONE" --machine-type "$MACHINE_TYPE"
  --image-project deeplearning-platform-release --image-family common-cu129-ubuntu-2204-nvidia-580
  --boot-disk-size "${BOOT_DISK_GB}GB" --boot-disk-type pd-balanced
  --maintenance-policy TERMINATE
  --metadata "install-nvidia-driver=True,trainer-image=${TRAINER_IMAGE},trainer-port=${TRAINER_PORT},ssh-keys=${VM_SSH_USER}:$(cat "$XPS_PUBKEY_FILE")"
  --metadata-from-file startup-script=./vm_startup.sh
  --scopes cloud-platform)
[ -n "${ACCELERATOR:-}" ] && args+=(--accelerator "$ACCELERATOR")
[ "${SPOT:-false}" = true ] && args+=(--provisioning-model SPOT --instance-termination-action STOP)
gcloud compute instances create "$VM_NAME" "${args[@]}"
IP=$(gcloud compute instances describe "$VM_NAME" --project "$GCP_PROJECT" --zone "$GCP_ZONE" \
      --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
echo "VM external IP: $IP   (put it in the XPS tunnel: ./xps_tunnel.sh start $IP)"
echo "Watch the trainer come up: gcloud compute ssh $VM_NAME --project $GCP_PROJECT --zone $GCP_ZONE -- sudo journalctl -u google-startup-scripts -f"
