#!/usr/bin/env bash
# Stop (disk kept, GPU billing stops; the trainer restarts on the next start) or delete the VM.
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env
case "${1:-stop}" in
  stop)   gcloud compute instances stop "$VM_NAME" --project "$GCP_PROJECT" --zone "$GCP_ZONE" ;;
  start)  gcloud compute instances start "$VM_NAME" --project "$GCP_PROJECT" --zone "$GCP_ZONE" ;;
  delete) gcloud compute instances delete "$VM_NAME" --project "$GCP_PROJECT" --zone "$GCP_ZONE" ;;
  *) echo "usage: $0 [stop|start|delete]"; exit 2 ;;
esac
