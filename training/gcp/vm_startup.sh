#!/usr/bin/env bash
# GCE startup script (runs as root on every boot of the trainer VM). Installs Docker + the NVIDIA container toolkit
# if the image lacks them, then starts the Physical AI trainer container bound to 127.0.0.1 only (it has no auth).
set -euo pipefail
md() { curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
IMAGE="$(md trainer-image)"; PORT="$(md trainer-port || echo 8001)"

until nvidia-smi >/dev/null 2>&1; do echo "waiting for NVIDIA driver"; sleep 10; done
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
if ! docker info 2>/dev/null | grep -q nvidia; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -q && apt-get install -yq nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
fi

docker volume create physicalai-trainer-data >/dev/null
docker pull "$IMAGE"
docker rm -f physicalai-trainer-cuda >/dev/null 2>&1 || true
# Same settings as Studio's docker-compose.trainer.yaml (cuda profile). HF token: Studio sends it per job.
docker run -d --name physicalai-trainer-cuda --restart unless-stopped \
  --gpus all --ipc host --read-only --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  -v physicalai-trainer-data:/var/lib/physicalai-trainer \
  -p "127.0.0.1:${PORT}:8001" --stop-timeout 35 "$IMAGE"
for _ in $(seq 1 60); do curl -fsS "http://127.0.0.1:${PORT}/health" && break; sleep 5; done
echo "physicalai trainer up on 127.0.0.1:${PORT}"
