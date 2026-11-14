# Remote training on GCP (Physical AI Studio "Self Managed Remote Trainer")

Studio on the XPS submits training jobs to Studio's own trainer container (`physicalai-trainer-cuda`) on a GCP GPU VM.
Studio uploads the dataset snapshot, streams progress, and downloads the trained model (with its OpenVINO export) when done.
This keeps training inside Studio (judged: "VLA & Physical AI Studio integration").

```
XPS: Studio container ──http──► 172.18.0.1:8001 ══ SSH tunnel (systemd user service) ══► GCP VM: 127.0.0.1:8001 trainer (GPU)
```

The trainer has **no authentication**. It only listens on the VM's loopback, and Studio reaches it through an SSH tunnel bound to the Studio
Docker network's gateway on the XPS. Never open port 8001 in a GCP firewall.

## Steps
1. **Config** (any machine with `gcloud`): `cp config.env.example config.env` and set `GCP_PROJECT`, `GCP_ZONE`, `MACHINE_TYPE`.
   Check GPU quota first: `gcloud compute regions describe us-central1 --project $GCP_PROJECT | grep -B1 -A1 A100`
   (new projects often have 0 GPU quota; request it under IAM & Admin > Quotas).
2. **Tunnel key** (on the XPS, in this folder): `./xps_tunnel.sh setup`, then copy `xps_tunnel_key.pub` next to `create_vm.sh`.
3. **Create the VM** (costs money from here): `./create_vm.sh`. Boot plus pulling the trainer image takes about 5–10 minutes. Follow it with the
   `journalctl` command the script prints until you see `physicalai trainer up`.
4. **Tunnel** (on the XPS): `./xps_tunnel.sh start <VM external IP>` should print the trainer's `/health` JSON.
5. **Register in Studio**: Settings > Training Targets > **Self Managed Remote Trainer**, URL `http://172.18.0.1:8001`.
   Studio warns about a non-loopback `http://` URL. The hop is the local Docker bridge and the WAN part is SSH-encrypted.
6. **Train**: Models > new model > pick the dataset, π0.5, and this trainer. Set `tokenizer_max_length` 64 (see `training/pi05`).
7. **Stop billing**: `./stop_vm.sh stop` (keeps the disk; `start` resumes) or `./stop_vm.sh delete`. Then run `./xps_tunnel.sh stop` on the XPS.

## Choices
- Image: `ghcr.io/open-edge-platform/physicalai-trainer-cuda:0.3.0-dev-8f132416`, which matches the XPS Studio commit `8f13241`.
  Bump both together.
- VM image: Deep Learning VM `common-cu129-ubuntu-2204-nvidia-580` (NVIDIA driver preinstalled); the startup script adds
  Docker / NVIDIA container toolkit if missing and (re)starts the trainer on every boot.
- π0.5 full fine-tune needs ≥ 40–80 GB per GPU: `a2-ultragpu-1g` (1× A100 80 GB) or `a2-highgpu-4g` (4× A100 40 GB).
  An L4 (24 GB) only suits smoke tests and smaller policies. `SPOT=true` is much cheaper but can be preempted.
- The dataset upload goes over the XPS's uplink (`duplo_teleop_rgb` is ~0.8 GB), so a slow venue Wi-Fi uplink adds minutes.

⚠️ Untested end to end: written against Studio's `docker-compose.trainer.yaml` and `backend/docs/remote-trainer.md` at `8f13241`.
