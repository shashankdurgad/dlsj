#!/usr/bin/env bash
# Run ON THE XPS. Keeps an SSH tunnel so Studio (Docker network docker_default) reaches the GCP trainer at
# http://${TUNNEL_BIND}:${TRAINER_PORT}. The trainer itself only listens on the VM's loopback; traffic is SSH-encrypted.
#   ./xps_tunnel.sh setup        create the tunnel key (prints the .pub to copy next to create_vm.sh)
#   ./xps_tunnel.sh start <IP>   install + start a systemd user service that keeps the tunnel up
#   ./xps_tunnel.sh status|stop
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env
KEY="$HOME/.ssh/dlsj_trainer_tunnel"; UNIT="$HOME/.config/systemd/user/dlsj-trainer-tunnel.service"
case "${1:-}" in
  setup)
    [ -f "$KEY" ] || ssh-keygen -t ed25519 -N "" -C "xps-trainer-tunnel" -f "$KEY"
    cp "$KEY.pub" ./xps_tunnel_key.pub; echo "Public key (also written to ./xps_tunnel_key.pub):"; cat "$KEY.pub" ;;
  start)
    IP="${2:?usage: $0 start <VM external IP>}"
    ip -4 addr | grep -q "inet ${TUNNEL_BIND}/" || { echo "${TUNNEL_BIND} is not on this host (is Studio's docker_default network up?)"; exit 1; }
    mkdir -p "$(dirname "$UNIT")"
    cat > "$UNIT" <<UNITEOF
[Unit]
Description=SSH tunnel to the dlsj GCP Physical AI trainer
After=network-online.target
[Service]
ExecStart=/usr/bin/ssh -N -i $KEY -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new -L ${TUNNEL_BIND}:${TRAINER_PORT}:127.0.0.1:${TRAINER_PORT} ${VM_SSH_USER}@${IP}
Restart=always
RestartSec=5
[Install]
WantedBy=default.target
UNITEOF
    systemctl --user daemon-reload && systemctl --user enable --now dlsj-trainer-tunnel.service
    sleep 3; curl -fsS "http://${TUNNEL_BIND}:${TRAINER_PORT}/health" && echo && echo "Trainer reachable. In Studio: Settings > Training Targets > Self Managed Remote Trainer, URL http://${TUNNEL_BIND}:${TRAINER_PORT}" ;;
  status) systemctl --user status dlsj-trainer-tunnel.service --no-pager; curl -fsS "http://${TUNNEL_BIND}:${TRAINER_PORT}/health"; echo ;;
  stop)   systemctl --user disable --now dlsj-trainer-tunnel.service ;;
  *) sed -n '2,7p' "$0"; exit 2 ;;
esac
