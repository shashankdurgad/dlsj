#!/bin/bash
# The showcase dashboard (live 3D twin + mission control + presentation) on http://<xps>:${PORT:-8030}/.
# Reads the loop controller at ${ROBOT:-http://localhost:8020}; safe to start/stop any time (never touches the arm).
cd "$(dirname "$0")"
exec python3 server.py --robot "${ROBOT:-http://localhost:8020}" --port "${PORT:-8030}"
