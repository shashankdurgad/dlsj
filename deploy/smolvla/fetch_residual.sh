#!/bin/bash
# Fetch a DICE-RL grasp-residual checkpoint (W&B artifact), export it in the twin container on a GB10, copy the
# export back here and, with STAGE=1, into the XPS test checkout's weights/<name>/. Rerun for every new round.
#   deploy/smolvla/fetch_residual.sh round03            # an alias of the artifact: roundNN, latest, vN
#   STAGE=1 deploy/smolvla/fetch_residual.sh latest
#   SRC_DIR=<rounds dir on $HOST> STAGE=1 deploy/smolvla/fetch_residual.sh round05   # a copy already on the GB10
#                                                  (e.g. a Modal fan-out's out/rl/<run>/rounds/): no W&B download
# Env: RUN (artifact prefix, default abl_c_4g_r0), WB_PROJECT (entity/project), HOST (GB10 in ~/.brev/ssh_config,
# default watson), OUT (local exports dir), XPS (ssh target), XPS_DIR (the test checkout on the XPS).
# The W&B key is the `wandb=<key>` line of the repo's .env (git-ignored); it is passed by environment only.
set -euo pipefail
ALIAS=${1:?usage: fetch_residual.sh <alias: roundNN | latest | vN>}
RUN=${RUN:-abl_c_4g_r0}
WB_PROJECT=${WB_PROJECT:-sfb5s44zhz-university-of-cambridge/dlsj-twin-dicerl}
HOST=${HOST:-watson}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$REPO/residual_exports}
XPS=${XPS:-intel@100.77.191.42}
XPS_DIR=${XPS_DIR:-desmond/residual_test}
TWIN_DIR=Documents/dlsj  # the twin checkout on $HOST (sim/twin/run.sh mounts it at /repo)
REMOTE_RUN=sim/twin/out/rl/$RUN/rounds
ssh_h() { perl -e 'alarm shift; exec @ARGV' "$1" ssh -F ~/.brev/ssh_config "$HOST" "${@:2}"; }

DL=$(mktemp -d)  # the download, kept apart from any code
trap 'rm -rf "$DL"' EXIT
if [ -n "${SRC_DIR:-}" ]; then
  echo "1/4 $HOST:$SRC_DIR/residual_$ALIAS.pt" >&2
  perl -e 'alarm 600; exec @ARGV' scp -q -F ~/.brev/ssh_config "$HOST:$SRC_DIR/residual_$ALIAS.pt" "$DL/"
else
KEY=$(grep -m1 '^wandb=' "$REPO/.env" 2>/dev/null | cut -d= -f2- || true)
[ -n "$KEY" ] || { echo "no wandb=<key> line in $REPO/.env" >&2; exit 1; }
echo "1/4 W&B $WB_PROJECT/$RUN-residual:$ALIAS" >&2
WANDB_API_KEY=$KEY WANDB_SILENT=true uv run -q --no-project --with wandb python -I -c '
import json, sys, wandb
a = wandb.Api().artifact(f"{sys.argv[1]}/{sys.argv[2]}-residual:{sys.argv[3]}")
a.download(root=sys.argv[4])
m = a.metadata or {}
print(json.dumps({"version": a.version, "aliases": a.aliases, "round": m.get("round"),
                  "eval/first_attempt_success": m.get("eval/first_attempt_success"),
                  "prior/first_attempt_success": m.get("prior/first_attempt_success"),
                  "eval_dr/first_attempt_success": m.get("eval_dr/first_attempt_success")}), file=sys.stderr)
' "$WB_PROJECT" "$RUN" "$ALIAS" "$DL"
fi
CKPT=$(cd "$DL" && ls *.pt | head -1)
[ -n "$CKPT" ] || { echo "no .pt in the artifact" >&2; exit 1; }
STEM=${CKPT%.pt}
NAME=$([ "$STEM" = residual ] && echo deploy || echo "deploy_$STEM")  # export_residual.py's output dir name
shasum -a 256 "$DL/$CKPT" >&2

echo "2/4 export on $HOST ($CKPT)" >&2
ssh_h 60 "mkdir -p ~/$TWIN_DIR/$REMOTE_RUN"
perl -e 'alarm 600; exec @ARGV' scp -q -F ~/.brev/ssh_config "$DL/$CKPT" "$HOST:$TWIN_DIR/$REMOTE_RUN/$CKPT"
ssh_h 900 "cd ~/$TWIN_DIR && NAME=dlsj-twin-resexport MEMORY=16g sim/twin/run.sh sim/twin/rl/export_residual.py \
  $REMOTE_RUN/$CKPT --onnx 2>&1 | grep -E '^EXPORT|Error|error' | cut -c1-400"

echo "3/4 copy to $OUT/$RUN/$NAME" >&2
mkdir -p "$OUT/$RUN"
ssh_h 300 "cd ~/$TWIN_DIR/$REMOTE_RUN && tar czf - $NAME" | tar xzf - -C "$OUT/$RUN"
cp "$DL/$CKPT" "$OUT/$RUN/$CKPT"
python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); print("export:", {k: d.get(k) for k in (
  "max_abs_diff_vs_torch", "obs_dim", "latent", "top_target", "max_step", "fixed_noise", "finetune", "tip_local")})' \
  "$OUT/$RUN/$NAME/export.json" >&2

if [ "${STAGE:-0}" = 1 ]; then
  DEST=${NAME#deploy_residual_}
  DEST=${DEST#deploy_}
  [ "$DEST" = deploy ] && DEST=final  # the run's last residual.pt
  DEST=${DEST_NAME:-$DEST}  # e.g. DEST_NAME=final for the last eval round
  echo "4/4 stage on $XPS:$XPS_DIR/weights/$DEST" >&2
  perl -e 'alarm 60; exec @ARGV' ssh "$XPS" "mkdir -p $XPS_DIR/weights/$DEST"
  perl -e 'alarm 300; exec @ARGV' scp -q "$OUT/$RUN/$NAME"/residual_actor.* "$OUT/$RUN/$NAME/export.json" \
    "$XPS:$XPS_DIR/weights/$DEST/"
  perl -e 'alarm 60; exec @ARGV' ssh "$XPS" "ls -la $XPS_DIR/weights/$DEST"
else
  echo "4/4 skipped (STAGE=1 copies it to $XPS:$XPS_DIR/weights/)" >&2
fi
echo "$OUT/$RUN/$NAME"
