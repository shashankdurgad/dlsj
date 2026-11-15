#!/usr/bin/env bash
# One realism audit of the CURRENT twin (sim/twin/audit; outputs in sim/twin/out/audit/runs/<stamp>/):
#   sim/twin/audit/run_audit.sh            [STAGES="state layouts render policy perceptual control report"]
# 0 state      git commit / dirty files / sha256 of every twin source file -> state.json (attribution)
# 1 layouts    the scene agent's layout detector on the audit episodes' first top frames (plain python)
# 2 render     the audit set in the twin at the recorded joint states (Isaac, via sim/twin/run.sh)
# 3 policy     MolmoAct2 real-vs-sim equivalence (zero-shot, and the fine-tune when its checkpoint is present)
# 4 perceptual DINOv2 KID / paired distances, real-vs-sim classifiers, matched-region colour + sharpness
# 5 control    positive control: the twin's closed-loop picks (RL harness logs) vs the real first-close outcomes
# 6 report     report.md + images + trend table over all runs
# Containers are throwaway (--rm --init) and named dlsj-twin-audit*; anything still running is removed on exit.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"
A=sim/twin/audit
O=sim/twin/out/audit
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
RUN=$O/runs/$STAMP
STAGES=${STAGES:-"state layouts render policy perceptual control report"}
mkdir -p "$RUN"
exec 9>"$O/.lock"
flock -n 9 || { echo "another audit is running"; exit 1; }
exec > >(tee -a "$RUN/audit.log") 2>&1
cleanup() { docker ps -aq --filter name=dlsj-twin-audit | xargs -r docker rm -f >/dev/null 2>&1 || true; }
trap cleanup EXIT
has() { [[ " $STAGES " == *" $1 "* ]]; }
py() { # plain python in the twin image (GPU unless GPU=0)
  local gpu=()
  [ "${GPU:-1}" = 1 ] && gpu=(--device nvidia.com/gpu=all)
  timeout -k 30 "${T:-3000}" docker run --rm --init --name "dlsj-twin-audit-x-$(date +%s)" "${gpu[@]}" \
    --network host --ipc host -v "$REPO:/repo" -w /repo -e PYTHONPATH=/repo -e PYTHONUNBUFFERED=1 \
    --entrypoint /opt/isaaclab-venv/bin/python dlsj/twin:latest "$@"
}
snapshot() { # sha256 of every twin source file (tracked + untracked, not ignored), minus the audit's own code
  git ls-files -co --exclude-standard -- sim/twin sim/residual_env sim/realism deploy/molmoact2 \
    | grep -v -e '^sim/twin/audit/' -e '__pycache__' | sort | while read -r f; do
      [ -f "$f" ] && sha256sum "$f"
    done
}
echo "== audit $STAMP on $(hostname) | stages: $STAGES"
if has state; then
  snapshot > "$RUN/files.sha256"
  python3 - "$RUN" "$STAMP" <<'PY'
import hashlib, json, subprocess, sys, socket, time
run, stamp = sys.argv[1:3]
sh = lambda *c: subprocess.run(c, capture_output=True, text=True).stdout.strip()
txt = open(f"{run}/files.sha256").read()
json.dump({"stamp": stamp, "host": socket.gethostname(), "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"), "commit": sh("git", "rev-parse", "HEAD"),
           "dirty": sh("git", "status", "--short").splitlines(),
           "twin_hash": hashlib.sha256(txt.encode()).hexdigest()[:16], "n_files": len(txt.splitlines()),
           "audit_hash": hashlib.sha256(b"".join(open(p, "rb").read() for p in sorted(
               __import__("glob").glob("sim/twin/audit/*.py") + __import__("glob").glob("sim/twin/audit/*.sh"))))
           .hexdigest()[:8]},
          open(f"{run}/state.json", "w"), indent=1)
PY
  cat "$RUN/state.json" | head -8
fi
[ -f "$O/set/set.json" ] || GPU=0 py $A/make_set.py || echo "STAGE_FAILED make_set"
if has layouts; then GPU=0 T=900 py $A/layouts.py --run "$RUN" || echo "STAGE_FAILED layouts"; fi
if has render; then
  # batches of 9 samples (3 episodes), each its own Isaac process with a timeout: a PhysX hang only costs one batch,
  # retried twice, the last time with rigid fins (gripper.finray.bend.hinges=0; recorded in render_meta)
  rm -rf "$RUN/parts"; : > "$RUN/render.log"
  NS=$(python3 -c "import json; print(len(json.load(open('$O/set/set.json'))['samples']))")
  for ((s = 0; s < NS; s += ${BATCH:-9})); do
    e=$((s + ${BATCH:-9})); [ $e -gt "$NS" ] && e=$NS
    for try in 1 2 3; do
      PS=""; [ $try = 3 ] && PS="gripper.finray.bend.hinges=0"
      NAME=dlsj-twin-audit timeout -k 30 ${RENDER_T:-900} sim/twin/run.sh $A/render_set.py --run "$RUN" --start $s \
        --stop $e --physics-set "$PS" ${RENDER_ARGS:-} > "$RUN/render_$s.log" 2>&1
      cleanup
      if grep -q "^RENDER_OK" "$RUN/render_$s.log"; then
        echo "batch $s-$e ok (try $try${PS:+, rigid fins})"; break
      fi
      echo "batch $s-$e failed (try $try): $(grep -E 'RENDER_FAILED|Traceback|Error:' "$RUN/render_$s.log" | tail -1)"
    done
    grep -E "RENDER_|\[audit\] sample" "$RUN/render_$s.log" >> "$RUN/render.log"
  done
  if GPU=0 T=600 py $A/merge_render.py --run "$RUN"; then echo "RENDER_OK merged" >> "$RUN/render.log"; fi
  grep -E "RENDER_FAILED|MERGE|\[audit\] sample" "$RUN/render.log" | tail -5
  # a render with black or failed frames is NOT audited (run 20261115-000228 fed all-black frames to every stage)
  grep -q "^RENDER_OK merged" "$RUN/render.log" || { echo "STAGE_FAILED render: no valid sim.npz, skipping policy + perceptual";
    [ -f "$RUN/sim.npz" ] && mv "$RUN/sim.npz" "$RUN/sim_invalid.npz"; }
  snapshot > "$RUN/files_after_render.sha256"
  cmp -s "$RUN/files.sha256" "$RUN/files_after_render.sha256" || echo "NOTE twin files changed during the render"
fi
if has policy && [ -f "$RUN/sim.npz" ]; then
  py $A/policy_eval.py --run "$RUN" --model zero ${POLICY_ARGS:-} 2> "$RUN/policy_zero.err" || echo "STAGE_FAILED policy zero"
  if ls sim/twin/data/models/molmoact2_ft_8k/*.ckpt >/dev/null 2>&1; then
    py $A/policy_eval.py --run "$RUN" --model ft ${POLICY_ARGS:-} 2> "$RUN/policy_ft.err" || echo "STAGE_FAILED policy ft"
  fi
fi
if has perceptual && [ -f "$RUN/sim.npz" ]; then py $A/perceptual.py --run "$RUN" || echo "STAGE_FAILED perceptual"; fi
if has control; then GPU=0 T=600 py $A/control.py --run "$RUN" || echo "STAGE_FAILED control"; fi
if has report; then GPU=0 T=900 py $A/report.py --run "$RUN" || echo "STAGE_FAILED report"; fi
echo "== audit $STAMP done"
