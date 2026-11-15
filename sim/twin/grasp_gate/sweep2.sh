#!/bin/bash
# sweep2.sh <listfile>...: one gate run per line "<tag> <gate args...>", sequentially (run on the GB10 host, in the
# repo). Skips tags that have a result; retries a run up to 3 times when it died of a CUDA out-of-memory error at start
# (busy box). Several streams can share lists: a tag is claimed with an atomic <tag>.lock directory.
cd "$(dirname "$0")/../../.." || exit 1
O=sim/twin/out/grasp_gate
mkdir -p $O
for list in "$@"; do
  while read -r tag rest; do
    [ -z "$tag" ] && continue
    [ -f "$O/$tag/result.json" ] && continue
    mkdir "$O/$tag.lock" 2>/dev/null || continue  # another stream has it
    for try in 1 2 3; do
      [ -f "$O/$tag/result.json" ] && break
      NAME=dlsj-twin-gate-$$ sim/twin/run.sh sim/twin/grasp_gate/gate.py --tag "$tag" $rest > "$O/$tag.log" 2>&1 < /dev/null
      grep -q "^GATE_" "$O/$tag.log" && break
      grep -q "out of memory" "$O/$tag.log" || break
      sleep 90
    done
    grep -h "^GATE_" "$O/$tag.log" | sed "s/^/$tag: /" >> "$O/sweep.log"
    grep -q "^GATE_" "$O/$tag.log" || echo "$tag: NO RESULT" >> "$O/sweep.log"
    rmdir "$O/$tag.lock"
  done < "$list"
done
