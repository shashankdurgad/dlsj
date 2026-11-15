#!/bin/bash
# sweep.sh <listfile>: one gate run per line "<tag> <gate args...>", sequentially (run on the GB10 host, in the repo)
cd "$(dirname "$0")/../../.." || exit 1
O=sim/twin/out/grasp_gate
mkdir -p $O
while read -r tag rest; do
  [ -z "$tag" ] && continue
  [ -f "$O/$tag/result.json" ] && continue
  NAME=dlsj-twin-gate sim/twin/run.sh sim/twin/grasp_gate/gate.py --tag "$tag" $rest > "$O/$tag.log" 2>&1 < /dev/null
  grep -h "^GATE_" "$O/$tag.log" | sed "s/^/$tag: /" >> "$O/sweep.log"
  grep -q "^GATE_" "$O/$tag.log" || echo "$tag: NO RESULT" >> "$O/sweep.log"
done < "$1"
