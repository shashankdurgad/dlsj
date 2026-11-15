"""Put the arms' current calibration into the Physical AI Studio robot records (SO101 Follower / SO101 Leader).

    python deploy/studio/sync_calibration.py [--repo ~/inference/dlsj-smolvla] [--backup FILE] [--dry-run]

The calibration is the LeRobot SO-101 format (per joint: id, drive_mode, homing_offset, range_min, range_max), the
same format Studio stores in each robot's `payload.calibration`. Source: `calibration/robots/so_follower/follower.json`
and `calibration/teleoperators/so_leader/leader.json` (the files the recorder, the loop and the training data use).
Robots are matched by type and USB serial number. Studio's update endpoint only writes its database: nothing
connects to the arms, so this is safe while the loop runs. The previous records are saved to --backup first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

STUDIO = "http://localhost:7860"
PROJECT = "0fad5636-5f89-4d15-b83c-bf069a7b23bf"
SOURCES = {"SO101_Follower": "calibration/robots/so_follower/follower.json",
           "SO101_Leader": "calibration/teleoperators/so_leader/leader.json"}
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")


def _call(method: str, url: str, body: dict | None = None):
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def load_calibration(path: Path) -> dict:
    cal = json.loads(path.read_text())
    missing = [j for j in JOINTS if j not in cal or any(f not in cal[j] for f in FIELDS)]
    if missing:
        raise ValueError(f"{path}: incomplete calibration for {missing}")
    return {j: {f: int(cal[j][f]) for f in FIELDS} for j in JOINTS}


def diff(old: dict | None, new: dict) -> dict:
    old = old or {}
    return {j: {f: [old.get(j, {}).get(f), new[j][f]] for f in FIELDS if old.get(j, {}).get(f) != new[j][f]}
            for j in JOINTS if old.get(j) != new[j]}


def plan(robots: list[dict], repo: Path) -> list[tuple[dict, dict]]:
    """(robot record, new calibration) for every robot whose type has a calibration file."""
    out = []
    for r in robots:
        src = SOURCES.get(r["type"])
        if src:
            out.append((r, load_calibration(repo / src)))
    if {r["type"] for r, _ in out} != set(SOURCES):
        raise ValueError(f"expected one robot of each type {sorted(SOURCES)}, found {[r['type'] for r in robots]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path("~/inference/dlsj-smolvla"))
    ap.add_argument("--studio", default=STUDIO)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--backup", type=Path, default=Path(f"~/inference/tmp/studio_robots_{time.strftime('%Y%m%d-%H%M%S')}.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    base = f"{a.studio}/api/projects/{a.project}/robots"
    robots = _call("GET", base)
    todo = plan(robots, a.repo.expanduser())
    for r, cal in todo:
        d = diff(r["payload"].get("calibration"), cal)
        print(f"{r['name']} ({r['type']}, serial {r['payload'].get('serial_number')}): "
              f"{'already current' if not d else f'{len(d)} joints change'}")
        for j, fs in d.items():
            print(f"  {j}: " + ", ".join(f"{f} {o} -> {n}" for f, (o, n) in fs.items()))
    if a.dry_run:
        return
    backup = a.backup.expanduser()
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(json.dumps(robots, indent=2))
    print(f"previous records saved to {backup}")
    for r, cal in todo:
        if not diff(r["payload"].get("calibration"), cal):
            continue
        body = {k: r[k] for k in ("id", "name", "type")} | {"payload": {**r["payload"], "calibration": cal}}
        _call("PUT", f"{base}/{r['id']}", body)
        back = _call("GET", f"{base}/{r['id']}")
        if diff(back["payload"].get("calibration"), cal) or back["payload"].get("serial_number") != r["payload"].get(
                "serial_number"):
            sys.exit(f"{r['name']}: Studio did not keep the update, restore from {backup}")
        print(f"{r['name']}: updated (updated_at {back.get('updated_at')})")


if __name__ == "__main__":
    main()
