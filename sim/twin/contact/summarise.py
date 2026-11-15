"""Table of gate arms: lift %, failure split, held reading and held brick centre (grasp_gate/heldpose.py's measure:
median brick centre in gripper_link over the lifted ticks of the successful demos; real (-0.8, 2.8, -88.0) mm, real
held reading 5.72), push events (brick moved > 1.5 mm on the table before the close: count, median drag at the close),
per-pose success. Plain python.

    python sim/twin/contact/summarise.py <dir with <tag>/result.json + trace.npz> [tags ...] [--vs base]
"""
import json
import sys
from pathlib import Path

import numpy as np

REAL = np.array([-0.8, 2.8, -88.0])
LABELS = Path(__file__).resolve().parents[1] / "grasp_gate/brick_labels.json"


def held(d: Path) -> tuple:
    tr = np.load(d / "trace.npz")
    per = {e["episode"]: e for e in json.loads((d / "result.json").read_text())["episodes"]}
    pts, drag, ev = [], [], 0
    for k, ep in enumerate(tr["episodes"].tolist()):
        n = int(tr["end"][k] - tr["start"][k]) + 1
        ic = int(tr["cf"][k] - tr["start"][k])
        disp = np.linalg.norm(tr["bp"][k, :ic + 1, :2] - tr["b0"][k, :2], axis=1)
        if (disp > 0.0015).any():
            ev += 1
            drag.append(disp[ic] * 1e3)
        if not per[ep]["success"]:
            continue
        h = tr["bp"][k, :n, 2] - tr["b0"][k, 2]
        near = np.linalg.norm(tr["bp"][k, :n] - tr["tip"][k, :n], axis=1) < 0.04
        sel = (h > 0.03) & near
        if sel.any():
            pts.append(np.median(tr["rel"][k, :n][sel], 0) * 1e3)
    m = np.median(np.array(pts), 0) if pts else np.full(3, np.nan)
    return m, ev, (float(np.median(drag)) if drag else 0.0)


def row(d: Path, base: dict | None = None) -> dict:
    res = json.loads((d / "result.json").read_text())
    s = res["summary"]
    m, ev, drag = held(d)
    labels = json.loads(LABELS.read_text())["episodes"] if LABELS.exists() else {}
    by = {}
    for e in res["episodes"]:
        p = labels.get(str(e["episode"]), {}).get("pose", "?")
        by.setdefault(p, []).append(e["success"])
    out = {"tag": d.name, "lift": round(100 * s["success"], 1), "pushed": round(100 * s["failures"]["pushed"], 1),
           "missed": round(100 * s["failures"]["missed"], 1), "tipped": round(100 * s["failures"]["tipped"], 1),
           "slipped": round(100 * s["failures"]["slipped"], 1), "held_reading": s["grip_held"]["sim_success"],
           "held_real": s["grip_held"]["real"], "held_xyz_mm": [round(float(v), 1) for v in m],
           "dx_dz_to_real": [round(float(m[0] - REAL[0]), 1), round(float(m[2] - REAL[2]), 1)],
           "push_events": ev, "median_drag_mm": round(drag, 1),
           "by_pose": {p: f"{sum(v)}/{len(v)}" for p, v in sorted(by.items())}}
    if base is not None:
        b = {e["episode"]: e["success"] for e in base["episodes"]}
        out["flips"] = {"+": sorted(e["episode"] for e in res["episodes"] if e["success"] and not b.get(e["episode"])),
                        "-": sorted(e["episode"] for e in res["episodes"] if not e["success"] and b.get(e["episode"]))}
    return out


def main():
    argv = sys.argv[1:]
    vs = None
    if "--vs" in argv:
        i = argv.index("--vs")
        vs = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    args = argv
    root = Path(args[0])
    tags = args[1:] or sorted(p.parent.name for p in root.glob("*/result.json"))
    tags = [t for t in tags if t != vs] if vs else tags
    base = json.loads((root / vs / "result.json").read_text()) if vs else None
    rows = [row(root / vs)] if vs else []
    rows += [row(root / t, base) for t in tags if (root / t / "trace.npz").exists()]
    print("| arm | lift % | pushed % | missed / tipped % | held reading | held x, y, z mm | "
          "push events (median drag mm) | by pose | flips vs base |")
    for r in rows:
        print(f"| {r['tag']} | {r['lift']} | {r['pushed']} | {r['missed']} / {r['tipped']} | {r['held_reading']} | "
              f"{r['held_xyz_mm']} | {r['push_events']} ({r['median_drag_mm']}) | {r['by_pose']} | "
              f"{r.get('flips', '')} |")
    (root / "contact_summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
