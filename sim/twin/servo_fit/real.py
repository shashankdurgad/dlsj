"""Real servo response of the SO-101 follower: metrics.py on the XPS controller traces (MolmoAct2 5.5k / 8k, zero-shot,
other policies) and the cal2 teleop episodes, plus the replay windows for the twin (twin_replay.py).

    sim/twin/dynamics/py.sh sim/twin/servo_fit/real.py [--no-fit]
    -> sim/twin/out/servo_fit/real.json, sim/twin/out/servo_fit/windows.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.dynamics import data as D  # noqa: E402
from sim.twin.servo_fit import metrics as M  # noqa: E402

OUT = REPO / "sim/twin/out/servo_fit"
WIN = 210  # replay window, ticks (7 s)


def load_traces(root: Path = D.TRACES, min_len: int = 60) -> list[dict]:
    """Contiguous 30 Hz runs of every trace with the export tag of its `run` record."""
    out = []
    for f in sorted(root.glob("*.jsonl")):
        export, rows = "", []
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "run":
                export = str((r.get("args") or {}).get("export") or (r.get("args") or {}).get("ckpt") or "")
            elif r.get("kind") == "tick" and r.get("measured") and r.get("cmd"):
                rows.append((int(r["t"]), float(r["wall"]), r["measured"], r["cmd"], r.get("phase", "")))
        if not rows:
            continue
        t = np.array([r[0] for r in rows])
        cut = np.where(np.diff(t) != 1)[0] + 1
        for si, (a, b) in enumerate(zip(np.r_[0, cut], np.r_[cut, len(rows)], strict=True)):
            if b - a < min_len:
                continue
            seg = rows[a:b]
            out.append({"file": f"{f.stem}#{si}", "export": export, "wall": np.array([r[1] for r in seg]),
                        "meas": np.array([r[2] for r in seg], float), "cmd": np.array([r[3] for r in seg], float),
                        "phase": np.array([r[4] for r in seg])})
    return out


def group(export: str) -> str:
    e = export.lower()
    if "ft_cal2_456_8k" in e:
        return "molmo_8k"
    if "ft_cal2_456" in e:
        return "molmo_5p5k"
    if "molmo" in e:
        return "molmo_zeroshot"
    return "other_policy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--teleop-max", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load_traces()
    groups: dict[str, list] = {}
    for s in tr:
        groups.setdefault(group(s["export"]), []).append(s)
    res = {"_doc": M.__doc__, "groups": {}}
    dts = np.concatenate([np.diff(s["wall"]) for s in tr])
    res["trace_tick_dt_ms"] = {q: round(float(np.percentile(dts, q)) * 1e3, 1) for q in (5, 50, 95, 99)}
    for g, segs in sorted(groups.items()):
        C, Mm = [s["cmd"] for s in segs], [s["meas"] for s in segs]
        res["groups"][g] = {"n_seg": len(segs), "ticks": int(sum(len(c) for c in C)),
                            "metrics": M.all_metrics(C, Mm, fit=not args.no_fit)}
        print(g, json.dumps(res["groups"][g]["metrics"].get("gripper_close")), flush=True)
    mol = [s for s in tr if group(s["export"]).startswith("molmo_") and "zeroshot" not in group(s["export"])]
    res["groups"]["molmo_ft_all"] = {"n_seg": len(mol), "metrics": M.all_metrics(
        [s["cmd"] for s in mol], [s["meas"] for s in mol], fit=not args.no_fit)}
    te = D.teleop()
    keys = [k for k in sorted(te) if not te[k]["excluded"]][:args.teleop_max]
    res["groups"]["teleop"] = {"n_seg": len(keys), "metrics": M.all_metrics(
        [te[k]["cmd"] for k in keys], [te[k]["meas"] for k in keys], fit=not args.no_fit, clamp=1e9)}
    print("teleop", json.dumps(res["groups"]["teleop"]["metrics"].get("gripper_close")), flush=True)
    # replay windows: MolmoAct2 fine-tune traces cut into 10 s windows with motion, + teleop episodes
    W = []
    for s in mol:
        for a in range(0, len(s["cmd"]) - WIN + 1, WIN):
            c, m = s["cmd"][a:a + WIN], s["meas"][a:a + WIN]
            if np.ptp(c[:, :5], 0).max() >= 5.0 or np.ptp(c[:, 5]) >= 10:
                W.append((f"{s['file']}@{a}", group(s["export"]), c, m))
    lens = [len(te[k]["cmd"]) for k in keys]
    res["teleop_len_p10_p50_p90"] = np.percentile(lens, [10, 50, 90]).tolist()
    nt = 0
    for k in keys:
        c, m = te[k]["cmd"], te[k]["meas"]
        for a in range(0, len(c) - WIN + 1, WIN):
            if nt < 120:
                W.append((f"teleop{k}@{a}", "teleop", c[a:a + WIN], m[a:a + WIN]))
                nt += 1
    np.savez_compressed(OUT / "windows.npz", name=np.array([w[0] for w in W]), src=np.array([w[1] for w in W]),
                        cmd=np.stack([w[2] for w in W]).astype(np.float32),
                        meas=np.stack([w[3] for w in W]).astype(np.float32))
    (OUT / "real.json").write_text(json.dumps(res, indent=1, default=float))
    print("windows", len(W), {g: sum(1 for w in W if w[1] == g) for g in set(w[1] for w in W)})


if __name__ == "__main__":
    main()
