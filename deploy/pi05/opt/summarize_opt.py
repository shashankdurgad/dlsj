"""Join accuracy (eval_actions.py report --out) and latency (bench_variants.py --json) into one Markdown table.

    python summarize_opt.py --acc ~/inference/logs/pi05/opt/acc.json --bench ~/inference/logs/pi05/opt/bench.jsonl \
        --map fp16=pi05_junction_fp16 --map int8=pi05_junction_int8 ...

`--map pred_name=export_dir_name` links a prediction file to its export. Latency is the median over all
rounds' medians (runtime call and OV-only) and the worst p90; speed-ups are against the 467 ms
PyTorch-XPU baseline and the 188 ms INT8 runtime number this study started from.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

BASELINE_MS = 467.0
START_MS = 188.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acc", type=Path, required=True)
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--exports", type=Path, default=Path("~/inference/exports").expanduser())
    ap.add_argument("--map", action="append", default=[], metavar="PRED=EXPORT")
    a = ap.parse_args()
    acc = json.loads(a.acc.read_text())
    floor = acc.get("seed_floor") or {}
    bench: dict[str, list[dict]] = defaultdict(list)
    for line in a.bench.read_text().splitlines():
        r = json.loads(line)
        if r.get("variant", "default") == "default":
            bench[Path(r["export"]).name].append(r)
    print("| Variant | Runtime median ms | p90 | OV-only median | vs 467 ms | vs 188 ms | MAE k10 | ΔMAE k10 vs FP16 [95% CI] "
          "| MAE k50 | Dev vs FP16 MAE / max | cos (centred) | Size GB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for kv in a.map:
        pred, exp = kv.split("=", 1)
        r = acc["variants"].get(pred)
        b = bench.get(exp, [])
        if b:
            med = statistics.median(x["runtime_call"]["median_ms"] for x in b)
            p90 = max(x["runtime_call"]["p90_ms"] for x in b)
            ov = statistics.median(x["ov_model_only"]["median_ms"] for x in b)
            lat = f"{med:.0f} | {p90:.0f} | {ov:.0f} | {BASELINE_MS / med:.2f}× | {START_MS / med:.2f}×"
        else:
            lat = "n/a | n/a | n/a | | "
        d = a.exports / exp
        size = sum(f.stat().st_size for f in d.iterdir() if f.is_file()) / 1e9 if d.is_dir() else float("nan")
        if r:
            v, dv, bt = r["vs_recorded"], r["vs_fp16"], r["k10_mae_change_vs_fp16"]
            accs = (f"{v['k10']['mae']:.2f} | {bt['rel']:+.1%} [{bt['lo']:+.1%}, {bt['hi']:+.1%}] | {v['k50']['mae']:.2f} | "
                    f"{dv['mae']:.2f} / {dv['max_abs']:.1f} | {dv['cos_centered']:.3f}")
        else:
            accs = "n/a | | | | "
        print(f"| {exp} | {lat} | {accs} | {size:.2f} |")
    if floor:
        print(f"\nSeed floor (FP16 vs FP16 with other noise seeds): dev MAE {floor['mae']:.2f}, max {floor['max_abs']:.1f}, "
              f"cos (centred) {floor['cos_centered']:.3f}")


if __name__ == "__main__":
    main()
