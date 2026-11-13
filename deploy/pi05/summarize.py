"""Turn the JSON lines written by `studio_bench.py` / `ov_bench.py` into a Markdown results table.

    python3 ~/inference/deploy/pi05/summarize.py ~/inference/logs/pi05/final_*.jsonl
"""

import json
import sys
from pathlib import Path


def row(d: dict) -> str:
    """One Markdown row; Studio runtime rows also show the OpenVINO-only time."""
    cfg = ", ".join(f"{k}={v}" for k, v in d.get("extra_config", {}).items())
    hint = d.get("precision_hint", "")
    opts = "; ".join(x for x in (cfg, "" if hint == "device default" else f"hint {hint}") if x) or "-"
    if "runtime_call" in d:
        name, rt, ov = Path(d["export"]).name, d["runtime_call"], d["ov_model_only"]
        return (f"| {name} | InferenceModel | {d['device']} | {opts} | {rt['median_ms']} | {rt['p90_ms']} | "
                f"{rt['p99_ms']} | {ov['median_ms']} | {d['peak_rss_gib']} | {d['date']} |")
    name = "/".join(Path(d["model"]).parts[-2:])
    return (f"| {name} | raw IR | {d['device']} | {opts} | {d['median_ms']} | {d['p90_ms']} | {d['p99_ms']} | "
            f"{d['median_ms']} | {d['peak_rss_gib']} | {d['date']} |")


def main() -> None:
    print("| export | harness | device | options | median ms | p90 ms | p99 ms | OV-only median ms | peak RSS GiB | date |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for path in sys.argv[1:]:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                print(row(json.loads(line)))


if __name__ == "__main__":
    main()
