"""Render the kernel-study JSON into Markdown tables (stdlib only).

    python summarize_kernel.py profile prof_int8_release.json [--top 15]
    python summarize_kernel.py compare prof_a.json prof_b.json ...      # side-by-side wall/kernel/GatedMLP counts
    python summarize_kernel.py sweep sweep_int8.jsonl [--metric runtime_call|ov_model_only]
    python summarize_kernel.py counters bapp_int8/benchmark_detailed_counters_report.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_lib as K  # noqa: E402

AGG_COLS = [("key", "Group"), ("ms", "ms / inference"), ("share", "Share"), ("count", "Kernels"), ("mean_us", "µs / kernel")]
AGG_FMT = {"ms": ".2f", "share": ".1%", "mean_us": ".1f"}


def load_profile(path: Path) -> dict:
    """Load a profile and recompute the aggregates from the stored rows with the current kernel_lib rules."""
    p = json.loads(path.read_text())
    rows = p["rows"]
    pr = K.infer_prefix_rows(r.get("out_shapes", []) for r in rows)
    for r in rows:
        r["subgraph"] = K.subgraph_of(r.get("out_shapes", []), prefix_rows=pr)
        r["family"] = K.op_family(r["layer_type"], r["exec_type"])
    kern = sorted((r for r in rows if K.is_kernel(r)), key=lambda r: r.get("order", -1))
    expert = [r for r in kern if r["subgraph"] == "expert"]
    if expert and p.get("expert_step_ms"):
        p["expert_step_ms"] = [sum(r["real_ms"] for r in g) for g in K.split_steps(expert, len(p["expert_step_ms"]))]
    p["by_family"] = K.aggregate(kern, lambda r: r["family"])
    p["by_subgraph"] = K.aggregate(kern, lambda r: r["subgraph"])
    p["by_subgraph_family"] = K.aggregate(kern, lambda r: f'{r["subgraph"]} | {r["family"]}')[:40]
    return p


def profile(path: Path, top: int) -> str:
    p = load_profile(path)
    d = p["dispatch"]
    out = [f"### {path.name} ({p['ov_version']}, {p.get('full_device_name', '')})\n",
           f"Env: `{p.get('env', {})}`. Compile {p['compile_s']} s. Wall without PERF_COUNT: median "
           f"{p['wall_no_perfcount']['median_ms']:.1f} ms (p90 {p['wall_no_perfcount']['p90_ms']:.1f}); with PERF_COUNT "
           f"{d['wall_ms']:.1f} ms. Sum of GPU kernel time {d['kernel_ms']:.1f} ms in {int(d['n_kernels'])} kernel launches; "
           f"uncovered gap {d['gap_ms']:.1f} ms ({d['gap_share']:.1%}) = {d['gap_us_per_kernel']:.1f} µs per launch; "
           f"{int(d.get('n_small', 0))} kernels shorter than 20 µs ({d.get('small_ms', 0):.1f} ms total).\n"]
    nopc = p["wall_no_perfcount"]["median_ms"]
    d2 = K.dispatch_estimate(nopc, d["kernel_ms"], int(d["n_kernels"]))
    out.append(f"Against the wall time without profiling ({nopc:.1f} ms), the uncovered host/launch gap is "
               f"{d2['gap_ms']:.1f} ms ({d2['gap_share']:.1%}), i.e. {d2['gap_us_per_kernel']:.1f} µs per launch "
               f"(PERF_COUNT itself adds {d['wall_ms'] - nopc:.1f} ms of profiling overhead).\n")
    if p.get("expert_step_ms"):
        out.append("Action expert per denoising step (ms): " + ", ".join(f"{x:.2f}" for x in p["expert_step_ms"]) + "\n")
    for key, title in (("by_subgraph", "Per sub-graph"), ("by_family", "Per op family"),
                       ("by_backend", "Per implementation library"), ("by_subgraph_family", "Sub-graph × op family"),
                       ("by_exec_type", "Top primitives (layer type | implementation)")):
        out.append(f"\n**{title}**\n\n" + K.md_table(p[key][:top], AGG_COLS, AGG_FMT))
    return "\n".join(out)


def compare(paths: list[Path]) -> str:
    rows = []
    for path in paths:
        p = load_profile(path)
        fam = {a["key"]: a for a in p["by_family"]}
        sub = {a["key"]: a for a in p["by_subgraph"]}
        rows.append({"name": path.stem, "wall": p["wall_no_perfcount"]["median_ms"], "kern": p["dispatch"]["kernel_ms"],
                     "n": int(p["dispatch"]["n_kernels"]), "gm": fam.get("GatedMLP", {}).get("count", 0),
                     "gm_ms": fam.get("GatedMLP", {}).get("ms", 0.0), "mm": fam.get("MatMul/FC", {}).get("ms", 0.0),
                     "act": fam.get("Activation/GLU", {}).get("ms", 0.0),
                     "expert": sub.get("expert", {}).get("ms", 0.0), "vlm": sub.get("vlm", {}).get("ms", 0.0),
                     "vision": sub.get("vision", {}).get("ms", 0.0)})
    cols = [("name", "Profile"), ("wall", "Wall ms (no PC)"), ("kern", "Σ kernel ms"), ("n", "Kernels"),
            ("gm", "GatedMLP prims"), ("gm_ms", "GatedMLP ms"), ("mm", "MatMul/FC ms"), ("act", "Act/GLU ms"),
            ("vision", "vision ms"), ("vlm", "VLM ms"), ("expert", "expert ms")]
    return K.md_table(rows, cols, {k: ".1f" for k in ("wall", "kern", "gm_ms", "mm", "act", "expert", "vlm", "vision")})


def sweep(path: Path, metric: str) -> str:
    res = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    errs = [r for r in res if "error" in r]
    rows = K.pooled_compare(res, metric=metric)
    for r in rows:
        cfg = next((x for x in res if x["variant"] == r["variant"] and "error" not in x), {})
        bits = [f"{k}={v}" for k, v in cfg.get("extra_config", {}).items()]
        bits += [f"env {k}={v}" for k, v in cfg.get("env", {}).items()] + (["taskset " + " ".join(cfg["prefix"][2:])]
                                                                           if cfg.get("prefix") else [])
        r["setting"] = ", ".join(bits) or "LATENCY, NUM_STREAMS=1, CACHE_DIR (studio_bench default)"
    cols = [("variant", "Variant"), ("setting", "Setting"), ("rounds", "Rounds"), ("median_ms", "Median ms"),
            ("p90_ms", "p90 ms"), ("delta_ms", "Δ vs default (paired) ms"), ("delta_pct", "Δ %")]
    out = K.md_table(rows, cols, {"median_ms": ".1f", "p90_ms": ".1f", "delta_ms": "+.1f", "delta_pct": "+.1f"})
    if errs:
        out += "\nFailed: " + "; ".join(f"{e['variant']} r{e['round']}: {e['error'].strip().splitlines()[-1][:200]}"
                                        for e in errs) + "\n"
    return out


def counters(path: Path, top: int) -> str:
    rows, total = K.parse_counters_csv(path.read_text())
    kern = [r for r in rows if K.is_kernel(r)]
    s = f"benchmark_app counters: {len(kern)} executed kernels, Σ {sum(r['real_ms'] for r in kern):.1f} ms (Total line {total}).\n\n"
    agg = K.aggregate(kern, lambda r: f"{r['layer_type']} | {K.strip_precision(r['exec_type'])}")
    return s + K.md_table(agg[:top], AGG_COLS, AGG_FMT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["profile", "compare", "sweep", "counters"])
    ap.add_argument("paths", type=Path, nargs="+")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--metric", default="runtime_call")
    a = ap.parse_args()
    if a.mode == "profile":
        print("\n\n".join(profile(p, a.top) for p in a.paths))
    elif a.mode == "compare":
        print(compare(a.paths))
    elif a.mode == "sweep":
        print(sweep(a.paths[0], a.metric))
    else:
        print(counters(a.paths[0], a.top))


if __name__ == "__main__":
    main()
