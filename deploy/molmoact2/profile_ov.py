"""Profile a MolmoAct2 OpenVINO export on the XPS: where does the time go?

1. Exact prefix token accounting from the prepared graph inputs (input_ids, images, token_pooling).
2. Wall-clock split: preprocessors (state/joint frame, image resize+patchify, tokenizer, input assembly) vs graph
   execution vs postprocessors.
3. Per-op GPU profile (PERF_COUNT) grouped by the PyTorch module the node came from:
   vision_backbone (ViT + pooling/projector), transformer (Molmo2-ER LM prefill), action_expert (10 flow steps,
   incl. context K/V projections), other. Profiling inflates totals, so shares are reported alongside.

    python profile_ov.py ~/inference/exports/molmoact2_so101_rtc_int8 [--frames-cache ...] [--iters 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import ZEROSHOT, load_frames  # noqa: E402

GROUPS = (("vision", ("vision_backbone",)), ("action_expert", ("action_expert",)),
          ("lm_prefill", ("transformer",)))


def group_of(name: str) -> str:
    for g, keys in GROUPS:
        if any(k in name for k in keys):
            return g
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--data", default="~/inference/data/duplo_teleop_cal2_3phase")
    ap.add_argument("--frames-cache", default="~/inference/logs/molmo/opt/frames_zeroshot_60.npz")
    ap.add_argument("--cache-dir", default="~/inference/ov_cache")
    ap.add_argument("--json")
    a = ap.parse_args()

    from physicalai.inference import InferenceModel

    frames = load_frames(a.data, 60, ZEROSHOT, cache=a.frames_cache)
    f = next(fr for fr in frames if fr["phase"] == "grasp")
    obs = {"state": f["state"][None], "task": [f["task"]]}
    for c, img in f["imgs"].items():
        obs[f"images.{c}"] = np.ascontiguousarray(img[None])

    res = {"export": a.export_dir, "device": a.device}
    # --- wall-clock split with a normal (non-profiling) compile
    m = InferenceModel(str(Path(a.export_dir).expanduser()), backend="openvino", device=a.device,
                       PERFORMANCE_HINT="LATENCY", NUM_STREAMS="1", CACHE_DIR=str(Path(a.cache_dir).expanduser()))
    captured = {}
    orig_predict = m.adapter.predict

    def spy(inputs):
        captured["inputs"] = inputs
        return orig_predict(inputs)

    m.adapter.predict = spy
    m.predict_action_chunk(obs)
    pre_t, graph_t, total_t = [], [], []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        x = dict(obs)
        for p in m.preprocessors:
            x = p(x)
        prepared = m._prepare_inputs(x)
        t1 = time.perf_counter()
        orig_predict(prepared)
        t2 = time.perf_counter()
        m.predict_action_chunk(obs)
        t3 = time.perf_counter()
        pre_t.append((t1 - t0) * 1e3)
        graph_t.append((t2 - t1) * 1e3)
        total_t.append((t3 - t2) * 1e3)
    inputs = captured["inputs"]
    shapes = {k: list(np.asarray(v).shape) for k, v in inputs.items()}
    ids = np.asarray(inputs.get("input_ids"))
    tok = {"input_shapes": shapes}
    if ids.size:
        cfgp = json.loads((Path(a.export_dir).expanduser() / "manifest.json").read_text())
        spec = next((p for p in cfgp["model"]["preprocessors"] if p.get("type") == "molmoact2_inputs"), {})
        patch, col = spec.get("image_patch_id"), spec.get("image_col_id")
        img_ids = set(spec.get("image_token_ids") or []) | {i for i in (patch, col) if i is not None}
        flat = ids.reshape(-1)
        tok |= {"prefix_tokens": int(flat.size), "image_patch_tokens": int((flat == patch).sum()) if patch else None,
                "image_col_tokens": int((flat == col).sum()) if col else None,
                "image_related_tokens": int(np.isin(flat, list(img_ids)).sum()),
                "pad_tokens": int((flat == spec.get("pad_token_id")).sum()) if spec.get("pad_token_id") is not None
                else None}
    res["tokens"] = tok
    res["wall_ms"] = {"preprocess_median": round(float(np.median(pre_t)), 1),
                      "graph_median": round(float(np.median(graph_t)), 1),
                      "end_to_end_median": round(float(np.median(total_t)), 1)}
    del m

    # --- per-op profile
    import openvino as ov

    core = ov.Core()
    xml = next(Path(a.export_dir).expanduser().glob("*.xml"))
    xml = next(p for p in Path(a.export_dir).expanduser().glob("*.xml") if "tokenizer" not in p.name)
    cm = core.compile_model(str(xml), a.device, {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": "YES"})
    req = cm.create_infer_request()
    req.infer(inputs)
    agg = defaultdict(float)
    cnt = defaultdict(int)
    by_type = defaultdict(float)
    prof_t = []
    for _ in range(a.iters):
        t = time.perf_counter()
        req.infer(inputs)
        prof_t.append((time.perf_counter() - t) * 1e3)
        for p in req.profiling_info:
            us = p.real_time.total_seconds() * 1e6
            g = group_of(p.node_name)
            agg[g] += us
            cnt[g] += 1
            by_type[(g, p.exec_type.split("_")[0] if p.exec_type else p.node_type)] += us
    tot = sum(agg.values())
    res["profile"] = {"wall_ms_with_profiling": round(float(np.median(prof_t)), 1),
                      "groups": {g: {"ms": round(v / a.iters / 1e3, 1), "share": round(v / tot, 3),
                                     "kernels": cnt[g] // a.iters} for g, v in sorted(agg.items(), key=lambda kv: -kv[1])},
                      "top_kernel_types": {f"{g}/{t}": round(v / a.iters / 1e3, 1)
                                           for (g, t), v in sorted(by_type.items(), key=lambda kv: -kv[1])[:15]}}
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
