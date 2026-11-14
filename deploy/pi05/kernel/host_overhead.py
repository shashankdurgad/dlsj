"""Host-side overhead of one π0.5 call on the GPU, and cold vs warm compile.

`calls`: the same compiled IR called five ways, interleaved in blocks (synthetic static-shape inputs):
  - `compiled_call`  : `compiled_model(inputs)` (what physicalai's adapter does: new request state, numpy copy in)
  - `req_infer_copy` : one reused `InferRequest`, `req.infer(inputs)` (copies numpy into its tensors)
  - `req_infer_share`: `req.infer(inputs, share_inputs=True, share_outputs=True)` (zero-copy wrap)
  - `preset_tensors` : `ov.Tensor`s created once and `set_tensor`'d once, then `req.infer()` with no arguments
  - `async_wait`     : preset tensors, `start_async()` + `wait()`
`compile`: compile time with no cache, an empty CACHE_DIR (cold: compiles and writes the blob) and the same
CACHE_DIR again (warm: loads the blob), each in a fresh subprocess.

    PY=~/inference/physical-ai-studio/library/.venv/bin/python
    $PY host_overhead.py calls ~/inference/exports/pi05_junction_int8/pi05.xml --out host_calls.json
    $PY host_overhead.py compile ~/inference/exports/pi05_junction_int8/pi05.xml --out host_compile.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_lib as K  # noqa: E402
from profile_kernels import make_inputs  # noqa: E402

CFG = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}


def calls(xml: Path, iters: int, blocks: int, out: Path) -> None:
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(xml))
    inputs = make_inputs(model)
    compiled = core.compile_model(model, "GPU", CFG)
    req = compiled.create_infer_request()
    preq = compiled.create_infer_request()
    for k, v in inputs.items():
        preq.set_tensor(k, ov.Tensor(np.ascontiguousarray(v)))

    def compiled_call():
        compiled(inputs)

    def req_infer_copy():
        req.infer(inputs)

    def req_infer_share():
        req.infer(inputs, share_inputs=True, share_outputs=True)

    def preset_tensors():
        preq.infer()

    def async_wait():
        preq.start_async()
        preq.wait()

    modes = [compiled_call, req_infer_copy, req_infer_share, preset_tensors, async_wait]
    for m in modes:
        for _ in range(3):
            m()
    ms: dict[str, list[float]] = {m.__name__: [] for m in modes}
    for b in range(blocks):
        for m in (modes if b % 2 == 0 else modes[::-1]):
            for _ in range(iters):
                t = time.perf_counter()
                m()
                ms[m.__name__].append((time.perf_counter() - t) * 1e3)
    res = {"xml": str(xml), "ov_version": ov.get_version(), "modes": {k: K.stats(v) for k, v in ms.items()}}
    out.write_text(json.dumps(res, indent=1))
    for k, v in res["modes"].items():
        print(f"{k:<16} median {v['median_ms']:.2f}  p90 {v['p90_ms']:.2f}  min {v['min_ms']:.2f} ms")


def _compile_once(xml: str, cache: str) -> None:
    import openvino as ov

    cfg = dict(CFG)
    if cache:
        cfg["CACHE_DIR"] = cache
    t0 = time.perf_counter()
    core = ov.Core()
    c = core.compile_model(xml, "GPU", cfg)
    t1 = time.perf_counter()
    req = c.create_infer_request()
    import openvino as _ov  # noqa: F401
    m = core.read_model(xml)
    for k, v in make_inputs(m).items():
        req.set_tensor(k, ov.Tensor(v))
    t2 = time.perf_counter()
    req.infer()
    t3 = time.perf_counter()
    print(json.dumps({"compile_s": round(t1 - t0, 2), "first_infer_s": round(t3 - t2, 3)}))


def compile_test(xml: Path, out: Path, scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.mkdtemp(prefix="ovcache_cold_", dir=scratch))
    res = {}
    for label, c in (("no_cache", ""), ("cache_cold", str(cache)), ("cache_warm", str(cache)),
                     ("cache_warm2", str(cache))):
        p = subprocess.run([sys.executable, __file__, "_compile_once", str(xml), "--cache", c],
                           capture_output=True, text=True)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("{")), None)
        res[label] = json.loads(line) if line else {"error": p.stderr[-1500:]}
        print(label, res[label], flush=True)
    res["cache_blob_mb"] = round(sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) / 2**20, 1)
    shutil.rmtree(cache, ignore_errors=True)
    out.write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["calls", "compile", "_compile_once"])
    ap.add_argument("xml", type=Path)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--cache", default="")
    ap.add_argument("--scratch", type=Path, default=Path("~/inference/tmp").expanduser())
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    xml = a.xml.expanduser()
    if a.mode == "calls":
        calls(xml, a.iters, a.blocks, a.out)
    elif a.mode == "compile":
        compile_test(xml, a.out, a.scratch)
    else:
        _compile_once(str(xml), a.cache)


if __name__ == "__main__":
    main()
