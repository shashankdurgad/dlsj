import time, numpy as np, torch, torchvision, openvino as ov
m = torchvision.models.resnet18(weights=None).eval()
x = torch.randn(1, 3, 256, 256)
ovm = ov.convert_model(m, example_input=x, input=[1,3,256,256])
core = ov.Core()
def bench(dev, hint="LATENCY", prec=None, n=200):
    cfg = {"PERFORMANCE_HINT": hint}
    if prec: cfg["INFERENCE_PRECISION_HINT"] = prec
    t0 = time.time(); cm = core.compile_model(ovm, dev, cfg); ct = time.time() - t0
    req = cm.create_infer_request(); a = x.numpy()
    for _ in range(20): req.infer({0: a})
    ts = []
    for _ in range(n):
        t = time.perf_counter(); req.infer({0: a}); ts.append((time.perf_counter() - t) * 1e3)
    print(f"{dev:4s} {(prec or chr(45)):7s} compile {ct:5.1f}s  median {np.median(ts):6.2f} ms  p99 {np.percentile(ts,99):6.2f} ms")
for d, p in [("CPU", None), ("GPU", "f16"), ("NPU", None)]:
    try: bench(d, prec=p)
    except Exception as e: print(d, "FAILED:", str(e)[:200])
xm = m.to("xpu"); xx = x.to("xpu")
with torch.no_grad():
    for _ in range(20): xm(xx)
    torch.xpu.synchronize(); ts = []
    for _ in range(200):
        t = time.perf_counter(); xm(xx); torch.xpu.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
print(f"PyTorch XPU eager fp32  median {np.median(ts):6.2f} ms  p99 {np.percentile(ts,99):6.2f} ms")
