"""Match timed windows (results.json 'wall', idle window) with /tmp/rapl.log -> average W and J per chunk."""
import json, sys
import numpy as np
log = np.loadtxt("/tmp/rapl.log", skiprows=1)
t, cols = log[:, 0], {"package": 1, "core": 2, "uncore": 3, "psys": 4}
def watts(a, b):
    i, j = np.searchsorted(t, a), np.searchsorted(t, b) - 1
    if j <= i:
        return None
    dt = t[j] - t[i]
    out = {}
    for k, c in cols.items():
        d = log[j, c] - log[i, c]
        if d < 0:  # counter wrapped: sum positive steps
            d = np.clip(np.diff(log[i:j + 1, c]), 0, None).sum()
        out[k] = d / 1e6 / dt
    return out
res = json.load(open(sys.argv[1]))
idle = json.load(open(sys.argv[2]))
wi = watts(*idle["wall"])
print("idle", {k: round(v, 1) for k, v in wi.items()})
summary = {"idle_w": wi, "rows": {}}
for k, r in res.items():
    w = watts(*r["wall"])
    if w is None:
        print(k, "no overlap with the log"); continue
    j = {kk: w[kk] * r["median_ms"] / 1e3 for kk in w}
    jd = {kk: (w[kk] - wi[kk]) * r["median_ms"] / 1e3 for kk in w}
    summary["rows"][k] = {"label": r.get("label"), "median_ms": r["median_ms"], "w": w, "j_per_chunk": j,
                          "j_per_chunk_above_idle": jd}
    print(r.get("label") or k, f"{r['median_ms']:.0f} ms", {kk: f"{v:.1f} W" for kk, v in w.items()},
          "J/chunk pkg %.2f psys %.2f (above idle pkg %.2f psys %.2f)" % (j["package"], j["psys"], jd["package"], jd["psys"]))
json.dump(summary, open(sys.argv[3], "w"), indent=1)
