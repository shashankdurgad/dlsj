"""Experiment: sharpness of the twin's renders vs RTX antialiasing / upscaling settings and render passes.

    sim/twin/run.sh sim/twin/visuals/fit/exp_render.py
Writes sim/twin/out/visual_fit/exp_render.json and exp_render_<cam>.jpg (one crop per setting).
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from sim.twin.visuals.fit.harness import OUT, Harness  # noqa: E402

h = Harness(passes=1)
import carb  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sim.twin.visuals.fit import eval_visuals as ev  # noqa: E402

st = carb.settings.get_settings()
keys = ["/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/post/aa/sharpness", "/rtx/post/taa/samples",
        "/rtx/post/taa/alpha", "/rtx/rendermode", "/rtx/post/resample/fractionalResolution",
        "/rtx/post/scaling/staticRatio", "/rtx-transient/dlssg/enabled", "/rtx/raytracing/subpixel/mode",
        "/rtx/post/tonemap/op", "/rtx/pathtracing/spp", "/rtx/directLighting/sampledLighting/samplesPerPixel",
        "/rtx/reflections/enabled", "/rtx/post/dlss/sharpness", "/rtx/hydra/mdlMaterialWarmup",
        "/rtx/textures/mipBias", "/rtx-transient/resourcemanager/enableTextureStreaming",
        "/rtx/post/lensFlares/enabled",
        "/rtx/post/motionblur/enabled", "/rtx/post/dof/enabled", "/rtx/post/chromaticAberration/enabled"]
defaults = {k: st.get(k) for k in keys}
print("DEFAULTS", json.dumps(defaults), flush=True)
samples, real = ev.load_eval()
ids = [i for i, s in enumerate(samples) if s["ep"] in (33, 173) and s["moment"] in ("start", "inspect")]
VAR = {"default": {}, "dlss_perf": {"/rtx/post/dlss/execMode": 0}, "dlaa": {"/rtx/post/aa/op": 4},
       "taa": {"/rtx/post/aa/op": 1}, "fxaa": {"/rtx/post/aa/op": 2}, "none": {"/rtx/post/aa/op": 0},
       "dlaa_mip-1": {"/rtx/post/aa/op": 4, "/rtx/textures/mipBias": -1.0},
       "dlss_q_sharp": {"/rtx/post/aa/sharpness": 0.8, "/rtx/post/dlss/sharpness": 0.8}}
res, crops = {}, {c: [] for c in ev.CAMS}
for name, kv in VAR.items():
    for k, v in defaults.items():
        if v is not None and k in sum((list(x) for x in VAR.values()), []):
            st.set(k, v)
    for k, v in kv.items():
        st.set(k, v)
    res[name] = {}
    for passes in (1, 2, 6):
        acc = {c: [] for c in ev.CAMS}
        t0 = time.time()
        for i in ids:
            h.pose(samples[i])
            for _ in range(passes):
                img = h.render()
            for c in ev.CAMS:
                acc[c].append(ev.lap_var(ev.gray(img[c])))
            if passes == 2 and i == ids[0]:
                for c in ev.CAMS:
                    cr = np.ascontiguousarray(img[c][200:440, 160:480])
                    cv2.putText(cr, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    crops[c].append(cr)
        res[name][passes] = {c: round(float(np.mean(v)), 1) for c, v in acc.items()} | {"s": round(time.time() - t0, 2)}
    print(name, json.dumps(res[name]), flush=True)
res["real"] = {c: round(float(np.mean([ev.lap_var(ev.gray(real[c][i])) for i in ids])), 1) for c in ev.CAMS}
print("real", res["real"], flush=True)
for c in ev.CAMS:
    crops[c].append(np.ascontiguousarray(real[c][ids[0]][200:440, 160:480]))
    cv2.imwrite(str(OUT / f"exp_render_{c}.jpg"), np.concatenate([np.concatenate(crops[c][:5], 1),
                np.concatenate((crops[c][5:] + [crops[c][-1]] * 5)[:5], 1)], 0)[..., ::-1])
(OUT / "exp_render.json").write_text(json.dumps(dict(defaults=defaults, results=res), indent=1))
print("EXP_OK", flush=True)
h.close()
