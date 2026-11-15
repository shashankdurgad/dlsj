"""Old vs new light rig at the recorded states of held-out episodes, against the real frames (Isaac).

    NAME=dlsj-twin-light sim/twin/run.sh sim/twin/lighting/validate.py --rig sim/twin/out/lighting/rig_candidate.json \
        [--eps 33 126 220 313] [--tag v1]

Per evaluation sample (visuals/fit/harness.Harness posing) renders the twin twice on the same pose: with the current
nominal look (TwinLook() from visuals/lighting.json, as every other agent renders it) and with the candidate rig
(`--rig`: a lighting.json-style file; its tubes / side lamps are built under /World/LightNew/*, the old lights are
switched off, dome / key / RTX ambient set from the file). Computes lighting/metrics.light_metrics for both against the
real frames (masks from visual_fit/render_before.npz) and writes, under sim/twin/out/lighting/:
  validate_<tag>.json            metrics old / new per camera,
  validate_<tag>.npz             renders (camera response applied, as the policy sees them),
  compare_<tag>_<cam>.jpg        real | old | new rows, compare_<tag>_<cam>_crops.jpg plate / table close-ups.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/lighting"
VF = REPO / "sim/twin/out/visual_fit"
LIGHT_FIELDS = ("dome_intensity", "dome_rgb", "key_intensity", "key_angle_deg", "key_rot_xyz_deg", "key_rgb",
                "ceiling_lights", "side_lights", "ambient_intensity", "exposure_scale")


class CandidateLook:
    """The nominal TwinLook with the light fields of a candidate lighting file on top (works before TwinLook has the
    rig's new fields)."""

    def __init__(self, base, over: dict):
        self.__dict__.update(vars(base))
        self.__dict__.update(over)
        self._s = float(over.get("exposure_scale") or base.scale())

    def scale(self) -> float:
        """Exposure scale of the candidate (lighting.json 'exposure_scale')."""
        return self._s


def _label(img, text):
    import cv2

    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", required=True)
    ap.add_argument("--eps", type=int, nargs="*", default=[33, 126, 220, 313])
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    import cv2

    from sim.twin.lighting import rig
    from sim.twin.lighting.metrics import light_metrics
    from sim.twin.visuals.fit.harness import Harness
    from sim.twin.visuals.twin_scene import TwinLook, color_correct

    h = Harness(passes=args.passes)
    import carb
    from pxr import Gf

    st = carb.settings.get_settings()
    stage = h.env.stage
    old = h.env.look
    cand = json.loads(Path(args.rig).read_text())
    new = CandidateLook(TwinLook(), {k: v for k, v in cand.items() if k in LIGHT_FIELDS})
    olds = [p for p in stage.Traverse() if p.GetTypeName().endswith("Light")]
    old_i = {str(p.GetPath()): p.GetAttribute("inputs:intensity").Get() for p in olds}
    old_amb = st.get(rig.AMBIENT_KEY)
    built = rig.build(stage, new, 1, prefix="/World/LightNew/Ceil", side_name="LightNewSide")
    new_paths = built["ceiling"] + built["side"]
    new_i = {p: stage.GetPrimAtPath(p).GetAttribute("inputs:intensity").Get() for p in new_paths}
    s_new = new.scale()

    def use(which: str):
        for p in olds:
            path = str(p.GetPath())
            v = old_i[path]
            if which == "new":
                v = {"/World/Light": new.dome_intensity * s_new, "/World/Sun": new.key_intensity * s_new}.get(path, 0.0)
            p.GetAttribute("inputs:intensity").Set(float(v))
        for p in new_paths:
            stage.GetPrimAtPath(p).GetAttribute("inputs:intensity").Set(float(new_i[p] if which == "new" else 0.0))
        dome, sun = stage.GetPrimAtPath("/World/Light"), stage.GetPrimAtPath("/World/Sun")
        lk = new if which == "new" else old
        dome.GetAttribute("inputs:color").Set(Gf.Vec3f(*map(float, lk.dome_rgb)))
        sun.GetAttribute("inputs:color").Set(Gf.Vec3f(*map(float, lk.key_rgb)))
        amb = new.ambient_intensity if which == "new" else old_amb
        st.set(rig.AMBIENT_KEY, float(old_amb if amb is None else amb))

    ids = [i for i, s in enumerate(h.samples) if s["ep"] in set(args.eps)]
    cams = ("top", "inspect")
    out = {f"{w}_{c}": [] for w in ("old", "new") for c in cams}
    for n, i in enumerate(ids):
        h.pose(h.samples[i])
        for which in ("old", "new"):
            use(which)
            h.render()
            img = h.render()
            for c in cams:
                out[f"{which}_{c}"].append(color_correct(img[c], c))
        if n % 5 == 0:
            print(f"[validate] {n + 1}/{len(ids)}", flush=True)
    use("old")
    out = {k: np.stack(v) for k, v in out.items()}
    real = np.load(VF / "evalset.npz")
    masks = np.load(VF / "render_before.npz")
    samples = [h.samples[i] for i in ids]
    res = {"rig": args.rig, "eps": args.eps, "n": len(ids)}
    for c in cams:
        r, m = real[c][ids], masks[f"mask_{c}"][ids]
        res[c] = {w: light_metrics(r, out[f"{w}_{c}"], m, samples, c) for w in ("old", "new")}
        print(c, json.dumps(res[c]), flush=True)
        rows, crops = [], []
        for k in range(0, len(ids), max(1, len(ids) // 5))[:5]:
            rows.append(np.concatenate([_label(r[k].copy(), f"real ep{samples[k]['ep']} {samples[k]['moment']}"),
                                        _label(out[f"old_{c}"][k].copy(), "sim old"),
                                        _label(out[f"new_{c}"][k].copy(), "sim new")], 1))
            y0, x0 = (60, 0) if c == "top" else (200, 0)
            crops.append(np.concatenate([x[k][y0:y0 + 260, x0:x0 + 480] for x in
                                         (r, out[f"old_{c}"], out[f"new_{c}"])], 1))
        cv2.imwrite(str(OUT / f"compare_{args.tag}_{c}.jpg"),
                    cv2.resize(np.concatenate(rows, 0), None, fx=0.5, fy=0.5)[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(OUT / f"compare_{args.tag}_{c}_crops.jpg"), np.concatenate(crops, 0)[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    np.savez_compressed(OUT / f"validate_{args.tag}.npz", ids=np.array(ids), **out)
    (OUT / f"validate_{args.tag}.json").write_text(json.dumps(res, indent=1))
    print("VALIDATE_OK", args.tag, flush=True)
    h.close()


if __name__ == "__main__":
    main()
