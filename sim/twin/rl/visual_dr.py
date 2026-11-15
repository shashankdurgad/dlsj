"""Small visual domain randomisation for the image path, ramped during training (sim-to-real shifts only).

`scaled_cfg(dr_scale)` shrinks every TwinRandomizer range (sim/twin/env/randomize.py DEFAULTS) toward its nominal
value by `dr_scale` (0 = the fitted twin, 1 = the defaults). At the default 0.4: cameras ~+-4 mm / 0.4 deg (inspect
~+-2 cm / 2 deg), focal +-1 %, exposure ~+-4 %, colour temperature ~+-300 K, brick hue +-2 deg, clutter rare, mild
image noise / blur. Spotlights are a separate knob `spot_scale` (may ramp to 1.0) passed to the lighting agent's
setter on the randomiser (`set_light_scale(s)` or `set_scale(s)`; skipped with a warning until it exists).

    ramp = Ramp(mode="ramp", start=3000, end=9000, dr_scale=0.4, spot_scale=1.0)
    rnd = TwinRandomizer(scaled_cfg(ramp.dr(step)));  ramp.apply(rnd, step)   # before each reset
"""
from __future__ import annotations

import copy

NOMINAL = {"exposure": 1.0, "key_gain": 1.0, "dome_kelvin": 6500.0, "key_kelvin": 6500.0, "key_angle_deg": 15.0,
           "brick_sat": 1.0, "brick_val": 1.0, "plate_sat": 1.0, "plate_val": 1.0, "table_gain": 1.0, "jaw_val": 1.0,
           "robot_body_val": 1.0, "backdrop_gain": 1.0, "gain": 1.0, "contrast": 1.0}
SPREAD = ("brick_hue_deg", "plate_hue_deg", "jaw_hue_deg", "table_tint", "table_shift_m", "key_tilt_deg",
          "key_azimuth_deg", "wb", "p_blur", "p_motion", "p_dome_texture", "p_cable", "p_blob", "p_jpeg")


def _shrink(rng, nom, s):
    lo, hi = rng
    return (nom + (lo - nom) * s, nom + (hi - nom) * s)


def scaled_cfg(s: float, defaults: dict | None = None, codec: str = "emulate") -> dict:
    """TwinRandomizer cfg with every range scaled by s around its nominal value. codec: the sensor pipeline's video
    codec mode for this cfg ("emulate" = in-process JPEG stand-in for training throughput; evaluation should run the
    full H.264 round-trip, codec=None keeps the pipeline default)."""
    if defaults is None:
        from sim.twin.env.randomize import DEFAULTS as defaults
    c = copy.deepcopy(defaults)
    s = float(max(0.0, s))
    c["camera"]["scale"] = s * float(c["camera"].get("scale", 1.0))
    c["image"]["scale"] = s * float(c["image"].get("scale", 1.0))  # sensor pipeline: one scale for the image section
    if codec is not None:
        c["image"]["codec"] = {"mode": codec}

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict) and not k.endswith("variants"):
                walk(v)
            elif k in NOMINAL and isinstance(v, (tuple, list)) and len(v) == 2:
                d[k] = _shrink(v, NOMINAL[k], s)
            elif k in SPREAD and isinstance(v, (int, float)):
                d[k] = v * s
            elif k == "noise" and isinstance(v, (tuple, list)):
                d[k] = (v[0] * s, v[1] * s)
            elif k == "blur_sigma" and isinstance(v, (tuple, list)):
                d[k] = (v[0] * s, max(v[0] * s, v[1] * s))

    for sec in ("light", "material", "clutter", "image"):
        walk(c[sec])
    c["clutter"]["enabled"] = s >= 0.2  # clutter rare at small scales (probabilities scaled above)
    return c


class Ramp:
    """off | fixed | ramp (linear 0 -> target between start and end training steps), separate spotlight knob."""

    def __init__(self, mode="off", start=3000, end=9000, dr_scale=0.4, spot_scale=1.0):
        self.mode, self.start, self.end = mode, start, end
        self.dr_scale, self.spot_scale = dr_scale, spot_scale
        self._warned = False

    def frac(self, step: int) -> float:
        if self.mode == "off":
            return 0.0
        if self.mode == "fixed":
            return 1.0
        return min(1.0, max(0.0, (step - self.start) / max(1, self.end - self.start)))

    def dr(self, step: int) -> float:
        return self.dr_scale * self.frac(step)

    def spot(self, step: int) -> float:
        return self.spot_scale * self.frac(step)

    def apply(self, randomizer, step: int) -> dict:
        """Set the randomiser's cfg for this step and the spotlight scale (if the setter exists)."""
        randomizer.cfg = scaled_cfg(self.dr(step))
        setter = getattr(randomizer, "set_light_scale", None) or getattr(randomizer, "set_scale", None)
        if setter is not None:
            setter(self.spot(step))
        elif not self._warned and self.spot_scale > 0 and self.mode != "off":
            print("[visual_dr] TwinRandomizer has no spotlight scale setter yet; spot_scale ignored", flush=True)
            self._warned = True
        return {"dr_scale": self.dr(step), "spot_scale": self.spot(step)}
