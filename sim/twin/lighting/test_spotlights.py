"""Plain-Python checks of the lighting-DR draw (no Isaac): python sim/twin/lighting/test_spotlights.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sim.twin.lighting.spotlights import LIGHT_DR, draw_lights  # noqa: E402

NOMINAL = [dict(kind="dome", path="/World/Light", irradiance=1.0, rgb=[1, 1, 1]),
           dict(kind="key", path="/World/Sun", dir=[0, 0, 1], irradiance=1.0, rgb=[1, 1, 1]),
           dict(kind="ceiling", path="/World/TwinCeil0", index=0, dir=[-0.4, 0, 0.9], irradiance=1.0, rgb=[1, 1, 1],
                cfg={}),
           dict(kind="side", index=0, dir=[0, -1, 0.3], irradiance=1.0, rgb=[1, 1, 1], cfg={"pos": [0, -5, 2.5]})]


def test_scale_zero_is_nominal():
    """scale 0: unit gains / colours, nominal directions, no spots, no light-off event."""
    d = draw_lights(LIGHT_DR, np.random.default_rng(0), NOMINAL, 2, scale=0.0)
    assert d["exposure"] == 1.0 and d["off"] is None and d["spots"] == [[], []]
    for L, n in zip(d["lights"], NOMINAL, strict=True):
        assert L["gain"] == 1.0 and np.allclose(L["rgb"], 1.0, atol=1e-3)
        if "dir" in L:
            assert np.allclose(L["dir"], np.asarray(n["dir"]) / np.linalg.norm(n["dir"]), atol=1e-3)


def test_spot_count_and_strength_grow_with_scale():
    """spots: 0 .. round(3 s) per env; strength upper bound 0.05 + 1.45 s."""
    for s, nmax in ((0.5, 2), (1.0, 3)):
        draws = [draw_lights(LIGHT_DR, np.random.default_rng(k), NOMINAL, 1, scale=s) for k in range(300)]
        counts = [len(d["spots"][0]) for d in draws]
        assert max(counts) == nmax and min(counts) == 0
        st = [sp["strength"] for d in draws for sp in d["spots"][0]]
        assert max(st) <= 0.05 + 1.45 * s + 1e-6


if __name__ == "__main__":
    test_scale_zero_is_nominal()
    test_spot_count_and_strength_grow_with_scale()
    print("test_spotlights OK")
