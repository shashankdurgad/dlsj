"""Presentation-only scene options (Isaac side; the twin's defaults are untouched):

* `pres_look(table=True, normal=True)`: TwinLook with the cleaned presentation table texture
  (out/presentation/tex/table_pres.jpg from pres_texture.py) instead of visuals/textures/table_top.jpg.
* `add_wire_camera(src="top_cam")`: before TwinEnv is built, adds a second camera sensor `wire_cam` with exactly the
  pose / intrinsics of `src`, so a wireframe pass renders frame-for-frame with the same camera and timeline.
* `set_wireframe(stage, "WireCam", mode)`: RTX wireframe on that camera's render product only (per-render-product
  `omni:rtx:wireframe:mode` = "shaded" | "emissive"; the other cameras keep the normal render).
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PRES_TABLE = REPO / "sim/twin/out/presentation/tex/table_pres2.jpg"


def pres_look(table: bool = True, normal: bool = True, clearcoat: float | None = None):
    from sim.twin.visuals.twin_scene import TwinLook

    look = TwinLook()
    kw = {}
    if table:
        if not PRES_TABLE.exists():
            raise SystemExit(f"{PRES_TABLE} missing: run sim/twin/presentation/pres_texture.py first")
        kw["table_texture"] = str(PRES_TABLE)
    if not normal:
        kw["table_normal"] = False
    if clearcoat is not None:
        kw["table_clearcoat"] = float(clearcoat)
    return dataclasses.replace(look, **kw) if kw else look


def add_wire_camera(src: str = "top_cam", prim: str = "WireCam", width: int = 640, height: int = 480):
    import sim.twin.visuals.twin_scene as ts

    orig = ts.twin_scene_cfg

    def twin_scene_cfg(*a, **k):
        cfg = orig(*a, **k)
        c = copy.deepcopy(getattr(cfg, src))
        c.prim_path = c.prim_path.rsplit("/", 1)[0] + "/" + prim
        c.width, c.height, c.data_types = width, height, ["rgb"]
        c.spawn.vertical_aperture = c.spawn.horizontal_aperture * height / width
        cfg.wire_cam = c
        return cfg

    ts.twin_scene_cfg = twin_scene_cfg


def render_products(stage, cam_name: str) -> list[str]:
    """Render product prims whose camera relationship targets a camera prim named `cam_name`."""
    out = []
    for p in stage.Traverse():
        if p.GetTypeName() != "RenderProduct":
            continue
        rel = p.GetRelationship("camera")
        if rel and any(t.name == cam_name for t in rel.GetTargets()):
            out.append(p.GetPath().pathString)
    return out


def set_wireframe(stage, cam_name: str = "WireCam", mode: str = "shaded", white: bool = False) -> list[str]:
    from pxr import Sdf, Usd

    paths = render_products(stage, cam_name)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in paths:
            p = stage.GetPrimAtPath(path)
            p.CreateAttribute("omni:rtx:wireframe:mode", Sdf.ValueTypeNames.Token).Set(mode)
            p.CreateAttribute("omni:rtx:wireframe:enabled", Sdf.ValueTypeNames.Bool).Set(mode != "instance")
            if white:
                p.CreateAttribute("omni:rtx:debugMaterialType", Sdf.ValueTypeNames.Int).Set(0)
    return paths
