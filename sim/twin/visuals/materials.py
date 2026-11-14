"""UsdPreviewSurface material helpers (pxr only; import after the Kit app is up).

Colours are given as sRGB 0-255 as a camera sees them (that is what we sample from the real frames) and converted to
the linear values UsdPreviewSurface expects. With the twin's lighting calibrated so a white Lambertian surface renders
at ~255 (twin_scene.calibrate_exposure), a material renders at roughly the sRGB colour it was given.
"""
from __future__ import annotations

import numpy as np


def srgb_to_linear(rgb255) -> tuple[float, float, float]:
    c = np.clip(np.asarray(rgb255, float) / 255.0, 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return tuple(float(v) for v in lin)


def preview_material(stage, path: str, rgb255, roughness: float = 0.5, metallic: float = 0.0,
                     clearcoat: float = 0.0, clearcoat_roughness: float = 0.05, texture: str | None = None,
                     specular: float = 0.5, linear: bool = False):
    """Define (or redefine) a UsdPreviewSurface material at `path`. `texture`: optional diffuse image file, read
    with primvars:st (sRGB); its colour is multiplied by `rgb255` (use (255, 255, 255) for the plain texture)."""
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    col = tuple(float(v) for v in rgb255) if linear else srgb_to_linear(rgb255)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(float(clearcoat))
    sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(float(clearcoat_roughness))
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
    if texture:
        st = UsdShade.Shader.Define(stage, path + "/St")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, path + "/Tex")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*col, 1.0))
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, material, stronger: bool = True):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.strongerThanDescendants if stronger else UsdShade.Tokens.weakerThanDescendants)
