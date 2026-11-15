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
                     specular: float = 0.5, linear: bool = False, normal_texture: str | None = None,
                     emissive: bool = False, wrap: str = "clamp"):
    """Define (or redefine) a UsdPreviewSurface material at `path`. `texture`: optional diffuse image file, read
    with primvars:st (sRGB); its colour is multiplied by `rgb255` (use (255, 255, 255) for the plain texture).
    `normal_texture`: optional tangent-space normal map (same st). `emissive`: the colour / texture goes to
    emissiveColor on a black diffuse surface (unlit: backdrops whose brightness must not follow the scene lights).
    `specular` is the dielectric specular level (UsdPreviewSurface: ior, 0.5 = ior 1.5). `wrap`: clamp | repeat."""
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
    f0 = 0.08 * float(specular)  # F0 = ((ior - 1) / (ior + 1))^2
    sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set((1 + f0 ** 0.5) / max(1 - f0 ** 0.5, 1e-3))
    target = "emissiveColor" if emissive else "diffuseColor"
    if emissive:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    if texture or normal_texture:
        st = UsdShade.Shader.Define(stage, path + "/St")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    if texture:
        tx = UsdShade.Shader.Define(stage, path + "/Tex")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap)
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp" if emissive else wrap)
        tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*col, 1.0))
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput(target, Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput(target, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col))
    if normal_texture:
        nx = UsdShade.Shader.Define(stage, path + "/Normal")
        nx.CreateIdAttr("UsdUVTexture")
        nx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(normal_texture))
        nx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        nx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap)
        nx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(wrap)
        nx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.0, 2.0, 2.0, 1.0))
        nx.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.0, -1.0, -1.0, 0.0))
        nx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        nx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(nx.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, material, stronger: bool = True):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.strongerThanDescendants if stronger else UsdShade.Tokens.weakerThanDescendants)
