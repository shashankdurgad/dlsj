"""Tone curve of the table's wood texture: contrast, mean and saturation (plain Python, cv2 / numpy).

    python sim/twin/visuals/wood_tone.py --contrast 1.4 --mean 0.9 --chroma 1.3 [--out path.jpg]

wood_texture.py builds textures/table_top_base.jpg (colours mapped to the top camera on the photo / median overlap).
Rendered under the fitted light rig and the camera model, that texture comes out brighter and flatter than the real
table (luma mean / std, chroma: lighting/metrics.contrast). This applies, in CIE Lab,
    L' = mean * L_mu + contrast * (L - L_mu),   (a', b') = chroma * (a, b)
(L_mu = mean L of the texture) to table_top_base.jpg and writes textures/table_top.jpg (+ the parameters and the new
mean colour in table_top.json), or `--out` (a variant for fit/sweep_looks.py "table_texture_file"). Tested on
held-out top / wrist / inspect table regions and NOT applied: the twin's missing table contrast is low-frequency, not
the wood's (fit/log.md); textures/table_top.jpg is the untoned base.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

TEX = Path(__file__).resolve().parent / "textures"
BASE = TEX / "table_top_base.jpg"


def tone(img_rgb: np.ndarray, contrast: float, mean: float, chroma: float) -> np.ndarray:
    """uint8 sRGB -> toned uint8 sRGB (see the module doc)."""
    lab = cv2.cvtColor(img_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    mu = float(lab[..., 0].mean())
    lab[..., 0] = np.clip(mean * mu + contrast * (lab[..., 0] - mu), 0, 100)
    lab[..., 1:] *= chroma
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--mean", type=float, default=1.0)
    ap.add_argument("--chroma", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="write a variant here instead of textures/table_top.jpg")
    a = ap.parse_args()
    if not BASE.exists():  # first use: the current (untoned) texture becomes the base
        shutil.copyfile(TEX / "table_top.jpg", BASE)
    src = cv2.imread(str(BASE))[..., ::-1]
    out = tone(src, a.contrast, a.mean, a.chroma)
    dst = Path(a.out) if a.out else TEX / "table_top.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not a.out:
        meta_p = TEX / "table_top.json"
        meta = json.loads(meta_p.read_text())
        meta["tone"] = dict(base=BASE.name, contrast=a.contrast, mean=a.mean, chroma=a.chroma,
                            note="wood_tone.py applied to the base texture")
        if "mean_rgb_visible" in meta:
            k = np.array(out.reshape(-1, 3).mean(0) / np.maximum(src.reshape(-1, 3).mean(0), 1e-3))
            meta["mean_rgb_visible"] = [float(v) for v in np.asarray(meta["mean_rgb_visible"]) * k]
        meta_p.write_text(json.dumps(meta, indent=1))
    print("TONE_OK", dst, "mean rgb", out.reshape(-1, 3).mean(0).round(1).tolist())


if __name__ == "__main__":
    main()
