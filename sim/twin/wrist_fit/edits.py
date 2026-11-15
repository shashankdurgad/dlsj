"""Image-space edits of the wrist frame for the wrist-gap diagnosis (numpy + cv2 only; no Isaac, no PyAV).

An edit is `f(img, ctx) -> img` on one HxWx3 uint8 RGB wrist frame. ctx carries the sample's other frames and masks:
    ctx["real"]  the recorded wrist frame            ctx["sim"]   the twin's wrist render at the same state
    ctx["mask"]  the twin's region bits for the sim wrist frame (audit BITS: table 1, plates 2, bricks 4, robot 8)
    ctx["s"]     the audit-set sample dict           ctx["lay"]   the episode layout (target colour etc.)
Variant spec strings: "<base>:<edit>+<edit>..." with base sim | real | simw (real top + sim wrist) | simt (sim top +
real wrist); the edits apply to the wrist image of the base.
"""
from __future__ import annotations

import cv2
import numpy as np

BITS = {"table": 1, "plates": 2, "bricks": 4, "robot": 8, "props": 16}
# real target-brick colours (sRGB as the top camera records them; visuals/twin_scene.py BRICK_RGB)
PALETTE = {"light_blue": (5, 138, 202), "yellow": (155, 173, 86), "orange": (166, 85, 41), "red": (124, 24, 43),
           "brown": (110, 60, 37), "cream": (205, 200, 160)}


def lab(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)


def rgb(lab_img):
    return cv2.cvtColor(np.clip(lab_img, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def region(ctx, name) -> np.ndarray:
    return (ctx["mask"] & BITS[name]) > 0


def real_fins(real: np.ndarray) -> np.ndarray:
    """Red TPU fins in a real wrist frame: saturated red, connected to the bottom border."""
    hsv = cv2.cvtColor(real, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    m = (((h < 9) | (h > 168)) & (s > 110) & (v > 50)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m)
    H = real.shape[0]
    keep = np.zeros_like(m, bool)
    for k in range(1, n):
        x, y, w, hh, a = st[k]
        if y + hh >= H - 3 and a > 400:
            keep |= lbl == k
    return cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


# ------------------------------------------------------------------ global edits
def sat(f):
    def e(img, ctx):
        x = lab(img)
        x[..., 1:] = 128 + (x[..., 1:] - 128) * f
        return rgb(x)
    return e


def contrast(f):
    def e(img, ctx):
        x = lab(img)
        m = x[..., 0].mean()
        x[..., 0] = m + (x[..., 0] - m) * f
        return rgb(x)
    return e


def gain(f):
    return lambda img, ctx: np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


def blur(sig):
    return lambda img, ctx: cv2.GaussianBlur(img, (0, 0), sig)


def noise(sig):
    def e(img, ctx):
        rng = np.random.default_rng(int(ctx["s"]["i"]))
        return np.clip(img + rng.normal(0, sig, img.shape), 0, 255).astype(np.uint8)
    return e


def highlights(amount):
    """Specular-like boost of the brightest table / plate pixels (real wrist highlights are brighter)."""
    def e(img, ctx):
        x = lab(img)
        L = x[..., 0]
        thr = np.percentile(L, 90)
        w = np.clip((L - thr) / max(255 - thr, 1), 0, 1)
        x[..., 0] = L + amount * w * 255 / 100
        return rgb(x)
    return e


def match_global(ctx_key="real"):
    """Per-channel LAB mean/std of the whole frame matched to the real frame (Reinhard)."""
    def e(img, ctx):
        a, b = lab(img), lab(ctx[ctx_key])
        a = (a - a.mean((0, 1))) / (a.std((0, 1)) + 1e-6) * b.std((0, 1)) + b.mean((0, 1))
        return rgb(a)
    return e


# ------------------------------------------------------------------ region edits
def region_match(name, use_real_mask=False):
    """LAB mean/std of a sim region matched to the same (sim-geometry) region of the real frame."""
    def e(img, ctx):
        m = region(ctx, name)
        if m.sum() < 50:
            return img
        a, b = lab(img), lab(ctx["real"])
        src, ref = a[m], b[m]
        a[m] = (src - src.mean(0)) / (src.std(0) + 1e-6) * ref.std(0) + ref.mean(0)
        return rgb(a)
    return e


def region_paste(name):
    """Real pixels pasted into the sim region (geometry of the sim, appearance of the real frame there)."""
    def e(img, ctx):
        m = region(ctx, name)
        out = img.copy()
        out[m] = ctx["real"][m]
        return out
    return e


def brick_palette(L_shift=0.0, chroma=1.0):
    """Target + distractor brick pixels recoloured to the real palette colour of the target (keeps sim shading)."""
    def e(img, ctx):
        m = region(ctx, "bricks")
        if m.sum() < 50:
            return img
        col = ((ctx["lay"] or {}).get("target") or {}).get("color") or "light_blue"
        pc = lab(np.array([[PALETTE.get(col, PALETTE["light_blue"])]], np.uint8))[0, 0]
        x = lab(img)
        Lm = x[..., 0][m]
        x[..., 0][m] = Lm - Lm.mean() + pc[0] + L_shift
        x[..., 1][m] = 128 + (pc[1] - 128) * chroma
        x[..., 2][m] = 128 + (pc[2] - 128) * chroma
        return rgb(x)
    return e


def brick_boost(dL=8.0, chroma=1.5):
    def e(img, ctx):
        m = region(ctx, "bricks")
        x = lab(img)
        x[..., 0][m] += dL
        for c in (1, 2):
            x[..., c][m] = 128 + (x[..., c][m] - 128) * chroma
        return rgb(x)
    return e


def fins_real():
    """Sim fins removed (inpainted) and the real fins pasted in at their real image position."""
    def e(img, ctx):
        sm = cv2.dilate(region(ctx, "robot").astype(np.uint8), np.ones((7, 7), np.uint8))
        out = cv2.inpaint(img, sm, 5, cv2.INPAINT_TELEA)
        rm = real_fins(ctx["real"])
        out[rm] = ctx["real"][rm]
        return out
    return e


def fins_remove():
    def e(img, ctx):
        sm = cv2.dilate(region(ctx, "robot").astype(np.uint8), np.ones((7, 7), np.uint8))
        return cv2.inpaint(img, sm, 5, cv2.INPAINT_TELEA)
    return e


def fins_sim_into_real():
    """Reverse test on REAL frames: real fins inpainted away, the sim fins pasted at their sim position."""
    def e(img, ctx):
        rm = real_fins(img)
        out = cv2.inpaint(img, rm.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        sm = region(ctx, "robot")
        out[sm] = ctx["sim"][sm]
        return out
    return e


def fins_color():
    """Sim fin (robot region) colour matched to the real fin pixels' LAB stats (geometry unchanged)."""
    def e(img, ctx):
        sm = region(ctx, "robot")
        rm = real_fins(ctx["real"])
        if sm.sum() < 50 or rm.sum() < 50:
            return img
        a, b = lab(img), lab(ctx["real"])
        src, ref = a[sm], b[rm]
        a[sm] = (src - src.mean(0)) / (src.std(0) + 1e-6) * ref.std(0) + ref.mean(0)
        return rgb(a)
    return e


def shift(dx, dy):
    return lambda img, ctx: cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (img.shape[1], img.shape[0]),
                                           borderMode=cv2.BORDER_REFLECT)


def zoom_scene(f, cx=320.0, cy=240.0):
    """The scene (everything but the robot) magnified by f about (cx, cy), the sim fins kept in place: a camera
    closer to the table by a factor 1/f at the same fins (the real gripper sits lower at the same joint state)."""
    def e(img, ctx):
        rob = cv2.dilate(region(ctx, "robot").astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        scene = cv2.inpaint(img, rob.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        M = np.float32([[f, 0, cx * (1 - f)], [0, f, cy * (1 - f)]])
        out = cv2.warpAffine(scene, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        fin = region(ctx, "robot")
        out[fin] = img[fin]
        return out
    return e


EDITS = {
    "none": lambda img, ctx: img,
    "sat14": sat(1.4), "sat18": sat(1.8), "sat06": sat(0.6),
    "con13": contrast(1.3), "con07": contrast(0.7),
    "gain115": gain(1.15), "gain085": gain(0.85),
    "blur15": blur(1.5), "noise4": noise(4.0),
    "hl20": highlights(20.0),
    "match": match_global(),
    "bricks_match": region_match("bricks"), "table_match": region_match("table"),
    "robot_match": region_match("robot"), "plates_match": region_match("plates"),
    "bricks_paste": region_paste("bricks"), "table_paste": region_paste("table"),
    "plates_paste": region_paste("plates"), "robot_paste": region_paste("robot"),
    "brick_pal": brick_palette(), "brick_pal_dark": brick_palette(L_shift=-15.0),
    "brick_pal_grey": brick_palette(chroma=0.3),
    "brick_boost": brick_boost(),
    "fins_real": fins_real(), "fins_remove": fins_remove(), "fins_sim": fins_sim_into_real(),
    "fins_color": fins_color(),
    "zoom110": zoom_scene(1.1), "zoom120": zoom_scene(1.2), "zoom130": zoom_scene(1.3),
    "zoom120b": zoom_scene(1.2, 320.0, 420.0), "zoom090": zoom_scene(0.9),
    "sx20": shift(20, 0), "sx-20": shift(-20, 0), "sy20": shift(0, 20), "sy-20": shift(0, -20),
}


def apply(spec_edits: str, img, ctx):
    for name in [x for x in spec_edits.split("+") if x]:
        img = EDITS[name](img, ctx)
    return img
