"""Table-top texture from the real top camera (plain Python, no Isaac).

1. per-pixel median of top-camera frames from many episodes (plates move every few dozen episodes, bricks and the
   arm every episode), with strongly saturated pixels (plates, bricks, blue clips) and the black arm masked out where
   enough clean samples exist;
2. back-projection of that median onto the table plane z = TABLE_TOP_Z with the top camera model (pinhole K + pose,
   parametric; default = the joe_cal2 profile's camera, i.e. the camera the sim renders with, so the textured plane
   reprojects onto the real pixels);
3. texels outside the camera frustum are filled by mirroring the visible rectangle.

    docker run ... python sim/twin/visuals/table_texture.py [--every 3] [--profile joe_cal2]
      [--fx 604.7 --fy 603.8 --cx 304.2 --cy 239.0]   # override intrinsics (default: profile f, centred pp)

Writes sim/twin/out/visuals/{top_median.png, table_texture.png, table_texture.json} and the scene's copy
sim/twin/visuals/textures/table_top.{jpg,json}. The json records the plane
extent (env frame) and the camera used; re-run whenever the top camera is re-calibrated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim/residual_env"))
OUT = REPO / "sim/twin/out/visuals"

# table cuboid of sim/residual_env/scene.py: centre (0.25, 0), 1.35 x 1.3 m, top at z = -0.002
TABLE_CENTER = (0.25, 0.0)
TABLE_SIZE = (1.35, 1.3)
TABLE_TOP_Z = -0.002
# textured visual quad (bigger than the physics table so the top camera's whole view lands on it; beyond the real
# table edge it carries the projected background, which is right for the top camera only)
VIS_X = (-0.9, 1.2)
VIS_Y = (-1.0, 1.0)
TEX_DIR = REPO / "sim/twin/visuals/textures"


def camera_from_profile(name: str = "joe_cal2", cam: str = "top"):
    """(R_world_to_cv 3x3, position 3, K 3x3) of a profile camera (top / inspect; sim convention: centred pp)."""
    from calib import load_profile

    p = load_profile(name)
    c = getattr(p, cam)
    rows = np.asarray(c.rows, float)
    r_cv = np.stack([rows[0, :3], -rows[1, :3], -rows[2, :3]])  # USD camera rows -> OpenCV rows
    f = c.f_px
    K = np.array([[f, 0, 320.0], [0, f, 240.0], [0, 0, 1.0]])
    return r_cv, rows[3, :3].copy(), K


def project(pts_w: np.ndarray, r_cv, pos, K) -> tuple[np.ndarray, np.ndarray]:
    """World (env) points Nx3 -> pixel uv Nx2 and depth N."""
    pc = (pts_w - pos) @ np.asarray(r_cv).T
    z = pc[:, 2]
    uv = (pc[:, :2] / np.maximum(z, 1e-6)[:, None]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def backproject(uv: np.ndarray, r_cv, pos, K, z_plane: float) -> np.ndarray:
    """Pixels Nx2 -> intersection with the plane z = z_plane (env frame) Nx3."""
    d_cv = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))]
    d_w = d_cv @ np.asarray(r_cv)  # R^T d
    t = (z_plane - pos[2]) / d_w[:, 2]
    return pos + t[:, None] * d_w


def median_top(ds, every: int, frames=(0, -1)) -> tuple[np.ndarray, np.ndarray]:
    """Masked per-pixel median over top frames; returns (median uint8, plain median uint8)."""
    stack = []
    for ep in range(0, len(ds), every):
        n = int(ds.eps.loc[ep, "length"])
        idxs = sorted({(f if f >= 0 else n + f) for f in frames})
        stack += ds.frames(ep, "top", idxs)
    stack = np.stack(stack)  # N H W 3
    print(f"median over {len(stack)} top frames", flush=True)
    plain = np.median(stack, axis=0).astype(np.uint8)
    hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in stack])
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # saturated plastic (plates, bricks) or off-wood hue (blue plate / bricks), or the black arm
    bad = (sat > 140) | (((h < 5) | (h > 30)) & (sat > 70)) | (val < 45)
    f = stack.astype(np.float32)
    f[bad] = np.nan
    good_n = (~bad).sum(0)
    with np.errstate(all="ignore"):
        import warnings

        warnings.simplefilter("ignore")
        med = np.nanmedian(f, axis=0)
    # keep the plain median for persistent non-wood things (robot base, clips, beyond-the-table background) and
    # where too few clean samples exist
    phsv = cv2.cvtColor(plain, cv2.COLOR_RGB2HSV)
    persistent = (phsv[..., 1] > 120) | (phsv[..., 2] < 70) | (((phsv[..., 0] < 5) | (phsv[..., 0] > 30)) & (phsv[..., 1] > 60))
    persistent = cv2.dilate(persistent.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    use_plain = persistent | (good_n < max(5, len(stack) // 6)) | np.isnan(med).any(-1)
    med[use_plain] = plain[use_plain]
    return np.clip(med, 0, 255).astype(np.uint8), plain


def to_plane(img: np.ndarray, r_cv, pos, K, mm_per_px: float = 1.0):
    """Back-project the image onto the table-top rectangle. Returns (texture, valid mask, extent dict).
    Texture row 0 = +y edge (far side), column 0 = -x edge, so USD uv = ((x - x0) / W, (y - y0) / H)."""
    x0, y0 = VIS_X[0], VIS_Y[0]
    W, H = VIS_X[1] - VIS_X[0], VIS_Y[1] - VIS_Y[0]
    nx, ny = int(round(W * 1000 / mm_per_px)), int(round(H * 1000 / mm_per_px))
    xs = x0 + (np.arange(nx) + 0.5) * W / nx
    ys = y0 + H - (np.arange(ny) + 0.5) * H / ny  # rows go from +y to -y
    gx, gy = np.meshgrid(xs, ys)
    pts = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, TABLE_TOP_Z)]
    uv, z = project(pts, r_cv, pos, K)
    h, w = img.shape[:2]
    mx = uv[:, 0].reshape(ny, nx).astype(np.float32)
    my = uv[:, 1].reshape(ny, nx).astype(np.float32)
    valid = ((z.reshape(ny, nx) > 0) & (mx >= 0.5) & (mx <= w - 1.5) & (my >= 0.5) & (my <= h - 1.5))
    tex = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    tex = fill_outside(tex, valid)
    ext = dict(x0=x0, y0=y0, width=W, height=H, z=TABLE_TOP_Z, nx=nx, ny=ny,
               note="texture row 0 = +y edge, col 0 = -x edge; uv = ((x-x0)/W, (y-y0)/H)")
    return tex, valid, ext


def fill_outside(tex: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill texels outside the camera footprint by mirroring the footprint across its nearest boundary point
    (continues the wood grain), falling back to the nearest visible texel; blend over a few cm at the seam."""
    inv = (~valid).astype(np.uint8)
    if not inv.any():
        return tex
    _, lab = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    vy, vx = np.nonzero(inv == 0)
    # label -> coordinates of that zero pixel (labels enumerate zero pixels in raster order)
    order = np.zeros(lab.max() + 1, np.int64)
    zl = lab[inv == 0]
    order[zl] = np.arange(len(zl))
    ny_, nx_ = vy[order[lab]], vx[order[lab]]
    yy, xx = np.mgrid[:tex.shape[0], :tex.shape[1]]
    my, mx = 2 * ny_ - yy, 2 * nx_ - xx
    H, W = tex.shape[:2]
    ok = (my >= 0) & (my < H) & (mx >= 0) & (mx < W)
    my, mx = np.where(ok, my, ny_).clip(0, H - 1), np.where(ok, mx, nx_).clip(0, W - 1)
    ok2 = valid[my, mx]
    my, mx = np.where(ok2, my, ny_), np.where(ok2, mx, nx_)
    out = tex.copy()
    out[~valid] = tex[my, mx][~valid]
    out = np.where(valid[..., None], tex, cv2.GaussianBlur(out, (0, 0), 1.5))
    return out


def median_cam(ds, cam: str, every: int, frame: int = 0) -> np.ndarray:
    stack = np.stack([ds.frame(ep, cam, frame) for ep in range(0, len(ds), every)])
    print(f"{cam}: median over {len(stack)} frames", flush=True)
    return np.median(stack, axis=0).astype(np.uint8)


def backdrop(img: np.ndarray, r_cv, pos, K, edge_row: float, width: float = 4.0, height: float = 0.45,
             mm_per_px: float = 2.0):
    """Vertical billboard standing on the table where the image row `edge_row` (centre column) meets the table plane,
    facing the camera (normal = horizontal backward camera axis), textured by projecting `img` from the camera (exact
    for that camera; edge-clamped outside its view). Returns (texture, plane dict: origin = bottom-left corner,
    u axis (width), v axis (up))."""
    p_edge = backproject(np.array([[K[0, 2], edge_row]]), r_cv, pos, K, TABLE_TOP_Z)[0]
    fwd = np.asarray(r_cv)[2].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    u_ax = np.cross(fwd, up)  # camera right (horizontal)
    origin = p_edge - u_ax * width / 2
    nx, ny = int(width * 1000 / mm_per_px), int(height * 1000 / mm_per_px)
    s = (np.arange(nx) + 0.5) / nx * width
    t = height - (np.arange(ny) + 0.5) / ny * height  # row 0 = top
    gs, gt = np.meshgrid(s, t)
    pts = origin + gs.reshape(-1, 1) * u_ax + gt.reshape(-1, 1) * up
    uv, z = project(pts, r_cv, pos, K)
    mx = uv[:, 0].reshape(ny, nx).astype(np.float32)
    my = uv[:, 1].reshape(ny, nx).astype(np.float32)
    h, w = img.shape[:2]
    tex = cv2.remap(img, mx.clip(0, w - 1), my.clip(0, h - 1), cv2.INTER_LINEAR)
    plane = dict(origin=origin.tolist(), u_axis=u_ax.tolist(), v_axis=up.tolist(), width=width, height=height,
                 edge_row=edge_row, note="texture row 0 = top (v = height), col 0 = origin side")
    return tex, plane


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3, help="use every n-th episode")
    ap.add_argument("--profile", default="joe_cal2")
    ap.add_argument("--fx", type=float)
    ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    ap.add_argument("--mm-per-px", type=float, default=1.0)
    ap.add_argument("--reuse-median", action="store_true")
    ap.add_argument("--inspect-edge-row", type=float, default=250.0,
                    help="row of the real inspect image where the table meets the background objects (laptop base); "
                         "250 puts the 0.45 m backdrop ~0.75 m from the inspect camera, outside the top camera view")
    a = ap.parse_args()
    from sim.twin.visuals.realdata import Dataset

    OUT.mkdir(parents=True, exist_ok=True)
    r_cv, pos, K = camera_from_profile(a.profile)
    for i, k in ((0, "fx"), (1, "fy")):
        if getattr(a, k):
            K[i, i] = getattr(a, k)
    if a.cx:
        K[0, 2] = a.cx
    if a.cy:
        K[1, 2] = a.cy
    mp = OUT / "top_median.png"
    if a.reuse_median and mp.exists():
        med = cv2.imread(str(mp))[..., ::-1]
    else:
        med, plain = median_top(Dataset(), a.every)
        cv2.imwrite(str(mp), med[..., ::-1])
        cv2.imwrite(str(OUT / "top_median_plain.png"), plain[..., ::-1])
    tex, valid, ext = to_plane(med, r_cv, pos, K, a.mm_per_px)
    cv2.imwrite(str(OUT / "table_texture.png"), tex[..., ::-1])
    cv2.imwrite(str(OUT / "table_texture_valid.png"), (valid * 255).astype(np.uint8))
    ext.update(camera=dict(profile=a.profile, R_world_to_cv=r_cv.tolist(), position=pos.tolist(), K=K.tolist()),
               mean_rgb_visible=[float(v) for v in tex[valid].mean(0)])
    (OUT / "table_texture.json").write_text(json.dumps(ext, indent=1))
    TEX_DIR.mkdir(parents=True, exist_ok=True)  # the copy the scene uses (small JPEG, committed)
    cv2.imwrite(str(TEX_DIR / "table_top.jpg"), tex[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    (TEX_DIR / "table_top.json").write_text(json.dumps(ext, indent=1))
    # inspect-camera backdrop (background objects / people behind the table, projection-mapped)
    ip = OUT / "inspect_median.png"
    if a.reuse_median and ip.exists():
        imed = cv2.imread(str(ip))[..., ::-1]
    else:
        imed = median_cam(Dataset(), "inspect", max(1, a.every))
        cv2.imwrite(str(ip), imed[..., ::-1])
    ir, ipos, iK = camera_from_profile(a.profile, "inspect")
    btex, plane = backdrop(imed, ir, ipos, iK, a.inspect_edge_row)
    plane.update(camera=dict(profile=a.profile, cam="inspect"))
    cv2.imwrite(str(TEX_DIR / "backdrop_inspect.jpg"), btex[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    (TEX_DIR / "backdrop_inspect.json").write_text(json.dumps(plane, indent=1))
    print("backdrop", btex.shape, plane["origin"])
    print("texture", tex.shape, "visible", round(float(valid.mean()), 3), "->", OUT / "table_texture.png")


if __name__ == "__main__":
    main()
