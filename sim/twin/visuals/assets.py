"""Procedural visual meshes for the twin: Duplo bricks with studs, round plates (shallow dish with rim and faint
concentric rings), and the textured table-top quad. Mesh builders are pure numpy (points, normals, face counts,
indices); `define_mesh` writes a UsdGeom.Mesh (pxr, import after the Kit app is up).

Dimensions
* Duplo (LEGO DUPLO = 2x LEGO System in plan): stud pitch 16 mm, 2x2 brick 31.7 x 31.7 mm, body height 19.2 mm,
  studs 9.4 mm diameter x 4.4 mm high (hollow top recess on real studs is ignored). The residual env's colliders are
  boxes of exactly the body size (31.8 x 31.8 x 19.2 mm / 63.6 x 31.8 x 19.2 mm) without studs, so the visual body
  matches the collider and the studs stick out on top (+z of the brick).
* Brick variants on the real table (sim/twin/data/photos/table_topdown.jpg): 2x2 (LDraw 3437), 2x4 (3011) and the
  curved brick, which is the Duplo "Brick 2 x 3 with Curved Top" (LDraw 2302b: 2 x 2 studs on the flat part, a
  stud-less quarter-round end, 47.7 x 31.7 x 19.2 mm body). `brick_mesh(kind)` returns the LDraw geometry converted by
  sim/twin/scene/ldraw.py (hollow studs, underside walls and tubes, 0.4 mm rounded outer edges on the box bricks,
  crease-angle normals; real size = LDraw nominal minus 0.15 mm play per side: 2x2 31.7 mm, 2x4 63.7 mm), falling back
  to the procedural meshes below when the converted .npz files are missing. `brick_hull_points(kind)` is the matching
  convex collider without studs (boxes for 2x2 / 2x4, the convex body of the curved brick). "curved_2x2" is kept as an
  alias of "curved_2x3" (the earlier name; there is no curved 2x2 on the station).
* Plates: user's measurement (14 Nov): radius 9.2 cm (0.184 m outer diameter; the top camera gave 0.182-0.189 m),
  total height ~23 mm, ~2 mm plastic. Profile: flat floor (faint concentric ripples, the photo shows them over ~3/4
  of the radius) whose top is 3 mm above the table (2 mm shell on a ~1 mm foot: assumed), sloping wall up to a
  rounded rim 23 mm above the table. z is in the plate frame of the residual env's bowl bodies (origin 3 mm above
  the table), so the floor top is at z = 0. The same profile is the plates' physics collider (sim/twin/physics:
  triangle mesh on the kinematic plate body; the env's flat 6 mm cylinder collider is disabled there).
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

LDRAW_MESHES = Path(__file__).resolve().parents[1] / "scene/assets"  # duplo_<kind>.npz from scene/ldraw.py

DUPLO_PITCH = 0.016
DUPLO_BODY_H = 0.0192
DUPLO_STUD_D = 0.0094
DUPLO_STUD_H = 0.0044
DUPLO_2X2 = (0.0317, 0.0317, DUPLO_BODY_H)
DUPLO_2X4 = (0.0636, 0.0317, DUPLO_BODY_H)
CURVED_LOW_H = 0.0064  # curved-slope 2x2: height of the rounded front edge (estimated from the photo)
PLATE_RADIUS = 0.092  # m, outer (user's measurement)
PLATE_RIM_H = 0.0238  # m above the table (total height; user 23 mm, inspect-silhouette fit 23.8 mm)
PLATE_WALL = 0.002  # m plastic thickness
PLATE_FLOOR_H = 0.003  # m, floor top above the table (assumed: 2 mm shell + ~1 mm foot)
PLATE_FLOOR_FRAC = 0.75  # flat floor out to this fraction of the radius (photo: rings over ~3/4; fit 0.748)
PLATE_WALL_POW = 3.0  # wall rise t ** p from the floor edge to the lip (inspect-silhouette fit, scene/plate_fit.py)
PLATE_FOOT_W = 0.004  # m, width of the foot ring the plate stands on
PLATE_VIS_RING_AMP = 0.0001  # m, ripple amplitude of the VISUAL plate's floor (plate_mesh; the collider profile
# keeps plate_profile's default 0.25 mm): rings barely visible from the top camera, as in the real frames (fit/log.md)


# ----------------------------------------------------------------------------------------------- mesh builders
class MeshBuf:
    def __init__(self):
        self.p, self.n, self.c, self.i = [], [], [], []
        self.nv = 0

    def add(self, pts, nrm, faces):
        pts, nrm = np.asarray(pts, float), np.asarray(nrm, float)
        for f in faces:
            self.c.append(len(f))
            self.i.extend(int(v) + self.nv for v in f)
        self.p.append(pts)
        self.n.append(nrm / np.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-9))
        self.nv += len(pts)

    def arrays(self):
        return np.concatenate(self.p), np.concatenate(self.n), np.array(self.c), np.array(self.i)


def box_mesh(size, center=(0, 0, 0), bottom=True) -> tuple:
    """Axis-aligned box with per-face normals (24 vertices)."""
    hx, hy, hz = (s / 2 for s in size)
    cx, cy, cz = center
    buf = MeshBuf()
    faces = [((1, 0, 0), [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
             ((-1, 0, 0), [(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)]),
             ((0, 1, 0), [(hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)]),
             ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
             ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)])]
    if bottom:
        faces.append(((0, 0, -1), [(-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz)]))
    for n, quad in faces:
        buf.add(np.array(quad) + [cx, cy, cz], [n] * 4, [[0, 1, 2, 3]])
    return buf.arrays()


def _merge(*meshes):
    buf = MeshBuf()
    for p, n, c, i in meshes:
        faces, k = [], 0
        for cnt in c:
            faces.append(i[k:k + cnt])
            k += cnt
        buf.add(p, n, faces)
    return buf.arrays()


def cylinder_mesh(r, h, center=(0, 0, 0), seg=24, bevel=0.0006) -> tuple:
    """Vertical cylinder (side + top cap with a small bevel ring), open bottom."""
    cx, cy, cz = center
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    buf = MeshBuf()
    z0, z1 = cz - h / 2, cz + h / 2 - bevel
    # side
    pts = np.r_[np.c_[cx + r * ca, cy + r * sa, np.full(seg, z0)], np.c_[cx + r * ca, cy + r * sa, np.full(seg, z1)]]
    nrm = np.r_[np.c_[ca, sa, np.zeros(seg)], np.c_[ca, sa, np.zeros(seg)]]
    buf.add(pts, nrm, [[k, (k + 1) % seg, seg + (k + 1) % seg, seg + k] for k in range(seg)])
    # bevel ring
    rb = r - bevel
    pts = np.r_[np.c_[cx + r * ca, cy + r * sa, np.full(seg, z1)],
                np.c_[cx + rb * ca, cy + rb * sa, np.full(seg, z1 + bevel)]]
    nb = np.c_[ca, sa, np.ones(seg)]
    buf.add(pts, np.r_[nb, nb], [[k, (k + 1) % seg, seg + (k + 1) % seg, seg + k] for k in range(seg)])
    # top cap (fan)
    pts = np.r_[[[cx, cy, z1 + bevel]], np.c_[cx + rb * ca, cy + rb * sa, np.full(seg, z1 + bevel)]]
    buf.add(pts, np.tile([0, 0, 1.0], (seg + 1, 1)), [[0, 1 + k, 1 + (k + 1) % seg] for k in range(seg)])
    return buf.arrays()


def duplo_mesh(nx: int, ny: int, size=None, stud_seg: int = 24) -> tuple:
    """Duplo brick nx x ny studs, centred on its body centre (the collider's centre); studs on +z. Known Duplo
    footprints use the LDraw geometry (2x2, 4x2 = 2x4, 3x2 = the curved 2x3; 2x4 / 2x3 along y are rotated), others
    the procedural box + studs."""
    key = {(2, 2): "2x2", (4, 2): "2x4", (2, 4): "2x4", (3, 2): "curved_2x3", (2, 3): "curved_2x3"}.get((nx, ny))
    if key is not None and _ldraw(key) is not None:
        sz = None if size is None else (tuple(size) if nx >= ny else (size[1], size[0], size[2]))
        p, n, c, i = ldraw_mesh(key, sz)
        if ny > nx:  # long side along y: rotate +90 deg about z
            p = np.c_[-p[:, 1], p[:, 0], p[:, 2]]
            n = np.c_[-n[:, 1], n[:, 0], n[:, 2]]
        return p, n, c, i
    return _procedural_duplo(nx, ny, size, stud_seg)


def _procedural_duplo(nx: int, ny: int, size=None, stud_seg: int = 24) -> tuple:
    sx, sy, sz = size if size is not None else (nx * DUPLO_PITCH - 0.0003, ny * DUPLO_PITCH - 0.0003, DUPLO_BODY_H)
    parts = [box_mesh((sx, sy, sz))]
    px, py = sx / nx, sy / ny
    for i in range(nx):
        for j in range(ny):
            x = -sx / 2 + px * (i + 0.5)
            y = -sy / 2 + py * (j + 0.5)
            parts.append(cylinder_mesh(DUPLO_STUD_D / 2, DUPLO_STUD_H, (x, y, sz / 2 + DUPLO_STUD_H / 2), stud_seg))
    return _merge(*parts)

def curved_slope_mesh(size=DUPLO_2X2, low_h: float = CURVED_LOW_H, stud_seg: int = 24, n_arc: int = 12) -> tuple:
    """Duplo curved-slope 2x2, centred on the body-box centre (same frame as a 2x2 box collider): the -y half keeps the
    full height with one row of two studs, the +y half rounds down (quarter ellipse) to `low_h` at the +y edge."""
    sx, sy, sz = size
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    prof = curved_slope_profile(size, low_h, n_arc)  # (y, z) over the top from -y to +y
    buf = MeshBuf()
    m = len(prof)
    # top surface strip (quads between consecutive profile points, x from -hx to +hx)
    for k in range(m - 1):
        (y0, z0), (y1, z1) = prof[k], prof[k + 1]
        n = np.array([0.0, -(z1 - z0), (y1 - y0)])
        n = n / max(np.linalg.norm(n), 1e-12)
        buf.add([(-hx, y0, z0), (hx, y0, z0), (hx, y1, z1), (-hx, y1, z1)], [n] * 4, [[0, 1, 2, 3]])
    # side walls (x = +-hx): polygon fan of the profile closed along the bottom
    for sgn in (-1.0, 1.0):
        pts = [(sgn * hx, y, z) for y, z in prof] + [(sgn * hx, hy, -hz), (sgn * hx, -hy, -hz)]
        idx = list(range(len(pts)))
        buf.add(pts, [(sgn, 0.0, 0.0)] * len(pts), [idx if sgn > 0 else idx[::-1]])
    # back (-y), front (+y, low), bottom
    buf.add([(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (hx, -hy, hz)], [(0, -1.0, 0)] * 4, [[0, 1, 2, 3]])
    zf = -hz + low_h
    buf.add([(-hx, hy, -hz), (hx, hy, -hz), (hx, hy, zf), (-hx, hy, zf)], [(0, 1.0, 0)] * 4, [[0, 1, 2, 3]])
    buf.add([(-hx, hy, -hz), (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz)], [(0, 0, -1.0)] * 4, [[0, 1, 2, 3]])
    parts = [buf.arrays()]
    for i in range(2):  # one stud row on the flat (-y) half
        x = -hx + sx / 2 * (i + 0.5)
        parts.append(cylinder_mesh(DUPLO_STUD_D / 2, DUPLO_STUD_H, (x, -hy + sy / 4, hz + DUPLO_STUD_H / 2), stud_seg))
    return _merge(*parts)


def curved_slope_profile(size=DUPLO_2X2, low_h: float = CURVED_LOW_H, n_arc: int = 12) -> np.ndarray:
    """(y, z) top profile of the curved-slope 2x2 in the body-centred frame: flat to y = 0, then a quarter ellipse
    down to -hz + low_h at y = +hy."""
    _, sy, sz = size
    hy, hz = sy / 2, sz / 2
    a = np.linspace(0.0, math.pi / 2, n_arc)
    ys = hy * np.sin(a)
    zs = -hz + low_h + (sz - low_h) * np.cos(a)
    return np.r_[[[-hy, hz]], np.c_[ys, zs]]


DUPLO_CURVED_2X3 = (0.0477, 0.0317, DUPLO_BODY_H)
BRICK_KINDS = {"2x2": DUPLO_2X2, "2x4": DUPLO_2X4, "curved_2x3": DUPLO_CURVED_2X3, "curved_2x2": DUPLO_CURVED_2X3}
KIND_ALIASES = {"curved_2x2": "curved_2x3"}


def brick_kind_from_size(size) -> str:
    """'2x2' | '2x4' | 'curved_2x3' from a body size (m): the long side decides (2x2 32, 2x3 48, 2x4 64 mm)."""
    long_ = max(size[0], size[1])
    return "2x2" if long_ < 0.040 else ("curved_2x3" if long_ < 0.056 else "2x4")


@lru_cache(maxsize=8)
def _ldraw(kind: str):
    kind = KIND_ALIASES.get(kind, kind)
    p = LDRAW_MESHES / f"duplo_{kind}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def ldraw_mesh(kind: str, size=None):
    """(points, normals, face counts, indices) of the converted LDraw brick, or None. With `size` (body x, y, z, m)
    that differs from the real body by more than 1 mm the mesh is scaled per axis (else used as is)."""
    m = _ldraw(kind)
    if m is None:
        return None
    p, n = m["points"].astype(float), m["normals"].astype(float)
    if size is not None:
        s = np.asarray(size, float) / m["body_size"]
        if np.any(np.abs(np.asarray(size) - m["body_size"]) > 0.001):
            p = p * s
            n = n / s
            n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    f = m["faces"]
    return p, n, np.full(len(f), 3, np.int32), f.reshape(-1).astype(np.int32)


def brick_mesh(kind: str = "2x2", size=None, source: str = "ldraw") -> tuple:
    """Visual mesh of a brick variant ('2x2' | '2x4' | 'curved_2x3'), centred on its body-box centre, studs on +z,
    long side along x. source='ldraw' (default; falls back to 'procedural' if the .npz files are missing)."""
    if source == "ldraw":
        m = ldraw_mesh(kind, size)
        if m is not None:
            return m
    size = tuple(size) if size is not None else BRICK_KINDS[kind]
    if KIND_ALIASES.get(kind, kind) == "curved_2x3":
        return curved_slope_mesh(size)
    nx = max(1, round(size[0] / DUPLO_PITCH))
    ny = max(1, round(size[1] / DUPLO_PITCH))
    return _procedural_duplo(nx, ny, size=size)


def brick_hull_points(kind: str = "2x2", size=None) -> np.ndarray:
    """Vertices of the brick's convex collider (studs excluded, as for the box colliders): the body box, or for the
    curved brick the convex hull of its LDraw body."""
    sx, sy, sz = tuple(size) if size is not None else BRICK_KINDS[kind]
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    if KIND_ALIASES.get(kind, kind) == "curved_2x3":
        m = _ldraw("curved_2x3")
        if m is not None:
            return m["hull"].astype(float) * (np.array([sx, sy, sz]) / m["body_size"])
        prof = curved_slope_profile((sx, sy, sz))
        yz = np.r_[prof, [[hy, -hz], [-hy, -hz]]]
        return np.r_[np.c_[np.full(len(yz), -hx), yz], np.c_[np.full(len(yz), hx), yz]]
    return np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)])


def _offset_polyline(P: np.ndarray, d: float) -> np.ndarray:
    """Offset an (r, z) polyline by d to its right-hand side (outward-down for a centre -> rim top surface)."""
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True).clip(1e-12)
    return P + d * np.c_[t[:, 1], -t[:, 0]]


def plate_profile(radius: float = PLATE_RADIUS, floor_z: float = 0.0, rim_h: float = PLATE_RIM_H,
                  wall: float = PLATE_WALL, floor_frac: float = PLATE_FLOOR_FRAC, rings: int = 7,
                  ring_amp: float = 0.00025, n: int = 90, floor_h: float = PLATE_FLOOR_H,
                  wall_pow: float = PLATE_WALL_POW, foot_w: float = PLATE_FOOT_W):
    """(r, z) polyline of the plate's cross-section, going over the top surface from the centre to the rim lip and
    back underneath. z is relative to the plate frame (the env's bowl body origin, 3 mm above the table): floor top
    at z = floor_z, table at z = floor_z - floor_h, rim top at table + rim_h.

    Top surface: flat floor (faint concentric ripples) out to floor_frac * radius, then the wall rises as
    t ** wall_pow (t = 0 at the floor edge, 1 at the lip; wall_pow > 1 leaves the floor tangentially and steepens
    towards the rim), then a rounded lip of diameter 1.5 * wall. Underside: the top surface offset by the shell
    thickness `wall` down to the foot ring (at floor_h - wall above the table), a foot ring of width foot_w standing
    on the table, and the flat underside of the floor inside it."""
    table = floor_z - floor_h
    lip_r = wall * 0.75
    top_rim = table + rim_h - lip_r  # centre height of the rounded lip
    r_in = radius - lip_r  # lip centre radius (outer edge of the lip = radius)
    rf = radius * floor_frac
    m = max(n // 2, 8)
    floor = [(r, floor_z + ring_amp * math.sin(2 * math.pi * rings * r / rf) * (r / rf))
             for r in np.linspace(0.0, rf, m)]
    t = np.linspace(0, 1, m)[1:]
    wall_pts = np.c_[rf + (r_in - lip_r - rf) * t, floor_z + (top_rim - floor_z) * t ** wall_pow]
    lip = [(r_in + lip_r * math.cos(a), top_rim + lip_r * math.sin(a)) for a in np.linspace(math.pi, 0, 9)[1:]]
    top = np.r_[np.asarray(floor), wall_pts, np.asarray(lip)]
    # outer (under) side of the wall: the top surface (floor edge -> lip start) offset by the shell thickness
    inner = np.r_[np.c_[np.linspace(rf * 0.75, rf, m // 4), np.full(m // 4, floor_z)][:-1], wall_pts]
    outer = _offset_polyline(inner, wall)[::-1]
    foot_z = table + max(floor_h - wall, 0.0005)
    outer = outer[(outer[:, 1] > foot_z + 1e-5) & (outer[:, 0] <= radius)]
    r_foot = float(outer[-1, 0]) if len(outer) else rf
    under = [(r_foot, table), (max(r_foot - foot_w, 0.0), table), (max(r_foot - foot_w, 0.0), foot_z), (0.0, foot_z)]
    prof = np.r_[top, [(radius, top_rim)], outer, np.asarray(under)]
    keep = np.r_[True, np.linalg.norm(np.diff(prof, axis=0), axis=1) > 1e-9]  # no zero-length segments
    return prof[keep]


def lathe_mesh(profile_rz: np.ndarray, seg: int = 96) -> tuple:
    """Revolve an (r, z) polyline about +z. Smooth vertex normals from the profile tangents."""
    pr = np.asarray(profile_rz, float)
    m = len(pr)
    t = np.gradient(pr, axis=0)
    n2 = np.c_[t[:, 1], -t[:, 0]]  # outward normal of the polyline (r, z) when traversed outward-on-top
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True).clip(1e-12)
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    P = np.zeros((m, seg, 3))
    N = np.zeros((m, seg, 3))
    P[..., 0] = pr[:, :1] * ca
    P[..., 1] = pr[:, :1] * sa
    P[..., 2] = pr[:, 1:2]
    N[..., 0] = n2[:, :1] * ca
    N[..., 1] = n2[:, :1] * sa
    N[..., 2] = n2[:, 1:2]
    faces = []
    for i in range(m - 1):
        for k in range(seg):
            k2 = (k + 1) % seg
            faces.append([i * seg + k, (i + 1) * seg + k, (i + 1) * seg + k2, i * seg + k2])
    buf = MeshBuf()
    buf.add(P.reshape(-1, 3), N.reshape(-1, 3), faces)
    return buf.arrays()


def plate_mesh(radius: float, **kw) -> tuple:
    """Render mesh of a plate: plate_profile revolved, with the visual ring amplitude PLATE_VIS_RING_AMP unless
    `ring_amp` is given."""
    kw.setdefault("ring_amp", PLATE_VIS_RING_AMP)
    p = plate_profile(radius, **kw)
    # normals: top surface faces up when traversed centre -> rim; (t_z, -t_r) is the right-hand normal -> flip sign
    pts, nrm, c, i = lathe_mesh(p)
    return pts, -nrm, c, i


def quad_mesh(x0, y0, w, h, z) -> tuple:
    """Upward quad with st = ((x - x0) / w, (y - y0) / h)."""
    pts = np.array([[x0, y0, z], [x0 + w, y0, z], [x0 + w, y0 + h, z], [x0, y0 + h, z]])
    nrm = np.tile([0, 0, 1.0], (4, 1))
    st = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
    return pts, nrm, np.array([4]), np.array([0, 1, 2, 3]), st


# ----------------------------------------------------------------------------------------------- USD
def define_mesh(stage, path: str, pts, nrm, counts, idx, st=None, double_sided: bool = False):
    from pxr import Sdf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(pts, np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.asarray(counts, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(idx, np.int32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(nrm, np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(double_sided)
    p = np.asarray(pts)
    mesh.CreateExtentAttr([tuple(map(float, p.min(0))), tuple(map(float, p.max(0)))])
    if st is not None:
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                     UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(st, np.float32)))
    return mesh


def hide_geometry(prim):
    """Hide a shape-spawner's own render geometry (Isaac Lab puts it at <prim>/geometry/mesh) without touching
    physics: purpose=guide is not rendered by cameras, and PhysX ignores purpose (the converted robot's collision
    meshes are guide too)."""
    from pxr import Usd, UsdGeom

    n = 0
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Gprim) and "/geometry" in str(p.GetPath()):
            UsdGeom.Imageable(p).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            n += 1
    return n
