"""Procedural visual meshes for the twin: Duplo bricks with studs, round plates (shallow dish with rim and faint
concentric rings), and the textured table-top quad. Mesh builders are pure numpy (points, normals, face counts,
indices); `define_mesh` writes a UsdGeom.Mesh (pxr, import after the Kit app is up).

Dimensions
* Duplo (LEGO DUPLO = 2x LEGO System in plan): stud pitch 16 mm, 2x2 brick 31.7 x 31.7 mm, body height 19.2 mm,
  studs 9.4 mm diameter x 4.4 mm high (hollow top recess on real studs is ignored). The residual env's colliders are
  boxes of exactly the body size (31.8 x 31.8 x 19.2 mm / 63.6 x 31.8 x 19.2 mm) without studs, so the visual body
  matches the collider and the studs stick out on top (+z of the brick).
* Brick variants on the real table (sim/twin/data/photos/table_topdown.jpg): 2x2 (31.7 x 31.7 mm), 2x4
  (63.6 x 31.7 mm) and the curved-slope 2x2 (cream / yellow: one stud row, the other half rounds down to ~6 mm at the
  front edge). `brick_mesh(kind)`; `brick_hull_points(kind)` is the matching convex collider (the curved slope is
  convex, so its convex hull is exact; the others are boxes). Hollow underside / stud recesses not modelled.
* Plates: user's measurement (14 Nov): radius 9.2 cm (0.184 m outer diameter; the top camera gave 0.182-0.189 m),
  total height ~23 mm, ~2 mm plastic. Profile: flat floor (faint concentric ripples, the photo shows them over ~3/4
  of the radius) whose top is 3 mm above the table (2 mm shell on a ~1 mm foot: assumed), sloping wall up to a
  rounded rim 23 mm above the table. z is in the plate frame of the residual env's bowl bodies (origin 3 mm above
  the table), so the floor top is at z = 0. The same profile is the plates' physics collider (sim/twin/physics:
  triangle mesh on the kinematic plate body; the env's flat 6 mm cylinder collider is disabled there).
"""
from __future__ import annotations

import math

import numpy as np

DUPLO_PITCH = 0.016
DUPLO_BODY_H = 0.0192
DUPLO_STUD_D = 0.0094
DUPLO_STUD_H = 0.0044
DUPLO_2X2 = (0.0317, 0.0317, DUPLO_BODY_H)
DUPLO_2X4 = (0.0636, 0.0317, DUPLO_BODY_H)
CURVED_LOW_H = 0.0064  # curved-slope 2x2: height of the rounded front edge (estimated from the photo)
PLATE_RADIUS = 0.092  # m, outer (user's measurement)
PLATE_RIM_H = 0.023  # m above the table (total height)
PLATE_WALL = 0.002  # m plastic thickness
PLATE_FLOOR_H = 0.003  # m, floor top above the table (assumed: 2 mm shell + ~1 mm foot)
PLATE_FLOOR_FRAC = 0.75  # flat floor out to this fraction of the radius (photo: rings over ~3/4 of the plate)


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
    pts = np.r_[np.c_[cx + r * ca, cy + r * sa, np.full(seg, z1)], np.c_[cx + rb * ca, cy + rb * sa, np.full(seg, z1 + bevel)]]
    nb = np.c_[ca, sa, np.ones(seg)]
    buf.add(pts, np.r_[nb, nb], [[k, (k + 1) % seg, seg + (k + 1) % seg, seg + k] for k in range(seg)])
    # top cap (fan)
    pts = np.r_[[[cx, cy, z1 + bevel]], np.c_[cx + rb * ca, cy + rb * sa, np.full(seg, z1 + bevel)]]
    buf.add(pts, np.tile([0, 0, 1.0], (seg + 1, 1)), [[0, 1 + k, 1 + (k + 1) % seg] for k in range(seg)])
    return buf.arrays()


def duplo_mesh(nx: int, ny: int, size=None, stud_seg: int = 24) -> tuple:
    """Duplo brick nx x ny studs, centred on its body centre (the collider's centre); studs on +z."""
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


BRICK_KINDS = {"2x2": DUPLO_2X2, "2x4": DUPLO_2X4, "curved_2x2": DUPLO_2X2}


def brick_mesh(kind: str = "2x2", size=None) -> tuple:
    """Visual mesh of a brick variant ('2x2' | '2x4' | 'curved_2x2'), centred on its body-box centre."""
    size = tuple(size) if size is not None else BRICK_KINDS[kind]
    if kind == "curved_2x2":
        return curved_slope_mesh(size)
    nx = max(1, round(size[0] / DUPLO_PITCH))
    ny = max(1, round(size[1] / DUPLO_PITCH))
    return duplo_mesh(nx, ny, size=size)


def brick_hull_points(kind: str = "2x2", size=None) -> np.ndarray:
    """Vertices of the brick's convex collider (studs excluded, as for the box colliders): the body box, or for the
    curved slope the extruded profile (convex, so its hull is exact)."""
    sx, sy, sz = tuple(size) if size is not None else BRICK_KINDS[kind]
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    if kind == "curved_2x2":
        prof = curved_slope_profile((sx, sy, sz))
        yz = np.r_[prof, [[hy, -hz], [-hy, -hz]]]
        return np.r_[np.c_[np.full(len(yz), -hx), yz], np.c_[np.full(len(yz), hx), yz]]
    return np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)])


def plate_profile(radius: float = PLATE_RADIUS, floor_z: float = 0.0, rim_h: float = PLATE_RIM_H,
                  wall: float = PLATE_WALL, floor_frac: float = PLATE_FLOOR_FRAC, rings: int = 7,
                  ring_amp: float = 0.00025, n: int = 90, floor_h: float = PLATE_FLOOR_H):
    """(r, z) polyline of the plate's cross-section, going over the top surface from the centre to the rim lip and
    back underneath. z is relative to the plate frame (the env's bowl body origin, 3 mm above the table): floor top
    at z = floor_z, table at z = floor_z - floor_h, rim top at table + rim_h."""
    table = floor_z - floor_h
    top_rim = table + rim_h - wall * 0.75  # inner top of the wall; the rounded lip adds wall * 0.75 -> table + rim_h
    rf = radius * floor_frac
    r_in = radius - wall * 1.5
    pts = []
    # floor with faint concentric ripples
    for r in np.linspace(0.0, rf, n // 2):
        pts.append((r, floor_z + ring_amp * math.sin(2 * math.pi * rings * r / rf) * (r / rf)))
    # wall: smooth (cosine) rise from the floor to the lip
    for t in np.linspace(0, 1, n // 2)[1:]:
        r = rf + (r_in - rf) * t
        z = floor_z + (top_rim - floor_z) * (0.5 - 0.5 * math.cos(math.pi * t)) ** 0.8
        pts.append((r, z))
    # rounded lip
    for a in np.linspace(math.pi, 0, 7)[1:]:
        pts.append((r_in + wall * 0.75 * (1 + math.cos(a)), top_rim + wall * 0.75 * math.sin(a)))
    # outer wall down to the foot (offset by the wall thickness)
    for t in np.linspace(1, 0, n // 3)[1:]:
        r = rf + (radius - rf) * t
        z = floor_z - wall + (top_rim - floor_z) * (0.5 - 0.5 * math.cos(math.pi * t)) ** 0.8
        pts.append((r, max(z, table + 0.001)))
    pts.append((rf * 0.9, table + 0.001))
    pts.append((rf * 0.9, table))
    pts.append((0.0, table))
    return np.array(pts)


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
