"""Procedural visual meshes for the twin: Duplo bricks with studs, round plates (shallow dish with rim and faint
concentric rings), and the textured table-top quad. Mesh builders are pure numpy (points, normals, face counts,
indices); `define_mesh` writes a UsdGeom.Mesh (pxr, import after the Kit app is up).

Dimensions
* Duplo (LEGO DUPLO = 2x LEGO System in plan): stud pitch 16 mm, 2x2 brick 31.7 x 31.7 mm, body height 19.2 mm,
  studs 9.4 mm diameter x 4.4 mm high (hollow top recess on real studs is ignored). The residual env's colliders are
  boxes of exactly the body size (31.8 x 31.8 x 19.2 mm / 63.6 x 31.8 x 19.2 mm) without studs, so the visual body
  matches the collider and the studs stick out on top (+z of the brick).
* Plates: measured from the real top camera (measure_real.py, joe_cal2 camera, rim plane 18 mm above the table):
  outer diameter orange 0.182 m, yellow 0.184 m, purple 0.189 m (IQR +-1-2 mm). Profile of a kids' plastic plate:
  flat floor (with faint concentric ripples) to ~60 % of the radius, sloping wall up to a ~18 mm high rounded rim,
  2 mm wall. The residual env's plate collider (6 mm high cylinder, centre 3 mm above the table) is kept; the dish
  floor sits on the collider top so bricks dropped in the plate rest on the visual floor.
"""
from __future__ import annotations

import math

import numpy as np

DUPLO_PITCH = 0.016
DUPLO_BODY_H = 0.0192
DUPLO_STUD_D = 0.0094
DUPLO_STUD_H = 0.0044


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


def plate_profile(radius: float, floor_z: float = 0.003, rim_h: float = 0.018, wall: float = 0.002,
                  floor_frac: float = 0.62, rings: int = 7, ring_amp: float = 0.00025, n: int = 90):
    """(r, z) polyline of the plate's cross-section, going over the top surface from the centre to the rim lip and
    back underneath. z is relative to the plate frame (collider centre, 3 mm above the table); the table is at
    z = floor_z - 0.006."""
    table = floor_z - 0.006
    top_rim = table + rim_h
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
