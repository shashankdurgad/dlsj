"""Duplo brick geometry for contact realism (pure numpy, no Isaac): convex collider pieces and mass properties
from the LDraw parts (sim/twin/scene/ldraw.py: 3437 2x2, 3011 2x4, 2302b curved 2x3).

Frame = the env's brick frame: body-box centred, studs on +z, long side along x (as scene/ldraw.py's npz files).

Measured from the LDraw meshes (real play applied, scene/ldraw.py):
  body 31.7 x 31.7 x 19.2 (2x2), 63.7 x 31.7 x 19.2 (2x4), 47.7 x 31.7 x 19.2 (curved 2x3) mm
  studs: radius 4.8 mm, 4.4 mm high, centres on the 16 mm grid at +-8 mm
  shell: side walls 1.45 mm, top deck 1.6 mm, open bottom, underside tubes radius 6.4 mm (2x2 one, 2x4 three)
  solid volume (divergence theorem on the welded LDraw triangles): 2x2 5.6-5.8 cm^3, 2x4 9.8-10.3, curved 7.1-7.2;
  x ABS 1.05 g/cm^3 = 5.9 / 10.5 / 7.5 g. BrickLink weighs 3437 at 6.75 g and 3011 at ~12 g: the same 1.15-1.2 x
  for both, so the real walls are thicker than LDraw's; the mass DISTRIBUTION (centre of mass, inertia / mass) is
  taken from the mesh and the mass itself from params.json (bricks.mass_kg) unless contact.brick_mass = "mesh".

Collider variants (`parts(kind, variant)`):
  studs   the body (box; the curved brick's LDraw body hull) + one 16-gon prism per stud. Solid body.
  shell   top deck + 4 side walls + underside tubes + studs, open underneath (an upside-down brick shows its 17.6 mm
          deep cavity); the curved 2x3's quarter-round deck / end wall is a chain of 1.6 mm slabs along the LDraw
          profile, its long walls follow the profile underneath.
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SCENE = REPO / "sim/twin/scene"
KINDS = ("2x2", "2x4", "curved_2x3")
BODY = {"2x2": (0.0317, 0.0317, 0.0192), "2x4": (0.0637, 0.0317, 0.0192), "curved_2x3": (0.0477, 0.0317, 0.0192)}
STUD_R, STUD_H = 0.0048, 0.0044
WALL, DECK = 0.00145, 0.0016
TUBE_R, TUBE_Z0 = 0.0064, -0.0080  # tube bottom 1.6 mm above the brick's bottom rim
STUDS = {"2x2": [(x, y) for x in (-0.008, 0.008) for y in (-0.008, 0.008)],
         "2x4": [(x, y) for x in (-0.024, -0.008, 0.008, 0.024) for y in (-0.008, 0.008)],
         "curved_2x3": [(x, y) for x in (-0.016, 0.0) for y in (-0.008, 0.008)]}
TUBES = {"2x2": [(0.0, 0.0)], "2x4": [(-0.016, 0.0), (0.0, 0.0), (0.016, 0.0)],
         "curved_2x3": [(-0.008, 0.0), (0.008, 0.0)]}
ABS_DENSITY = 1050.0  # kg/m^3
NGON = 16


def kind_from_size(dims) -> str:
    """Kind from the body's bounding dimensions (m): the long side (31.7 / 47.7 / 63.7 mm)."""
    long_side = max(float(dims[0]), float(dims[1]))
    if long_side > 0.056:
        return "2x4"
    return "curved_2x3" if long_side >= 0.040 else "2x2"


def box(lo, hi) -> np.ndarray:
    """8 corner points of an axis-aligned box."""
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def prism(cx: float, cy: float, r: float, z0: float, z1: float, n: int = NGON) -> np.ndarray:
    """Points of an n-gon prism circumscribing a cylinder of radius r (flat sides touch the cylinder)."""
    rr = r / math.cos(math.pi / n)
    a = (np.arange(n) + 0.5) * 2 * math.pi / n
    ring = np.c_[cx + rr * np.cos(a), cy + rr * np.sin(a)]
    return np.r_[np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)]]


@lru_cache(maxsize=None)
def _npz(kind: str):
    p = SCENE / "assets" / f"duplo_{kind}.npz"
    return dict(np.load(p)) if p.exists() else None


def body_hull(kind: str) -> np.ndarray:
    """Convex body without studs: a box for 2x2 / 2x4, the LDraw body hull for the curved brick."""
    hx, hy, hz = (v / 2 for v in BODY[kind])
    if kind == "curved_2x3":
        m = _npz(kind)
        if m is not None:
            return m["hull"].astype(float) * (np.array(BODY[kind]) / m["body_size"])
    return box((-hx, -hy, -hz), (hx, hy, hz))


def parts(kind: str, variant: str) -> list[tuple[str, np.ndarray]]:
    """[(name, convex points (n, 3))] of the collider variant ('studs' | 'shell'), brick frame."""
    hx, hy, hz = (v / 2 for v in BODY[kind])
    studs = [(f"stud{i}", prism(x, y, STUD_R, hz - 0.0005, hz + STUD_H)) for i, (x, y) in enumerate(STUDS[kind])]
    if variant == "studs":
        return [("body", body_hull(kind)), *studs]
    if variant != "shell":
        raise ValueError(f"contact.brick_collider {variant!r}: studs | shell | vhacd")
    if kind == "curved_2x3":
        return _curved_shell(hx, hy, hz) + studs
    zt = hz - DECK
    out = [("deck", box((-hx, -hy, zt), (hx, hy, hz))),
           ("wall_px", box((hx - WALL, -hy, -hz), (hx, hy, zt))),
           ("wall_nx", box((-hx, -hy, -hz), (-hx + WALL, hy, zt))),
           ("wall_py", box((-hx + WALL, hy - WALL, -hz), (hx - WALL, hy, zt))),
           ("wall_ny", box((-hx + WALL, -hy, -hz), (hx - WALL, -hy + WALL, zt)))]
    out += [(f"tube{i}", prism(x, y, TUBE_R, TUBE_Z0, zt + 0.0002)) for i, (x, y) in enumerate(TUBES[kind])]
    return out + studs


def curved_profile(hx: float, hz: float) -> np.ndarray:
    """(x, z) of the curved brick's top outline (y = const), from the flat deck's end over the quarter-round to the
    bottom of the +x end: the upper envelope of the LDraw body hull (fallback: a quarter circle)."""
    m = _npz("curved_2x3")
    if m is not None:
        H = m["hull"].astype(float) * (np.array(BODY["curved_2x3"]) / m["body_size"])
        xs = np.unique(np.round(H[:, 0], 6))
        prof = np.array([(x, H[np.isclose(H[:, 0], x, atol=1e-6), 2].max()) for x in xs])
        prof = prof[prof[:, 1] < hz - 1e-6]
        start = prof[0, 0] - (prof[1, 0] - prof[0, 0]) if len(prof) > 1 else 0.008
        flat_end = max(x for x in xs if np.isclose(H[np.isclose(H[:, 0], x, atol=1e-6), 2].max(), hz))
        return np.r_[[[flat_end, hz]], prof] if start else prof
    a = np.linspace(0, math.pi / 2, 9)
    r = 2 * hz
    return np.c_[hx - r + r * np.sin(a), -hz + r * np.cos(a)]


def _curved_shell(hx: float, hy: float, hz: float) -> list:
    """Hollow curved 2x3: flat deck, -x end wall, long walls, the curved deck as 1.6 mm slabs, two tubes."""
    prof = curved_profile(hx, hz)
    x0 = float(prof[0, 0])
    zt = hz - DECK
    out = [("deck", box((-hx, -hy, zt), (x0, hy, hz))), ("wall_nx", box((-hx, -hy, -hz), (-hx + WALL, hy, zt))),
           ("wall_py", box((-hx + WALL, hy - WALL, -hz), (x0, hy, zt))),
           ("wall_ny", box((-hx + WALL, -hy, -hz), (x0, -hy + WALL, zt)))]
    inner = []
    for i in range(len(prof) - 1):
        (xa, za), (xb, zb) = prof[i], prof[i + 1]
        t = np.array([xb - xa, zb - za])
        n = np.array([t[1], -t[0]]) / max(np.linalg.norm(t), 1e-9)  # outward (up / +x) normal of the segment
        if n[1] < 0 and n[0] < 0:
            n = -n
        ia, ib = np.array([xa, za]) - DECK * n, np.array([xb, zb]) - DECK * n
        inner.append((ia, ib))
        P2 = np.array([[xa, za], [xb, zb], ia, ib])
        out.append((f"curve{i}", np.array([[x, y, z] for x, z in P2 for y in (-hy, hy)])))
        for side, (y0, y1) in (("p", (hy - WALL, hy)), ("n", (-hy, -hy + WALL))):
            za_in, zb_in = max(ia[1], -hz), max(ib[1], -hz)
            if max(za_in, zb_in) - (-hz) < 1e-4:
                continue
            Q = [(ia[0], -hz), (ib[0], -hz), (ia[0], za_in), (ib[0], zb_in)]
            out.append((f"wall{side}{i}", np.array([[x, y, z] for x, z in Q for y in (y0, y1)])))
    for k, (cx, cy) in enumerate(TUBES["curved_2x3"]):
        top = zt
        for ia, ib in inner:  # stay under the curved deck's inner surface
            if min(ia[0], ib[0]) <= cx + TUBE_R and max(ia[0], ib[0]) >= cx - TUBE_R:
                top = min(top, ia[1], ib[1])
        out.append((f"tube{k}", prism(cx, cy, TUBE_R, TUBE_Z0, top + 0.0002)))
    return out


def hull_faces(pts: np.ndarray) -> np.ndarray:
    """Triangle faces of the convex hull of pts (scipy when available; else a fan PhysX accepts for convexHull)."""
    try:
        from scipy.spatial import ConvexHull

        h = ConvexHull(pts)
        F = h.simplices.copy()
        c = pts.mean(0)
        n = np.cross(pts[F[:, 1]] - pts[F[:, 0]], pts[F[:, 2]] - pts[F[:, 0]])
        flip = np.einsum("ij,ij->i", n, pts[F[:, 0]] - c) < 0
        F[flip] = F[flip][:, ::-1]
        return F
    except Exception:  # noqa: BLE001
        return np.array([(0, k, k + 1) for k in range(1, len(pts) - 1)])


@lru_cache(maxsize=None)
def ldraw_solid(kind: str):
    """(V, F) of the raw welded LDraw part in the brick frame (play applied, long side on x); None if the LDraw
    library is missing (python sim/twin/scene/ldraw.py fetch)."""
    sys.path.insert(0, str(SCENE))
    try:
        import ldraw as L

        T = L.load_triangles(L.PARTS[kind]) @ L.TO_ZUP.T * L.LDU
    except Exception:  # noqa: BLE001
        return None
    V, F = L.weld(T)
    lo, hi = V.min(0), V.max(0)
    on_x = np.isclose(np.abs(V[:, 0] - (lo[0] + hi[0]) / 2), (hi[0] - lo[0]) / 2, atol=1e-7)
    top = V[on_x, 2].max()
    V = V - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + top) / 2])
    hx, hy = (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2
    for ax, h in ((0, hx), (1, hy)):
        on = np.isclose(np.abs(V[:, ax]), h, atol=1e-7)
        V[on, ax] -= np.sign(V[on, ax]) * L.PLAY
    if hy > hx:
        V = np.c_[-V[:, 1], V[:, 0], V[:, 2]]
    return V, F


def solid_props(V: np.ndarray, F: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(volume m^3, centre of mass (3,), inertia tensor about the COM per unit volume (3, 3)) of a closed mesh."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    d = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = d.sum() / 6
    com = (d[:, None] * (a + b + c)).sum(0) / 24 / vol
    can = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 120.0
    A = np.stack([a, b, c], 2)
    C = np.einsum("n,nij,jk,nlk->il", d, A, can, A) - vol * np.outer(com, com)
    return float(vol), com, (np.trace(C) * np.eye(3) - C) / vol


def mass_props(kind: str, mass: float | None = None) -> dict:
    """Mass properties of the real (hollow) brick from the LDraw solid: mass (kg; `mass` or volume x ABS density),
    centre of mass (m), principal inertia (kg m^2) and principal axes (quaternion w, x, y, z). Falls back to the
    solid body box when the LDraw library is missing."""
    s = ldraw_solid(kind)
    if s is None:
        dims = np.array(BODY[kind])
        vol, com = float(np.prod(dims)), np.zeros(3)
        Iv = np.diag([dims[1] ** 2 + dims[2] ** 2, dims[0] ** 2 + dims[2] ** 2, dims[0] ** 2 + dims[1] ** 2]) / 12
        src = "box"
    else:
        vol, com, Iv = solid_props(*s)
        src = "ldraw"
    m = float(mass) if mass is not None else vol * ABS_DENSITY
    w, R = np.linalg.eigh(Iv * m)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return {"mass": m, "com": com, "inertia": w, "axes_wxyz": _mat_to_quat(R), "volume_cm3": vol * 1e6,
            "mesh_mass_g": vol * ABS_DENSITY * 1e3, "source": src}


def _mat_to_quat(R: np.ndarray) -> tuple:
    """Rotation matrix -> quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


def rest_heights(kind: str, variant: str) -> dict:
    """Body-centre height above the table for the resting poses with this collider (m)."""
    hx, hy, hz = (v / 2 for v in BODY[kind])
    studded = variant in ("studs", "shell", "vhacd")
    return {"up": hz, "down": hz + (STUD_H if studded else 0.0), "side_x": hy, "side_y": hy}


if __name__ == "__main__":
    import json

    rep = {}
    for k in KINDS:
        mp = mass_props(k)
        rep[k] = {"volume_cm3": round(mp["volume_cm3"], 3), "mesh_mass_g_abs1050": round(mp["mesh_mass_g"], 2),
                  "com_mm": np.round(mp["com"] * 1e3, 2).tolist(),
                  "inertia_per_kg_mm2": np.round(mp["inertia"] / mp["mass"] * 1e6, 1).tolist(),
                  "solid_box_inertia_per_kg_mm2": np.round(
                      [(BODY[k][1] ** 2 + BODY[k][2] ** 2) / 12 * 1e6, (BODY[k][0] ** 2 + BODY[k][2] ** 2) / 12 * 1e6,
                       (BODY[k][0] ** 2 + BODY[k][1] ** 2) / 12 * 1e6], 1).tolist(),
                  "parts": {v: len(parts(k, v)) for v in ("studs", "shell")}, "source": mp["source"]}
    print(json.dumps(rep, indent=1))
