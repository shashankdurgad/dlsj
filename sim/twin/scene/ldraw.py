"""LDraw -> triangle mesh for the real Duplo bricks (pure numpy; no Isaac).

    python sim/twin/scene/ldraw.py fetch            # download the parts + every sub-part / primitive they use
    python sim/twin/scene/ldraw.py build            # -> sim/twin/scene/assets/duplo_<kind>.npz (+ report json)

Parts (official LDraw library, https://library.ldraw.org, CC BY 4.0, see assets/ldraw/LICENSE.md):
    2x2         3437.dat   Duplo Brick 2 x 2
    2x4         3011.dat   Duplo Brick 2 x 4
    curved_2x3  2302b.dat  Duplo Brick 2 x 3 with Curved Top (the cream / yellow / brown "curved" bricks on the
                           station: 2 x 2 studs on the flat part plus a stud-less quarter-round end, 2 x 3 footprint;
                           identified from sim/twin/data/photos/table_topdown.jpg and the user's phone photos)

Conversion
* LDraw line types 1 (sub-file reference with a 3x4 transform), 3 (triangle) and 4 (quad) are expanded recursively
  through parts/, parts/s/, p/, p/48/. Lines 2 / 5 (edge / optional lines) are drawing hints and are dropped.
* BFC: every file used here is `BFC CERTIFY` (CCW or CW); winding follows the spec (CW files and INVERTNEXT flip,
  a mirroring transform (det < 0) flips), so all triangles end up counter-clockwise seen from outside.
* Frame: LDraw is -Y up in LDU (1 LDU = 0.4 mm). Output is Z-up metres: (x, y, z)_ld -> 0.0004 * (x, z, -y)
  (a proper rotation, winding preserved), centred on the body box (studs on +z), the same frame as the env's box
  colliders.
* Real size: LDraw bricks are nominal (2x2 = 80 LDU = 32.0 mm). Real Duplo bricks are 0.15 mm smaller per side for
  play (2x2 31.7 mm, 2x4 63.7 mm; measured with calipers by many, e.g. https://www.cailliau.org/en/Alphabetical/L/Lego/Dimensions/
  and the LDraw Duplo dimension notes), so vertices ON the outer side walls move 0.15 mm inward (studs, tubes and
  inner walls stay where LDraw puts them).
* Welding (1e-7 m), degenerate (area < 1e-12 m^2) and duplicate triangles (same vertex set) removed.
* Normals: per-corner, smoothed across neighbours within a 40 deg crease angle (16- and 48-segment cylinders and
  the curved top come out smooth, the 90 deg box edges hard); vertices split per normal.
* Bevels: real ABS bricks have ~0.3-0.5 mm rounded edges that catch highlights. For the box bricks the flat outer
  walls and the flat top deck are replaced by a rounded-box shell (radius `BEVEL`, analytic normals) with the same
  extents; studs, tubes and the underside stay LDraw geometry. The curved brick keeps its LDraw shell (its curved top
  is smooth-shaded) and gets rounded vertical corners only where LDraw has them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB = HERE / "assets/ldraw"
OUT = HERE / "assets"
BASE_URL = "https://library.ldraw.org/library/official/"
LDU = 0.0004  # m
PARTS = {"2x2": "3437.dat", "2x4": "3011.dat", "curved_2x3": "2302b.dat"}
PLAY = 0.00015  # m inward per outer side wall (real vs nominal)
BEVEL = 0.0004  # m edge radius of the box bricks' outer shell
CREASE_DEG = 40.0


# ------------------------------------------------------------------------------------------------ library access
def _norm(name: str) -> str:
    return name.strip().replace("\\", "/").lower()


def _local(name: str) -> Path | None:
    n = _norm(name)
    for d in ("parts", "p"):  # "s/xxx.dat" lives in parts/s/, "48/xxx.dat" in p/48/
        p = LIB / d / n
        if p.exists():
            return p
    return None


def _fetch_one(name: str) -> Path:
    import time
    import urllib.error
    import urllib.request

    n = _norm(name)
    for d in ("parts", "p"):
        url = BASE_URL + d + "/" + n
        for attempt in range(8):
            time.sleep(1.0)  # the library rate-limits (HTTP 429)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ldraw-fetch/1.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(float(e.headers.get("Retry-After") or 10 * (attempt + 1)))
                    continue
                data = None
            except Exception:  # noqa: BLE001
                data = None
            break
        if not data:
            continue
        dst = LIB / d / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        return dst
    raise FileNotFoundError(f"LDraw file {name} not found under {BASE_URL}{{parts,p}}/")


def fetch(parts=None) -> list[str]:
    """Download the parts and, recursively, every referenced sub-file (skips files already on disk)."""
    todo, seen, got = list(parts or PARTS.values()), set(), []
    while todo:
        n = _norm(todo.pop())
        if n in seen:
            continue
        seen.add(n)
        p = _local(n) or _fetch_one(n)
        got.append(str(p.relative_to(LIB)))
        for line in p.read_text(errors="replace").splitlines():
            t = line.split()
            if t and t[0] == "1" and len(t) >= 15:
                todo.append(" ".join(t[14:]))
    return sorted(got)


# ------------------------------------------------------------------------------------------------ parsing
def _read(name: str) -> list[str]:
    p = _local(name)
    if p is None:
        raise FileNotFoundError(f"{name}: run `python sim/twin/scene/ldraw.py fetch` first")
    return p.read_text(errors="replace").splitlines()


def load_triangles(name: str) -> np.ndarray:
    """(N, 3, 3) triangles in LDU, LDraw frame, counter-clockwise from outside (BFC)."""
    tris: list[np.ndarray] = []

    def walk(fname, M, t, invert, depth=0):
        if depth > 32:
            raise RecursionError(fname)
        ccw = True  # BFC CERTIFY [CCW|CW]; every file used here is certified
        invert_next = False
        for line in _read(fname):
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "0":
                up = [s.upper() for s in tok[1:]]
                if up[:1] == ["BFC"]:
                    if "CW" in up:
                        ccw = False
                    if "CCW" in up:
                        ccw = True
                    if "INVERTNEXT" in up:
                        invert_next = True
                continue
            if tok[0] == "1":
                v = list(map(float, tok[2:14]))
                tt = np.array(v[0:3])
                R = np.array([v[3:6], v[6:9], v[9:12]])
                sub = " ".join(tok[14:])
                walk(sub, M @ R, M @ tt + t, invert ^ invert_next, depth + 1)
                invert_next = False
                continue
            if tok[0] in ("3", "4"):
                k = int(tok[0])
                pts = np.array(list(map(float, tok[2:2 + 3 * k]))).reshape(k, 3)
                pts = pts @ M.T + t
                flip = (not ccw) ^ invert ^ (np.linalg.det(M) < 0)
                faces = [pts[[0, 1, 2]]] if k == 3 else [pts[[0, 1, 2]], pts[[0, 2, 3]]]
                for f in faces:
                    tris.append(f[::-1] if flip else f)
                invert_next = False

    walk(name, np.eye(3), np.zeros(3), False)
    return np.asarray(tris)


# ------------------------------------------------------------------------------------------------ mesh processing
TO_ZUP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # (x, y, z)_ld -> (x, z, -y): det +1


def weld(tris: np.ndarray, tol: float = 1e-7):
    """(N,3,3) -> (V (M,3), F (N,3)) with shared vertices; drops degenerate and duplicate triangles."""
    P = tris.reshape(-1, 3)
    key = np.round(P / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V = P[first]
    F = inv.reshape(-1, 3)
    a = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = 0.5 * np.linalg.norm(a, axis=1)
    ok = (area > 1e-12) & (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[ok]
    s = np.sort(F, axis=1)
    _, keep = np.unique(s, axis=0, return_index=True)
    F = F[np.sort(keep)]
    return V, F


def crease_normals(V, F, crease_deg: float = CREASE_DEG):
    """Split-vertex mesh with crease-angle normals: returns (points (K,3), normals (K,3), faces (N,3))."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fa = np.linalg.norm(fn, axis=1, keepdims=True)
    fu = fn / fa.clip(1e-20)
    cos_c = np.cos(np.radians(crease_deg))
    adj: dict[int, list[int]] = {}
    for fi, f in enumerate(F):
        for v in f:
            adj.setdefault(int(v), []).append(fi)
    pts, nrm, out = [], [], np.zeros_like(F)
    cache: dict[tuple, int] = {}
    for fi, f in enumerate(F):
        for c, v in enumerate(f):
            nb = adj[int(v)]
            sel = [g for g in nb if fu[g] @ fu[fi] > cos_c]
            n = fn[sel].sum(0)  # area-weighted
            n = n / max(np.linalg.norm(n), 1e-20)
            k = (int(v), tuple(np.round(n, 4)))
            if k not in cache:
                cache[k] = len(pts)
                pts.append(V[v])
                nrm.append(n)
            out[fi, c] = cache[k]
    return np.asarray(pts), np.asarray(nrm), out


def rounded_box_shell(hx, hy, zb, zt, r, n_arc: int = 4):
    """Outer shell of a box with rounded vertical edges and rounded top edges (radius r), open bottom, flat top cap.
    Returns (points, normals, faces) with analytic (smooth) normals, CCW from outside."""
    # perimeter of the rounded rectangle, CCW seen from +z: (inner point Q, outward normal N)
    Q, N = [], []
    for cx, cy, a0 in ((hx - r, hy - r, 0.0), (-(hx - r), hy - r, 90.0), (-(hx - r), -(hy - r), 180.0),
                       (hx - r, -(hy - r), 270.0)):
        for a in np.radians(np.linspace(a0, a0 + 90.0, n_arc + 1)):
            Q.append((cx, cy))
            N.append((np.cos(a), np.sin(a)))
    Q, N = np.asarray(Q), np.asarray(N)
    m = len(Q)
    rows_p, rows_n = [], []
    # wall bottom -> wall top -> top-edge arc (theta 0..90) -> inner top ring
    for z in (zb, zt - r):
        rows_p.append(np.c_[Q + r * N, np.full(m, z)])
        rows_n.append(np.c_[N, np.zeros(m)])
    for th in np.radians(np.linspace(0, 90, n_arc + 1))[1:]:
        rows_p.append(np.c_[Q + r * np.cos(th) * N, np.full(m, zt - r + r * np.sin(th))])
        rows_n.append(np.c_[np.cos(th) * N, np.full(m, np.sin(th))])
    P = np.concatenate(rows_p)
    Nn = np.concatenate(rows_n)
    faces = []
    for i in range(len(rows_p) - 1):
        for k in range(m):
            k2 = (k + 1) % m
            a, b, c, d = i * m + k, i * m + k2, (i + 1) * m + k2, (i + 1) * m + k
            faces += [(a, b, c), (a, c, d)]
    # top cap (fan around the centre, its own vertices with +z normals)
    top0 = len(P)
    cap = np.c_[Q, np.full(m, zt)]
    P = np.r_[P, [[0.0, 0.0, zt]], cap]
    Nn = np.r_[Nn, np.tile([0, 0, 1.0], (m + 1, 1))]
    for k in range(m):
        faces.append((top0, top0 + 1 + k, top0 + 1 + (k + 1) % m))
    F = np.asarray(faces)
    # drop zero-area faces (coincident inner points at the corners)
    a = np.linalg.norm(np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]), axis=1)
    return P, Nn, F[a > 1e-14]


def build(kind: str) -> dict:
    """Mesh of one Duplo kind in the env brick frame: dict(points, normals, faces, body_size, report)."""
    T = load_triangles(PARTS[kind]) @ TO_ZUP.T * LDU
    V, F = weld(T)
    # body extents: side walls span the full footprint; the body top is the highest z of the outer walls
    lo, hi = V.min(0), V.max(0)
    on_x = np.isclose(np.abs(V[:, 0] - (lo[0] + hi[0]) / 2), (hi[0] - lo[0]) / 2, atol=1e-7)
    body_top = V[on_x, 2].max()
    zb = lo[2]
    c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (zb + body_top) / 2])
    V = V - c
    hx, hy = (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2
    hz = (body_top - zb) / 2
    # play: outer side-wall vertices 0.15 mm inward
    for ax, h in ((0, hx), (1, hy)):
        on = np.isclose(np.abs(V[:, ax]), h, atol=1e-7)
        V[on, ax] -= np.sign(V[on, ax]) * PLAY
    hx, hy = hx - PLAY, hy - PLAY
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fu = fn / np.linalg.norm(fn, axis=1, keepdims=True)
    rep = {"part": PARTS[kind], "triangles_ldraw": int(len(T)), "triangles_welded": int(len(F))}
    shell = None
    if kind in ("2x2", "2x4"):
        # remove the flat outer walls and the flat top deck; a rounded shell replaces them
        Vf = V[F]
        wall = np.zeros(len(F), bool)
        for ax, h in ((0, hx), (1, hy)):
            on = np.isclose(np.abs(Vf[..., ax]), h, atol=1e-7).all(1)
            wall |= on & (np.abs(fu[:, ax]) > 0.99) & (np.sign(fu[:, ax]) == np.sign(Vf[:, 0, ax]))
        deck = np.isclose(Vf[..., 2], hz, atol=1e-7).all(1) & (fu[:, 2] > 0.99)
        rep.update(removed_outer_wall_tris=int(wall.sum()), removed_top_deck_tris=int(deck.sum()))
        F = F[~(wall | deck)]
        shell = rounded_box_shell(hx, hy, -hz, hz, BEVEL)
    if hy > hx:  # long axis along +x like the env's 2x4 collider (rotate +90 deg about z: (x, y) -> (-y, x))
        V = np.c_[-V[:, 1], V[:, 0], V[:, 2]]
        hx, hy = hy, hx
        rep["rotated_90deg_z"] = True
    P, Nn, Fo = crease_normals(V, F)
    if shell is not None:
        sp, sn, sf = shell
        Fo = np.r_[Fo, sf + len(P)]
        P = np.r_[P, sp]
        Nn = np.r_[Nn, sn]
    # orientation check: corner normals should agree with the face normals
    fn2 = np.cross(P[Fo[:, 1]] - P[Fo[:, 0]], P[Fo[:, 2]] - P[Fo[:, 0]])
    agree = (np.einsum("ij,ij->i", fn2, Nn[Fo].mean(1)) > 0).mean()
    # outward check on the outer hull: faces whose centroid is within 0.3 mm of the bbox should point outward
    cen = P[Fo].mean(1)
    outer = (np.abs(np.abs(cen[:, 0]) - hx) < 3e-4) | (np.abs(np.abs(cen[:, 1]) - hy) < 3e-4)
    outward = (np.einsum("ij,ij->i", fn2[outer], np.c_[cen[outer, :2], np.zeros(outer.sum())]) > 0).mean()
    rep.update(points=int(len(P)), faces=int(len(Fo)),
               body_size_m=[round(2 * hx, 5), round(2 * hy, 5), round(2 * hz, 5)],
               total_height_m=round(float(P[:, 2].max() + hz), 5),
               normals_agree_with_winding=round(float(agree), 4), outer_faces_point_outward=round(float(outward), 4))
    studs = P[P[:, 2] > hz + 1e-5]
    if len(studs):
        rep["stud_height_m"] = round(float(studs[:, 2].max() - hz), 5)
    return dict(points=P.astype(np.float32), normals=Nn.astype(np.float32), faces=Fo.astype(np.int32),
                body_size=np.array([2 * hx, 2 * hy, 2 * hz], np.float32),
                hull=_hull_points(V[np.abs(V[:, 2]) <= hz + 1e-7]).astype(np.float32), report=rep)


def _hull_points(V):
    try:
        from scipy.spatial import ConvexHull

        return V[ConvexHull(V).vertices]
    except Exception:  # noqa: BLE001
        return V


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "fetch":
        got = fetch()
        print("FETCHED", len(got), "files")
        return
    rep = {}
    for kind in PARTS:
        m = build(kind)
        np.savez_compressed(OUT / f"duplo_{kind}.npz", points=m["points"], normals=m["normals"], faces=m["faces"],
                            body_size=m["body_size"], hull=m["hull"])
        rep[kind] = m["report"]
        print(kind, json.dumps(m["report"]), flush=True)
    (OUT / "duplo_meshes.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
