"""Push forensics: which shape touches the target brick first, where on the brick, and how hard.

In Isaac (via gate_run.py --c_forensics, with --set contact.report=true): `Recorder(env)` wraps the env's physics
step and keeps every brick contact pair of the PhysX contact report (per physics substep).
Offline (plain python):
    python sim/twin/contact/forensics.py <tag> [--episodes ...]   -> out/contact/<tag>/forensics.json + table
For each demo: the first tick a robot shape touches the target brick (relative to the close frame), the shape
(fixed / moving side, base or fin slice, fin slice index), the contact point in the brick frame (which face:
top / side / edge), the normal, and the brick's displacement over the next ticks; plus the shape that carried the
largest impulse before the close.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "sim/twin/out/contact"


class Recorder:
    """Per-substep brick contacts from PhysX tensor contact views (one per brick body: Brick, Brick2x4, BrickCurved),
    filtered against the gripper / jaw links, every fin segment and the table. Needs PhysxContactReportAPI on the
    bricks (contact.report=true)."""

    BODIES = ("Brick", "Brick2x4", "BrickCurved")

    def __init__(self, env):
        import omni.usd
        from isaaclab_physx.physics import PhysxManager
        from pxr import Usd, UsdPhysics

        self.env = env
        stage = omni.usd.get_context().get_stage()
        root = "/World/envs/env_0"
        filt = []
        grip = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root + "/Robot")) if p.GetName() == "gripper_link"]
        for p in Usd.PrimRange(grip[0]):  # the gripper's rigid bodies: both links and every fin segment
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                filt.append(str(p.GetPath()))
        table = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root + "/Table"))
                 if p.HasAPI(UsdPhysics.CollisionAPI)]
        filt += table[:1]
        self.names = [f.replace(root + "/", "") for f in filt]
        sv = PhysxManager.get_physics_sim_view()
        n = env.scene.num_envs
        # explicit per-env paths: a glob over the nested fixed-fin bodies matches twice per env
        pats = [[f.replace("env_0", f"env_{e}") for f in filt] for e in range(n)]
        self.views = []
        for b in self.BODIES:
            if stage.GetPrimAtPath(f"{root}/{b}"):
                self.views.append(sv.create_rigid_contact_view([f"/World/envs/env_{e}/{b}" for e in range(n)],
                                                               filter_patterns=pats,
                                                               max_contact_data_count=n * len(filt) * 16))
        print("forensics filters:", self.names, "views", len(self.views), flush=True)
        self.dt = 1.0 / 120.0
        self.tick = 0
        self.sub = 0
        self.mat = []  # (tick, sub, view, env, |normal force| per filter ..., net force xyz)
        self.rows = []  # (tick, sub, env, filter, force N, px, py, pz, nx, ny, nz, min separation, n points)
        phys, step = env.physics, env.sim.step

        def physics(*a, **k):
            self.sub = 0
            r = phys(*a, **k)
            self.tick += 1
            return r

        def sim_step(*a, **k):
            r = step(*a, **k)
            self._collect()
            self.sub += 1
            return r

        env.physics = physics
        env.sim.step = sim_step

    @staticmethod
    def _np(a):
        import warp as wp

        try:
            return wp.to_torch(a).detach().cpu().numpy()
        except Exception:  # noqa: BLE001
            return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)

    def _collect(self):
        for kv, v in enumerate(self.views):
            fm = self._np(v.get_contact_force_matrix(dt=self.dt)).reshape(self.env.scene.num_envs, -1, 3)
            net = self._np(v.get_net_contact_forces(dt=self.dt)).reshape(self.env.scene.num_envs, 3)
            for e in np.nonzero((np.abs(fm).sum((1, 2)) > 0) | (np.abs(net).sum(1) > 0))[0]:
                self.mat.append((self.tick, self.sub, kv, int(e), *np.linalg.norm(fm[e], axis=1), *net[e]))
            f, pts, nrm, sep, cnt, st = (self._np(x) for x in v.get_contact_data(dt=self.dt))
            cnt, st = cnt.reshape(cnt.shape[0], -1), st.reshape(st.shape[0], -1)
            f = f.reshape(-1)
            for e, j in zip(*np.nonzero(cnt), strict=True):
                s0, c = int(st[e, j]), int(cnt[e, j])
                w = np.abs(f[s0:s0 + c])
                P, Nn = pts[s0:s0 + c], nrm[s0:s0 + c]
                p = (P * w[:, None]).sum(0) / w.sum() if w.sum() > 0 else P.mean(0)
                self.rows.append((self.tick, self.sub, int(e), int(j), float(w.sum()), *p, *Nn.mean(0),
                                  float(sep[s0:s0 + c].min()), c))

    def save(self, path, tr=None):
        o = self.env.scene.env_origins.cpu().numpy()
        R = np.array(self.rows, float) if self.rows else np.zeros((0, 13))
        M = np.array(self.mat, float) if self.mat else np.zeros((0, 4 + len(self.names) + 3))
        np.savez_compressed(path, rows=R, mat=M, names=np.array(self.names), origins=o)
        print(f"forensics: {len(R)} brick contact rows, {len(self.names)} filters -> {path}", flush=True)


# ------------------------------------------------------------------------------------------------ offline
def _quat_rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def shape_label(name: str) -> str:
    """Short label of a filter shape path: fixed / jaw base, fin root piece or fin slice a-d (root -> tip)."""
    if "Table" in name:
        return "table"
    side = "jaw" if "moving_jaw" in name else "fixed"
    segs = re.findall(r"TwinFinSeg_([a-f])", name)
    leaf = name.split("/")[-1]
    if segs and leaf.startswith("TwinFinSeg"):
        return f"{side}_fin_{segs[-1]}"
    if leaf in ("gripper_link", "moving_jaw_so101_v1_link"):
        return f"{side}_link"
    return f"{side}_{leaf}"


def face_of(p_b: np.ndarray, half: np.ndarray, tol: float = 0.0025) -> str:
    """Which face / edge of the body box (brick frame) a contact point is on."""
    hits = []
    for ax, nm in ((2, "top"), (0, "x"), (1, "y")):
        if abs(abs(p_b[ax]) - half[ax]) < tol:
            hits.append(("top" if p_b[2] > 0 else "bottom") if ax == 2 else f"side_{nm}")
    if p_b[2] > half[2] + 0.001:
        return "stud"
    return "+".join(hits) if hits else "inside"


def analyse(tag: str, episodes=None) -> list:
    d = OUT / tag
    tr = np.load(d / "trace.npz")
    C = np.load(d / "contacts.npz")
    res = {e["episode"]: e for e in json.loads((d / "result.json").read_text())["episodes"]}
    rows, names, org = C["rows"], C["names"].tolist(), C["origins"]
    labels = [shape_label(n) for n in names]
    half = {0: np.array([0.01585, 0.01585, 0.0096]), 1: np.array([0.03185, 0.01585, 0.0096]),
            2: np.array([0.02385, 0.01585, 0.0096])}
    out = []
    for k, ep in enumerate(tr["episodes"].tolist()):
        if episodes and ep not in episodes:
            continue
        ic = int(tr["cf"][k] - tr["start"][k])
        R = rows[(rows[:, 2] == k)]
        robot = R[[labels[int(i)] not in ("table",) for i in R[:, 3]]] if len(R) else R
        b0 = tr["b0"][k]
        disp = np.linalg.norm(tr["bp"][k, :, :2] - b0[:2], axis=1) * 1e3
        rec = {"episode": ep, "outcome": res[ep]["outcome"], "kind": int(tr["kind"][k]), "close_tick": ic}
        if len(robot):
            t0 = int(robot[0, 0])
            first = robot[robot[:, 0] == t0]
            j = int(np.argmax(first[:, 4]))
            row = first[j]
            # the trace's brick pose is after tick t0's physics: use the tick before (pose at contact)
            tb = max(t0 - 1, 0)
            Rb = _quat_rot(tr["bq"][k, tb])
            p_b = Rb.T @ (row[5:8] - org[k] - tr["bp"][k, tb])
            n_b = Rb.T @ row[8:11]
            pre = robot[robot[:, 0] <= ic]
            tot = {}
            for r in pre:
                tot[labels[int(r[3])]] = tot.get(labels[int(r[3])], 0.0) + r[4] / 120.0
            rec.update(first_tick_rel_close=t0 - ic, first_shape=labels[int(row[3])],
                       first_shapes_same_tick=sorted({labels[int(i)] for i in first[:, 3]}),
                       point_brick_mm=np.round(p_b * 1e3, 1).tolist(), normal_brick=np.round(n_b, 2).tolist(),
                       face=face_of(p_b, half.get(rec["kind"], half[0])),
                       disp_mm_at_first=round(float(disp[t0]), 1),
                       disp_mm_first_plus10=round(float(disp[min(t0 + 10, len(disp) - 1)]), 1),
                       disp_mm_close=round(float(disp[ic]), 1),
                       impulse_pre_close_mNs={s: round(v * 1e3, 2)
                                              for s, v in sorted(tot.items(), key=lambda x: -x[1])},
                       rel_first_mm=np.round(tr["rel"][k, tb] * 1e3, 1).tolist())
        out.append(rec)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--all", action="store_true", help="every demo, not just the failures")
    a = ap.parse_args()
    out = analyse(a.tag, a.episodes)
    (OUT / a.tag / "forensics.json").write_text(json.dumps(out, indent=1))
    for r in out:
        if not a.all and not a.episodes and r["outcome"] == "success":
            continue
        print(r["episode"], r["outcome"], "kind", r["kind"], "| first", r.get("first_tick_rel_close"),
              r.get("first_shape"), r.get("face"), "pt", r.get("point_brick_mm"), "n", r.get("normal_brick"),
              "| disp first/+10/close",
              r.get("disp_mm_at_first"), r.get("disp_mm_first_plus10"), r.get("disp_mm_close"),
              "| impulse", list((r.get("impulse_pre_close_mNs") or {}).items())[:3], flush=True)
    s = [r for r in out if r["outcome"] == "success" and "first_shape" in r]
    if s:
        from collections import Counter

        print("successes: first shape", Counter(r["first_shape"] for r in s).most_common(),
              "face", Counter(r["face"] for r in s).most_common(), "median first tick",
              float(np.median([r["first_tick_rel_close"] for r in s])), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    main()
