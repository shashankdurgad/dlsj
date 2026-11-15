"""Brick variety for TwinEnv: a pool of 2x2 / 2x4 / curved 2x3 Duplo bodies, layouts sampled from the real-data
statistics (scene_stats.json from layout.py) or copied from a real episode (layout.layout_from_episode).

The residual env has one target body ("Brick") and N distractor bodies with fixed box colliders, so variety is a
pool: TwinEnv builds the scene with `pool_specs()` as the distractor list, and every reset assigns pool bodies to the
target and the distractors of the drawn kinds (the residual env's `brick` / `distractors` handles are re-pointed;
unused bodies are parked off the table by the residual env). Colours are material rebinds (twin_scene palette).

Colliders stay boxes of the body size (studs excluded, as before): 2x2 31.8 x 31.8 x 19.2 mm, 2x4 63.6 x 31.8 x
19.2 mm, curved 2x3 47.7 x 31.8 x 19.2 mm (its quarter-round end is inside the box: a box approximation).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STATS_JSON = HERE / "scene_stats.json"
SIZES = {"2x2": (0.0318, 0.0318, 0.0192), "2x4": (0.0636, 0.0318, 0.0192), "curved_2x3": (0.0477, 0.0318, 0.0192)}
# kg: 2x2 / 2x4 from physics/params.json (BrickLink); curved 2x3 = 2x2 mass x LDraw surface-area ratio 130.5 / 94.3 cm^2
# (the same ratio gives 12.6 g for the 2x4 vs BrickLink ~12 g)
MASS = {"2x2": 0.00675, "2x4": 0.012, "curved_2x3": 0.0093}
DEFAULT_POOL = {"2x2": 6, "2x4": 3, "curved_2x3": 3}  # incl. the target body "Brick" (a 2x2)
# bricks lying upside down (studs on the table): 3 of 6 checked targets show their hollow underside in the top view and
# in the wrist camera while held (check_scene.py bricks, episodes 6 / 165 / 195); a rough estimate
UPSIDE_DOWN_FRAC = 0.4
TABLE_Z = -0.002
PLATE_FLOOR = 0.003  # plate floor top above the table
PARK = (1.5, 1.5)


def pool_specs(Brick, pool: dict | None = None) -> list:
    """Distractor Brick specs of the pool (the residual env's target body "Brick" is the first 2x2)."""
    pool = dict(pool or DEFAULT_POOL)
    pool["2x2"] = max(0, pool.get("2x2", 0) - 1)
    out = []
    for kind, n in pool.items():
        for k in range(n):
            out.append(Brick(f"pool_{kind}_{k}", SIZES[kind], PARK, (0.8, 0.8, 0.8), MASS[kind]))
    return out


def kind_of(spec) -> str:
    long_ = max(spec.size[0], spec.size[1])
    return "2x2" if long_ < 0.040 else ("curved_2x3" if long_ < 0.056 else "2x4")


def load_stats(path: Path = STATS_JSON) -> dict:
    """Real-data frequencies (layout.py stats); a uniform-ish fallback when the file is missing."""
    if path.exists():
        return json.loads(path.read_text())
    return {"target_kind": {"2x2": 0.6, "2x4": 0.2, "curved_2x3": 0.2},
            "brick_kind": {"2x2": 0.6, "2x4": 0.2, "curved_2x3": 0.2},
            "color_kind": {}, "target_color": {"light_blue": 0.3, "yellow": 0.2, "brown": 0.2, "red": 0.1,
                                               "orange": 0.1, "cream": 0.1},
            "brick_color": {"light_blue": 0.3, "yellow": 0.2, "brown": 0.2, "red": 0.1, "orange": 0.1, "cream": 0.1},
            "distractor_count": {"1": 1, "2": 2, "3": 2, "4": 1}, "distractor_on_plate_frac": 0.4,
            "plates_visible_count": {"2": 1, "3": 1}}


def _pick(rng, probs: dict, allowed=None):
    items = [(k, float(v)) for k, v in probs.items() if (allowed is None or k in allowed) and float(v) > 0]
    if not items:
        return None
    k, p = zip(*items, strict=True)
    p = np.asarray(p) / sum(p)
    return k[int(rng.choice(len(k), p=p))]


def _color_given_kind(rng, stats, kind, key_fallback):
    ck = {k.split("|")[0]: v for k, v in stats.get("color_kind", {}).items() if k.endswith("|" + kind)}
    return _pick(rng, ck) or _pick(rng, stats.get(key_fallback, {})) or "light_blue"


def sample_plan(rng, stats: dict, pool_counts: dict) -> dict:
    """A brick plan from the real-data statistics: target kind + colour, distractors (kind, colour, on_plate), and how
    many plates are on the table (the cal2 top frames show 2 or 3)."""
    tk = _pick(rng, stats["target_kind"]) or "2x2"
    flip = stats.get("upside_down_frac", UPSIDE_DOWN_FRAC)
    plan = {"target": {"kind": tk, "color": _color_given_kind(rng, stats, tk, "target_color"),
                       "upside_down": bool(rng.random() < flip)}, "distractors": []}
    counts = {int(k): v for k, v in stats["distractor_count"].items()}
    n = int(_pick(rng, {str(k): v for k, v in counts.items()}) or 2)
    left = dict(pool_counts)
    left[tk] -= 1
    for _ in range(n):
        k = _pick(rng, stats["brick_kind"], allowed={x for x, c in left.items() if c > 0})
        if k is None:
            break
        left[k] -= 1
        plan["distractors"].append({"kind": k, "color": _color_given_kind(rng, stats, k, "brick_color"),
                                    "on_plate": bool(rng.random() < stats.get("distractor_on_plate_frac", 0.4)),
                                    "upside_down": bool(rng.random() < flip)})
    pv = stats.get("plates_visible_count", {"3": 1})
    plan["n_plates"] = int(_pick(rng, pv) or 3)
    return plan


def plan_from_layout(lay: dict) -> dict:
    """A brick plan (with exact poses) from a real-episode layout (layout.layout_from_episode)."""
    t = lay.get("target") or {}
    plan = {"target": {"kind": t.get("kind") or "2x2", "color": t.get("color") or "light_blue",
                       "pose": (t["x"], t["y"], t.get("yaw", 0.0)) if t else None},
            "distractors": [{"kind": d["kind"], "color": d["color"], "on_plate": bool(d.get("on_plate")),
                             "pose": (d["x"], d["y"], d.get("yaw", 0.0))} for d in lay.get("distractors", [])],
            "n_plates": len(lay.get("plates", {}))}
    return plan


def _quat(yaw: float, upside_down: bool) -> tuple:
    """xyzw quaternion: yaw about z, then (optionally) half a turn about the brick's x axis (studs down)."""
    c, s = math.cos(yaw / 2), math.sin(yaw / 2)
    return (c, s, 0.0, 0.0) if upside_down else (0.0, 0.0, s, c)


class BrickPool:
    """The pool bodies of one TwinEnv: [(prim name, RigidObject, spec, kind)], target first."""

    def __init__(self, env):
        r = env.residual
        lay = env.layout
        self.env = env
        self.items = [("Brick", r.brick, lay.target, kind_of(lay.target))]
        self.items += [(f"Distractor{i}", obj, spec, kind_of(spec))
                       for i, (obj, spec) in enumerate(zip(r.distractors, lay.distractors, strict=True))]
        self.counts = {}
        for it in self.items:
            self.counts[it[3]] = self.counts.get(it[3], 0) + 1
        self.target0 = lay.target
        self.current: dict = {}

    def assign(self, plan: dict) -> dict:
        """Point the residual env's target / distractor handles at pool bodies of the planned kinds (before
        ResidualEnv.reset, which then places the first n distractors and parks the rest)."""
        r, lay = self.env.residual, self.env.layout
        free = list(self.items)

        def take(kind):
            for i, it in enumerate(free):
                if it[3] == kind:
                    return free.pop(i)
            return None

        tgt = take(plan["target"]["kind"]) or take("2x2")
        used = []
        for d in plan["distractors"]:
            it = take(d["kind"])
            if it is not None:
                used.append((it, d))
        rest = free
        r.brick = tgt[1]
        lay.target = type(self.target0)(self.target0.name, tgt[2].size, self.target0.pos, tgt[2].color, tgt[2].mass)
        r.distractors = [it[1] for it, _ in used] + [it[1] for it in rest]
        lay.distractors = [it[2] for it, _ in used] + [it[2] for it in rest]
        lay.distractor_count = (len(used), len(used))
        self.current = {"target": (tgt[0], plan["target"]), "distractors": [(it[0], d, it[1]) for it, d in used],
                        "parked": [it[0] for it in rest]}
        return self.current

    def place(self, plan: dict, rng, exact: bool) -> None:
        """After the residual reset and the plate layout: distractor poses (exact from a real layout, else on-plate
        draws / de-overlapped table draws) and colours of every assigned body."""
        import torch

        env, r = self.env, self.env.residual
        dev = env.device
        o = env.scene.env_origins
        plates = [(r.bowl_xy_env[0, k].detach().cpu().numpy(), float(r.bowl_r[k])) for k in range(len(r.bowl_objs))
                  if abs(float(r.bowl_xy_env[0, k, 0])) < 1.0]
        tgt = r.brick_local()[0, :2].detach().cpu().numpy()
        taken = [tgt]
        lay = env.layout
        for _name, d, obj in self.current["distractors"]:
            h = SIZES[d["kind"]][2]
            if exact and d.get("pose") is not None:
                x, y, yaw = d["pose"]
                on = bool(d.get("on_plate"))
            else:
                x, y, yaw, on = self._draw(rng, d, plates, taken, lay)
            taken.append(np.array([x, y]))
            z = TABLE_Z + h / 2 + 0.0005 + (PLATE_FLOOR if on else 0.0)
            pose = torch.tensor([[x, y, z, *_quat(yaw, d.get("upside_down", False))]], device=dev)
            pose[:, :3] += o[:1]
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        if plan["target"].get("upside_down"):  # the residual reset placed it studs up: same pose, flipped
            p = r.brick.data.root_pose_w[:1].clone() if hasattr(r.brick.data, "root_pose_w") else \
                r.brick.data.root_state_w[:1, :7].clone()
            q = p[0, 3:7].detach().cpu().numpy()
            yaw = 2.0 * math.atan2(float(q[2]), float(q[3]))
            p[0, 3:7] = torch.tensor(_quat(yaw, True), device=dev)
            r.brick.write_root_pose_to_sim(p)
        self.colors(plan)

    @staticmethod
    def _draw(rng, d, plates, taken, lay, tries: int = 60):
        """(x, y, yaw, on_plate) for a sampled distractor: on a plate floor (r < 5 cm) or on the table from the
        residual env's distractor distribution, >= 4.5 cm from other bricks and clear of plate rims."""
        yaw = float(rng.uniform(-math.pi, math.pi))
        if d.get("on_plate") and plates:
            for _ in range(tries):
                c, _r = plates[int(rng.integers(len(plates)))]
                a, rr = rng.uniform(0, 2 * math.pi), 0.05 * math.sqrt(rng.random())
                p = c + rr * np.array([math.cos(a), math.sin(a)])
                if all(np.linalg.norm(p - q) > 0.045 for q in taken):
                    return float(p[0]), float(p[1]), yaw, True
        if lay.distractor_xy is not None:
            m, s, lo, hi = (np.asarray(v, float) for v in lay.distractor_xy)
        else:
            m, s = np.array([0.22, -0.03]), np.array([0.05, 0.08])
            lo, hi = np.array([0.12, -0.2]), np.array([0.32, 0.15])
        p = m
        for _ in range(tries):
            p = np.clip(m + rng.normal(size=2) * s, lo, hi)
            if all(np.linalg.norm(p - q) > 0.045 for q in taken) and \
                    all(np.linalg.norm(p - c) > r + 0.03 for c, r in plates):
                break
        return float(p[0]), float(p[1]), yaw, False

    def colors(self, plan: dict) -> None:
        from sim.twin.visuals.twin_scene import set_brick_color

        palette = getattr(self.env.look, "brick_rgb", {})
        for e in range(self.env.cfg.num_envs):
            name, t = self.current["target"]
            if t.get("color") in palette:
                set_brick_color(e, name, t["color"])
            for name, d, _ in self.current["distractors"]:
                if d.get("color") in palette:
                    set_brick_color(e, name, d["color"])

    def describe(self) -> dict:
        """The current assignment (for info / logs)."""
        name, t = self.current.get("target", (None, {}))
        return {"target": {"prim": name, "kind": t.get("kind"), "color": t.get("color"),
                           "upside_down": bool(t.get("upside_down"))},
                "distractors": [{"prim": n, "kind": d["kind"], "color": d["color"], "on_plate": d.get("on_plate"),
                                 "upside_down": bool(d.get("upside_down"))}
                                for n, d, _ in self.current.get("distractors", [])]}
