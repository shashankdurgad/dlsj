"""Software fin-ray curl for the hinged fins (opt-in, `gripper.finray.bend.soft_curl_gain`; default off).

A real fin-ray finger wraps: when its contact face is pushed in, the ribs turn the distal part of the fin TOWARD the
object, so the face closes around it and draws it in. The twin's independent spring hinges bend away from the brick
and make a V that opens downward (the squeeze pushes the brick toward the tips). PhysX mimic joints for the coupling
were unstable (README, Results), so here the coupling is a control law instead of a constraint: before every physics
step the distal hinges' PD drive TARGETS are set from the measured proximal bend,

    target_j = rest_j + CURL_SIGN * min(g_j * max(s, 0), soft_curl_max_deg)      (tip curl = -g x root bend)

with s = the low-passed sum of the source hinges' outward bend (raw angle, + = away from the other fin) and g_j the
gain per hinge (root -> tip; a scalar applies to the distal half). The hinge still is a spring (stiffness from the
bend spec) about the moving target, so the brick can push the tip back: the coupling has the compliance of the
drive instead of the stiffness of a constraint.

Keys (gripper.finray.bend):
  soft_curl_gain     scalar (distal half of the hinges) or list per hinge root -> tip; 0 / missing = off
  soft_curl_source   number of proximal hinges whose summed bend is the input (default 1 = the root hinge)
  soft_curl_tau_s    first-order low-pass of the input (s, default 0.02; 0 = none)
  soft_curl_max_deg  clamp of the added curl per hinge (deg, default 30)

Installed by model.add_colliders: wraps InteractiveScene.write_data_to_sim (called before every physics step by the
bench, the residual / grasp env and TwinEnv), so every articulation with fin_<fixed|jaw>_<a..e> joints is coupled.
"""
from __future__ import annotations

import math

SPEC: dict = {}
_STATE: dict = {}
CURL_SIGN = {"fixed": -1.0, "jaw": -1.0}  # raw hinge angle sign that turns each fin toward the other (model.py)


def enabled(spec: dict | None) -> bool:
    g = (spec or {}).get("soft_curl_gain", 0.0) or 0.0
    return any(float(v) != 0.0 for v in g) if isinstance(g, list) else float(g) != 0.0


def gains(spec: dict, hinges: int) -> list[float]:
    """Gain per hinge root -> tip (list, or a scalar for the distal half)."""
    g = spec.get("soft_curl_gain", 0.0) or 0.0
    if isinstance(g, list):
        return [float(g[min(i, len(g) - 1)]) for i in range(hinges)]
    return [float(g) if i >= (hinges + 1) // 2 else 0.0 for i in range(hinges)]


def install(spec: dict) -> bool:
    """Wrap InteractiveScene.write_data_to_sim once (idempotent). Returns True when the coupling is on."""
    SPEC.clear()
    SPEC.update(spec or {})
    if not enabled(SPEC):
        return False
    from isaaclab.scene import InteractiveScene

    if getattr(InteractiveScene.write_data_to_sim, "_fin_curl", False):
        return True
    orig = InteractiveScene.write_data_to_sim

    def write_data_to_sim(self, *a, **kw):
        if SPEC and enabled(SPEC):
            for art in (getattr(self, "_articulations", None) or {}).values():
                _couple(art)
        return orig(self, *a, **kw)

    write_data_to_sim._fin_curl = True
    InteractiveScene.write_data_to_sim = write_data_to_sim
    return True


def _t(x):
    return getattr(x, "torch", x)


def _layout(art):
    key = id(art)
    if key not in _STATE:
        import torch

        from sim.twin.gripper import model

        names = list(art.joint_names)
        fins = {}
        for which in ("fixed", "jaw"):
            ids = [names.index(f"fin_{which}_{c}") for c in model.SEG_NAMES if f"fin_{which}_{c}" in names]
            if ids:
                fins[which] = ids
        st = {"fins": fins, "s": None}
        if fins:
            h = len(next(iter(fins.values())))
            dev = art.device
            st["g"] = torch.tensor(gains(SPEC, h), device=dev, dtype=torch.float32)
            st["rest"] = {w: torch.tensor([float(model.rest_angle(SPEC, i, w)) for i in range(h)], device=dev,
                                          dtype=torch.float32) for w in fins}
            st["src"] = max(1, min(int(SPEC.get("soft_curl_source", 1)), h))
            st["drv"] = [i for i in range(h) if float(st["g"][i]) != 0.0]
        _STATE[key] = st
    return _STATE[key]


def _couple(art):
    import torch

    st = _layout(art)
    if not st["fins"] or not st["drv"]:
        return
    q = _t(art.data.joint_pos)
    dt = 1.0 / 120.0
    try:
        from isaaclab.sim import SimulationContext

        dt = float(SimulationContext.instance().get_physics_dt())
    except Exception:  # noqa: BLE001 - the bench / env always have a context; keep the default otherwise
        pass
    tau = float(SPEC.get("soft_curl_tau_s", 0.02))
    a = 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)
    cap = math.radians(float(SPEC.get("soft_curl_max_deg", 30.0)))
    src = st["src"]
    s_now = torch.stack([q[:, ids[:src]].sum(1) for ids in st["fins"].values()], 1)  # (n, fins) raw outward bend
    s = s_now if st["s"] is None or st["s"].shape != s_now.shape else st["s"] + a * (s_now - st["s"])
    st["s"] = s
    drv = st["drv"]
    full = _t(art.data.joint_pos_target).clone().to(torch.float32)  # the env's targets (arm, gripper, fins)
    for k, (which, ids) in enumerate(st["fins"].items()):
        curl = (st["g"][drv][None, :] * s[:, k:k + 1].clamp(min=0.0)).clamp(max=cap)  # (n, len(drv))
        full[:, [ids[i] for i in drv]] = st["rest"][which][drv][None, :] + CURL_SIGN[which] * curl
    art.set_joint_position_target(full.contiguous())


def reset_filter():
    """Drop the low-pass state (after env resets the bend restarts from the new state)."""
    for st in _STATE.values():
        st["s"] = None
