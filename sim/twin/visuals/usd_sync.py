"""Make the RTX cameras see the current physics state on the GB10 render path.

On the GB10 the twin runs with the Fabric scene delegate OFF (sim/residual_env/app.py KIT_ARGS; otherwise every camera
renders from the world origin), so the renderer reads plain USD. Isaac Lab 3.0's PhysX backend does not write body
poses back to USD (measured: even with SimulationCfg(use_fabric=False) and /physics/updateToUsd=True the gripper link
stays at its spawn pose in USD while PhysX moves it), so the cameras keep rendering the robot, bricks, plates and the
wrist camera at their spawn poses. Writing the physics prims' xforms ourselves is not an option either: omni.physx
re-parses them and invalidates the tensor views ("Simulation view object is invalidated"), even with change
tracking paused.

So the twin renders *visual proxies* that carry no physics (twin_scene.apply_twin_visuals builds them):
* /World/envs/env_i/RobotVisual: a second reference of the robot USD with its Physics variant = "none" (collision
  instances deactivated), painted; its link xforms are set from the physics link poses;
* /World/envs/env_i/TwinVis/<Prim>: Duplo / plate meshes, set from the rigid-object root poses;
* the wrist camera (a child of the physics gripper_link, whose USD pose is stale) gets its local transform corrected
  so that its world pose = authored offset * physics gripper pose.
The physics prims' own render geometry is hidden (purpose = guide). Call the VisualSync object after the physics
step(s) and before `sim.render()`.
"""
from __future__ import annotations

import re

import numpy as np


def env_path(prim_path: str, e: int) -> str:
    """'{ENV_REGEX_NS}/Robot' or '/World/envs/env_[^/]+/Robot' (or env_.*) -> '/World/envs/env_<e>/Robot'."""
    p = prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{e}")
    return re.sub(r"env_(\[\^/\]\+|\.\*)", f"env_{e}", p)


def _mat(pos, quat_xyzw):
    from pxr import Gf

    x, y, z, w = (float(v) for v in quat_xyzw)
    return Gf.Matrix4d().SetRotate(Gf.Quatd(w, x, y, z)) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, pos)))


def _parent_world(prim):
    from pxr import Usd, UsdGeom

    return UsdGeom.Xformable(prim).ComputeParentToWorldTransform(Usd.TimeCode.Default())


def set_local(prim, local):
    """Author a local transform through the prim's translate / orient ops when it has them, else a transform op."""
    from pxr import Gf, UsdGeom

    t = local.ExtractTranslation()
    q = local.ExtractRotationQuat().GetNormalized()
    ta, oa = prim.GetAttribute("xformOp:translate"), prim.GetAttribute("xformOp:orient")
    order = list(UsdGeom.Xformable(prim).GetXformOpOrderAttr().Get() or [])
    if ta and oa and order[:2] == ["xformOp:translate", "xformOp:orient"]:
        ta.Set(Gf.Vec3f(t) if str(ta.GetTypeName()) == "float3" else Gf.Vec3d(t))
        oa.Set(Gf.Quatf(q) if str(oa.GetTypeName()) == "quatf" else Gf.Quatd(q))
        return
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    if len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform:
        ops[0].Set(local)
    else:
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(local)


class VisualSync:
    """Copies physics poses onto the twin's visual proxies (see module doc). Build after sim.reset()."""

    def __init__(self, scene, robot_key: str = "robot", robot_vis: str = "RobotVisual", vis_root: str = "TwinVis",
                 wrist_cam_rel: str | None = None):
        import omni.usd
        from pxr import Usd, UsdGeom

        self.stage = omni.usd.get_context().get_stage()
        self.scene = scene
        self.robot = scene[robot_key] if robot_key in scene.articulations else None
        self.envs = list(range(scene.num_envs))
        self.links, self.rigid, self.cams = {}, {}, {}
        for e in self.envs:
            root = self.stage.GetPrimAtPath(f"/World/envs/env_{e}/{robot_vis}")
            if root and self.robot is not None:
                names = set(self.robot.body_names)
                found = {}
                for p in Usd.PrimRange(root):
                    if p.GetName() in names and p.GetName() not in found:
                        found[p.GetName()] = p
                order = sorted(found, key=lambda n: str(found[n].GetPath()).count("/"))
                self.links[e] = [(self.robot.body_names.index(n), found[n]) for n in order]
            for key, obj in scene.rigid_objects.items():
                name = env_path(obj.cfg.prim_path, e).rsplit("/", 1)[1]
                vp = self.stage.GetPrimAtPath(f"/World/envs/env_{e}/{vis_root}/{name}")
                if vp:
                    self.rigid.setdefault(e, []).append((key, vp, _parent_world(vp).GetInverse()))
            if wrist_cam_rel and self.robot is not None:
                cam = self.stage.GetPrimAtPath(f"/World/envs/env_{e}/{wrist_cam_rel}")
                if cam:
                    local0 = UsdGeom.Xformable(cam).GetLocalTransformation(Usd.TimeCode.Default())
                    self.cams[e] = (cam, self.robot.body_names.index(cam.GetParent().GetName()), local0,
                                    _parent_world(cam).GetInverse())

    def __call__(self) -> int:
        n = 0
        bp = bq = None
        if self.robot is not None:
            bp = self.robot.data.body_pos_w.detach().cpu().numpy()
            bq = self.robot.data.body_quat_w.detach().cpu().numpy()
        roots = {k: (o.data.root_pos_w.detach().cpu().numpy(), o.data.root_quat_w.detach().cpu().numpy())
                 for k, o in self.scene.rigid_objects.items()}
        for e in self.envs:
            for i, prim in self.links.get(e, []):  # parents first, so _parent_world sees the updated parent
                set_local(prim, _mat(bp[e, i], bq[e, i]) * _parent_world(prim).GetInverse())
                n += 1
            for key, prim, pw_inv in self.rigid.get(e, []):
                p, q = roots[key]
                set_local(prim, _mat(p[e], q[e]) * pw_inv)
                n += 1
            if e in self.cams:
                cam, bi, local0, stale_inv = self.cams[e]
                set_local(cam, local0 * _mat(bp[e, bi], bq[e, bi]) * stale_inv)
                n += 1
        return n
