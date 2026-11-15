"""Paint the converted SO-101 USD (sim/twin/assets/so101/usd/so101_new_calib) like the real robot.

Why the residual env's `paint_robot` leaves it yellow: the URDF converter (v0.3.2) writes every link mesh as an
*instanceable* Xform (e.g. `gripper_link/wrist_roll_follower_so101_v1`) referencing `instances.usda`, and the Mesh
prims live inside the instance prototype (instance proxies: no authoring allowed, and `IsA(UsdGeom.Mesh)` never sees
them from a normal traversal). The prototype binds `tn__3d_printed_` (yellow) / `sts3215` (black). Here we
de-instance only the *visual* instance prims (the collision twins `<name>_1` are purpose=guide and stay instanced),
then bind our own materials on their meshes.

Real robot (phone photos + recorded frames): matte black printed body and servos, red TPU fin-ray fingers on black
printed bases (sim/twin/gripper: the stock jaw visuals wrist_roll_follower_so101_v1 on gripper_link and
moving_jaw_so101_v1 are swapped for them; with the stock jaw selected they are painted red instead), and an orange
printed
wrist-camera mount carrying a black camera board on the gripper. The URDF has no camera-mount mesh, so
`add_wrist_camera_mount` builds the real one (sim/twin/robot/wrist_mount.py: flat orange frame + neck screwed to the
follower, official 32x32 camera module mesh, USB lead) around the profile's wrist camera (visual only, no collision).
"""
from __future__ import annotations

from dataclasses import dataclass

from sim.twin.visuals.materials import bind, preview_material


@dataclass
class RobotLook:
    # sRGB 0-255 as seen by the cameras (jaw colour: median of red finger pixels in real wrist frames; body: dark
    # robot-base pixels in the real top frames read (8, 24, 42) -> a slightly blue-tinted near-black)
    body: tuple = (24, 27, 33)
    body_roughness: float = 0.55
    servo: tuple = (20, 20, 23)
    servo_roughness: float = 0.4
    jaw: tuple = (165, 42, 55)
    jaw_roughness: float = 0.3
    jaw_clearcoat: float = 0.3
    mount: tuple = (215, 118, 50)
    mount_roughness: float = 0.45
    board: tuple = (25, 25, 28)


JAW_PREFIXES = ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1")
SERVO_PREFIXES = ("sts3215",)


def _is_visual_instance(prim) -> bool:
    """An instanceable prim whose prototype holds a renderable (non-guide) Mesh."""
    from pxr import Usd, UsdGeom

    if not prim.IsInstance():
        return False
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.IsA(UsdGeom.Mesh):
            purpose = UsdGeom.Imageable(p).ComputePurpose()
            if purpose != UsdGeom.Tokens.guide:
                return True
    return False


def robot_roots(stage, num_envs: int, robot: str = "Robot"):
    return [r for r in (stage.GetPrimAtPath(f"/World/envs/env_{e}/{robot}") for e in range(num_envs)) if r]


def paint_robot(num_envs: int = 1, look: RobotLook | None = None, robot: str = "Robot",
                looks_root: str = "/World/Looks/Twin") -> dict:
    """De-instance the visual link prims and bind body / servo / jaw materials. Returns counts per material."""
    import omni.usd
    from pxr import Usd, UsdGeom

    look = look or RobotLook()
    stage = omni.usd.get_context().get_stage()
    mats = {
        "body": preview_material(stage, f"{looks_root}/RobotBody", look.body, look.body_roughness),
        "servo": preview_material(stage, f"{looks_root}/RobotServo", look.servo, look.servo_roughness),
        "jaw": preview_material(stage, f"{looks_root}/RobotJaw", look.jaw, look.jaw_roughness,
                                clearcoat=look.jaw_clearcoat),
    }
    counts = {k: 0 for k in mats}
    finray = _finray_model()
    for root in robot_roots(stage, num_envs, robot):
        visual = [p for p in Usd.PrimRange(root) if _is_visual_instance(p)]
        for p in visual:
            name = p.GetName()
            kind = "jaw" if name.startswith(JAW_PREFIXES) else "servo" if name.startswith(SERVO_PREFIXES) else "body"
            p.SetInstanceable(False)
            for m in Usd.PrimRange(p):
                if m.IsA(UsdGeom.Mesh):
                    bind(m, mats[kind])
            bind(p, mats[kind])
            counts[kind] += 1
        if finray is not None:  # the real fingers: black printed bases + red TPU fins instead of the stock jaws
            counts["finray_meshes"] = counts.get("finray_meshes", 0) + finray.replace_visuals(
                stage, root, mats["body"], mats["jaw"], bind, extend=_finray_extend(), **_finray_bend())
    return counts


def _finray_extend() -> float:
    """The physics layer's fin mount offset (params.json gripper.finray.extend; 0 = the CAD mount)."""
    from sim.twin.physics import apply as phys

    return float(phys.summary()["gripper"]["finray"].get("extend", 0.0))


def _finray_bend() -> dict:
    """The physics layer's bending-fin segmentation (params.json gripper.finray.bend; hinges 0 = rigid fins), as
    keyword arguments of model.replace_visuals."""
    from sim.twin.physics import apply as phys

    b = phys.fin_bend()
    if not b:
        return {"hinges": 0}
    return {"hinges": int(b["hinges"]), "slices": bool(b.get("slices", False)),
            "root_len": float(b.get("root_len", 0.012))}


def _finray_model():
    """sim/twin/gripper/model.py when the physics layer's active fingers are the fin-ray ones, else None (stock
    jaw visuals, painted red)."""
    try:
        from sim.twin.gripper import model
        from sim.twin.physics import apply as phys
    except ImportError:
        return None
    return model if phys.finray() else None


def add_wrist_camera_mount(num_envs: int, wrist_rows, look: RobotLook | None = None,
                           gripper_rel: str = "Robot/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/"
                                              "wrist_link/gripper_link",
                           looks_root: str = "/World/Looks/Twin") -> int:
    """The real wrist-camera mount (sim/twin/robot/wrist_mount.py) on gripper_link, visual only: orange printed frame
    + bent neck + screw tab on the follower's M3 holes, black screws, the 32x32 camera module (official mesh), USB lead.

    wrist_rows: the rendered wrist camera's USD row matrix in gripper_link (calib.Profile.wrist.rows). The mount is
    built around the physical camera behind it (the sim camera is a centred pinhole; the physical one is the same
    pose rotated by wrist_mount.json's principal-point correction), and its neck is anchored to the screw holes, so it
    stays attached for any profile / DR draw. Returns the number of meshes defined."""
    import omni.usd

    from sim.twin.robot.wrist_mount import load, mount_parts, rows_to_cv
    from sim.twin.visuals.assets import define_mesh

    look = look or RobotLook()
    stage = omni.usd.get_context().get_stage()
    mats = {"plate": preview_material(stage, f"{looks_root}/CamMount", look.mount, look.mount_roughness),
            "board": preview_material(stage, f"{looks_root}/CamBoard", look.board, 0.4)}
    mats["screws"] = mats["cable"] = mats["board"]
    m = load()
    R_sim, t = rows_to_cv(wrist_rows)
    parts = mount_parts(m.R_cv @ m.R_sim.T @ R_sim, t, m.dims)
    n = 0
    for e in range(num_envs):
        g = stage.GetPrimAtPath(f"/World/envs/env_{e}/{gripper_rel}")
        if not g:
            continue
        for name, (pts, nrm, counts, idx) in parts.items():
            mesh = define_mesh(stage, f"{g.GetPath()}/WristMount_{name}", pts, nrm, counts, idx)
            bind(mesh.GetPrim(), mats[name])
            n += 1
    return n
