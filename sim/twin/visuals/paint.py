"""Paint the converted SO-101 USD (sim/twin/assets/so101/usd/so101_new_calib) like the real robot.

Why the residual env's `paint_robot` leaves it yellow: the URDF converter (v0.3.2) writes every link mesh as an
*instanceable* Xform (e.g. `gripper_link/wrist_roll_follower_so101_v1`) referencing `instances.usda`, and the Mesh
prims live inside the instance prototype (instance proxies: no authoring allowed, and `IsA(UsdGeom.Mesh)` never sees
them from a normal traversal). The prototype binds `tn__3d_printed_` (yellow) / `sts3215` (black). Here we
de-instance only the *visual* instance prims (the collision twins `<name>_1` are purpose=guide and stay instanced),
then bind our own materials on their meshes.

Real robot (phone photos + recorded frames): matte black printed body and servos, red / orange-red glossy jaws
(fixed jaw = wrist_roll_follower_so101_v1 on gripper_link, moving jaw = moving_jaw_so101_v1), and an orange printed
wrist-camera mount plate carrying a black camera board on the gripper. The URDF has no camera-mount mesh, so
`add_wrist_camera_mount` adds a simple plate + board at the profile's wrist-camera pose (visual only, no collision).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    return counts


def _box_points(half, center, rot):
    """8 corners of an oriented box: rot columns = box axes in the parent frame."""
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return (s * half) @ np.asarray(rot).T + center


def add_wrist_camera_mount(num_envs: int, wrist_rows, look: RobotLook | None = None,
                           gripper_rel: str = "Robot/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/"
                                              "wrist_link/gripper_link",
                           plate=(0.042, 0.050, 0.004), board=(0.032, 0.032, 0.003),
                           plate_offset=(0.0, 0.004, 0.0), looks_root: str = "/World/Looks/Twin") -> int:
    """Orange mount plate + black camera board behind the wrist camera (gripper_link frame, visual only).

    wrist_rows: the wrist camera's USD row matrix in the gripper_link frame (calib.Profile.wrist.rows): rows 0-2 =
    camera x / y / -optical axes, row 3 = position. Plate and board lie in the camera's image plane, just behind the
    lens (so they never enter the wrist camera's view): plate first, camera board on its back. plate_offset shifts the plate in camera (x, y_up) metres."""
    import omni.usd
    from pxr import Gf, UsdGeom

    from sim.twin.visuals.assets import box_mesh, define_mesh

    look = look or RobotLook()
    stage = omni.usd.get_context().get_stage()
    m_plate = preview_material(stage, f"{looks_root}/CamMount", look.mount, look.mount_roughness)
    m_board = preview_material(stage, f"{looks_root}/CamBoard", look.board, 0.4)
    R = np.asarray(wrist_rows, float)
    ax, ay, az = R[0, :3], R[1, :3], R[2, :3]  # az = backwards (USD camera looks along -z)
    rot = np.stack([ax, ay, az], 1)
    c = R[3, :3]
    n = 0
    for e in range(num_envs):
        g = stage.GetPrimAtPath(f"/World/envs/env_{e}/{gripper_rel}")
        if not g:
            continue
        # real mount (photos): the lens looks through the orange plate, the black camera board sits on the plate's
        # back (visible from above). Plate from 1 mm behind the lens, board behind the plate.
        cp = c + az * (0.001 + plate[2] / 2) + ax * plate_offset[0] + ay * plate_offset[1]
        cb = c + az * (0.001 + plate[2] + board[2] / 2)
        for name, size, ctr, mat in (("CamBoard", board, cb, m_board), ("CamMountPlate", plate, cp, m_plate)):
            pts, nrm, counts, idx = box_mesh(size)
            pts = pts @ rot.T + ctr
            nrm = nrm @ rot.T
            mesh = define_mesh(stage, f"{g.GetPath()}/{name}", pts, nrm, counts, idx)
            bind(mesh.GetPrim(), mat)
            n += 1
    return n
