"""Isaac Lab scene for the SO-101 brick task: N cloned envs built from Joe's Isaac scene geometry.

Per env: SO-101 (Joe's converted URDF USD, fixed base at the env origin), table (Joe's cube), target brick (2x2) +
distractor bricks, three plates (bowls), top camera (fixed) and wrist camera (on gripper_link). Poses/intrinsics come
from Joe's scene-final.usda / camera-fit.json; layout and randomisation ranges can be overridden by
sim/realism/scene_params.json (stage A2) via `load_layout`.
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

JOE = Path(os.environ.get("DLSJ_JOE_ROOT", "/home/joe-spark-2/pi05-isaac-so101"))
ROBOT_USD = str(JOE / "assets/converted/so101_new_calib_2/so101_new_calib.usda")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER_LINK = "Robot/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link"
JAW_LINK = GRIPPER_LINK + "/moving_jaw_so101_v1_link"

# Joe's scene-final.usda (row-vector USD matrices: rows = camera x/y/z axes in the parent frame, row 4 = position)
TOP_CAM_M = np.array([[-0.12946073859057658, 0.9913676740636833, 0.02073769478860113, 0],
                      [-0.9324356225862437, -0.12882673169332085, 0.3375907032686496, 0],
                      [0.3373480797274333, 0.024368176435258765, 0.9410645382126742, 0],
                      [0.465762347632199, 0.026429952710681763, 0.6473618296007448, 1]])
TOP_CAM_INTR = dict(focal_length=20.0, horizontal_aperture=20.92898, vertical_aperture=15.696734)
# Joe's OverviewCamera (scene-final.usda), USD row-vector transform, robot base at the origin; rendered 16:9 for videos.
OVERVIEW_CAM_M = np.array([[0.752576694706878, 0.6585046078685182, 0.0, 0],
                           [-0.28935841100753085, 0.3306953268657496, 0.8982829792266755, 0],
                           [0.591523480990626, -0.6760268354178585, 0.43941744302160796, 0],
                           [0.70, -0.80, 0.65, 1]])
OVERVIEW_INTR = {"focal_length": 18.0, "horizontal_aperture": 24.0, "vertical_aperture": 13.5}
WRIST_CAM_M = np.array([[-0.9609081717487479, 0.017280304596569828, 0.2763274806086624, 0],
                        [-0.13272283184622244, -0.9046498261338383, -0.4049609141420139, 0],
                        [0.24298175934254865, -0.42580521738121774, 0.8715789014643105, 0],
                        [0.0, -0.073, 0.0066, 1]])
WRIST_CAM_INTR = dict(focal_length=20.0, horizontal_aperture=36.57143, vertical_aperture=27.428572)
INSPECT_CAM_M = np.array([[-0.999530856702204, 0.03062787129655299, 0, 0],
                          [0.0066718593539116895, 0.21773401198020886, 0.9759853412422564, 0],
                          [0.029892353418890183, 0.9755274642606655, -0.2178362083773864, 0],
                          [0.19372239888012727, 0.30782613116852947, 0.09, 1]])
INSPECT_CAM_INTR = dict(focal_length=20.0, horizontal_aperture=27.234043, vertical_aperture=20.425531)
TABLE_TOP_Z = -0.002  # Joe's table cube: centre z -0.027, height 0.05


def mat_to_quat_xyzw(m_rows: np.ndarray) -> tuple[float, float, float, float]:
    """USD row-vector 4x4 -> (x, y, z, w) quaternion of the frame's rotation (columns = frame axes)."""
    r = np.asarray(m_rows)[:3, :3].T
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def _euler_xyz_to_quat_xyzw(rx, ry, rz):
    """USD rotateXYZ (degrees, applied X then Y then Z) -> (x, y, z, w)."""
    def q(axis, deg):
        h = math.radians(deg) / 2
        v = [0.0, 0.0, 0.0]
        v[axis] = math.sin(h)
        return np.array([*v, math.cos(h)])

    def mul(a, b):  # Hamilton product, xyzw
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                         aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])

    r = mul(q(2, rz), mul(q(1, ry), q(0, rx)))
    return tuple(float(v) for v in r / np.linalg.norm(r))


@dataclass
class Brick:
    name: str
    size: tuple[float, float, float]  # m (x, y, z)
    pos: tuple[float, float]  # nominal table xy (env frame)
    color: tuple[float, float, float]
    mass: float  # kg


@dataclass
class Layout:
    """Scene layout + randomisation (env frame, metres). Defaults = Joe's episode-0 fit; A2 can override."""

    target: Brick = field(default_factory=lambda: Brick("target", (0.0318, 0.0318, 0.0192), (0.1719, -0.1064),
                                                        (0.43, 0.73, 0.55), 0.0093))
    distractors: list[Brick] = field(default_factory=lambda: [
        Brick("blue_brick", (0.0636, 0.0318, 0.0192), (0.2431, -0.0675), (0.02, 0.64, 0.83), 0.018),
        Brick("yellow_brick", (0.0636, 0.0318, 0.0192), (0.2410, -0.0007), (0.82, 0.72, 0.26), 0.018),
        Brick("cream_brick", (0.0318, 0.0318, 0.0192), (0.1966, 0.0046), (0.73, 0.76, 0.48), 0.0093),
    ])
    # bowl name -> (x, y, radius, rgb). purple/orange from Joe's fitted plates; yellow is a placeholder until A2.
    bowls: dict = field(default_factory=lambda: {
        "purple": (0.3854, -0.1140, 0.095, (0.45, 0.35, 0.80)),
        "orange": (0.3218, 0.0883, 0.095, (0.86, 0.25, 0.045)),
        "yellow": (0.52, 0.17, 0.095, (0.95, 0.85, 0.15)),
    })
    target_xy_jitter: float = 0.04  # +- m, uniform
    target_yaw_range: float = math.pi  # +- rad
    distractor_xy_jitter: float = 0.03
    init_joint_deg: tuple = (24.26, -58.07, 67.91, 67.03, 31.34, 11.57)  # Joe's recorded episode-0 start
    init_joint_jitter_deg: tuple = (8.0, 6.0, 6.0, 6.0, 8.0, 3.0)
    inspection_pose_deg: tuple | None = None  # median joint state at the inspection mark (A2)
    source: str = "defaults (Joe's episode-0 fit)"
    # ---- data-derived sampling (A2: sim/realism/scene_params.json); None = use the uniform jitter above
    target_xy_std: tuple | None = None
    target_xy_box: tuple | None = None  # (lo_xy, hi_xy) clip (p5, p95)
    distractor_xy: tuple | None = None  # (mean, std, lo, hi)
    distractor_count: tuple = (1, 3)
    plate_box: tuple | None = None  # (lo_xy, hi_xy) for plate centres; None = fixed bowls
    start_joint: tuple | None = None  # (median, std, lo, hi) over 6 joints
    inspection_tip: tuple | None = None  # median fingertip xyz at the inspection mark
    gripper_hold_real: float = 5.75
    top_cam_rows: list | None = None  # refined top camera (USD rows) + intrinsics
    top_cam_intr: dict | None = None


def load_layout(path: str | Path | None, camera_source: str = "joe") -> Layout:
    """Layout defaults, overridden by stage A2's scene_params.json when it exists (data-derived distributions)."""
    lay = Layout()
    if not path or not Path(path).exists():
        return lay
    p = json.loads(Path(path).read_text())
    lay.source = str(path)

    def stat(d):
        return tuple(d["median"]), tuple(d["std"]), tuple(d["p5"]), tuple(d["p95"])

    if "target_brick_xy_from_fk_at_grasp" in p:
        med, std, lo, hi = stat(p["target_brick_xy_from_fk_at_grasp"])
        lay.target.pos, lay.target_xy_std, lay.target_xy_box = med, std, (lo, hi)
    tb = p.get("table_bricks_at_start", {})
    if "xy" in tb:
        lay.distractor_xy = stat(tb["xy"])
    if "count_per_episode" in tb:
        c = tb["count_per_episode"]
        lay.distractor_count = (int(max(0, c.get("p10", 1))), int(min(3, max(1, c.get("p90", 3)))))
    pl = p.get("plates", {})
    if "centre_xy_all_sessions" in pl:
        _, _, lo, hi = stat(pl["centre_xy_all_sessions"])
        lay.plate_box = (lo, hi)
        radii = pl.get("radius_m", {}).get("by_colour_median_m", {})
        for name, key in (("yellow", "yellow"), ("orange", "orange"), ("purple", "blue(=purple bowl)")):
            if key in radii and name in lay.bowls:
                x, y, _, rgb = lay.bowls[name]
                lay.bowls[name] = (x, y, float(radii[key]), rgb)
    if "start_joint_state_deg" in p:
        lay.start_joint = stat(p["start_joint_state_deg"])
        lay.init_joint_deg = lay.start_joint[0]
    ins = p.get("inspection", {})
    if "joint_state_deg" in ins:
        lay.inspection_pose_deg = tuple(ins["joint_state_deg"]["median"])
    if "tip_xyz" in ins:
        lay.inspection_tip = tuple(ins["tip_xyz"]["median"])
    g = p.get("gripper", {}).get("holding_value_after_grasp", {})
    if "median" in g:
        lay.gripper_hold_real = float(g["median"])
    # top camera: Joe's camera-fit.json by default. A2's refinement was rejected (it absorbed joint-offset /
    # segmentation error: up to 8.5 deg, 64-90 mm, focal x0.93; held-out 50->43 px) -> opt-in only.
    rc = p.get("top_camera", {}).get("realism_check", {}).get("refined_camera")
    if camera_source == "a2_refined" and rc and "world_to_cv" in rc:
        w2c = np.asarray(rc["world_to_cv"])  # rows = OpenCV camera axes in world (x right, y down, z forward)
        rows = np.eye(4)
        rows[0, :3], rows[1, :3], rows[2, :3] = w2c[0], -w2c[1], -w2c[2]  # USD camera: y up, looks along -z
        rows[3, :3] = rc["position"]
        f = float(rc["focal_px"])
        lay.top_cam_rows = rows.tolist()
        lay.top_cam_intr = dict(focal_length=20.0, horizontal_aperture=20.0 * 640 / f, vertical_aperture=20.0 * 480 / f)
    return lay


def _robot_usd(profile) -> str:
    if profile is not None and getattr(profile, "robot_usd", None):
        from calib import check_robot_usd

        return check_robot_usd(profile.robot_usd)
    return ROBOT_USD


def build_scene_cfg(num_envs: int, layout: Layout, cameras: bool, env_spacing: float = 2.5,
                    drive_stiffness: float = 60.0, drive_damping: float = 2.0, effort_limit: float = 2.9,
                    profile=None, overview: bool = False):
    """InteractiveSceneCfg with robot, table, bricks, bowls (+ optional wrist/top cameras)."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg

    off = np.array([0, -0.2756362497772003, 0.056384559242785785, -0.45744995615777856, 0, 0])
    gup = math.radians(100.000046)
    init = np.deg2rad(np.array(layout.init_joint_deg[:5])) + off[:5]
    init_pos = {j: float(v) for j, v in zip(JOINTS[:5], init)}
    init_pos["gripper"] = float(layout.init_joint_deg[5] / 100 * gup)

    cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=False)  # USD root_joint anchors to world: replicated clones would all be pinned to env_0
    cfg.dome = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=700.0))
    # Joe's Sun: intensity 600, rotateXYZ (25, -30, 20) deg -> xyzw quaternion
    cfg.sun = AssetBaseCfg(prim_path="/World/Sun", spawn=sim_utils.DistantLightCfg(intensity=600.0, angle=0.53),
                           init_state=AssetBaseCfg.InitialStateCfg(rot=_euler_xyz_to_quat_xyzw(25, -30, 20)))
    cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_robot_usd(profile),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16,
                solver_velocity_iteration_count=4),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos=init_pos),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=JOINTS[:5], stiffness=drive_stiffness,
                                       damping=drive_damping, effort_limit_sim=effort_limit,
                                       velocity_limit_sim=math.radians(240.0)),
            "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper"], stiffness=drive_stiffness,
                                           damping=drive_damping,
                                           effort_limit_sim=(profile.gripper_effort if profile is not None and
                                                             getattr(profile, "gripper_effort", None) else effort_limit),
                                           velocity_limit_sim=math.radians(240.0)),
        },
    )
    cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.35, 1.3, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.43, 0.32), roughness=0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, 0.0, -0.027)),
    )

    def brick_cfg(b: Brick, prim: str) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + prim,
            spawn=sim_utils.CuboidCfg(
                size=b.size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=b.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                # ABS brick vs 3D-printed jaw / wood: static ~0.6, dynamic ~0.5 (PLA/ABS literature values)
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=b.color, roughness=0.4)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(b.pos[0], b.pos[1], TABLE_TOP_Z + b.size[2] / 2)),
        )

    cfg.brick = brick_cfg(layout.target, "Brick")
    for i, d in enumerate(layout.distractors):
        setattr(cfg, f"distractor{i}", brick_cfg(d, f"Distractor{i}"))
    for name, (x, y, r, rgb) in layout.bowls.items():
        setattr(cfg, f"bowl_{name}", RigidObjectCfg(  # kinematic: repositioned per episode (plates move in the data)
            prim_path="{ENV_REGEX_NS}/Bowl_" + name,
            spawn=sim_utils.CylinderCfg(
                radius=r, height=0.006, axis="Z", collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.25)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, TABLE_TOP_Z + 0.003))))
    if cameras:
        def cam(prim, m, intr):
            return CameraCfg(
                prim_path=prim, update_period=0.0, height=480, width=640, data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=intr["focal_length"],
                                                 horizontal_aperture=intr["horizontal_aperture"],
                                                 vertical_aperture=intr["vertical_aperture"],
                                                 clipping_range=(0.005, 20.0)),
                offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in m[3, :3]), rot=mat_to_quat_xyzw(m),
                                           convention="opengl"))
        if profile is not None and profile.top is not None:  # calibration profile (calib.py) wins
            top_m, top_i = np.asarray(profile.top.rows), profile.top.intr
        else:
            top_m = np.asarray(layout.top_cam_rows) if layout.top_cam_rows else TOP_CAM_M
            top_i = layout.top_cam_intr or TOP_CAM_INTR
        cfg.top_cam = cam("{ENV_REGEX_NS}/TopCam", top_m, top_i)
        if profile is not None and profile.wrist is not None:
            wr_m, wr_i = np.asarray(profile.wrist.rows), profile.wrist.intr
        else:
            wr_m, wr_i = WRIST_CAM_M, WRIST_CAM_INTR
        cfg.wrist_cam = cam("{ENV_REGEX_NS}/" + GRIPPER_LINK + "/WristCam", wr_m, wr_i)
    if overview:  # video rendering only (1280x720 RGB)
        from isaaclab.sensors import CameraCfg as _CamCfg

        m = OVERVIEW_CAM_M
        cfg.overview_cam = _CamCfg(
            prim_path="{ENV_REGEX_NS}/OverviewCam", update_period=0.0, height=720, width=1280, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=OVERVIEW_INTR["focal_length"],
                                             horizontal_aperture=OVERVIEW_INTR["horizontal_aperture"],
                                             vertical_aperture=OVERVIEW_INTR["vertical_aperture"],
                                             clipping_range=(0.01, 20.0)),
            offset=_CamCfg.OffsetCfg(pos=tuple(float(v) for v in m[3, :3]), rot=mat_to_quat_xyzw(m),
                                     convention="opengl"))
    return cfg
