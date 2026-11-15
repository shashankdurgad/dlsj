"""Live mirror of the real SO-101: poll the follower's pose, pose the twin kinematically, render a hero view, stream
MJPEG (see README.md).

    sim/twin/stream/run_live.sh [--pose_url URL] [--look real|minimal] ...     # detached container (normal use)
    sim/twin/run.sh sim/twin/stream/live.py --pose_url http://127.0.0.1:8041/api/pose   # foreground, fake feed

Process layout: this script forks the HTTP / JPEG / pose-poller child (server.py) BEFORE Isaac boots, then builds
TwinEnv (the twin's scene, materials, fitted lighting, fin-ray gripper, VisualSync) with the three robot cameras
replaced by one hero camera. Every frame: critically damped filter towards the latest pose -> joint state written
straight into PhysX (no physics step; `sim.forward()` updates the link poses) -> VisualSync -> render -> shared
memory. Bricks and plates never move (no physics stepping), so the arm cannot knock them over.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim.twin.stream.server import S_BLACK, S_FRAMES, S_HEARTBEAT, S_LOOP_MS, S_READY, S_RENDER_MS, Shared, serve

OUT = REPO / "sim/twin/out/stream"
REST_DEG = [16.1, -101.9, 90.5, 7.7, 2.9, 1.1]  # until the first pose arrives (the real follower at rest, cal2)
# hero views (eye, target) in the env frame (robot base at the origin on the table's back edge, table toward +x, plates
# at y -0.1..-0.3 in the real data, inspect camera at y +0.18, inspect backdrop billboard at y ~ -0.6)
HEROES = {
    "hero": ((0.625, 0.128, 0.453), (0.24, -0.12, 0.15)),  # 3/4 from the inspect side; working reach ~60 % height
    "hero_low": ((0.66, -0.30, 0.30), (0.12, 0.02, 0.14)),
    "front_right": ((0.80, -0.52, 0.46), (0.16, 0.02, 0.10)),
    "front_left": ((0.78, 0.55, 0.46), (0.16, -0.02, 0.10)),
    "front": ((0.95, 0.05, 0.55), (0.15, 0.0, 0.10)),
    "side_right": ((0.15, -0.85, 0.42), (0.18, 0.0, 0.12)),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose_url", default="http://127.0.0.1:8040/api/pose",
                    help="robot pose feed (on watson: the Mac's SSH tunnel to the XPS /api/pose)")
    ap.add_argument("--bind", default="127.0.0.1",
                    help="listen address (never 0.0.0.0 by default); 'auto' = Tailscale IPv4, else 127.0.0.1")
    ap.add_argument("--port", type=int, default=8030)
    ap.add_argument("--look", choices=("real", "minimal"), default="real")
    ap.add_argument("--bg", choices=("dark", "hall"), default="dark",
                    help="real look, behind the table: 'dark' = dark room (hall panorama black, floor near-black), "
                         "'hall' = the twin's hall panorama + blue floor (the inspect backdrop is always hidden)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--max_fps", type=float, default=30.0)
    ap.add_argument("--smooth_ms", type=float, default=100.0, help="critically damped filter rise time")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality")
    ap.add_argument("--hero", default="hero", help=f"named view {sorted(HEROES)} or 'ex,ey,ez,tx,ty,tz'")
    ap.add_argument("--focal", type=float, default=18.0, help="hero focal length mm (35 mm-ish aperture 20.955)")
    ap.add_argument("--seed", type=int, default=3, help="layout seed (bricks / plates as the real data has them)")
    ap.add_argument("--render_passes", type=int, default=1)
    ap.add_argument("--aa", type=int, default=3,
                    help="RTX /rtx/post/aa/op: 3 = DLSS quality (fast, the stream default), 4 = DLAA (the twin's fit)")
    ap.add_argument("--black_max", type=int, default=12, help="frames whose max pixel <= this are skipped")
    ap.add_argument("--duration", type=float, default=0.0, help="exit after N s (0 = run forever)")
    ap.add_argument("--tune", action="store_true", help="render the hero views (hall / dark / minimal) to out/stream/")
    ap.add_argument("--tune_views", default="", help="--tune: ';'-separated 'ex,ey,ez,tx,ty,tz' views instead")
    return ap.parse_args(argv)


def hero_pose(spec: str):
    if spec in HEROES:
        return HEROES[spec]
    v = [float(x) for x in spec.split(",")]
    return tuple(v[:3]), tuple(v[3:6])


def look_at_rows(eye, target) -> np.ndarray:
    """USD row-vector camera matrix (rows: x right, y up, z back; row 3 = position), -Z looks at target, +Z up."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = x, y, z, eye
    return m


class CritDamped:
    """Critically damped 2nd-order follower (exact update for a piecewise-constant target), per joint."""

    def __init__(self, x0, rise_s: float):
        self.x = np.asarray(x0, float).copy()
        self.v = np.zeros_like(self.x)
        self.w = 2.15 / max(1e-3, rise_s)  # 63 % rise time of a critically damped system ~ 2.15 / omega

    def __call__(self, target, dt: float) -> np.ndarray:
        dt = min(max(dt, 0.0), 0.25)
        e0 = self.x - np.asarray(target, float)
        a = self.v + self.w * e0
        k = math.exp(-self.w * dt)
        self.x = np.asarray(target, float) + (e0 + a * dt) * k
        self.v = (self.v - self.w * a * dt) * k
        return self.x


def patch_scene(width: int, height: int, eye, target, focal: float):
    """TwinEnv's scene cfg with the top / wrist / inspect cameras replaced by one hero camera (render cost)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    import sim.twin.visuals.twin_scene as ts
    from sim.twin.visuals.twin_scene import _rscene

    orig = ts.twin_scene_cfg

    def twin_scene_cfg(*a, **k):
        cfg = orig(*a, **k)
        for n in ("top_cam", "wrist_cam", "inspect_cam"):
            if hasattr(cfg, n):
                setattr(cfg, n, None)
        ha = 20.955
        cfg.hero_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/HeroCam", update_period=0.0, width=width, height=height, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=ha,
                                             vertical_aperture=ha * height / width, clipping_range=(0.01, 30.0)),
            offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in eye),
                                       rot=_rscene().mat_to_quat_xyzw(look_at_rows(eye, target)), convention="opengl"))
        return cfg

    ts.twin_scene_cfg = twin_scene_cfg


def set_hero(stage, eye, target, env: int = 0):
    """Move the hero camera at run time (USD xform; the GB10 render path reads USD)."""
    from pxr import Gf

    from sim.twin.visuals.usd_sync import _parent_world, set_local

    prim = stage.GetPrimAtPath(f"/World/envs/env_{env}/HeroCam")
    set_local(prim, Gf.Matrix4d(look_at_rows(eye, target).tolist()) * _parent_world(prim).GetInverse())


def apply_minimal(stage, look, env: int = 0):
    """Black void, light matte arm, soft key + rim light: hide everything but the robot visual, switch every light
    off, add our own two distant lights."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    keep = {"RobotVisual", "HeroCam", "Robot"}
    for child in stage.GetPrimAtPath(f"/World/envs/env_{env}").GetChildren():
        if child.GetName() not in keep and child.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(child).MakeInvisible()
    for p in stage.Traverse():
        if p.HasAPI(UsdLux.LightAPI) or "Light" in p.GetTypeName():
            a = p.GetAttribute("inputs:intensity")
            if a:
                a.Set(0.0)
            if p.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(p).MakeInvisible()
        elif p.GetPath().pathString.startswith("/World/") and p.GetPath().pathElementCount == 2 and \
                p.GetName() not in ("envs", "Looks", "MinimalLights") and p.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(p).MakeInvisible()  # ground plane, hall, etc.
    s = look.scale()
    base = float(look.key_intensity) * s

    def distant(name, d, inten, rgb, angle):
        from sim.twin.visuals.twin_scene import quat_z_to

        lt = UsdLux.DistantLight.Define(stage, f"/World/MinimalLights/{name}")
        lt.CreateIntensityAttr(inten)
        lt.CreateAngleAttr(angle)
        lt.CreateColorAttr(Gf.Vec3f(*rgb))
        x, y, z, w = quat_z_to(d)
        xf = UsdGeom.Xformable(lt.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    distant("Key", (0.6, -0.5, 0.8), base * 3.5, (1.0, 0.97, 0.93), 25.0)
    distant("Fill", (0.4, 0.8, 0.3), base * 1.0, (0.85, 0.9, 1.0), 40.0)
    distant("Rim", (-0.8, 0.3, 0.45), base * 4.0, (0.9, 0.95, 1.0), 8.0)
    mat = UsdShade.Material.Define(stage, "/World/Looks/MinimalArm")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/MinimalArm/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.85, 0.85, 0.87))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    sh.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.25)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    root = stage.GetPrimAtPath(f"/World/envs/env_{env}/RobotVisual")
    UsdShade.MaterialBindingAPI.Apply(root).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def apply_background(stage, bg: str, env: int = 0):
    """Live-mirror background (this process only; the twin's defaults are untouched). The inspect backdrop billboard
    (a photo projected for the inspect camera) only makes sense from that camera: hide it. bg='dark': the hall
    cylinder's unlit panorama goes black and the floor beyond the table edge near-black, so the lit table reads as
    the real table in a dark room; 'hall': keep the panorama and the floor."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    def hide(path):
        p = stage.GetPrimAtPath(path)
        if p and p.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(p).MakeInvisible()

    hide(f"/World/envs/env_{env}/BackdropVisual")
    if bg != "dark":
        return
    tex = stage.GetPrimAtPath("/World/Looks/Twin/Hall/Tex")
    if tex and tex.GetAttribute("inputs:scale"):
        tex.GetAttribute("inputs:scale").Set(Gf.Vec4f(0.0, 0.0, 0.0, 1.0))
    else:  # no panorama material: hide the cylinder
        hide(f"/World/envs/env_{env}/HallVisual")
    mat = UsdShade.Material.Define(stage, "/World/Looks/LiveDarkFloor")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/LiveDarkFloor/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.004, 0.004, 0.005))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    floor = stage.GetPrimAtPath(f"/World/envs/env_{env}/TwinProps/Floor")
    if floor:
        UsdShade.MaterialBindingAPI.Apply(floor).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


class Mirror:
    def __init__(self, args, sh: Shared):
        self.args, self.sh = args, sh
        from sim.twin.env.twin_env import launch_app

        launch_app(headless=True, cameras=True)  # Kit first: isaaclab.sim needs it (GB10 KIT_ARGS from app.py)
        eye, target = hero_pose(args.hero)
        patch_scene(args.width, args.height, eye, target, args.focal)
        from sim.twin.env import TwinEnv, TwinEnvCfg

        cfg = TwinEnvCfg(cameras=(), color_correct=False, play=False, randomize=False, layout="sample",
                         max_relative_target=1000.0)
        self.env = env = TwinEnv(cfg)
        import omni.usd
        import torch

        self.torch = torch
        self.stage = omni.usd.get_context().get_stage()
        env.reset(seed=args.seed, render=False, randomize=False)
        self.robot = env.residual.robot
        self.cam = env.scene.sensors["hero_cam"]
        self.q_full = self.robot.data.joint_pos.clone()  # passive joints (fin-ray segments) keep the settled state
        self.zeros = torch.zeros_like(self.q_full)
        lim = getattr(self.robot.data, "soft_joint_pos_limits", None)
        self.lim = lim.clone() if lim is not None else None
        if args.look == "minimal":
            apply_minimal(self.stage, env.look)
        elif not args.tune:
            apply_background(self.stage, args.bg)
        import carb

        carb.settings.get_settings().set("/rtx/post/aa/op", int(args.aa))
        self.split = np.zeros(3)  # EMA ms: sync, sim.render, fetch

    def pose(self, deg):
        torch = self.torch
        r = self.env.residual
        d = torch.as_tensor(np.asarray(deg, np.float32), device=self.env.device).reshape(1, 6)
        q = self.q_full.clone()
        q[:, r.jidx] = r.to_sim(d)
        if self.lim is not None:
            q = torch.maximum(torch.minimum(q, self.lim[..., 1]), self.lim[..., 0])
        self.robot.write_joint_state_to_sim(q, self.zeros)
        self.robot.set_joint_position_target(q)
        self.robot.write_data_to_sim()
        self.env.sim.forward()  # link poses from the joint state, no physics step
        self.robot.update(1.0 / 120.0)

    def render(self):
        """RGB uint8 (H, W, 3) numpy, or None for an all-black frame."""
        env = self.env
        t0 = time.perf_counter()
        env.sync()
        t1 = time.perf_counter()
        for _ in range(max(1, self.args.render_passes)):
            env.sim.render()
        t2 = time.perf_counter()
        env.scene.update(0.0)
        img = self.cam.data.output["rgb"][0, ..., :3]
        black = int(img.max()) <= self.args.black_max
        out = None if black else img.detach().cpu().numpy().astype(np.uint8)
        t3 = time.perf_counter()
        self.split = 0.9 * self.split + 0.1 * np.array([t1 - t0, t2 - t1, t3 - t2]) * 1e3
        return out

    def gripper_xyz(self):
        r = self.env.residual
        return (r.robot.data.body_pos_w[0, r.grip_body] - self.env.scene.env_origins[0]).detach().cpu().numpy()


def tune(m: Mirror, args):
    """Every hero view at several arm poses: real look with the hall / dark background, then minimal ->
    out/stream/tune_<look>_<view>_<pose>.jpg."""
    import cv2

    OUT.mkdir(parents=True, exist_ok=True)
    poses = {"a": [26.42, -51.69, 30.11, 73.23, 15.56, 37.23], "b": [-20.0, -10.0, -20.0, 60.0, 60.0, 70.0],
             "reach": [21.0, 69.6, -85.1, 93.1, -21.7, 46.4], "far": [50.0, 40.0, -50.0, 90.0, 0.0, 40.0],
             "rest": [16.1, -101.9, 90.5, 7.7, 2.9, 1.1]}  # reach / rest: real follower poses
    views = dict(HEROES)
    if args.tune_views:
        views = {f"v{i}": hero_pose(v) for i, v in enumerate(args.tune_views.split(";"))}
    looks = ["hall", "dark", "minimal"] if args.look == "real" else ["minimal"]
    for look in looks:
        if look in ("hall", "dark"):
            apply_background(m.stage, look)
        if look == "minimal" and args.look == "real":
            apply_minimal(m.stage, m.env.look)
        for view in views:
            set_hero(m.stage, *views[view])
            imgs = {}
            for k, p in poses.items():
                m.pose(p)
                img = None
                for _ in range(6):  # a few passes so RTX settles after the jump
                    img = m.render()
                imgs[k] = img
                if img is not None:
                    cv2.imwrite(str(OUT / f"tune_{look}_{view}_{k}.jpg"), img[..., ::-1])
            if imgs["a"] is not None and imgs["b"] is not None:
                d = np.abs(imgs["a"].astype(np.float32) - imgs["b"].astype(np.float32)).mean(-1)
                print(f"[live/tune] {look} {view}: pose a vs b changed {(d > 20).mean() * 100:.1f}% of pixels | "
                      f"gripper {np.round(m.gripper_xyz(), 3).tolist()}", flush=True)
            else:
                print(f"[live/tune] {look} {view}: black frame(s)", flush=True)


def run(m: Mirror, args, sh: Shared, child):
    st = sh.stats
    flt = None
    last_seq = 0
    target = list(REST_DEG)
    t_prev = time.perf_counter()
    t_start = time.time()
    last_frame = time.time()
    last_log = 0.0
    frames = black = 0
    render_ms = loop_ms = 0.0
    period = 1.0 / max(1.0, args.max_fps)
    st[S_READY] = 1.0
    print(f"[live] rendering {args.width}x{args.height} look={args.look} hero={args.hero}", flush=True)
    while True:
        t0 = time.perf_counter()
        seq, _, _, q = sh.read_pose()
        if seq != last_seq:
            last_seq, target = seq, list(q)
        if flt is None:
            flt = CritDamped(target, args.smooth_ms / 1e3)
        cur = flt(target, t0 - t_prev)
        t_prev = t0
        m.pose(cur)
        t1 = time.perf_counter()
        img = m.render()
        t2 = time.perf_counter()
        if img is None:
            black += 1
            st[S_BLACK] = black
        else:
            sh.publish(img)
            frames += 1
            st[S_FRAMES] = frames
            last_frame = time.time()
        render_ms = 0.9 * render_ms + 0.1 * (t2 - t1) * 1e3 if frames > 1 else (t2 - t1) * 1e3
        loop_ms = 0.9 * loop_ms + 0.1 * (t2 - t0) * 1e3 if frames > 1 else (t2 - t0) * 1e3
        st[S_RENDER_MS], st[S_LOOP_MS], st[S_HEARTBEAT] = render_ms, loop_ms, time.time()
        now = time.time()
        if now - last_log > 10:
            last_log = now
            print(f"[live] frames {frames} black {black} render {render_ms:.1f} ms loop {loop_ms:.1f} ms "
                  f"(sync / render / fetch {np.round(m.split, 1).tolist()}) | "
                  f"pose seq {last_seq} q {np.round(cur, 1).tolist()} | gripper "
                  f"{np.round(m.gripper_xyz(), 3).tolist()}", flush=True)
        if not child.is_alive():
            print("[live] HTTP child died; exiting for a restart", flush=True)
            return 3
        if now - last_frame > 60:
            print("[live] no non-black frame for 60 s; exiting for a restart", flush=True)
            return 2
        if args.duration and now - t_start > args.duration:
            return 0
        wait = period - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)


def main():
    args = parse_args()
    if "av" in sys.modules:
        raise SystemExit("PyAV must not be imported in the Isaac process")
    sh = Shared(args.width, args.height)
    # the HTTP / JPEG / pose child is forked before Isaac (Kit, CUDA) exists in this process
    child = mp.get_context("fork").Process(
        target=serve, args=(sh, args.pose_url, args.bind, args.port, args.look, args.quality, args.max_fps),
        daemon=True, name="live-http")
    child.start()
    m = Mirror(args, sh)
    if args.tune:
        tune(m, args)
        code = 0
    else:
        code = run(m, args, sh, child)
    child.terminate()
    sys.stdout.flush()
    os._exit(code)  # skip Kit's slow shutdown; the container restart policy takes over


if __name__ == "__main__":
    main()
