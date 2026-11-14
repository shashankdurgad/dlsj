"""Spin-and-inspect in Joe's SO-101 Isaac scene (read-only use of his USD; runs in yan's own Isaac Sim).

Attaches a Duplo-sized brick with an optional mark (plain/cross/circle) on one side face to the gripper,
puts the arm in an inspect pose, finds the arm pose that centres the brick in the inspect camera,
then steps wrist_roll through N angles and saves the inspect-camera frames.
"""
import argparse, json, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/simsweep/out"))
ap.add_argument("--marks", default="plain,cross,circle")
ap.add_argument("--angles", type=int, default=12)
ap.add_argument("--color", default="0.95,0.80,0.25")
ap.add_argument("--search", type=int, default=1)
args = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 640, "height": 480})

import numpy as np
from PIL import Image, ImageDraw
import omni.usd, omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation

ROOT = Path("/home/joe-spark-2/pi05-isaac-so101")
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
omni.usd.get_context().open_stage(str(ROOT / "outputs/scene-final.usda"))
for _ in range(20): app.update()
stage = omni.usd.get_context().get_stage()
fit = json.loads((ROOT / "assets/camera-fit.json").read_text())
OFF = np.array(fit["joint_offsets_rad"], dtype=np.float32)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# hide the other props' bricks on the table? keep scene as is. Find gripper link.
grip = next(p for p in stage.Traverse() if p.GetName() == "gripper_link" and str(p.GetPath()).startswith("/World/SO101"))
jaw = next(p for p in stage.Traverse() if p.GetName() == "moving_jaw_so101_v1_link" and str(p.GetPath()).startswith("/World/SO101"))
print("gripper", grip.GetPath(), "jaw", jaw.GetPath(), flush=True)

def tex(mark, color, path):
    s = 256; c = tuple(int(255 * float(v)) for v in color.split(","))
    im = Image.new("RGB", (s, s), c); d = ImageDraw.Draw(im); w = 22
    if mark == "cross":
        d.line([(60, 60), (196, 196)], fill=(20, 20, 20), width=w); d.line([(196, 60), (60, 196)], fill=(20, 20, 20), width=w)
    elif mark == "circle":
        d.ellipse([64, 64, 192, 192], outline=(20, 20, 20), width=w)
    im.save(path); return path

def material(name, rgb=None, texture=None):
    path = f"/World/SimSweepMat/{name}"
    m = UsdShade.Material.Define(stage, path); sh = UsdShade.Shader.Define(stage, path + "/S"); sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    if texture:
        r = UsdShade.Shader.Define(stage, path + "/UV"); r.CreateIdAttr("UsdPrimvarReader_float2"); r.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        t = UsdShade.Shader.Define(stage, path + "/T"); t.CreateIdAttr("UsdUVTexture"); t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB"); t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(r.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(v) ** 2.2 for v in rgb.split(",")]))
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface"); return m

# brick: 31.8 x 31.8 x 19.2 mm body (+ studs) built from 6 quads so one face can carry a texture.
BX, BY, BZ = 0.0318, 0.0318, 0.0192
def quad(path, corners, mat):
    m = UsdGeom.Mesh.Define(stage, path); m.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    m.CreateFaceVertexCountsAttr([4]); m.CreateFaceVertexIndicesAttr([0, 1, 2, 3]); m.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)

def build_brick(parent, mark):
    path = parent + "/SimBrick"
    if stage.GetPrimAtPath(path): stage.RemovePrim(path)
    x = UsdGeom.Xform.Define(stage, path)
    plain = material("plain", rgb=args.color)
    marked = material(f"m_{mark}", texture=tex(mark, args.color, str(OUT / f"tex_{mark}.png")))
    hx, hy, hz = BX / 2, BY / 2, BZ / 2
    faces = {  # name: corners (counter-clockwise seen from outside, uv 0..1)
        "px": [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)],
        "nx": [(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)],
        "py": [(hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)],
        "ny": [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)],
        "pz": [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)],
        "nz": [(-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz)],
    }
    for n, c in faces.items():
        quad(f"{path}/{n}", c, marked if n == "px" else plain)
    for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        s = UsdGeom.Cylinder.Define(stage, f"{path}/stud{i}"); s.CreateRadiusAttr(0.0048); s.CreateHeightAttr(0.0045); s.CreateAxisAttr("Z")
        s.AddTranslateOp().Set(Gf.Vec3d(sx * 0.008, sy * 0.008, hz + 0.0022)); UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(plain)
    return x

world = World(stage_units_in_meters=1.0)
world.reset()
arm = SingleArticulation("/World/SO101"); arm.initialize()
names = arm.dof_names; idx = [names.index(j) for j in JOINTS]
lim = arm.get_articulation_controller().get_joint_limits() if hasattr(arm.get_articulation_controller(), "get_joint_limits") else None
print("dofs", names, flush=True)

def set_pose(deg):
    q = np.deg2rad(np.asarray(deg[:5], np.float32)) + OFF[:5]
    full = arm.get_joint_positions().copy()
    for k, j in enumerate(idx[:5]): full[j] = q[k]
    full[idx[5]] = 0.45  # open to brick width
    arm.set_joint_positions(full); arm.set_joint_velocities(np.zeros_like(full))
    for _ in range(3): world.step(render=True)

cam = stage.GetPrimAtPath("/World/InspectCamera")
rp = rep.create.render_product(str(cam.GetPath()), (640, 480))
ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
ov = rep.create.render_product("/World/OverviewCamera", (640, 480)); ann_ov = rep.AnnotatorRegistry.get_annotator("rgb"); ann_ov.attach([ov])

def grab(a):
    for _ in range(30):
        rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
        d = np.asarray(a.get_data())
        if d.ndim == 3: return d[:, :, :3].copy()
    raise RuntimeError("annotator returned no image")

def world_pos(prim):
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())

C = np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))  # row-vector convention
def project(p):
    v = np.append(p, 1) @ np.linalg.inv(C)  # world -> camera (camera looks down -Z)
    ca = UsdGeom.Camera(cam); f = ca.GetFocalLengthAttr().Get(); ha = ca.GetHorizontalApertureAttr().Get()
    fx = f / ha * 640
    return np.array([320 + fx * v[0] / -v[2], 240 - fx * v[1] / -v[2]]), -v[2]

# brick sits between the jaws: offset in gripper_link frame, estimated from the jaw position
build_brick(str(grip.GetPath()), "plain")
brick = stage.GetPrimAtPath(str(grip.GetPath()) + "/SimBrick")
pose = np.array([-30.8, 13.2, 3.3, 57.1, 14.3])  # median state over dataset frames with inspect.prob >= 0.999
set_pose(pose)
g = UsdGeom.Xformable(grip).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
jw = world_pos(jaw)
local_jaw = np.array(g.GetInverse().Transform(Gf.Vec3d(*jw)))
print("jaw in gripper frame", local_jaw, flush=True)
GOFF = Gf.Vec3d(-0.008, 0.0, -0.118)  # between the finger tips (measured jaw bounds in gripper_link frame)
UsdGeom.Xformable(brick).AddTranslateOp().Set(GOFF)
for _ in range(3): world.step(render=True)

def score(p):
    set_pose(p); uv, z = project(world_pos(brick))
    return np.hypot(*(uv - [320, 240])) + 400 * abs(z - 0.20), uv, z

s0, uv, z = score(pose); print("initial", pose, uv, z, s0, flush=True)
Image.fromarray(grab(ann)).save(OUT / "pose_initial_inspect.png"); Image.fromarray(grab(ann_ov)).save(OUT / "pose_initial_overview.png")
if args.search:
    best = (s0, pose.copy())
    rng = np.random.default_rng(0); step = np.array([12, 12, 12, 12, 0.0])
    for it in range(220):
        cand = best[1] + rng.normal(0, 1, 5) * step
        s, uv, z = score(cand)
        if s < best[0]: best = (s, cand)
        if it % 40 == 39: step *= 0.6
    pose = best[1]
s, uv, z = score(pose); print("inspect pose", np.round(pose, 1).tolist(), "uv", uv, "z", z, flush=True)
json.dump({"pose_deg": pose.tolist(), "brick_uv": uv.tolist(), "brick_depth_m": float(z), "gripper_brick_offset": list(GOFF)}, open(OUT / "inspect_pose.json", "w"), indent=1)
Image.fromarray(grab(ann_ov)).save(OUT / "pose_found_overview.png")

angles = np.linspace(-150, 150, args.angles)
for mark in args.marks.split(","):
    b = build_brick(str(grip.GetPath()), mark); b.AddTranslateOp().Set(GOFF)
    d = OUT / mark; d.mkdir(exist_ok=True)
    for i, a in enumerate(angles):
        p = pose.copy(); p[4] = a; set_pose(p)
        Image.fromarray(grab(ann)).save(d / f"{i:02d}_roll{a:+.0f}.png")
    print("MARK_DONE", mark, flush=True)
app.close()
