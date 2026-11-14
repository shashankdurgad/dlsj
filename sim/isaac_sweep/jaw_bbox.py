from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import numpy as np, omni.usd
from pxr import Usd, UsdGeom
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
omni.usd.get_context().open_stage("/home/joe-spark-2/pi05-isaac-so101/outputs/scene-final.usda")
for _ in range(20): app.update()
st = omni.usd.get_context().get_stage()
w = World(); w.reset(); a = SingleArticulation("/World/SO101"); a.initialize()
q = a.get_joint_positions(); q[a.dof_names.index("gripper")] = 0.05; a.set_joint_positions(q)
for _ in range(3): w.step(render=False)
grip = next(p for p in st.Traverse() if p.GetName() == "gripper_link")
G = UsdGeom.Xformable(grip).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
for p in Usd.PrimRange(grip):
    if p.IsA(UsdGeom.Mesh):
        r = cache.ComputeRelativeBound(p, grip).ComputeAlignedRange()
        lo = np.array(r.GetMin()); hi = np.array(r.GetMax())
        print("BBOX", p.GetName(), np.round(lo, 4).tolist(), np.round(hi, 4).tolist(), flush=True)
app.close()
