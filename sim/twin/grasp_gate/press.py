import json, sys, numpy as np, torch
sys.path.insert(0, "/repo")
from sim.twin.grasp_gate import common as C
from sim.twin.calib.profile import twin_profile
from sim.twin.rl.kin import Kin
FIN = (-0.00821, 0.00097, -0.10522)
kin = Kin(twin_profile().joint_offsets, "cpu", tip_local=FIN)
demos = C.load_demos()
out = []
for ep in demos["keep"]:
    d = demos["eps"][ep]; cf = d["cf"]
    S, A = torch.tensor(d["state"]), torch.tensor(d["action"])
    zs = kin.fk(S)[0][:, 2].numpy() + 0.002
    za = kin.fk(A)[0][:, 2].numpy() + 0.002
    lag = 2
    gap = (za[:-lag] - zs[lag:]) * 1e3     # commanded (2 ticks earlier) minus measured fin-tip height
    w = slice(cf - 5, cf + 10)
    free = slice(max(cf - 45, 0), cf - 25)
    # joint-level: action - state for lift, elbow, wflex at close vs free
    je = (d["action"][:-lag] - d["state"][lag:])
    out.append([ep, zs[w].min(), gap[w].min(), np.median(gap[free]), *np.median(je[w], 0)[:4], *np.median(je[free], 0)[:4], np.hypot(*kin.fk(S[cf])[0][:2].numpy())])
a = np.array(out)
np.save("/repo/sim/twin/out/grasp_gate/press.npy", a)
print("commanded - measured fin-tip z (mm): at close p10/50/90", np.percentile(a[:,2],[10,50,90]).round(1), "| free approach", np.percentile(a[:,3],[10,50,90]).round(1))
print("action-state deg at close (pan,lift,elbow,wflex) median", np.median(a[:,4:8],0).round(2), "free", np.median(a[:,8:12],0).round(2))
print("corr(measured zmin, cmd-meas gap at close)", np.corrcoef(a[:,1], a[:,2])[0,1].round(2))
for lo, hi in ((-20,-5),(-5,0),(0,5),(5,10),(10,15),(15,40)):
    m = (a[:,1]*1e3>=lo)&(a[:,1]*1e3<hi)
    print(f"FK fin tip {lo:4d}..{hi:3d} mm: n {m.sum():3d}  cmd-meas z gap at close median {np.median(a[m,2]):6.1f} mm, lift err {np.median(a[m,5]):5.2f} elbow err {np.median(a[m,6]):5.2f} reach {np.median(a[m,12]):.3f}")
