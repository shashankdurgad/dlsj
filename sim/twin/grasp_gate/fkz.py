import json, sys, numpy as np, torch
sys.path.insert(0, "/repo")
from sim.twin.grasp_gate import common as C
from sim.twin.calib.profile import twin_profile
from sim.twin.rl.kin import Kin
FIN = (-0.00821, 0.00097, -0.10522)
kin = Kin(twin_profile().joint_offsets, "cpu", tip_local=FIN)
demos = C.load_demos()
rows = []
for ep in demos["keep"]:
    d = demos["eps"][ep]; cf = d["cf"]
    s = torch.tensor(d["state"][max(cf-10,0):cf+25])
    p, R = kin.fk(s)
    z = p[:, 2].numpy() + 0.002
    i = int(np.argmin(z))
    held = float(np.median(d["state"][cf+15:cf+40, 5]))
    rows.append([ep, p[10,0].item(), p[10,1].item(), z[10], z.min(), i-10, *d["state"][cf][:5], held, R[10,2,2].item()])
a = np.array(rows)
np.save("/repo/sim/twin/out/grasp_gate/fkz.npy", a)
x, y, zc, zm = a[:,1], a[:,2], a[:,3]*1e3, a[:,4]*1e3
print("fin tip z above table at close: p10/50/90", np.percentile(zc,[10,50,90]).round(1), " min in window:", np.percentile(zm,[10,50,90]).round(1), "argmin offset median", np.median(a[:,5]))
r = np.hypot(x, y); pan = np.degrees(np.arctan2(y, x))
for name, X in (("plane x,y", np.c_[np.ones_like(x), x, y]), ("r", np.c_[np.ones_like(x), r]), ("x,y,r", np.c_[np.ones_like(x), x, y, r]),
                ("joints", np.c_[np.ones_like(x), a[:,6:11]]), ("x,y,x2,xy,y2", np.c_[np.ones_like(x), x, y, x*x, x*y, y*y])):
    for zz, nm in ((zc, "close"), (zm, "min")):
        c, *_ = np.linalg.lstsq(X, zz, rcond=None); res = zz - X @ c
        print(f"{name:14s} {nm:5s} coef {np.round(c,2).tolist()} resid std {res.std():.1f} mm (raw std {zz.std():.1f})")
print("corr z_min with r", np.corrcoef(zm, r)[0,1].round(2), "pan", np.corrcoef(zm, pan)[0,1].round(2), "held", np.corrcoef(zm, a[:,11])[0,1].round(2), "grip axis z (R22)", np.corrcoef(zm, a[:,12])[0,1].round(2))
print("r range", r.min().round(3), r.max().round(3), "bins of r -> median zmin:", [(round(lo,2), round(float(np.median(zm[(r>=lo)&(r<lo+0.04)])),1), int(((r>=lo)&(r<lo+0.04)).sum())) for lo in np.arange(0.12,0.40,0.04)])
