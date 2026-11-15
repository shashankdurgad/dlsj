import json, sys, numpy as np, torch
sys.path.insert(0, "/repo")
from sim.twin.grasp_gate import rung0 as R0, common as C
from sim.twin.calib.profile import twin_profile
from sim.twin.rl.kin import Kin
ref = json.load(open("/repo/sim/residual_env/assets/joe_cal2/reference-cal2.json"))
print({k: (v if not isinstance(v, (list, dict)) or len(str(v)) < 300 else "...") for k, v in ref.items() if k not in ("camera","objects","observed_pixels","predicted_pixels")})
kin = Kin(twin_profile().joint_offsets, "cpu", tip_local=(0, 0, 0))
cam = R0.cameras()["joe"]
demos = C.load_demos()
lm = torch.tensor(ref["housing_landmark_local"])
init=np.array(ref["initial_joint_state"])
best=sorted((float(np.abs(d["state"][0]-init).sum()),e) for e,d in demos["eps"].items())[:3]
print("closest start", best)
for ep in [best[0][1]]:
    s = torch.tensor(demos["eps"][ep]["state"])
    out = []
    for i, f in enumerate(ref["frames"]):
        p, R = kin.fk(s[f])
        w = (p + R @ lm).numpy().astype(float)
        out.append(R0.project(cam, w)[0])
    out = np.array(out)
    print(ep, "mine", out.round(0).tolist()); print("joe pred", np.array(ref["predicted_pixels"]).round(0).tolist())
    print("rms vs joe pred", np.sqrt(((out - np.array(ref["predicted_pixels"]))**2).sum(1).mean()).round(1),
          "rms vs obs", np.sqrt(((out - np.array(ref["observed_pixels"]))**2).sum(1).mean()).round(1))
