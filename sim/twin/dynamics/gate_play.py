"""Run the grasp gate (sim/twin/grasp_gate/gate.py, unchanged) with the play model installed on its GraspEnv.

    sim/twin/run.sh sim/twin/dynamics/gate_play.py --play compensate --tag p_comp --rung 3 --n 60 --place fk
    --play none | compensate | physical | sag | band | encoder      (what play.py layer(s) to install; see below)
    --delay                                 one extra command tick on pan / wrist_flex / wrist_roll
    --link_sag a,b,c                        camera-only compliance (deg / N m) on lift / elbow / wrist_flex
Every other argument goes to gate.py. Outputs go to sim/twin/out/dynamics/gate/<tag>/ (gate's result.json +
trace.npz), not the gate's own folder.
  none        the twin as is (servo model with constant bias, PD 60 N m/rad)
  compensate  play.py default: hold band + stiction + load sag (minus the PD's own sag) + quantised encoder
  physical    as compensate, but the arm PD runs at the real servo stiffness (params isaac_drive_physical) and the
              play model adds no sag (gravity and contacts deflect the arm in PhysX)
  sag         only the load sag (no band / stiction), constant bias removed
  band        only the band / stiction (no load sag): c0 + band around the goal, PD sag as is
"""
import argparse
import json
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim/residual_env"))

ap = argparse.ArgumentParser()
ap.add_argument("--play", default="compensate", choices=["none", "compensate", "physical", "sag", "band", "encoder"])
ap.add_argument("--delay", action="store_true")
ap.add_argument("--link_sag", default="")
ap.add_argument("--tag", required=True)
args, rest = ap.parse_known_args(sys.argv[1:])

P = json.loads((REPO / "sim/twin/dynamics/params.json").read_text())

if args.play == "physical":  # arm drive at the real servo stiffness (gripper drive untouched)
    import scene as rscene

    _build = rscene.build_scene_cfg

    def build_scene_cfg(*a, **k):
        cfg = _build(*a, **k)
        arm = cfg.robot.actuators["arm"]
        arm.stiffness = P["isaac_drive_physical"]["stiffness_nm_per_rad"]
        arm.damping = P["isaac_drive_physical"]["damping_nms_per_rad"]
        print("gate_play: arm drive", arm.stiffness, arm.damping, flush=True)
        return cfg

    rscene.build_scene_cfg = build_scene_cfg

from sim.twin.grasp_gate import common as C  # noqa: E402
from sim.twin.rl import grasp_env as G  # noqa: E402

C.OUT = REPO / "sim/twin/out/dynamics/gate"
_make = G.make


def make(cfg):
    env = _make(cfg)
    if args.play != "none":
        from sim.twin.dynamics.play import PlayCfg, install

        ls = tuple(float(v) for v in args.link_sag.split(",")) if args.link_sag else None
        pc = PlayCfg(band=args.play in ("compensate", "physical", "band"),
                     sag=args.play in ("compensate", "physical", "sag"),
                     encoder=True, controller_delay=args.delay, link_sag=ls,
                     drive="physical" if args.play == "physical" else "compensate")
        if args.play == "encoder":
            pc.band, pc.sag = False, False
        play = install(env, pc)
        if args.play == "band":  # keep the PD's own sag (no compensation term)
            play.c_drive[:] = 0.0
        print("gate_play: installed", args.play, "delay", args.delay, "link_sag", ls, flush=True)
    return env


G.make = make
sys.argv = ["gate.py", "--tag", args.tag] + rest
runpy.run_path(str(REPO / "sim/twin/grasp_gate/gate.py"), run_name="__main__")
