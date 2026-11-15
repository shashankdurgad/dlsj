"""The grasp gate (sim/twin/grasp_gate/gate.py, unchanged on disk) with the contact agent's protocol options.

    sim/twin/run.sh sim/twin/contact/gate_run.py [--c_studs] [--c_curved] [--c_forensics] <gate.py args ...>

gate.py's source is executed with these patches (each asserted to apply; without options it runs as is). Since
2026-11-15 gate.py does --c_studs / --c_curved natively (rest_lift_m, curved labels), so only --c_forensics patches:
  --c_studs      resting heights for studded colliders (contact.brick_collider): the 2x4 upside down rests on its
                 studs (centre 14.0 mm instead of 9.6 mm; gate.py already uses 14.0 for the 2x2).
  --c_curved     curved 2x3 labels spawn the curved brick (GraspEnv's BrickCurved: the LDraw body hull) instead of the
                 2x2 box; adds its resting poses (up, down, side_x = on a flat side, studs along the jaw axis).
  --c_forensics  PhysX contact reports on the target bricks (needs --set contact.report=true): every brick contact
                 per physics step -> <out>/contacts.npz (tick, env, other shape, impulse, point, normal, separation).
The tag is written under sim/twin/out/contact/ (gate.py's out dir + ../contact/).
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GATE = REPO / "sim/twin/grasp_gate/gate.py"

own = {"--c_studs": False, "--c_curved": False, "--c_forensics": False}
rest = []
for a in sys.argv[1:]:
    if a in own:
        own[a] = True
    else:
        rest.append(a)
if "--tag" in rest:
    i = rest.index("--tag") + 1
    rest[i] = "../contact/" + rest[i]
src = GATE.read_text()


def patch(old: str, new: str) -> None:
    global src
    assert src.count(old) == 1, f"gate.py changed: cannot patch {old[:60]!r}"
    src = src.replace(old, new)


native = "rest_lift_m" in src  # gate.py now does --c_studs / --c_curved itself (heights from contact.brick_collider)
if own["--c_studs"] and not native:
    patch('(1, "down"): ((1, 0, 0, 0), 0.0096)', '(1, "down"): ((1, 0, 0, 0), 0.0140)')
if own["--c_curved"] and not native:
    patch('kind[k] = 1 if lab["kind"] == "2x4" else 0',
          'kind[k] = 1 if lab["kind"] == "2x4" else (2 if lab["kind"] == "curved_2x3" and env.objs[2] is not None '
          'else 0)')
    patch('(1, "side_y"): ((-_S, 0, 0, _S), 0.0159)}',
          '(1, "side_y"): ((-_S, 0, 0, _S), 0.0159),\n'
          '        (2, "up"): ((0, 0, 0, 1), 0.0096), (2, "down"): ((1, 0, 0, 0), 0.0140),\n'
          '        (2, "side_x"): ((0.5, 0.5, 0.5, 0.5), 0.01585), (2, "side_y"): ((0.5, 0.5, 0.5, 0.5), 0.01585)}')
if own["--c_forensics"]:
    patch("env = G.make(cfg)\n", "env = G.make(cfg)\n_FORENSICS = __import__('sim.twin.contact.forensics', "
                                 "fromlist=['x']).Recorder(env)\n")
    patch("np.savez_compressed(out / \"trace.npz\", **tr)\n",
          "np.savez_compressed(out / \"trace.npz\", **tr)\n_FORENSICS.save(out / \"contacts.npz\", tr)\n")
sys.argv = [str(GATE), *rest]
print("gate_run options:", {k: v for k, v in own.items() if v}, flush=True)
code = compile(src, str(GATE), "exec")
exec(code, {"__name__": "__main__", "__file__": str(GATE), "re": re})  # noqa: S102
