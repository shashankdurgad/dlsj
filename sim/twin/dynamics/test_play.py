"""Smoke test of play.py without Isaac: a mock ResidualEnv with a perfect joint (sim joint = PD target).

    sim/twin/dynamics/py.sh sim/twin/dynamics/test_play.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.dynamics import data as D  # noqa: E402
from sim.twin.dynamics.play import Play, PlayCfg, install  # noqa: E402


class MockEnv:
    def __init__(self, n):
        self.n, self.device = n, "cpu"
        self.off = torch.tensor(D.w_offsets(), dtype=torch.float32)
        self.K = torch.tensor([0.30, 0.25, 0.25, 0.29, 0.29, 0.20])
        self.vmax = torch.full((6,), 6.0)
        self.bias = torch.tensor([-0.09, 0.66, 1.93, 0.22, -0.09, 1.23])
        self.cfg = SimpleNamespace(max_relative_target=10.0)
        self.stats = {"clamp_hits": torch.zeros(n)}
        self.jidx = list(range(6))
        self.r = torch.zeros(n, 6)
        self.q = torch.zeros(n, 6)
        self.robot = SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.zeros(n, 6)),
                                     set_joint_position_target=self._set)

    def _set(self, q):
        self.q = q.clone()  # perfect drive: joint = target (rad)

    def to_sim(self, ours):
        return ours.clone()

    def measured(self, noisy=True):
        return self.q.clone()

    def apply_goal(self, goal):
        raise RuntimeError("replaced by install()")


def main():
    n = 4
    env = MockEnv(n)
    start = torch.tensor([0.0, -40.0, 60.0, 70.0, 0.0, 30.0]).expand(n, 6).clone()
    env.r[:], env.q[:] = start, start
    play = install(env, PlayCfg(dr=True))
    assert torch.all(env.bias == 0)
    # ramp lift up 10 deg, hold, ramp back down: hold error depends on the approach direction
    traj = [start + torch.tensor([0, a, 0, 0, 0, 0.0]) for a in np.r_[np.linspace(0, 10, 20), [10] * 40,
                                                                          np.linspace(10, 0, 20), [0] * 40]]
    errs = []
    for g in traj:
        env.apply_goal(g)
        errs.append((env.q - g)[:, 1].mean().item())
    print("lift hold error after rising / falling:", round(errs[59], 3), round(errs[-1], 3))
    m = env.measured()
    k = play.count
    frac = ((m - play.phase) / k)
    assert torch.allclose(frac, frac.round(), atol=1e-3), "readings not on the encoder grid"
    print("tau at start (N m):", play.tau(start)[0].numpy().round(3))
    # delay flag: one-tick lag on pan / wrist_flex / wrist_roll
    p2 = Play(PlayCfg(controller_delay=True, band=False, sag=False), 1, D.w_offsets())
    p2.reset(torch.arange(1), torch.zeros(1, 6))
    a = p2.target(torch.ones(1, 6), torch.zeros(1, 6))
    print("delay: first-tick targets", a.numpy().round(3))
    assert a[0, 0] < 0.5 and a[0, 1] > 0.5
    # reset detection: env.r changed outside apply_goal -> play state restarts there
    env.r[0] = start[0] + 5
    env.apply_goal(start + 5)
    print("DR draws c (lift):", play.P["c"][:, 1].numpy().round(3), "d0 (pan):", play.P["d0"][:, 0].numpy().round(3))
    print("OK")


if __name__ == "__main__":
    main()
