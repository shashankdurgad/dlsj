"""Replay n-step targets, MC returns and the zero-initialised residual (CPU).

    sim/twin/run.sh -m pytest sim/twin/rl/test_dice.py     (or python -m pytest inside the image)
"""
import torch

from sim.twin.rl.dice import Dice, DiceCfg, Replay, finalize_returns


def _buf():
    b = Replay(64, 3, 2, 4, 6, "cpu")
    # one env, episode 0 = 4 rows (last terminal), episode 1 = 2 rows (truncated end, not terminal)
    rew = torch.tensor([0.1, 0.2, 0.3, 1.0, 0.5, 0.5])
    term = torch.tensor([False, False, False, True, False, False])
    end = torch.tensor([False, False, False, True, False, True])
    epi = torch.tensor([0, 0, 0, 0, 1, 1])
    prev = torch.tensor([-1])
    for i in range(6):
        ix = b.add(s=torch.full((1, 3), float(i)), s2=torch.full((1, 3), float(i + 1)), A=torch.zeros(1, 2, 4),
                   A2=torch.zeros(1, 2, 4), a_sel=torch.zeros(1, 4), r=torch.zeros(1, 6), rew=rew[i:i + 1],
                   term=term[i:i + 1], end=end[i:i + 1], epi=epi[i:i + 1])
        b.link(prev, ix)
        prev = ix if not end[i] else torch.tensor([-1])
    return b


def test_nstep_and_mc():
    b = _buf()
    finalize_returns(b, torch.tensor([[0, 1, 2, 3], [4, 5, -1, -1]]), torch.tensor([4, 2]), 0.9)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        s = b.sample(32, 0.9, 3, g)
        i = s["s"][:, 0].long()
        for j in range(32):
            k = int(i[j])
            if k == 0:  # rows 0,1,2 -> bootstrap from s2 of row 2
                assert abs(float(s["R"][j]) - (0.1 + 0.9 * 0.2 + 0.81 * 0.3)) < 1e-6
                assert abs(float(s["disc"][j]) - 0.729) < 1e-6 and float(s["s2"][j, 0]) == 3.0
            if k == 2:  # rows 2,3 (terminal) -> no bootstrap
                assert abs(float(s["R"][j]) - (0.3 + 0.9 * 1.0)) < 1e-6 and float(s["disc"][j]) == 0.0
            if k == 4:  # rows 4,5 end of episode (truncation stored non-terminal) -> bootstrap gamma^2
                assert abs(float(s["R"][j]) - (0.5 + 0.45)) < 1e-6 and abs(float(s["disc"][j]) - 0.81) < 1e-6
    assert abs(float(b.G[0]) - (0.1 + 0.9 * 0.2 + 0.81 * 0.3 + 0.729 * 1.0)) < 1e-5
    assert abs(float(b.G[4]) - 0.95) < 1e-6


def test_zero_init_residual():
    m = Dice(10, 30, DiceCfg(hidden=(32, 32), ensemble=3), "cpu")
    s, a = torch.randn(5, 10), torch.randn(5, 30)
    assert float(m.pi(s, a).abs().max()) == 0.0
    k, r, q = m.select(s, a.view(5, 1, 30).expand(5, 4, 30))
    assert r.shape == (5, 6) and q.shape == (5, 4)
