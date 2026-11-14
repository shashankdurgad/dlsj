"""Compact SAC (twin Q, tanh-Gaussian actor, auto entropy) for the batched residual env. All tensors on the GPU."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, o, h=256, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, h=256):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, h)
        # start near zero residual: the base policy alone is the initial behaviour
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias[:act_dim] = 0.0
            self.net[-1].bias[act_dim:] = -1.5

    def forward(self, obs, deterministic=False):
        mu, log_std = self.net(obs).chunk(2, -1)
        log_std = log_std.clamp(-5, 1)
        if deterministic:
            return torch.tanh(mu), None
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.9189385).sum(-1)
        logp -= (2 * (0.6931472 - u - F.softplus(-2 * u))).sum(-1)
        return a, logp


class SAC:
    def __init__(self, obs_dim, act_dim, device, lr=3e-4, gamma=0.98, tau=0.01, buffer=500_000, batch=512):
        self.d, self.gamma, self.tau, self.batch = device, gamma, tau, batch
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.q1, self.q2 = mlp(obs_dim + act_dim, 1).to(device), mlp(obs_dim + act_dim, 1).to(device)
        self.q1t, self.q2t = mlp(obs_dim + act_dim, 1).to(device), mlp(obs_dim + act_dim, 1).to(device)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        self.log_alpha = torch.tensor(-2.0, device=device, requires_grad=True)
        self.target_ent = -float(act_dim)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=lr)
        self.N, self.i, self.full = buffer, 0, False
        self.o = torch.zeros(buffer, obs_dim, device=device)
        self.a = torch.zeros(buffer, act_dim, device=device)
        self.r = torch.zeros(buffer, device=device)
        self.o2 = torch.zeros(buffer, obs_dim, device=device)
        self.dn = torch.zeros(buffer, device=device)

    def add(self, o, a, r, o2, done):
        n = o.shape[0]
        idx = (torch.arange(n, device=self.d) + self.i) % self.N
        self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.dn[idx] = o, a, r, o2, done.float()
        self.i = (self.i + n) % self.N
        self.full |= self.i < n

    def size(self):
        return self.N if self.full else self.i

    @torch.no_grad()
    def act(self, o, deterministic=False):
        return self.actor(o, deterministic)[0]

    def update(self):
        idx = torch.randint(0, self.size(), (self.batch,), device=self.d)
        o, a, r, o2, dn = self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.dn[idx]
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            x2 = torch.cat([o2, a2], -1)
            qt = torch.min(self.q1t(x2), self.q2t(x2)).squeeze(-1) - alpha * logp2
            y = r + self.gamma * (1 - dn) * qt
        x = torch.cat([o, a], -1)
        lq = F.mse_loss(self.q1(x).squeeze(-1), y) + F.mse_loss(self.q2(x).squeeze(-1), y)
        self.opt_q.zero_grad()
        lq.backward()
        self.opt_q.step()
        an, logp = self.actor(o)
        xn = torch.cat([o, an], -1)
        la = (alpha * logp - torch.min(self.q1(xn), self.q2(xn)).squeeze(-1)).mean()
        self.opt_a.zero_grad()
        la.backward()
        self.opt_a.step()
        lal = -(self.log_alpha * (logp.detach() + self.target_ent)).mean()
        self.opt_al.zero_grad()
        lal.backward()
        self.opt_al.step()
        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1t), (self.q2, self.q2t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)
        return {"q_loss": float(lq), "a_loss": float(la), "alpha": float(alpha)}

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                    "log_alpha": self.log_alpha.detach()}, path)

    def load(self, path):
        sd = torch.load(path, map_location=self.d)
        self.actor.load_state_dict(sd["actor"])
        self.q1.load_state_dict(sd["q1"])
        self.q2.load_state_dict(sd["q2"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"])
