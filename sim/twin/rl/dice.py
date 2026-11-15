"""DICE-RL residual learner (arXiv 2603.10263), ported from the user's reference implementation
(~/Documents/dice-rl-wam*/model/rl/distill_residual_rl*.py; pi0.5 LIBERO config
brev/dicerl/cfg/libero/*/ft_distill_residual_flow_unet_img_pi05.yaml) to the low-dim grasp residual.

* frozen prior proposes K candidate chunks per decision; the residual is conditioned on the base action
  (condition_residual_on_base_action: true): r = s_theta(state, a_base^k), zero-initialised output (tanh-bounded),
  so training starts exactly at the prior;
* "state" = the env's low-dim observation (+ the prior's pooled latent for a visual prior) + the first K_state
  candidate chunks (as in the pi0.5 features: latent + all K chunks);
* critic: ensemble of 10 MLPs Q(state, a_base, r), min over heads; summed per-head MSE to an n-step (3) target,
  gamma 0.99, bootstrapped with the Polyak target (tau 0.01) averaged over the stored next-state candidates corrected
  by the current actor; truncation bootstraps (not terminal), termination does not;
* actor: -Q normalised by mean |Q| over online rows (Q loss off on expert rows) + bc_weight * ||r||^2 (pull to the
  prior), soft Q-filter of the BC term after `qf_warmup` steps: BC is dropped where Q(s, a_pi) > Q(s, a_base) AND
  Q(s, a_stored) - MC return < qf_threshold (-0.55);
* best-of-K: execute argmax_k min_j Q(s, a_k + s_theta(s, a_k)) (max_q_min), uniform during the warm-up;
* RLPD expert mixing: expert rows (a_base = demo chunk, r = 0) at an adaptive ratio 0.5 -> 0.1 over 10k steps;
* Adam 1e-4, weight decay 1e-5, cosine restarts (1000 / 10 warm-up / 1e-6), grad-norm clip 1, actor + target
  update every 2nd critic step.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiceCfg:
    hidden: tuple = (256, 256, 256)  # reference: [1024]x3 for 3072-d tokens; low-dim here
    ensemble: int = 10
    gamma: float = 0.99
    n_step: int = 3
    tau: float = 0.01
    lr: float = 1e-4
    weight_decay: float = 1e-5
    cycle_steps: int = 1000
    warmup_lr_steps: int = 10
    min_lr: float = 1e-6
    max_grad_norm: float = 1.0
    batch: int = 256
    grad_steps: int = 20  # per update
    actor_every: int = 2
    bc_weight: float = 50.0
    soft_q_filter: bool = True
    qf_threshold: float = -0.55
    qf_warmup: int = 3000  # gradient steps
    expert_ratio: tuple = (0.5, 0.1, 10000)  # start, end, over gradient steps
    disable_q_expert: bool = True
    k_state: int = 4  # candidates included in the state
    act_dim: int = 6


class Ens(nn.Module):
    """E parallel MLPs with GELU + LayerNorm (per member), batched by einsum."""

    def __init__(self, E, din, hidden, dout, zero_last=False):
        super().__init__()
        dims = [din, *hidden, dout]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        self.g = nn.ParameterList()
        self.beta = nn.ParameterList()
        for i in range(len(dims) - 1):
            w = torch.empty(E, dims[i], dims[i + 1])
            for e in range(E):
                nn.init.kaiming_uniform_(w[e].T, a=math.sqrt(5))
            bound = 1 / math.sqrt(dims[i])
            self.W.append(nn.Parameter(w))
            self.b.append(nn.Parameter(torch.empty(E, 1, dims[i + 1]).uniform_(-bound, bound)))
            if i < len(dims) - 2:
                self.g.append(nn.Parameter(torch.ones(E, 1, dims[i + 1])))
                self.beta.append(nn.Parameter(torch.zeros(E, 1, dims[i + 1])))
        if zero_last:
            nn.init.zeros_(self.W[-1])
            nn.init.zeros_(self.b[-1])

    def forward(self, x):  # x (E, B, din) or (B, din)
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.W[0].shape[0], -1, -1)
        for i in range(len(self.W)):
            x = torch.baddbmm(self.b[i], x, self.W[i])
            if i < len(self.W) - 1:
                x = F.layer_norm(x, x.shape[-1:]) * self.g[i] + self.beta[i]
                x = F.gelu(x)
        return x


class Dice(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, cfg: DiceCfg, device="cuda:0"):
        super().__init__()
        self.cfg, self.device = cfg, device
        self.s_dim, self.a_dim = s_dim, a_dim  # a_dim = candidate feature size
        self.actor = Ens(1, s_dim + a_dim, cfg.hidden, cfg.act_dim, zero_last=True)
        self.critic = Ens(cfg.ensemble, s_dim + a_dim + cfg.act_dim, cfg.hidden, 1)
        self.critic_t = Ens(cfg.ensemble, s_dim + a_dim + cfg.act_dim, cfg.hidden, 1)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.critic_t.requires_grad_(False)
        self.to(device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.steps = 0

    # ---------------------------------------------------------------- nets
    def pi(self, s, a):
        return torch.tanh(self.actor(torch.cat([s, a], -1))[0])

    def q_all(self, s, a, r, target=False):
        net = self.critic_t if target else self.critic
        return net(torch.cat([s, a, r], -1))[..., 0]  # (E, B)

    def q_min(self, s, a, r, target=False):
        return self.q_all(s, a, r, target).min(0).values

    @torch.no_grad()
    def select(self, s, A, mode="max_q_min", residual=True):
        """s (n, s_dim), A (n, K, a_dim) -> chosen index (n,), residual for it (n, act_dim)."""
        n, K, _ = A.shape
        sK = s.unsqueeze(1).expand(n, K, s.shape[-1]).reshape(n * K, -1)
        aK = A.reshape(n * K, -1)
        r = self.pi(sK, aK) if residual else torch.zeros(n * K, self.cfg.act_dim, device=s.device)
        q = self.q_min(sK, aK, r).view(n, K)
        k = q.argmax(1) if mode == "max_q_min" else torch.zeros(n, dtype=torch.long, device=s.device)
        return k, r.view(n, K, -1)[torch.arange(n, device=s.device), k], q

    # ---------------------------------------------------------------- schedule
    def _lr(self):
        c = self.cfg
        t = self.steps % c.cycle_steps
        if t < c.warmup_lr_steps:
            return c.lr * (t + 1) / c.warmup_lr_steps
        p = (t - c.warmup_lr_steps) / max(1, c.cycle_steps - c.warmup_lr_steps)
        return c.min_lr + 0.5 * (c.lr - c.min_lr) * (1 + math.cos(math.pi * p))

    def expert_ratio(self):
        a, b, n = self.cfg.expert_ratio
        return a + (b - a) * min(1.0, self.steps / max(1, n))

    def _polyak(self):
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters(), strict=False):
                pt.lerp_(p, self.cfg.tau)

    # ---------------------------------------------------------------- update
    def update(self, B: dict) -> dict:
        """B: s (B,S), A (B,K,a) stored candidates, a_sel (B,a) executed candidate, r (B,act) executed residual,
        R (B,) n-step reward, disc (B,) gamma^m * (1 - terminal), s2 (B,S), A2 (B,K,a), G (B,) MC return (nan
        unknown), expert (B,) bool."""
        c = self.cfg
        for o in (self.opt_a, self.opt_c):
            for gp in o.param_groups:
                gp["lr"] = self._lr()
        s, A, a_sel, r, R, disc, s2, A2 = (B[k] for k in ("s", "A", "a_sel", "r", "R", "disc", "s2", "A2"))
        nb, K, ad = A2.shape
        with torch.no_grad():
            s2K = s2.unsqueeze(1).expand(nb, K, s2.shape[-1]).reshape(nb * K, -1)
            a2K = A2.reshape(nb * K, -1)
            q2 = self.q_min(s2K, a2K, self.pi(s2K, a2K), target=True).view(nb, K).mean(1)
            y = R + disc * q2
        qa = self.q_all(s, a_sel, r)
        lc = (qa - y.unsqueeze(0)).pow(2).mean(1).sum()
        self.opt_c.zero_grad(set_to_none=True)
        lc.backward()
        gn_c = nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
        self.opt_c.step()
        info = {"critic_loss": float(lc.detach()), "q_data": float(qa.detach().mean()), "target": float(y.mean()),
                "grad_c": float(gn_c), "lr": self._lr()}
        self.steps += 1
        if self.steps % c.actor_every == 0:
            info.update(self._actor(B))
            self._polyak()
        return info

    def _actor(self, B):
        c = self.cfg
        s, A, a_sel, r_st, G, expert = B["s"], B["A"], B["a_sel"], B["r"], B["G"], B["expert"]
        nb, K, ad = A.shape
        sK = s.unsqueeze(1).expand(nb, K, s.shape[-1]).reshape(nb * K, -1)
        aK = A.reshape(nb * K, -1)
        rK = self.pi(sK, aK)
        self.critic.requires_grad_(False)
        q = self.q_min(sK, aK, rK).view(nb, K)
        self.critic.requires_grad_(True)
        online = (~expert).float()
        with torch.no_grad():
            q_base = self.q_min(sK, aK, torch.zeros_like(rK)).view(nb, K)
            if c.disable_q_expert:
                cnt = online.sum() * K
                qbar = (q.abs() * online[:, None]).sum() / cnt.clamp(min=1) if cnt > 0 else q.abs().mean()
            else:
                qbar = q.abs().mean()
            qbar = qbar.clamp(min=1e-6)
            kappa = torch.ones(nb, K, device=s.device)
            frac_filtered = 0.0
            if c.soft_q_filter and self.steps > c.qf_warmup:
                q_st = self.q_min(s, a_sel, r_st)
                under = torch.nan_to_num(q_st - G, nan=0.0) < c.qf_threshold  # nan (unfinished episode): keep BC
                kappa = 1.0 - ((q > q_base) & under[:, None]).float()
                frac_filtered = float(1 - kappa.mean())
        qw = online if c.disable_q_expert else torch.ones_like(online)
        lq = -(q * qw[:, None]).sum() / (nb * K) / qbar
        bc = (kappa * rK.view(nb, K, -1).pow(2).mean(-1)).mean()
        la = lq + c.bc_weight * bc
        self.opt_a.zero_grad(set_to_none=True)
        la.backward()
        gn = nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
        self.opt_a.step()
        return {"actor_loss": float(la.detach()), "actor_q": float(lq.detach()), "bc": float(bc.detach()),
                "res_rms": float(rK.detach().pow(2).mean().sqrt()),
                "q_gain": float((q.detach() - q_base).mean()), "bc_filtered": frac_filtered, "grad_a": float(gn)}

    def save(self, path, extra=None):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "critic_t": self.critic_t.state_dict(), "opt_a": self.opt_a.state_dict(),
                    "opt_c": self.opt_c.state_dict(), "cfg": asdict(self.cfg), "s_dim": self.s_dim,
                    "a_dim": self.a_dim, "steps": self.steps, **(extra or {})}, path)

    @classmethod
    def load(cls, path, device="cuda:0"):
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = DiceCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
        m = cls(ck["s_dim"], ck["a_dim"], cfg, device)
        m.actor.load_state_dict(ck["actor"])
        m.critic.load_state_dict(ck["critic"])
        m.critic_t.load_state_dict(ck["critic_t"])
        for o, kk in ((m.opt_a, "opt_a"), (m.opt_c, "opt_c")):
            if kk in ck:
                o.load_state_dict(ck[kk])
        m.steps = ck["steps"]
        return m, ck


class Replay:
    """Flat GPU replay of decision rows with in-episode `nxt` links (n-step targets at sample time) and MC returns
    filled at episode end. Online and expert rows live in separate instances."""

    def __init__(self, cap, s_dim, K, a_dim, act_dim, device):
        self.cap, self.d, self.n = cap, device, 0
        self.ptr = 0

        def z(*s, dt=torch.float32):
            return torch.zeros(cap, *s, dtype=dt, device=device)

        self.s, self.s2 = z(s_dim), z(s_dim)
        self.A, self.A2 = z(K, a_dim), z(K, a_dim)
        self.a_sel, self.r = z(a_dim), z(act_dim)
        self.rew, self.term, self.end = z(), z(dt=torch.bool), z(dt=torch.bool)
        self.G = torch.full((cap,), float("nan"), device=device)
        self.nxt = torch.full((cap,), -1, dtype=torch.long, device=device)
        self.epi = torch.full((cap,), -1, dtype=torch.long, device=device)

    def add(self, **kw) -> torch.Tensor:
        m = len(kw["s"])
        idx = (self.ptr + torch.arange(m, device=self.d)) % self.cap
        for k, v in kw.items():
            if v.is_floating_point():  # a PhysX blow-up must not poison the critics
                v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            getattr(self, k)[idx] = v
        self.nxt[idx] = -1
        self.G[idx] = float("nan")
        self.ptr = (self.ptr + m) % self.cap
        self.n = min(self.cap, self.n + m)
        return idx

    def link(self, prev_idx, new_idx):
        ok = (prev_idx >= 0) & (self.epi[prev_idx.clamp(min=0)] == self.epi[new_idx])
        self.nxt[prev_idx[ok]] = new_idx[ok]

    def sample(self, b, gamma, n_step, gen=None):
        i = torch.randint(self.n, (b,), device=self.d, generator=gen)
        R = torch.zeros(b, device=self.d)
        disc = torch.ones(b, device=self.d)
        cur = i.clone()
        alive = torch.ones(b, dtype=torch.bool, device=self.d)
        last = i.clone()
        for _ in range(n_step):
            R += alive * disc * self.rew[cur]
            last = torch.where(alive, cur, last)
            disc = torch.where(alive, disc * gamma, disc)
            stop = self.end[cur] | (self.nxt[cur] < 0) | (self.epi[self.nxt[cur].clamp(min=0)] != self.epi[cur])
            alive = alive & ~stop
            cur = torch.where(alive, self.nxt[cur].clamp(min=0), cur)
        disc = disc * (~self.term[last]).float()
        return {"s": self.s[i], "A": self.A[i], "a_sel": self.a_sel[i], "r": self.r[i], "R": R, "disc": disc,
                "s2": self.s2[last], "A2": self.A2[last], "G": self.G[i]}


def mix(online: dict, expert: dict | None):
    if expert is None:
        online["expert"] = torch.zeros(len(online["R"]), dtype=torch.bool, device=online["R"].device)
        return online
    out = {k: torch.cat([online[k], expert[k]]) for k in online}
    out["expert"] = torch.cat([torch.zeros(len(online["R"]), dtype=torch.bool, device=online["R"].device),
                               torch.ones(len(expert["R"]), dtype=torch.bool, device=online["R"].device)])
    return out


def finalize_returns(buf: Replay, rows: torch.Tensor, lens: torch.Tensor, gamma: float):
    """rows (m, L) row indices of finished episodes (padded with -1), lens (m,): MC discounted return-to-go."""
    m, L = rows.shape
    G = torch.zeros(m, device=buf.d)
    for j in range(L - 1, -1, -1):
        valid = j < lens
        ix = rows[:, j].clamp(min=0)
        G = torch.where(valid, buf.rew[ix] + gamma * G, G)
        buf.G[ix[valid]] = G[valid]
