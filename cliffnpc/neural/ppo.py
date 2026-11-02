"""PPO pretraining for the shared brain, laid out for the cache.

Rollout buffers are allocated once, time-major [T, N, ...].  Each NPC has a preference
vector omega over the goals; the brain sees it (observation features 22-25) and its
advantage is scalarised with it, so one network learns every personality.  Values and
advantages are per goal (GAE per goal), so the four value heads stay meaningful.

Minibatches are contiguous blocks of NPC slots (a random permutation of blocks), and
each block is replayed through the recurrent layer from its hidden state at the start
of the rollout (full-episode truncated BPTT), reading [T, block, ...] slices.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .brain import ActBuffers, Brain, NPCState, bern_logp, entropy, gauss_logp
from .layout import Layout
from .world import VecWorld


@dataclass
class PPOConfig:
    gamma: float = 0.99              # per tick (0.2 s): horizon ~20 s
    lam: float = 0.95
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 4
    block: int = 256                 # NPCs per minibatch (contiguous)
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad: float = 0.5


class Rollout:
    """Preallocated [T, N, ...] buffers."""

    def __init__(self, L: Layout):
        T, N, F, G = L.episode, L.n, L.features, L.goals
        z = torch.zeros
        self.obs = z(T, N, F)
        self.u = z(T, N, 2)
        self.fight = z(T, N, dtype=torch.bool)
        self.logp = z(T, N)
        self.value = z(T + 1, N, G)
        self.reward = z(T, N, G)
        self.h0 = z(N, L.hidden)
        self.adv = z(T, N, G)
        self.ret = z(T, N, G)


class RunningStd:
    """Per-goal running standard deviation of rewards (reward scaling)."""

    def __init__(self, g: int):
        self.n, self.mean, self.m2 = 0, torch.zeros(g), torch.zeros(g)

    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])
        n, mean, var = x.shape[0], x.mean(0), x.var(0, unbiased=False)
        tot = self.n + n
        d = mean - self.mean
        self.mean = self.mean + d * n / tot
        self.m2 = self.m2 + var * n + d * d * self.n * n / tot
        self.n = tot

    @property
    def std(self) -> torch.Tensor:
        return (self.m2 / max(self.n, 1)).sqrt().clamp_min(1e-6)


def sample_preferences(L: Layout, gen: torch.Generator) -> torch.Tensor:
    """Random personalities: each goal weight U(0, 1), normalised to sum 2."""
    p = torch.rand(L.worlds, L.agents, L.goals, generator=gen)
    return 2.0 * p / p.sum(-1, keepdim=True)


class Trainer:
    def __init__(self, L: Layout, cfg: PPOConfig = PPOConfig(), seed: int = 0, world_cfg=None,
                 individual: dict | None = None):
        """individual: e.g. dict(p_allergic=0.5, p_fragile=0.3) -- re-randomise each NPC's
        allergy and fragility every episode (pretraining for adaptability)."""
        self.individual = individual
        self.L, self.cfg = L, cfg
        torch.manual_seed(seed)
        self.gen = torch.Generator().manual_seed(seed + 1)
        self.brain = Brain(L)
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=cfg.lr, eps=1e-5)
        self.world = VecWorld(L, world_cfg, seed=seed + 2) if world_cfg is not None else VecWorld(L, seed=seed + 2)
        self.state = NPCState.zeros(L.n, L.hidden)
        self.acts = ActBuffers.zeros(L.n, L.goals)
        self.buf = Rollout(L)
        self.rstd = RunningStd(L.goals)
        self._zeros_block = torch.zeros(cfg.block, L.hidden)   # adapters off during pretraining

    # ---------------------------------------------------------------- rollout
    def collect(self, pref: torch.Tensor | None = None, greedy: bool = False) -> dict:
        L, b, st, a = self.L, self.buf, self.state, self.acts
        pref = sample_preferences(L, self.gen) if pref is None else pref
        if self.individual:
            self.world.randomize_individuals(**self.individual)
        obs = self.world.reset(pref).view(L.n, L.features)
        st.hidden.zero_()
        b.h0.copy_(st.hidden)
        for t in range(L.episode):
            b.obs[t].copy_(obs)
            self.brain.act(obs, st, a, self.gen, greedy=greedy)
            b.u[t].copy_(a.u)
            b.fight[t].copy_(a.fight)
            b.logp[t].copy_(a.logp)
            b.value[t].copy_(a.value)
            o, r = self.world.step(torch.tanh(a.u).view(L.worlds, L.agents, 2), a.fight.view(L.worlds, L.agents))
            b.reward[t].copy_(r.view(L.n, L.goals))
            obs = o.view(L.n, L.features)
        self.brain.act(obs, st, a, self.gen, greedy=True)             # bootstrap value at T
        b.value[L.episode].copy_(a.value)
        self.pref = pref.view(L.n, L.goals).clone()
        per_npc = b.reward.sum(0)                                     # [N, G] episode returns
        return dict(returns=per_npc.mean(0), score=(per_npc * self.pref).sum(-1).mean(),
                    per_npc=per_npc)

    # ---------------------------------------------------------------- update
    def advantages(self) -> None:
        """Per-goal GAE.  Rewards are scaled by the running spread of each goal's
        discounted return, so every value head regresses order-1 targets."""
        c, b, L = self.cfg, self.buf, self.L
        G = torch.zeros(L.n, L.goals)
        disc = torch.empty_like(b.reward)
        for t in reversed(range(L.episode)):
            G = b.reward[t] + c.gamma * G
            disc[t] = G
        self.rstd.update(disc)
        r = b.reward / self.rstd.std
        gae = torch.zeros(L.n, L.goals)
        for t in reversed(range(L.episode)):
            delta = r[t] + c.gamma * b.value[t + 1] - b.value[t]
            gae = delta + c.gamma * c.lam * gae
            b.adv[t] = gae
        b.ret.copy_(b.adv + b.value[:-1])

    def update(self) -> dict:
        c, b, L = self.cfg, self.buf, self.L
        self.advantages()
        scal = (b.adv * self.pref.unsqueeze(0)).sum(-1)               # [T, N] scalarised by preference
        scal = (scal - scal.mean()) / (scal.std() + 1e-8)
        n_blocks = L.n // c.block
        stats = dict(pi=0.0, v=0.0, ent=0.0, kl=0.0)
        for _ in range(c.epochs):
            for bi in torch.randperm(n_blocks, generator=self.gen).tolist():
                s, e = bi * c.block, (bi + 1) * c.block
                zs = self._zeros_block
                po, vals = self.brain.sequence(b.obs[:, s:e], b.h0[s:e], zs, zs)   # one fused pass
                mu, fl = po[..., :2], po[..., 2]
                logp = gauss_logp(b.u[:, s:e], mu, self.brain.log_std) + bern_logp(b.fight[:, s:e], fl)
                ents = entropy(self.brain.log_std, fl)
                ratio = (logp - b.logp[:, s:e]).exp()
                A = scal[:, s:e]
                pi_loss = -torch.min(ratio * A, ratio.clamp(1 - c.clip, 1 + c.clip) * A).mean()
                v_loss = ((vals - b.ret[:, s:e]) ** 2).mean()
                ent = ents.mean()
                loss = pi_loss + c.vf_coef * v_loss - c.ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.brain.parameters(), c.max_grad)
                self.opt.step()
                k = c.epochs * n_blocks
                stats["pi"] += pi_loss.item() / k
                stats["v"] += v_loss.item() / k
                stats["ent"] += ent.item() / k
                stats["kl"] += (b.logp[:, s:e] - logp.detach()).mean().item() / k
        return stats
