"""Online, per-NPC adaptation during play: the shared brain stays frozen, each NPC learns
only its own adapter row (a scale and shift on the hidden units) from its own experience.

Every `window` ticks (16 = 3.2 s):
  * the window is replayed through ONE fused GRU pass with the adapters requiring grad;
  * advantages come from the NPC's own value heads -- TD(lambda) per goal, bootstrapped at
    the window's end -- scalarised by its preference vector;
  * loss = sum over NPCs of (policy-gradient loss + value loss): summed per NPC, so each
    NPC's adapter gradient is the same whatever the population size, and -- because the
    adapters act row-wise -- one backward pass yields every NPC's gradient at once;
  * SGD on the adapters, then a pull back toward the pretrained behaviour (decay toward
    zero) and a cap on each row's norm: guards against drift and against players
    training NPCs into degenerate behaviour.

Buffers are a preallocated time-major ring [window, N, ...] (layout.py rules).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .brain import Brain, NPCState, bern_logp, gauss_logp
from .layout import Layout


@dataclass
class AdaptConfig:
    window: int = 16
    lr: float = 0.02
    decay: float = 0.002             # per update: pull toward the pretrained network (adapters -> 0)
    max_norm: float = 2.0            # per NPC, per adapter tensor
    gamma: float = 0.99
    lam: float = 0.9
    vf_coef: float = 0.5
    hidden: bool = True              # learn the hidden-unit scale/shift
    input: bool = False              # learn the per-NPC input adapter (cue -> encoder)
    input_lr: float = 0.05
    input_max_norm: float = 2.0


class OnlineAdapter:
    def __init__(self, brain: Brain, L: Layout, reward_scale: torch.Tensor, cfg: AdaptConfig = AdaptConfig()):
        self.brain, self.L, self.cfg = brain, L, cfg
        for p in brain.parameters():
            p.requires_grad_(False)                       # shared weights frozen
        U, N = cfg.window, L.n
        z = torch.zeros
        self.obs = z(U, N, L.features)
        self.u = z(U, N, 2)
        self.fight = z(U, N, dtype=torch.bool)
        self.reward = z(U, N, L.goals)
        self.h_start = z(N, L.hidden)
        self.scale = reward_scale.clone()                 # per-goal reward scale from pretraining
        self.k = 0
        self.updates = 0
        self.seconds = 0.0

    def begin_window(self, st: NPCState) -> None:
        if self.k == 0:
            self.h_start.copy_(st.hidden)

    def record(self, obs: torch.Tensor, u: torch.Tensor, fight: torch.Tensor, reward: torch.Tensor) -> None:
        k = self.k
        self.obs[k].copy_(obs)
        self.u[k].copy_(u)
        self.fight[k].copy_(fight)
        self.reward[k].copy_(reward)
        self.k += 1

    def ready(self) -> bool:
        return self.k == self.cfg.window

    def update(self, st: NPCState, next_obs: torch.Tensor, pref: torch.Tensor) -> dict:
        """One adapter step for every NPC.  next_obs [N, F]; pref [N, G]; st.hidden is the
        hidden state after the window (used for the bootstrap value)."""
        import time
        t0 = time.perf_counter()
        c, U = self.cfg, self.cfg.window
        scale = st.scale.detach().requires_grad_(c.hidden)
        shift = st.shift.detach().requires_grad_(c.hidden)
        inp = st.inp.detach().requires_grad_(c.input)
        po, vals = self.brain.sequence(self.obs, self.h_start, scale, shift, inp)  # [U, N, 3], [U, N, G]
        with torch.no_grad():
            _, v_end, _ = self.brain.step(next_obs, st.hidden, st.scale, st.shift, st.inp)
            r = self.reward / self.scale
            v = vals.detach()
            adv = torch.zeros_like(r)
            gae = torch.zeros_like(v_end)
            nxt = v_end
            for t in reversed(range(U)):
                delta = r[t] + c.gamma * nxt - v[t]
                gae = delta + c.gamma * c.lam * gae
                adv[t] = gae
                nxt = v[t]
            ret = adv + v
            a = (adv * pref.unsqueeze(0)).sum(-1)
            a = (a - a.mean()) / (a.std() + 1e-8)
        mu, fl = po[..., :2], po[..., 2]
        logp = gauss_logp(self.u, mu, self.brain.log_std) + bern_logp(self.fight, fl)
        loss = -(logp * a).sum() / U + c.vf_coef * ((vals - ret) ** 2).sum() / U
        loss.backward()
        with torch.no_grad():
            if c.hidden:
                for buf, g in ((st.scale, scale.grad), (st.shift, shift.grad)):
                    buf.sub_(c.lr * g).mul_(1.0 - c.decay)
                    n = buf.norm(dim=-1, keepdim=True)
                    buf.mul_((c.max_norm / n.clamp_min(1e-12)).clamp_max(1.0))
            if c.input:
                st.inp.sub_(c.input_lr * inp.grad).mul_(1.0 - c.decay)
                n = st.inp.flatten(1).norm(dim=-1).view(-1, 1, 1)
                st.inp.mul_((c.input_max_norm / n.clamp_min(1e-12)).clamp_max(1.0))
        self.k = 0
        self.updates += 1
        self.seconds += time.perf_counter() - t0
        return dict(loss=float(loss.detach()), scale_norm=float(st.scale.norm(dim=-1).mean()),
                    shift_norm=float(st.shift.norm(dim=-1).mean()))
