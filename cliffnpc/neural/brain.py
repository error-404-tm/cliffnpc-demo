"""The NPC brain: one small recurrent network shared by every NPC, plus per-NPC state.

    obs (F=32) -> Linear -> ReLU -> GRU (H=64) -> per-NPC adapter -> heads
        policy: 2-D velocity (Gaussian on pre-tanh values) and a fight flag (Bernoulli)
        value:  one head per goal (food, safety, social, knowledge) -- interrogable
                ("it fled because predicted safety dropped")

Shared weights: ~29k parameters (~115 KB float32), small enough to stay in L2 while
NPC rows stream through.  Per-NPC state lives in NPCState, as contiguous [N, ...]
tensors: hidden state, and an adapter -- a scale and shift on the hidden units,
h_eff = h * (1 + scale) + shift -- that gameplay can update per NPC without touching
the shared weights.  Because the adapter acts row-wise, one batched backward pass gives
every NPC's adapter gradient at once.  And because it acts on the GRU's output, not its
recurrence, a whole episode's hidden states come from ONE fused nn.GRU call over a
contiguous [T, B, H] block (sequence), instead of T separate step calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .layout import Layout


@dataclass
class NPCState:
    """Per-NPC tensors, all [N, ...], contiguous, preallocated."""
    hidden: torch.Tensor
    scale: torch.Tensor                 # adapter
    shift: torch.Tensor                 # adapter

    @classmethod
    def zeros(cls, n: int, hidden: int) -> "NPCState":
        z = lambda: torch.zeros(n, hidden, dtype=torch.float32)  # noqa: E731
        return cls(z(), z(), z())


class Brain(nn.Module):
    def __init__(self, layout: Layout):
        super().__init__()
        F, H, G = layout.features, layout.hidden, layout.goals
        self.L = layout
        self.enc = nn.Linear(F, H)
        self.gru = nn.GRU(H, H)                      # (seq, batch, feature); fused sequence kernel
        self.pi = nn.Linear(H, 3)                    # 2 velocity means + fight logit
        self.v = nn.Linear(H, G)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- one step for a batch of rows
    def step(self, obs: torch.Tensor, h: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        """obs [B, F], h [B, H] -> (pi_out [B, 3], value [B, G], h_new [B, H])."""
        out, _ = self.gru(torch.relu(self.enc(obs)).unsqueeze(0), h.unsqueeze(0))
        h_new = out[0]
        he = h_new * (1.0 + scale) + shift
        return self.pi(he), self.v(he), h_new

    def sequence(self, obs: torch.Tensor, h0: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        """obs [T, B, F], h0 [B, H] -> (pi_out [T, B, 3], value [T, B, G]) in one fused pass."""
        out, _ = self.gru(torch.relu(self.enc(obs)), h0.unsqueeze(0))
        he = out * (1.0 + scale) + shift
        return self.pi(he), self.v(he)

    # ---- acting: tiles, preallocated outputs, no autograd
    @torch.inference_mode()
    def act(self, obs: torch.Tensor, st: NPCState, out: "ActBuffers", gen: torch.Generator | None = None,
            greedy: bool = False) -> None:
        """obs [N, F]; writes actions, log-probs and values into `out`; updates st.hidden
        in place.  Processes NPCs in tiles of layout.tile rows."""
        N, T = obs.shape[0], self.L.tile
        std = self.log_std.exp()
        for s in range(0, N, T):
            e = min(s + T, N)
            po, val, hn = self.step(obs[s:e], st.hidden[s:e], st.scale[s:e], st.shift[s:e])
            mu, fl = po[:, :2], po[:, 2]
            if greedy:
                u = mu
                f = fl > 0
            else:
                u = mu + std * torch.randn(mu.shape, generator=gen)
                f = torch.rand(fl.shape, generator=gen) < torch.sigmoid(fl)
            out.u[s:e] = u
            out.fight[s:e] = f
            out.logp[s:e] = gauss_logp(u, mu, self.log_std) + bern_logp(f, fl)
            out.value[s:e] = val
            st.hidden[s:e] = hn


@dataclass
class ActBuffers:
    u: torch.Tensor                     # [N, 2] pre-tanh velocity
    fight: torch.Tensor                 # [N] bool
    logp: torch.Tensor                  # [N]
    value: torch.Tensor                 # [N, G]

    @classmethod
    def zeros(cls, n: int, goals: int) -> "ActBuffers":
        return cls(torch.zeros(n, 2), torch.zeros(n, dtype=torch.bool), torch.zeros(n), torch.zeros(n, goals))


_LOG2PI = math.log(2 * math.pi)


def gauss_logp(u: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    z = (u - mu) / log_std.exp()
    return (-0.5 * z * z - log_std - 0.5 * _LOG2PI).sum(-1)


def bern_logp(f: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    f = f.to(logit.dtype)
    return -nn.functional.binary_cross_entropy_with_logits(logit, f, reduction="none")


def entropy(log_std: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    g = (0.5 + 0.5 * _LOG2PI + log_std).sum()
    p = torch.sigmoid(logit)
    b = -(p * torch.log(p.clamp_min(1e-8)) + (1 - p) * torch.log((1 - p).clamp_min(1e-8)))
    return g + b
