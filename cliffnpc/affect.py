"""Affect space: pain and reward, and pain-aversion metric learning.

The affect space is a second small Clifford space, Cl(3,0), with one basis
vector per pain source:

    eps_hunger, eps_social, eps_injury

Each axis is bipolar like the drives: + pain, - reward.
    hunger  +hungry (pain)          -sated (pleasure)
    social  +lonely (pain)          -company (pleasure)    [anti-social: crowding hurts]
    injury  +injury taken (pain)    -injury dealt to a rival (triumph)

A linear morphism maps the agent's drive/body state into this space every
tick.  The agent keeps an *expected* affect vector a_hat in the same space,
and the pain prediction error is the affect error projected on its pain
sensitivities w (a covector, i.e. personality):

    delta = < w , a - a_hat >        (> 0: worse than expected)

Pain aversion drives metric learning.  The drive dynamics are dC/dt = M C with
M = -beta g, linear in the metric g.  How strongly the current metric amplifies
the agent's current action tendencies s = tanh(C / c) is

    d/dt (s . C) = -s^T beta g C,   so   d/dg [s . dC/dt] = sym( (beta s) C^T ) =: E

(beta s is the push the forcing gives the active tendencies, paired with the
state).

Credit assignment uses the affect morphism itself.  Each pain axis has a drive
it is "about": hunger pain -> e1 (hunger), social pain -> e2 (social, with the
anti-social sign flip), injury -> e4 (fight/flight: approaching is what makes
contact).  Writing R for this affect -> drive responsibility map, the error is
pulled back to drive space,  v = R^T (w * (a - a_hat)),  and each drive axis k
keeps its own decaying eligibility trace T_k of  E_k = sym((beta s_k e_k) C^T).
The metric moves against pain:

    g <- g - rate * sum_k v_k T_k      (realized as L <- exp(A) L, A eta-self-adjoint)

Pain after an action dampens that action tendency *in that context*; relief or
triumph amplifies it.  Nothing tells the agent which context is safe: the
context is whatever the state C held at the time (e.g. loneliness), entering
through C^T in E_k.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import algebra as ga
from .cognition import apply_frame_update, metric_delta_generator
from .config import AffectConfig

SOURCES = ("hunger", "social", "injury")
DRIVE_OF = (0, 1, 3)        # responsibility map R: affect axis -> drive axis


def affect_vector(C: np.ndarray, injury_taken: float, injury_dealt: float,
                  cfg: AffectConfig) -> np.ndarray:
    """The morphism (drive state, body) -> affect space."""
    return np.array([
        C[0],                                     # hungry hurts, sated pleases
        cfg.social_sign * C[1],                   # lonely hurts (or crowded, if anti-social)
        injury_taken - cfg.triumph_weight * injury_dealt,
    ])


def eligibility_increment(beta: np.ndarray, C: np.ndarray, scale: float) -> np.ndarray:
    """E = sym((beta s) C^T), s = tanh(C/scale): d/dg of s . dC/dt under dC/dt = -beta g C."""
    s = np.tanh(C / scale)
    u = beta @ s
    E = np.outer(u, C)
    return 0.5 * (E + E.T)


def axis_eligibilities(beta: np.ndarray, C: np.ndarray, scale: float) -> np.ndarray:
    """E_k = sym((beta s_k e_k) C^T) for each drive axis k, shape (4, 4, 4).
    Their sum is eligibility_increment(beta, C, scale)."""
    s = np.tanh(C / scale)
    U = beta * s[None, :]                    # column k = beta e_k s_k
    E = np.einsum("ik,j->kij", U, C)
    return 0.5 * (E + E.transpose(0, 2, 1))


def responsibility(cfg: AffectConfig) -> np.ndarray:
    """R (3 x 4): affect axis -> drive axis, with the chain-rule signs of the morphism."""
    R = np.zeros((3, 4))
    R[0, 0] = 1.0
    R[1, 1] = cfg.social_sign
    R[2, 3] = 1.0
    return R


def metric_ok(g: np.ndarray, cfg: AffectConfig) -> bool:
    """Semantic integrity of a learned metric.

    exp-updates never change the signature, but strong learning can still swap
    which *drive* is timelike (e.g. explore becomes timelike and threat
    spacelike), and the straight path back to g0 then crosses a degenerate
    metric.  So: hunger/social/explore stay spacelike, threat stays timelike,
    each by a margin, and g stays well conditioned (checked on g, which is
    gauge-invariant; L is not).
    """
    d = np.diag(g)
    m = cfg.min_axis_norm
    return bool(np.all(d[:3] >= m) and d[3] <= -m and np.linalg.cond(g) <= cfg.max_cond)


@dataclass
class AffectState:
    cfg: AffectConfig
    a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    a_hat: np.ndarray | None = None
    trace: np.ndarray = field(default_factory=lambda: np.zeros((4, 4, 4)))   # per drive axis
    taken_rate: float = 0.0       # smoothed injury taken per second
    dealt_rate: float = 0.0       # smoothed injury dealt per second
    delta: float = 0.0            # < w, a - a_hat >
    blame: np.ndarray = field(default_factory=lambda: np.zeros(4))   # v = R^T (w * err)
    pain: float = 0.0             # < w, a >

    def observe_injury(self, taken: float, dealt: float, h: float) -> None:
        """Feed the injury taken/dealt during the last h seconds."""
        k = np.exp(-h / self.cfg.injury_tau)
        self.taken_rate = k * self.taken_rate + (1 - k) * taken / max(h, 1e-9)
        self.dealt_rate = k * self.dealt_rate + (1 - k) * dealt / max(h, 1e-9)

    def step(self, C: np.ndarray, beta_total: np.ndarray, h: float) -> float:
        """Update affect, prediction error and eligibility; return delta."""
        c = self.cfg
        self.a = affect_vector(C, self.taken_rate, self.dealt_rate, c)
        if self.a_hat is None:
            self.a_hat = self.a.copy()
        err = self.a - self.a_hat
        self.delta = float(c.weights @ err)
        self.blame = responsibility(c).T @ (c.weights * err)
        self.pain = float(c.weights @ self.a)
        self.a_hat = self.a_hat + c.expect_rate * err
        self.trace = (np.exp(-h / c.trace_tau) * self.trace
                      + h * axis_eligibilities(beta_total, C, c.action_scale))
        return self.delta

    def learn(self, L: np.ndarray) -> np.ndarray:
        """Pain aversion: g <- g - rate * sum_k v_k T_k (via the frame).

        Returns the raw proposal; the caller commits it through the frame law
        (NPC.commit_frame)."""
        dG = -self.cfg.learn_rate * np.einsum("k,kij->ij", self.blame, self.trace)
        if not np.any(dG):
            return L
        return apply_frame_update(L, metric_delta_generator(L, dG), 1.0, self.cfg.max_step)
