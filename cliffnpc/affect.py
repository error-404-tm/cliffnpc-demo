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
                  cfg: AffectConfig, fear: float = 0.0) -> np.ndarray:
    """The morphism (drive state, body, perception) -> affect space."""
    return np.array([
        C[0],                                     # hungry hurts, sated pleases
        cfg.social_sign * C[1],                   # lonely hurts (or crowded, if anti-social)
        injury_taken - cfg.triumph_weight * injury_dealt,
        fear,                                     # anticipated injury (rival perceived)
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
    """R (4 x 4): affect axis -> drive axis, with the chain-rule signs of the morphism.
    Injury and fear are both blamed on the fight/flight axis: approaching is
    what brings contact."""
    R = np.zeros((4, 4))
    R[0, 0] = 1.0
    R[1, 1] = cfg.social_sign
    R[2, 3] = 1.0
    R[3, 3] = 1.0
    return R


def metric_ok(g: np.ndarray, cfg: AffectConfig, eta: np.ndarray | None = None) -> bool:
    """Semantic integrity of a learned metric.

    exp-updates never change the signature, but strong learning can still swap
    which *drive* is timelike (e.g. explore becomes timelike and threat
    spacelike), and the straight path back to g0 then crosses a degenerate
    metric.  So every drive keeps the causal character its signature gives it
    (eta), by a margin, and g stays well conditioned (checked on g, which is
    gauge-invariant; L is not).
    """
    s = np.diag(ga.ETA if eta is None else eta)
    d = np.diag(g)
    return bool(np.all(s * d >= cfg.min_axis_norm) and np.linalg.cond(g) <= cfg.max_cond)


@dataclass
class AffectState:
    cfg: AffectConfig
    a: np.ndarray = field(default_factory=lambda: np.zeros(4))
    a_hat: np.ndarray | None = None
    trace: np.ndarray = field(default_factory=lambda: np.zeros((4, 4, 4)))   # per drive axis
    taken_rate: float = 0.0       # smoothed injury taken per second
    dealt_rate: float = 0.0       # smoothed injury dealt per second
    delta: float = 0.0            # < w, a - a_hat >
    blame: np.ndarray = field(default_factory=lambda: np.zeros(4))   # v = R^T (w * err)
    pain: float = 0.0             # < w, a >
    fear_now: float = 0.0         # perceived threat level (set each tick)
    fear_learned: float = 0.0     # learned fear sensitivity, added to the innate weight

    @property
    def weights(self) -> np.ndarray:
        w = self.cfg.weights.copy()
        w[3] += self.fear_learned
        return w

    def observe_injury(self, taken: float, dealt: float, h: float) -> None:
        """Feed the injury taken/dealt during the last h seconds.  Injury also
        sensitizes fear; learned fear slowly fades."""
        c = self.cfg
        k = np.exp(-h / c.injury_tau)
        self.taken_rate = k * self.taken_rate + (1 - k) * taken / max(h, 1e-9)
        self.dealt_rate = k * self.dealt_rate + (1 - k) * dealt / max(h, 1e-9)
        self.fear_learned *= np.exp(-c.fear_decay * h)
        self.sensitize(c.fear_from_injury * taken)

    def sensitize(self, amount: float) -> None:
        self.fear_learned = min(self.fear_learned + amount, self.cfg.fear_max)

    def step(self, C: np.ndarray, beta_total: np.ndarray, h: float) -> float:
        """Update affect, prediction error and eligibility; return delta."""
        c = self.cfg
        w = self.weights
        self.a = affect_vector(C, self.taken_rate, self.dealt_rate, c, self.fear_now)
        if self.a_hat is None:
            self.a_hat = self.a.copy()
        err = self.a - self.a_hat
        self.delta = float(w @ err)
        self.blame = responsibility(c).T @ (w * err)
        self.pain = float(w @ self.a)
        self.a_hat = self.a_hat + c.expect_rate * err
        self.trace = (np.exp(-h / c.trace_tau) * self.trace
                      + h * axis_eligibilities(beta_total, C, c.action_scale))
        return self.delta

    def learn(self, L: np.ndarray, eta: np.ndarray | None = None) -> np.ndarray:
        """Pain aversion: g <- g - rate * sum_k v_k T_k (via the frame).

        Returns the raw proposal; the caller commits it through the frame law
        (NPC.commit_frame)."""
        dG = -self.cfg.learn_rate * np.einsum("k,kij->ij", self.blame, self.trace)
        if not np.any(dG):
            return L
        return apply_frame_update(L, metric_delta_generator(L, dG, eta), 1.0, self.cfg.max_step)


# ===========================================================================
# Affects as trivectors
# ===========================================================================
#
# Drives are vectors (hunger, social, explore, fight/flight).  The grade-3 space
# of the same algebra is 4-dimensional and dual to the vectors (T = V I), so it
# has exactly one affect per drive:
#
#     e123  hunger^social^explore      dual to fight/flight   (no-threat affect)
#     e124  hunger^social^fight        dual to explore        (fear, type 1)
#     e134  hunger^explore^fight       dual to social         (fear, type 2)
#     e234  social^explore^fight       dual to hunger         (fear, type 3)
#
# For a drive state C and a push B, CB = C.B + C^B: the vector part moves the
# drives (behaviour), the trivector part is what the drives cannot absorb.  That
# part is the affect.  Felt affect uses the metric g induces on trivectors (its
# 3x3 minors), which by the cofactor identity is det(g) g^{-1} seen through the
# duality: affect sensitivity is the drive personality, not a separate object.

from .algebra import TRIPLES, bivector_from_planes, vector_wedge_bivector  # noqa: E402
from .config import TrivectorConfig  # noqa: E402

AFFECT_NAMES = ("e123", "e124 fear", "e134 fear", "e234 fear")
INCIDENCE = np.array([[1.0 if i in t else 0.0 for i in range(4)] for t in TRIPLES])  # affect x drive

THREAT_PLANES = bivector_from_planes({(0, 3): 1.0, (1, 3): 1.0, (2, 3): 1.0})
FORAGE_PLANE = bivector_from_planes({(0, 2): 1.0})
SOCIAL_PLANE = bivector_from_planes({(1, 2): 1.0})
MOB_PLANE = bivector_from_planes({(3, 1): 1.0})       # e4 ^ e2: social-threat (option A)


def induced_trivector_metric(g: np.ndarray) -> np.ndarray:
    """G[a, b] = <e_a, e_b> for basis trivectors: the 3x3 minors of g (Gram determinants)."""
    G = np.empty((4, 4))
    for a, ta in enumerate(TRIPLES):
        for b, tb in enumerate(TRIPLES):
            G[a, b] = np.linalg.det(g[np.ix_(ta, tb)])
    return G


def felt_matrix(g: np.ndarray) -> np.ndarray:
    """Induced metric with each row's sign normalized so its diagonal is positive,
    divided by |det g|^(3/4) so it depends on the *shape* of the personality and
    not its overall scale (the minors grow as scale^3, det as scale^4).  At
    g = eta this is the identity (P is felt as is); a learned g adds entanglement
    between affect types and rescales each by its cofactor."""
    G = induced_trivector_metric(g)
    s = np.sign(np.diag(G))
    s[s == 0] = 1.0
    return s[:, None] * G / max(abs(np.linalg.det(g)), 1e-12) ** 0.75


def contract(C: np.ndarray, T: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Bivector array of C _| T for a vector C and trivector T (components over TRIPLES):
    C _| (e_i^e_j^e_k) = (C.e_i) e_j^e_k - (C.e_j) e_i^e_k + (C.e_k) e_i^e_j."""
    u = g @ C
    beta = np.zeros((4, 4))

    def add(p, q, v):
        beta[p, q] += v
        beta[q, p] -= v
    for (i, j, k), t in zip(TRIPLES, T):
        add(j, k, t * u[i])
        add(i, k, -t * u[j])
        add(i, j, t * u[k])
    return beta


@dataclass
class TrivectorAffect:
    """Affect state P (signed: + pain, - reward) on the four trivectors."""
    tri: TrivectorConfig
    cfg: AffectConfig                       # learning settings (pain aversion)
    P: np.ndarray = field(default_factory=lambda: np.zeros(4))
    F: np.ndarray = field(default_factory=lambda: np.zeros(4))      # felt affect
    F_hat: np.ndarray | None = None
    C_hat: np.ndarray | None = None
    trace: np.ndarray = field(default_factory=lambda: np.zeros((4, 4, 4)))
    taken_rate: float = 0.0
    dealt_rate: float = 0.0
    pending_death: float = 0.0
    pending_false: float = 0.0
    delta: float = 0.0
    blame: np.ndarray = field(default_factory=lambda: np.zeros(4))
    pain: float = 0.0
    fear_now: float = 0.0                   # interface compatibility (unused)
    fear_learned: float = 0.0               # interface compatibility: sensitivity comes from g
    # interrogation: how much affect came from real events vs replayed memory
    src_real: float = 0.0                   # cumulative |source| from live events
    src_primed: float = 0.0                 # cumulative |source| from replayed events

    @property
    def a(self) -> np.ndarray:
        return self.F

    @property
    def weights(self) -> np.ndarray:
        return np.ones(4)

    def sensitize(self, amount: float) -> None:      # 1a: no separate learned sensitivity
        pass

    def observe_injury(self, taken: float, dealt: float, h: float) -> None:
        k = np.exp(-h / self.cfg.injury_tau)
        self.taken_rate = k * self.taken_rate + (1 - k) * taken / max(h, 1e-9)
        self.dealt_rate = k * self.dealt_rate + (1 - k) * dealt / max(h, 1e-9)

    def add_death(self, factor: float = 1.0) -> None:
        self.pending_death += factor

    def add_false_food(self) -> None:
        self.pending_false += 1.0

    def events(self, C: np.ndarray, beta_ext: np.ndarray, eat: float, density: float, h: float):
        t = self.tri
        ev = [
            (+1.0, t.need_hunger * max(C[0], 0.0) * FORAGE_PLANE),
            (-1.0, t.meal * eat * FORAGE_PLANE),
            (+1.0, t.need_social * max(C[1], 0.0) * SOCIAL_PLANE),
            (-1.0, t.company * density * SOCIAL_PLANE),
            (+1.0, t.injury * self.taken_rate * THREAT_PLANES),
            (-1.0, t.triumph * self.dealt_rate * THREAT_PLANES),
            (+1.0, t.death * self.pending_death / h * THREAT_PLANES),
            (+1.0, t.false_food * self.pending_false / h * FORAGE_PLANE),
            (+1.0, t.frustration * beta_ext),
        ]
        self.pending_death = self.pending_false = 0.0
        return ev

    def _felt_source(self, C, surprise_vec, v, B) -> np.ndarray:
        change = vector_wedge_bivector(C, B)                   # felt now
        surprise = vector_wedge_bivector(surprise_vec, B)      # departure from expectation
        return v * (np.abs(change) + self.tri.surprise_gain * np.abs(surprise))

    def step(self, C: np.ndarray, g: np.ndarray, beta_ext: np.ndarray, beta_total: np.ndarray,
             h: float, eat: float = 0.0, density: float = 0.0,
             primed: list | None = None) -> float:
        """primed: replayed memory events (valence, bivector impulse), felt like real
        events at reduced strength (the memory system already scaled them)."""
        t = self.tri
        if self.C_hat is None:
            self.C_hat = C.copy()
        surprise_vec = C - self.C_hat
        src = np.zeros(4)
        for v, B in self.events(C, beta_ext, eat, density, h):
            if not np.any(B):
                continue
            src += self._felt_source(C, surprise_vec, v, B)
        self.src_real += float(np.abs(src).sum()) * h
        if primed:
            ps = np.zeros(4)
            for v, B in primed:
                ps += self._felt_source(C, surprise_vec, v, B / h)   # an impulse this tick
            src += ps
            self.src_primed += float(np.abs(ps).sum()) * h
        # leak + cubic saturation, semi-implicit (stable for any size of P)
        self.P = (self.P + h * src) / (1.0 + h * (t.decay + t.sat_gain * (self.P / t.sat_scale) ** 2))
        self.C_hat = self.C_hat + t.expect_rate * (C - self.C_hat)
        self.F = felt_matrix(g) @ self.P
        if self.F_hat is None:
            self.F_hat = self.F.copy()
        err = self.F - self.F_hat
        self.delta = float(err.sum())
        self.blame = INCIDENCE.T @ err                         # each affect -> its three drives
        self.pain = float(self.F.sum())
        self.F_hat = self.F_hat + t.baseline_rate * err
        self.trace = (np.exp(-h / self.cfg.trace_tau) * self.trace
                      + h * axis_eligibilities(beta_total, C, self.cfg.action_scale))
        return self.delta

    def behaviour_bivector(self, C: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Affect acts back on the drives: the contraction C _| F, as a generator
        (strength capped so a single shock cannot spin the drives arbitrarily fast)."""
        beta = self.tri.contraction_gain * contract(C, self.F, g)
        n = np.linalg.norm(beta)
        cap = self.tri.contraction_cap
        return beta if n <= cap else beta * (cap / n)

    def learn(self, L: np.ndarray, eta: np.ndarray | None = None) -> np.ndarray:
        dG = -self.cfg.learn_rate * np.einsum("k,kij->ij", self.blame, self.trace)
        if not np.any(dG):
            return L
        return apply_frame_update(L, metric_delta_generator(L, dG, eta), 1.0, self.cfg.max_step)
