"""NPC cognitive state, dynamics step, readouts, prediction and metric learning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from . import algebra as ga
from .config import CognitionConfig

ETA = ga.ETA


# ---------------------------------------------------------------------------
# Metric (personality) updates on the frame L   (§2.4, §5.2)
# ---------------------------------------------------------------------------

def _cap(A: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(A)
    return A if n <= max_norm else A * (max_norm / n)


def salience_generator(c: np.ndarray, L: np.ndarray) -> np.ndarray:
    """eta-self-adjoint frame generator that makes the surprised direction salient.

    The (sensitivity-weighted) residual is read as a *covector* c (lower
    index, like a gradient).  With v = L^{-T} c (c in the orthonormal frame)
    and sigma = sign(c^T g^{-1} c):
        A = 1/2 sigma eta v v^T.
    eta A = 1/2 sigma v v^T is symmetric, so A has no gauge part, and to first
    order L <- exp(eps A) L gives the Fisher/precision-like accumulation
        g <- g + eps sigma c c^T.
    For u = g^{-1} c, |u.u| grows: repeated surprise along u makes it stiffer,
    and sigma keeps its causal character (det(g + t sigma c c^T) =
    det g (1 + t |c^T g^{-1} c|) never changes sign).  Null c gives no update.

    Why a covector: the earlier choice A = sigma (L u)(L u)^T eta gives
    Delta g ~ (g u)(g u)^T, which feeds back on g and grows exponentially;
    this form is additive in g, so with forgetting toward g0 the metric
    settles at g0 + (eps/lambda) E[sigma c c^T], a bounded running statistic.
    """
    v = np.linalg.solve(L.T, c)
    sigma = np.sign(float(v @ ETA @ v))  # = sign(c^T g^{-1} c)
    return 0.5 * sigma * ETA @ np.outer(v, v)


def felt_covector(r: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Lower the residual vector r with the *current* metric, keeping its size.

        c = |r| * (g r) / |g r|

    The NPC learns from the surprise as it *feels* it (projection readout,
    §2.5), so a later event is interpreted through the geometry left by
    earlier ones.  This is what makes the update order-dependent at first
    order (with c = r alone, Delta g = eps sigma r r^T would commute to first
    order and only the O(eps^2) exp commutator would remember order).  The
    magnitude stays |r|, so the accumulation remains bounded.
    sign(c^T g^{-1} c) = sign(r^T g r): the residual's own causal character.
    """
    gr = g @ r
    n = np.linalg.norm(gr)
    if n == 0.0:
        return np.zeros_like(r)
    return np.linalg.norm(r) * gr / n


def metric_delta_generator(L: np.ndarray, dG: np.ndarray) -> np.ndarray:
    """eta-self-adjoint generator realizing a symmetric metric change dG to first order:
    A = eta L^{-T} dG L^{-1} / 2  gives  g <- g + dG  under  L <- exp(A) L."""
    Linv = np.linalg.inv(L)
    return 0.5 * ETA @ (Linv.T @ (0.5 * (dG + dG.T)) @ Linv)


def toward_metric_generator(L: np.ndarray, g_target: np.ndarray) -> np.ndarray:
    """eta-self-adjoint generator moving g = L^T eta L toward g_target.

    H = L^{-T} g_target L^{-1} is the target metric seen in the current
    orthonormal frame.  A = eta (H - eta)/2 is eta-self-adjoint and gives, to
    first order, g <- g + (g_target - g) under L <- exp(A) L.  It depends only
    on g and g_target (not on the gauge of L).  Used for forgetting toward the
    temperament prior g0 and for gossip diffusion.
    """
    Linv = np.linalg.inv(L)
    H = Linv.T @ g_target @ Linv
    return 0.5 * ETA @ (H - ETA)


def apply_frame_update(L: np.ndarray, A: np.ndarray, eps: float, max_norm: float) -> np.ndarray:
    """L <- exp(eps A) L with ||eps A|| capped.  exp is always invertible, so
    det L never crosses zero and the signature (3,1) is preserved exactly."""
    return expm(_cap(eps * A, max_norm)) @ L


def saturate_frame(L_old: np.ndarray, L_raw: np.ndarray, horizon: float) -> np.ndarray:
    """The frame law: continuous boosts, diminishing returns, never singular.

    Metric-changing generators (eta-self-adjoint, p) do not close: [p, p] lies in
    so(3,1), so composed personality updates also turn the frame by a Lorentz
    transformation, the holonomy of the path through personality space (the
    analogue of the Thomas-Wigner rotation).  Under option B that holonomy is kept:
    it records the order of experience.

    What can become singular is the frame's *stretch*: its singular values.
    So, as with rapidity and velocity in relativity, each singular direction
    accumulates an unbounded experience rapidity phi = horizon * artanh(s / horizon),
    an update adds its change of log-singular-value to phi, and the expressed
    log-stretch is s = horizon * tanh(phi / horizon).  Small early shifts pass
    almost unchanged; repeated pushes the same way saturate.  The rotations,
    holonomy included, are compact and pass through unchanged.
    Hence cond(L) <= exp(2 horizon) for any history.
    """
    U, S, Vt = np.linalg.svd(L_raw)
    # Old stretch measured along each *new* singular direction (pairing by
    # direction, not by sorted order, so crossings don't swap rapidities).
    s_old = np.log(np.linalg.norm(L_old @ Vt.T, axis=0))
    ratio = np.clip(s_old / horizon, -1 + 1e-12, 1 - 1e-12)
    phi = horizon * np.arctanh(ratio) + (np.log(S) - s_old)
    s_new = horizon * np.tanh(phi / horizon)
    return (U * np.exp(s_new)) @ Vt


def learn_frame(L: np.ndarray, r: np.ndarray, frustration: float, c: CognitionConfig,
                rate: float | None = None) -> np.ndarray:
    """One learning step from drive residual r (§5.2): sensitivity weighting
    |g^ii| (§3.4), felt covector, salience generator scaled by frustration."""
    g = ga.metric_from_frame(L)
    cov = ga.sensitivities(g, c.sensitivity_cap) * r if c.use_sensitivity else r.copy()
    if c.felt_residual:
        cov = felt_covector(cov, g)
    scale = 1.0 + c.frustration_gain * frustration
    eps = (c.learn_rate if rate is None else rate) * scale
    return apply_frame_update(L, salience_generator(cov, L), eps, c.max_frame_step)


def forget_frame(L: np.ndarray, L0: np.ndarray, c: CognitionConfig) -> np.ndarray:
    """Decay toward the temperament prior g0 = L0^T eta L0 (not toward zero)."""
    if c.forget_rate <= 0:
        return L
    A = toward_metric_generator(L, ga.metric_from_frame(L0))
    return apply_frame_update(L, A, c.forget_rate, c.max_frame_step)


# ---------------------------------------------------------------------------
# NPC
# ---------------------------------------------------------------------------

INVARIANTS = ("food_dist", "threat_closing", "peer_density", "novelty")


@dataclass
class NPC:
    cfg: CognitionConfig
    L: np.ndarray = field(default_factory=lambda: np.eye(4))
    C: np.ndarray | None = None
    L0: np.ndarray | None = None       # temperament prior frame; g0 = L0^T eta L0
    X_hat: np.ndarray | None = None    # expected invariants (for the next tick)
    X_vel: np.ndarray | None = None    # invariant velocity estimate ("cv" predictor)
    hol_accum: float = 0.0             # accumulated holonomy integral since last tick
    _h_last: float = 0.0
    # bookkeeping of the last tick (for logging)
    last: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.C is None:
            self.C = self.cfg.C0.copy()
        if self.L0 is None:
            self.L0 = self.L.copy()

    # --- views ---------------------------------------------------------
    @property
    def g(self) -> np.ndarray:
        return ga.metric_from_frame(self.L)

    @property
    def g0(self) -> np.ndarray:
        return ga.metric_from_frame(self.L0)

    def commit_frame(self, L_raw: np.ndarray, check=None) -> bool:
        """Adopt a new frame through the frame law; transport the inner state.

        check: optional predicate on the resulting metric; if it fails nothing
        changes.  Returns whether the frame was updated.
        """
        h = self.cfg.frame_horizon
        L_new = saturate_frame(self.L, L_raw, h) if h else L_raw
        if check is not None and not check(ga.metric_from_frame(L_new)):
            return False
        if self.cfg.transport_state:
            # C_o = L C is the agent's experiential state; it stays, its
            # reading in the drive basis changes.  The interval is unchanged.
            self.C = np.linalg.solve(L_new, self.L @ self.C)
        self.L = L_new
        return True

    def coefficients(self) -> np.ndarray:
        """C^i = C . e^i (reciprocal frame).  Drives *action*."""
        return self.C.copy()

    def projections(self) -> np.ndarray:
        """C . e_i = (g C)_i.  What the NPC *feels*."""
        return self.g @ self.C

    def interval(self) -> float:
        return ga.interval(self.C, self.g)

    def cone(self) -> float:
        """Normalized interval s/|C|^2 (scale-free light-cone position)."""
        return ga.normalized_interval(self.C, self.g)

    def regime(self) -> str:
        return ga.regime(self.cone(), self.cfg.null_tol)

    # --- generators ----------------------------------------------------
    def omega_int(self, g: np.ndarray) -> np.ndarray:
        beta = ga.bivector_from_planes(self.cfg.omega_int)
        return ga.generator_from_bivector(beta, g)

    # --- one cognitive tick ---------------------------------------------
    def dynamics_step(self, beta_ext: np.ndarray, h: float, eat: float = 0.0,
                      company: float = 0.0) -> None:
        """Split rotor steps + damping + hunger drift + rapidity clamp (§3)."""
        c = self.cfg
        g = self.g
        # Frustration is measured on the state the world is pushing (§3.3).
        fr_norm, fr_sq = ga.trivector_norms(self.C, beta_ext, self.L)

        # Lie-Trotter: C <- exp(h O_ext) exp(h O_int) C
        C, phi_int = ga.rotor_step(self.C, self.omega_int(g), h, self.L, c.max_rapidity)
        C, phi_ext = ga.rotor_step(C, ga.generator_from_bivector(beta_ext, g), h,
                                   self.L, c.max_rapidity)
        s_after_rotor = ga.interval(C, g)
        # Non-rotor terms
        C = C - h * c.gamma * (C - c.C0)
        C[0] += h * (c.hunger_drift - c.eat_rate * eat * (C[0] - c.sated_level))
        C[1] += h * (c.social_drift - c.company_rate * company * (C[1] - c.company_level))
        n = np.linalg.norm(C)
        if not np.isfinite(n):
            raise FloatingPointError("cognitive state became non-finite")
        if n > c.state_cap:
            C *= c.state_cap / n
        self.C = C
        self.last.update(frustration=fr_norm, frustration_sq=fr_sq,
                         rapidity=max(phi_int, phi_ext), s_after_rotor=s_after_rotor)

    def drive_residual(self, X: np.ndarray, h: float = 1.0) -> np.ndarray:
        """Compare invariants with the prediction, update it, return drive residual r.

        "ema": persistence with a leaky expectation.  "cv": constant velocity,
        X_hat(next) = X + X_vel h, so smooth motion is not surprising and only
        discontinuities (food appears, threat turns) are.
        """
        if self.X_hat is None:
            self.X_hat = X.copy()
            self.X_vel = np.zeros_like(X)
        res_inv = X - self.X_hat
        # For scalar invariants the group is (R, +): R = X - X_hat, log R = R,
        # and X_hat <- exp(eta log R) X_hat is X_hat + eta * R.
        if self.cfg.predictor == "cv":
            prev = self.X_hat - self.X_vel * self._h_last
            self.X_vel = self.X_vel + self.cfg.expect_rate * ((X - prev) / h - self.X_vel)
            self.X_hat = X + self.X_vel * h
            self._h_last = h
        else:
            self.X_hat = self.X_hat + self.cfg.expect_rate * res_inv
        r = self.cfg.residual_map @ res_inv
        self.last.update(residual_inv=res_inv, residual=r)
        return r

    def metric_update(self, r: np.ndarray, learn: bool = True, forget: bool = True) -> bool:
        """L <- exp(eps A(r)) L, then decay toward g0 (§5.2).  Returns True if
        the learning step ran (it is gated by the surprise threshold)."""
        c = self.cfg
        ran = False
        if learn and np.linalg.norm(r) > c.surprise_threshold:
            self.commit_frame(learn_frame(self.L, r, self.last.get("frustration", 0.0), c))
            ran = True
        if forget:
            self.commit_frame(forget_frame(self.L, self.L0, c))
        return ran

    def holonomy_transport(self, alpha: float) -> None:
        """C <- exp(alpha M_h) C for the accumulated connection integral alpha."""
        if alpha == 0.0:
            return
        i, j = self.cfg.holonomy_plane
        M = ga.basis_generator(i, j, self.g)
        self.C = expm(alpha * M) @ self.C

    def tick(self, beta_ext: np.ndarray, X: np.ndarray, h: float, learn: bool = True,
             eat: float = 0.0, forget: bool | None = None) -> np.ndarray:
        """One cognitive tick (spec Phase 1, steps 3, 5, 6).  Returns residual r.

        Step 4 (act) is done by the caller with the steering readout between
        the dynamics and the learning, as the spec orders it.
        """
        self.dynamics_step(beta_ext, h, eat)
        r = self.drive_residual(X, h)
        forget = learn if forget is None else forget
        self.last["learned"] = self.metric_update(r, learn=learn, forget=forget)
        self.holonomy_transport(self.hol_accum)
        self.last["holonomy"] = self.hol_accum
        self.hol_accum = 0.0
        return r
