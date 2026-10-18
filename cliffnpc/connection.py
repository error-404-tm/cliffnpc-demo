"""A learned connection: situation -> drive-space bivector (learned tendencies).

The spec's connection (§4.1) maps position to a bivector; here it maps the
agent's *situation* (a context vector x) to a bivector that pushes the drives:

    beta_conn(x) = sum_f x_f W_f          (W_f: one learned bivector per feature)

so the drives move as dC/dt = -(beta_world + beta_conn(x)) g C.  This is a
learned gauge field over situation space.  The metric g (personality) acts the
same way in every situation; the connection can say "rivals near and allies near
-> rotate toward fight; rivals near and alone -> toward flight".

Learning is the same pain aversion as for the metric.  beta_conn is linear in
W, so the effect of W on each drive's rate has a closed form:

    d(s_k dC_k/dt)/d W_f[i, j] = -x_f s_k (delta_ki u_j - delta_kj u_i),  u = g C

accumulated in a decaying eligibility trace per drive axis; pain blamed on axis
k moves W against what was amplifying that axis in that situation.

Context features (8, a power of two), each in [0, 1]:
    food_near, rival_near, rival_closing, allies, rivals, outnumbered, injured, backed
"outnumbered" (rivals per ally) and "backed" (rivals x allies) are the two
interaction features a linear map needs to make fight/flight depend on both
rivals and allies.  There is no constant feature: context-free tendencies
belong to the metric (personality), not to the habits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .algebra import PLANES, generator_from_bivector
from .config import ConnectionConfig

FEATURES = ("food_near", "rival_near", "rival_closing", "allies", "rivals", "outnumbered",
            "injured", "backed")
N_FEAT = len(FEATURES)
DRIVES = ("hunger", "social", "explore", "fight")
PLANE_NAMES = tuple(f"{DRIVES[i]}^{DRIVES[j]}" for i, j in PLANES)


def context_vector(food_dist: float, rival_dist: float | None, closing: float, allies: int,
                   rivals: int, health: float, sensor_radius: float) -> np.ndarray:
    R = sensor_radius
    return np.array([
        max(0.0, 1.0 - food_dist / R),
        0.0 if rival_dist is None else max(0.0, 1.0 - rival_dist / R),
        float(np.clip(closing / 5.0, 0.0, 1.0)),
        min(allies / 4.0, 1.0),
        min(rivals / 4.0, 1.0),
        min(rivals / (1.0 + allies) / 2.0, 1.0),
        1.0 - health,
        min(rivals * allies / 4.0, 1.0),
    ])


def _antisym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M - np.swapaxes(M, -1, -2))


@dataclass
class LearnedConnection:
    cfg: ConnectionConfig
    W: np.ndarray = field(default_factory=lambda: np.zeros((N_FEAT, 4, 4)))
    trace: np.ndarray = field(default_factory=lambda: np.zeros((4, N_FEAT, 4, 4)))  # axis, feature
    # learning ledger (for interrogation)
    updates: int = 0
    pain_updates: int = 0
    reward_updates: int = 0
    dW_total: np.ndarray = field(default_factory=lambda: np.zeros((N_FEAT, 4, 4)))   # signed sum
    dW_abs: np.ndarray = field(default_factory=lambda: np.zeros(N_FEAT))             # |dW| per feature
    last_x: np.ndarray = field(default_factory=lambda: np.zeros(N_FEAT))

    # --- forcing ------------------------------------------------------------
    def beta(self, x: np.ndarray) -> np.ndarray:
        """Bivector the connection contributes in situation x."""
        return self.cfg.gain * np.einsum("f,fij->ij", x, self.W)

    def response(self, x: np.ndarray, C: np.ndarray, g: np.ndarray) -> np.ndarray:
        """dC/dt the connection alone would cause in situation x (for interrogation)."""
        return generator_from_bivector(self.beta(x), g) @ C

    # --- learning -------------------------------------------------------------
    def accumulate(self, x: np.ndarray, C: np.ndarray, g: np.ndarray, h: float,
                   action_scale: float, trace_tau: float) -> None:
        """Eligibility: d(s_k dC_k/dt)/dW_f for each drive axis k and feature f."""
        self.last_x = x
        u = g @ C
        s = np.tanh(C / action_scale)
        E = np.zeros((4, 4, 4))                       # axis k, bivector (i, j)
        for k in range(4):
            E[k, k, :] -= s[k] * u                    # -s_k u_j at (k, j)
            E[k, :, k] += s[k] * u                    # +s_k u_i at (i, k)
        # E[k] holds the derivative for each antisymmetric pair at both (i, j) and
        # (j, i); as a gradient over matrix entries that counts every pair twice.
        E = 0.5 * _antisym(E) * self.cfg.gain
        self.trace = (np.exp(-h / trace_tau) * self.trace
                      + h * np.einsum("f,kij->kfij", x, E))

    def learn(self, blame: np.ndarray) -> np.ndarray:
        """W_f <- W_f - rate * sum_k blame_k T_kf, with decay and a norm cap.  Returns dW."""
        c = self.cfg
        dW = -c.learn_rate * np.einsum("k,kfij->fij", blame, self.trace)
        dW = _antisym(dW)
        self.W = (1.0 - c.decay) * self.W + dW
        n = np.linalg.norm(self.W, axis=(1, 2))
        over = n > c.max_norm
        self.W[over] *= (c.max_norm / n[over])[:, None, None]
        # ledger
        self.updates += 1
        if blame.sum() > 0:
            self.pain_updates += 1
        elif blame.sum() < 0:
            self.reward_updates += 1
        self.dW_total += dW
        self.dW_abs += np.linalg.norm(dW, axis=(1, 2))
        return dW

    def inherit(self, parents: list["LearnedConnection"], factor: float) -> None:
        self.W = factor * np.mean([p.W for p in parents], axis=0)

    # --- interrogation --------------------------------------------------------
    def planes(self) -> dict:
        """Learned bivector per feature, as {feature: {plane: coefficient}}."""
        return {f: {pn: float(self.W[k, i, j]) for pn, (i, j) in zip(PLANE_NAMES, PLANES)}
                for k, f in enumerate(FEATURES)}

    def strongest(self, n: int = 5) -> list[tuple]:
        """The n largest learned (feature, plane, coefficient) entries."""
        rows = [(f, pn, float(self.W[k, i, j])) for k, f in enumerate(FEATURES)
                for pn, (i, j) in zip(PLANE_NAMES, PLANES)]
        return sorted(rows, key=lambda r: -abs(r[2]))[:n]

    def holonomy(self, contexts: list[np.ndarray], g: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Transport around a loop of situations: prod exp(dt * M(x_i)).  Its distance
        from the identity measures how path-dependent the learned habits are."""
        from scipy.linalg import expm
        H = np.eye(4)
        for x in contexts:
            H = expm(dt * generator_from_bivector(self.beta(x), g)) @ H
        return H
