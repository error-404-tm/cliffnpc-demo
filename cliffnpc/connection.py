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

Context features (16, a power of two), each in [0, 1]:
    food_near, rival_near, rival_closing, allies, rivals, outnumbered, injured, backed,
    food_cue_a, food_cue_b (one-hot: how the nearest seen food looks, look A or B),
    place_cue (nearness to a marked place), and 5 spare slots (zero)
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
            "injured", "backed", "food_cue_a", "food_cue_b", "place_cue",
            "spare0", "spare1", "spare2", "spare3", "spare4")
N_FEAT = len(FEATURES)
DRIVES = ("hunger", "social", "explore", "fight")
PLANE_NAMES = tuple(f"{DRIVES[i]}^{DRIVES[j]}" for i, j in PLANES)


def context_vector(food_dist: float, rival_dist: float | None, closing: float, allies: int,
                   rivals: int, health: float, sensor_radius: float,
                   food_cue: int = 0, place_cue: float = 0.0) -> np.ndarray:
    """food_cue: 0 no food in view (or no visible mark), 1 look A, 2 look B."""
    R = sensor_radius
    x = np.zeros(N_FEAT)
    x[:11] = [
        max(0.0, 1.0 - food_dist / R),
        0.0 if rival_dist is None else max(0.0, 1.0 - rival_dist / R),
        float(np.clip(closing / 5.0, 0.0, 1.0)),
        min(allies / 4.0, 1.0),
        min(rivals / 4.0, 1.0),
        min(rivals / (1.0 + allies) / 2.0, 1.0),
        1.0 - health,
        min(rivals * allies / 4.0, 1.0),
        float(food_cue == 1),
        float(food_cue == 2),
        place_cue,
    ]
    return x


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
    # TD actor-critic state (rule == "td")
    rng: np.random.Generator | None = None
    v: np.ndarray = field(default_factory=lambda: np.zeros(N_FEAT + 1))   # critic (features + bias)
    xi: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))       # exploration bivector
    e_actor: np.ndarray = field(default_factory=lambda: np.zeros((N_FEAT, 4, 4)))
    e_critic: np.ndarray = field(default_factory=lambda: np.zeros(N_FEAT + 1))
    prev_phi: np.ndarray | None = None
    S: np.ndarray = field(default_factory=lambda: np.zeros((N_FEAT, 4, 4)))   # p part (td + cartan), symmetric
    xs: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))          # exploration in p
    e_S: np.ndarray = field(default_factory=lambda: np.zeros((N_FEAT, 4, 4)))
    U: np.ndarray = field(default_factory=lambda: np.zeros((4, N_FEAT)))      # affine push (td + affine)
    zeta: np.ndarray = field(default_factory=lambda: np.zeros(4))
    e_push: np.ndarray = field(default_factory=lambda: np.zeros((4, N_FEAT)))
    td_errors: list = field(default_factory=list)                          # (delta, reward) per tick

    # --- forcing ------------------------------------------------------------
    def beta(self, x: np.ndarray) -> np.ndarray:
        """Bivector the connection contributes in situation x."""
        return self.cfg.gain * np.einsum("f,fij->ij", x, self.W)

    def act(self, x: np.ndarray) -> np.ndarray:
        """Bivector actually applied: the learned habit plus exploration (td only)."""
        return self.beta(x) + self.xi

    def p_generator(self, x: np.ndarray, g: np.ndarray, explore: bool = False) -> np.ndarray:
        """The p part in situation x: P = g^-1 S(x), g-symmetric (g P is symmetric)."""
        S = np.einsum("f,fij->ij", x, self.S) + (self.xs if explore else 0.0)
        return np.linalg.solve(g, S)

    def push(self, x: np.ndarray, C: np.ndarray, g: np.ndarray) -> np.ndarray | None:
        """Drive-rate terms beyond the bivector: the p part P C (cartan) and the
        direct push U x (affine), with exploration.  None when neither is on."""
        c = self.cfg
        if c.rule != "td" or not (c.cartan or c.affine):
            return None
        out = np.zeros(4)
        if c.cartan:
            out += self.p_generator(x, g, explore=True) @ C
        if c.affine:
            out += self.U @ x + self.zeta
        return out

    def response(self, x: np.ndarray, C: np.ndarray, g: np.ndarray) -> np.ndarray:
        """dC/dt the connection alone would cause in situation x (for interrogation),
        without exploration: bivector part, plus the p part and push when learned."""
        r = generator_from_bivector(self.beta(x), g) @ C
        if self.cfg.rule == "td" and self.cfg.cartan:
            r = r + self.p_generator(x, g) @ C
        if self.cfg.rule == "td" and self.cfg.affine:
            r = r + self.U @ x
        return r

    def response_parts(self, x: np.ndarray, C: np.ndarray, g: np.ndarray) -> dict:
        """The response split by algebra part: so(g) (redirects, keeps C.gC) and p (rescales)."""
        rot = generator_from_bivector(self.beta(x), g) @ C
        p = self.p_generator(x, g) @ C if self.cfg.cartan else np.zeros(4)
        return dict(rotation=rot, p=p, interval_rate=float(2 * C @ g @ (rot + p)))

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

    def td_update(self, x: np.ndarray, reward: float, h: float) -> float | None:
        """One actor-critic step.  `reward` was earned over the tick that just ended,
        under the previous situation and the exploration then applied.  Returns the TD error."""
        c = self.cfg
        phi = np.append(x, 1.0)
        gamma = np.exp(-h / c.td_horizon)
        delta = None
        if self.prev_phi is not None:
            delta = reward + gamma * self.v @ phi - self.v @ self.prev_phi
            self.v += c.critic_rate * delta * self.e_critic / (1.0 + self.prev_phi @ self.prev_phi)
            dW = c.actor_rate * delta * self.e_actor
            n = np.linalg.norm(dW, axis=(1, 2))           # bound each update per feature
            big = n > c.actor_step_max
            dW[big] *= (c.actor_step_max / n[big])[:, None, None]
            if c.cartan and not c.learn_rotation:
                dW[:] = 0.0
            self.W = self.W + dW
            if c.cartan:
                dS = c.actor_rate * delta * self.e_S
                n = np.linalg.norm(dS, axis=(1, 2))
                big = n > c.actor_step_max
                dS[big] *= (c.actor_step_max / n[big])[:, None, None]
                self.S = self.S + dS
                n = np.linalg.norm(self.S, axis=(1, 2))
                over = n > c.td_max_norm
                self.S[over] *= (c.td_max_norm / n[over])[:, None, None]
            if c.affine:
                dU = c.actor_rate * delta * self.e_push
                n = np.linalg.norm(dU, axis=0)
                big = n > c.actor_step_max
                dU[:, big] *= c.actor_step_max / n[big]
                self.U = np.clip(self.U + dU, -c.td_max_norm, c.td_max_norm)
            n = np.linalg.norm(self.W, axis=(1, 2))
            over = n > c.td_max_norm
            self.W[over] *= (c.td_max_norm / n[over])[:, None, None]
            self.updates += 1
            self.pain_updates += delta < 0
            self.reward_updates += delta > 0
            self.dW_total += dW
            self.dW_abs += np.linalg.norm(dW, axis=(1, 2))
            self.td_errors.append((float(delta), float(reward)))
        # exploration for the coming tick: Ornstein-Uhlenbeck in the six planes
        k = np.exp(-h / c.noise_tau)
        z = np.zeros((4, 4))
        for (i, j), r in zip(PLANES, self.rng.normal(size=len(PLANES))):
            z[i, j], z[j, i] = r, -r
        self.xi = k * self.xi + np.sqrt(1.0 - k * k) * c.noise_sigma * z
        decay = gamma * c.td_lambda
        if c.cartan:
            z = np.triu(self.rng.normal(size=(4, 4)))
            z = (z + z.T) / np.sqrt(2.0)                  # symmetric; diagonal and off-diagonal alike
            z[np.diag_indices(4)] /= np.sqrt(2.0)
            self.xs = k * self.xs + np.sqrt(1.0 - k * k) * c.p_noise_sigma * z
            self.e_S = decay * self.e_S + np.einsum("f,ij->fij", x, self.xs) / c.p_noise_sigma
        if c.affine:
            self.zeta = k * self.zeta + np.sqrt(1.0 - k * k) * c.noise_sigma * self.rng.normal(size=4)
            self.e_push = decay * self.e_push + np.outer(self.zeta, x) / c.noise_sigma
        self.e_critic = decay * self.e_critic + phi
        self.e_actor = decay * self.e_actor + np.einsum("f,ij->fij", x, self.xi) / c.noise_sigma
        self.prev_phi = phi
        self.last_x = x
        return delta

    def reset_episode(self) -> None:
        """Forget the TD bootstrap and traces (after a discontinuity such as a teleport)."""
        self.prev_phi = None
        self.e_S[:] = 0.0
        self.e_push[:] = 0.0
        self.e_actor[:] = 0.0
        self.e_critic[:] = 0.0

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
