"""Phase C, rung 0: an agent whose state is the geometry it induces (spec-v2 section 6).

The agent acts like the frontier panel's policies -- weights theta on the six commons
primitives, plus smooth exploration noise -- so its result can be placed against the
measured frontier.  What it learns, online and about itself:

    A   (goals x primitives)  the vector action gradient: how a sustained lean on each
                              primitive changes each goal's progress over a record,
                              estimated from its own exploration noise against vector
                              TD errors (vector return minus a vector critic baseline)
    M = A A^T                 its local goal metric: how goals co-move under its control
                              (the circumstance- and behaviour-induced inner product)

Preferences omega (agent-specific weights on normalised goals) pick a point on the
frontier.  Update rules for theta (the movement toward the frontier):

    "pg"      d = A^T omega                          (Euclidean in control space)
    "ng"      d = A^T (M + lam I)^-1 omega           (natural gradient, written in goal space:
                                                      omega is raised with the inverse local
                                                      goal metric -- goals that are hard to move
                                                      get more weight; damping lam = tr(M)/k
                                                      keeps noise in barely-controllable goals
                                                      from dominating, and if the step would
                                                      not ascend omega it falls back to pg)
    "pareto"  min-norm point of the convex hull of {omega_i a_i}: first improve every
              valued goal at once; when that is impossible (on the frontier), trade off
              along omega slowly

Policy classes: "global" -- one theta for the whole episode (the frontier panel's
class); "ctx" -- situation-dependent theta(s) = Theta phi(s), phi = (bias, hunger,
rival in view, forager near, food in view), 6 x 5 parameters; the action gradient is
then taken with respect to Theta, from the noise times the situation.

Interrogation (report): goal modes (eigenvectors of M), trade-off planes a_i ^ a_j (area
= how independently two goals can be controlled; cosine = aligned or opposed), and the
connection -- how M turns as the agent moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .commons import GOALS, N_PHI, PRIMITIVES, Commons, Dials, record_windows

# per-episode goal scales: spread of the base world's frontier panel (E23), so that
# omega means the same thing in every experiment
GOAL_SCALE = np.array([0.4625, 0.0791, 18.106, 0.6183])
PREFERENCES = {
    "forager":  np.array([1.0, 0.5, 0.2, 0.0]),
    "balanced": np.array([1.0, 1.0, 0.5, 0.0]),
    "cautious": np.array([0.5, 2.0, 0.5, 0.0]),
    "curious":  np.array([1.0, 1.0, 0.5, 1.0]),
}
THETA_BOUNDS = (-1.5, 2.5)


@dataclass
class GoalGeometryAgent:
    dials: Dials
    omega: np.ndarray
    rule: str = "ng"
    seed: int = 0
    noise_sigma: float = 0.5
    lr0: float = 0.25                  # step size in theta per episode (normalised direction)
    lr_tau: float = 80.0               # episodes: lr = lr0 / (1 + episode / lr_tau)
    record_T: int = 32                 # ticks per record (6.4 s)
    forget: float = 0.95               # per-episode decay of the running estimates
    warmup: int = 8                    # episodes of estimation before theta moves
    # critic: "mc" = window return minus a regression baseline V(s0) (E24);
    #         "td" = a per-tick TD critic of the discounted vector return, with a
    #         surprise gate, and bootstrapped advantages
    #         sum_t gamma^t r_t + gamma^T V(s_T) - V(s_0): a rare setback's value shows
    #         up when its predictor (rivals appearing) does, not only when it lands
    policy: str = "global"             # "global" theta or situation-dependent "ctx" Theta
    ng_damping: float = 1.0            # lam = ng_damping * tr(M) / k
    critic: str = "mc"
    td_tau: float = 10.0               # s, discount horizon of the TD critic
    td_rate: float = 0.05
    surprise_z: float = 8.0
    theta: np.ndarray = field(default_factory=lambda: np.full(len(PRIMITIVES), 0.5))
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.omega = np.asarray(self.omega, float)
        self.env = Commons(self.dials, seed=self.seed, noise_override=self.noise_sigma)
        k, n = len(GOALS), len(PRIMITIVES) * (N_PHI if self.policy == "ctx" else 1)
        if self.policy == "ctx":
            self.Theta = np.zeros((len(PRIMITIVES), N_PHI))
            self.Theta[:, 0] = self.theta                    # start: the same policy as "global"
        n_s = 25                                            # senses + bias
        self._SS = np.eye(n_s) * 1e-3                       # critic: sum s s^T
        self._SG = np.zeros((n_s, k))                       #         sum s G^T
        self._LL = np.eye(n) * 1e-3                         # gradient: sum lean lean^T
        self._DL = np.zeros((k, n))                         #           sum delta lean^T
        self.A = np.zeros((k, n))
        self.episode = 0
        self.Vw = np.zeros((n_s, k))                        # TD critic weights (senses + bias -> goals)
        self.gamma = float(np.exp(-0.2 / self.td_tau))      # per tick (0.2 s)
        self._dsd = np.zeros(k)                             # running spread of TD errors per goal

    # -- one learning episode
    def step_episode(self) -> dict:
        c = self.env
        c.theta_fixed = self.theta.copy()
        c.Theta = self.Theta.copy() if self.policy == "ctx" else None
        n0 = len(c.log["r"])
        c.run_episode()
        log = {k: np.asarray(v[n0:]) for k, v in c.log.items()}
        for key in c.log:
            del c.log[key][n0:]
        J = log["r"].sum(0)
        if self.critic == "td":
            self._td_pass(log)
        W = record_windows(log, self.record_T)
        if len(W["G"]):
            G = W["G"] / GOAL_SCALE * (300 / self.record_T)     # per-episode units, normalised
            s0 = np.hstack([W["s"][:, 0, :], np.ones((len(G), 1))])
            lean = self._lean(W)
            f = self.forget
            if self.critic == "td":
                delta = self._bootstrapped_advantage(log) * (300 / self.record_T)
            else:
                self._SS = f * self._SS + s0.T @ s0
                self._SG = f * self._SG + s0.T @ G
                V = s0 @ np.linalg.solve(self._SS, self._SG)   # vector critic baseline
                delta = G - V
            self._LL = f * self._LL + lean.T @ lean
            self._DL = f * self._DL + delta.T @ lean
            self.A = self._DL @ np.linalg.inv(self._LL)       # d goal / d theta
        d = self.direction() if self.episode >= self.warmup else np.zeros(self.A.shape[1])
        lr = self.lr0 / (1 + self.episode / self.lr_tau)
        nd = np.linalg.norm(d)
        if nd > 1e-12:
            if self.policy == "ctx":
                self.Theta = np.clip(self.Theta + lr * (d / nd).reshape(self.Theta.shape), *THETA_BOUNDS)
                self.theta = self.Theta[:, 0].copy()
            else:
                self.theta = np.clip(self.theta + lr * d / nd, *THETA_BOUNDS)
        rec = dict(episode=self.episode, theta=self.theta.copy(), J=J, Jn=J / GOAL_SCALE,
                   score=float(self.omega @ (J / GOAL_SCALE)), A=self.A.copy(), step=float(lr * (nd > 1e-12)))
        self.history.append(rec)
        self.episode += 1
        return rec

    def _lean(self, W):
        """Per record: the exploration noise over the first half (global), or the noise
        times the situation features, flattened as Theta is (ctx)."""
        h = self.record_T // 2
        xi = W["xi"][:, :h, :len(PRIMITIVES)]
        if self.policy != "ctx":
            return xi.mean(axis=1)
        ph = W["phi"][:, :h, :]
        return np.einsum("nti,ntj->nij", xi, ph).reshape(len(xi), -1) / h

    def _phi(self, s):
        return np.hstack([s, np.ones((len(s), 1))])

    def _td_pass(self, log):
        """TD(0) over the episode's ticks, per goal, with a surprise gate (big errors --
        a first ambush -- take a larger, normalised step)."""
        X = self._phi(log["s"])
        R = log["r"] / GOAL_SCALE
        for t in range(len(X) - 1):
            x, x1 = X[t], X[t + 1]
            d = R[t] + self.gamma * (x1 @ self.Vw) - x @ self.Vw
            z = np.abs(d) / np.where(self._dsd > 0, self._dsd, np.inf)
            rate = np.where(z > self.surprise_z, 1.0, self.td_rate)
            self.Vw += np.outer(x, rate * d) / (1.0 + x @ x)
            self._dsd = np.sqrt(0.995 * self._dsd ** 2 + 0.005 * d ** 2)

    def _bootstrapped_advantage(self, log):
        """Per record (same windows as record_windows): discounted return over the window
        plus the discounted value at its end, minus the value at its start."""
        T, stride = self.record_T, self.record_T // 2
        X = self._phi(log["s"])
        V = X @ self.Vw
        R = log["r"] / GOAL_SCALE
        disc = self.gamma ** np.arange(T)
        out = []
        for i in range(0, len(R) - T + 1, stride):
            end = min(i + T, len(V) - 1)
            out.append(disc @ R[i:i + T] + self.gamma ** T * V[end] - V[i])
        return np.array(out)

    def run(self, n_episodes: int) -> list:
        for _ in range(n_episodes):
            self.step_episode()
        return self.history

    # -- movement rules
    def direction(self) -> np.ndarray:
        A, w = self.A, self.omega
        if self.rule == "pg":
            return A.T @ w
        if self.rule == "ng":
            M = A @ A.T
            lam = self.ng_damping * np.trace(M) / len(M) + 1e-12
            d = A.T @ np.linalg.solve(M + lam * np.eye(len(M)), w)
            g = A.T @ w
            return d if d @ g > 0 else g                       # never step against the estimated ascent
        if self.rule == "pareto":
            g = np.array([w[i] * A[i] for i in range(len(w)) if w[i] > 0])
            lam = np.full(len(g), 1.0 / len(g))
            for it in range(100):                              # Frank-Wolfe: min-norm point of hull
                v = lam @ g
                j = int(np.argmin(g @ v))
                step = 2.0 / (it + 3)
                lam = (1 - step) * lam
                lam[j] += step
            d = lam @ g
            if np.linalg.norm(d) < 0.05 * np.linalg.norm(g, axis=1).max():
                d = 0.2 * (A.T @ w)                            # Pareto-stationary: trade off slowly
            return d
        raise ValueError(self.rule)

    # -- evaluation and interrogation
    def evaluate(self, n_episodes: int = 16, noise: float = 0.2) -> np.ndarray:
        """Goal returns of the current theta, with the frontier panel's noise level."""
        ev = Commons(self.dials, seed=self.seed + 10_000, noise_override=noise)
        if self.policy == "ctx":
            return ev.run_ctx_policies(self.Theta[None], n_episodes)[0]
        return ev.run_policies(self.theta[None, :], n_episodes)[0]

    def report(self) -> dict:
        A = self.A
        M = A @ A.T
        ev, V = np.linalg.eigh(M)
        norms = np.linalg.norm(A, axis=1)
        planes = {}
        for i in range(len(GOALS)):
            for j in range(i + 1, len(GOALS)):
                dot = float(A[i] @ A[j])
                area = float(np.sqrt(max(norms[i] ** 2 * norms[j] ** 2 - dot ** 2, 0.0)))
                cos = dot / (norms[i] * norms[j] + 1e-12)
                planes[f"{GOALS[i]}^{GOALS[j]}"] = dict(cos=cos, area=area)
        modes = [dict(eigenvalue=float(ev[k]), goals=dict(zip(GOALS, np.round(V[:, k], 3))))
                 for k in np.argsort(-ev)]
        return dict(theta=dict(zip(PRIMITIVES, np.round(self.theta, 3))), M=M, modes=modes, planes=planes,
                    gradient_norms=dict(zip(GOALS, np.round(norms, 3))))


def connection(history: list, every: int = 20) -> list:
    """How the local goal metric turns as the agent moves: the change of the normalised
    M between successive stages (Frobenius), and the theta distance moved."""
    out = []
    for k in range(every, len(history), every):
        A0, A1 = history[k - every]["A"], history[k]["A"]
        M0, M1 = A0 @ A0.T, A1 @ A1.T
        n0, n1 = np.linalg.norm(M0), np.linalg.norm(M1)
        dM = np.linalg.norm(M1 / (n1 + 1e-12) - M0 / (n0 + 1e-12))
        out.append(dict(episode=k, dM=float(dM),
                        dtheta=float(np.linalg.norm(history[k]["theta"] - history[k - every]["theta"]))))
    return out
