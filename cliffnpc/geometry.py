"""Estimators of the geometry induced on goal and record space (spec-v2 section 3).

All take records cut from a Commons log (commons.record_windows) or the per-tick
log itself.  Only invariants should be read off: cosines, signatures, plane
types, eigenstructure -- the raw numbers depend on how goals and records are scaled.

    action_gradients(W)    A[i, k] = d G_i / d (lean on primitive k): the effect of
                           the exploration noise on each goal's progress over the
                           record, with the situation and the episode weights
                           regressed out
    goal_metric(A)         M = A A^T and its cosines: the circumstance-induced inner
                           product of goals (social . food > 0 when following the
                           skilled forager feeds you)
    outcome_moment(W, w)   S = E[w (x - xbar)(x - xbar)^T] over record features x,
                           w a scalarised standardised outcome: indefinite in general
    return_curvature(log)  per goal, the Hessian of the episode return in the
                           episode weights theta (quadratic regression)
    signature(Q)           (p, q, z) of a symmetric form, with a relative threshold
    bootstrap(fn, groups)  resample episodes for confidence intervals
"""

from __future__ import annotations

import numpy as np

from .commons import GOALS, PRIMITIVES


def _std(X, axis=0):
    s = X.std(axis=axis, keepdims=True)
    return np.where(s > 1e-12, s, 1.0)


def action_gradients(W: dict, lean_frac: float = 0.5) -> np.ndarray:
    """A (k x 6).  Lean = the mean exploration noise over the first part of the
    record; regress the record's (standardised) vector outcome on [lean, senses at
    the start, episode weights, 1] and keep the lean coefficients."""
    T = W["xi"].shape[1]
    n_lean = max(1, int(T * lean_frac))
    lean = W["xi"][:, :n_lean, :len(PRIMITIVES)].mean(axis=1)
    lean = lean / _std(lean)
    s0 = W["s"][:, 0, :]
    th = W["theta"][:, 0, :]
    X = np.hstack([lean, s0, th, np.ones((len(lean), 1))])
    G = W["G"]
    Gz = (G - G.mean(0)) / _std(G)
    coef, *_ = np.linalg.lstsq(X, Gz, rcond=None)
    return coef[:len(PRIMITIVES)].T                  # goals x primitives


def goal_metric(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """M = A A^T (goals x goals) and its cosine matrix."""
    M = A @ A.T
    d = np.sqrt(np.clip(np.diag(M), 1e-18, None))
    return M, M / np.outer(d, d)


def record_features(W: dict) -> np.ndarray:
    """Per record: mean control, mean primitive weights (theta + noise), mean senses."""
    T = W["xi"].shape[1]
    lean = W["theta"][:, 0, :] + W["xi"][:, :, :len(PRIMITIVES)].mean(axis=1)
    return np.hstack([W["u"].mean(axis=1), lean, W["s"].mean(axis=1)])


def outcome_weight(W: dict, omega=(1.0, 1.0, 0.5, 0.0)) -> np.ndarray:
    """Scalarised standardised outcome; omega[3] is the agent-specific value of knowledge."""
    G = W["G"]
    Gz = (G - G.mean(0)) / _std(G)
    w = Gz @ np.asarray(omega, float)
    return (w - w.mean()) / (w.std() + 1e-12)


def outcome_moment(W: dict, w: np.ndarray) -> np.ndarray:
    """S = E[w (x - xbar)(x - xbar)^T], record features standardised (dead ones dropped)."""
    X = record_features(W)
    keep = X.std(0) > 1e-9
    X = X[:, keep]
    Xz = (X - X.mean(0)) / _std(X)
    return (Xz * w[:, None]).T @ Xz / len(w)


def return_curvature(log: dict, goal: int | None = None, omega=(1.0, 1.0, 0.5, 0.0)) -> np.ndarray:
    """Hessian (6 x 6) of the episode return in theta, by quadratic regression over
    episodes.  goal=None: the scalarised standardised return."""
    eps = np.unique(log["episode"])
    th = np.array([log["theta"][log["episode"] == e][0] for e in eps])
    J = np.array([log["r"][log["episode"] == e].sum(0) for e in eps])
    Jz = (J - J.mean(0)) / _std(J)
    y = Jz @ np.asarray(omega, float) if goal is None else Jz[:, goal]
    n = th.shape[1]
    iu = np.triu_indices(n)
    quad = np.stack([th[:, i] * th[:, j] for i, j in zip(*iu)], axis=1)
    X = np.hstack([np.ones((len(th), 1)), th, quad])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    H = np.zeros((n, n))
    for c, (i, j) in zip(coef[1 + n:], zip(*iu)):
        if i == j:
            H[i, i] = 2 * c
        else:
            H[i, j] = H[j, i] = c
    return H


def signature(Q: np.ndarray, rel: float = 0.05) -> tuple[int, int, int]:
    """(p, q, z): eigenvalues above +rel*max|ev|, below -rel*max|ev|, and near zero."""
    ev = np.linalg.eigvalsh((Q + Q.T) / 2)
    tol = rel * np.abs(ev).max() if ev.size else 0.0
    return int((ev > tol).sum()), int((ev < -tol).sum()), int((np.abs(ev) <= tol).sum())


def bootstrap(fn, W: dict, n: int = 200, seed: int = 0) -> np.ndarray:
    """Resample whole episodes of a record set; returns fn on each resample, stacked."""
    rng = np.random.default_rng(seed)
    eps = np.unique(W["episode"])
    idx_by_ep = {e: np.nonzero(W["episode"] == e)[0] for e in eps}
    out = []
    for _ in range(n):
        pick = rng.choice(eps, size=len(eps), replace=True)
        idx = np.concatenate([idx_by_ep[e] for e in pick])
        out.append(fn({k: v[idx] for k, v in W.items()}))
    return np.array(out)


def merge(logs: list[dict]) -> dict:
    """Concatenate run logs from several seeds, keeping episode ids distinct."""
    out = {k: [] for k in logs[0]}
    offset = 0
    for L in logs:
        for k, v in L.items():
            out[k].append(v + offset if k == "episode" else v)
        offset += int(L["episode"].max()) + 1
    return {k: np.concatenate(v) for k, v in out.items()}


__all__ = ["GOALS", "PRIMITIVES", "action_gradients", "goal_metric", "record_features",
           "outcome_weight", "outcome_moment", "return_curvature", "signature", "bootstrap", "merge"]
