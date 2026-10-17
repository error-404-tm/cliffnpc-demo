"""Drive-space geometric algebra: metrics, frames, bivector generators, blades.

Conventions (fixed throughout the project)
------------------------------------------
* Drive basis e1..e4 = hunger, social, explore, threat (indices 0..3 in code).
* A drive-space vector is stored as its coefficient array C = (C^1..C^4) in the
  *semantic* basis, so ``C = sum_i C^i e_i``.
* The metric is g_ij = e_i . e_j, a symmetric 4x4 matrix of signature (3,1).
* The orthonormal frame is a 4x4 matrix L with g = L^T eta L.  Column i of L
  is e_i written in an orthonormal basis f_a (f_a . f_b = eta_ab).  Vectors
  are mapped to the orthonormal frame by ``C_o = L @ C``.
* Bivector generator convention: the simple bivector a^b acts on vectors as
      M(a, b): v -> (v.a) b - (v.b) a,      i.e.  M = b a^T g - a b^T g.
  This is (minus) the commutator action v -> v . (a^b).  With it, the flow
  exp(t M(e1, e3)) rotates hungry (e1) -> explore (e3) -> sated (-e1) ->
  home (-e3), and M(a,b)^2 restricted to the plane equals (a^b)^2.
* A general bivector is stored as an antisymmetric 4x4 coefficient array
  beta, B = 1/2 sum_ij beta^{ij} e_i ^ e_j, so e_i^e_j has beta^{ij} = 1,
  beta^{ji} = -1.  Its generator is M = sum_{i<j} beta^{ij} M(e_i, e_j)
  = -beta g  (see ``generator_from_bivector``).

Everything here is pure NumPy/SciPy with no I/O.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.linalg import expm

ETA = np.diag([1.0, 1.0, 1.0, -1.0])     # default signature (3,1): threat timelike


def eta_of(signature) -> np.ndarray:
    """Base metric for a drive signature, e.g. (1, 1, 1, -1) or (-1, 1, 1, -1)."""
    return np.diag(np.asarray(signature, dtype=float))


def _eta(eta):
    return ETA if eta is None else eta
DRIVES = ("hunger", "social", "explore", "threat")
N = 4
PLANES = list(combinations(range(N), 2))  # (0,1),(0,2),...,(2,3)
TRIPLES = list(combinations(range(N), 3))


# ---------------------------------------------------------------------------
# Metrics and frames
# ---------------------------------------------------------------------------

def metric_from_frame(L: np.ndarray, eta: np.ndarray | None = None) -> np.ndarray:
    """g = L^T eta L."""
    return L.T @ _eta(eta) @ L


def signature(g: np.ndarray, tol: float = 1e-10) -> tuple[int, int, int]:
    """(n_positive, n_negative, n_zero) eigenvalue counts of a symmetric matrix."""
    w = np.linalg.eigvalsh(0.5 * (g + g.T))
    return int((w > tol).sum()), int((w < -tol).sum()), int((abs(w) <= tol).sum())


def frame_from_metric(g: np.ndarray, eta: np.ndarray | None = None) -> np.ndarray:
    """Some L with L^T eta L = g, for g with the signature of eta.

    Built from the eigendecomposition g = Q diag(w) Q^T, with the negative
    eigenvalues placed in eta's timelike slots.  The result is one
    representative of the gauge orbit {Lambda L : Lambda in O(eta)}.
    """
    eta = _eta(eta)
    g = 0.5 * (g + g.T)
    w, Q = np.linalg.eigh(g)
    neg = [i for i in range(N) if w[i] < 0]
    pos = [i for i in range(N) if w[i] >= 0]
    t_slots = [i for i in range(N) if eta[i, i] < 0]
    s_slots = [i for i in range(N) if eta[i, i] > 0]
    if len(neg) != len(t_slots):
        raise ValueError(f"metric signature does not match eta; eigenvalues {w}")
    order = [0] * N
    for slot, k in zip(s_slots + t_slots, pos + neg):
        order[slot] = k
    w, Q = w[order], Q[:, order]
    return np.diag(np.sqrt(np.abs(w))) @ Q.T


def reciprocal_frame(g: np.ndarray) -> np.ndarray:
    """g^{-1}: row i gives e^i in the semantic basis (e^i . e_j = delta^i_j)."""
    return np.linalg.inv(g)


def dot(u: np.ndarray, v: np.ndarray, g: np.ndarray) -> float:
    return float(u @ g @ v)


def interval(C: np.ndarray, g: np.ndarray) -> float:
    """s = C^T g C.  s<0 timelike (reactive), s>0 spacelike (deliberative)."""
    return float(C @ g @ C)


def normalized_interval(C: np.ndarray, g: np.ndarray) -> float:
    """s / |C|^2 (Euclidean norm of the coefficients): the scale-free position
    relative to the light cone, roughly in [-1, 1].  Used for regime decisions so
    that a tiny resting state does not read as "null" merely for being small."""
    n2 = float(C @ C)
    return interval(C, g) / n2 if n2 > 1e-12 else 0.0


def regime(s: float, null_tol: float) -> str:
    if s < -null_tol:
        return "reactive"
    if s > null_tol:
        return "deliberative"
    return "null"


# ---------------------------------------------------------------------------
# Bivectors and generators
# ---------------------------------------------------------------------------

def generator(a: np.ndarray, b: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Matrix of v -> (v.a) b - (v.b) a, i.e. M = b a^T g - a b^T g."""
    return np.outer(b, a) @ g - np.outer(a, b) @ g


def basis_generator(i: int, j: int, g: np.ndarray) -> np.ndarray:
    e = np.eye(N)
    return generator(e[i], e[j], g)


def bivector_from_planes(coeffs: dict[tuple[int, int], float]) -> np.ndarray:
    """Antisymmetric coefficient array beta from {(i,j): value} for e_i^e_j."""
    beta = np.zeros((N, N))
    for (i, j), v in coeffs.items():
        beta[i, j] += v
        beta[j, i] -= v
    return beta


def wedge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """beta for a^b: beta^{ij} = a^i b^j - a^j b^i."""
    return np.outer(a, b) - np.outer(b, a)


def generator_from_bivector(beta: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Generator of B = sum_{i<j} beta^{ij} e_i^e_j.

    Linear extension of ``generator``: for beta = a b^T - b a^T this reduces
    to b a^T g - a b^T g = -beta g.  Hence M = -beta @ g in general.
    """
    return -beta @ g


def bivector_square_simple(a: np.ndarray, b: np.ndarray, g: np.ndarray) -> float:
    """(a^b)^2 = (a.b)^2 - (a.a)(b.b)."""
    return dot(a, b, g) ** 2 - dot(a, a, g) * dot(b, b, g)


def plane_type(a: np.ndarray, b: np.ndarray, g: np.ndarray, tol: float = 1e-9) -> str:
    sq = bivector_square_simple(a, b, g)
    if sq < -tol:
        return "elliptic"
    if sq > tol:
        return "hyperbolic"
    return "parabolic"


def basis_plane_type(i: int, j: int, g: np.ndarray, tol: float = 1e-9) -> str:
    e = np.eye(N)
    return plane_type(e[i], e[j], g, tol)


def is_in_so(M: np.ndarray, g: np.ndarray, tol: float = 1e-10) -> bool:
    """M^T g + g M = 0 (M is g-antisymmetric, i.e. preserves the metric)."""
    return bool(np.allclose(M.T @ g + g @ M, 0.0, atol=tol))


def split_eta_generator(A: np.ndarray, eta: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Split a frame generator into (so(eta) part, eta-self-adjoint part).

    A_gauge = (A - eta A^T eta)/2 satisfies A^T eta + eta A = 0 and leaves
    g = L^T eta L unchanged under L <- exp(A) L.  A_sym = (A + eta A^T eta)/2
    satisfies eta A_sym = (eta A_sym)^T and is the part that deforms g.
    """
    eta = _eta(eta)
    swapped = eta @ A.T @ eta
    return 0.5 * (A - swapped), 0.5 * (A + swapped)


def clamp_boost(M_o: np.ndarray, max_rapidity: float,
                eta: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Cap the boost part of an so(eta) matrix (already scaled by the step h).

    In the orthonormal frame the boost part of an so(eta) matrix is the set of
    entries that mix a spacelike with a timelike axis (symmetric there); the
    rotation part mixes axes of the same type.  For (3,1) this is the familiar
    [[R, b], [b^T, 0]].  phi = norm of the mixed entries (each pair once).
    Returns the clamped matrix and the rapidity before clamping.
    """
    s = np.diag(_eta(eta))
    mixed = np.outer(s, s) < 0
    phi = float(np.sqrt(0.5 * np.sum(M_o[mixed] ** 2)))
    if phi <= max_rapidity or phi == 0.0:
        return M_o, phi
    out = M_o.copy()
    out[mixed] *= max_rapidity / phi
    return out, phi


def rotor_step(C: np.ndarray, M: np.ndarray, h: float, L: np.ndarray,
               max_rapidity: float = np.inf,
               eta: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """C <- exp(h M) C, with the boost rapidity of h M capped.

    The clamp is applied in the orthonormal frame (M_o = L M L^{-1} is in
    so(eta)), so the clamped step is still an exact isometry of g.
    """
    Linv = np.linalg.inv(L)
    M_o = L @ (h * M) @ Linv
    M_o, phi = clamp_boost(M_o, max_rapidity, eta)
    return Linv @ expm(M_o) @ L @ C, phi


# ---------------------------------------------------------------------------
# Trivectors (frustration) and Gram volumes (sensitivity)
# ---------------------------------------------------------------------------

def vector_wedge_bivector(C: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Trivector components T^{ijk} (i<j<k) of C ^ B, in the given basis.

    For B = 1/2 beta^{jk} e_j^e_k,
    C^B = sum_{i<j<k} (C^i beta^{jk} + C^j beta^{ki} + C^k beta^{ij}) e_ijk.
    """
    return np.array([C[i] * beta[j, k] + C[j] * beta[k, i] + C[k] * beta[i, j]
                     for i, j, k in TRIPLES])


def trivector_norms(C: np.ndarray, beta: np.ndarray, L: np.ndarray,
                    eta: np.ndarray | None = None) -> tuple[float, float]:
    """(Euclidean norm, signed square) of C^B, computed in the orthonormal frame.

    The Euclidean norm of the orthonormal-frame components is the stress meter
    (positive-definite but depends on the gauge of L).  The signed square
    T . T~ = sum T_abc^2 eta_a eta_b eta_c is Lorentz invariant but indefinite.
    """
    C_o = L @ C
    beta_o = L @ beta @ L.T
    T = vector_wedge_bivector(C_o, beta_o)
    e = _eta(eta)
    signs = np.array([e[i, i] * e[j, j] * e[k, k] for i, j, k in TRIPLES])
    return float(np.linalg.norm(T)), float((signs * T * T).sum())


def gram_minor(g: np.ndarray, i: int) -> float:
    """Squared g-volume of the three drives other than i: det of g without row/col i."""
    keep = [k for k in range(N) if k != i]
    return float(np.linalg.det(g[np.ix_(keep, keep)]))


def sensitivities(g: np.ndarray, w_max: float = 5.0) -> np.ndarray:
    """Per-drive learning-rate weights |g^{ii}| (cofactor identity), capped."""
    return np.minimum(np.abs(np.diag(np.linalg.inv(g))), w_max)
