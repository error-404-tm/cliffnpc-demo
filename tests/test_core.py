"""Phase 1 required tests (spec §6, "Tests")."""

import numpy as np
import pytest
from scipy.linalg import expm

from cliffnpc import algebra as ga
from cliffnpc.cognition import (NPC, apply_frame_update, felt_covector, salience_generator,
                                toward_metric_generator)
from cliffnpc.config import CognitionConfig, Config
from cliffnpc.sim import Simulation
from cliffnpc.world import HauntedDisc

ETA = ga.ETA


def random_frame(rng, scale=0.4):
    """Random L near the identity (so g has signature (3,1) and is non-diagonal)."""
    return expm(scale * rng.normal(size=(4, 4)))


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


# ---------------------------------------------------------------------------

def test_random_frames_give_signature_31(rng):
    for _ in range(50):
        g = ga.metric_from_frame(random_frame(rng))
        assert ga.signature(g) == (3, 1, 0)


def test_frame_from_metric_roundtrip(rng):
    for _ in range(20):
        g = ga.metric_from_frame(random_frame(rng))
        L = ga.frame_from_metric(g)
        assert np.allclose(ga.metric_from_frame(L), g, atol=1e-10)


def test_generator_in_so_g(rng):
    """M^T g + g M = 0 for arbitrary g of signature (3,1)."""
    for _ in range(50):
        g = ga.metric_from_frame(random_frame(rng))
        a, b = rng.normal(size=4), rng.normal(size=4)
        assert ga.is_in_so(ga.generator(a, b, g), g)
        beta = rng.normal(size=(4, 4))
        beta = beta - beta.T
        assert ga.is_in_so(ga.generator_from_bivector(beta, g), g)


def test_generator_from_bivector_matches_simple(rng):
    g = ga.metric_from_frame(random_frame(rng))
    a, b = rng.normal(size=4), rng.normal(size=4)
    assert np.allclose(ga.generator_from_bivector(ga.wedge(a, b), g), ga.generator(a, b, g))


def test_generator_square_is_bivector_square(rng):
    """M(a,b)^2 restricted to the plane = (a^b)^2 (fixes the sign convention)."""
    g = ga.metric_from_frame(random_frame(rng))
    a, b = rng.normal(size=4), rng.normal(size=4)
    M = ga.generator(a, b, g)
    sq = ga.bivector_square_simple(a, b, g)
    assert np.allclose(M @ M @ a, sq * a)
    assert np.allclose(M @ M @ b, sq * b)


def test_rotation_direction_convention():
    """exp(t M(e1,e3)) takes hungry -> explore at t = pi/2 (foraging order)."""
    M = ga.basis_generator(0, 2, ETA)
    assert np.allclose(expm(np.pi / 2 * M) @ np.eye(4)[0], np.eye(4)[2])


def test_rotor_step_preserves_interval(rng):
    for _ in range(30):
        L = random_frame(rng)
        g = ga.metric_from_frame(L)
        beta = rng.normal(size=(4, 4))
        beta = beta - beta.T
        M = ga.generator_from_bivector(beta, g)
        C = rng.normal(size=4)
        C2, _ = ga.rotor_step(C, M, 0.1, L)
        assert np.isclose(ga.interval(C2, g), ga.interval(C, g), rtol=1e-9, atol=1e-10)
        # also with an active rapidity clamp: still an exact isometry
        C3, phi = ga.rotor_step(C, M * 20, 0.1, L, max_rapidity=0.05)
        assert phi > 0.05
        assert np.isclose(ga.interval(C3, g), ga.interval(C, g), rtol=1e-9, atol=1e-10)


def test_plane_classification_eta():
    """§2.2: planes among e1..e3 elliptic, planes with e4 hyperbolic."""
    for i, j in ga.PLANES:
        expected = "hyperbolic" if j == 3 else "elliptic"
        assert ga.basis_plane_type(i, j, ETA) == expected


@pytest.mark.parametrize("g23,expected", [(0.0, "elliptic"), (0.5, "elliptic"),
                                          (-0.99, "elliptic"), (1.0, "parabolic"),
                                          (1.01, "hyperbolic"), (-1.5, "hyperbolic")])
def test_plane_classification_g23_threshold(g23, expected):
    g = ETA.copy()
    g[1, 2] = g[2, 1] = g23
    assert ga.basis_plane_type(1, 2, g) == expected


def test_metric_update_preserves_signature(rng):
    """Learning + forgetting updates keep signature (3,1) and det L away from 0."""
    L = np.eye(4)
    L0 = L.copy()
    g0 = ga.metric_from_frame(L0)
    for _ in range(500):
        u = rng.normal(size=4) * 3
        L = apply_frame_update(L, salience_generator(u, L), 0.05, 0.2)
        L = apply_frame_update(L, toward_metric_generator(L, g0), 0.01, 0.2)
        assert ga.signature(ga.metric_from_frame(L)) == (3, 1, 0)
        assert abs(np.linalg.det(L)) > 1e-6


def test_gauge_updates_leave_metric_unchanged(rng):
    """so(eta)-generated frame updates are pure gauge."""
    L = random_frame(rng)
    g = ga.metric_from_frame(L)
    for _ in range(20):
        A = rng.normal(size=(4, 4))
        A_gauge, A_sym = ga.split_eta_generator(A)
        assert np.allclose(A_gauge + A_sym, A)
        assert np.allclose(A_gauge.T @ ETA + ETA @ A_gauge, 0)
        assert np.allclose(ETA @ A_sym, (ETA @ A_sym).T)
        L2 = expm(A_gauge) @ L
        assert np.allclose(ga.metric_from_frame(L2), g, atol=1e-9)
        # and the symmetric part does deform it
        assert not np.allclose(ga.metric_from_frame(expm(0.1 * A_sym) @ L), g, atol=1e-6)


def test_salience_and_decay_generators_are_gauge_free(rng):
    L = random_frame(rng)
    for A in (salience_generator(rng.normal(size=4), L),
              toward_metric_generator(L, ga.metric_from_frame(random_frame(rng)))):
        A_gauge, _ = ga.split_eta_generator(A)
        assert np.allclose(A_gauge, 0, atol=1e-12)


def test_salience_first_order_effect(rng):
    """g <- g + eps sigma c c^T to first order; |u.u| grows for u = g^{-1} c."""
    L = random_frame(rng, 0.2)
    g = ga.metric_from_frame(L)
    for _ in range(10):
        c = rng.normal(size=4)
        eps = 1e-5
        g2 = ga.metric_from_frame(expm(eps * salience_generator(c, L)) @ L)
        u = np.linalg.solve(g, c)
        sigma = np.sign(c @ u)
        assert np.allclose((g2 - g) / eps, sigma * np.outer(c, c), rtol=1e-3, atol=1e-3)
        assert abs(u @ g2 @ u) > abs(u @ g @ u)


def test_learning_with_forgetting_stays_bounded(rng):
    """Repeated surprise along one direction saturates instead of blowing up."""
    L = np.eye(4)
    g0 = ga.metric_from_frame(L)
    c = np.array([0.0, 0.0, 0.0, 1.0])
    for _ in range(3000):
        L = apply_frame_update(L, salience_generator(c, L), 0.02, 0.2)
        L = apply_frame_update(L, toward_metric_generator(L, g0), 0.01, 0.2)
    g = ga.metric_from_frame(L)
    # equilibrium g44 ~ -1 - eps/lambda = -3
    assert np.isclose(g[3, 3], -3.0, rtol=0.05)
    assert np.linalg.cond(L) < 10


def test_decay_moves_toward_prior(rng):
    L0 = np.eye(4)
    L = random_frame(rng, 0.3)
    g0 = ga.metric_from_frame(L0)
    d0 = np.linalg.norm(ga.metric_from_frame(L) - g0)
    for _ in range(200):
        L = apply_frame_update(L, toward_metric_generator(L, g0), 0.05, 1.0)
    assert np.linalg.norm(ga.metric_from_frame(L) - g0) < 0.05 * d0


def test_swapping_updates_changes_L(rng):
    """Non-commutativity: trauma then comfort != comfort then trauma."""
    L = np.eye(4)
    r_trauma = np.array([0.0, 0.3, 0.0, 2.0])
    r_comfort = np.array([1.5, 0.8, 0.0, 0.0])

    def run(seq):
        Lx = L.copy()
        for r in seq:
            c = felt_covector(r, ga.metric_from_frame(Lx))
            Lx = apply_frame_update(Lx, salience_generator(c, Lx), 0.3, 2.0)
        return Lx

    L_tc, L_ct = run([r_trauma, r_comfort]), run([r_comfort, r_trauma])
    assert np.linalg.norm(L_tc - L_ct) > 1e-3
    g_tc, g_ct = ga.metric_from_frame(L_tc), ga.metric_from_frame(L_ct)
    # Order dependence is a commutator effect, O(eps_a eps_b): for two strong
    # events it is a few percent of the total change.
    assert np.linalg.norm(g_tc - g_ct) > 0.01 * np.linalg.norm(g_tc - ETA)


def test_cofactor_identity(rng):
    """Gram minor of the drives other than i = det(g) (g^{-1})_ii."""
    for _ in range(20):
        g = ga.metric_from_frame(random_frame(rng))
        ginv = np.linalg.inv(g)
        for i in range(4):
            assert np.isclose(ga.gram_minor(g, i), np.linalg.det(g) * ginv[i, i])


def test_trivector_duality_signs():
    """I^2 = -1; e123 squares to -1, trivectors containing e4 square to +1."""
    # Use T.T~ = sum T^2 eta eta eta on unit blades (T~ = -T for grade 3,
    # so the square is -T.T~).
    for idx, (i, j, k) in enumerate(ga.TRIPLES):
        prod = ETA[i, i] * ETA[j, j] * ETA[k, k]
        square = -prod
        assert square == (-1 if 3 not in (i, j, k) else 1)
    # pseudoscalar: e1234^2 = (+1)(sign of reversal for grade 4 = +1)*prod(eta) = -1
    assert np.prod(np.diag(ETA)) * 1 == -1


def test_frustration_zero_when_state_in_plane():
    """C in the plane of B: C^B = 0, all of the push is realisable."""
    beta = ga.bivector_from_planes({(0, 2): 1.0})
    C = np.array([0.7, 0.0, -0.3, 0.0])
    n, _ = ga.trivector_norms(C, beta, np.eye(4))
    assert n < 1e-12
    C2 = np.array([0.0, 1.0, 0.0, 0.0])
    n2, _ = ga.trivector_norms(C2, beta, np.eye(4))
    assert np.isclose(n2, 1.0)


# ---------------------------------------------------------------------------
# Holonomy
# ---------------------------------------------------------------------------

def _loop(center, radius, n=400):
    th = np.linspace(0, 2 * np.pi, n + 1)
    return np.stack([center[0] + radius * np.cos(th), center[1] + radius * np.sin(th)], 1)


def _loop_integral(disc, pts):
    return sum(disc.segment_integral(p, q) for p, q in zip(pts[:-1], pts[1:]))


def test_holonomy_outside_loop_is_identity():
    disc = HauntedDisc(np.array([50.0, 50.0]), 8.0, 0.01)
    alpha = _loop_integral(disc, _loop((80.0, 50.0), 10.0))
    assert abs(alpha) < 1e-12
    M = ga.basis_generator(1, 2, ETA)
    assert np.allclose(expm(alpha * M), np.eye(4))
    # crossing the disc but not enclosing its centre's full flux still works
    # out: a loop through part of the disc picks up Phi * (overlap area)
    alpha2 = _loop_integral(disc, _loop((58.0, 50.0), 4.0, n=4000))
    # overlap: small circle r=4 centred 8 from the centre, i.e. on the rim;
    # the lens area of two circles (R=8, r=4, d=8):
    R, r, d = 8.0, 4.0, 8.0
    lens = (r**2 * np.arccos((d**2 + r**2 - R**2) / (2 * d * r))
            + R**2 * np.arccos((d**2 + R**2 - r**2) / (2 * d * R))
            - 0.5 * np.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)))
    assert np.isclose(alpha2, 0.01 * lens, rtol=1e-3)


def test_holonomy_enclosing_loop_picks_up_total_flux():
    disc = HauntedDisc(np.array([50.0, 50.0]), 8.0, 0.01)
    alpha = _loop_integral(disc, _loop((52.0, 49.0), 20.0))
    assert np.isclose(alpha, 0.01 * np.pi * 64, rtol=1e-9)
    alpha_in = _loop_integral(disc, _loop((50.0, 50.0), 3.0, n=2000))
    assert np.isclose(alpha_in, 0.01 * np.pi * 9, rtol=1e-4)


# ---------------------------------------------------------------------------
# Smoke test of the whole loop
# ---------------------------------------------------------------------------

def test_simulation_runs_and_stays_finite():
    sim = Simulation(Config())
    log = sim.run(300.0)
    for k in ("C", "s", "frustration", "g", "detL"):
        assert np.all(np.isfinite(log[k]))
    for g in log["g"][::50]:
        assert ga.signature(g) == (3, 1, 0)


def test_surprise_gating_skips_ticks_and_stays_finite():
    cfg = Config()
    cfg.cog.predictor = "cv"
    cfg.world.surprise_gate = 0.05
    sim = Simulation(cfg)
    log = sim.run(200.0)
    assert sim.cog_ticks < 0.9 * sim.cog_slots
    assert np.all(np.isfinite(log["C"]))
