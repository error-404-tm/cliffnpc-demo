"""Affects as trivectors: the grade-3 dual of the drives."""

import numpy as np
import pytest
from scipy.linalg import expm

from cliffnpc import algebra as ga
from cliffnpc.affect import (INCIDENCE, MOB_PLANE, THREAT_PLANES, TrivectorAffect, contract,
                             felt_matrix, induced_trivector_metric)
from cliffnpc.config import AffectConfig, TrivectorConfig

SIGS = {"Cl(4,0)": (1.0, 1.0, 1.0, 1.0), "Cl(3,1)": (1.0, 1.0, 1.0, -1.0),
        "Cl(2,2)": (-1.0, 1.0, 1.0, -1.0)}


def _g(sig, rng, scale=0.3):
    eta = ga.eta_of(sig)
    X = scale * rng.normal(size=(4, 4))
    _, Xs = ga.split_eta_generator(X, eta)
    return ga.metric_from_frame(expm(Xs), eta), eta


@pytest.mark.parametrize("name", SIGS)
def test_induced_metric_signs_follow_the_signature(name):
    eta = ga.eta_of(SIGS[name])
    G = induced_trivector_metric(eta)
    s = np.diag(eta)
    expected = [s[i] * s[j] * s[k] for i, j, k in ga.TRIPLES]
    assert np.allclose(G, np.diag(expected))
    assert np.allclose(felt_matrix(eta), np.eye(4))      # at g = eta affect is felt as is


@pytest.mark.parametrize("name", SIGS)
def test_induced_metric_is_the_cofactor_matrix(name):
    """3x3 minors of g = det(g) g^{-1}, up to the Hodge index map and signs."""
    rng = np.random.default_rng(3)
    g, _ = _g(SIGS[name], rng)
    G = induced_trivector_metric(g)
    cof = np.linalg.det(g) * np.linalg.inv(g)           # cof[l, m]: minors without row l / col m
    left_out = [({0, 1, 2, 3} - set(t)).pop() for t in ga.TRIPLES]
    for a in range(4):
        for b in range(4):
            assert np.isclose(abs(G[a, b]), abs(cof[left_out[a], left_out[b]]))


def test_contraction_matches_the_expansion_formula():
    rng = np.random.default_rng(0)
    g, _ = _g(SIGS["Cl(3,1)"], rng)
    a, b, c, C = rng.normal(size=(4, 4))
    T = ga.vector_wedge_bivector(a, ga.wedge(b, c))      # components of a^b^c
    dot = lambda x, y: x @ g @ y  # noqa: E731
    expected = dot(C, a) * ga.wedge(b, c) - dot(C, b) * ga.wedge(a, c) + dot(C, c) * ga.wedge(a, b)
    assert np.allclose(contract(C, T, g), expected)


def test_threat_events_are_felt_as_the_three_fears():
    """C ^ (threat planes) only has components containing fight/flight."""
    C = np.array([0.4, -0.3, 0.2, 0.1])
    T = ga.vector_wedge_bivector(C, THREAT_PLANES)
    names = [t for t in ga.TRIPLES]
    for comp, t in zip(T, names):
        if 3 not in t:
            assert abs(comp) < 1e-12                       # e123: no fear
    assert np.abs(T[1:]).sum() > 0


def test_injury_hurts_and_a_meal_rewards():
    """Against an otherwise identical agent: injury adds pain on the fear trivectors only,
    a meal removes pain (reward)."""
    tri = TrivectorConfig()
    eta = ga.eta_of(SIGS["Cl(3,1)"])
    C = np.array([0.5, 0.2, -0.3, 0.1])
    Z = np.zeros((4, 4))

    def felt(injury=0.0, eat=0.0):
        a = TrivectorAffect(tri, AffectConfig())
        a.observe_injury(injury, 0.0, 0.2)
        a.step(C, eta, Z, Z, 0.2, eat=eat)
        return a
    base, hurt, fed = felt(), felt(injury=0.2), felt(eat=1.0)
    d = hurt.F - base.F
    assert hurt.pain > base.pain and abs(d[0]) < 1e-12 and np.all(d[1:] >= 0) and d[1:].sum() > 0
    assert fed.pain < base.pain


def test_fear_is_blamed_on_fight_flight():
    assert np.allclose(INCIDENCE.T @ np.array([0.0, 1.0, 1.0, 1.0]), [2, 2, 2, 3])
    assert np.allclose(INCIDENCE.T @ np.array([1.0, 0.0, 0.0, 0.0]), [1, 1, 1, 0])


def test_affect_acts_back_through_a_generator():
    rng = np.random.default_rng(1)
    g, eta = _g(SIGS["Cl(2,2)"], rng)
    aff = TrivectorAffect(TrivectorConfig(), AffectConfig())
    aff.F = np.array([0.2, 1.0, 0.5, 0.8])
    C = rng.normal(size=4)
    beta = aff.behaviour_bivector(C, g)
    assert np.allclose(beta, -beta.T)
    assert ga.is_in_so(ga.generator_from_bivector(beta, g), g)   # an isometry of the drives


def test_mob_plane_is_hyperbolic_except_in_the_euclidean_signature():
    for name, sig in SIGS.items():
        kind = ga.basis_plane_type(1, 3, ga.eta_of(sig))
        assert kind == ("elliptic" if name == "Cl(4,0)" else "hyperbolic")
    assert MOB_PLANE[3, 1] == 1.0


@pytest.mark.parametrize("name", SIGS)
def test_trivector_arena_runs_for_every_signature(name):
    from cliffnpc.arena import Arena, life_config
    cfg = life_config(SIGS[name], seed=0)
    cfg.n_npcs = 3
    ar = Arena(cfg)
    log = ar.run(120.0)
    assert np.all(np.isfinite(log["affect"][log["alive"]]))
    for a in ar.agents:
        assert isinstance(a.affect, TrivectorAffect)
        assert ga.signature(a.npc.g)[:2] == (sum(x > 0 for x in SIGS[name]), sum(x < 0 for x in SIGS[name]))


def test_felt_affect_is_scale_invariant_and_affect_saturates():
    rng = np.random.default_rng(4)
    g, _ = _g(SIGS["Cl(3,1)"], rng)
    assert np.allclose(felt_matrix(3.0 * g), felt_matrix(g))
    aff = TrivectorAffect(TrivectorConfig(), AffectConfig())
    eta = ga.eta_of(SIGS["Cl(3,1)"])
    C = np.array([0.5, 0.5, 0.5, 0.5])
    for _ in range(2000):
        aff.observe_injury(1.0, 0.0, 0.2)                 # relentless injury
        aff.step(C, eta, np.zeros((4, 4)), np.zeros((4, 4)), 0.2)
    assert np.abs(aff.P).max() < 15.0
