"""Affect space and pain-aversion learning."""

import numpy as np
from scipy.linalg import expm

from cliffnpc import algebra as ga
from cliffnpc.affect import (AffectState, affect_vector, axis_eligibilities,
                             eligibility_increment)
from cliffnpc.cognition import metric_delta_generator
from cliffnpc.config import AffectConfig


def test_affect_morphism_signs():
    cfg = AffectConfig()
    a = affect_vector(np.array([0.5, 0.4, 0.0, 0.0]), 0.2, 0.0, cfg)
    assert np.all(a > 0)                                   # hungry, lonely, injured: all pain
    a = affect_vector(np.array([-0.5, -0.4, 0.0, 0.0]), 0.0, 0.3, cfg)
    assert np.all(a < 0)                                   # sated, in company, triumphant
    anti = AffectConfig(social_sign=-1.0)
    assert affect_vector(np.array([0, -0.4, 0, 0.0]), 0, 0, anti)[1] > 0   # crowding hurts


def test_eligibility_is_metric_derivative_of_action_growth():
    """E = d/dg [ s . dC/dt ] for dC/dt = -beta g C (checked by finite differences)."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        beta = rng.normal(size=(4, 4))
        beta = beta - beta.T
        C = rng.normal(size=4)
        g = ga.metric_from_frame(expm(0.3 * rng.normal(size=(4, 4))))
        s = np.tanh(C / 0.2)
        f = lambda gg: s @ (-beta @ gg @ C)  # noqa: E731
        E = eligibility_increment(beta, C, 0.2)
        dG = rng.normal(size=(4, 4))
        dG = dG + dG.T
        eps = 1e-6
        fd = (f(g + eps * dG) - f(g - eps * dG)) / (2 * eps)
        assert np.isclose(fd, np.sum(E * dG), rtol=1e-6, atol=1e-9)


def test_axis_eligibilities_sum_to_total():
    rng = np.random.default_rng(2)
    beta = rng.normal(size=(4, 4))
    beta = beta - beta.T
    C = rng.normal(size=4)
    assert np.allclose(axis_eligibilities(beta, C, 0.2).sum(axis=0), eligibility_increment(beta, C, 0.2))


def test_metric_delta_generator_first_order():
    rng = np.random.default_rng(1)
    L = expm(0.2 * rng.normal(size=(4, 4)))
    g = ga.metric_from_frame(L)
    dG = rng.normal(size=(4, 4))
    dG = 1e-6 * (dG + dG.T)
    g2 = ga.metric_from_frame(expm(metric_delta_generator(L, dG)) @ L)
    assert np.allclose(g2 - g, dG, atol=1e-10)
    A_gauge, _ = ga.split_eta_generator(metric_delta_generator(L, dG))
    assert np.allclose(A_gauge, 0, atol=1e-14)


def test_pain_dampens_and_triumph_amplifies_the_active_tendency():
    """After pain the same forcing grows the active drive less; after triumph more."""
    cfg = AffectConfig(learn_rate=0.5, max_step=1.0)
    beta = ga.bivector_from_planes({(3, 2): 0.5})      # threat channel M(e4, e3)
    C = np.array([0.3, 0.4, -0.3, 0.2])               # hungry, lonely, homey, fighting

    def fight_rate(L):
        g = ga.metric_from_frame(L)
        return (-beta @ g @ C)[3]

    out = {}
    for sign in (+1, -1):                              # +1 injury pain, -1 triumph
        st = AffectState(cfg)
        st.step(C, beta, 0.2)
        st.trace = axis_eligibilities(beta, C, cfg.action_scale)
        st.blame = np.array([0.0, 0.0, 0.0, sign * 1.0])   # injury is blamed on e4
        L = st.learn(np.eye(4))
        assert ga.signature(ga.metric_from_frame(L)) == (3, 1, 0)
        out[sign] = fight_rate(L)
    base = fight_rate(np.eye(4))
    assert out[+1] < base < out[-1]


def test_injury_alone_teaches_flight_when_lonely_and_fight_in_company():
    """Pain while fighting alone and triumph while fighting in company both build
    the same coupling: afterwards loneliness pushes toward flight and company
    toward fight, under the same threat forcing."""
    cfg = AffectConfig(learn_rate=0.3, max_step=1.0)
    beta = ga.bivector_from_planes({(3, 2): 0.5, (0, 3): 0.1})   # threat channels
    alone = np.array([0.3, 0.6, -0.4, 0.3])       # lonely
    group = np.array([0.3, -0.4, -0.4, 0.3])      # in company
    L = np.eye(4)
    st = AffectState(cfg)
    for C, injury_sign in ((alone, +1.0), (group, -1.0)) * 5:
        st.trace = axis_eligibilities(beta, C, cfg.action_scale)
        st.blame = np.array([0.0, 0.0, 0.0, injury_sign])
        L = st.learn(L)
    g = ga.metric_from_frame(L)

    def fight_rate(C, gg):
        return (-beta @ gg @ C)[3]

    before = fight_rate(alone, ga.ETA) - fight_rate(group, ga.ETA)
    after = fight_rate(alone, g) - fight_rate(group, g)
    assert after < before            # relative to company, loneliness now favours flight
    assert ga.signature(g) == (3, 1, 0)


def test_metric_ok_rejects_swapped_causal_character():
    from cliffnpc.affect import metric_ok
    cfg = AffectConfig()
    assert metric_ok(ga.ETA, cfg)
    g = ga.ETA.copy()
    g[2, 2], g[3, 3] = -1.0, 1.0           # explore timelike, threat spacelike: still (3,1)
    assert ga.signature(g) == (3, 1, 0)
    assert not metric_ok(g, cfg)
