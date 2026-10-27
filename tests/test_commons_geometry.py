"""Commons environment and induced-geometry estimators."""

import numpy as np

from cliffnpc.commons import (GOALS, PRIMITIVES, Commons, Env, environments, record_dtype,
                              record_windows, tick_dtype)
from cliffnpc.geometry import (action_gradients, goal_metric, outcome_moment, return_curvature,
                               signature)


def test_record_layout_is_powers_of_two():
    assert tick_dtype().itemsize == 256
    for T in (16, 32, 64):
        assert record_dtype(T).itemsize == 256 * T


def test_sweep_is_centre_plus_one_factor_changes():
    envs = environments()
    assert len(envs) == 7 and len({e.name() for e in envs}) == 7


def test_commons_logs_controls_senses_and_goal_progress():
    log = Commons(Env(threat=0.0), seed=0).run(2)
    n = len(log["r"])
    assert n == 600 and log["s"].shape == (n, 24) and log["r"].shape == (n, len(GOALS))
    assert log["xi"].shape == (n, len(PRIMITIVES)) and np.all(log["r"][:, 1] <= 1e-12)  # no threat: no injury
    W = record_windows(log, 32)
    assert W["G"].shape[1] == len(GOALS) and W["xi"].shape[1] == 32


def test_knowledge_comes_from_sight_only_when_cues_are_valid():
    valid = Commons(Env(threat=0.0, cue_validity=1), seed=0).run(1)
    random_cues = Commons(Env(threat=0.0, cue_validity=0), seed=0).run(1)
    assert valid["r"][0, 3] >= 1.0                 # patches in view are resolved at once
    assert random_cues["r"][0, 3] == 0.0            # only probing at a patch resolves them


def test_policy_families_set_theta_ranges():
    from cliffnpc.commons import POLICIES
    c = Commons(Env(), seed=0, policy="cautious")
    c.new_episode()
    m = np.asarray(POLICIES["cautious"]["mean"])
    assert np.all(np.abs(c.theta - m) <= POLICIES["cautious"]["spread"] + 1e-12)


def test_action_gradients_recover_a_planted_effect():
    rng = np.random.default_rng(0)
    n, T = 2000, 16
    xi = rng.normal(size=(n, T, 6))
    lean = xi[:, :8].mean(axis=1)
    A_true = np.zeros((4, 6))
    A_true[0, 0], A_true[2, 1], A_true[0, 1] = 1.0, 1.0, 0.5      # food <- food, forager; social <- forager
    G = lean @ A_true.T + 0.1 * rng.normal(size=(n, 4))
    W = dict(xi=xi, s=rng.normal(size=(n, T, 24)) * 0.0, theta=np.zeros((n, T, 6)), G=G,
             u=np.zeros((n, T, 3)), r=np.zeros((n, T, 4)), episode=np.arange(n))
    A = action_gradients(W)
    _, cos = goal_metric(A)
    assert cos[0, 2] > 0.3                                         # food and social share the forager lean
    assert np.argmax(np.abs(A[0])) == 0


def test_signature_and_moment_are_indefinite_for_contrasting_outcomes():
    assert signature(np.diag([2.0, 1.0, -1.5, 0.0])) == (2, 1, 1)
    rng = np.random.default_rng(1)
    n, T = 1000, 16
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    W = dict(u=np.zeros((n, T, 3)), xi=np.zeros((n, T, 6)), theta=np.zeros((n, T, 6)),
             s=np.repeat(np.stack([x1, x2] + [np.zeros(n)] * 22, 1)[:, None], T, 1),
             G=np.zeros((n, 4)), r=np.zeros((n, T, 4)), episode=np.arange(n))
    w = x1 ** 2 - x2 ** 2                                          # extremes of x1 pay, of x2 cost
    S = outcome_moment(W, w / w.std())
    p, q, _ = signature(S)
    assert p >= 1 and q >= 1


def test_return_curvature_finds_a_planted_saddle():
    rng = np.random.default_rng(2)
    E = 400
    th = rng.uniform(-1, 1, (E, 6))
    J = th[:, 0] ** 2 - th[:, 1] ** 2
    log = dict(episode=np.repeat(np.arange(E), 2), theta=np.repeat(th, 2, axis=0),
               r=np.repeat(np.stack([J / 2] + [rng.normal(size=E) * 0.01] * 3, 1), 2, axis=0))
    H = return_curvature(log, goal=0)
    assert H[0, 0] > 0 > H[1, 1]


def test_frontier_tools_find_trade_offs_and_synergies():
    from cliffnpc.geometry import frontier_report, pareto_mask, smooth_returns
    J = np.array([[1, 0], [0, 1], [0.5, 0.5], [0.4, 0.4], [0.8, 0.6]], float)
    assert list(pareto_mask(J)) == [True, True, False, False, True]
    rng = np.random.default_rng(0)
    t = rng.uniform(0, np.pi / 2, 300)
    trade = np.stack([np.cos(t), np.sin(t)], 1) * rng.uniform(0.2, 1.0, (300, 1))   # quarter disc
    r = frontier_report(np.hstack([trade, np.zeros((300, 2))]))
    assert r["corr"][0, 1] < -0.5 and r["gaps"][0, 1] > 0.2                       # genuine trade-off
    syn = rng.uniform(0, 1, 300)
    r2 = frontier_report(np.stack([syn, syn, np.zeros(300), np.zeros(300)], 1))
    assert r2["n_pareto"] == 1 and r2["gaps"][0, 1] < 1e-9                        # one policy wins both
    th = rng.uniform(-1, 1, (80, 6))
    Je = np.repeat((th[:, :1] ** 2)[:, None, :], 3, axis=1) + 0.01 * rng.normal(size=(80, 3, 1))
    assert np.allclose(smooth_returns(th, np.repeat(Je, 4, axis=2))[:, 0], th[:, 0] ** 2, atol=0.05)
