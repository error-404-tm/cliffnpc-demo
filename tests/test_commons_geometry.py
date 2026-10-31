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
    at_patch = lambda log: log["s"][:, 0] > 0.9                # food_near: within ~eat radius  # noqa: E731
    valid = Commons(Env(threat=0.0, cue_validity=1), seed=0).run(3)
    random_cues = Commons(Env(threat=0.0, cue_validity=0), seed=0).run(3)
    gain_v, gain_r = valid["r"][:, 3] > 0, random_cues["r"][:, 3] > 0
    assert np.any(gain_v & ~at_patch(valid))                   # valid cues: learned from a distance
    assert gain_r.any() and np.all(at_patch(random_cues)[gain_r] | np.roll(at_patch(random_cues), 1)[gain_r])


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


def test_harness_pairs_worlds_and_flags_untargeted_effects():
    from cliffnpc.harness import BASE, Sizes, compare, world_for, world_run
    sz = Sizes(n_frontier=12, eps_frontier=2, eps_local=4, families=("broad",))
    a = world_run(BASE, 0, sz)
    b = world_run(world_for("threat", 0.0), 0, sz)
    assert np.allclose(a["J"][:, :, 0], b["J"][:, :, 0], atol=0.3)     # CRN: food barely moves
    r = compare(a, b, sz, n_boot=8, targets=("food·safety",))
    assert r["live_a"][1] and not r["live_b"][1]                         # safety dimension vanishes
    assert set(r["flags"]) <= set(r["touched"])


def test_phase_c_rules_ascend_a_known_gradient_and_report_geometry():
    from cliffnpc.commons import Dials
    from cliffnpc.phase_c import GoalGeometryAgent
    A = np.array([[1.0, 0, 0, 0, 0, 0], [-1.0, 1.0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0], [0, 0, 0, 0, 0, 1.0]])
    w = np.array([1.0, 1.0, 0.0, 0.0])
    for rule in ("pg", "ng", "pareto"):
        ag = GoalGeometryAgent(Dials(threat=0.0), w, rule=rule, seed=0)
        ag.A = A
        d = ag.direction()
        assert w @ (A @ d) > 0                                   # the preferred combination improves
        if rule == "pareto":
            assert np.all((A @ d)[:2] >= -1e-9)                  # no valued goal gets worse
    ag = GoalGeometryAgent(Dials(threat=0.0), w, rule="ng", seed=0, warmup=1)
    ag.run(3)
    rep = ag.report()
    assert len(ag.history) == 3 and "food^safety" in rep["planes"] and len(rep["modes"]) == 4


def test_new_dials_absent_leave_the_world_unchanged_and_ctx_reduces_to_global():
    from cliffnpc.commons import N_PHI, Commons, Dials
    th = np.array([[0.8, 1.5, 0.2, 0.6, 0.1, 0.4]])
    a = Commons(Dials(need=1.0), seed=2, noise_override=0.2).run_policies(th, 2)
    b = Commons(Dials(need=1.0, depletion=0.0, allies=0.0, rivals=0.0, false_frac=0.5), seed=2,
                noise_override=0.2).run_policies(th, 2)
    assert np.allclose(a, b)
    Th = np.zeros((1, 6, N_PHI))
    Th[0, :, 0] = th[0]                                        # no situation terms
    c = Commons(Dials(need=1.0), seed=2, noise_override=0.2).run_ctx_policies(Th, 2)
    assert np.allclose(a, c)


def test_stabilised_natural_gradient_always_ascends():
    from cliffnpc.commons import Dials
    from cliffnpc.phase_c import GoalGeometryAgent
    rng = np.random.default_rng(0)
    for policy in ("global", "ctx"):
        ag = GoalGeometryAgent(Dials(threat=0.0), np.array([1.0, 2.0, 0.5, 0.0]), rule="ng", policy=policy)
        for _ in range(20):
            ag.A = rng.normal(size=ag.A.shape) * rng.uniform(0.01, 1.0, (4, 1))   # some goals barely movable
            assert ag.omega @ (ag.A @ ag.direction()) > 0
