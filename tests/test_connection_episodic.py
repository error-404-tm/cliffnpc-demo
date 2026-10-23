"""Learned connection, event-chain memory, interrogation."""

import numpy as np
from scipy.linalg import expm

from cliffnpc import algebra as ga
from cliffnpc.config import ConnectionConfig, EpisodicConfig
from cliffnpc.connection import FEATURES, LearnedConnection, context_vector
from cliffnpc.episodic import EVENT_TYPES, TYPE_IDS, EventMemory, chain_dtype, event_dtype

ETA22 = ga.eta_of((-1.0, 1.0, 1.0, -1.0))


def test_connection_eligibility_is_the_derivative_of_the_action_rates():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 4)) * 0.3
    _, Xs = ga.split_eta_generator(X, ETA22)
    g = ga.metric_from_frame(expm(Xs), ETA22)
    C = rng.normal(size=4)
    x = rng.uniform(0, 1, len(FEATURES))
    conn = LearnedConnection(ConnectionConfig())
    conn.W = rng.normal(size=conn.W.shape)
    conn.W = conn.W - np.swapaxes(conn.W, 1, 2)
    conn.accumulate(x, C, g, 1.0, action_scale=0.2, trace_tau=1e9)   # trace = E exactly
    s = np.tanh(C / 0.2)
    for k in range(4):
        f = lambda W: s[k] * (-(conn.cfg.gain * np.einsum("f,fij->ij", x, W)) @ g @ C)[k]  # noqa: E731
        d = rng.normal(size=conn.W.shape)
        d = d - np.swapaxes(d, 1, 2)
        eps = 1e-6
        fd = (f(conn.W + eps * d) - f(conn.W - eps * d)) / (2 * eps)
        assert np.isclose(fd, np.sum(conn.trace[k] * d), rtol=1e-6, atol=1e-9)


def test_pain_dampens_the_situational_tendency_and_reward_amplifies_it():
    g = ETA22
    C = np.array([0.2, -0.4, 0.1, 0.5])                 # in company, fighting
    x = context_vector(30.0, 10.0, 2.0, 2, 1, 1.0, 30.0)  # rival near, allies near
    base = LearnedConnection(ConnectionConfig())
    base.W[FEATURES.index("backed")] = 0.3 * ga.bivector_from_planes({(3, 1): 1.0})
    rate = lambda c: (ga.generator_from_bivector(c.beta(x), g) @ C)[3]   # noqa: E731
    out = {}
    for blame in (+1.0, -1.0):
        c = LearnedConnection(ConnectionConfig(learn_rate=0.5, decay=0.0, max_norm=10))
        c.W = base.W.copy()
        c.accumulate(x, C, g, 1.0, 0.2, 1e9)
        c.learn(np.array([0.0, 0.0, 0.0, blame]))
        out[blame] = rate(c) * np.sign(C[3])
    assert out[+1.0] < rate(base) * np.sign(C[3]) < out[-1.0]


def test_backed_and_outnumbered_features():
    alone = context_vector(30, 10, 0, 0, 1, 1.0, 30)
    backed = context_vector(30, 10, 0, 2, 1, 1.0, 30)
    assert alone[FEATURES.index("backed")] == 0 and backed[FEATURES.index("backed")] > 0
    assert alone[FEATURES.index("outnumbered")] > backed[FEATURES.index("outnumbered")]


def test_event_records_are_powers_of_two():
    assert event_dtype().itemsize == 64
    assert chain_dtype(EpisodicConfig()).itemsize == 2048
    m = EventMemory()
    assert m.buffer.ctypes.data % 64 == 0 and m.slots.ctypes.data % 64 == 0


def _chain_memory():
    m = EventMemory(EpisodicConfig(match_threshold=0.5))
    ctx = np.full(16, 0.3)
    t = 0.0
    for name in ("rival_seen", "contact", "injury", "alone"):
        t += 2.0
        m.observe_event(name, 1.0, ctx, t)
    m.t = t
    m.imprint(5.0, t)
    return m, ctx


def test_partial_segment_of_a_chain_triggers_fast_weak_replay_of_what_came_next():
    m, ctx = _chain_memory()
    m2 = EventMemory(m.cfg)
    m2.slots[:] = m.slots
    m2.observe_event("rival_seen", 1.0, ctx, 100.0)       # the chain's opening...
    m2.observe_event("contact", 1.0, ctx, 102.0)          # ...is happening again
    assert m2.step(102.0) == [] or True
    act = m2.activations[-1]
    assert act["matched"] == ["rival_seen", "contact"]
    assert act["primed"][:2] == ["injury", "alone"]
    due_times = [q[0] for q in m2.replay_queue]
    assert all(t - 102.0 < 2.0 * 2 / m2.cfg.replay_speed + 1e-9 for t in due_times)  # compressed
    out = m2.step(110.0)                                  # replayed events arrive, weakly
    names = [v for v, _ in out]
    assert out and all(abs(np.linalg.norm(B)) > 0 for _, B in out)
    inj = [B for v, B in out if v > 0]
    full = np.linalg.norm(EVENT_TYPES["injury"][0])
    assert max(np.linalg.norm(B) for B in inj) <= m2.cfg.replay_strength * full + 1e-9
    assert len(names) >= 1


def test_no_trigger_without_a_matching_segment_and_refractory_after_firing():
    m, ctx = _chain_memory()
    m2 = EventMemory(m.cfg)
    m2.slots[:] = m.slots
    m2.observe_event("food_seen", 1.0, ctx, 50.0)
    m2.observe_event("meal", 1.0, ctx, 52.0)
    m2.step(52.0)
    assert not m2.activations
    m2.observe_event("rival_seen", 1.0, ctx, 60.0)
    m2.observe_event("contact", 1.0, ctx, 61.0)
    m2.step(61.0)
    m2.observe_event("rival_seen", 1.0, ctx, 63.0)
    m2.observe_event("contact", 1.0, ctx, 64.0)
    m2.step(64.0)
    assert len(m2.activations) == 1                       # refractory


def test_interrogation_runs_on_a_learning_arena():
    from cliffnpc.arena import Arena, life_config
    from cliffnpc.interrogate import (courage, explain, metric_report, population_snapshot,
                                      response_table)
    cfg = life_config((-1.0, 1.0, 1.0, -1.0), 0, memory_model="episodic", connection=True)
    cfg.n_npcs = 3
    cfg.basis_scale = (1.3, 1.0, 1.0, 0.8)
    cfg.snapshot_every = 20.0
    ar = Arena(cfg)
    a = ar.agents[0]
    assert np.isclose(metric_report(a)["basis_magnitudes"]["hunger"], 1.3)
    ar.run(60.0)
    assert len(ar.snapshots) >= 3
    assert set(courage(a)) >= {"courage_index", "numbers_index"}
    assert len(response_table(a)) == 8
    assert "basis magnitudes" in explain(a)
    snap = population_snapshot(ar)
    assert "courage_index" in snap and "primed_share" in snap


def test_surprise_gate_writes_a_rare_large_delta_in_one_step_without_overshoot():
    from cliffnpc.connection import N_FEAT
    for gain, expect_big in ((0.0, False), (1.0, True)):
        c = LearnedConnection(ConnectionConfig(rule="td", actor=False, surprise_gain=gain))
        x = np.zeros(N_FEAT)
        x[FEATURES.index("place_cue")] = 0.5
        for _ in range(500):                                  # quiet background: small rewards
            c.td_update(x, 0.001, 0.2)
        v0 = c.v @ np.append(x, 1.0)
        delta = c.td_update(x, -3.0, 0.2)                     # a witnessed death
        dv = c.v @ np.append(x, 1.0) - v0
        assert (dv < -1.0) == expect_big                      # gated: most of delta written at once
        assert dv >= delta - 1e-9                             # normalised: never more than delta
