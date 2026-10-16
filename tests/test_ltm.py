"""Long-term sequence memory: layout, storage, attractor recall, memory Clifford space."""

import numpy as np

from cliffnpc.arena import Arena, ArenaConfig
from cliffnpc.config import LTMConfig
from cliffnpc.ltm import (ALIGN, APPETITIVE, AVERSIVE, SequenceMemory, is_pow2, memory_dtype)


def test_every_buffer_and_struct_is_power_of_two_and_aligned():
    m = SequenceMemory(LTMConfig())
    assert memory_dtype(m.cfg).itemsize == 512
    for buf in (m.stm, m.beh, m.pain, m.sense_hist, m.slots):
        assert all(is_pow2(n) for n in buf.shape), buf.shape
        assert is_pow2(buf.dtype.itemsize)
        assert buf.ctypes.data % ALIGN == 0
    X, _ = m.memory_vectors()
    assert is_pow2(X.shape[1])


def _feed(m, sense, C, pains):
    """One behaviour sample per pain value (stride ticks each); first event or None."""
    first = None
    for p in pains:
        for _ in range(m.cfg.stride):
            ev = m.observe(sense, C, p, 0.0)
            first = first or ev
    return first


def test_relief_stores_an_aversive_sequence_with_its_starting_context():
    m = SequenceMemory(LTMConfig())
    ctx = np.linspace(0.1, 0.8, 8).astype(np.float32)
    C = np.array([0.1, 0.2, -0.3, -0.6])                  # fleeing
    ev = _feed(m, ctx, C, [1.5] * 8 + list(np.linspace(1.5, 0.0, 8)))
    assert ev is not None and ev["what"] == "stored" and ev["kind"] == AVERSIVE
    s = m.slots[ev["slot"]]
    assert np.allclose(s["key"], ctx)
    assert np.allclose(s["seq"], C)


def test_reward_is_appetitive_and_similar_events_merge_into_one_attractor():
    m = SequenceMemory(LTMConfig())
    ctx = np.full(8, 0.3, np.float32)
    C = np.array([0.6, -0.2, 0.0, 0.4])                   # fighting together
    ev1 = _feed(m, ctx, C, list(np.linspace(0.0, -0.8, 16)))
    assert ev1["kind"] == APPETITIVE
    _feed(m, ctx, C, [0.0] * m.cfg.cooldown)
    ev2 = _feed(m, ctx + 0.05, C, list(np.linspace(0.0, -0.8, 16)))
    assert ev2["what"] == "merged" and ev2["slot"] == ev1["slot"]
    assert (m.slots["active"] == 1).sum() == 1


def test_recall_fires_only_when_the_context_is_in_stm_and_replays_in_order():
    m = SequenceMemory(LTMConfig(match_radius=0.5, activation=0.5))
    key = np.array([0.9, 0.5, 0.0, 0.4, 0.2, 1.0, 0.1, 0.0], np.float32)
    seq = np.stack([np.array([0.0, 0.0, 0.1 * k, -0.5]) for k in range(16)])
    m.store(key, seq, AVERSIVE, 1.0)
    other = np.zeros(8, np.float32)
    for _ in range(40):                                   # unrelated situation
        m.observe(other + 0.9, np.zeros(4), 0.0, 0.0)
    assert m.recall() is None
    m.observe(key + 0.05, np.zeros(4), 0.0, 0.0)          # approximately present
    first = m.recall()
    assert first is not None and np.allclose(first, seq[0])
    replayed = [first] + [m.recall() for _ in range(16 * m.cfg.stride - 1)]
    steps = np.array(replayed[::m.cfg.stride])
    assert np.allclose(steps, seq)                        # whole sequence, in order


def test_attractor_settles_on_the_stronger_of_two_nearby_memories():
    m = SequenceMemory(LTMConfig(merge_radius=0.01, match_radius=0.5, activation=0.5))
    k1 = np.full(8, 0.5, np.float32)
    k2 = k1 + np.float32(0.1)
    m.store(k1, np.full((16, 4), 1.0), APPETITIVE, 0.2)
    m.store(k2, np.full((16, 4), -1.0), APPETITIVE, 3.0)
    m.observe(0.5 * (k1 + k2), np.zeros(4), 0.0, 0.0)     # ambiguous cue, between them
    target = m.recall()
    assert target is not None and np.allclose(target, -1.0)


def test_memory_space_signature_counts_approach_and_escape_modes():
    m = SequenceMemory(LTMConfig())
    rng = np.random.default_rng(0)
    for i in range(6):
        m.store(rng.normal(size=8).astype(np.float32), rng.normal(size=(16, 4)),
                APPETITIVE if i < 4 else AVERSIVE, 1.0)
    sp = m.build_space()
    assert sp.signature == (4, 2)
    assert np.allclose(sp.basis @ sp.basis.T, np.eye(6), atol=1e-9)
    X, v = m.memory_vectors()
    y = sp.coords(X[0])
    assert sp.interval(y) > 0                             # an approach memory reads spacelike
    assert sp.interval(sp.coords(X[5])) < 0               # an escape memory reads timelike
    assert sp.plane_type(0, 1) in ("elliptic", "hyperbolic")


def test_arena_with_ltm_runs():
    ar = Arena(ArenaConfig(n_npcs=3, seed=0, ltm=LTMConfig()))
    log = ar.run(200.0)
    assert np.all(np.isfinite(log["C"]))
    assert "replay" in log


def test_loss_aversion_forming_pain_counts_double():
    c = LTMConfig()
    ctx = np.full(8, 0.2, np.float32)
    C = np.zeros(4)
    # a relief of x and a reward of x: the relief memory is twice as strong
    m1, m2 = SequenceMemory(c), SequenceMemory(c)
    e1 = _feed(m1, ctx, C, [0.5] * 8 + list(np.linspace(0.5, 0.0, 8)))     # relief 0.5
    e2 = _feed(m2, ctx, C, [0.0] * 8 + list(np.linspace(0.0, -0.5, 8)))    # reward 0.5
    assert e1["kind"] == AVERSIVE and e2["kind"] == APPETITIVE
    assert np.isclose(m1.slots["strength"][e1["slot"]], 2 * m2.slots["strength"][e2["slot"]], rtol=1e-5)
    # half the formation threshold is enough for pain, not for reward
    x = 0.6 * c.form_threshold
    m3, m4 = SequenceMemory(c), SequenceMemory(c)
    assert _feed(m3, ctx, C, [x] * 8 + list(np.linspace(x, 0.0, 8))) is not None
    assert _feed(m4, ctx, C, [0.0] * 8 + list(np.linspace(0.0, -x, 8))) is None


def test_loss_aversion_triggering_escape_basin_is_wider():
    c = LTMConfig()
    key = np.full(8, 0.5, np.float32)
    seq = np.zeros((16, 4))
    # a cue at squared mismatch 1.5 x match_radius^2 (activation exp(-1.5) for a reward
    # memory, exp(-0.75) for an escape memory) against threshold exp(-x) in between
    off = np.sqrt(1.5) * c.match_radius / np.sqrt(8)
    cue = key + np.float32(off)
    c2 = LTMConfig(activation=float(np.exp(-1.0)))
    fired = {}
    for kind in (APPETITIVE, AVERSIVE):
        m = SequenceMemory(c2)
        m.store(key, seq, kind, 1.0)
        m.observe(cue, np.zeros(4), 0.0, 0.0)
        fired[kind] = m.recall() is not None
    assert fired[AVERSIVE] and not fired[APPETITIVE]
