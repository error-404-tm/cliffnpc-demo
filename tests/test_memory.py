"""Phase 3 tests: memory tiers, consolidation, promotion, gossip."""

import numpy as np
import pytest
from scipy.linalg import expm

from cliffnpc import algebra as ga
from cliffnpc.cognition import NPC
from cliffnpc.config import CognitionConfig
from cliffnpc.memory import (BACKGROUND, EPISODIC, METRIC, CollectiveMetric, Episode,
                             EpisodicMemory, PlaceMemory, TieredNPC, fold_episodes, gossip_step)

ETA = ga.ETA


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def make_episode(rng, t):
    r = rng.normal(size=4)
    return Episode(t, rng.normal(size=4), r, rng.uniform(0, 100, 2), 0.1, float(np.linalg.norm(r)))


def test_reservoir_capacity_and_conservation(rng):
    mem = EpisodicMemory(k=5, rng=rng)
    offered, out = [], []
    for t in range(200):
        ep = make_episode(rng, float(t))
        offered.append(ep)
        out += mem.offer(ep)
        assert len(mem.slots) <= 5
    kept = mem.drain()
    # nothing silently lost: every episode is either kept or handed back for folding
    assert len(kept) + len(out) == len(offered)
    assert {id(e) for e in kept + out} == {id(e) for e in offered}


def test_reservoir_prefers_surprising_episodes(rng):
    kept_surprise = []
    for trial in range(30):
        mem = EpisodicMemory(k=5, rng=np.random.default_rng(trial))
        for t in range(100):
            s = 10.0 if t % 20 == 0 else 0.1
            mem.offer(Episode(float(t), np.zeros(4), np.ones(4), np.zeros(2), 0.0, s))
        kept_surprise += [e.surprise for e in mem.episodes()]
    # 5% of offers are surprising; weighted sampling keeps them far above that share
    assert np.mean(np.array(kept_surprise) == 10.0) > 0.5


def test_fold_is_chronological(rng):
    cfg = CognitionConfig(learn_rate=0.2, max_frame_step=1.0)
    eps = [make_episode(rng, float(t)) for t in range(6)]
    shuffled = [eps[i] for i in rng.permutation(6)]
    L1 = fold_episodes(np.eye(4), eps, cfg)
    L2 = fold_episodes(np.eye(4), shuffled, cfg)
    assert np.allclose(L1, L2)
    assert ga.signature(ga.metric_from_frame(L1)) == (3, 1, 0)


def test_demotion_consolidates_episodes(rng):
    cfg = CognitionConfig(learn_rate=0.05)
    col = CollectiveMetric(np.eye(4), np.eye(4), cfg)
    a = TieredNPC(NPC(cfg), EPISODIC, col, rng)
    for t in range(5):
        a.episodes.offer(make_episode(rng, float(t)))
    L_before = a.npc.L.copy()
    a.demote(n_population=10)
    assert a.tier == METRIC
    assert a.places is None and len(a.episodes.slots) == 0
    assert a.folds == 5
    assert not np.allclose(a.npc.L, L_before)   # lossy, never amnesiac


def test_promotion_from_background_starts_at_collective(rng):
    cfg = CognitionConfig()
    Lc = expm(0.2 * ga.split_eta_generator(rng.normal(size=(4, 4)))[1])
    col = CollectiveMetric(Lc, np.eye(4), cfg)
    a = TieredNPC(NPC(cfg), BACKGROUND, col, rng)
    assert np.allclose(a.g, col.g)
    a.promote(jitter=0.005)
    assert a.tier == METRIC
    assert np.linalg.norm(a.g - col.g) < 0.1
    assert np.allclose(a.npc.L0, a.npc.L)
    a.promote()
    assert a.tier == EPISODIC and a.episodes is not None


def test_demotion_to_background_feeds_collective(rng):
    cfg = CognitionConfig()
    col = CollectiveMetric(np.eye(4), np.eye(4), cfg)
    L = expm(0.3 * ga.split_eta_generator(rng.normal(size=(4, 4)))[1])
    a = TieredNPC(NPC(cfg, L=L), METRIC, col, rng)
    g_ind = a.g.copy()
    a.demote(n_population=4)
    assert a.tier == BACKGROUND
    d_after = np.linalg.norm(col.g - g_ind)
    assert d_after < np.linalg.norm(ETA - g_ind)
    assert np.allclose(a.g, col.g)


def test_place_memory_lru_and_recall():
    pm = PlaceMemory(m=2, cell=10.0)
    pm.accumulate(np.array([5.0, 5.0]), 0.1, t=0.0)
    pm.accumulate(np.array([6.0, 4.0]), 0.2, t=1.0)      # same cell
    pm.accumulate(np.array([25.0, 5.0]), 0.5, t=2.0)
    assert np.isclose(pm.recall(np.array([1.0, 1.0]), t=3.0), 0.3)
    pm.accumulate(np.array([55.0, 55.0]), 0.7, t=4.0)    # evicts the LRU cell (25, 5)
    assert len(pm.slots) == 2
    assert pm.recall(np.array([25.0, 5.0]), t=5.0) == 0.0
    assert np.isclose(pm.recall(np.array([55.0, 55.0]), t=5.0), 0.7)


def _metric_agents(rng, n, col, cfg, spread=0.3):
    out = []
    for _ in range(n):
        L = expm(spread * ga.split_eta_generator(rng.normal(size=(4, 4)))[1])
        out.append(TieredNPC(NPC(cfg, L=L, L0=L.copy()), METRIC, col, rng))
    return out


def test_gossip_contracts_metric_differences(rng):
    cfg = CognitionConfig()
    col = CollectiveMetric(np.eye(4), np.eye(4), cfg)
    agents = _metric_agents(rng, 4, col, cfg)
    A = np.ones((4, 4), dtype=int) - np.eye(4, dtype=int)

    def spread():
        gs = [a.g for a in agents]
        return max(np.linalg.norm(gi - gj) for gi in gs for gj in gs)

    s0 = spread()
    for _ in range(200):
        gossip_step(agents, A, 0.05, col)
    assert spread() < 0.05 * s0
    for a in agents:
        assert ga.signature(a.g) == (3, 1, 0)


def test_gossip_is_gauge_invariant(rng):
    """A Lorentz gauge change of a neighbour's frame doesn't change the update."""
    cfg = CognitionConfig()
    col = CollectiveMetric(np.eye(4), np.eye(4), cfg)
    A = np.array([[0, 1], [1, 0]])
    a1 = _metric_agents(np.random.default_rng(3), 2, col, cfg)
    a2 = _metric_agents(np.random.default_rng(3), 2, col, cfg)
    Lam = expm(ga.split_eta_generator(rng.normal(size=(4, 4)))[0])   # in SO(3,1)
    assert np.allclose(Lam.T @ ETA @ Lam, ETA)
    a2[1].npc.L = Lam @ a2[1].npc.L
    gossip_step(a1, A, 0.1, col)
    gossip_step(a2, A, 0.1, col)
    assert np.allclose(a1[0].g, a2[0].g)
