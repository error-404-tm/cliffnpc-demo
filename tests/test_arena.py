"""Arena mechanics: combat split, immobility, perception of immobile agents."""

import numpy as np

from cliffnpc import algebra as ga
from cliffnpc.arena import NPC_FACTION, RIVAL_FACTION, Arena, ArenaConfig


def _arena(n_npcs=3):
    ar = Arena(ArenaConfig(n_npcs=n_npcs, seed=0))
    rival = next(a for a in ar.agents if a.faction == RIVAL_FACTION)
    npcs = [a for a in ar.agents if a.faction == NPC_FACTION]
    for a in ar.agents:                 # park everyone far apart
        a.pos = np.array([5.0, 5.0]) + 20 * np.array([ar.agents.index(a), 0.0])
    return ar, rival, npcs


def test_injury_split_favours_groups():
    k = ArenaConfig().combat.rival_strength
    for n in (1, 3):
        ar, rival, npcs = _arena(3)
        rival.pos = np.array([50.0, 50.0])
        for i, b in enumerate(npcs):
            b.pos = np.array([90.0, 90.0])
        for b in npcs[:n]:
            b.pos = rival.pos + np.array([0.5, 0.0])
            b.npc.C[3] = 0.5            # fighting
        h0 = rival.health
        ar._combat(1.0)
        D = ar.cfg.combat.damage_rate
        assert np.isclose(h0 - rival.health, D * n / (n + k))
        for b in npcs[:n]:
            assert np.isclose(1.0 - b.health, D * k / (n + k) / n)


def test_immobility_hysteresis_and_rivals_ignore_the_immobile():
    ar, rival, npcs = _arena(1)
    b = npcs[0]
    b.pos = rival.pos + np.array([3.0, 0.0])
    assert ar._rival_of(rival) is b
    b.health = 0.25
    ar._combat(0.1)
    assert not b.mobile
    assert ar._rival_of(rival) is None          # no longer interesting
    b.health = 0.5
    ar._combat(0.1)
    assert not b.mobile                          # still below mobile_above
    b.health = 0.65
    ar._combat(0.1)
    assert b.mobile


def test_arena_runs_and_metrics_stay_valid():
    ar = Arena(ArenaConfig(n_npcs=3, seed=1))
    log = ar.run(120.0)
    assert np.all(np.isfinite(log["C"]))
    for g in log["g"][-1]:
        assert ga.signature(g) == (3, 1, 0)
        assert np.all(np.diag(g)[:3] > 0) and g[3, 3] < 0
