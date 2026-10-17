"""Life in the arena: per-simulation signature, lethal injury, starvation, witnessing
a death, reproduction with memory inheritance."""

import numpy as np

from cliffnpc import algebra as ga
from cliffnpc.arena import NPC_FACTION, RIVAL_FACTION, Arena, ArenaConfig
from cliffnpc.config import AffectConfig, LifeConfig, LTMConfig
from cliffnpc.ltm import AVERSIVE

SIG22 = (-1.0, 1.0, 1.0, -1.0)


def _arena(n_npcs=2, signature=None, ltm=True, **life_kw):
    cfg = ArenaConfig(n_npcs=n_npcs, seed=0, ltm=LTMConfig() if ltm else None,
                      life=LifeConfig(**life_kw), signature=signature)
    cfg.npc_affect.fear_from_death = 5.0
    return Arena(cfg)


def _park(ar):
    for k, a in enumerate(ar.agents):
        a.pos = np.array([5.0 + 20.0 * k, 5.0])
        a.npc.C[3] = 0.0


def test_one_signature_for_every_agent_including_newborns():
    ar = _arena(signature=SIG22)
    assert all(a.npc.cfg.signature == SIG22 for a in ar.agents)
    assert all(np.allclose(a.npc.g, ga.eta_of(SIG22)) for a in ar.agents)
    npcs = [a for a in ar.agents if a.faction == NPC_FACTION]
    child = ar._birth((npcs[0], npcs[1]))
    assert child.npc.cfg.signature == SIG22
    assert ga.signature(child.npc.g)[:2] == (2, 2)


def test_injury_kills_and_is_recorded_as_injury():
    ar = _arena(n_npcs=1)
    _park(ar)
    rival = next(a for a in ar.agents if a.faction == RIVAL_FACTION)
    npc = next(a for a in ar.agents if a.faction == NPC_FACTION)
    npc.pos = rival.pos + np.array([0.5, 0.0])
    npc.health = 0.01
    ar._combat(1.0)
    assert npc not in ar.agents
    assert ar.deaths[-1]["cause"] == "injury" and ar.deaths[-1]["name"] == npc.name


def test_speed_falls_in_proportion_to_health():
    speeds = []
    for health in (1.0, 0.5, 0.25):
        ar = _arena(n_npcs=1)
        _park(ar)
        npc = ar.agents[0]
        npc.health = health
        npc.npc.C[:] = [1.0, 0.0, 0.0, 0.0]            # hungry: heads for food
        x0 = npc.pos.copy()
        ar.step()
        speeds.append(np.linalg.norm(npc.pos - x0))
    assert np.allclose(np.array(speeds) / speeds[0], [1.0, 0.5, 0.25], rtol=1e-6)


def test_starvation_kills():
    ar = _arena(n_npcs=1, starve_rate=1.0)
    _park(ar)
    npc = ar.agents[0]
    npc.npc.C[0] = 3.0
    npc.health = 0.05
    ar._combat(0.1)
    assert npc not in ar.agents and ar.deaths[-1]["cause"] == "starvation"


def test_witnessing_a_death_shocks_sensitizes_and_imprints_a_memory():
    ar = _arena(n_npcs=3)
    _park(ar)
    npcs = [a for a in ar.agents if a.faction == NPC_FACTION]
    victim, witness, far = npcs
    witness.pos = victim.pos + np.array([3.0, 0.0])
    far.pos = victim.pos + np.array([80.0, 80.0])
    c4 = witness.npc.C[3]
    victim.health = 0.0
    ar._combat(0.1)
    d = ar.deaths[-1]
    assert witness.name in d["witnesses"] and far.name not in d["witnesses"]
    assert witness.npc.C[3] < c4 - 1.0                   # recoil toward flight
    assert witness.affect.fear_learned > 0 and far.affect.fear_learned == 0
    assert witness.taken > 0                             # felt like an injury
    assert len(witness.ltm.pending) == 1
    for _ in range(witness.ltm.cfg.seq_len * witness.ltm.cfg.stride * ar.wcfg.cog_every + 8):
        ar.step()                                         # its reaction becomes the memory
    imp = [e for e in witness.ltm.events if e.get("origin") == "imprinted"]
    assert imp and imp[0]["kind"] == AVERSIVE


def test_reproduction_inherits_memories_frame_and_fear_and_costs_the_parents():
    ar = _arena(n_npcs=2)
    _park(ar)
    a, b = [x for x in ar.agents if x.faction == NPC_FACTION]
    rng = np.random.default_rng(0)
    for p in (a, b):
        for _ in range(5):
            p.ltm.store(rng.normal(size=8).astype(np.float32), rng.normal(size=(16, 4)), AVERSIVE, 2.0)
        p.affect.fear_learned = 4.0
    a.npc.L = a.npc.L @ np.diag([1.1, 1.0, 0.9, 1.0])
    child = ar._birth((a, b))
    assert np.allclose(child.npc.L, a.npc.L) and np.allclose(child.npc.L0, a.npc.L)
    inh = [e for e in child.ltm.events if e.get("origin") == "inherited"]
    assert len(inh) == 10
    assert np.isclose(child.affect.fear_learned, ar.life.inherit_fear * 4.0)
    assert child.generation == 1 and child.parents == (a.name, b.name)
    assert a.health < 1.0 and b.health < 1.0 and a.last_birth == ar.t


def test_births_need_a_partner_for_npcs_and_respect_caps():
    ar = _arena(n_npcs=1, maturity=0.0, birth_cooldown=0.0, npc_cap=2)
    _park(ar)
    npc = next(a for a in ar.agents if a.faction == NPC_FACTION)
    npc.npc.C[0] = -1.0
    ar._births()
    assert not [b for b in ar.births if b["faction"] == NPC_FACTION]   # alone: no child
    ar2 = _arena(n_npcs=2, maturity=0.0, birth_cooldown=0.0, npc_cap=2)
    _park(ar2)
    for a in ar2.agents:
        a.npc.C[0] = -1.0
        a.pos = np.array([50.0, 50.0])
    ar2._births()
    assert not [b for b in ar2.births if b["faction"] == NPC_FACTION]  # cap of 2 reached


def test_life_off_keeps_the_old_arena():
    ar = Arena(ArenaConfig(n_npcs=2, seed=0))
    log = ar.run(30.0)
    assert log["alive"].all() and ar.capacity == 3


def test_saturation_law_contains_a_boost_runaway():
    from cliffnpc.cognition import NPC
    from cliffnpc.config import CognitionConfig
    beta = ga.bivector_from_planes({(0, 2): 1.0})        # e13 forcing: a boost in Cl(2,2)
    peaks = {}
    for gain in (0.0, 0.5):
        c = CognitionConfig(signature=SIG22, sat_gain=gain, hunger_drift=0.0, omega_int={})
        npc = NPC(c, C=np.array([-0.5, 0.0, 0.3, 0.0]))
        for _ in range(500):
            npc.dynamics_step(beta, 0.2)
        peaks[gain] = np.abs(npc.C).max()
    assert peaks[0.0] > 4.0                                # runs away (only damping and the cap resist)
    assert peaks[0.5] < 0.5 * peaks[0.0]                   # settles far lower


def test_satiety_slows_a_little():
    speeds = []
    for hunger in (0.0, -1.0, -4.0):
        ar = _arena(n_npcs=1)
        _park(ar)
        npc = ar.agents[0]
        npc.npc.C[:] = [hunger, 0.0, 0.5, 0.0]           # explore drives motion
        npc.heading = 0.0
        x0 = npc.pos.copy()
        ar.rng = np.random.default_rng(1)
        ar.step()
        speeds.append(np.linalg.norm(npc.pos - x0))
    assert speeds[0] > speeds[1] > speeds[2]
    assert np.isclose(speeds[2] / speeds[0], 1.0 - ar.life.satiety_slow, rtol=0.05)


def _food_arena(n_npcs=2, **kw):
    from cliffnpc.config import FoodConfig
    cfg = ArenaConfig(n_npcs=n_npcs, seed=0, life=LifeConfig(), food_dynamics=FoodConfig(**kw))
    return Arena(cfg)


def test_food_is_depleted_faster_by_more_eaters_and_then_disappears():
    for n in (1, 3):
        ar = _food_arena(n_npcs=n, initial_patches=1, spawn_rate=0.0, stock=5.0)
        p = ar.patches[0]
        for a in ar.agents:
            a.pos = p["pos"] + np.array([0.0, 2.0 if a.faction == RIVAL_FACTION else 0.5])
        ar.agents = [a for a in ar.agents if a.faction == NPC_FACTION]
        ar._food_step(1.0)
        assert np.isclose(5.0 - p["stock"], ar.fd.take_rate * n)
    ar._food_step(100.0)
    assert not ar.patches and ar.food_events[-1]["what"] == "depleted"


def test_new_patches_spawn_elsewhere_up_to_the_maximum():
    ar = _food_arena(initial_patches=1, spawn_rate=1e6, max_patches=4)
    first = ar.patches[0]["pos"].copy()
    for _ in range(10):
        ar._food_step(0.1)
    assert len(ar.patches) == 4
    assert all(not np.allclose(p["pos"], first) for p in ar.patches[1:])


def test_agents_only_know_food_they_have_seen_and_forget_vanished_patches():
    ar = _food_arena(n_npcs=1, initial_patches=1, spawn_rate=0.0)
    a = next(x for x in ar.agents if x.faction == NPC_FACTION)
    p = ar.patches[0]
    a.pos = p["pos"] + np.array([45.0, 0.0])          # out of sight
    assert ar.percept(a).food_memory is None
    a.pos = p["pos"] + np.array([10.0, 0.0])          # sees it, remembers it
    ar.percept(a)
    a.pos = p["pos"] + np.array([45.0, 0.0])
    assert ar.percept(a).food_memory is not None
    ar.patches.clear()                                 # eaten up elsewhere
    assert ar.percept(a).food_memory is not None       # still believes it is there
    a.pos = p["pos"] + np.array([10.0, 0.0])          # comes to look: gone
    ar.percept(a)
    assert p["id"] not in a.known_food


def test_search_grows_with_hunger_squared_when_no_food_is_known():
    from cliffnpc.config import SteeringConfig
    from cliffnpc.coupling import Percept
    from cliffnpc.steering import steer
    cfg = SteeringConfig(search_gain=2.0, urgency=0.002)  # tiny urgency: tanh is linear
    p = Percept(None, None, np.zeros(2), np.zeros((0, 2)), np.array([0.0, 0.0]), None)
    speed = lambda h: np.linalg.norm(steer(np.array([h, 0.0, 0.0, 0.0]), 1.0, p, 0.0, cfg))  # noqa: E731
    s1, s2 = speed(0.5), speed(1.5)
    assert np.isclose(s2 / s1, (1.5 + 2 * 1.5 ** 2) / (0.5 + 2 * 0.5 ** 2), rtol=0.01)


# ---------------------------------------------------------------- tribes
from cliffnpc.arena import TRIBE_A, TRIBE_B, life_config  # noqa: E402


def _tribes(n=3, **food_kw):
    cfg = life_config(seed=0)
    cfg.n_npcs = n
    for k, v in food_kw.items():
        setattr(cfg.food_dynamics, k, v)
    return Arena(cfg)


def test_two_identical_tribes_with_equal_numbers():
    ar = _tribes(4)
    by = {f: [a for a in ar.agents if a.faction == f] for f in (TRIBE_A, TRIBE_B)}
    assert len(by[TRIBE_A]) == len(by[TRIBE_B]) == 4
    a, b = by[TRIBE_A][0], by[TRIBE_B][0]
    assert a.npc.cfg.C0.tolist() == b.npc.cfg.C0.tolist()
    assert a.npc.cfg.signature == b.npc.cfg.signature
    assert a.affect.cfg.social_sign == b.affect.cfg.social_sign == 1.0


def test_tribes_regard_only_their_own_as_peers_and_the_other_as_rival():
    ar = _tribes(2)
    a1, a2 = [x for x in ar.agents if x.faction == TRIBE_A]
    b1, b2 = [x for x in ar.agents if x.faction == TRIBE_B]
    for k, x in enumerate((a1, a2, b1, b2)):
        x.pos = np.array([50.0 + 3.0 * k, 50.0])
    assert set(ar._peers_of(a1)) == {a2}
    assert ar._rival_of(a1) in (b1, b2)
    assert ar.allies_near(a1) == 1


def test_tribe_combat_is_symmetric_and_numbers_decide():
    D = ArenaConfig().combat.damage_rate
    for n_a, n_b in ((1, 1), (3, 1)):
        ar = _tribes(3)
        A = [x for x in ar.agents if x.faction == TRIBE_A]
        B = [x for x in ar.agents if x.faction == TRIBE_B]
        for x in A + B:
            x.pos = np.array([5.0, 5.0]) + 20 * np.array([ar.agents.index(x), 0.0])
            x.npc.C[3] = 1.0                          # all willing to fight
        center = np.array([50.0, 50.0])
        for k, x in enumerate(A[:n_a]):
            x.pos = center + np.array([-0.5, 0.3 * k])
        for k, x in enumerate(B[:n_b]):
            x.pos = center + np.array([0.5, 0.3 * k])
        ar._combat(1.0)
        hit_a = sum(1.0 - x.health for x in A[:n_a])
        hit_b = sum(1.0 - x.health for x in B[:n_b])
        assert np.isclose(hit_a, D * n_b / (n_a + n_b))
        assert np.isclose(hit_b, D * n_a / (n_a + n_b))


def test_both_tribes_reproduce_in_pairs():
    ar = _tribes(2)
    ar.life.maturity = 0.0
    ar.life.birth_cooldown = 0.0
    for x in ar.agents:
        x.npc.C[0] = -1.0
        x.pos = ar.homes[x.faction].copy()
    ar._births()
    assert sorted(b["faction"] for b in ar.births) == [TRIBE_A, TRIBE_B]
    assert all(len(b["parents"]) == 2 for b in ar.births)


def test_half_of_food_spawns_at_the_fixed_sites():
    ar = _tribes(1, spawn_rate=1e6, max_patches=3)
    sites = ar._fixed_sites()
    fixed = rand = 0
    for _ in range(400):
        ar.patches.clear()
        pos = ar._spawn_spot()
        if any(np.allclose(pos, s) for s in sites):
            fixed += 1
        else:
            rand += 1
    assert 0.4 < fixed / (fixed + rand) < 0.6
    assert all(any(np.allclose(p["pos"], s) for s in sites) for p in _tribes(1).patches)
