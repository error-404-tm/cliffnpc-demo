"""Arena: a pack of NPCs and an anti-social rival competing for food.

Both factions run the same cognition (Cl(3,1) drives, own metric).  What
differs is temperament: the rival is withdrawn (social baseline < 0), finds
crowding painful (affect social_sign = -1), and is a little more aggressive.

Combat is world physics, never told to the agents:
  * contact (distance < contact_radius) between a mobile NPC and a mobile rival
    is a mutual attack;
  * every mobile NPC within join_radius of the rival with fight tendency
    C^4 > 0 joins the attack;
  * with n attackers, a contact second removes damage_rate health in total:
    the rival takes n/(n+k) of it, the attackers share the rest equally;
  * health < immobile_below -> immobile until it recovers above mobile_above;
    an immobile agent can't move or eat and is ignored as a rival.

Learning is pain aversion (affect.py): injury, hunger and loneliness are pain,
being fed, in company and hurting a rival are reward; the pain prediction error
reshapes each agent's metric along its eligibility trace.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from . import algebra as ga
from . import coupling
from .affect import MOB_PLANE, AffectState, TrivectorAffect, metric_ok
from .cognition import NPC, forget_frame
from .config import (AffectConfig, CognitionConfig, CombatConfig, Config, ConnectionConfig,
                     EpisodicConfig, FoodConfig, LifeConfig, LTMConfig, SteeringConfig,
                     TrivectorConfig)
from .connection import FEATURES, LearnedConnection, context_vector
from .episodic import EventMemory
from .ltm import SequenceMemory
from .steering import steer

NPC_FACTION, RIVAL_FACTION = "npc", "rival"     # legacy arena: NPC pack vs anti-social rival
TRIBE_A, TRIBE_B = "A", "B"                       # tribes arena: two identical tribes
# sense features (sense_vector): food dist, closing, peers, from home, rival dist, health,
# pain, allies.  Dread memories are matched on the danger cues only.
DANGER_MASK = np.array([0, 1, 0, 0, 1, 1, 0, 1], dtype=np.float32)


def npc_cognition() -> CognitionConfig:
    c = CognitionConfig()
    c.social_drift = 0.15       # solitude builds loneliness fast enough to be a usable context
    c.company_rate = 0.5
    c.peer_gain = 0.05          # weak e23 "group wanderlust" rotation, so company dominates
    c.learn_rate = 0.0          # salience learning off: pain aversion drives the metric
    c.forget_rate = 0.0003      # slow drift back toward temperament
    return c


def arena_affect(**kw) -> AffectConfig:
    a = AffectConfig(weights=np.array([0.5, 0.5, 15.0, 0.0]), learn_rate=0.1,
                     expect_rate=0.002,    # slow baseline: respond to pain, not to its derivative
                     max_step=0.05)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def rival_cognition() -> CognitionConfig:
    c = npc_cognition()
    c.C0 = np.array([-0.3, -0.3, -0.2, 0.05])   # withdrawn, slightly aggressive
    c.social_drift = 0.0                         # never lonely
    return c


@dataclass
class ArenaConfig:
    n_npcs: int = 5                    # tribes mode: members per tribe
    n_rivals: int = 1
    # Two identical tribes (same dynamics, same rules, n_npcs each, homes at
    # `home` and `lair`) instead of an NPC pack and an anti-social rival.
    tribes: bool = False
    home: tuple = (20.0, 50.0)
    lair: tuple = (85.0, 50.0)
    food: tuple = ((55.0, 50.0),)
    learning: bool = True
    ltm: LTMConfig | None = None       # long-term sequence memory (None = off)
    life: LifeConfig | None = None     # death, birth, inheritance, witnessing (None = off)
    food_dynamics: FoodConfig | None = None   # finite, moving food (None = fixed infinite food)
    # "vector": the 4-axis affect space (affect.AffectState); "trivector": affects
    # are the grade-3 dual of the drives (affect.TrivectorAffect)
    affect_model: str = "vector"
    tri: TrivectorConfig = field(default_factory=TrivectorConfig)
    # episodic event-chain memory (replaces the sequence memory's motor replay)
    episodic: EpisodicConfig | None = None
    # learned connection: situation -> drive bivector (learned tendencies)
    connection: ConnectionConfig | None = None
    # starting magnitudes of the drive basis vectors: L0 = diag(basis_scale), so
    # g0 = eta * basis_scale^2.  (1, 1, 1, 1) = unit bases.
    basis_scale: tuple = (1.0, 1.0, 1.0, 1.0)
    snapshot_every: float = 0.0        # s between learning snapshots (0 = off)
    # marked places: ((x, y), radius); place_cue = max(0, 1 - distance / radius), the
    # largest over all places (a landmark seen from up to `radius` away)
    place_cues: tuple = ()
    # One drive signature for every agent in the simulation (both factions);
    # None keeps the factions' own configs.  Vary it across concurrent runs.
    signature: tuple | None = None
    seed: int = 0
    base: Config = field(default_factory=Config)
    combat: CombatConfig = field(default_factory=CombatConfig)
    npc_affect: AffectConfig = field(default_factory=arena_affect)
    rival_affect: AffectConfig = field(default_factory=lambda: arena_affect(social_sign=-1.0))
    npc_cog: CognitionConfig = field(default_factory=npc_cognition)
    rival_cog: CognitionConfig = field(default_factory=rival_cognition)


def life_config(signature=(1.0, 1.0, 1.0, -1.0), seed: int = 0, sat_gain: float = 0.5,
                affect_model: str = "trivector", memory_model: str = "sequence",
                connection: bool = False) -> ArenaConfig:
    """The full-life arena used by E9 and the arena viewer: sequence memory, life
    (death, birth, inheritance, witnessing, injury reflex), learned fear, the
    saturation law and finite moving food, with one drive signature for every agent."""
    cfg = ArenaConfig(seed=seed, ltm=LTMConfig(), life=LifeConfig(), signature=signature,
                      tribes=True, home=(30.0, 100.0), lair=(170.0, 100.0),
                      food=((100.0, 60.0), (100.0, 140.0)),    # fixed sites, equidistant
                      food_dynamics=FoodConfig(max_patches=6, spawn_rate=1.0 / 30.0,
                                               false_fraction=0.3))
    cfg.base.world.size = 200.0        # a bigger world: room to explore and to miss each other
    cfg.affect_model = affect_model
    if memory_model == "episodic":
        cfg.ltm = None
        cfg.episodic = EpisodicConfig()
    if connection:
        cfg.connection = ConnectionConfig()
    if affect_model == "trivector":
        for a in (cfg.npc_affect, cfg.rival_affect):
            a.learn_rate = 0.01            # blame sums over three affects per drive: learn slower
            a.max_step = 0.02
    for a in (cfg.npc_affect, cfg.rival_affect):
        a.fear_from_injury, a.fear_from_death, a.fear_decay = 10.0, 10.0, 0.002
        a.fear_max = 12.0              # fear cannot saturate into a constant pain
    for c in (cfg.npc_cog, cfg.rival_cog):
        c.sat_gain = sat_gain
        c.hunger_drift = 0.08          # hunger builds faster ...
    cfg.life.starve_level = 0.8        # ... and becomes dangerous sooner
    cfg.base.steer.search_gain = 2.0   # hungry with no food known: search harder (hunger^2)
    return cfg


@dataclass(eq=False)          # agents are individuals: identity, hashable
class Agent:
    name: str
    faction: str
    npc: NPC
    affect: AffectState
    home: np.ndarray
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0
    health: float = 1.0
    mobile: bool = True
    taken: float = 0.0          # injury taken since the last cognitive tick
    dealt: float = 0.0          # injury dealt since the last cognitive tick
    taken_total: float = 0.0    # running totals (never reset; for scoring)
    dealt_total: float = 0.0
    in_contact: bool = False
    ltm: SequenceMemory | None = None
    replaying: int = -1         # memory slot being replayed (-1: none)
    episodic: EventMemory | None = None
    conn: LearnedConnection | None = None
    ctx: np.ndarray | None = None                    # last situation (context) vector
    prev: dict = field(default_factory=dict)          # last tick's event flags
    ledger: dict = field(default_factory=lambda: dict(dg_diag=0.0, dg_off=0.0, metric_steps=0))
    # optional external controller (baselines, scripted agents): f(arena, agent, percept) -> velocity.
    # Controlled agents skip cognition; they set npc.C[3] themselves to signal fight (+) / flight (-).
    controller: object = None
    slot: int = -1              # index in the capacity-wide logs
    known_food: dict = field(default_factory=dict)   # patch id -> position (seen, finite food)
    known_false: set = field(default_factory=set)    # patch ids found to be false food
    probe: dict = field(default_factory=dict)         # patch id -> seconds tried without satiety
    generation: int = 0
    born: float = 0.0
    last_birth: float = -1e9
    parents: tuple = ()


def sense_vector(ar: "Arena", a: Agent, X: np.ndarray, p: coupling.Percept) -> np.ndarray:
    """8 sense features (power of two) for short-term memory, each roughly O(1)."""
    R = ar.wcfg.sensor_radius
    rival_d = np.linalg.norm(p.threat_vec) / R if p.threat_vec is not None else 1.0
    return np.array([
        X[0] / R,                                  # food distance
        np.clip(X[1] / 5.0, -1.0, 1.0),            # rival closing speed
        min(X[2] / 3.0, 1.0),                      # peer density
        X[3] / 50.0,                               # distance from home
        rival_d,                                   # rival distance
        a.health,
        np.clip(a.affect.pain / 5.0, -1.0, 1.0),   # interoception
        min(ar.allies_near(a) / 4.0, 1.0),         # allies near
    ], dtype=np.float32)


class Arena:
    def __init__(self, cfg: ArenaConfig | None = None):
        self.cfg = cfg = cfg or ArenaConfig()
        self.wcfg = cfg.base.world
        self.steer_cfg: SteeringConfig = cfg.base.steer
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0.0
        self.fd = cfg.food_dynamics
        self.food_rng = np.random.default_rng(cfg.seed + 7919)   # own stream: agents' RNG unaffected
        self.patches: list[dict] = []
        self.food_events: list[dict] = []
        self._patch_id = 0
        if self.fd is None:
            for f in cfg.food:
                self._spawn(np.array(f, float), np.inf)
        else:
            sites = self._fixed_sites()
            for k in range(self.fd.initial_patches):
                pos = sites[k] if k < len(sites) else self._random_spot()
                self._spawn(pos.copy(), self.fd.stock)
        self.n_steps = 0
        if cfg.signature is not None:
            for c in (cfg.npc_cog, cfg.rival_cog):
                c.signature = tuple(float(x) for x in cfg.signature)
        self.life = cfg.life
        self.tribes = cfg.tribes
        self.factions = (TRIBE_A, TRIBE_B) if cfg.tribes else (NPC_FACTION, RIVAL_FACTION)
        n_second = cfg.n_npcs if cfg.tribes else cfg.n_rivals
        self.capacity = cfg.life.capacity if cfg.life else cfg.n_npcs + n_second
        self._free = list(range(self.capacity))
        self._count = {f: 0 for f in self.factions}
        self.homes = {self.factions[0]: np.array(cfg.home, float),
                      self.factions[1]: np.array(cfg.lair, float)}
        self.agents: list[Agent] = []
        for _ in range(cfg.n_npcs):
            self.agents.append(self._make(self.factions[0], self.homes[self.factions[0]]))
        for _ in range(n_second):
            self.agents.append(self._make(self.factions[1], self.homes[self.factions[1]]))
        self.log: dict = {k: [] for k in ("t", "pos", "C", "health", "mobile", "g", "pain",
                                          "delta", "affect", "contact", "replay", "alive",
                                          "faction", "generation", "fear")}
        self.engagements: list[dict] = []     # one record per contact episode
        self.decisions: list[tuple] = []      # (t, slot, n_allies, C1..C4, faction 0/1) when a rival is near
        self.deaths: list[dict] = []
        self.births: list[dict] = []
        self.snapshots: list[dict] = []          # learning snapshots (interrogate.population_snapshot)
        self._next_snapshot = 0.0
        self._episode: dict | None = None

    def _make(self, faction, home, pos=None) -> Agent:
        cfg = self.cfg
        cog, aff = ((cfg.npc_cog, cfg.npc_affect) if faction != RIVAL_FACTION
                    else (cfg.rival_cog, cfg.rival_affect))   # tribes share one config
        name = f"{faction}{self._count[faction]}"
        self._count[faction] += 1
        cog = copy.deepcopy(cog)
        C = cog.C0.copy()
        C[0] = self.rng.uniform(-1.5, 0.5)            # stagger hunger so arrivals vary
        npc = NPC(cog, C=C)
        pos = home + self.rng.normal(0, 3.0, 2) if pos is None else pos
        ltm = SequenceMemory(copy.deepcopy(cfg.ltm)) if cfg.ltm is not None else None
        affect = (TrivectorAffect(copy.deepcopy(cfg.tri), copy.deepcopy(aff))
                  if cfg.affect_model == "trivector" else AffectState(copy.deepcopy(aff)))
        if tuple(cfg.basis_scale) != (1.0, 1.0, 1.0, 1.0):
            npc.L = np.diag(np.asarray(cfg.basis_scale, float))
            npc.L0 = npc.L.copy()
        ep = EventMemory(copy.deepcopy(cfg.episodic)) if cfg.episodic is not None else None
        conn = LearnedConnection(copy.deepcopy(cfg.connection)) if cfg.connection is not None else None
        return Agent(name, faction, npc, affect, home, pos,
                     heading=self.rng.uniform(0, 2 * np.pi), ltm=ltm,
                     slot=self._free.pop(0), born=self.t, episodic=ep, conn=conn)

    # ---------------------------------------------------------------- food
    @property
    def food(self) -> list[np.ndarray]:
        return [p["pos"] for p in self.patches]

    def _fixed_sites(self) -> list[np.ndarray]:
        sites = self.fd.fixed_sites if self.fd.fixed_sites is not None else self.cfg.food
        return [np.array(s, float) for s in sites]

    def _spawn_spot(self) -> np.ndarray:
        """Half the spawns (fixed_fraction) at a free fixed site, the rest anywhere."""
        if self.food_rng.random() < self.fd.fixed_fraction:
            free = [s for s in self._fixed_sites()
                    if all(np.linalg.norm(p["pos"] - s) > 3.0 for p in self.patches)]
            if free:
                return free[int(self.food_rng.integers(len(free)))].copy()
        return self._random_spot()

    def _random_spot(self) -> np.ndarray:
        m = self.fd.margin
        return self.food_rng.uniform(m, self.wcfg.size - m, 2)

    def _spawn(self, pos: np.ndarray, stock: float, false: bool = False, cue: float = 0.0) -> None:
        self.patches.append(dict(id=self._patch_id, pos=pos, stock=stock, stock0=stock,
                                 false=false, born=self.t, cue=cue))
        self.food_events.append(dict(t=self.t, what="spawned", id=self._patch_id, pos=pos.copy(),
                                     false=false))
        self._patch_id += 1

    def _eating(self, a: Agent) -> bool:
        """At a real patch with stock left (false food gives no satiety)."""
        r = self.cfg.npc_cog.eat_radius
        return a.mobile and any(not p["false"] and p["stock"] > 0
                                and np.linalg.norm(a.pos - p["pos"]) < r for p in self.patches)

    def _food_step(self, dt: float) -> None:
        """Eaters deplete patches; empty patches vanish; new ones may spawn."""
        if self.fd is None:
            return
        r = self.cfg.npc_cog.eat_radius
        for p in list(self.patches):
            at = [a for a in self.agents if a.mobile and np.linalg.norm(a.pos - p["pos"]) < r]
            if p["false"]:
                # trying to eat here yields nothing; after a while the agent knows
                for a in at:
                    a.probe[p["id"]] = a.probe.get(p["id"], 0.0) + dt
                    if a.probe[p["id"]] >= self.fd.discover_time and p["id"] not in a.known_false:
                        if self.fd.remember_false:
                            a.known_false.add(p["id"])
                            a.known_food.pop(p["id"], None)
                        else:
                            a.probe[p["id"]] = -1e9     # felt once per visit, nothing remembered
                        if isinstance(a.affect, TrivectorAffect):
                            a.affect.add_false_food()       # disappointment
                        if a.episodic is not None and a.ctx is not None:
                            a.episodic.observe_event("false_food", 1.0, a.ctx, self.t)
                        self.food_events.append(dict(t=self.t, what="discovered_false",
                                                     id=p["id"], by=a.name))
                if self.fd.false_lifetime > 0 and self.t - p["born"] >= self.fd.false_lifetime:
                    self.patches.remove(p)
                    self.food_events.append(dict(t=self.t, what="expired", id=p["id"], pos=p["pos"]))
                continue
            p["stock"] -= self.fd.take_rate * len(at) * dt
            if p["stock"] <= 0:
                self.patches.remove(p)
                self.food_events.append(dict(t=self.t, what="depleted", id=p["id"], pos=p["pos"]))
        if len(self.patches) < self.fd.max_patches and self.food_rng.random() < self.fd.spawn_rate * dt:
            false = self.food_rng.random() < self.fd.false_fraction
            self._spawn(self._spawn_spot(), self.fd.stock, false)

    # ---------------------------------------------------------------- perception
    def _rival_of(self, a: Agent) -> Agent | None:
        best, bd = None, self.wcfg.sensor_radius
        for b in self.agents:
            if b.faction != a.faction and b.mobile:
                d = np.linalg.norm(b.pos - a.pos)
                if d <= bd:
                    best, bd = b, d
        return best

    def _peers_of(self, a: Agent) -> list[Agent]:
        R = self.wcfg.sensor_radius
        if a.faction == RIVAL_FACTION:     # legacy anti-social rival: every other agent is a crowd
            pool = [b for b in self.agents if b is not a]
        else:
            pool = [b for b in self.agents if b is not a and b.faction == a.faction]
        return [b for b in pool if np.linalg.norm(b.pos - a.pos) <= R]

    def allies_near(self, a: Agent, radius: float | None = None) -> int:
        """Mobile agents near `a` that would fight on its side (NPCs: other NPCs;
        a rival has none) -- or, for a rival, the number of mobile NPCs near it."""
        r = self.wcfg.peer_radius if radius is None else radius
        if a.faction == RIVAL_FACTION:      # legacy rival: how many NPCs crowd it
            return sum(b.faction == NPC_FACTION and b.mobile
                       and np.linalg.norm(b.pos - a.pos) <= r for b in self.agents)
        return sum(b is not a and b.faction == a.faction and b.mobile
                   and np.linalg.norm(b.pos - a.pos) <= r for b in self.agents)

    def percept(self, a: Agent) -> coupling.Percept:
        R = self.wcfg.sensor_radius
        seen = [p for p in self.patches if np.linalg.norm(p["pos"] - a.pos) <= R
                and p["id"] not in a.known_false]          # known-false food is ignored
        foods = [p["pos"] - a.pos for p in seen]
        food_vec = min(foods, key=np.linalg.norm) if foods else None
        rival = self._rival_of(a)
        peers = self._peers_of(a)
        peer_vecs = np.array([b.pos - a.pos for b in peers]).reshape(-1, 2)
        if self.fd is None:
            known = [f - a.pos for f in self.food]          # fixed food: always known
        else:
            # remember what is seen; forget remembered spots seen to be empty
            for p in seen:
                a.known_food[p["id"]] = p["pos"]
            ids = {p["id"] for p in self.patches}
            for pid, pos in list(a.known_food.items()):
                if pid not in ids and np.linalg.norm(pos - a.pos) <= R:
                    del a.known_food[pid]
            known = [pos - a.pos for pos in a.known_food.values()]
        return coupling.Percept(
            food_vec=food_vec,
            threat_vec=None if rival is None else rival.pos - a.pos,
            threat_vel=(np.zeros(2) if rival is None else rival.vel) - a.vel,
            peer_vecs=peer_vecs,
            home_vec=a.home - a.pos,
            food_memory=min(known, key=np.linalg.norm) if known else None,
        )

    # ---------------------------------------------------------------- main loop
    def step(self) -> None:
        cfg, w = self.cfg, self.wcfg
        dt = w.dt
        cog_tick = self.n_steps % w.cog_every == 0
        h = dt * w.cog_every
        percepts = [self.percept(a) for a in self.agents]
        for i, (a, p) in enumerate(zip(self.agents, percepts)):
            if a.controller is not None:
                v = a.controller(self, a, p) if a.mobile else np.zeros(2)
                new = np.clip(a.pos + v * dt, 0.0, w.size)
                a.vel = (new - a.pos) / dt
                a.pos = new
                continue
            if cog_tick:
                self._think(i, a, p, h)
            if a.mobile:
                a.heading += self.steer_cfg.wander_turn * np.sqrt(dt) * self.rng.normal()
                v = steer(a.npc.coefficients(), a.npc.cone(), p, a.heading, self.steer_cfg,
                          a.npc.cfg.signature)
                if self.life is not None and not self.life.immobility:
                    v = v * a.health                     # speed falls in proportion to injury
                    sat = np.clip(-a.npc.C[0] / self.life.satiety_ref, 0.0, 1.0)
                    v = v * (1.0 - self.life.satiety_slow * sat)   # satiety slows a little
                else:
                    v = v * (0.5 + 0.5 * a.health)      # injured agents are slower
            else:
                v = np.zeros(2)
            new = np.clip(a.pos + v * dt, 0.0, w.size)
            a.vel = (new - a.pos) / dt
            a.pos = new
        self._combat(dt)
        self._food_step(dt)
        self.t = (self.n_steps + 1) * dt
        self.n_steps += 1
        if cog_tick and self.life is not None:
            self._births()
        if cog_tick:
            self._log()
            se = self.cfg.snapshot_every
            if se > 0 and self.t >= self._next_snapshot:
                from .interrogate import population_snapshot
                self.snapshots.append(population_snapshot(self))
                self._next_snapshot += se

    def _think(self, i: int, a: Agent, p: coupling.Percept, h: float) -> None:
        npc = a.npc
        X = coupling.invariants(p, self.wcfg)
        beta, _ = coupling.omega_ext(X, npc.cfg)
        if a.ltm is not None:
            beta = beta + self._replay_bivector(a)
        beta_world = beta
        trivector = isinstance(a.affect, TrivectorAffect)
        if trivector:
            # option A: rivals near drive a boost in the social-threat plane,
            # aimed by the social drive (company -> fight, isolation -> flight)
            beta_world = beta_world + a.affect.tri.mob_gain * self._rivals_near(a) * MOB_PLANE
            # affect acts back on behaviour: the contraction C _| F as a generator
            beta = beta_world + a.affect.behaviour_bivector(npc.C, npc.g)
        eat = float(self._eating(a)) if self.fd is not None else float(a.mobile and X[0] < npc.cfg.eat_radius)
        # situation vector (context for the connection and for episodic events)
        allies_n = self.allies_near(a)
        a.ctx = self.context_of(a, p, X)
        if a.conn is not None:                       # learned tendencies
            beta = beta + a.conn.beta(a.ctx)
        if a.episodic is not None:
            self._detect_events(a, p, X, eat, allies_n)
        a.affect.observe_injury(a.taken, a.dealt, h)
        a.taken = a.dealt = 0.0
        a.affect.fear_now = self._fear_level(a, p, X)
        push = None
        if self.life is not None:     # injury reflex: the strongest repulsive drive
            push = np.array([0.0, 0.0, 0.0, -self.life.injury_reflex * a.affect.taken_rate])
        npc.dynamics_step(beta, h, eat=eat, company=X[2], push=push)
        npc.drive_residual(X, h)
        ok = lambda g: metric_ok(g, a.affect.cfg, npc.eta)  # noqa: E731
        npc.commit_frame(forget_frame(npc.L, npc.L0, npc.cfg), ok)
        beta_total = beta + ga.bivector_from_planes(npc.cfg.omega_int)
        if trivector:
            primed = a.episodic.step(self.t) if a.episodic is not None else None
            a.affect.step(npc.C, npc.g, beta_world, beta_total, h, eat=eat, density=X[2],
                          primed=primed)
            if a.episodic is not None:
                a.episodic.observe_pain(a.affect.pain, self.t)
        else:
            a.affect.step(npc.C, beta_total, h)
        if a.conn is not None:
            a.conn.accumulate(a.ctx, npc.C, npc.g, h, a.affect.cfg.action_scale,
                              a.affect.cfg.trace_tau)
        if self.cfg.learning:
            g_before = npc.g
            if npc.commit_frame(a.affect.learn(npc.L, npc.eta), ok):
                d = npc.g - g_before
                a.ledger["dg_diag"] += float(np.abs(np.diag(d)).sum())
                a.ledger["dg_off"] += float(np.abs(d - np.diag(np.diag(d))).sum())
                a.ledger["metric_steps"] += 1
            if a.conn is not None:
                a.conn.learn(a.affect.blame)
        if a.ltm is not None:
            a.ltm.observe(sense_vector(self, a, X, p), npc.C, a.affect.pain, self.t,
                          affect=a.affect.weights * a.affect.a)
        if a.mobile and p.threat_vec is not None:
            self.decisions.append((self.t, a.slot, self.allies_near(a), *map(float, npc.C),
                                   self.factions.index(a.faction)))

    def _detect_events(self, a: Agent, p: coupling.Percept, X: np.ndarray, eat: float,
                       allies_n: int) -> None:
        """Turn this tick's changes into discrete events for the episodic memory."""
        ep, pv, t, x = a.episodic, a.prev, self.t, a.ctx
        gap = ep.cfg.event_refractory

        def emit(name, mag):                       # debounced: one event per type per `gap` s
            if t - pv.get(name + "_t", -1e9) >= gap:
                ep.observe_event(name, mag, x, t)
                pv[name + "_t"] = t
        rival = p.threat_vec is not None
        food = p.food_vec is not None
        if rival and not pv.get("rival"):
            emit("rival_seen", 1.0 + max(X[1], 0.0) / 5.0)
        if a.in_contact and not pv.get("contact"):
            emit("contact", 1.0)
        if a.taken > 1e-3:
            emit("injury", 10.0 * a.taken)
        if a.dealt > 1e-3:
            emit("triumph", 10.0 * a.dealt)
        if food and not pv.get("food"):
            emit("food_seen", 1.0)
        if eat and not pv.get("eat"):
            emit("meal", 1.0)
        if allies_n > 0 and pv.get("allies", 0) == 0:
            emit("company", float(allies_n))
        if allies_n == 0 and pv.get("allies", 0) > 0:
            emit("alone", 1.0)
        pv.update(rival=rival, contact=a.in_contact, food=food, eat=bool(eat), allies=allies_n)

    def context_of(self, a: Agent, p: coupling.Percept, X: np.ndarray | None = None) -> np.ndarray:
        """The agent's situation vector (connection.FEATURES order)."""
        if X is None:
            X = coupling.invariants(p, self.wcfg)
        return context_vector(X[0], None if p.threat_vec is None else float(np.linalg.norm(p.threat_vec)),
                              X[1], self.allies_near(a), self._rivals_near(a), a.health,
                              self.wcfg.sensor_radius, food_cue=self.food_cue(a),
                              place_cue=self.place_cue(a))

    def food_cue(self, a: Agent) -> int:
        """How the nearest seen food patch looks: 0 none in view / unmarked, 1 look A, 2 look B."""
        R = self.wcfg.sensor_radius
        seen = [p for p in self.patches if np.linalg.norm(p["pos"] - a.pos) <= R
                and p["id"] not in a.known_false]
        if not seen:
            return 0
        return int(min(seen, key=lambda p: np.linalg.norm(p["pos"] - a.pos)).get("cue", 0))

    def place_cue(self, a: Agent) -> float:
        """Nearness to the closest marked place: 1 at its centre, 0 at `radius` and beyond."""
        best = 0.0
        for c, r in self.cfg.place_cues:
            best = max(best, 1.0 - np.linalg.norm(np.asarray(c, float) - a.pos) / r)
        return float(best)

    def _rivals_near(self, a: Agent) -> int:
        R = self.wcfg.sensor_radius
        return sum(b.faction != a.faction and b.mobile and np.linalg.norm(b.pos - a.pos) <= R
                   for b in self.agents)

    def _fear_level(self, a: Agent, p: coupling.Percept, X: np.ndarray) -> float:
        """Perceived threat, scaled by how outnumbered the agent is:
            (proximity + closing speed) * rivals in view / (1 + allies near)
        Alone among enemies is frightening; inside one's group much less."""
        if p.threat_vec is None:
            return 0.0
        d = np.linalg.norm(p.threat_vec) / self.wcfg.sensor_radius
        base = max(0.0, 1.0 - d) + max(X[1], 0.0) / 5.0
        R = self.wcfg.sensor_radius
        rivals = sum(b.faction != a.faction and b.mobile and np.linalg.norm(b.pos - a.pos) <= R
                     for b in self.agents)
        return float(base * rivals / (1.0 + self.allies_near(a)))

    def _replay_bivector(self, a: Agent) -> np.ndarray:
        """Recall: a generator in the plane C ^ C_target pulling the drives along
        the remembered sequence (rotation, or boost across the light cone)."""
        target = a.ltm.recall()
        a.replaying = a.ltm.replay_slot
        if target is None:
            return np.zeros((4, 4))
        C = a.npc.C
        n = np.linalg.norm(C) * np.linalg.norm(target)
        if n < 1e-9:
            return np.zeros((4, 4))
        orient = np.sign(ga.interval(C, a.npc.g)) or 1.0   # M(C, T) C = (C.C) T - (C.T) C
        gain = a.ltm.cfg.replay_gain * max(a.ltm.activation, a.ltm.cfg.activation)
        return orient * gain * ga.wedge(C, target) / n

    def _combat(self, dt: float) -> None:
        cc = self.cfg.combat
        for a in self.agents:
            a.in_contact = False
        contact_any = self._tribe_combat(dt) if self.tribes else False
        for r in ([] if self.tribes else self.agents):
            if r.faction != RIVAL_FACTION or not r.mobile:
                continue
            npcs = [b for b in self.agents if b.faction == NPC_FACTION and b.mobile]
            touching = [b for b in npcs if np.linalg.norm(b.pos - r.pos) < cc.contact_radius]
            if not touching:
                continue
            joiners = [b for b in npcs if b not in touching and b.npc.C[3] > 0
                       and np.linalg.norm(b.pos - r.pos) < cc.join_radius]
            group = touching + joiners
            n = len(group)
            D = cc.damage_rate * dt
            to_rival = D * n / (n + cc.rival_strength)
            to_each = (D - to_rival) / n
            r.health -= to_rival
            r.taken += to_rival
            r.dealt += D - to_rival
            r.in_contact = True
            for b in group:
                b.health -= to_each
                b.taken += to_each
                b.dealt += to_rival / n
                b.in_contact = True
            contact_any = True
            self._track_episode(r, group, to_rival, to_each)
        if not contact_any and self._episode is not None:
            self._close_episode()
        life = self.life
        dead = []
        for a in self.agents:
            if life is not None and a.npc.C[0] > life.starve_level:
                a.health -= life.starve_rate * (a.npc.C[0] - life.starve_level) * dt
            if life is not None and a.health <= 0.0:
                dead.append(a)
                continue
            if not a.in_contact:
                a.health = min(1.0, a.health + cc.recovery_rate * dt)
            a.health = max(a.health, 0.0)
            if life is not None and not life.immobility:
                continue                                 # continuous slowing instead
            if a.mobile and a.health < cc.immobile_below:
                a.mobile = False
            elif not a.mobile and a.health > cc.mobile_above:
                a.mobile = True
        for a in dead:
            self._die(a, "injury" if a.in_contact else "starvation")

    # ---------------------------------------------------------------- life
    def _die(self, a: Agent, cause: str) -> None:
        """Remove `a`; every agent that sees it die gets a strong shock and an
        imprinted escape memory of its own reaction."""
        life = self.life
        self.agents.remove(a)
        self._free.append(a.slot)
        self._free.sort()
        witnesses = []
        for w in self.agents:
            if np.linalg.norm(w.pos - a.pos) > self.wcfg.sensor_radius:
                continue
            f = 1.0 if w.faction == a.faction else life.foe_death_factor
            w.affect.sensitize(w.affect.cfg.fear_from_death * f)
            if isinstance(w.affect, TrivectorAffect):
                w.affect.add_death(f)                      # felt as a strong threat event
            if w.episodic is not None:
                ctx = w.ctx if w.ctx is not None else np.zeros(len(FEATURES))
                w.episodic.observe_event("death_seen", f, ctx, self.t)
                w.episodic.imprint(life.dread_valence * f * w.episodic.cfg.loss_aversion, self.t)
            w.npc.C[3] -= life.shock_flight * f          # recoil: flight
            w.taken += life.vicarious_injury * f          # felt like an injury
            if w.ltm is not None:
                p = self.percept(w)
                key = sense_vector(self, w, coupling.invariants(p, self.wcfg), p)
                w.ltm.imprint(key, life.dread_valence * f * w.ltm.cfg.loss_aversion,
                              np.array([0.0, 0.0, 0.5, 1.0]),   # about injury and fear
                              mask=DANGER_MASK)                 # matched on danger cues only
            witnesses.append(w.name)
        self.deaths.append(dict(t=self.t, name=a.name, faction=a.faction, cause=cause,
                                generation=a.generation, pos=a.pos.copy(),
                                age=self.t - a.born, witnesses=witnesses))

    def _eligible(self, a: Agent) -> bool:
        life = self.life
        return (a.mobile and a.health >= life.birth_health and a.npc.C[0] <= life.birth_sated
                and self.t - a.last_birth >= life.birth_cooldown
                and self.t - a.born >= life.maturity)

    def _births(self) -> None:
        life = self.life
        caps = ({f: life.npc_cap for f in self.factions} if self.tribes
                else {NPC_FACTION: life.npc_cap, RIVAL_FACTION: life.rival_cap})
        used = set()
        for a in list(self.agents):
            if a.name in used or not self._eligible(a) or not self._free:
                continue
            n_fac = sum(b.faction == a.faction for b in self.agents)
            if n_fac >= caps[a.faction]:
                continue
            if a.faction != RIVAL_FACTION:         # NPCs and both tribes need a partner
                partners = [b for b in self.agents if b is not a and b.faction == a.faction
                            and b.name not in used and self._eligible(b)
                            and np.linalg.norm(b.pos - a.pos) <= life.partner_radius]
                if not partners:
                    continue
                parents = (a, partners[0])
            else:
                parents = (a,)                     # anti-social: reproduces alone
            self._birth(parents)
            used |= {p.name for p in parents}

    def _birth(self, parents: tuple) -> Agent:
        life = self.life
        p0 = parents[0]
        child = self._make(p0.faction, p0.home, pos=np.clip(p0.pos + self.rng.normal(0, 1.0, 2),
                                                           0.0, self.wcfg.size))
        child.npc.L = p0.npc.L.copy()              # inherited personality ...
        child.npc.L0 = p0.npc.L.copy()             # ... as the child's temperament
        child.npc.C[0] = 0.0
        child.generation = 1 + max(p.generation for p in parents)
        child.parents = tuple(p.name for p in parents)
        child.affect.fear_learned = life.inherit_fear * float(np.mean([p.affect.fear_learned
                                                                       for p in parents]))
        n_mem = 0
        if child.episodic is not None:
            for p in parents:
                n_mem += child.episodic.inherit_from(p.episodic, child.episodic.cfg.inherit_memories,
                                                     child.episodic.cfg.inherit_strength)
        if child.conn is not None:                 # habits are inherited too
            child.conn.inherit([p.conn for p in parents], child.conn.cfg.inherit)
        if child.ltm is not None:
            for p in parents:
                n_mem += child.ltm.inherit_from(p.ltm, life.inherit_memories, life.inherit_strength)
        for p in parents:
            p.health -= life.birth_cost_health
            p.npc.C[0] += life.birth_cost_hunger
            p.last_birth = self.t
        self.agents.append(child)
        self.births.append(dict(t=self.t, name=child.name, faction=child.faction,
                                parents=child.parents, generation=child.generation,
                                memories=n_mem, fear=child.affect.fear_learned))
        return child

    def _tribe_combat(self, dt: float) -> bool:
        """Symmetric fights between two tribes.  Agents of different tribes in
        contact form a fight (connected clusters); fighting members of either
        side within join_radius of the other side join.  With n on one side and
        m on the other, one contact second removes damage_rate in total: a side
        takes (other side's size)/(n + m) of it, shared equally."""
        cc = self.cfg.combat
        A = [a for a in self.agents if a.faction == self.factions[0] and a.mobile]
        B = [b for b in self.agents if b.faction == self.factions[1] and b.mobile]
        pairs = [(i, j) for i, a in enumerate(A) for j, b in enumerate(B)
                 if np.linalg.norm(a.pos - b.pos) < cc.contact_radius]
        if not pairs:
            if self._episode is not None:
                self._close_episode()
            return False
        # connected components of the contact graph (union-find over A ∪ B)
        parent = list(range(len(A) + len(B)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, j in pairs:
            parent[find(i)] = find(len(A) + j)
        comps: dict = {}
        for i, j in pairs:
            comps.setdefault(find(i), (set(), set()))
            comps[find(i)][0].add(i)
            comps[find(i)][1].add(j)
        D = cc.damage_rate * dt
        for ia, jb in comps.values():
            side_a = {A[i] for i in ia}
            side_b = {B[j] for j in jb}
            side_a |= {a for a in A if a.npc.C[3] > 0
                       and any(np.linalg.norm(a.pos - b.pos) < cc.join_radius for b in side_b)}
            side_b |= {b for b in B if b.npc.C[3] > 0
                       and any(np.linalg.norm(b.pos - a.pos) < cc.join_radius for a in side_a)}
            n, m = len(side_a), len(side_b)
            to_a, to_b = D * m / (n + m), D * n / (n + m)
            for x in side_a:
                x.health -= to_a / n
                x.taken += to_a / n
                x.dealt += to_b / n
                x.taken_total += to_a / n
                x.dealt_total += to_b / n
                x.in_contact = True
            for x in side_b:
                x.health -= to_b / m
                x.taken += to_b / m
                x.dealt += to_a / m
                x.taken_total += to_b / m
                x.dealt_total += to_a / m
                x.in_contact = True
            self._track_tribe_episode(side_a, side_b, to_a, to_b)
        return True

    def _track_tribe_episode(self, side_a, side_b, to_a, to_b):
        ep = self._episode
        if ep is None:
            ep = self._episode = dict(t0=self.t, n_max=0, sides={f: 0 for f in self.factions},
                                      taken={f: 0.0 for f in self.factions}, members=set(),
                                      rival_taken=0.0, npc_taken=0.0)
        fa, fb = self.factions
        ep["sides"][fa] = max(ep["sides"][fa], len(side_a))
        ep["sides"][fb] = max(ep["sides"][fb], len(side_b))
        ep["n_max"] = max(ep["n_max"], len(side_a), len(side_b))
        ep["taken"][fa] += to_a
        ep["taken"][fb] += to_b
        ep["npc_taken"] += to_a                  # legacy keys: tribe A / tribe B
        ep["rival_taken"] += to_b
        ep["members"] |= {x.name for x in side_a | side_b}

    def _track_episode(self, r, group, to_rival, to_each):
        ep = self._episode
        if ep is None:
            ep = self._episode = dict(t0=self.t, n_max=0, rival_taken=0.0, npc_taken=0.0,
                                      members=set())
        ep["n_max"] = max(ep["n_max"], len(group))
        ep["rival_taken"] += to_rival
        ep["npc_taken"] += to_each * len(group)
        ep["members"] |= {b.name for b in group}

    def _close_episode(self):
        ep = self._episode
        ep["t1"] = self.t
        ep["rival_down"] = any(not a.mobile for a in self.agents if a.faction == self.factions[1])
        ep["npc_down"] = [a.name for a in self.agents if a.faction == self.factions[0]
                          and not a.mobile and a.name in ep["members"]]
        ep["members"] = sorted(ep["members"])
        self.engagements.append(ep)
        self._episode = None

    def _log(self) -> None:
        """Capacity-wide rows: slot i holds the agent living in slot i (empty: NaN / False / -1)."""
        L, cap = self.log, self.capacity
        nan = np.nan
        rows = dict(pos=np.full((cap, 2), nan), C=np.full((cap, 4), nan), health=np.full(cap, nan),
                    mobile=np.zeros(cap, bool), g=np.full((cap, 4, 4), nan),
                    pain=np.full(cap, nan), delta=np.full(cap, nan), affect=np.full((cap, 4), nan),
                    contact=np.zeros(cap, bool), replay=np.full(cap, -1),
                    alive=np.zeros(cap, bool), faction=np.full(cap, -1),
                    generation=np.full(cap, -1), fear=np.full(cap, nan))
        for a in self.agents:
            i = a.slot
            rows["pos"][i] = a.pos
            rows["C"][i] = a.npc.C
            rows["health"][i] = a.health
            rows["mobile"][i] = a.mobile
            rows["g"][i] = a.npc.g
            rows["pain"][i] = a.affect.pain
            rows["delta"][i] = a.affect.delta
            rows["affect"][i] = a.affect.a
            rows["contact"][i] = a.in_contact
            rows["replay"][i] = a.replaying
            rows["alive"][i] = True
            rows["faction"][i] = self.factions.index(a.faction)
            rows["generation"][i] = a.generation
            rows["fear"][i] = a.affect.fear_learned
        L["t"].append(self.t)
        for k, v in rows.items():
            L[k].append(v)

    def run(self, T: float) -> dict:
        for _ in range(int(round(T / self.wcfg.dt))):
            self.step()
        if self._episode is not None:
            self._close_episode()
        return {k: np.array(v) for k, v in self.log.items()}
