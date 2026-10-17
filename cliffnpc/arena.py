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
from .affect import AffectState, metric_ok
from .cognition import NPC, forget_frame
from .config import (AffectConfig, CognitionConfig, CombatConfig, Config, LifeConfig, LTMConfig,
                     SteeringConfig)
from .ltm import SequenceMemory
from .steering import steer

NPC_FACTION, RIVAL_FACTION = "npc", "rival"


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
    n_npcs: int = 5
    n_rivals: int = 1
    home: tuple = (20.0, 50.0)
    lair: tuple = (85.0, 50.0)
    food: tuple = ((55.0, 50.0),)
    learning: bool = True
    ltm: LTMConfig | None = None       # long-term sequence memory (None = off)
    life: LifeConfig | None = None     # death, birth, inheritance, witnessing (None = off)
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


@dataclass
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
    in_contact: bool = False
    ltm: SequenceMemory | None = None
    replaying: int = -1         # memory slot being replayed (-1: none)
    slot: int = -1              # index in the capacity-wide logs
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
        self.food = [np.array(f, float) for f in cfg.food]
        self.t = 0.0
        self.n_steps = 0
        if cfg.signature is not None:
            for c in (cfg.npc_cog, cfg.rival_cog):
                c.signature = tuple(float(x) for x in cfg.signature)
        self.life = cfg.life
        self.capacity = cfg.life.capacity if cfg.life else cfg.n_npcs + cfg.n_rivals
        self._free = list(range(self.capacity))
        self._count = {NPC_FACTION: 0, RIVAL_FACTION: 0}
        self.agents: list[Agent] = []
        for _ in range(cfg.n_npcs):
            self.agents.append(self._make(NPC_FACTION, np.array(cfg.home, float)))
        for _ in range(cfg.n_rivals):
            self.agents.append(self._make(RIVAL_FACTION, np.array(cfg.lair, float)))
        self.log: dict = {k: [] for k in ("t", "pos", "C", "health", "mobile", "g", "pain",
                                          "delta", "affect", "contact", "replay", "alive",
                                          "faction", "generation", "fear")}
        self.engagements: list[dict] = []     # one record per contact episode
        self.decisions: list[tuple] = []      # (t, slot, n_allies, C1..C4, faction 0/1) when a rival is near
        self.deaths: list[dict] = []
        self.births: list[dict] = []
        self._episode: dict | None = None

    def _make(self, faction, home, pos=None) -> Agent:
        cfg = self.cfg
        cog, aff = ((cfg.npc_cog, cfg.npc_affect) if faction == NPC_FACTION
                    else (cfg.rival_cog, cfg.rival_affect))
        name = f"{faction}{self._count[faction]}"
        self._count[faction] += 1
        cog = copy.deepcopy(cog)
        C = cog.C0.copy()
        C[0] = self.rng.uniform(-1.5, 0.5)            # stagger hunger so arrivals vary
        npc = NPC(cog, C=C)
        pos = home + self.rng.normal(0, 3.0, 2) if pos is None else pos
        ltm = SequenceMemory(copy.deepcopy(cfg.ltm)) if cfg.ltm is not None else None
        return Agent(name, faction, npc, AffectState(copy.deepcopy(aff)), home, pos,
                     heading=self.rng.uniform(0, 2 * np.pi), ltm=ltm,
                     slot=self._free.pop(0), born=self.t)

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
        if a.faction == RIVAL_FACTION:     # anti-social: every other agent is a crowd
            pool = [b for b in self.agents if b is not a]
        else:
            pool = [b for b in self.agents if b is not a and b.faction == a.faction]
        return [b for b in pool if np.linalg.norm(b.pos - a.pos) <= R]

    def allies_near(self, a: Agent, radius: float | None = None) -> int:
        """Mobile agents near `a` that would fight on its side (NPCs: other NPCs;
        a rival has none) -- or, for a rival, the number of mobile NPCs near it."""
        r = self.wcfg.peer_radius if radius is None else radius
        if a.faction == RIVAL_FACTION:
            return sum(b.faction == NPC_FACTION and b.mobile
                       and np.linalg.norm(b.pos - a.pos) <= r for b in self.agents)
        return sum(b is not a and b.faction == a.faction and b.mobile
                   and np.linalg.norm(b.pos - a.pos) <= r for b in self.agents)

    def percept(self, a: Agent) -> coupling.Percept:
        R = self.wcfg.sensor_radius
        foods = [f - a.pos for f in self.food if np.linalg.norm(f - a.pos) <= R]
        food_vec = min(foods, key=np.linalg.norm) if foods else None
        rival = self._rival_of(a)
        peers = self._peers_of(a)
        peer_vecs = np.array([b.pos - a.pos for b in peers]).reshape(-1, 2)
        known = [f - a.pos for f in self.food]
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
            if cog_tick:
                self._think(i, a, p, h)
            if a.mobile:
                a.heading += self.steer_cfg.wander_turn * np.sqrt(dt) * self.rng.normal()
                v = steer(a.npc.coefficients(), a.npc.cone(), p, a.heading, self.steer_cfg,
                          a.npc.cfg.signature)
                if self.life is not None and not self.life.immobility:
                    v = v * a.health                     # speed falls in proportion to injury
                else:
                    v = v * (0.5 + 0.5 * a.health)      # injured agents are slower
            else:
                v = np.zeros(2)
            new = np.clip(a.pos + v * dt, 0.0, w.size)
            a.vel = (new - a.pos) / dt
            a.pos = new
        self._combat(dt)
        self.t = (self.n_steps + 1) * dt
        self.n_steps += 1
        if cog_tick and self.life is not None:
            self._births()
        if cog_tick:
            self._log()

    def _think(self, i: int, a: Agent, p: coupling.Percept, h: float) -> None:
        npc = a.npc
        X = coupling.invariants(p, self.wcfg)
        beta, _ = coupling.omega_ext(X, npc.cfg)
        if a.ltm is not None:
            beta = beta + self._replay_bivector(a)
        eat = float(a.mobile and X[0] < npc.cfg.eat_radius)
        a.affect.observe_injury(a.taken, a.dealt, h)
        a.taken = a.dealt = 0.0
        a.affect.fear_now = self._fear_level(p, X)
        push = None
        if self.life is not None:     # injury reflex: the strongest repulsive drive
            push = np.array([0.0, 0.0, 0.0, -self.life.injury_reflex * a.affect.taken_rate])
        npc.dynamics_step(beta, h, eat=eat, company=X[2], push=push)
        npc.drive_residual(X, h)
        ok = lambda g: metric_ok(g, a.affect.cfg, npc.eta)  # noqa: E731
        npc.commit_frame(forget_frame(npc.L, npc.L0, npc.cfg), ok)
        beta_total = beta + ga.bivector_from_planes(npc.cfg.omega_int)
        a.affect.step(npc.C, beta_total, h)
        if self.cfg.learning:
            npc.commit_frame(a.affect.learn(npc.L, npc.eta), ok)
        if a.ltm is not None:
            a.ltm.observe(sense_vector(self, a, X, p), npc.C, a.affect.pain, self.t,
                          affect=a.affect.weights * a.affect.a)
        if a.mobile and p.threat_vec is not None:
            self.decisions.append((self.t, a.slot, self.allies_near(a), *map(float, npc.C),
                                   0 if a.faction == NPC_FACTION else 1))

    def _fear_level(self, p: coupling.Percept, X: np.ndarray) -> float:
        """Perceived threat: rival proximity plus its closing speed (0 if none seen)."""
        if p.threat_vec is None:
            return 0.0
        d = np.linalg.norm(p.threat_vec) / self.wcfg.sensor_radius
        return float(max(0.0, 1.0 - d) + max(X[1], 0.0) / 5.0)

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
        contact_any = False
        for r in self.agents:
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
            w.npc.C[3] -= life.shock_flight * f          # recoil: flight
            w.taken += life.vicarious_injury * f          # felt like an injury
            if w.ltm is not None:
                p = self.percept(w)
                key = sense_vector(self, w, coupling.invariants(p, self.wcfg), p)
                w.ltm.imprint(key, life.dread_valence * f * w.ltm.cfg.loss_aversion,
                              np.array([0.0, 0.0, 0.5, 1.0]))   # about injury and fear
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
        caps = {NPC_FACTION: life.npc_cap, RIVAL_FACTION: life.rival_cap}
        used = set()
        for a in list(self.agents):
            if a.name in used or not self._eligible(a) or not self._free:
                continue
            n_fac = sum(b.faction == a.faction for b in self.agents)
            if n_fac >= caps[a.faction]:
                continue
            if a.faction == NPC_FACTION:
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
        ep["rival_down"] = any(not a.mobile for a in self.agents if a.faction == RIVAL_FACTION)
        ep["npc_down"] = [a.name for a in self.agents if a.faction == NPC_FACTION and not a.mobile
                          and a.name in ep["members"]]
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
            rows["faction"][i] = 0 if a.faction == NPC_FACTION else 1
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
