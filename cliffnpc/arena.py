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
from .config import AffectConfig, CognitionConfig, CombatConfig, Config, LTMConfig, SteeringConfig
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
    a = AffectConfig(weights=np.array([0.5, 0.5, 15.0]), learn_rate=0.1,
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
        self.agents: list[Agent] = []
        for i in range(cfg.n_npcs):
            self.agents.append(self._make(f"npc{i}", NPC_FACTION, cfg.npc_cog, cfg.npc_affect,
                                          np.array(cfg.home, float)))
        for i in range(cfg.n_rivals):
            self.agents.append(self._make(f"rival{i}", RIVAL_FACTION, cfg.rival_cog,
                                          cfg.rival_affect, np.array(cfg.lair, float)))
        self.log: dict = {k: [] for k in ("t", "pos", "C", "health", "mobile", "g", "pain",
                                          "delta", "affect", "contact", "replay")}
        self.engagements: list[dict] = []     # one record per contact episode
        self.decisions: list[tuple] = []      # (t, agent idx, n_allies, C1..C4) when a rival is near
        self._episode: dict | None = None

    def _make(self, name, faction, cog, aff, home) -> Agent:
        cog = copy.deepcopy(cog)
        C = cog.C0.copy()
        C[0] = self.rng.uniform(-1.5, 0.5)            # stagger hunger so arrivals vary
        npc = NPC(cog, C=C)
        pos = home + self.rng.normal(0, 3.0, 2)
        ltm = SequenceMemory(copy.deepcopy(self.cfg.ltm)) if self.cfg.ltm is not None else None
        return Agent(name, faction, npc, AffectState(copy.deepcopy(aff)), home, pos,
                     heading=self.rng.uniform(0, 2 * np.pi), ltm=ltm)

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
                v = steer(a.npc.coefficients(), a.npc.cone(), p, a.heading, self.steer_cfg)
                v = v * (0.5 + 0.5 * a.health)          # injured agents are slower
            else:
                v = np.zeros(2)
            new = np.clip(a.pos + v * dt, 0.0, w.size)
            a.vel = (new - a.pos) / dt
            a.pos = new
        self._combat(dt)
        self.t = (self.n_steps + 1) * dt
        self.n_steps += 1
        if cog_tick:
            self._log()

    def _think(self, i: int, a: Agent, p: coupling.Percept, h: float) -> None:
        npc = a.npc
        X = coupling.invariants(p, self.wcfg)
        beta, _ = coupling.omega_ext(X, npc.cfg)
        if a.ltm is not None:
            beta = beta + self._replay_bivector(a)
        eat = float(a.mobile and X[0] < npc.cfg.eat_radius)
        npc.dynamics_step(beta, h, eat=eat, company=X[2])
        npc.drive_residual(X, h)
        ok = lambda g: metric_ok(g, a.affect.cfg)  # noqa: E731
        npc.commit_frame(forget_frame(npc.L, npc.L0, npc.cfg), ok)
        a.affect.observe_injury(a.taken, a.dealt, h)
        a.taken = a.dealt = 0.0
        beta_total = beta + ga.bivector_from_planes(npc.cfg.omega_int)
        a.affect.step(npc.C, beta_total, h)
        if self.cfg.learning:
            npc.commit_frame(a.affect.learn(npc.L), ok)
        if a.ltm is not None:
            a.ltm.observe(sense_vector(self, a, X, p), npc.C, a.affect.pain, self.t,
                          affect=a.affect.cfg.weights * a.affect.a)
        if a.mobile and p.threat_vec is not None:
            self.decisions.append((self.t, i, self.allies_near(a), *map(float, npc.C)))

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
        for a in self.agents:
            if not a.in_contact:
                a.health = min(1.0, a.health + cc.recovery_rate * dt)
            a.health = max(a.health, 0.0)
            if a.mobile and a.health < cc.immobile_below:
                a.mobile = False
            elif not a.mobile and a.health > cc.mobile_above:
                a.mobile = True

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
        L = self.log
        L["t"].append(self.t)
        L["pos"].append(np.array([a.pos for a in self.agents]))
        L["C"].append(np.array([a.npc.C for a in self.agents]))
        L["health"].append(np.array([a.health for a in self.agents]))
        L["mobile"].append(np.array([a.mobile for a in self.agents]))
        L["g"].append(np.array([a.npc.g for a in self.agents]))
        L["pain"].append(np.array([a.affect.pain for a in self.agents]))
        L["delta"].append(np.array([a.affect.delta for a in self.agents]))
        L["affect"].append(np.array([a.affect.a for a in self.agents]))
        L["contact"].append(np.array([a.in_contact for a in self.agents]))
        L["replay"].append(np.array([a.replaying for a in self.agents]))

    def run(self, T: float) -> dict:
        for _ in range(int(round(T / self.wcfg.dt))):
            self.step()
        if self._episode is not None:
            self._close_episode()
        return {k: np.array(v) for k, v in self.log.items()}
