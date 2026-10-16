"""Phase 3: a village of tiered NPCs sharing one world, one collective metric
and a social graph for gossip."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from . import algebra as ga
from . import coupling
from .cognition import NPC
from .config import Config
from .memory import BACKGROUND, EPISODIC, METRIC, CollectiveMetric, TieredNPC, gossip_step
from .steering import steer
from .world import World


def small_world_graph(n: int, k: int, n_extra: int, rng: np.random.Generator) -> np.ndarray:
    """Ring lattice (k neighbours each side) plus a few random long links."""
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for d in range(1, k + 1):
            A[i, (i + d) % n] = A[(i + d) % n, i] = 1
    for _ in range(n_extra):
        i, j = rng.choice(n, 2, replace=False)
        A[i, j] = A[j, i] = 1
    return A


def graph_distances(A: np.ndarray, sources: list[int]) -> np.ndarray:
    n = len(A)
    dist = np.full(n, np.inf)
    frontier = list(sources)
    for s in sources:
        dist[s] = 0
    while frontier:
        nxt = []
        for i in frontier:
            for j in np.nonzero(A[i])[0]:
                if dist[j] == np.inf:
                    dist[j] = dist[i] + 1
                    nxt.append(j)
        frontier = nxt
    return dist


@dataclass
class VillageConfig:
    n: int = 30
    n_episodic: int = 3
    n_metric: int = 9
    ring_radius: float = 35.0
    food_offset: float = 8.0       # each villager's own food patch, radially outward
    sensor_radius: float = 12.0
    center: tuple = (50.0, 50.0)
    graph_k: int = 2
    graph_extra: int = 3
    gossip_eps: float = 0.01
    gossip_every: int = 5          # cognitive ticks between gossip rounds
    personality_spread: float = 0.05
    seed: int = 1


class Village:
    def __init__(self, cfg: Config | None = None, vcfg: VillageConfig | None = None):
        self.cfg = cfg or Config()
        self.vcfg = vc = vcfg or VillageConfig()
        self.rng = np.random.default_rng(vc.seed)
        self.cfg.world.sensor_radius = vc.sensor_radius
        self.world = World(self.cfg.world)
        self.world.haunt.flux = 0.0
        self.world.threat.active = False
        self.collective = CollectiveMetric(np.eye(4), np.eye(4), self.cfg.cog)
        ang = 2 * np.pi * np.arange(vc.n) / vc.n
        c = np.array(vc.center)
        radial = np.stack([np.cos(ang), np.sin(ang)], 1)
        self.homes = c + vc.ring_radius * radial
        self.radial = radial
        self.world.food = list(self.homes + vc.food_offset * radial)
        self.pos = self.homes.copy()
        self.vel = np.zeros_like(self.pos)
        self.heading = self.rng.uniform(0, 2 * np.pi, vc.n)
        self.agents: list[TieredNPC] = []
        for i in range(vc.n):
            tier = EPISODIC if i < vc.n_episodic else METRIC if i < vc.n_episodic + vc.n_metric else BACKGROUND
            X = vc.personality_spread * self.rng.normal(size=(4, 4))
            _, X_sym = ga.split_eta_generator(X)
            L = expm(X_sym)
            npc = NPC(self.cfg.cog, L=L.copy(), L0=L.copy())
            npc.C = npc.C + 0.1 * self.rng.normal(size=4)
            self.agents.append(TieredNPC(npc, tier, self.collective, self.rng))
        # interleave tiers around the ring so every tier is near the event
        order = self.rng.permutation(vc.n)
        self.agents = [self.agents[k] for k in order]
        self.adj = small_world_graph(vc.n, vc.graph_k, vc.graph_extra, self.rng)
        self._step = 0
        self._cog_tick = 0
        self._last_threat = self.world.threat.pos.copy()
        self.target = 0
        self.log: dict = {"t": [], "g": [], "C": [], "pos": [], "tier": [], "g_col": [], "threat": []}

    def step(self) -> None:
        w, cfg, vc = self.world, self.cfg, self.vcfg
        dt = cfg.world.dt
        cog = self._step % cfg.world.cog_every == 0
        h = dt * cfg.world.cog_every
        threat_vel = (w.threat.pos - self._last_threat) / dt
        self._last_threat = w.threat.pos.copy()
        bg_r, bg_fr = [], []
        new_pos = self.pos.copy()
        for i, a in enumerate(self.agents):
            w.peers = np.delete(self.pos, i, axis=0)
            w.home = self.homes[i]
            p = coupling.sense(w, self.pos[i], self.vel[i], threat_vel, cfg.world)
            X = coupling.invariants(p, cfg.world)
            if cog:
                beta, _ = coupling.omega_ext(X, cfg.cog)
                eat = float(X[0] < cfg.cog.eat_radius)
                r = a.tick(beta, X, h, eat, w.t, self.pos[i], 0.0)
                if a.tier == BACKGROUND:
                    bg_r.append(r)
                    bg_fr.append(a.npc.last.get("frustration", 0.0))
            self.heading[i] += cfg.steer.wander_turn * np.sqrt(dt) * self.rng.normal()
            v = steer(a.npc.coefficients(), a.npc.cone(), p, self.heading[i], cfg.steer)
            new_pos[i] = w.clamp(self.pos[i] + v * dt)
        if cog:
            self.collective.aggregate(bg_r, bg_fr)
            self._cog_tick += 1
            if self._cog_tick % vc.gossip_every == 0:
                gossip_step(self.agents, self.adj, vc.gossip_eps, self.collective)
            self._log()
        self.vel = (new_pos - self.pos) / dt
        self.pos = new_pos
        w.step(self.pos[self.target])
        self._step += 1

    def _log(self):
        L = self.log
        L["t"].append(self.world.t)
        L["g"].append(np.array([a.g for a in self.agents]))
        L["C"].append(np.array([a.npc.C for a in self.agents]))
        L["pos"].append(self.pos.copy())
        L["tier"].append([a.tier for a in self.agents])
        L["g_col"].append(self.collective.g)
        L["threat"].append(self.world.threat.pos.copy())

    def run(self, T: float, events: dict | None = None) -> dict:
        """events: {time: callable(village)} fired once when t passes them."""
        events = dict(events or {})
        n = int(round(T / self.cfg.world.dt))
        for _ in range(n):
            for te in sorted(list(events)):
                if self.world.t >= te:
                    events.pop(te)(self)
            self.step()
        return {k: np.array(v) for k, v in self.log.items()}
