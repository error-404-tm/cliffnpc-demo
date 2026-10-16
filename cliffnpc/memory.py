"""Memory tiers / cognitive LOD (§5.3): episodes, place rotors, consolidation,
collective metric, promotion/demotion, gossip as metric diffusion.

Tiers
  1 EPISODIC   own L, k episode slots (surprise-weighted reservoir), m place
               rotors.  L is *not* learned online: experiences reach L when an
               episode leaves the reservoir (rejected or evicted) or when the
               NPC is demoted.  Lossy, never amnesiac.
  2 METRIC     own L only, learned online and gated by a surprise threshold.
  3 BACKGROUND no own L: reads the collective (town/faction) metric, and its
               residuals feed the collective in aggregate.

Pure NumPy, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from . import algebra as ga
from .cognition import (NPC, apply_frame_update, forget_frame, learn_frame,
                        saturate_frame, toward_metric_generator)
from .config import CognitionConfig

EPISODIC, METRIC, BACKGROUND = 1, 2, 3


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    t: float
    C: np.ndarray           # state (grade-1 part)
    r: np.ndarray           # residual (what was surprising)
    pos: np.ndarray         # where (world)
    frustration: float
    surprise: float         # |r|, the reservoir weight


@dataclass
class EpisodicMemory:
    """k slots, filled by weighted reservoir sampling (Efraimidis-Spirakis A-Res).

    Each offered episode gets key u^(1/w) with u ~ U(0,1), w = surprise; the k
    largest keys are kept.  ``offer`` returns the episodes that leave memory
    (the offered one if rejected, or the evicted one), which the caller folds
    into L, so nothing is lost without trace.
    """
    k: int
    rng: np.random.Generator
    slots: list = field(default_factory=list)   # (key, Episode)

    def offer(self, ep: Episode) -> list[Episode]:
        key = self.rng.random() ** (1.0 / max(ep.surprise, 1e-9))
        if len(self.slots) < self.k:
            self.slots.append((key, ep))
            return []
        j = min(range(len(self.slots)), key=lambda i: self.slots[i][0])
        if key > self.slots[j][0]:
            out = self.slots[j][1]
            self.slots[j] = (key, ep)
            return [out]
        return [ep]

    def drain(self) -> list[Episode]:
        eps = [e for _, e in self.slots]
        self.slots = []
        return eps

    def episodes(self) -> list[Episode]:
        return sorted((e for _, e in self.slots), key=lambda e: e.t)


def fold_episodes(L: np.ndarray, episodes: list[Episode], cfg: CognitionConfig,
                  rate: float | None = None) -> np.ndarray:
    """Consolidate episodes into the frame, in chronological order."""
    for ep in sorted(episodes, key=lambda e: e.t):
        L = learn_frame(L, ep.r, ep.frustration, cfg, rate)
    return L


# ---------------------------------------------------------------------------
# Place rotors
# ---------------------------------------------------------------------------

@dataclass
class PlaceMemory:
    """m place slots on a grid, each holding the accumulated holonomy angle
    of the (single) haunted bivector B_h experienced in that cell.

    With one fixed B_h a place rotor is exp(alpha B_h), so one number per slot
    suffices; with several connection bivectors this would be a full element
    of the even subalgebra (8 numbers).  Slots are evicted least-recently-used.
    ``recall`` returns the stored angle so re-entering a place can replay a
    fraction of its rotation (memory of place, the persistence E4 lacks).
    """
    m: int
    cell: float = 10.0
    slots: dict = field(default_factory=dict)   # cell -> [alpha, last_t]

    def key(self, pos: np.ndarray) -> tuple:
        return tuple((np.asarray(pos) // self.cell).astype(int))

    def accumulate(self, pos: np.ndarray, alpha: float, t: float) -> None:
        if alpha == 0.0:
            return
        k = self.key(pos)
        if k not in self.slots and len(self.slots) >= self.m:
            oldest = min(self.slots, key=lambda q: self.slots[q][1])
            del self.slots[oldest]
        a, _ = self.slots.get(k, [0.0, t])
        self.slots[k] = [a + alpha, t]

    def recall(self, pos: np.ndarray, t: float) -> float:
        k = self.key(pos)
        if k in self.slots:
            self.slots[k][1] = t
            return self.slots[k][0]
        return 0.0


# ---------------------------------------------------------------------------
# Collective metric
# ---------------------------------------------------------------------------

@dataclass
class CollectiveMetric:
    """One frame per settlement/faction, shared by all background NPCs."""
    L: np.ndarray
    L0: np.ndarray
    cfg: CognitionConfig
    agg_rate: float = 0.5        # learning-rate multiplier for aggregate events

    @property
    def g(self) -> np.ndarray:
        return ga.metric_from_frame(self.L)

    def aggregate(self, residuals: list[np.ndarray], frustrations: list[float]) -> None:
        """Update from the mean residual of its background members this tick."""
        if not residuals:
            return
        r = np.mean(residuals, axis=0)
        if np.linalg.norm(r) > self.cfg.surprise_threshold:
            self._commit(learn_frame(self.L, r, float(np.mean(frustrations)), self.cfg,
                                     self.cfg.learn_rate * self.agg_rate))
        self._commit(forget_frame(self.L, self.L0, self.cfg))

    def _commit(self, L_raw: np.ndarray) -> None:
        h = self.cfg.frame_horizon
        self.L = saturate_frame(self.L, L_raw, h) if h else L_raw

    def absorb(self, g_i: np.ndarray, weight: float) -> None:
        """Fold an individual metric in with a small weight (on demotion)."""
        self._commit(apply_frame_update(self.L, toward_metric_generator(self.L, g_i), weight, 1.0))


# ---------------------------------------------------------------------------
# Tiered NPC wrapper
# ---------------------------------------------------------------------------

@dataclass
class TieredNPC:
    npc: NPC
    tier: int
    collective: CollectiveMetric
    rng: np.random.Generator
    episodes: EpisodicMemory | None = None
    places: PlaceMemory | None = None
    k_episodes: int = 8
    m_places: int = 6
    episode_threshold: float = 0.3
    metric_threshold: float = 0.3
    place_recall: float = 0.02   # fraction of a place rotor replayed per tick there
    folds: int = 0               # number of episodes consolidated so far

    def __post_init__(self):
        if self.tier == EPISODIC:
            self._start_episodic()
        if self.tier == BACKGROUND:
            self.npc.L = self.collective.L.copy()

    def _start_episodic(self):
        self.episodes = self.episodes or EpisodicMemory(self.k_episodes, self.rng)
        self.places = self.places or PlaceMemory(self.m_places)

    @property
    def g(self) -> np.ndarray:
        return self.collective.g if self.tier == BACKGROUND else self.npc.g

    # --- per tick ---------------------------------------------------------
    def tick(self, beta: np.ndarray, X: np.ndarray, h: float, eat: float, t: float,
             pos: np.ndarray, alpha: float) -> np.ndarray:
        npc = self.npc
        if self.tier == BACKGROUND:
            npc.L = self.collective.L          # read the shared metric
        npc.hol_accum = alpha
        r = npc.tick(beta, X, h, learn=False, eat=eat, forget=False)
        fr = npc.last.get("frustration", 0.0)
        surprise = float(np.linalg.norm(r))
        if self.tier == EPISODIC:
            if surprise > self.episode_threshold:
                out = self.episodes.offer(Episode(t, npc.C.copy(), r.copy(), pos.copy(), fr, surprise))
                if out:
                    self._fold(out)
            self.places.accumulate(pos, alpha, t)
            stored = self.places.recall(pos, t)
            if stored:
                npc.holonomy_transport(self.place_recall * stored)
            npc.commit_frame(forget_frame(npc.L, npc.L0, npc.cfg))
        elif self.tier == METRIC:
            if surprise > self.metric_threshold:
                npc.commit_frame(learn_frame(npc.L, r, fr, npc.cfg))
            npc.commit_frame(forget_frame(npc.L, npc.L0, npc.cfg))
        return r

    def _fold(self, episodes: list[Episode]) -> None:
        """Consolidate episodes chronologically, each step through the frame law."""
        for ep in sorted(episodes, key=lambda e: e.t):
            self.npc.commit_frame(learn_frame(self.npc.L, ep.r, ep.frustration, self.npc.cfg))
        self.folds += len(episodes)

    # --- LOD transitions --------------------------------------------------
    def promote(self, jitter: float = 0.005) -> None:
        if self.tier == BACKGROUND:
            # Instantiate from the *current* collective metric (recent town
            # events included), with a small personal perturbation.
            X = jitter * self.rng.normal(size=(4, 4))
            _, X_sym = ga.split_eta_generator(X)
            self.npc.L = self.collective.L @ expm(X_sym)
            self.npc.L0 = self.npc.L.copy()
            self.tier = METRIC
        elif self.tier == METRIC:
            self.tier = EPISODIC                 # past stays metric-only
            self._start_episodic()

    def demote(self, n_population: int) -> None:
        if self.tier == EPISODIC:
            self._fold(self.episodes.drain())
            self.places = None
            self.tier = METRIC
        elif self.tier == METRIC:
            self.collective.absorb(self.npc.g, 1.0 / max(n_population, 1))
            self.tier = BACKGROUND
            self.npc.L = self.collective.L.copy()


# ---------------------------------------------------------------------------
# Gossip
# ---------------------------------------------------------------------------

def gossip_step(agents: list[TieredNPC], adjacency: np.ndarray, eps: float,
                collective: CollectiveMetric) -> None:
    """Metric diffusion on the social graph (§5.3).

    Spec form: L_i <- exp(eps sum_j w_ij log(L_j L_i^{-1})) L_i.  Implemented
    with the gauge-invariant first-order generator toward_metric_generator(L_i,
    g_j) = eta (L_i^{-T} g_j L_i^{-1} - eta)/2 in place of log(L_j L_i^{-1}):
    it is the eta-self-adjoint (metric-changing) part of that log to first
    order, it doesn't depend on the arbitrary Lorentz gauge of L_j, and it
    needs no logm.  Background agents talk through the collective metric:
    their incoming diffusion is averaged into it.
    """
    gs = [a.g for a in agents]
    deg = adjacency.sum(axis=1)
    new_L = {}
    bg_targets = []
    for i, a in enumerate(agents):
        nbrs = np.nonzero(adjacency[i])[0]
        if len(nbrs) == 0:
            continue
        if a.tier == BACKGROUND:
            bg_targets.extend(gs[j] for j in nbrs if agents[j].tier != BACKGROUND)
            continue
        A = sum(toward_metric_generator(a.npc.L, gs[j]) for j in nbrs) / deg[i]
        new_L[i] = apply_frame_update(a.npc.L, A, eps, 0.2)
    for i, L in new_L.items():
        agents[i].npc.commit_frame(L)
    if bg_targets:
        # Average over all background agents' links; links between two
        # background agents carry no difference (same metric), so they only
        # dilute the pull of the individuals.
        bg_deg = sum(deg[i] for i, a in enumerate(agents) if a.tier == BACKGROUND)
        A = sum(toward_metric_generator(collective.L, g) for g in bg_targets) / bg_deg
        collective._commit(apply_frame_update(collective.L, A, eps, 0.2))
