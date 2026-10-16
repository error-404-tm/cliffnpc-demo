"""Main loop for a single NPC in the 2D world, with per-tick logging.

Physics runs at ``world.dt``; cognition runs every ``cog_every`` physics steps
on invariants averaged over the interval (multirate, §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import algebra as ga
from . import coupling
from .cognition import NPC
from .config import Config
from .steering import steer
from .world import World


@dataclass
class Agent:
    npc: NPC
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0


@dataclass
class Log:
    rows: dict = field(default_factory=dict)

    def add(self, **kw):
        for k, v in kw.items():
            self.rows.setdefault(k, []).append(np.array(v, dtype=float) if not isinstance(v, str) else v)

    def arr(self, k: str) -> np.ndarray:
        return np.array(self.rows[k])

    def __getitem__(self, k):
        return self.arr(k)


class Simulation:
    def __init__(self, cfg: Config | None = None, npc: NPC | None = None,
                 start: tuple | None = None,
                 path: Callable[[float], np.ndarray] | None = None,
                 active_channels: set[str] | None = None,
                 learn: bool = True,
                 world_hook: Callable[["Simulation"], None] | None = None):
        """``path``: optional scripted trajectory t -> position (overrides steering).
        ``world_hook``: called every physics step before sensing (scripted events)."""
        self.cfg = cfg or Config()
        self.world = World(self.cfg.world)
        npc = npc or NPC(self.cfg.cog)
        start = np.array(start if start is not None else self.cfg.world.home, float)
        self.agent = Agent(npc, start.copy())
        self.path = path
        self.active_channels = active_channels
        self.learn = learn
        self.world_hook = world_hook
        self.log = Log()
        self._X_acc = np.zeros(4)
        self._n_acc = 0
        self._step = 0
        self._last_threat = self.world.threat.pos.copy()
        self._beta = np.zeros((4, 4))
        self._acts: dict = {}
        self._h_pending = 0.0
        self._skipped = 0
        self.cog_slots = 0                # cognitive ticks scheduled by the multirate clock
        self.cog_ticks = 0                # cognitive ticks actually run (surprise gating)

    def step(self) -> None:
        w, a, cfg = self.world, self.agent, self.cfg
        dt = cfg.world.dt
        if self.world_hook:
            self.world_hook(self)
        threat_vel = (w.threat.pos - self._last_threat) / dt
        self._last_threat = w.threat.pos.copy()

        # 1. sense
        p = coupling.sense(w, a.pos, a.vel, threat_vel, cfg.world)
        X = coupling.invariants(p, cfg.world)
        self._X_acc += X
        self._n_acc += 1

        # cognitive tick (multirate, optionally surprise-gated)
        if self._step % cfg.world.cog_every == 0:
            self.cog_slots += 1
            self._h_pending += dt * cfg.world.cog_every
            Xbar = self._X_acc / self._n_acc
            gate = cfg.world.surprise_gate
            if gate > 0 and a.npc.X_hat is not None and self._skipped < cfg.world.max_skip:
                surprise = np.linalg.norm(cfg.cog.residual_map @ (Xbar - a.npc.X_hat))
                if surprise < gate:
                    self._skipped += 1
                    return self._act(p)
            self._skipped = 0
            self.cog_ticks += 1
            self._X_acc[:] = 0
            self._n_acc = 0
            h, self._h_pending = self._h_pending, 0.0
            # 2. Omega_ext
            self._beta, self._acts = coupling.omega_ext(Xbar, cfg.cog, self.active_channels)
            # 3, 5, 6. dynamics, prediction + metric update, holonomy
            eat = float(Xbar[0] < cfg.cog.eat_radius)
            a.npc.tick(self._beta, Xbar, h, learn=self.learn, eat=eat)
            self._log(Xbar)
        self._act(p)

    def _act(self, p) -> None:
        """Step 4 (act), physics, holonomy accumulation."""
        w, a, cfg = self.world, self.agent, self.cfg
        dt = cfg.world.dt
        a.heading += cfg.steer.wander_turn * np.sqrt(dt) * w.rng.normal()
        if self.path is not None:
            new = np.asarray(self.path(w.t + dt), float)
        else:
            v = steer(a.npc.coefficients(), a.npc.cone(), p, a.heading, cfg.steer)
            new = w.clamp(a.pos + v * dt)
        a.npc.hol_accum += w.haunt.segment_integral(a.pos, new)
        a.vel = (new - a.pos) / dt
        a.pos = new

        w.step(a.pos)
        self._step += 1

    def _log(self, X: np.ndarray) -> None:
        npc = self.agent.npc
        g = npc.g
        s = npc.interval()
        last = npc.last
        self.log.add(
            t=self.world.t, pos=self.agent.pos, threat_pos=self.world.threat.pos,
            threat_dist=np.linalg.norm(self.world.threat.pos - self.agent.pos),
            C=npc.coefficients(), proj=npc.projections(), s=s, cone=npc.cone(),
            regime=npc.regime(),
            frustration=last.get("frustration", 0.0),
            frustration_sq=last.get("frustration_sq", 0.0),
            rapidity=last.get("rapidity", 0.0),
            g_offdiag=g[np.triu_indices(4, 1)], g=g,
            detL=np.linalg.det(npc.L), X=X,
            residual=last.get("residual", np.zeros(4)),
            learned=float(last.get("learned", False)),
            holonomy=last.get("holonomy", 0.0),
            in_haunt=float(self.world.haunt.contains(self.agent.pos)),
            **{f"act_{k}": v for k, v in self._acts.items()},
        )

    def run(self, T: float) -> Log:
        n = int(round(T / self.cfg.world.dt))
        for _ in range(n):
            self.step()
        return self.log
