"""2D world: food, home, static peers, a periodically attacking threat, and a
haunted disc that supplies a cognitive connection (§4.1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import WorldConfig


# ---------------------------------------------------------------------------
# Haunted disc: connection A = Phi B_h * a(x), with flux confined to the disc
# ---------------------------------------------------------------------------

def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class HauntedDisc:
    """Uniform curvature Phi (of a fixed bivector B_h) inside a disc of radius R.

    Gauge choice (symmetric/vortex gauge, relative to the centre c):
        inside:  a(x) = Phi/2 (-y, x)                 curl a = Phi
        outside: a(x) = Phi R^2 / (2 r^2) (-y, x)      curl a = 0
    a is continuous at r = R.  The line integral along a straight segment is
    computed exactly: Phi/2 (p x q) for the part inside (signed swept area)
    and Phi R^2/2 * (change in polar angle) outside.  Consequences:
      * a loop entirely outside that does not enclose the disc has holonomy
        exactly the identity (zero winding);
      * a loop enclosing the disc picks up exp(Phi pi R^2 B_h) wherever it
        runs (Aharonov-Bohm-like: memory of having circled the place);
      * a small loop inside picks up exp(Phi * area * B_h).
    """
    center: np.ndarray
    radius: float
    flux: float

    def segment_integral(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.asarray(p, float) - self.center
        q = np.asarray(q, float) - self.center
        d = q - p
        # Split the segment at its intersections with the circle |p + t d| = R.
        ts = [0.0, 1.0]
        a2 = d @ d
        if a2 > 0:
            b = 2 * p @ d
            c = p @ p - self.radius ** 2
            disc = b * b - 4 * a2 * c
            if disc > 0:
                sq = np.sqrt(disc)
                for t in ((-b - sq) / (2 * a2), (-b + sq) / (2 * a2)):
                    if 0.0 < t < 1.0:
                        ts.append(t)
        ts.sort()
        total = 0.0
        R2 = self.radius ** 2
        for t0, t1 in zip(ts[:-1], ts[1:]):
            x0, x1 = p + t0 * d, p + t1 * d
            mid = 0.5 * (x0 + x1)
            if mid @ mid <= R2:
                total += 0.5 * self.flux * (x0[0] * x1[1] - x0[1] * x1[0])
            else:
                dth = _wrap(np.arctan2(x1[1], x1[0]) - np.arctan2(x0[1], x0[0]))
                total += 0.5 * self.flux * R2 * dth
        return total

    def contains(self, x: np.ndarray) -> bool:
        return float(np.linalg.norm(np.asarray(x) - self.center)) <= self.radius


# ---------------------------------------------------------------------------
# Threat agent
# ---------------------------------------------------------------------------

@dataclass
class Threat:
    """Wanders near its lair; periodically chases the NPC until it makes
    contact (or the attack window ends), then retreats to the lair."""
    pos: np.ndarray
    speed: float
    period: float
    attack_len: float
    lair: np.ndarray
    # Optional scripted attack windows [(t_start, t_end), ...]; overrides period.
    schedule: list | None = None
    # Extra one-off attack windows (e.g. triggered from the viewer); always honoured.
    forced: list = field(default_factory=list)
    active: bool = True
    contact_radius: float = 1.5
    wander_sigma: float = 3.0        # spread of idle wandering around the lair
    _done_window: float = -1.0       # start time of the window that ended in contact
    _goal: np.ndarray | None = None

    def window(self, t: float) -> float | None:
        """Start time of the attack window containing t, or None."""
        if not self.active:
            return None
        for a, b in self.forced:
            if a <= t < b:
                return a
        if self.schedule is not None:
            for a, b in self.schedule:
                if a <= t < b:
                    return a
            return None
        start = (t // self.period) * self.period + self.period - self.attack_len
        return start if t >= start else None

    def force_attack(self, t: float, length: float | None = None) -> None:
        """Start an attack now, independent of the periodic schedule."""
        self.forced.append((t, t + (self.attack_len if length is None else length)))

    def attacking(self, t: float) -> bool:
        w = self.window(t)
        return w is not None and w != self._done_window

    def step(self, t: float, dt: float, target: np.ndarray, rng: np.random.Generator) -> None:
        if not self.active:
            return
        if self.attacking(t):
            goal, speed = target, self.speed
            if np.linalg.norm(target - self.pos) < self.contact_radius:
                self._done_window = self.window(t)
        else:
            if self._goal is None or np.linalg.norm(self._goal - self.pos) < 0.5:
                self._goal = self.lair + self.wander_sigma * rng.normal(0, 1.0, 2)
            goal, speed = self._goal, 0.6 * self.speed
        d = goal - self.pos
        n = np.linalg.norm(d)
        if n > 1e-9:
            self.pos = self.pos + min(speed * dt, n) * d / n


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

@dataclass
class World:
    cfg: WorldConfig
    t: float = 0.0
    n_steps: int = 0
    home: np.ndarray = field(init=False)
    food: list = field(init=False)
    peers: np.ndarray = field(init=False)
    threat: Threat = field(init=False)
    haunt: HauntedDisc = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        c = self.cfg
        self.rng = np.random.default_rng(c.seed)
        self.home = np.array(c.home, float)
        self.food = [np.array(f, float) for f in c.food]
        self.peers = np.array(c.peers, float).reshape(-1, 2)
        start = np.array(c.threat_start, float)
        self.threat = Threat(pos=start.copy(), speed=c.threat_speed, period=c.threat_period,
                             attack_len=c.threat_attack_len, lair=start.copy())
        self.haunt = HauntedDisc(np.array(c.haunt_center, float), c.haunt_radius, c.haunt_flux)

    def step(self, npc_pos: np.ndarray) -> None:
        self.threat.step(self.t, self.cfg.dt, npc_pos, self.rng)
        self.n_steps += 1
        self.t = self.n_steps * self.cfg.dt   # no float accumulation drift

    def clamp(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, self.cfg.size)
