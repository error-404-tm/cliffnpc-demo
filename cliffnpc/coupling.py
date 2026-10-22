"""World -> cognition coupling: Omega_ext = L_c o I o P_S (world)   (§4).

* P_S (sensor): a disc of radius ``sensor_radius`` around the NPC.  Anything
  outside is rejected, i.e. not perceived.
* I (invariants): distances, a closing speed and a count; all invariant under
  rigid motions of the world, so cognition never depends on world coordinates.
* L_c (low-rank coupling): Omega_ext = sum_i f_i(invariants) B_i over r
  designer-chosen channels.  ``CHANNELS`` lists them; r = len(CHANNELS) is the
  explicit budget knob (``active_channels`` lets callers drop some).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import algebra as ga
from .config import CognitionConfig, WorldConfig
from .world import World


@dataclass
class Percept:
    """Projection of the world onto the sensor (the P_S step)."""
    food_vec: np.ndarray | None      # nearest food, relative position
    threat_vec: np.ndarray | None    # threat, relative position
    threat_vel: np.ndarray           # threat velocity relative to the NPC
    peer_vecs: np.ndarray            # relative positions of perceived peers
    home_vec: np.ndarray
    food_memory: np.ndarray | None = None  # remembered nearest food (steering only)
    danger_vec: np.ndarray | None = None   # nearest marked place in range (steering only)


def sense(world: World, pos: np.ndarray, vel: np.ndarray, threat_vel: np.ndarray,
          wcfg: WorldConfig) -> Percept:
    R = wcfg.sensor_radius
    foods = [f - pos for f in world.food]
    foods = [f for f in foods if np.linalg.norm(f) <= R]
    food_vec = min(foods, key=np.linalg.norm) if foods else None
    tv = world.threat.pos - pos
    threat_vec = tv if (world.threat.active and np.linalg.norm(tv) <= R) else None
    peer_vecs = world.peers - pos
    peer_vecs = peer_vecs[np.linalg.norm(peer_vecs, axis=1) <= R]
    known = [f - pos for f in world.food]
    food_memory = min(known, key=np.linalg.norm) if known else None
    return Percept(food_vec, threat_vec, threat_vel - vel, peer_vecs, world.home - pos,
                   food_memory)


def invariants(p: Percept, wcfg: WorldConfig) -> np.ndarray:
    """(food distance, threat closing speed, peer density, novelty)."""
    R = wcfg.sensor_radius
    food_dist = float(np.linalg.norm(p.food_vec)) if p.food_vec is not None else R
    if p.threat_vec is not None:
        d = np.linalg.norm(p.threat_vec)
        closing = float(-(p.threat_vec @ p.threat_vel) / max(d, 1e-6))
    else:
        closing = 0.0
    if len(p.peer_vecs):
        dists = np.linalg.norm(p.peer_vecs, axis=1)
        density = float(np.sum(np.clip(1.0 - dists / wcfg.peer_radius, 0.0, None)))
    else:
        density = 0.0
    novelty = float(np.linalg.norm(p.home_vec))  # distance from home
    return np.array([food_dist, closing, density, novelty])


# Channel table: (name, plane (i, j), activation f(invariants, cfg)).
def _food(X, c: CognitionConfig):
    return c.food_gain * np.exp(-X[0] / c.food_length)


def _threat34(X, c: CognitionConfig):
    return c.threat_gain * max(X[1] - c.threat_deadband, 0.0)


def _threat14(X, c: CognitionConfig):
    return c.threat_gain * c.threat_e14_share * max(X[1] - c.threat_deadband, 0.0)


def _peers(X, c: CognitionConfig):
    return c.peer_gain * X[2]


CHANNELS = [
    # food proximity rotates hunger -> explore -> sated -> home (e13)
    ("food", (0, 2), _food),
    # threat closing boosts e34 with generator M(e4, e3): explorers flee,
    # homebodies fight (the eigen-direction picked depends on sign of C^3)
    ("threat_e34", (3, 2), _threat34),
    # threat closing boosts e14 with M(e1, e4): starving -> fight, sated -> flight
    ("threat_e14", (0, 3), _threat14),
    # peer density rotates lonely -> explore (group wanderlust) in e23
    ("peers", (1, 2), _peers),
]


def omega_ext(X: np.ndarray, cfg: CognitionConfig,
              active_channels: set[str] | None = None) -> tuple[np.ndarray, dict]:
    """Bivector coefficient array beta of Omega_ext, plus per-channel activations."""
    coeffs, acts = {}, {}
    for name, plane, f in CHANNELS:
        if active_channels is not None and name not in active_channels:
            continue
        a = float(f(X, cfg))
        acts[name] = a
        coeffs[plane] = coeffs.get(plane, 0.0) + a
    return ga.bivector_from_planes(coeffs), acts
