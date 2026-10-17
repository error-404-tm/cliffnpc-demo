"""Cognition -> steering readout (spec Phase 1 step 4).

This is the most arbitrary piece of the demo, so it is isolated here behind a
single function with the signature

    steer(coeffs, s, percept, heading, cfg) -> desired velocity (2,)

Swap it for anything else with the same signature.
"""

from __future__ import annotations

import numpy as np

from .config import SteeringConfig
from .coupling import Percept


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(2)


def reactivity(s: float, cfg: SteeringConfig) -> float:
    """0 (deliberative) .. 1 (reactive): a soft step at the light cone.
    ``s`` is the normalized interval s/|C|^2."""
    return float(1.0 / (1.0 + np.exp(s / cfg.reactive_scale)))


def steer(coeffs: np.ndarray, s: float, p: Percept, heading: float,
          cfg: SteeringConfig, signature=(1.0, 1.0, 1.0, -1.0)) -> np.ndarray:
    """In the reactive regime (timelike state) the timelike drives take over:
    they are boosted by (1 + reactive_boost * rho) and the spacelike ones damped
    by (1 - 0.8 rho).  With the default signature only threat is timelike; with
    hunger timelike too, a starving agent forages urgently instead of calmly."""
    hunger, social, explore, threat = coeffs
    rho = reactivity(s, cfg)
    v = np.zeros(2)
    timelike = np.asarray(signature) < 0
    k = np.where(timelike, 1.0 + cfg.reactive_boost * rho, 1.0 - 0.8 * rho)

    if hunger > 0:
        # head for food in view, else for the remembered patch
        target = p.food_vec if p.food_vec is not None else p.food_memory
        if target is not None:
            v += k[0] * hunger * _unit(target)
        else:
            # hungry and no food known: search
            v += k[0] * hunger * np.array([np.cos(heading), np.sin(heading)])
    if len(p.peer_vecs):
        centroid = p.peer_vecs.mean(axis=0)
        v += k[1] * social * _unit(centroid)  # social<0 pushes away
    if explore > 0:
        v += k[2] * explore * np.array([np.cos(heading), np.sin(heading)])
    else:
        v += k[2] * (-explore) * _unit(p.home_vec)

    # Threat: + approach (fight), - flee (flight).
    if p.threat_vec is not None:
        w = abs(threat) * k[3]
        v += w * np.sign(threat) * _unit(p.threat_vec)

    speed = cfg.max_speed * np.tanh(cfg.urgency * np.linalg.norm(v))
    return speed * _unit(v)
