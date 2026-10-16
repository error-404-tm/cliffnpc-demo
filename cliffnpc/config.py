"""All tunable constants in one place (spec §7: expose every constant)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CognitionConfig:
    # --- dynamics (§3) ---
    # Intrinsic temperament bivector Omega_int, as {(i, j): coefficient of e_i^e_j}.
    omega_int: dict = field(default_factory=lambda: {(0, 2): 0.02})
    # Baseline state C0 the damping pulls toward (slightly sated, slightly homey).
    C0: np.ndarray = field(default_factory=lambda: np.array([-0.3, 0.1, -0.2, -0.05]))
    # Per-drive damping rates gamma_i.  Threat is damped more slowly than
    # explore ("fear lingers").  This anisotropy is what lets a boost carry the
    # state across the light cone; boosts alone conserve s (README, E2).
    gamma: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.3, 0.5, 0.04]))
    hunger_drift: float = 0.05          # k in dC^1 += k dt
    eat_rate: float = 1.0               # while at food: dC^1 -= eat_rate (C^1 - sated_level) dt
    sated_level: float = -2.0
    eat_radius: float = 2.0             # "at food" means within this distance
    max_rapidity: float = 0.25          # boost rapidity cap per cognitive step
    null_tol: float = 0.05              # |s|/|C|^2 below this counts as "null"
    state_cap: float = 6.0              # hard safety cap on |C| (NaN guard)

    # --- coupling channels (§4) ---
    food_gain: float = 0.6              # rotation rate in e13 at the food patch
    food_length: float = 3.0            # food proximity kernel length (world units)
    threat_gain: float = 0.08           # boost rate per unit closing speed
    threat_deadband: float = 0.5        # closing speeds below this are ignored
    threat_e14_share: float = 0.15      # weight of e14 relative to e34
    peer_gain: float = 0.25             # rotation rate in e23 per unit peer density

    # --- prediction & metric learning (§5) ---
    expect_rate: float = 0.2            # eta in X_hat <- exp(eta log R) X_hat
    predictor: str = "ema"              # "ema" (persistence) or "cv" (constant velocity)
    learn_rate: float = 0.004           # epsilon for L <- exp(eps A(r)) L
    frustration_gain: float = 1.0       # learning scale = 1 + gain * |C^B|
    forget_rate: float = 0.002          # lambda, decay of g toward g0
    max_frame_step: float = 0.05        # cap on ||A|| per update (Frobenius)
    sensitivity_cap: float = 5.0
    use_sensitivity: bool = True        # weight drive i by |g^ii| (§3.4); ablation flag
    surprise_threshold: float = 0.0     # tier-2 gating: skip metric update below this
    felt_residual: bool = True          # lower r with the current g before learning (ablation flag)
    # Invariant residual -> drive residual map (rows: drives, cols: invariants
    # food_dist, threat_closing, peer_density, novelty).  Scaled so a typical
    # surprise is O(1).
    residual_map: np.ndarray = field(default_factory=lambda: np.array([
        [0.15, 0.0, 0.0, 0.0],   # food distance  -> hunger
        [0.0, 0.0, 1.0, 0.0],    # peer density   -> social
        [0.0, 0.0, 0.0, 0.1],    # novelty        -> explore
        [0.0, 0.6, 0.0, 0.0],    # threat closing -> threat
    ]))

    # --- holonomy (§4.1) ---
    holonomy_plane: tuple = (1, 2)      # haunted disc bivector B_h = e23


@dataclass
class SteeringConfig:
    max_speed: float = 5.0
    urgency: float = 2.0                # speed = max_speed * tanh(urgency |v_desired|)
    reactive_boost: float = 3.0         # extra threat weight when s < 0
    reactive_scale: float = 0.1         # softness of the regime switch
    wander_turn: float = 0.6            # heading noise (rad/sqrt(s))


@dataclass
class WorldConfig:
    size: float = 100.0
    dt: float = 0.1                     # physics step
    cog_every: int = 2                  # cognition runs every n physics steps
    surprise_gate: float = 0.0          # skip a cognitive tick if |predicted residual| < this (0 = off)
    max_skip: int = 4                   # never skip more than this many ticks in a row
    sensor_radius: float = 30.0
    home: tuple = (20.0, 20.0)
    food: tuple = ((65.0, 60.0),)
    peers: tuple = ((22.0, 52.0), (27.0, 55.0), (19.0, 57.0))
    peer_radius: float = 15.0
    threat_start: tuple = (85.0, 15.0)
    threat_speed: float = 4.5
    threat_period: float = 120.0        # an approach starts every period
    threat_attack_len: float = 15.0     # seconds spent chasing the NPC
    haunt_center: tuple = (45.0, 40.0)
    haunt_radius: float = 8.0
    haunt_flux: float = 0.004           # Phi: holonomy angle per unit area
    seed: int = 0


@dataclass
class Config:
    cog: CognitionConfig = field(default_factory=CognitionConfig)
    steer: SteeringConfig = field(default_factory=SteeringConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
