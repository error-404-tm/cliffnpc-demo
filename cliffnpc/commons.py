"""Commons: an environment for measuring induced geometry (spec-v2, phase B).

One controlled learner in a 100 x 100 world with scripted allies, rivals and
food.  Five environment factors, each chosen for the goal relation it should
induce (spec-v2 section 4):

    ally_skill     0: the forager goes to the nearest patch; 1: it knows the real ones
    threat         probability that reaching a patch brings an ambush (0 / 0.3 / 0.7)
    shelter        a disc around home rivals cannot enter (False / True)
    cue_validity   1: look A is real, look B false; 0: looks are random
    scarcity       0: rich (4 patches, big stock); 1: scarce (2 patches, small stock)

Goal progress per tick (0.2 s), a vector over GOALS:
    food       satiety gained (eating real food; satiety saturates and decays)
    safety     minus injury taken
    social     tick fraction with an ally within peer range
    knowledge  newly seen area (grid cells) + learning a patch's true type on first visit

The learner is driven by an exploratory controller: a weighted sum of six
PRIMITIVES (unit vectors) with episode weights theta and smooth OU noise xi on
the weights.  Every tick logs controls, noise, senses and goal progress; records
at several scales are cut from that log (record_windows).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arena import Agent, Arena, ArenaConfig
from .config import FoodConfig

GOALS = ("food", "safety", "social", "knowledge")
PRIMITIVES = ("food", "forager", "shelter", "flee", "fight", "wander")
N_SENSE = 24                         # 16 situation features + 8 commons senses
SCALES = (16, 32, 64)                # record lengths in ticks
TICK = 0.2                           # s per record tick (= cognitive tick)
CENTER = np.array([50.0, 50.0])
HOME = np.array([20.0, 50.0])


@dataclass
class Env:
    """One point in environment space."""
    ally_skill: int = 1
    threat: float = 0.3
    shelter: bool = True
    cue_validity: int = 1
    scarcity: int = 0

    def name(self) -> str:
        return (f"skill{self.ally_skill}_thr{self.threat}_shel{int(self.shelter)}"
                f"_cue{self.cue_validity}_scar{self.scarcity}")


CENTRE = Env()
SWEEP = {                            # one-factor-at-a-time around the centre (spec-v2 section 4)
    "ally_skill": [0, 1], "threat": [0.0, 0.3, 0.7], "shelter": [False, True],
    "cue_validity": [0, 1], "scarcity": [0, 1],
}


def environments() -> list[Env]:
    out, seen = [CENTRE], {CENTRE.name()}
    for k, levels in SWEEP.items():
        for v in levels:
            e = Env(**{**CENTRE.__dict__, k: v})
            if e.name() not in seen:
                out.append(e)
                seen.add(e.name())
    return out


# ------------------------------------------------------------------ record layout
def tick_dtype() -> np.dtype:
    """One tick of a record, padded to 256 bytes (a power of two)."""
    return np.dtype([("u", "<f4", (4,)),            # control: vx, vy, fight, pad
                     ("xi", "<f4", (8,)),           # exploration noise on the 6 primitive weights, pad
                     ("s", "<f4", (32,)),           # senses (N_SENSE used), pad
                     ("r", "<f4", (4,)),            # goal progress
                     ("pad", "<f4", (16,))])


def record_dtype(T: int) -> np.dtype:
    """A record: T ticks (T a power of two)."""
    assert T & (T - 1) == 0
    return np.dtype([("ticks", tick_dtype(), (T,))])


def record_windows(log: dict, T: int, stride: int | None = None) -> dict:
    """Cut a run log into windows of T ticks: per window the ticks' arrays and the
    summed goal progress (the record's vector outcome)."""
    stride = stride or T // 2
    n = len(log["r"])
    starts = np.arange(0, n - T + 1, stride)
    out = {k: np.stack([log[k][i:i + T] for i in starts]) for k in ("u", "xi", "s", "r", "theta")}
    out["G"] = out["r"].sum(axis=1)                 # vector outcome of the record
    out["episode"] = np.array([log["episode"][i] for i in starts])
    return out


# ------------------------------------------------------------------ the world
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(2)


@dataclass
class Commons:
    env: Env
    seed: int = 0
    episode_len: float = 60.0
    noise_sigma: float = 0.5           # OU noise on primitive weights
    noise_tau: float = 1.0
    theta_low: float = -0.5            # episode weights ~ U(low, high) per primitive
    theta_high: float = 1.5
    ar: Arena = field(init=False)
    log: dict = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed + 5003)
        cfg = ArenaConfig(seed=self.seed, tribes=True, n_npcs=1, home=tuple(HOME), lair=(95.0, 95.0),
                          food=((50.0, 50.0),),
                          food_dynamics=FoodConfig(initial_patches=0, max_patches=0, spawn_rate=0.0,
                                                   stock=1e9, false_fraction=0.0, false_lifetime=0.0,
                                                   remember_false=False))
        cfg.base.world.size = 100.0
        cfg.learning = False
        ar = self.ar = Arena(cfg)
        used = {x.slot for x in ar.agents}
        ar.capacity = 8                              # learner, two allies, ambushers
        ar._free = [i for i in range(ar.capacity) if i not in used]
        self.learner = ar.agents[0]
        for b in list(ar.agents[1:]):
            self._remove(b)
        self.learner.controller = self._control
        self.forager = self._add("A", HOME + np.array([3.0, 0.0]), self._forager_control)
        self.homebody = self._add("A", HOME + np.array([0.0, 3.0]), self._homebody_control)
        self.rivals: list[Agent] = []
        self.log = {k: [] for k in ("u", "xi", "s", "r", "theta", "episode", "t")}
        self.episode = -1
        self.theta = np.zeros(len(PRIMITIVES))
        self.xi = np.zeros(len(PRIMITIVES))
        self.heading = 0.0
        self.u = np.zeros(3)
        self.satiety = 0.0
        self.seen = np.zeros((10, 10), bool)
        self.visited: set = set()
        self.amb_patches: set = set()
        self.ambush_until = -1.0
        self.forager_target = None

    # -- agent management (as in gym)
    def _remove(self, a):
        if a in self.ar.agents:
            self.ar.agents.remove(a)
            self.ar._free.append(a.slot)
            self.ar._free.sort()

    def _add(self, faction, pos, ctrl):
        a = self.ar._make(faction, np.asarray(pos, float), pos=np.asarray(pos, float).copy())
        a.controller = ctrl
        self.ar.agents.append(a)
        return a

    # -- food
    def _patch_spec(self):
        rich = self.env.scarcity == 0
        return (4, 1e9) if rich else (2, 3.0)        # number of patches, stock (scarce: ~30 meal-s)

    def _spawn_patch(self):
        n, stock = self._patch_spec()
        pos = self.rng.uniform(10, 90, 2)
        real = self.rng.random() < 0.5
        if self.env.cue_validity:
            cue = 1 if real else 2                   # look A real, look B false
        else:
            cue = int(self.rng.integers(1, 3))
        self.ar._spawn(pos, stock, false=not real, cue=cue)

    def _keep_food(self):
        n, _ = self._patch_spec()
        while len(self.ar.patches) < n:
            self._spawn_patch()

    # -- scripted agents
    def _in_shelter(self, pos):
        return self.env.shelter and np.linalg.norm(pos - HOME) < 10.0

    def _forager_control(self, ar, me, p):
        me.npc.C[3] = 1.0
        pool = [q for q in ar.patches if (not q["false"]) or not self.env.ally_skill]
        if not pool:
            return np.zeros(2)
        tgt = min(pool, key=lambda q: np.linalg.norm(q["pos"] - me.pos))
        d = tgt["pos"] - me.pos
        return _unit(d) * min(4.0, np.linalg.norm(d) / ar.wcfg.dt)

    def _homebody_control(self, ar, me, p):
        me.npc.C[3] = 1.0
        d = HOME - me.pos
        return _unit(d) * min(3.0, max(0.0, np.linalg.norm(d) - 2.0))

    def _rival_control(self, ar, me, p):
        me.npc.C[3] = 1.0
        v = _unit(self.learner.pos - me.pos) * 4.0
        nxt = me.pos + v * ar.wcfg.dt
        if self._in_shelter(nxt):                    # rivals cannot enter the shelter
            return np.zeros(2)
        return v

    # -- the learner's exploratory controller
    def _primitive_vectors(self, ar, a):
        R = ar.wcfg.sensor_radius
        seen = [q for q in ar.patches if np.linalg.norm(q["pos"] - a.pos) <= R]
        food = _unit(min(seen, key=lambda q: np.linalg.norm(q["pos"] - a.pos))["pos"] - a.pos) if seen else np.zeros(2)
        foe = ar._rival_of(a)
        rival = _unit(foe.pos - a.pos) if foe is not None else np.zeros(2)
        return np.array([
            food,
            _unit(self.forager.pos - a.pos),
            _unit(HOME - a.pos),
            -rival,
            rival,
            np.array([np.cos(self.heading), np.sin(self.heading)]),
        ])

    def _control(self, ar, a, p):
        dt = ar.wcfg.dt
        k = np.exp(-dt / self.noise_tau)
        self.xi = k * self.xi + np.sqrt(1 - k * k) * self.noise_sigma * self.rng.normal(size=len(PRIMITIVES))
        self.heading += 0.6 * np.sqrt(dt) * self.rng.normal()
        w = self.theta + self.xi
        P = self._primitive_vectors(ar, a)
        v = w @ P
        speed = ar.steer_cfg.max_speed * np.tanh(np.linalg.norm(v))
        fight = float(w[4] > w[3])                    # fight flag: leaning in more than away
        a.npc.C[3] = 1.0 if fight else -1.0
        vel = speed * _unit(v)
        self.u = np.array([vel[0], vel[1], fight])
        return vel

    # -- senses and goal progress
    def senses(self) -> np.ndarray:
        a, ar = self.learner, self.ar
        x = ar.context_of(a, ar.percept(a))
        R = ar.wcfg.sensor_radius
        extra = np.array([
            max(0.0, 1 - np.linalg.norm(self.forager.pos - a.pos) / R),
            max(0.0, 1 - np.linalg.norm(HOME - a.pos) / R),
            float(self._in_shelter(a.pos)),
            self.satiety,
            a.health,
            float(len(self.rivals) > 0),
            float(self.env.shelter),
            0.0,
        ])
        return np.concatenate([x, extra])

    def _see(self):
        a = self.learner
        ij = np.indices((10, 10)).reshape(2, -1).T
        centres = ij * 10.0 + 5.0
        near = np.linalg.norm(centres - a.pos, axis=1) <= self.ar.wcfg.sensor_radius
        new = near & ~self.seen.reshape(-1)
        self.seen.reshape(-1)[near] = True
        return new.sum() / 100.0

    # -- episodes
    def new_episode(self):
        self.episode += 1
        for b in self.rivals:
            self._remove(b)
        self.rivals = []
        self.ar.patches.clear()
        self._keep_food()
        a = self.learner
        a.pos, a.vel, a.health, a.mobile = HOME.copy(), np.zeros(2), 1.0, True
        self.forager.pos = HOME + np.array([3.0, 0.0])
        self.homebody.pos = HOME + np.array([0.0, 3.0])
        self.satiety = float(self.rng.uniform(0.0, 0.5))
        self.seen[:] = False
        self.visited = set()
        self.amb_patches = set()
        self.theta = self.rng.uniform(self.theta_low, self.theta_high, len(PRIMITIVES))
        self.xi[:] = 0.0

    def run_episode(self):
        self.new_episode()
        ar, a = self.ar, self.learner
        steps_per_tick = int(round(TICK / ar.wcfg.dt))
        n_ticks = int(round(self.episode_len / TICK))
        r_eat = ar.cfg.npc_cog.eat_radius
        for _ in range(n_ticks):
            s = self.senses()
            xi0, u0 = self.xi.copy(), None
            taken0 = a.taken_total
            food = social = know = 0.0
            for _ in range(steps_per_tick):
                ar.step()
                u0 = self.u.copy() if u0 is None else u0
                if ar._eating(a):
                    gain = min(1.0 - self.satiety, 0.2 * ar.wcfg.dt)
                    self.satiety += gain
                    food += gain
                self.satiety = max(0.0, self.satiety - 0.01 * ar.wcfg.dt)
                for q in ar.patches:
                    if np.linalg.norm(a.pos - q["pos"]) < r_eat:
                        if q["id"] not in self.visited:
                            self.visited.add(q["id"])
                            know += 1.0                     # learned this patch's true type
                        if q["id"] not in self.amb_patches:
                            self.amb_patches.add(q["id"])
                            if self.rng.random() < self.env.threat:
                                self._ambush()
                if ar.t > self.ambush_until and self.rivals:
                    for b in self.rivals:
                        self._remove(b)
                    self.rivals = []
                self._keep_food()
            know += self._see()
            social = float(ar.allies_near(a) > 0) * TICK
            safety = -(a.taken_total - taken0)
            self._record(u0, xi0, s, np.array([food, safety, social, know]))
            if a.health < 0.15:                             # beaten: end the episode
                break

    def _ambush(self):
        th = self.rng.uniform(0, 2 * np.pi)
        for j in range(2):
            ang = th + np.pi * j
            pos = np.clip(self.learner.pos + 8.0 * np.array([np.cos(ang), np.sin(ang)]), 1, 99)
            self.rivals.append(self._add("B", pos, self._rival_control))
        self.ambush_until = self.ar.t + 8.0

    def _record(self, u, xi, s, r):
        L = self.log
        L["u"].append(u)
        L["xi"].append(xi)
        L["s"].append(s)
        L["r"].append(r)
        L["theta"].append(self.theta.copy())
        L["episode"].append(self.episode)
        L["t"].append(self.ar.t)

    def run(self, n_episodes: int) -> dict:
        for _ in range(n_episodes):
            self.run_episode()
        return {k: np.asarray(v) for k, v in self.log.items()}
