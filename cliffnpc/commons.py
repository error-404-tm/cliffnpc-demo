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
    social     tick fraction with an ally within peer range (the forager, or a
               companion who wanders the world)
    knowledge  bits resolved about patch types: each patch is one unknown bit,
               resolved by seeing its look (only if cues are valid) or by probing
               it at the patch (always)

Exploratory policies (POLICIES) draw episode weights theta = mean + U(-spread,
spread) per primitive; varying them separates the environment's share of the
induced geometry from the policy's.

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


# exploratory policy families: per-primitive mean and spread of theta, OU noise sigma
POLICIES = {
    "broad":    dict(mean=(0.5,) * 6, spread=1.0, noise=0.5),
    "narrow":   dict(mean=(0.5,) * 6, spread=0.5, noise=0.3),
    "bold":     dict(mean=(1.2, 0.3, -0.2, 0.0, 0.8, 0.5), spread=0.4, noise=0.5),
    "cautious": dict(mean=(0.4, 0.3, 1.0, 1.2, -0.2, 0.2), spread=0.4, noise=0.5),
    "social":   dict(mean=(0.4, 1.2, 0.3, 0.6, 0.2, 0.2), spread=0.4, noise=0.5),
}

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
    policy: str = "broad"              # exploratory policy family (POLICIES)
    noise_tau: float = 1.0
    noise_override: float | None = None  # fixed-policy runs: small noise so theta defines the policy
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
        ar._log = lambda: None                       # commons keeps its own compact log (memory)
        used = {x.slot for x in ar.agents}
        ar.capacity = 16                             # learner, two allies, ambushers
        ar._free = [i for i in range(ar.capacity) if i not in used]
        self.learner = ar.agents[0]
        for b in list(ar.agents[1:]):
            self._remove(b)
        self.learner.controller = self._control
        self.forager = self._add("A", HOME + np.array([3.0, 0.0]), self._forager_control)
        self.companion = self._add("A", HOME + np.array([0.0, 3.0]), self._companion_control)
        self.comp_heading = 0.0
        pol = POLICIES[self.policy]
        self.theta_mean = np.asarray(pol["mean"], float)
        self.theta_spread = float(pol["spread"])
        self.noise_sigma = float(pol["noise"]) if self.noise_override is None else self.noise_override
        self.theta_fixed: np.ndarray | None = None
        self.known: set = set()
        self.rivals: list[Agent] = []
        self.log = {k: [] for k in ("u", "xi", "s", "r", "theta", "episode", "t")}
        self.episode = -1
        self.theta = np.zeros(len(PRIMITIVES))
        self.xi = np.zeros(len(PRIMITIVES))
        self.heading = 0.0
        self.u = np.zeros(3)
        self.satiety = 0.0
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

    def _companion_control(self, ar, me, p):
        """Wanders the world (company is available anywhere, sometimes)."""
        me.npc.C[3] = 1.0
        self.comp_heading += 0.5 * np.sqrt(ar.wcfg.dt) * self.rng.normal()
        d = np.array([np.cos(self.comp_heading), np.sin(self.comp_heading)])
        if np.any(me.pos < 8) or np.any(me.pos > 92):          # turn back from the walls
            d = _unit(CENTER - me.pos)
            self.comp_heading = float(np.arctan2(d[1], d[0]))
        return d * 3.0

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
            max(0.0, 1 - np.linalg.norm(self.companion.pos - a.pos) / R),
        ])
        return np.concatenate([x, extra])

    def _resolve_by_sight(self):
        """Bits resolved by seeing patches' looks -- informative only if cues are valid."""
        if not self.env.cue_validity:
            return 0.0
        a, R = self.learner, self.ar.wcfg.sensor_radius
        bits = 0.0
        for q in self.ar.patches:
            if q["id"] not in self.known and np.linalg.norm(q["pos"] - a.pos) <= R:
                self.known.add(q["id"])
                bits += 1.0
        return bits

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
        self.companion.pos = self.rng.uniform(15, 85, 2)
        self.comp_heading = float(self.rng.uniform(0, 2 * np.pi))
        self.satiety = float(self.rng.uniform(0.0, 0.5))
        self.known = set()
        self.amb_patches = set()
        self.theta = (self.theta_fixed.copy() if self.theta_fixed is not None else
                      self.theta_mean + self.rng.uniform(-self.theta_spread, self.theta_spread, len(PRIMITIVES)))
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
                        if q["id"] not in self.known:       # probing at the patch resolves it
                            self.known.add(q["id"])
                            know += 1.0
                        if q["id"] not in self.amb_patches:
                            self.amb_patches.add(q["id"])
                            if self.rng.random() < self.env.threat:
                                self._ambush()
                if ar.t > self.ambush_until and self.rivals:
                    for b in self.rivals:
                        self._remove(b)
                    self.rivals = []
                self._keep_food()
            know += self._resolve_by_sight()
            social = float(ar.allies_near(a) > 0) * TICK
            safety = -(a.taken_total - taken0)
            self._record(u0, xi0, s, np.array([food, safety, social, know]))
            if a.health < 0.15:                             # beaten: end the episode
                break

    def _ambush(self):
        for b in self.rivals:                        # a new ambush replaces one still running
            self._remove(b)
        self.rivals = []
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

    def run_policies(self, thetas: np.ndarray, episodes_each: int) -> np.ndarray:
        """Run each policy theta for episodes_each episodes; returns the episode goal
        returns, shape (n_policies, episodes_each, n_goals)."""
        out = np.zeros((len(thetas), episodes_each, len(GOALS)))
        for i, th in enumerate(thetas):
            self.theta_fixed = np.asarray(th, float)
            for k in range(episodes_each):
                n0 = len(self.log["r"])
                self.run_episode()
                out[i, k] = np.sum(self.log["r"][n0:], axis=0)
                for key in self.log:                       # keep memory flat: drop the ticks
                    del self.log[key][n0:]
        self.theta_fixed = None
        return out

    def run(self, n_episodes: int) -> dict:
        for _ in range(n_episodes):
            self.run_episode()
        return {k: np.asarray(v) for k, v in self.log.items()}
