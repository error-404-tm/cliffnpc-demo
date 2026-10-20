"""Learning gym: small controlled tasks with known answers.

Each task runs one learner in a 100 x 100 world, trial by trial, with scripted
food, rivals and allies.  The same task is run under five policies:

    clifford   the full agent, Cl(2,2): metric learning, learned connection,
               episodic event-chain memory, trivector affect
    frozen     the same agent with all learning and memory storage off
    random     a random option every decision
    linear_ac  a non-geometric learner: linear softmax actor-critic (TD(lambda))
               on the same 16 situation features, trained on objective reward
               (food eaten, injury dealt minus taken)
    optimal    a hand-coded policy that knows the task's current rule

Controllers (random, linear_ac, optimal) choose among five options every
DECIDE_EVERY seconds: approach / move away from the nearest seen food,
approach (fight) / flee the nearest rival, wander.  They perceive exactly what
the Clifford agent perceives: the nearest food, its look, the nearest rival,
allies and rivals in view, a nearness-to-landmark cue.

Tasks (Level 1: can it learn; Level 2: context-dependent policy)

    discrimination   two patches on opposite sides of the learner, one looks A
                     (real), one looks B (false: no satiety).  Positions random
                     each trial, so only the look is informative.
    reversal         as discrimination; the meaning of the looks swaps halfway.
    fear             both patches real; one sits at a landmark.  Arriving there
                     brings an ambush (a rival attacks) during acquisition, and
                     nothing during extinction.
    one_shot         both patches real, one at a landmark.  On one trial an ally
                     dies at the landmark in view; control: no death.
    courage          the learner with k scripted allies meets m scripted rivals;
                     fighting pays when k + 1 > m (the n / (n + m) damage rule).

Every task returns per-trial records; scoring lives in the experiment script.
"""

from __future__ import annotations

import numpy as np

from .arena import Agent, Arena, life_config
from .config import FoodConfig
from .connection import N_FEAT

SIG = (-1.0, 1.0, 1.0, -1.0)
# Body: give hunger an aversive pole (steering.food_aversion) for all Clifford
# variants.  E12 ran without it; E13 sets this True.
FOOD_AVERSION = False
POLICIES = ("clifford", "frozen", "random", "linear_ac", "optimal")
TD_POLICIES = ("clifford_td", "clifford_td_affect", "clifford_td_metric", "clifford_td_affine")
CARTAN_POLICIES = ("clifford_cartan", "clifford_cartan_affect", "clifford_p_only")
PAV_POLICIES = ("clifford_pav_drive", "clifford_pav_body", "clifford_pav_cartan")
OPTIONS = ("approach_food", "avoid_food", "fight", "flee", "wander")
DECIDE_EVERY = 0.5          # s between controller decisions
CENTER = np.array([50.0, 50.0])
LOOK_A, LOOK_B = 1, 2
DEFEATED = 0.15             # learner health below this ends the trial (the gym never kills it)


# ------------------------------------------------------------------ world
def gym_arena(seed: int, policy: str) -> tuple[Arena, Agent]:
    """A one-learner arena: no births, no starvation, no random food, no false-food memory."""
    cfg = life_config(SIG, seed, memory_model="episodic", connection=True)
    cfg.base.world.size = 100.0
    cfg.n_npcs = 1
    cfg.home, cfg.lair = tuple(CENTER), (90.0, 90.0)
    cfg.food_dynamics = FoodConfig(initial_patches=0, max_patches=0, spawn_rate=0.0, stock=1e9,
                                   false_fraction=0.0, false_lifetime=0.0, remember_false=False)
    cfg.life.birth_cooldown = 1e12
    cfg.life.starve_level = 1e12
    cfg.base.steer.food_aversion = FOOD_AVERSION
    if policy == "frozen":
        cfg.learning = False
        cfg.episodic.store = False
    if policy.startswith("clifford_td"):          # actor-critic connection (step 2)
        cfg.connection.rule = "td"
        cfg.connection.reward = "affect" if policy == "clifford_td_affect" else "objective"
        cfg.metric_learning = policy == "clifford_td_metric"
        cfg.connection.affine = policy == "clifford_td_affine"
    if policy.startswith("clifford_cartan") or policy == "clifford_p_only":   # Cartan-split connection
        cfg.connection.rule = "td"
        cfg.connection.cartan = True
        cfg.connection.learn_rotation = policy != "clifford_p_only"
        cfg.connection.reward = "affect" if policy == "clifford_cartan_affect" else "objective"
        cfg.metric_learning = False
    if policy.startswith("clifford_pav"):        # Pavlovian valence from the TD critic
        cfg.connection.rule = "td"
        cfg.connection.reward = "objective"
        cfg.metric_learning = False
        cfg.connection.valence = "body" if policy == "clifford_pav_body" else "drive"
        cfg.connection.cartan = policy == "clifford_pav_cartan"
        cfg.connection.actor = policy == "clifford_pav_cartan"
    ar = Arena(cfg)
    learner = ar.agents[0]
    for b in list(ar.agents[1:]):                 # tribes mode made one B agent: remove it
        _remove(ar, b)
    learner.pos = CENTER.copy()
    if policy in ("random", "linear_ac", "optimal"):
        learner.controller = Controller(policy, ar.rng)
    return ar, learner


def _remove(ar: Arena, a: Agent) -> None:
    if a in ar.agents:
        ar.agents.remove(a)
        ar._free.append(a.slot)
        ar._free.sort()


def _add(ar: Arena, faction: str, pos, controller) -> Agent:
    a = ar._make(faction, np.asarray(pos, float), pos=np.asarray(pos, float).copy())
    a.controller = controller
    ar.agents.append(a)
    return a


def _clear_food(ar: Arena) -> None:
    ar.patches.clear()


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(2)


def _reset_learner(ar: Arena, a: Agent, hunger: float | None = 1.0) -> None:
    a.pos = CENTER.copy()
    a.vel = np.zeros(2)
    a.health, a.mobile = 1.0, True
    a.known_food.clear()
    a.known_false.clear()
    a.probe.clear()
    if hunger is not None:
        a.npc.C[0] = hunger                         # hungry at the start of every trial
    if a.controller is not None:
        a.controller.new_trial()
    if a.conn is not None and a.conn.cfg.rule == "td":
        a.conn.reset_episode()                      # teleported: no bootstrap across trials


# ------------------------------------------------------------------ controllers
def nearest_food(ar: Arena, a: Agent):
    R = ar.wcfg.sensor_radius
    seen = [p for p in ar.patches if np.linalg.norm(p["pos"] - a.pos) <= R]
    return min(seen, key=lambda p: np.linalg.norm(p["pos"] - a.pos)) if seen else None


def nearest_foe(ar: Arena, a: Agent):
    return ar._rival_of(a)


def option_velocity(ar: Arena, a: Agent, option: int, heading: float) -> np.ndarray:
    v_max = ar.steer_cfg.max_speed
    food, foe = nearest_food(ar, a), nearest_foe(ar, a)
    name = OPTIONS[option]
    if name == "approach_food" and food is not None:
        d = food["pos"] - a.pos
        return _unit(d) * min(v_max, np.linalg.norm(d) / ar.wcfg.dt)   # stop on the patch
    if name == "avoid_food" and food is not None:
        return -_unit(food["pos"] - a.pos) * v_max
    if name == "fight" and foe is not None:
        return _unit(foe.pos - a.pos) * v_max
    if name == "flee" and foe is not None:
        return -_unit(foe.pos - a.pos) * v_max
    return np.array([np.cos(heading), np.sin(heading)]) * v_max * 0.6


class Controller:
    """Policy for a controlled learner.  `optimal` needs a task-specific rule set by the task."""

    def __init__(self, policy: str, rng: np.random.Generator, gamma=0.95, lam=0.8,
                 lr_pi=0.05, lr_v=0.05, temp=1.0):
        self.policy, self.rng = policy, rng
        self.rule = None                                  # optimal: f(ar, a) -> option
        self.n_in = N_FEAT + 1
        self.theta = np.zeros((len(OPTIONS), self.n_in))
        self.w = np.zeros(self.n_in)
        self.e_pi = np.zeros_like(self.theta)
        self.e_v = np.zeros_like(self.w)
        self.gamma, self.lam, self.lr_pi, self.lr_v, self.temp = gamma, lam, lr_pi, lr_v, temp
        self.option, self.heading, self.next_t = 4, 0.0, 0.0
        self.reward, self.prev_phi, self.prev_pi = 0.0, None, None
        self.choices: list[tuple] = []

    def new_trial(self):
        """Trial boundary: terminal TD update (the learner is teleported), clear traces."""
        if self.policy == "linear_ac" and self.prev_phi is not None:
            delta = self.reward - self.w @ self.prev_phi
            self.w += self.lr_v * delta * self.e_v
            self.theta += self.lr_pi * delta * self.e_pi
        self.prev_phi, self.reward, self.next_t = None, 0.0, 0.0
        self.e_pi[:] = 0.0
        self.e_v[:] = 0.0

    def phi(self, ar, a):
        return np.append(ar.context_of(a, ar.percept(a)), 1.0)

    def __call__(self, ar: Arena, a: Agent, p) -> np.ndarray:
        # reward since the last decision: eating, injury dealt minus taken
        self.reward += float(ar._eating(a)) * ar.wcfg.dt + 10.0 * (a.dealt - a.taken)
        a.dealt = a.taken = 0.0
        self.heading += ar.steer_cfg.wander_turn * np.sqrt(ar.wcfg.dt) * self.rng.normal()
        if ar.t >= self.next_t:
            self.next_t = ar.t + DECIDE_EVERY
            self.decide(ar, a)
        a.npc.C[3] = {2: 1.0, 3: -1.0}.get(self.option, 0.0)   # fight flag for the combat rule
        return option_velocity(ar, a, self.option, self.heading)

    def decide(self, ar, a):
        if self.policy == "random":
            self.option = int(self.rng.integers(len(OPTIONS)))
        elif self.policy == "optimal":
            self.option = self.rule(ar, a)
        else:
            phi = self.phi(ar, a)
            z = self.theta @ phi / self.temp
            pi = np.exp(z - z.max())
            pi /= pi.sum()
            if self.prev_phi is not None:                    # TD(lambda) actor-critic update
                delta = self.reward + self.gamma * self.w @ phi - self.w @ self.prev_phi
                self.w += self.lr_v * delta * self.e_v
                self.theta += self.lr_pi * delta * self.e_pi
            self.option = int(self.rng.choice(len(OPTIONS), p=pi))
            grad = -np.outer(pi, phi)
            grad[self.option] += phi
            self.e_v = self.gamma * self.lam * self.e_v + phi
            self.e_pi = self.gamma * self.lam * self.e_pi + grad
            self.prev_phi = phi
        self.reward = 0.0
        self.choices.append((ar.t, self.option))


# ------------------------------------------------------------------ food tasks
def _place_pair(ar: Arena, rng: np.random.Generator, looks=(0, 0)):
    """Two real-or-false patches on opposite sides of the centre at random distances."""
    th = rng.uniform(0, 2 * np.pi)
    out = []
    for k, ang in enumerate((th, th + np.pi)):
        d = rng.uniform(8.0, 22.0)
        pos = CENTER + d * np.array([np.cos(ang), np.sin(ang)])
        ar._spawn(pos, ar.fd.stock, false=False, cue=looks[k])
        out.append(ar.patches[-1])
    return out


def _arrived(ar, a, patches):
    r = ar.cfg.npc_cog.eat_radius
    for k, p in enumerate(patches):
        if np.linalg.norm(a.pos - p["pos"]) < r:
            return k
    return None


def _run_food_trial(ar, a, patches, T_max=30.0, eat_for=2.0, on_step=None):
    """Run until the learner has eaten eat_for seconds of real food or T_max passes.
    Returns (index of the first patch reached or None, time to first meal or nan)."""
    first, eaten, t0, t_meal = None, 0.0, ar.t, np.nan
    while ar.t - t0 < T_max and eaten < eat_for and a.health >= DEFEATED:
        ar.step()
        if on_step is not None:
            on_step()
        k = _arrived(ar, a, patches)
        if first is None and k is not None:
            first = k
        if ar._eating(a):
            if np.isnan(t_meal):
                t_meal = ar.t - t0
            eaten += ar.wcfg.dt
    return first, t_meal


def discrimination(seed: int, policy: str, n_trials: int = 60, reverse_at: int | None = None) -> list:
    """Look A real, look B false (swapped from `reverse_at` on).  Records which side was
    reached first (looks of the patches are randomised across sides)."""
    ar, a = gym_arena(seed, policy)
    rng = np.random.default_rng(seed + 101)
    real_look = [LOOK_A]

    def rule(ar, a):                                  # optimal: approach only the real look
        f = nearest_food(ar, a)
        if f is None:
            return 4
        return 0 if f["cue"] == real_look[0] else 1
    if a.controller is not None:
        a.controller.rule = rule
    recs = []
    for trial in range(n_trials):
        if reverse_at is not None and trial == reverse_at:
            real_look[0] = LOOK_B
        _clear_food(ar)
        looks = (LOOK_A, LOOK_B) if rng.random() < 0.5 else (LOOK_B, LOOK_A)
        pats = _place_pair(ar, rng, looks)
        for p in pats:
            p["false"] = p["cue"] != real_look[0]
        _reset_learner(ar, a)
        first, t_meal = _run_food_trial(ar, a, pats)
        recs.append(dict(trial=trial, first=None if first is None else
                         ("false" if pats[first]["false"] else "real"),
                         t_meal=t_meal, phase="post" if reverse_at is not None and trial >= reverse_at
                         else "pre"))
    return recs


def fear(seed: int, policy: str, n_trials: int = 80, extinguish_at: int = 40,
         ambush_len: float = 6.0) -> list:
    """Both patches real; one at a landmark.  Reaching it brings an ambush (acquisition);
    from `extinguish_at` on, nothing happens there (extinction)."""
    ar, a = gym_arena(seed, policy)
    rng = np.random.default_rng(seed + 202)
    state = dict(cursed=None, acq=True)

    def rule(ar, a):                                  # optimal: avoid the landmark while it bites
        f = nearest_food(ar, a)
        foe = nearest_foe(ar, a)
        if foe is not None:
            return 3
        if f is None:
            return 4
        return 1 if (state["acq"] and f is state["cursed"]) else 0
    if a.controller is not None:
        a.controller.rule = rule
    recs = []
    for trial in range(n_trials):
        state["acq"] = trial < extinguish_at
        _clear_food(ar)
        pats = _place_pair(ar, rng)
        k = int(rng.integers(2))
        state["cursed"] = pats[k]
        ar.cfg.place_cues = ((tuple(pats[k]["pos"]), 20.0),)
        _reset_learner(ar, a)
        amb = dict(rival=None, t=None, injury=0.0)

        def on_step():
            if (state["acq"] and amb["t"] is None
                    and np.linalg.norm(a.pos - state["cursed"]["pos"]) < 5.0):
                pos = a.pos + 6.0 * _unit(rng.normal(size=2))
                amb["rival"] = _add(ar, "B", pos, chaser(a))
                amb["t"] = ar.t
            if amb["rival"] is not None:
                if ar.t - amb["t"] > ambush_len:
                    _remove(ar, amb["rival"])
                    amb["rival"] = None
        h0 = a.health
        first, t_meal = _run_food_trial(ar, a, pats, on_step=on_step)
        if amb["rival"] is not None:
            _remove(ar, amb["rival"])
        recs.append(dict(trial=trial, first=None if first is None else
                         ("cursed" if pats[first] is state["cursed"] else "safe"),
                         t_meal=t_meal, ambushed=amb["t"] is not None, health_end=a.health,
                         health_start=h0, phase="acquisition" if state["acq"] else "extinction"))
    return recs


def one_shot(seed: int, policy: str, n_trials: int = 50, death_at: int | None = 10,
             split: int = 10) -> list:
    """Both patches real, one at a landmark.  On trial `death_at` an ally dies at the
    landmark while the learner is near; afterwards: does the learner avoid it?"""
    ar, a = gym_arena(seed, policy)
    rng = np.random.default_rng(seed + 303)
    if a.controller is not None:
        a.controller.rule = lambda ar, a: 0 if nearest_food(ar, a) is not None else 4
    recs = []
    for trial in range(n_trials):
        _clear_food(ar)
        pats = _place_pair(ar, rng)
        k = int(rng.integers(2))
        marked = pats[k]
        ar.cfg.place_cues = ((tuple(marked["pos"]), 20.0),)
        _reset_learner(ar, a)
        ev = dict(done=False)

        def on_step():
            if (trial == death_at and not ev["done"]
                    and np.linalg.norm(a.pos - marked["pos"]) < 12.0):
                ally = _add(ar, a.faction, marked["pos"] + np.array([1.0, 0.0]), still)
                ally.ctx = None
                ar._die(ally, "injury")
                ev["done"] = True
        first, t_meal = _run_food_trial(ar, a, pats, on_step=on_step)
        recs.append(dict(trial=trial, first=None if first is None else
                         ("marked" if pats[first] is marked else "unmarked"),
                         t_meal=t_meal, death=ev["done"],
                         phase="before" if trial <= split else "after"))
    return recs


# ------------------------------------------------------------------ scripted agents
def chaser(target: Agent, speed: float = 4.0):
    """A rival that runs at `target` and fights."""
    def f(ar, me, p):
        me.npc.C[3] = 1.0
        return _unit(target.pos - me.pos) * speed
    return f


def still(ar, me, p):
    return np.zeros(2)


def hunter(speed: float = 4.0):
    """A rival that runs at the nearest agent of the other tribe and fights."""
    def f(ar, me, p):
        me.npc.C[3] = 1.0
        foes = [b for b in ar.agents if b.faction != me.faction and b.mobile]
        if not foes:
            return np.zeros(2)
        t = min(foes, key=lambda b: np.linalg.norm(b.pos - me.pos))
        return _unit(t.pos - me.pos) * speed
    return f


def ally_of(leader: Agent, speed: float = 5.0):
    """Stays near the leader; attacks rivals that come within 12 of the leader."""
    def f(ar, me, p):
        me.npc.C[3] = 1.0
        foes = [b for b in ar.agents if b.faction != me.faction and b.mobile
                and np.linalg.norm(b.pos - leader.pos) < 12.0]
        if foes:
            t = min(foes, key=lambda b: np.linalg.norm(b.pos - me.pos))
            return _unit(t.pos - me.pos) * speed
        d = leader.pos - me.pos
        return _unit(d) * min(speed, max(0.0, np.linalg.norm(d) - 3.0))
    return f


# ------------------------------------------------------------------ Level 2
def courage(seed: int, policy: str, n_trials: int = 120, T_trial: float = 25.0) -> list:
    """k allies (0-3) with the learner meet m rivals (1-3) approaching from 25 away.
    Fighting pays iff k + 1 > m.  Records whether the learner fought and its payoff."""
    ar, a = gym_arena(seed, policy)
    rng = np.random.default_rng(seed + 404)

    def rule(ar, a):                                  # optimal: fight iff our side is bigger
        foe = nearest_foe(ar, a)
        if foe is None:
            return 4
        R = ar.wcfg.sensor_radius
        m = sum(b.faction != a.faction and b.mobile and np.linalg.norm(b.pos - a.pos) <= R
                for b in ar.agents)
        n = 1 + sum(b is not a and b.faction == a.faction and b.mobile
                    and np.linalg.norm(b.pos - a.pos) <= R for b in ar.agents)
        return 2 if n > m else 3
    if a.controller is not None:
        a.controller.rule = rule
    recs = []
    for trial in range(n_trials):
        k, m = int(rng.integers(0, 4)), int(rng.integers(1, 4))
        _clear_food(ar)
        _reset_learner(ar, a, hunger=None)
        a.npc.C[0] = min(a.npc.C[0], 0.0)              # not hungry: the trial is about the fight
        extra = []
        for j in range(k):
            ang = 2 * np.pi * j / max(k, 1)
            extra.append(_add(ar, a.faction, a.pos + 3.0 * np.array([np.cos(ang), np.sin(ang)]),
                              ally_of(a)))
        th = rng.uniform(0, 2 * np.pi)
        for j in range(m):
            pos = a.pos + 25.0 * np.array([np.cos(th), np.sin(th)]) + rng.normal(0, 2.0, 2)
            extra.append(_add(ar, "B", np.clip(pos, 1, 99), hunter()))
        d0, k0 = a.dealt_total, a.taken_total
        fight_ticks = view_ticks = 0
        contact = False
        t0 = ar.t
        defeated = False
        while ar.t - t0 < T_trial:
            ar.step()
            if a.health < DEFEATED:                      # beaten: end the trial before death
                defeated = True
                break
            contact |= a.in_contact
            if nearest_foe(ar, a) is not None:
                view_ticks += 1
                fight_ticks += a.npc.C[3] > 0
        dealt, taken = a.dealt_total - d0, a.taken_total - k0
        for b in extra:
            _remove(ar, b)
        recs.append(dict(trial=trial, k=k, m=m, advantage=k + 1 - m, contact=contact,
                         defeated=defeated,
                         fight_share=fight_ticks / view_ticks if view_ticks else np.nan,
                         dealt=dealt, taken=taken, payoff=dealt - taken))
    return recs


TASKS = {
    "discrimination": lambda s, p: discrimination(s, p, n_trials=60),
    "reversal": lambda s, p: discrimination(s, p, n_trials=100, reverse_at=50),
    "fear": lambda s, p: fear(s, p),
    "one_shot": lambda s, p: one_shot(s, p, death_at=10),
    "one_shot_control": lambda s, p: one_shot(s, p, death_at=None),
    "courage": lambda s, p: courage(s, p),
}
