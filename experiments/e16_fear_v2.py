"""E16 Fear conditioning, redesigned (fear v2).

E12-E15's fear task was flawed: the ambush was one rival, and a 1-vs-1 fight
under the n/(n+m) rule deals as much as it takes, so under the objective reward
the ambush cost nothing and the cursed patch was just food.  No drive owned a
landmark and the body could not flee a place.

fear v2: the ambush is three rivals (outnumbered 3:1: about -1.8 per trial at the
cursed patch vs +2.0 at the safe one), a landmark is a danger cue owned by
fight/flight (drive-layer valence), and with no rival in view a negative
fight/flight drive moves the agent away from a nearby landmark (flee_places).
Acquisition 100 trials, extinction 100.  All policies re-run (body flags on).

Pass criteria (fixed before running): ambush-site-first rate over the last 30
acquisition trials below frozen's (paired CI excludes 0), and back up toward
0.5 in extinction.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

OUT = outdir("e16_fear_v2")
SEEDS = tuple(range(20))
POLICIES = ("frozen", "clifford", "clifford_p_only", "clifford_pav_drive", "clifford_pav_body",
            "clifford_pav_cartan", "clifford_pav_drive_affect", "random", "linear_ac", "optimal")
COLORS = dict(frozen="#7f8c8d", clifford="#e6a0a0", clifford_p_only="#f0c090",
              clifford_pav_drive="#c0392b", clifford_pav_body="#8e44ad", clifford_pav_cartan="#16a085",
              clifford_pav_drive_affect="#d35400", random="#bdc3c7", linear_ac="#2980b9", optimal="#27ae60")
EXT = 100


def one(job):
    policy, seed = job
    from cliffnpc import gym
    gym.FOOD_AVERSION = gym.FLEE_PLACES = True
    grab = {}
    orig = gym.gym_arena

    def wrapped(s, p):
        ar, a = orig(s, p)
        grab["a"] = a
        return ar, a
    gym.gym_arena = wrapped
    recs = gym.fear_v2(seed, policy, extinguish_at=EXT)
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, (np.floating, np.bool_)):
                r[k] = v.item()
    a = grab["a"]
    v_place = None
    if a.conn is not None and a.conn.cfg.rule == "td":
        from cliffnpc.connection import FEATURES
        v_place = float(a.conn.v[FEATURES.index("place_cue")])
    return dict(policy=policy, seed=seed, recs=recs, v_place=v_place)


res = pmap(one, [(p, s) for p in POLICIES for s in SEEDS])
R, VP = {}, {}
for r in res:
    R.setdefault(r["policy"], []).append(r["recs"])
    VP.setdefault(r["policy"], []).append(r["v_place"])
with open(OUT / "records.json", "w") as fh:
    json.dump(R, fh)


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def fmt(m):
    return f"{m[0]:.3f} ± {m[1]:.3f}"


def rate(recs, lo, hi):
    sel = [r for r in recs[lo:hi] if r["first"] in ("cursed", "safe")]
    return np.mean([r["first"] == "cursed" for r in sel]) if sel else np.nan


def ate(r):
    return r["t_meal"] is not None and not np.isnan(r["t_meal"])


def objective(r):
    return 2.0 * ate(r) + 10.0 * (r["dealt"] - r["taken"])


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for p in POLICIES:
    xs = np.arange(0, 200, 20)
    ys = np.array([[rate(recs, x, x + 20) for x in xs] for recs in R[p]])
    axes[0].plot(xs + 10, np.nanmean(ys, axis=0), color=COLORS[p], label=p)
    ys = np.array([[np.mean([objective(r) for r in recs[x:x + 20]]) for x in xs] for recs in R[p]])
    axes[1].plot(xs + 10, ys.mean(axis=0), color=COLORS[p], label=p)
for ax in axes:
    ax.axvline(EXT, color="k", ls="--", lw=0.8)
    ax.set_xlabel("trial")
axes[0].set_title("P(first patch is the ambush site); extinction from trial 100")
axes[0].set_ylim(-0.02, 1.02)
axes[1].set_title("objective reward per trial (meal 2, injury dealt - taken x10)")
axes[1].legend(fontsize=7)
save(fig, OUT, "fear_v2.png")

lines = ["E16 Fear conditioning v2 (Cl(2,2))", "=" * 34,
         f"{len(SEEDS)} seeds per policy; mean ± 95% CI over seeds.", "",
         "ambush-site-first rate: acquisition first 10 / last 30, extinction last 30;",
         "objective reward per trial (acq last 30); meals (acq last 30); critic value of the landmark"]
for p in POLICIES:
    a0 = ci([rate(r, 0, 10) for r in R[p]])
    a1 = ci([rate(r, 70, 100) for r in R[p]])
    e1 = ci([rate(r, 170, 200) for r in R[p]])
    ob = ci([np.mean([objective(x) for x in r[70:100]]) for r in R[p]])
    me = ci([np.mean([ate(x) for x in r[70:100]]) for r in R[p]])
    vp = "" if VP[p][0] is None else f"   v_place {fmt(ci(VP[p]))}"
    lines.append(f"   {p:<26s} acq first 10 {fmt(a0)}   acq last 30 {fmt(a1)}   ext last 30 {fmt(e1)}"
                 f"   reward {fmt(ob)}   meals {fmt(me)}{vp}")
lines += ["", "LEARNING EFFECT vs frozen (paired by seed): ambush-site rate, acq last 30 (negative = better);"
          " extinction last 30 minus acq last 30 (positive = extinguished)"]
z = np.array([rate(r, 70, 100) for r in R["frozen"]])
for p in POLICIES[1:]:
    c = np.array([rate(r, 70, 100) for r in R[p]])
    ext = np.array([rate(r, 170, 200) for r in R[p]]) - c
    lines.append(f"   {p:<26s} {fmt(ci(c - z))}   extinction {fmt(ci(ext))}")
write_summary(OUT, lines)
