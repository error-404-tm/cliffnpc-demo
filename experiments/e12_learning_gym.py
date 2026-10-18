"""E12 Learning gym, Levels 0-2.

Can the agent learn at all, and does it learn the context-dependent fight policy?
Five policies on each task (cliffnpc/gym.py): the full Clifford agent (Cl(2,2)),
the same agent frozen, random, a non-geometric linear actor-critic on the same
features, and a hand-coded optimum.  20 seeds per (task, policy), in parallel.

Level 1: discrimination, reversal, fear conditioning + extinction, one-shot
         (witnessed death) vs a no-death control.
Level 2: courage -- fight iff our side outnumbers theirs.

Pass criteria (fixed before running):
  discrimination  error rate (first patch reached is false) over the last 30
                  trials below frozen's, CI not overlapping
  reversal        post-reversal error falls from the first 30 to the last 30 trials
  fear            cursed-first rate in the last 30 acquisition trials below the
                  first 10; rises again in extinction
  one_shot        marked-first rate after the death below the no-death control
  courage         fight share with the advantage minus without it (discrimination
                  index) above frozen's, and payoff above frozen's, last 100 trials
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

from cliffnpc.gym import TASKS  # noqa: E402

OUT = outdir("e12_learning_gym")
SEEDS = tuple(range(20))
POLICIES = ("clifford", "frozen", "random", "linear_ac", "optimal")
COLORS = dict(clifford="#c0392b", frozen="#7f8c8d", random="#bdc3c7", linear_ac="#2980b9",
              optimal="#27ae60")
RUN = {
    "discrimination": dict(n_trials=150),
    "reversal": dict(n_trials=300, reverse_at=150),
    "fear": dict(n_trials=200, extinguish_at=100),
    "one_shot": dict(n_trials=60, death_at=10, split=10),
    "one_shot_control": dict(n_trials=60, death_at=None, split=10),
    "courage": dict(n_trials=300),
}


def one(job):
    task, policy, seed = job
    from cliffnpc import gym
    fn = {"discrimination": gym.discrimination, "reversal": gym.discrimination, "fear": gym.fear,
          "one_shot": gym.one_shot, "one_shot_control": gym.one_shot,
          "courage": gym.courage}[task]
    recs = fn(seed, policy, **RUN[task])
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, (np.floating, np.bool_)):
                r[k] = v.item()
    return dict(task=task, policy=policy, seed=seed, recs=recs)


assert set(RUN) == set(TASKS)
jobs = [(t, p, s) for t in RUN for p in POLICIES for s in SEEDS
        if not (t.startswith("one_shot") and p == "optimal")]
R = {}
if os.environ.get("E12_REUSE") and (OUT / "records.json").exists():   # re-score saved runs
    with open(OUT / "records.json") as fh:
        R = {tuple(k.split("|")): v for k, v in json.load(fh).items()}
else:
    for r in pmap(one, jobs):
        R.setdefault((r["task"], r["policy"]), []).append(r["recs"])
    with open(OUT / "records.json", "w") as fh:
        json.dump({f"{t}|{p}": v for (t, p), v in R.items()}, fh)
for runs in R.values():                     # JSON null -> nan for float fields
    for recs in runs:
        for x in recs:
            for k in ("fight_share", "t_meal"):
                if k in x and x[k] is None:
                    x[k] = np.nan


# ---------------------------------------------------------------- helpers
def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def rate(recs, lo, hi, key, bad, among=None):
    """Share of trials in [lo, hi) with recs[key] == bad, among trials where key is in `among`."""
    sel = [r for r in recs[lo:hi] if among is None or r[key] in among]
    return np.mean([r[key] == bad for r in sel]) if sel else np.nan


def per_seed(task, policy, f):
    return [f(recs) for recs in R.get((task, policy), [])]


def fmt(m):
    return f"{m[0]:.3f} ± {m[1]:.3f}"


def curve(task, policy, key, bad, among, block):
    runs = R.get((task, policy), [])
    if not runs:
        return None, None
    n = len(runs[0])
    xs = np.arange(0, n, block)
    ys = np.array([[rate(recs, x, x + block, key, bad, among) for x in xs] for recs in runs])
    return xs + block / 2, ys


def plot_curves(ax, task, key, bad, among, block, title, vlines=()):
    for p in POLICIES:
        xs, ys = curve(task, p, key, bad, among, block)
        if xs is None:
            continue
        m = np.nanmean(ys, axis=0)
        e = 1.96 * np.nanstd(ys, axis=0) / np.sqrt(np.sum(~np.isnan(ys), axis=0).clip(1))
        ax.plot(xs, m, color=COLORS[p], label=p)
        ax.fill_between(xs, m - e, m + e, color=COLORS[p], alpha=0.15)
    for v in vlines:
        ax.axvline(v, color="k", ls="--", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("trial")
    ax.set_ylim(-0.02, 1.02)


lines = ["E12 Learning gym, Levels 0-2 (Cl(2,2))", "=" * 38,
         f"{len(SEEDS)} seeds per (task, policy); mean ± 95% CI over seeds.", ""]

# ---------------------------------------------------------------- Level 1
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
plot_curves(axes[0][0], "discrimination", "first", "false", ("real", "false"), 15,
            "discrimination: P(first patch reached is false)")
plot_curves(axes[0][1], "reversal", "first", "false", ("real", "false"), 20,
            "reversal: P(first patch is false); rule swaps at 150", vlines=(150,))
plot_curves(axes[0][2], "fear", "first", "cursed", ("cursed", "safe"), 20,
            "fear: P(first patch is the ambush site)", vlines=(100,))
ax = axes[1][0]
for task, ls in (("one_shot", "-"), ("one_shot_control", ":")):
    for p in ("clifford", "frozen", "linear_ac"):
        xs, ys = curve(task, p, "first", "marked", ("marked", "unmarked"), 10)
        ax.plot(xs, np.nanmean(ys, axis=0), ls, color=COLORS[p],
                label=f"{p} ({'death' if task == 'one_shot' else 'no death'})")
ax.axvline(10, color="k", ls="--", lw=0.8)
ax.set_title("one-shot: P(first patch is the landmark); death at trial 10", fontsize=10)
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=7)
axes[0][0].legend(fontsize=8)

lines.append("LEVEL 1")
# discrimination
lines.append("discrimination: error rate (first patch reached is false), among trials that reached one")
for p in POLICIES:
    e0 = ci(per_seed("discrimination", p, lambda r: rate(r, 0, 30, "first", "false", ("real", "false"))))
    e1 = ci(per_seed("discrimination", p, lambda r: rate(r, 120, 150, "first", "false", ("real", "false"))))
    nr = ci(per_seed("discrimination", p, lambda r: np.mean([x["first"] is None for x in r[120:]])))
    lines.append(f"   {p:<10s} first 30: {fmt(e0)}   last 30: {fmt(e1)}   no arrival (last 30): {fmt(nr)}")
# reversal
lines.append("reversal: error rate before the swap (trials 120-150), just after (150-180), late (270-300)")
for p in POLICIES:
    a = ci(per_seed("reversal", p, lambda r: rate(r, 120, 150, "first", "false", ("real", "false"))))
    b = ci(per_seed("reversal", p, lambda r: rate(r, 150, 180, "first", "false", ("real", "false"))))
    c = ci(per_seed("reversal", p, lambda r: rate(r, 270, 300, "first", "false", ("real", "false"))))
    lines.append(f"   {p:<10s} pre: {fmt(a)}   just after: {fmt(b)}   late: {fmt(c)}")
# fear
lines.append("fear: share of trials reaching the ambush site first; health lost per acquisition trial")
for p in POLICIES:
    a = ci(per_seed("fear", p, lambda r: rate(r, 0, 10, "first", "cursed", ("cursed", "safe"))))
    b = ci(per_seed("fear", p, lambda r: rate(r, 70, 100, "first", "cursed", ("cursed", "safe"))))
    c = ci(per_seed("fear", p, lambda r: rate(r, 170, 200, "first", "cursed", ("cursed", "safe"))))
    h = ci(per_seed("fear", p, lambda r: np.mean([x["health_start"] - x["health_end"] for x in r[70:100]])))
    lines.append(f"   {p:<10s} acq first 10: {fmt(a)}   acq last 30: {fmt(b)}   ext last 30: {fmt(c)}"
                 f"   health lost (acq last 30): {fmt(h)}")
# one-shot
lines.append("one-shot: share of trials reaching the landmark first, trials 11-60 (death at trial 10)")
for p in ("clifford", "frozen", "random", "linear_ac"):
    d = per_seed("one_shot", p, lambda r: rate(r, 11, 60, "first", "marked", ("marked", "unmarked")))
    c = per_seed("one_shot_control", p, lambda r: rate(r, 11, 60, "first", "marked", ("marked", "unmarked")))
    early = per_seed("one_shot", p, lambda r: rate(r, 11, 21, "first", "marked", ("marked", "unmarked")))
    early_c = per_seed("one_shot_control", p, lambda r: rate(r, 11, 21, "first", "marked", ("marked", "unmarked")))
    lines.append(f"   {p:<10s} death: {fmt(ci(d))}   control: {fmt(ci(c))}   paired diff: "
                 f"{fmt(ci(np.array(d) - np.array(c)))}   first 10 after, diff: "
                 f"{fmt(ci(np.array(early) - np.array(early_c)))}")

# ---------------------------------------------------------------- Level 2
ax = axes[1][1]
for p in POLICIES:
    runs = R[("courage", p)]
    xs = np.arange(0, 300, 30)
    ys = []
    for recs in runs:
        row = []
        for x in xs:
            blk = recs[x:x + 30]
            good = [b["fight_share"] for b in blk if b["advantage"] > 0 and not np.isnan(b["fight_share"])]
            bad = [b["fight_share"] for b in blk if b["advantage"] <= 0 and not np.isnan(b["fight_share"])]
            row.append(np.mean(good) - np.mean(bad) if good and bad else np.nan)
        ys.append(row)
    ys = np.array(ys)
    ax.plot(xs + 15, np.nanmean(ys, axis=0), color=COLORS[p], label=p)
ax.axhline(0, color="k", lw=0.5)
ax.set_title("courage: fight share when outnumbering − when not", fontsize=10)
ax.set_xlabel("trial")
ax = axes[1][2]
for p in POLICIES:
    runs = R[("courage", p)]
    xs = np.arange(0, 300, 30)
    ys = np.array([[np.mean([b["payoff"] for b in recs[x:x + 30]]) for x in xs] for recs in runs])
    ax.plot(xs + 15, ys.mean(axis=0), color=COLORS[p], label=p)
ax.axhline(0, color="k", lw=0.5)
ax.set_title("courage: payoff per encounter (dealt − taken)", fontsize=10)
ax.set_xlabel("trial")
ax.legend(fontsize=8)
save(fig, OUT, "gym.png")


def courage_index(recs, lo, hi):
    blk = recs[lo:hi]
    good = [b["fight_share"] for b in blk if b["advantage"] > 0 and not np.isnan(b["fight_share"])]
    bad = [b["fight_share"] for b in blk if b["advantage"] <= 0 and not np.isnan(b["fight_share"])]
    return np.mean(good) - np.mean(bad) if good and bad else np.nan


lines += ["", "LEVEL 2", "courage: discrimination index (fight share outnumbering − not), payoff, defeats"]
pay = {}
for p in POLICIES:
    i0 = ci(per_seed("courage", p, lambda r: courage_index(r, 0, 100)))
    i1 = ci(per_seed("courage", p, lambda r: courage_index(r, 200, 300)))
    pay[p] = per_seed("courage", p, lambda r: np.mean([b["payoff"] for b in r[200:]]))
    d = ci(per_seed("courage", p, lambda r: np.mean([b["defeated"] for b in r[200:]])))
    lines.append(f"   {p:<10s} index first 100: {fmt(i0)}   last 100: {fmt(i1)}   "
                 f"payoff last 100: {fmt(ci(pay[p]))}   defeated: {fmt(d)}")
lo, hi = np.mean(pay["random"]), np.mean(pay["optimal"])
lines.append("   normalised payoff, last 100 (0 = random, 1 = optimal): " + "   ".join(
    f"{p}: {(np.mean(pay[p]) - lo) / (hi - lo):.2f}" for p in POLICIES))

# ---------------------------------------------------------------- learning vs frozen, paired
lines += ["", "LEARNING EFFECT (clifford − frozen, paired by seed; negative error = better)"]
pairs = [
    ("discrimination error, last 30", "discrimination",
     lambda r: rate(r, 120, 150, "first", "false", ("real", "false"))),
    ("reversal error, late", "reversal", lambda r: rate(r, 270, 300, "first", "false", ("real", "false"))),
    ("fear: ambush-site rate, acq last 30", "fear",
     lambda r: rate(r, 70, 100, "first", "cursed", ("cursed", "safe"))),
    ("courage index, last 100", "courage", lambda r: courage_index(r, 200, 300)),
    ("courage payoff, last 100", "courage", lambda r: np.mean([b["payoff"] for b in r[200:]])),
]
for lab, task, f in pairs:
    c = np.array(per_seed(task, "clifford", f), float)
    z = np.array(per_seed(task, "frozen", f), float)
    a = np.array(per_seed(task, "linear_ac", f), float)
    lines.append(f"   {lab:<38s} clifford − frozen: {fmt(ci(c - z))}   linear_ac − frozen: {fmt(ci(a - z))}")
write_summary(OUT, lines)
