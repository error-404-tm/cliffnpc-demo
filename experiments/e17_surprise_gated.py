"""E17 Surprise-gated value learning: does tying learning to the value delta give
low-shot learning?

The critic's step grows with how surprising the TD error delta is relative to its
running spread (Pearce-Hall associability; connection.surprise_gain):
    rate = critic_rate * (1 + gain * max(0, |delta|/sd - z)),  z = 8, capped at 1.
Ordinary ticks learn as before; a rare large delta -- of either sign -- writes its
value into the stimuli present at once.  All Clifford variants use drive-layer
Pavlovian valence, food aversion and place flight (as E16).

Policies: frozen; pav_drive (objective reward), pav_drive_surprise; pav_drive_affect
(reward = -felt affect), pav_drive_affect_surprise.
Tasks: one_shot (fixed: the ally dies when the learner perceives the landmark cue,
on the first trial from 10 on) vs a no-death control; fear_v2; discrimination;
reversal; courage.

Pass criteria (fixed before running):
  one-shot   landmark-first rate in the 10 trials after the death, minus the
             no-death control (paired by seed), below 0 for a *_surprise variant
             and below its ungated counterpart
  low-shot   fear_v2 ambush-site rate over acquisition trials 1-10 lower with the gate
  no harm    discrimination / reversal / courage not worse than ungated
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

OUT = outdir("e17_surprise_gated")
SEEDS = tuple(range(20))
POLICIES = ("frozen", "clifford_pav_drive", "clifford_pav_drive_surprise",
            "clifford_pav_drive_affect", "clifford_pav_drive_affect_surprise")
COLORS = dict(frozen="#7f8c8d", clifford_pav_drive="#e6a0a0", clifford_pav_drive_surprise="#c0392b",
              clifford_pav_drive_affect="#f0c090", clifford_pav_drive_affect_surprise="#d35400")
TASKS = ("one_shot", "one_shot_control", "fear_v2", "discrimination", "reversal", "courage")


def one(job):
    task, policy, seed = job
    from cliffnpc import gym
    from cliffnpc.connection import FEATURES
    gym.FOOD_AVERSION = gym.FLEE_PLACES = True
    grab = {}
    orig = gym.gym_arena

    def wrapped(s, p):
        ar, a = orig(s, p)
        grab["a"] = a
        return ar, a
    gym.gym_arena = wrapped
    fn = {"one_shot": lambda: gym.one_shot(seed, policy, n_trials=60, death_at=10, split=10),
          "one_shot_control": lambda: gym.one_shot(seed, policy, n_trials=60, death_at=None, split=10),
          "fear_v2": lambda: gym.fear_v2(seed, policy),
          "discrimination": lambda: gym.discrimination(seed, policy, n_trials=150),
          "reversal": lambda: gym.discrimination(seed, policy, n_trials=300, reverse_at=150),
          "courage": lambda: gym.courage(seed, policy, n_trials=300)}[task]
    recs = fn()
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, (np.floating, np.bool_)):
                r[k] = v.item()
    c = grab["a"].conn
    probe = dict(v_place=float(c.v[FEATURES.index("place_cue")]), gated=len(c.gated_steps),
                 gated_neg=sum(d < 0 for d, _, _ in c.gated_steps))
    return dict(task=task, policy=policy, seed=seed, recs=recs, probe=probe)


res = pmap(one, [(t, p, s) for t in TASKS for p in POLICIES for s in SEEDS])
R, PR = {}, {}
for r in res:
    R.setdefault((r["task"], r["policy"]), []).append(r["recs"])
    PR.setdefault((r["task"], r["policy"]), []).append(r["probe"])
with open(OUT / "records.json", "w") as fh:
    json.dump({f"{t}|{p}": v for (t, p), v in R.items()}, fh)
with open(OUT / "probes.json", "w") as fh:
    json.dump({f"{t}|{p}": v for (t, p), v in PR.items()}, fh)


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def fmt(m):
    return f"{m[0]:+.3f} ± {m[1]:.3f}"


def rate(recs, lo, hi, bad, pair):
    sel = [r for r in recs[lo:hi] if r["first"] in pair]
    return np.mean([r["first"] == bad for r in sel]) if sel else np.nan


def after_death(recs, k):
    """Landmark-first rate in the k trials after the death (death runs) -- or, for a
    control run, over the same trial window as its paired death run."""
    sel = [r for r in recs if r.get("since_death") is not None and 1 <= r["since_death"] <= k]
    return sel


def courage_index(recs, lo, hi):
    blk = recs[lo:hi]
    g = [b["fight_share"] for b in blk if b["advantage"] > 0 and b["fight_share"] is not None
         and not np.isnan(b["fight_share"])]
    b_ = [b["fight_share"] for b in blk if b["advantage"] <= 0 and b["fight_share"] is not None
          and not np.isnan(b["fight_share"])]
    return np.mean(g) - np.mean(b_) if g and b_ else np.nan


def trials_to_criterion(recs, hi=100, run=5):
    """First acquisition trial after which the ambush site is avoided `run` times in a row."""
    streak = 0
    for r in recs[:hi]:
        streak = streak + 1 if r["first"] == "safe" else 0 if r["first"] == "cursed" else streak
        if streak >= run:
            return r["trial"] - run + 1
    return hi


lines = ["E17 Surprise-gated value learning (Cl(2,2))", "=" * 43,
         f"{len(SEEDS)} seeds; mean ± 95% CI over seeds.", ""]

# ---- one-shot: death runs vs the same trial window in the paired control run
lines.append("ONE-SHOT: landmark-first rate in the 10 trials after the death, death run minus paired control")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for p in POLICIES:
    diffs, d_abs, c_abs, curves = [], [], [], []
    for dr, cr in zip(R[("one_shot", p)], R[("one_shot_control", p)]):
        sel = after_death(dr, 10)
        if not sel:
            continue
        idx = [r["trial"] for r in sel]
        d = np.mean([r["first"] == "marked" for r in sel if r["first"] in ("marked", "unmarked")] or [np.nan])
        c = np.mean([cr[i]["first"] == "marked" for i in idx if cr[i]["first"] in ("marked", "unmarked")]
                    or [np.nan])
        diffs.append(d - c)
        d_abs.append(d)
        c_abs.append(c)
        k0 = sel[0]["trial"] - 1
        curves.append([float(dr[k0 + j]["first"] == "marked") - float(cr[k0 + j]["first"] == "marked")
                       if k0 + j < len(dr) else np.nan for j in range(-5, 30)])
    vp = ci([q["v_place"] for q in PR[("one_shot", p)]])
    vpc = ci([q["v_place"] for q in PR[("one_shot_control", p)]])
    lines.append(f"   {p:<36s} death {np.nanmean(d_abs):.3f}  control {np.nanmean(c_abs):.3f}  "
                 f"paired diff {fmt(ci(diffs))}   (n={len(diffs)})   landmark value at end: death "
                 f"{fmt(vp)}, control {fmt(vpc)}")
    if curves:
        cm = np.nanmean(np.array(curves), axis=0)
        k = 5
        sm = np.convolve(np.nan_to_num(cm), np.ones(k) / k, mode="same")
        axes[0].plot(np.arange(-5, 30), sm, color=COLORS[p], label=p)
axes[0].axvline(0.5, color="k", ls="--", lw=0.8)
axes[0].axhline(0, color="k", lw=0.5)
axes[0].set_title("one-shot: landmark-first, death − control (5-trial mean); death after trial 0")
axes[0].set_xlabel("trials since the death")
axes[0].legend(fontsize=7)

# ---- fear_v2 low-shot
lines += ["", "FEAR v2: ambush-site rate, acquisition trials 1-10 and last 30; extinction last 30; "
          "trials to criterion (5 avoidances in a row)"]
for p in POLICIES:
    xs = np.arange(0, 200, 10)
    ys = np.array([[rate(r, x, x + 10, "cursed", ("cursed", "safe")) for x in xs] for r in R[("fear_v2", p)]])
    axes[1].plot(xs + 5, np.nanmean(ys, axis=0), color=COLORS[p], label=p)
    a10 = ci([rate(r, 0, 10, "cursed", ("cursed", "safe")) for r in R[("fear_v2", p)]])
    a30 = ci([rate(r, 70, 100, "cursed", ("cursed", "safe")) for r in R[("fear_v2", p)]])
    e30 = ci([rate(r, 170, 200, "cursed", ("cursed", "safe")) for r in R[("fear_v2", p)]])
    ttc = ci([trials_to_criterion(r) for r in R[("fear_v2", p)]])
    lines.append(f"   {p:<36s} acq 1-10 {fmt(a10)}   acq last 30 {fmt(a30)}   ext last 30 {fmt(e30)}   "
                 f"trials to criterion {ttc[0]:.1f} ± {ttc[1]:.1f}")
axes[1].axvline(100, color="k", ls="--", lw=0.8)
axes[1].set_title("fear v2: P(first patch is the ambush site)")
axes[1].set_ylim(-0.02, 1.02)

# ---- no harm
lines += ["", "NO HARM: discrimination error (last 30), reversal error (just after / late), courage index "
          "and payoff (last 100)"]
for p in POLICIES:
    d = ci([rate(r, 120, 150, "false", ("real", "false")) for r in R[("discrimination", p)]])
    ja = ci([rate(r, 150, 180, "false", ("real", "false")) for r in R[("reversal", p)]])
    la = ci([rate(r, 270, 300, "false", ("real", "false")) for r in R[("reversal", p)]])
    ci_ = ci([courage_index(r, 200, 300) for r in R[("courage", p)]])
    py = ci([np.mean([b["payoff"] for b in r[200:]]) for r in R[("courage", p)]])
    g = ci([q["gated"] for q in PR[("courage", p)]])
    lines.append(f"   {p:<36s} discr {fmt(d)}   rev just after {fmt(ja)} late {fmt(la)}   courage {fmt(ci_)}"
                 f"   payoff {fmt(py)}   gated steps (courage) {g[0]:.0f}")
    xs = np.arange(0, 300, 20)
    ys = np.array([[rate(r, x, x + 20, "false", ("real", "false")) for x in xs] for r in R[("reversal", p)]])
    axes[2].plot(xs + 10, np.nanmean(ys, axis=0), color=COLORS[p], label=p)
axes[2].axvline(150, color="k", ls="--", lw=0.8)
axes[2].set_title("reversal: P(first patch is false); rule swaps at 150")
axes[2].set_ylim(-0.02, 1.02)
save(fig, OUT, "surprise_gated.png")

lines += ["", "GATED (surprise) VS UNGATED, paired by seed"]
for base in ("clifford_pav_drive", "clifford_pav_drive_affect"):
    gtd = base + "_surprise"
    f10 = [rate(a, 0, 10, "cursed", ("cursed", "safe")) - rate(b, 0, 10, "cursed", ("cursed", "safe"))
           for a, b in zip(R[("fear_v2", gtd)], R[("fear_v2", base)])]
    ttc = [trials_to_criterion(a) - trials_to_criterion(b) for a, b in zip(R[("fear_v2", gtd)], R[("fear_v2", base)])]
    lines.append(f"   {gtd} − {base}:  fear acq 1-10 {fmt(ci(f10))}   trials to criterion {fmt(ci(ttc))}")
write_summary(OUT, lines)
