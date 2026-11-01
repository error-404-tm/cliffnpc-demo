"""E28 Less estimation noise: paired common-random-number perturbations (SPSA), rung 5.

The agent's estimate of its action gradient A was the bottleneck in E24-E27.  The
"spsa" estimator runs pairs of episodes theta + D and theta - D with the same random
streams and no within-episode noise, so the environment's noise cancels in J+ - J-.

World: rung 5 (need 1, depletion 0.5, rivals 1), the E27 competition world.
Grid: estimator regress / spsa x class global / ctx x rule pg / ng / pareto x
preference balanced / cautious x 6 seeds x 300 episodes.  Scores are fractions of the
re-evaluated best (top candidates of the fixed and situation-dependent panels re-run).

Criteria (fixed before running):
  N1  spsa beats regress (paired by class, rule, preference, seed)
  N2  with spsa, a geometry rule (ng or pareto) beats pg -- the thesis, with less noise
  N3  with spsa, ctx beats global
  N4  with spsa, some cell reaches >= 0.8 of the best for each preference
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pickle  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

from cliffnpc.commons import GOALS, Dials  # noqa: E402
from cliffnpc.harness import Sizes, merge_parts, reevaluate_best, run_ctx_panel, world_parts, world_run  # noqa: E402
from cliffnpc.phase_c import GOAL_SCALE, PREFERENCES, GoalGeometryAgent  # noqa: E402

OUT = outdir("e28_noise_fix")
SIZES = Sizes()
RUNG5 = Dials(need=1.0, depletion=0.5, rivals=1.0)
ESTIMATORS = ("regress", "spsa")
CLASSES = ("global", "ctx")
RULES = ("pg", "ng", "pareto")
PREFS = {p: PREFERENCES[p] for p in ("balanced", "cautious")}
SEEDS = tuple(range(6))
EPISODES = 300


def frontier_part(part):
    return world_run(RUNG5, 0, SIZES, part)


e27 = OUT.parent / "e27_rung5_competition" / "runs.pkl"
if e27.exists():
    with open(e27, "rb") as fh:
        runs27, J_ctx = pickle.load(fh)
    J_glob = runs27["rivals=1"]["J"]
else:
    J_glob = merge_parts(pmap(frontier_part, [p for p in world_parts(SIZES) if p.startswith("frontier")]), SIZES)["J"]
    J_ctx = run_ctx_panel(RUNG5, pmap=pmap)
BEST = reevaluate_best(RUNG5, J_glob, J_ctx, PREFS, GOAL_SCALE, pmap=pmap)


def one(job):
    est, cls, rule, pref, seed = job
    ag = GoalGeometryAgent(RUNG5, PREFS[pref], rule=rule, seed=seed, policy=cls, estimator=est)
    ag.run(EPISODES)
    final = ag.evaluate(16).mean(0) / GOAL_SCALE
    return dict(est=est, cls=cls, rule=rule, pref=pref, seed=seed, final=final,
                curve=[(h["episode"], h["score"]) for h in ag.history])


agents = pmap(one, [(e, c, r, p, s) for e in ESTIMATORS for c in CLASSES for r in RULES for p in PREFS for s in SEEDS])
with open(OUT / "agents.pkl", "wb") as fh:
    pickle.dump(agents, fh)


def frac(a):
    b = BEST[a["pref"]]
    return (a["final"] @ PREFS[a["pref"]] - b["panel_mean"]) / (b["score"] - b["panel_mean"])


def sel(**kw):
    return [a for a in agents if all(a[k] == v for k, v in kw.items())]


def key(a):
    return (a["cls"], a["rule"], a["pref"], a["seed"])


def paired(xs, ys):
    yd = {key(y): frac(y) for y in ys}
    return [frac(x) - yd[key(x)] for x in xs if key(x) in yd]


def pairedk(xs, ys, drop):
    k2 = lambda a: tuple(v for k, v in zip(("cls", "rule", "pref", "seed"), key(a)) if k != drop)  # noqa: E731
    yd = {k2(y): frac(y) for y in ys}
    return [frac(x) - yd[k2(x)] for x in xs if k2(x) in yd]


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def fmt(m):
    return f"{m[0]:+.2f} ± {m[1]:.2f}"


lines = ["E28 Less estimation noise: paired CRN perturbations (SPSA), rung 5", "=" * 64,
         f"{len(SEEDS)} seeds x {EPISODES} episodes; fraction of the re-evaluated best.", ""]
for p, b in BEST.items():
    lines.append(f"   best {p}: {b['score']:+.2f} ({b['kind']} #{b['index']}), panel mean {b['panel_mean']:+.2f}")
lines.append("")
for pref in PREFS:
    lines.append(f"## {pref}")
    for est in ESTIMATORS:
        for cls in CLASSES:
            lines.append(f"   {est:<8s}{cls:<7s} " + "   ".join(
                f"{r} {fmt(ci([frac(a) for a in sel(est=est, cls=cls, rule=r, pref=pref)]))}" for r in RULES))
d_est = paired(sel(est="spsa"), sel(est="regress"))
sp = sel(est="spsa")
d_ng = pairedk([a for a in sp if a["rule"] == "ng"], [a for a in sp if a["rule"] == "pg"], "rule")
d_pa = pairedk([a for a in sp if a["rule"] == "pareto"], [a for a in sp if a["rule"] == "pg"], "rule")
d_ctx = pairedk([a for a in sp if a["cls"] == "ctx"], [a for a in sp if a["cls"] == "global"], "cls")
best_cell = {p: max(np.mean([frac(a) for a in sel(est="spsa", cls=c, rule=r, pref=p)]) for c in CLASSES for r in RULES)
             for p in PREFS}
checks = [
    (f"N1 spsa beats regress ({fmt(ci(d_est))})", ci(d_est)[0] - ci(d_est)[1] > 0),
    (f"N2 geometry rule beats pg with spsa (ng - pg {fmt(ci(d_ng))}; pareto - pg {fmt(ci(d_pa))})",
     ci(d_ng)[0] - ci(d_ng)[1] > 0 or ci(d_pa)[0] - ci(d_pa)[1] > 0),
    (f"N3 ctx beats global with spsa ({fmt(ci(d_ctx))})", ci(d_ctx)[0] - ci(d_ctx)[1] > 0),
    ("N4 best cell >= 0.8: " + ", ".join(f"{p} {v:.2f}" for p, v in best_cell.items()), all(v >= 0.8 for v in best_cell.values())),
]
lines += ["", "CRITERIA (fixed before running)"] + [f"   {'PASS' if ok else 'FAIL'}  {lab}" for lab, ok in checks]

fig, axes = plt.subplots(1, len(PREFS), figsize=(6 * len(PREFS), 4))
for ax, pref in zip(axes, PREFS):
    for est, ls in (("regress", ":"), ("spsa", "-")):
        for rule, col in zip(RULES, ("#7f8c8d", "#c0392b", "#2980b9")):
            C = [np.array(a["curve"]) for a in sel(est=est, cls="ctx", rule=rule, pref=pref)]
            grid = np.arange(0, EPISODES, 10)
            ys = np.array([np.interp(grid, c[:, 0], np.convolve(c[:, 1], np.ones(5) / 5, mode="same")) for c in C])
            ax.plot(grid, ys.mean(0), ls, color=col, label=f"{est} {rule}")
    ax.axhline(BEST[pref]["score"], color="k", ls="--", lw=0.8)
    ax.set_title(f"{pref} (ctx agents): score vs episodes", fontsize=9)
axes[0].legend(fontsize=7)
save(fig, OUT, "learning_curves.png")
write_summary(OUT, lines)
