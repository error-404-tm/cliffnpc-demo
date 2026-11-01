"""E29 Ladder rung 6: allies come to defend (strength in numbers).

World: rung 5 (need 1, depletion 0.5, rivals 1) + two extra allies (rung 4 dial at 1),
plus the `defend` dial: 0 = allies join only if already near (as before); 0.5 = the
skilled forager comes to the learner's defence when a rival is within 15; 1 = every
ally comes.  When backed, fighting for food can pay, so the frontier should change shape.

Part A (harness): defend 0 vs 0.5 and 1.
Part B: fixed-theta and situation-dependent panels in the defend=1 world, top candidates
  re-evaluated (unbiased best per preference).
Part C: agents in the defend=1 world: class global / ctx x rule pg / ng / pareto x 4
  preferences x 6 seeds x 300 episodes, estimator from E28 (AGENT_ESTIMATOR).

Criteria (fixed before running):
  R1  defend changes the frontier's shape (a shape effect, not only local) at defend=1
  R2  a geometry rule (ng or pareto) beats pg, pooled
  R3  ctx beats global (fight when backed is situation-dependent)
  R4  absorption: some cell >= 0.8 of the best for each preference
  R5  preferences differentiate (ctx): cautious safer than forager, forager more fed
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pickle  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, pmap, write_summary  # noqa: E402

from cliffnpc.commons import GOALS, PHI_NAMES, PRIMITIVES, Dials  # noqa: E402
from cliffnpc.harness import (DIALS, PAIR_NAMES, Sizes, compare, merge_parts, reevaluate_best,  # noqa: E402
                              run_ctx_panel, world_for, world_parts, world_run)
from cliffnpc.phase_c import GOAL_SCALE, PREFERENCES, GoalGeometryAgent  # noqa: E402

OUT = outdir("e29_rung6_defend")
SIZES = Sizes()
EST = os.environ.get("AGENT_ESTIMATOR", "spsa")
BASE6 = Dials(need=1.0, depletion=0.5, rivals=1.0, allies=1.0)
WORLDS = {"defend=0": BASE6, "defend=0.5": world_for("defend", 0.5, BASE6), "defend=1": world_for("defend", 1.0, BASE6)}
RUNG6 = WORLDS["defend=1"]
CLASSES = ("global", "ctx")
RULES = ("pg", "ng", "pareto")
SEEDS = tuple(range(6))
EPISODES = 300


def run_part(job):
    return job[0], world_run(WORLDS[job[0]], 0, SIZES, job[1])


def cmp_job(name):
    return name, compare(runs["defend=0"], runs[name], SIZES, n_boot=100, seed=1, targets=DIALS["defend"].targets)


pk = OUT / "runs.pkl"
if os.environ.get("E29_REUSE") and pk.exists():
    with open(pk, "rb") as fh:
        runs, J_ctx = pickle.load(fh)
else:
    parts = {}
    for name, res in pmap(run_part, [(n, p) for n in WORLDS for p in world_parts(SIZES)]):
        parts.setdefault(name, []).append(res)
    runs = {n: merge_parts(v, SIZES) for n, v in parts.items()}
    J_ctx = run_ctx_panel(RUNG6, pmap=pmap)
    with open(pk, "wb") as fh:
        pickle.dump((runs, J_ctx), fh)
CMP = dict(pmap(cmp_job, ["defend=0.5", "defend=1"]))
BEST = reevaluate_best(RUNG6, runs["defend=1"]["J"], J_ctx, PREFERENCES, GOAL_SCALE, pmap=pmap)


def one(job):
    cls, rule, pref, seed = job
    ag = GoalGeometryAgent(RUNG6, PREFERENCES[pref], rule=rule, seed=seed, policy=cls, estimator=EST)
    ag.run(EPISODES)
    final = ag.evaluate(16).mean(0) / GOAL_SCALE
    return dict(cls=cls, rule=rule, pref=pref, seed=seed, final=final,
                Theta=(ag.Theta.copy() if cls == "ctx" else ag.theta.copy()))


agents = pmap(one, [(c, r, p, s) for c in CLASSES for r in RULES for p in PREFERENCES for s in SEEDS])
with open(OUT / "agents.pkl", "wb") as fh:
    pickle.dump(agents, fh)


def frac(a):
    b = BEST[a["pref"]]
    return (a["final"] @ PREFERENCES[a["pref"]] - b["panel_mean"]) / (b["score"] - b["panel_mean"])


def sel(**kw):
    return [a for a in agents if all(a[k] == v for k, v in kw.items())]


def pairedk(xs, ys, drop):
    names = ("cls", "rule", "pref", "seed")
    k2 = lambda a: tuple(a[k] for k in names if k != drop)  # noqa: E731
    yd = {k2(y): frac(y) for y in ys}
    return [frac(x) - yd[k2(x)] for x in xs if k2(x) in yd]


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def fmt(m):
    return f"{m[0]:+.2f} ± {m[1]:.2f}"


def fmt_ci(d, c, k):
    return f"{d[k]:+.3f} [{c[0][k]:+.3f},{c[1][k]:+.3f}]"


lines = ["E29 Ladder rung 6: allies come to defend", "=" * 40, f"agent estimator: {EST}", "",
         "PART A: what defence does to the geometry (vs defend 0)"]
for name, r in CMP.items():
    fr = r["frontier"]
    lines.append(f"## defend=0 -> {name}")
    lines.append("   extent, average: " + "  ".join(f"{g} {fmt_ci(fr['mean']['d'], fr['mean']['ci'], k)}" for k, g in enumerate(GOALS)))
    lines.append("   frontier gap (smoothed/raw): " + "  ".join(
        f"{p} {fr['gap_s']['d'][k]:+.2f}/{fr['gap_r']['d'][k]:+.2f}{'*' if r['gap_sig'][k] else ''}" for k, p in enumerate(PAIR_NAMES)))
    lines.append("   frontier corr (smoothed/raw): " + "  ".join(
        f"{p} {fr['corr_s']['d'][k]:+.2f}/{fr['corr_r']['d'][k]:+.2f}{'*' if r['corr_sig'][k] else ''}" for k, p in enumerate(PAIR_NAMES)))
    lines.append("   local: families (of 5) with a significant cosine change: " + "  ".join(
        f"{p} {r['local_count'][k]}" for k, p in enumerate(PAIR_NAMES)))
    kinds = []
    for kk, lab, names in (("extent", "extent", GOALS), ("shape", "shape", PAIR_NAMES),
                           ("local_robust", "local", PAIR_NAMES), ("new_dim", "new dimension", GOALS)):
        if r[kk].any():
            kinds.append(f"{lab} (" + ",".join(n for n, v in zip(names, r[kk]) if v) + ")")
    lines.append("   KINDS: " + ("; ".join(kinds) if kinds else "none") + "   flags: " + (", ".join(r["flags"]) or "none"))
lines += ["", "PART B: best policies in the defend=1 world (re-evaluated)"]
for p, b in BEST.items():
    lines.append(f"   {p:<9s} best {b['score']:+.2f} ({b['kind']} #{b['index']})   fixed {b['best_global']:+.2f}   "
                 f"situation-dependent {b['best_ctx']:+.2f}   panel mean {b['panel_mean']:+.2f}")
lines += ["", "PART C: agents (fraction of the re-evaluated best)"]
for pref in PREFERENCES:
    lines.append(f"## {pref}")
    for cls in CLASSES:
        J1 = np.mean([a["final"] for a in sel(cls=cls, pref=pref)], axis=0)
        lines.append(f"   {cls:<7s} " + "   ".join(f"{r} {fmt(ci([frac(a) for a in sel(cls=cls, rule=r, pref=pref)]))}"
                                              for r in RULES) + "   J/scale " + " ".join(f"{g} {v:+.2f}" for g, v in zip(GOALS, J1)))
th = np.mean([a["Theta"] for a in sel(cls="ctx", pref="balanced")], axis=0)
lines += ["", "Learned situation-dependent policy (ctx, balanced, all rules): rows primitives, columns " + ", ".join(PHI_NAMES)]
lines += [f"   {p:<8s} " + " ".join(f"{v:+.2f}" for v in th[i]) for i, p in enumerate(PRIMITIVES)]

r1 = CMP["defend=1"]
d_ng = pairedk(sel(rule="ng"), sel(rule="pg"), "rule")
d_pa = pairedk(sel(rule="pareto"), sel(rule="pg"), "rule")
d_ctx = pairedk(sel(cls="ctx"), sel(cls="global"), "cls")
best_cell = {p: max(np.mean([frac(a) for a in sel(cls=c, rule=r, pref=p)]) for c in CLASSES for r in RULES) for p in PREFERENCES}
fin = {p: np.mean([a["final"] for a in sel(cls="ctx", pref=p)], axis=0) for p in PREFERENCES}
checks = [
    ("R1 defend changes the frontier's shape", bool(r1["shape"].any())),
    (f"R2 geometry rule beats pg (ng - pg {fmt(ci(d_ng))}; pareto - pg {fmt(ci(d_pa))})",
     ci(d_ng)[0] - ci(d_ng)[1] > 0 or ci(d_pa)[0] - ci(d_pa)[1] > 0),
    (f"R3 ctx beats global ({fmt(ci(d_ctx))})", ci(d_ctx)[0] - ci(d_ctx)[1] > 0),
    ("R4 absorption: " + ", ".join(f"{p} {v:.2f}" for p, v in best_cell.items()), all(v >= 0.8 for v in best_cell.values())),
    ("R5 preferences differentiate (ctx: cautious safer, forager more fed)",
     fin["cautious"][1] > fin["forager"][1] and fin["forager"][0] > fin["cautious"][0]),
]
lines += ["", "CRITERIA (fixed before running)"] + [f"   {'PASS' if ok else 'FAIL'}  {lab}" for lab, ok in checks]
write_summary(OUT, lines)
