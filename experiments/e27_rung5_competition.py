"""E27 Ladder rung 5: competition -- the first rung expected to bend the frontier.

World: rung 1 (need 1) + depletion 0.5, plus rival foragers of the other tribe (dial
`rivals`: 0 / 0.5 -> one / 1 -> two).  They know the real patches, eat the same stock,
and contact is combat under the arena's n/(n+m) rule (allies near join).

Part A (harness): rivals 0 vs 0.5 and 1 -- does competition bend the frontier?
Part B (frontier panels in the rivals=1 world): the fixed-theta panel (192 x 4) and a
  situation-dependent panel (192 Theta x 4); for each preference the top 10 candidates of
  both are re-evaluated with 16 fresh episodes, so "best" is not inflated by luck.
Part C (agents in the rivals=1 world): policy class global / ctx x rule pg / ng
  (stabilised) / pareto x 4 preferences x 6 seeds x 200 episodes.

Criteria (fixed before running):
  C1  rivals bend the frontier (shape or robust local change at rivals=1)
  C2  ctx agents beat global agents (paired, pooled over rules, preferences, seeds)
  C3  stabilised ng never collapses (no cell below 0 of best) and is not significantly
      worse than pg
  C4  a geometry rule (ng or pareto) beats pg, pooled -- the thesis test
  C5  absorption: some (class, rule) reaches >= 0.8 of the re-evaluated best, every preference
  C6  preferences differentiate (ctx agents): cautious safer than forager, forager more fed
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pickle  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

from cliffnpc.commons import GOALS, PHI_NAMES, PRIMITIVES, Commons, Dials  # noqa: E402
from cliffnpc.harness import (DIALS, PAIR_NAMES, Sizes, compare, ctx_panel, frontier_panel,  # noqa: E402
                              merge_parts, world_for, world_parts, world_run)
from cliffnpc.phase_c import GOAL_SCALE, PREFERENCES, GoalGeometryAgent, connection  # noqa: E402

OUT = outdir("e27_rung5_competition")
SIZES = Sizes()
BASE5 = Dials(need=1.0, depletion=0.5)
WORLDS = {"rivals=0": BASE5, "rivals=0.5": world_for("rivals", 0.5, BASE5), "rivals=1": world_for("rivals", 1.0, BASE5)}
RUNG5 = WORLDS["rivals=1"]
CLASSES = ("global", "ctx")
RULES = ("pg", "ng", "pareto")
SEEDS = tuple(range(6))
EPISODES = 200
N_CTX, CTX_CHUNKS, TOP = 192, 12, 10
CTX = ctx_panel(N_CTX)
THETA = frontier_panel(SIZES.n_frontier)


# ---- part A
def run_part(job):
    return job[0], world_run(WORLDS[job[0]], 0, SIZES, job[1])


def cmp_job(name):
    return name, compare(runs["rivals=0"], runs[name], SIZES, n_boot=100, seed=1, targets=DIALS["rivals"].targets)


pk = OUT / "runs.pkl"
if os.environ.get("E27_REUSE") and pk.exists():
    with open(pk, "rb") as fh:
        runs, J_ctx = pickle.load(fh)
else:
    parts = {}
    for name, res in pmap(run_part, [(n, p) for n in WORLDS for p in world_parts(SIZES)]):
        parts.setdefault(name, []).append(res)
    runs = {n: merge_parts(v, SIZES) for n, v in parts.items()}

    def ctx_part(k):
        idx = np.arange(k, N_CTX, CTX_CHUNKS)
        return idx, Commons(RUNG5, seed=500 + k, noise_override=0.2).run_ctx_policies(CTX[idx], 4)
    J_ctx = np.zeros((N_CTX, 4, len(GOALS)))
    for idx, out in pmap(ctx_part, range(CTX_CHUNKS)):
        J_ctx[idx] = out
    with open(pk, "wb") as fh:
        pickle.dump((runs, J_ctx), fh)
CMP = dict(pmap(cmp_job, ["rivals=0.5", "rivals=1"]))

# ---- part B: re-evaluate the top candidates per preference
Jg = runs["rivals=1"]["J"].mean(1) / GOAL_SCALE
Jc = J_ctx.mean(1) / GOAL_SCALE
cands = []
for pref, w in PREFERENCES.items():
    for kind, Jm in (("global", Jg), ("ctx", Jc)):
        for i in np.argsort(-(Jm @ w))[:TOP]:
            cands.append((pref, kind, int(i)))


def reeval(job):
    pref, kind, i = job
    c = Commons(RUNG5, seed=900, noise_override=0.2)
    J = (c.run_policies(THETA[i][None], 16) if kind == "global" else c.run_ctx_policies(CTX[i][None], 16))[0]
    return job, J.mean(0) / GOAL_SCALE


RE = dict(pmap(reeval, sorted(set(cands))))
best = {}
for pref, w in PREFERENCES.items():
    vals = {k: v @ w for k, v in RE.items() if k[0] == pref}
    kbest = max(vals, key=vals.get)
    best[pref] = dict(score=float(vals[kbest]), kind=kbest[1], index=kbest[2], J=RE[kbest],
                      best_global=float(max(v for k, v in vals.items() if k[1] == "global")),
                      best_ctx=float(max(v for k, v in vals.items() if k[1] == "ctx")),
                      panel_mean=float((Jg @ w).mean()))


# ---- part C: agents
def one(job):
    cls, rule, pref, seed = job
    ag = GoalGeometryAgent(RUNG5, PREFERENCES[pref], rule=rule, seed=seed, policy=cls)
    start = ag.evaluate(16).mean(0) / GOAL_SCALE
    ag.run(EPISODES)
    final = ag.evaluate(16).mean(0) / GOAL_SCALE
    rep = ag.report()
    return dict(cls=cls, rule=rule, pref=pref, seed=seed, start=start, final=final,
                Theta=(ag.Theta.copy() if cls == "ctx" else ag.theta.copy()),
                scores=[h["score"] for h in ag.history], planes=rep["planes"], conn=connection(ag.history, 20))


agents = pmap(one, [(c, r, p, s) for c in CLASSES for r in RULES for p in PREFERENCES for s in SEEDS])
with open(OUT / "agents.pkl", "wb") as fh:
    pickle.dump(agents, fh)


def sel(cls=None, rule=None, pref=None):
    return [a for a in agents if (cls is None or a["cls"] == cls) and (rule is None or a["rule"] == rule)
            and (pref is None or a["pref"] == pref)]


def frac(a):
    b = best[a["pref"]]
    return (a["final"] @ PREFERENCES[a["pref"]] - b["panel_mean"]) / (b["score"] - b["panel_mean"])


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def fmt(m):
    return f"{m[0]:+.2f} ± {m[1]:.2f}"


def fmt_ci(d, c, k):
    return f"{d[k]:+.3f} [{c[0][k]:+.3f},{c[1][k]:+.3f}]"


lines = ["E27 Ladder rung 5: competition", "=" * 30, "", "PART A: what rival foragers do to the geometry (vs rivals 0)"]
for name, r in CMP.items():
    fr = r["frontier"]
    lines.append(f"## rivals=0 -> {name}")
    lines.append("   extent, average: " + "  ".join(f"{g} {fmt_ci(fr['mean']['d'], fr['mean']['ci'], k)}" for k, g in enumerate(GOALS)))
    lines.append("   frontier gap (smoothed/raw): " + "  ".join(
        f"{p} {fr['gap_s']['d'][k]:+.2f}/{fr['gap_r']['d'][k]:+.2f}{'*' if r['gap_sig'][k] else ''}" for k, p in enumerate(PAIR_NAMES)))
    lines.append("   frontier corr (smoothed/raw): " + "  ".join(
        f"{p} {fr['corr_s']['d'][k]:+.2f}/{fr['corr_r']['d'][k]:+.2f}{'*' if r['corr_sig'][k] else ''}" for k, p in enumerate(PAIR_NAMES)))
    lines.append("   local: families (of 5) with a significant cosine change: " + "  ".join(
        f"{p} {r['local_count'][k]}" for k, p in enumerate(PAIR_NAMES)))
    kinds = []
    for key, lab, names in (("extent", "extent", GOALS), ("shape", "shape", PAIR_NAMES),
                            ("local_robust", "local", PAIR_NAMES), ("new_dim", "new dimension", GOALS)):
        if r[key].any():
            kinds.append(f"{lab} (" + ",".join(n for n, v in zip(names, r[key]) if v) + ")")
    lines.append("   KINDS: " + ("; ".join(kinds) if kinds else "none") + "   flags: " + (", ".join(r["flags"]) or "none"))
lines += ["", "PART B: best policies in the rivals=1 world (top 10 per panel re-evaluated, 16 episodes)"]
for pref, b in best.items():
    lines.append(f"   {pref:<9s} best {b['score']:+.2f} ({b['kind']} #{b['index']})   best fixed-theta {b['best_global']:+.2f}   "
                 f"best situation-dependent {b['best_ctx']:+.2f}   panel mean {b['panel_mean']:+.2f}")
lines += ["", "PART C: agents in the rivals=1 world (fraction of the re-evaluated best)"]
for pref in PREFERENCES:
    lines.append(f"## {pref}")
    for cls in CLASSES:
        row = []
        for rule in RULES:
            rs = sel(cls, rule, pref)
            row.append(f"{rule} {fmt(ci([frac(a) for a in rs]))}")
        J1 = np.mean([a["final"] for a in sel(cls, None, pref)], axis=0)
        lines.append(f"   {cls:<7s} " + "   ".join(row) + "   J/scale " + " ".join(f"{g} {v:+.2f}" for g, v in zip(GOALS, J1)))
th = np.mean([a["Theta"] for a in sel("ctx", "pareto", "balanced")], axis=0)
lines += ["", "Learned situation-dependent policy (ctx, pareto, balanced; mean over seeds): rows primitives, "
          "columns " + ", ".join(PHI_NAMES)]
lines += [f"   {p:<8s} " + " ".join(f"{v:+.2f}" for v in th[i]) for i, p in enumerate(PRIMITIVES)]

# ---- criteria
r1 = CMP["rivals=1"]
d_ctx = [frac(a) - frac(b) for a, b in zip(sel("ctx"), sel("global"))]
cells_ng = [np.mean([frac(a) for a in sel(c, "ng", p)]) for c in CLASSES for p in PREFERENCES]
d_ngpg = [frac(a) - frac(b) for c in CLASSES for a, b in zip(sel(c, "ng"), sel(c, "pg"))]
d_papg = [frac(a) - frac(b) for c in CLASSES for a, b in zip(sel(c, "pareto"), sel(c, "pg"))]
best_cell = {p: max(np.mean([frac(a) for a in sel(c, r, p)]) for c in CLASSES for r in RULES) for p in PREFERENCES}
fin = {p: np.mean([a["final"] for a in sel("ctx", None, p)], axis=0) for p in PREFERENCES}
checks = [
    ("C1 rivals bend the frontier", bool(r1["shape"].any() or r1["local_robust"].any())),
    (f"C2 ctx beats global ({fmt(ci(d_ctx))})", ci(d_ctx)[0] - ci(d_ctx)[1] > 0),
    (f"C3 stabilised ng: worst cell {min(cells_ng):+.2f}, ng - pg {fmt(ci(d_ngpg))}",
     min(cells_ng) >= 0 and ci(d_ngpg)[0] + ci(d_ngpg)[1] >= 0),
    (f"C4 geometry rule beats pg (ng - pg {fmt(ci(d_ngpg))}; pareto - pg {fmt(ci(d_papg))})",
     ci(d_ngpg)[0] - ci(d_ngpg)[1] > 0 or ci(d_papg)[0] - ci(d_papg)[1] > 0),
    ("C5 absorption: " + ", ".join(f"{p} {v:.2f}" for p, v in best_cell.items()), all(v >= 0.8 for v in best_cell.values())),
    ("C6 preferences differentiate (ctx: cautious safer, forager more fed)",
     fin["cautious"][1] > fin["forager"][1] and fin["forager"][0] > fin["cautious"][0]),
]
lines += ["", "CRITERIA (fixed before running)"] + [f"   {'PASS' if ok else 'FAIL'}  {lab}" for lab, ok in checks]

fig, axes = plt.subplots(1, len(PREFERENCES), figsize=(4.5 * len(PREFERENCES), 3.8))
for ax, pref in zip(axes, PREFERENCES):
    for cls, ls in (("global", ":"), ("ctx", "-")):
        for rule, col in zip(RULES, ("#7f8c8d", "#c0392b", "#2980b9")):
            S = np.array([a["scores"] for a in sel(cls, rule, pref)])
            k = 10
            sm = np.array([np.convolve(s, np.ones(k) / k, mode="valid") for s in S]).mean(0)
            ax.plot(np.arange(len(sm)) + k, sm, ls, color=col, label=f"{cls} {rule}")
    ax.axhline(best[pref]["score"], color="k", ls="--", lw=0.8)
    ax.set_title(f"{pref}: score per episode", fontsize=9)
axes[0].legend(fontsize=6)
save(fig, OUT, "learning_curves.png")
write_summary(OUT, lines)
