"""E24 Phase C on rung 0: an agent that learns its own induced geometry and moves toward
the frontier (spec-v2 sections 6 and 8.7b).

GoalGeometryAgent (cliffnpc/phase_c.py) learns the vector action gradient A and local
goal metric M = A A^T online from its exploration, and moves its primitive weights
theta by one of three rules: "pg" (plain gradient A^T omega), "ng" (natural gradient
with the pullback metric), "pareto" (improve every valued goal at once first).

Base world (rung 0); four preference profiles; 8 seeds x 200 learning episodes; the
final theta is evaluated over 16 episodes at the frontier panel's noise level and
placed against the base world's frontier panel (192 fixed policies, E23).

Pass criteria (fixed before running):
  P1  learning beats the starting theta (paired by seed), rule ng, every preference
  P2  some rule reaches >= 0.8 of the panel's best (smoothed) score for its preference
  P3  preferences pull agents to different frontier points: cautious safer than forager,
      forager more fed than cautious, curious more knowledgeable than balanced
  P4  a geometry rule (ng or pareto) beats pg (paired by seed, pooled over preferences)
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pickle  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

from cliffnpc.commons import GOALS, PRIMITIVES, Dials  # noqa: E402
from cliffnpc.geometry import pareto_mask, smooth_returns  # noqa: E402
from cliffnpc.harness import BASE, Sizes, frontier_panel, merge_parts, world_parts, world_run  # noqa: E402
from cliffnpc.phase_c import GOAL_SCALE, PREFERENCES, GoalGeometryAgent, connection  # noqa: E402

OUT = outdir("e24_phase_c_rung0")
RULES = ("pg", "ng", "pareto")
SEEDS = tuple(range(8))
EPISODES = 200
EVAL = 16
SIZES = Sizes()

# ---- the base world's frontier panel (from E23 if saved locally, else recomputed)
def base_part(part):
    return world_run(BASE, 0, SIZES, part)



pk = OUT.parent / "e23_harness_validation" / "runs.pkl"
if pk.exists():
    with open(pk, "rb") as fh:
        J_panel = pickle.load(fh)["base"]["J"]
else:
    parts = [p for p in world_parts(SIZES) if p.startswith("frontier")]
    J_panel = merge_parts(pmap(base_part, parts), SIZES)["J"]
THETA_PANEL = frontier_panel(SIZES.n_frontier)
Jp_raw = J_panel.mean(1) / GOAL_SCALE
Jp_s = smooth_returns(THETA_PANEL, J_panel) / GOAL_SCALE


def one(job):
    rule, pref, seed = job
    ag = GoalGeometryAgent(Dials(), PREFERENCES[pref], rule=rule, seed=seed)
    start = ag.evaluate(EVAL).mean(0) / GOAL_SCALE
    ag.run(EPISODES)
    final = ag.evaluate(EVAL).mean(0) / GOAL_SCALE
    rep = ag.report()
    return dict(rule=rule, pref=pref, seed=seed, start=start, final=final,
                scores=[h["score"] for h in ag.history], thetas=np.array([h["theta"] for h in ag.history]),
                planes=rep["planes"], modes=rep["modes"], conn=connection(ag.history, 20),
                theta=ag.theta.copy())


res = pmap(one, [(r, p, s) for r in RULES for p in PREFERENCES for s in SEEDS])
with open(OUT / "agents.pkl", "wb") as fh:
    pickle.dump(res, fh)


def sel(rule=None, pref=None):
    return [r for r in res if (rule is None or r["rule"] == rule) and (pref is None or r["pref"] == pref)]


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def fmt(m):
    return f"{m[0]:+.2f} ± {m[1]:.2f}"


lines = ["E24 Phase C on rung 0 (spec-v2)", "=" * 31,
         f"{len(SEEDS)} seeds x {EPISODES} learning episodes per (rule, preference); final theta evaluated over "
         f"{EVAL} episodes; scores in normalised goal units (omega . J / scale).", ""]
panel = {}
for pref, w in PREFERENCES.items():
    sp_raw, sp_s = Jp_raw @ w, Jp_s @ w
    panel[pref] = dict(mean=float(sp_raw.mean()), best_raw=float(sp_raw.max()), best_s=float(sp_s.max()),
                       best_theta=THETA_PANEL[int(np.argmax(sp_s))])
    lines.append(f"## preference {pref}  omega {w}   panel: mean {sp_raw.mean():.2f}, best raw {sp_raw.max():.2f}, "
                 f"best smoothed {sp_s.max():.2f} (theta {np.round(panel[pref]['best_theta'], 2)})")
    for rule in RULES:
        rs = sel(rule, pref)
        s0 = [r["start"] @ w for r in rs]
        s1 = [r["final"] @ w for r in rs]
        rel = [(x - panel[pref]["mean"]) / (panel[pref]["best_s"] - panel[pref]["mean"]) for x in s1]
        dom = [float(np.mean(np.all(Jp_s >= r["final"], axis=1) & np.any(Jp_s > r["final"], axis=1))) for r in rs]
        lines.append(f"   {rule:<7s} start {fmt(ci(s0))}  final {fmt(ci(s1))}  gain {fmt(ci(np.subtract(s1, s0)))}  "
                     f"fraction of panel best {fmt(ci(rel))}  panel policies dominating it {fmt(ci(dom))}")
        th = np.mean([r["theta"] for r in rs], axis=0)
        J1 = np.mean([r["final"] for r in rs], axis=0)
        lines.append(f"           final theta {dict(zip(PRIMITIVES, np.round(th, 2)))}   final J/scale "
                     f"{dict(zip(GOALS, np.round(J1, 2)))}")
    lines.append("")

# ---- geometry the agents end with
lines.append("LOCAL GEOMETRY AT THE END (rule ng, mean over seeds): plane cosines and areas (a_i ^ a_j)")
for pref in PREFERENCES:
    rs = sel("ng", pref)
    pl = {k: (np.mean([r["planes"][k]["cos"] for r in rs]), np.mean([r["planes"][k]["area"] for r in rs]))
          for k in rs[0]["planes"]}
    lines.append(f"   {pref:<9s} " + "  ".join(f"{k} cos {c:+.2f} area {a:.2f}" for k, (c, a) in pl.items()))
lines.append("CONNECTION (rule ng): change of the normalised local metric per 20 episodes, and theta moved")
for pref in PREFERENCES:
    rs = sel("ng", pref)
    dM = np.mean([[c["dM"] for c in r["conn"]] for r in rs], axis=0)
    dt = np.mean([[c["dtheta"] for c in r["conn"]] for r in rs], axis=0)
    lines.append(f"   {pref:<9s} dM " + " ".join(f"{x:.2f}" for x in dM) + "   dtheta " + " ".join(f"{x:.2f}" for x in dt))

# ---- criteria
checks = []
p1 = []
for pref, w in PREFERENCES.items():
    rs = sel("ng", pref)
    m, h = ci([r["final"] @ w - r["start"] @ w for r in rs])
    p1.append(m - h > 0)
checks.append(("P1 rule ng beats its starting theta for every preference", all(p1)))
best_rel = {}
for pref, w in PREFERENCES.items():
    best_rel[pref] = max(np.mean([(r["final"] @ w - panel[pref]["mean"]) / (panel[pref]["best_s"] - panel[pref]["mean"])
                                  for r in sel(rule, pref)]) for rule in RULES)
checks.append(("P2 some rule reaches >= 0.8 of the panel's best for its preference: "
               + ", ".join(f"{p} {v:.2f}" for p, v in best_rel.items()), all(v >= 0.8 for v in best_rel.values())))
fin = {p: np.mean([r["final"] for r in sel("ng", p)], axis=0) for p in PREFERENCES}
g = {x: i for i, x in enumerate(GOALS)}
p3 = [fin["cautious"][g["safety"]] > fin["forager"][g["safety"]], fin["forager"][g["food"]] > fin["cautious"][g["food"]],
      fin["curious"][g["knowledge"]] > fin["balanced"][g["knowledge"]]]
checks.append((f"P3 preferences differentiate (cautious safer {p3[0]}, forager more fed {p3[1]}, curious more "
               f"knowledgeable {p3[2]})", all(p3)))
diff_ng = [a["final"] @ PREFERENCES[a["pref"]] - b["final"] @ PREFERENCES[b["pref"]]
           for a, b in zip(sel("ng"), sel("pg"))]
diff_pa = [a["final"] @ PREFERENCES[a["pref"]] - b["final"] @ PREFERENCES[b["pref"]]
           for a, b in zip(sel("pareto"), sel("pg"))]
m1, h1 = ci(diff_ng)
m2, h2 = ci(diff_pa)
checks.append((f"P4 a geometry rule beats pg (ng - pg {m1:+.2f} ± {h1:.2f}; pareto - pg {m2:+.2f} ± {h2:.2f})",
               (m1 - h1 > 0) or (m2 - h2 > 0)))
lines += ["", "CRITERIA (fixed before running)"] + [f"   {'PASS' if ok else 'FAIL'}  {lab}" for lab, ok in checks]

# ---- figures
fig, axes = plt.subplots(1, len(PREFERENCES), figsize=(4.5 * len(PREFERENCES), 3.8))
for ax, pref in zip(axes, PREFERENCES):
    for rule, col in zip(RULES, ("#7f8c8d", "#c0392b", "#2980b9")):
        S = np.array([r["scores"] for r in sel(rule, pref)])
        k = 10
        sm = np.array([np.convolve(s, np.ones(k) / k, mode="valid") for s in S])
        ax.plot(np.arange(sm.shape[1]) + k, sm.mean(0), color=col, label=rule)
    ax.axhline(panel[pref]["best_s"], color="k", ls="--", lw=0.8, label="panel best")
    ax.axhline(panel[pref]["mean"], color="k", ls=":", lw=0.8, label="panel mean")
    ax.set_title(f"{pref}: score per episode (10-ep mean)", fontsize=9)
    ax.set_xlabel("episode")
axes[0].legend(fontsize=7)
save(fig, OUT, "learning_curves.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (a, b) in zip(axes, ((1, 0), (2, 0), (3, 0))):
    P = pareto_mask(Jp_s)
    ax.scatter(Jp_s[:, a], Jp_s[:, b], s=4, c="#bdc3c7")
    ax.scatter(Jp_s[P, a], Jp_s[P, b], s=8, c="#7f8c8d", label="panel frontier")
    for pref, mk in zip(PREFERENCES, "osD^"):
        F = np.array([r["final"] for r in sel("ng", pref)])
        ax.scatter(F[:, a], F[:, b], marker=mk, s=30, label=f"ng {pref}")
    ax.set_xlabel(GOALS[a] + " / scale")
    ax.set_ylabel(GOALS[b] + " / scale")
axes[0].legend(fontsize=7)
save(fig, OUT, "agents_on_frontier.png")
write_summary(OUT, lines)
