"""E25 Ladder rung 1: internal need (spec-v2 sections 8 and 9).

Part A (harness): what the need dial does to the induced geometry -- base (need 0 =
rung 0) against need 0.5 and need 1, paired CRN comparisons, four kinds of effect,
construct checks.

Part B (phase C): GoalGeometryAgents learn in the rung-1 world (need 1) with the agent
configuration accepted in E24b, judged against that world's own frontier panel.

Expectations / criteria (fixed before running):
  A1  need changes the food extent (best or average food), and safety / social /
      knowledge extents do not move (need only touches food by construction)
  A2  construct check: any effect on a pair not involving food is flagged
  B1  some rule reaches >= 0.8 of the rung-1 panel's best for each preference (absorption)
  B2  a geometry rule (ng or pareto) beats pg (paired by seed, pooled over preferences)
  B3  preferences differentiate: cautious safer than forager, forager more fed than cautious
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pickle  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

from cliffnpc.commons import GOALS, PRIMITIVES  # noqa: E402
from cliffnpc.geometry import pareto_mask, smooth_returns  # noqa: E402
from cliffnpc.harness import (BASE, DIALS, PAIR_NAMES, Sizes, compare, frontier_panel, merge_parts,  # noqa: E402
                              world_for, world_parts, world_run)
from cliffnpc.phase_c import GOAL_SCALE, PREFERENCES, GoalGeometryAgent, connection  # noqa: E402

OUT = outdir("e25_rung1_need")
SIZES = Sizes()
CRITIC = os.environ.get("AGENT_CRITIC", "td")
FORGET = float(os.environ.get("AGENT_FORGET", "0.98"))
RULES = ("pg", "ng", "pareto")
SEEDS = tuple(range(8))
EPISODES = 200
WORLDS = {"base": BASE, "need=0.5": world_for("need", 0.5), "need=1": world_for("need", 1.0)}
RUNG1 = WORLDS["need=1"]

# ---- part A: harness
def run_part(job):
    return job[0], world_run(WORLDS[job[0]], 0, SIZES, job[1])


def cmp_job(name):
    return name, compare(runs["base"], runs[name], SIZES, n_boot=100, seed=1, targets=DIALS["need"].targets)


pk = OUT / "runs.pkl"
if os.environ.get("E25_REUSE") and pk.exists():
    with open(pk, "rb") as fh:
        runs = pickle.load(fh)
else:
    parts = {}
    for name, res in pmap(run_part, [(n, p) for n in WORLDS for p in world_parts(SIZES)]):
        parts.setdefault(name, []).append(res)
    runs = {n: merge_parts(v, SIZES) for n, v in parts.items()}
    with open(pk, "wb") as fh:
        pickle.dump(runs, fh)
CMP = dict(pmap(cmp_job, ["need=0.5", "need=1"]))


# ---- part B: phase C in the rung-1 world
def one(job):
    rule, pref, seed = job
    ag = GoalGeometryAgent(RUNG1, PREFERENCES[pref], rule=rule, seed=seed, critic=CRITIC, forget=FORGET)
    start = ag.evaluate(16).mean(0) / GOAL_SCALE
    ag.run(EPISODES)
    final = ag.evaluate(16).mean(0) / GOAL_SCALE
    rep = ag.report()
    return dict(rule=rule, pref=pref, seed=seed, start=start, final=final, theta=ag.theta.copy(),
                scores=[h["score"] for h in ag.history], planes=rep["planes"], conn=connection(ag.history, 20))


agents = pmap(one, [(r, p, s) for r in RULES for p in PREFERENCES for s in SEEDS])
THETA = frontier_panel(SIZES.n_frontier)
Jp_raw = runs["need=1"]["J"].mean(1) / GOAL_SCALE
Jp_s = smooth_returns(THETA, runs["need=1"]["J"]) / GOAL_SCALE


def sel(rule=None, pref=None):
    return [a for a in agents if (rule is None or a["rule"] == rule) and (pref is None or a["pref"] == pref)]


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def fmt(m):
    return f"{m[0]:+.2f} ± {m[1]:.2f}"


def fmt_ci(d, c, k):
    return f"{d[k]:+.3f} [{c[0][k]:+.3f},{c[1][k]:+.3f}]"


lines = ["E25 Ladder rung 1: internal need", "=" * 32, f"agent: critic {CRITIC}, forget {FORGET}", "",
         "PART A: what need does to the geometry (paired against the base world)"]
for name, r in CMP.items():
    fr = r["frontier"]
    lines.append(f"## base -> {name}")
    lines.append("   extent, best:    " + "  ".join(f"{g} {fmt_ci(fr['extent']['d'], fr['extent']['ci'], k)}" for k, g in enumerate(GOALS)))
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
    lines.append("   KINDS: " + ("; ".join(kinds) if kinds else "none"))
    lines.append("   construct-check flags: " + (", ".join(r["flags"]) or "none"))
    lines.append("")

lines.append("PART B: phase-C agents in the rung-1 world (need 1)")
panel = {}
for pref, w in PREFERENCES.items():
    sp_raw, sp_s = Jp_raw @ w, Jp_s @ w
    panel[pref] = (float(sp_raw.mean()), float(sp_s.max()))
    lines.append(f"## {pref}  panel mean {sp_raw.mean():.2f}, best smoothed {sp_s.max():.2f} "
                 f"(theta {np.round(THETA[int(np.argmax(sp_s))], 2)})")
    for rule in RULES:
        rs = sel(rule, pref)
        s0, s1 = [a["start"] @ w for a in rs], [a["final"] @ w for a in rs]
        rel = [(x - panel[pref][0]) / (panel[pref][1] - panel[pref][0]) for x in s1]
        dom = [float(np.mean(np.all(Jp_s >= a["final"], axis=1) & np.any(Jp_s > a["final"], axis=1))) for a in rs]
        J1 = np.mean([a["final"] for a in rs], axis=0)
        lines.append(f"   {rule:<7s} start {fmt(ci(s0))}  final {fmt(ci(s1))}  fraction of best {fmt(ci(rel))}  "
                     f"dominated by {fmt(ci(dom))} of panel   J/scale " + " ".join(f"{g} {v:+.2f}" for g, v in zip(GOALS, J1)))
    lines.append("")

best_rel = {p: max(np.mean([(a["final"] @ PREFERENCES[p] - panel[p][0]) / (panel[p][1] - panel[p][0])
                            for a in sel(r, p)]) for r in RULES) for p in PREFERENCES}
d_ng = [a["final"] @ PREFERENCES[a["pref"]] - b["final"] @ PREFERENCES[b["pref"]] for a, b in zip(sel("ng"), sel("pg"))]
d_pa = [a["final"] @ PREFERENCES[a["pref"]] - b["final"] @ PREFERENCES[b["pref"]] for a, b in zip(sel("pareto"), sel("pg"))]
fin = {p: np.mean([a["final"] for r in RULES for a in sel(r, p)], axis=0) for p in PREFERENCES}
r1 = CMP["need=1"]
other_extent = [k for k, g in enumerate(GOALS) if g != "food"]
checks = [
    ("A1 need moves the food extent, not the others",
     bool(r1["extent"][0]) and not any(r1["extent"][k] for k in other_extent)),
    ("A2 construct flags (effects on pairs not involving food): " + (", ".join(p for p in r1["flags"] if "food" not in p) or "none"),
     not any("food" not in p for p in r1["flags"])),
    ("B1 absorption, fraction of best per preference: " + ", ".join(f"{p} {v:.2f}" for p, v in best_rel.items()),
     all(v >= 0.8 for v in best_rel.values())),
    (f"B2 geometry rule beats pg (ng - pg {fmt(ci(d_ng))}; pareto - pg {fmt(ci(d_pa))})",
     ci(d_ng)[0] - ci(d_ng)[1] > 0 or ci(d_pa)[0] - ci(d_pa)[1] > 0),
    ("B3 preferences differentiate (cautious safer, forager more fed)",
     fin["cautious"][1] > fin["forager"][1] and fin["forager"][0] > fin["cautious"][0]),
]
lines += ["CRITERIA (fixed before running)"] + [f"   {'PASS' if ok else 'FAIL'}  {lab}" for lab, ok in checks]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (a, b) in zip(axes, ((1, 0), (2, 0), (3, 0))):
    P = pareto_mask(Jp_s)
    ax.scatter(Jp_s[:, a], Jp_s[:, b], s=4, c="#bdc3c7")
    ax.scatter(Jp_s[P, a], Jp_s[P, b], s=8, c="#7f8c8d", label="rung-1 frontier")
    for pref, mk in zip(PREFERENCES, "osD^"):
        F = np.array([x["final"] for r in RULES for x in sel(r, pref)])
        ax.scatter(F[:, a], F[:, b], marker=mk, s=18, label=pref)
    ax.set_xlabel(GOALS[a] + " / scale")
    ax.set_ylabel(GOALS[b] + " / scale")
axes[0].legend(fontsize=7)
save(fig, OUT, "agents_on_frontier.png")
write_summary(OUT, lines)
