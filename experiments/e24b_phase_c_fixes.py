"""E24b Two fixes to the phase-C agent, tested on rung 0 before climbing.

  memory    forget 0.98 instead of 0.95 (estimates remember ~50 episodes, not ~20)
  critic    "td": per-tick TD critic with a surprise gate and bootstrapped advantages
            (a rare setback's value appears when its predictor appears) instead of the
            window-return regression baseline ("mc")

Configurations: old (mc, 0.95), memory (mc, 0.98), critic (td, 0.95), both (td, 0.98);
preferences cautious and balanced; rules pg / ng / pareto; 8 seeds x 200 episodes.

Acceptance (fixed before running): a fix is accepted if the cautious agents' fraction of
the panel's best rises (paired by seed and rule vs old) and balanced agents do not get
worse; memory is also expected to steady the local metric (smaller late dM).
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pickle  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, pmap, write_summary  # noqa: E402

from cliffnpc.commons import Dials  # noqa: E402
from cliffnpc.geometry import smooth_returns  # noqa: E402
from cliffnpc.harness import Sizes, frontier_panel  # noqa: E402
from cliffnpc.phase_c import GOAL_SCALE, PREFERENCES, GoalGeometryAgent, connection  # noqa: E402

OUT = outdir("e24b_phase_c_fixes")
CONFIGS = {"old": ("mc", 0.95), "memory": ("mc", 0.98), "critic": ("td", 0.95), "both": ("td", 0.98)}
PREFS = ("cautious", "balanced")
RULES = ("pg", "ng", "pareto")
SEEDS = tuple(range(8))
SIZES = Sizes()

with open(OUT.parent / "e23_harness_validation" / "runs.pkl", "rb") as fh:
    J_panel = pickle.load(fh)["base"]["J"]
Jp_s = smooth_returns(frontier_panel(SIZES.n_frontier), J_panel) / GOAL_SCALE
Jp_raw = J_panel.mean(1) / GOAL_SCALE


def one(job):
    cfg, pref, rule, seed = job
    critic, forget = CONFIGS[cfg]
    ag = GoalGeometryAgent(Dials(), PREFERENCES[pref], rule=rule, seed=seed, critic=critic, forget=forget)
    ag.run(200)
    final = ag.evaluate(16).mean(0) / GOAL_SCALE
    conn = connection(ag.history, 20)
    return cfg, pref, rule, seed, final, float(np.mean([c["dM"] for c in conn[-4:]]))


res = pmap(one, [(c, p, r, s) for c in CONFIGS for p in PREFS for r in RULES for s in SEEDS])
R = {(c, p, r, s): (f, dm) for c, p, r, s, f, dm in res}


def frac(pref, final):
    w = PREFERENCES[pref]
    m, b = float((Jp_raw @ w).mean()), float((Jp_s @ w).max())
    return (final @ w - m) / (b - m)


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))


lines = ["E24b Phase-C agent fixes on rung 0", "=" * 34, "fraction of the panel's best (mean ± 95% CI over seeds x rules); "
         "late dM = change of the normalised local metric per 20 episodes over the last 80.", ""]
for pref in PREFS:
    lines.append(f"## {pref}")
    for cfg in CONFIGS:
        fr = [frac(pref, R[(cfg, pref, r, s)][0]) for r in RULES for s in SEEDS]
        dm = [R[(cfg, pref, r, s)][1] for r in RULES for s in SEEDS]
        per_rule = "  ".join(f"{r} {np.mean([frac(pref, R[(cfg, pref, r, s)][0]) for s in SEEDS]):.2f}" for r in RULES)
        lines.append(f"   {cfg:<7s} fraction {ci(fr)[0]:.2f} ± {ci(fr)[1]:.2f}  ({per_rule})   late dM {np.mean(dm):.2f}")
    for cfg in ("memory", "critic", "both"):
        d = [frac(pref, R[(cfg, pref, r, s)][0]) - frac(pref, R[("old", pref, r, s)][0]) for r in RULES for s in SEEDS]
        lines.append(f"   {cfg} - old: {ci(d)[0]:+.2f} ± {ci(d)[1]:.2f}")
    lines.append("")
lines.append("ACCEPTANCE (fixed before running)")
for cfg in ("memory", "critic", "both"):
    dc = ci([frac("cautious", R[(cfg, "cautious", r, s)][0]) - frac("cautious", R[("old", "cautious", r, s)][0])
             for r in RULES for s in SEEDS])
    db = ci([frac("balanced", R[(cfg, "balanced", r, s)][0]) - frac("balanced", R[("old", "balanced", r, s)][0])
             for r in RULES for s in SEEDS])
    ok = dc[0] - dc[1] > 0 and db[0] + db[1] >= 0
    lines.append(f"   {'ACCEPT' if ok else 'reject'}  {cfg}: cautious {dc[0]:+.2f} ± {dc[1]:.2f}, balanced {db[0]:+.2f} ± {db[1]:.2f}")
write_summary(OUT, lines)
