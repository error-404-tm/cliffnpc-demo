"""E30 Geometry where it is strong: detecting that the trade-off structure has changed.

The agent keeps a fast and a slow estimate of its action gradient.  When the *shape* of
the fast local goal metric (its cosine matrix) departs from the slow one beyond a running
threshold (mean + 3 sd), it treats the world as changed: it drops the stale estimate,
restarts its learning-rate schedule at double rate, and explores 1.5x more for 10 pairs.

Test: the world switches midway -- rung 5 without rivals (need 1, depletion 0.5) for 200
episodes, then rivals appear (rivals 1) for 200 more.  Agents: ctx policies, spsa
estimator, rule pg / pareto, detector off / on, preference balanced / cautious, 8 seeds.

Criteria (fixed before running):
  D1  the detector notices the switch: >= 50% of detector runs flag a change within 20
      pairs (40 episodes) after it, with on average <= 1 false alarm per run before it
  D2  detector agents score higher over the 60 episodes after the switch (paired)
  D3  no cost at the end: final 60 episodes not worse (paired)
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pickle  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

from cliffnpc.commons import Dials  # noqa: E402
from cliffnpc.phase_c import PREFERENCES, GoalGeometryAgent  # noqa: E402

OUT = outdir("e30_change_detector")
BEFORE = Dials(need=1.0, depletion=0.5, rivals=0.0)
AFTER = Dials(need=1.0, depletion=0.5, rivals=1.0)
RULES = ("pg", "pareto")
PREFS = ("balanced", "cautious")
SEEDS = tuple(range(8))
HALF = 200
SWITCH_PAIR = HALF // 2


def one(job):
    rule, pref, seed, det = job
    ag = GoalGeometryAgent(BEFORE, PREFERENCES[pref], rule=rule, seed=seed, policy="ctx", estimator="spsa", detector=det)
    ag.run(HALF)
    ag.set_world(AFTER)
    ag.run(HALF)
    return dict(rule=rule, pref=pref, seed=seed, det=det, detections=list(ag.detections),
                scores=np.array([h["score"] for h in ag.history]))


agents = pmap(one, [(r, p, s, d) for r in RULES for p in PREFS for s in SEEDS for d in (False, True)])
with open(OUT / "agents.pkl", "wb") as fh:
    pickle.dump(agents, fh)


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def fmt(m):
    return f"{m[0]:+.2f} ± {m[1]:.2f}"


on = [a for a in agents if a["det"]]
hit = [any(SWITCH_PAIR <= d < SWITCH_PAIR + 20 for d in a["detections"]) for a in on]
false = [sum(d < SWITCH_PAIR for d in a["detections"]) for a in on]
delay = [min([d - SWITCH_PAIR for d in a["detections"] if d >= SWITCH_PAIR], default=np.nan) for a in on]
key = lambda a: (a["rule"], a["pref"], a["seed"])  # noqa: E731
off = {key(a): a for a in agents if not a["det"]}
post = [a["scores"][SWITCH_PAIR:SWITCH_PAIR + 30].mean() - off[key(a)]["scores"][SWITCH_PAIR:SWITCH_PAIR + 30].mean() for a in on]
late = [a["scores"][-30:].mean() - off[key(a)]["scores"][-30:].mean() for a in on]

lines = ["E30 Detecting a change in the trade-off structure", "=" * 49,
         f"switch at episode {HALF} (pair {SWITCH_PAIR}): rivals appear; {len(SEEDS)} seeds per (rule, preference); "
         "scores are omega . J / scale per pair.", "",
         f"detection within 20 pairs of the switch: {np.mean(hit):.0%} of runs; median delay "
         f"{np.nanmedian(delay):.0f} pairs; false alarms before the switch {np.mean(false):.2f} per run", ""]
for rule in RULES:
    for pref in PREFS:
        sub = [a for a in on if a["rule"] == rule and a["pref"] == pref]
        p = [x for x, a in zip(post, on) if a["rule"] == rule and a["pref"] == pref]
        lt = [x for x, a in zip(late, on) if a["rule"] == rule and a["pref"] == pref]
        lines.append(f"   {rule:<7s}{pref:<9s} detected {np.mean([any(SWITCH_PAIR <= d < SWITCH_PAIR + 20 for d in a['detections']) for a in sub]):.0%}   "
                     f"after-switch gain {fmt(ci(p))}   final gain {fmt(ci(lt))}")
checks = [
    (f"D1 detects the switch ({np.mean(hit):.0%} within 20 pairs; {np.mean(false):.2f} false alarms per run)",
     np.mean(hit) >= 0.5 and np.mean(false) <= 1.0),
    (f"D2 faster recovery after the switch ({fmt(ci(post))})", ci(post)[0] - ci(post)[1] > 0),
    (f"D3 no cost at the end ({fmt(ci(late))})", ci(late)[0] + ci(late)[1] >= 0),
]
lines += ["", "CRITERIA (fixed before running)"] + [f"   {'PASS' if ok else 'FAIL'}  {lab}" for lab, ok in checks]

fig, ax = plt.subplots(figsize=(9, 4))
for det, col in ((False, "#7f8c8d"), (True, "#c0392b")):
    S = np.array([a["scores"] for a in agents if a["det"] == det])
    sm = np.array([np.convolve(s, np.ones(5) / 5, mode="same") for s in S]).mean(0)
    ax.plot(np.arange(len(sm)) * 2, sm, color=col, label=f"detector {'on' if det else 'off'}")
ax.axvline(HALF, color="k", ls="--", lw=0.8)
ax.set_xlabel("episode")
ax.set_ylabel("score")
ax.set_title("rivals appear at episode 200")
ax.legend()
save(fig, OUT, "recovery.png")
write_summary(OUT, lines)
