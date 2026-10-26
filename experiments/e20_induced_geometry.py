"""E20 The geometry induced by circumstance (spec-v2, phase B).

No agent dynamics: an exploratory controller (episode weights theta on six
primitives + smooth noise xi) generates experience records in the Commons world
under 7 environments (centre + one factor changed at a time), 4 seeds x 64
episodes of 60 s each.  For each environment and record scale (16 / 32 / 64
ticks) we estimate:

  * the goal metric M = A A^T from action gradients (A[i,k] = effect of leaning on
    primitive k on goal i's progress) and its cosines, with bootstrap CIs;
  * the signature of the outcome-weighted record moment S (knowledge valued 0 or 1);
  * the signature of the return curvature in theta.

A second exploratory policy (theta ~ U(0, 1), noise 0.3) at the centre shows how
much of the geometry belongs to the policy rather than the environment.

Expectations (fixed before running; spec-v2 section 4):
  skilled forager    cos(social, food) higher than with an unskilled forager
  threat             cos(safety, food) more negative as threat rises
  cue validity       cos(knowledge, food) higher with predictive cues
  scarcity           food relations stronger when scarce
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

from cliffnpc.commons import CENTRE, GOALS, PRIMITIVES, SCALES, Env, environments, record_windows  # noqa: E402
from cliffnpc.connection import FEATURES  # noqa: E402
from cliffnpc.geometry import (action_gradients, bootstrap, goal_metric, merge, outcome_moment,  # noqa: E402
                               outcome_weight, record_features, return_curvature, signature)

OUT = outdir("e20_induced_geometry")
SEEDS = (0, 1, 2, 3)
EPISODES = 64
EXTRA = ("forager_near", "home_near", "in_shelter", "satiety", "health", "ambush", "shelter_exists", "pad")
REC_NAMES = ["vx", "vy", "fight"] + [f"lean_{p}" for p in PRIMITIVES] + list(FEATURES) + list(EXTRA)
PAIRS = [(0, 2, "social·food"), (0, 1, "safety·food"), (0, 3, "knowledge·food"), (1, 2, "safety·social"),
         (1, 3, "safety·knowledge"), (2, 3, "social·knowledge")]


def conditions():
    out = [(e.name(), e, {}) for e in environments()]
    out.append(("centre, policy B", CENTRE, dict(theta_low=0.0, theta_high=1.0, noise_sigma=0.3)))
    return out


def one(job):
    name, env, kw, seed = job
    from cliffnpc.commons import Commons
    c = Commons(Env(**env.__dict__), seed=seed, **kw)
    return name, c.run(EPISODES)


jobs = [(n, e, kw, s) for n, e, kw in conditions() for s in SEEDS]
logs = {}
for name, log in pmap(one, jobs):
    logs.setdefault(name, []).append(log)
logs = {k: merge(v) for k, v in logs.items()}
names = [n for n, _, _ in conditions()]

res = {}
for name in names:
    log = logs[name]
    res[name] = {}
    for T in SCALES:
        W = record_windows(log, T)
        A = action_gradients(W)
        _, cos = goal_metric(A)
        boot = bootstrap(lambda w: goal_metric(action_gradients(w))[1], W, n=200)
        lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
        sig = {}
        for kv in (0.0, 1.0):
            S = outcome_moment(W, outcome_weight(W, (1, 1, 0.5, kv)))
            sig[kv] = signature(S)
        res[name][T] = dict(A=A, cos=cos, lo=lo, hi=hi, sig=sig, n=len(W["G"]))
    res[name]["curv"] = {kv: signature(return_curvature(log, None, (1, 1, 0.5, kv))) for kv in (0.0, 1.0)}
    res[name]["curv_goal"] = [signature(return_curvature(log, g)) for g in range(len(GOALS))]
    res[name]["mean_return"] = [float(log["r"][:, g].sum() / (log["episode"].max() + 1)) for g in range(len(GOALS))]

# ---- outcome-moment eigen-directions at the centre (what the signature is made of)
Wc = record_windows(logs[CENTRE.name()], 32)
X = record_features(Wc)
keep = X.std(0) > 1e-9
kept_names = [n for n, k in zip(REC_NAMES, keep) if k]
S = outcome_moment(Wc, outcome_weight(Wc))
ev, V = np.linalg.eigh(S)

# ---- figures
fig, axes = plt.subplots(len(names), len(SCALES), figsize=(4.2 * len(SCALES), 3.6 * len(names)))
for i, name in enumerate(names):
    for j, T in enumerate(SCALES):
        ax = axes[i][j]
        C = res[name][T]["cos"]
        ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        for a in range(4):
            for b in range(4):
                ax.text(b, a, f"{C[a, b]:+.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(4), GOALS, fontsize=7, rotation=30)
        ax.set_yticks(range(4), GOALS, fontsize=7)
        ax.set_title(f"{name}\nT={T}", fontsize=7)
save(fig, OUT, "goal_cosines.png")

fig, axes = plt.subplots(1, len(PAIRS), figsize=(4 * len(PAIRS), 4))
for ax, (a, b, lab) in zip(axes, PAIRS):
    for j, T in enumerate(SCALES):
        y = [res[n][T]["cos"][a, b] for n in names]
        e = [[res[n][T]["cos"][a, b] - res[n][T]["lo"][a, b] for n in names],
             [res[n][T]["hi"][a, b] - res[n][T]["cos"][a, b] for n in names]]
        ax.errorbar(np.arange(len(names)) + 0.2 * (j - 1), y, yerr=e, fmt="o", ms=3, label=f"T={T}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(names)), names, rotation=80, fontsize=6)
    ax.set_title(lab)
    ax.set_ylim(-1.05, 1.05)
axes[0].legend(fontsize=7)
save(fig, OUT, "relations_by_environment.png")

# ---- summary
lines = ["E20 Geometry induced by circumstance (spec-v2 phase B)", "=" * 54,
         f"{len(SEEDS)} seeds x {EPISODES} episodes x 60 s per condition; exploratory controller "
         "(6 primitives, episode weights + OU noise).",
         "Cosines of the action-gradient goal metric M = A A^T, [95% bootstrap CI over episodes].", ""]
for name in names:
    lines.append(f"## {name}   mean return per episode: " +
                 "  ".join(f"{g} {v:.2f}" for g, v in zip(GOALS, res[name]["mean_return"])))
    for T in SCALES:
        r = res[name][T]
        lines.append(f"   T={T:<3d} (n={r['n']:5d}) " + "  ".join(
            f"{lab} {r['cos'][a, b]:+.2f} [{r['lo'][a, b]:+.2f},{r['hi'][a, b]:+.2f}]" for a, b, lab in PAIRS))
        lines.append(f"            outcome-moment signature (p,q,0): knowledge valued 0 {r['sig'][0.0]}, "
                     f"valued 1 {r['sig'][1.0]}")
    lines.append(f"   return curvature signature in theta: scalarised (k=0) {res[name]['curv'][0.0]}, "
                 f"(k=1) {res[name]['curv'][1.0]};  per goal " +
                 "  ".join(f"{g} {s}" for g, s in zip(GOALS, res[name]["curv_goal"])))
    lines.append("   strongest action gradients (T=32): " + "  ".join(
        f"{g}<-{PRIMITIVES[int(np.argmax(np.abs(res[name][32]['A'][i])))]}"
        f"({res[name][32]['A'][i][int(np.argmax(np.abs(res[name][32]['A'][i])))]:+.2f})"
        for i, g in enumerate(GOALS)))
    lines.append("")
lines.append("Centre, T=32: outcome-moment eigen-directions (largest |eigenvalue| first), top features")
for k in np.argsort(-np.abs(ev))[:6]:
    top = np.argsort(-np.abs(V[:, k]))[:4]
    lines.append(f"   eigenvalue {ev[k]:+.3f}: " + ", ".join(f"{kept_names[t]} {V[t, k]:+.2f}" for t in top))
lines += ["", "EXPECTATIONS (fixed before running)"]


def cosT(name, a, b, T=32):
    return res[name][T]["cos"][a, b]


e = {x.name(): x for x in environments()}
c0 = CENTRE.name()
skill0 = Env(**{**CENTRE.__dict__, "ally_skill": 0}).name()
thr = [Env(**{**CENTRE.__dict__, "threat": t}).name() for t in (0.0, 0.3, 0.7)]
cue0 = Env(**{**CENTRE.__dict__, "cue_validity": 0}).name()
scar1 = Env(**{**CENTRE.__dict__, "scarcity": 1}).name()
for T in SCALES:
    lines.append(f"   T={T}: social·food skilled {cosT(c0, 0, 2, T):+.2f} vs unskilled {cosT(skill0, 0, 2, T):+.2f};  "
                 f"safety·food at threat 0/0.3/0.7: " + "/".join(f"{cosT(n, 0, 1, T):+.2f}" for n in thr) + ";  "
                 f"knowledge·food valid {cosT(c0, 0, 3, T):+.2f} vs random cues {cosT(cue0, 0, 3, T):+.2f};  "
                 f"food relations rich vs scarce: social·food {cosT(c0, 0, 2, T):+.2f}/{cosT(scar1, 0, 2, T):+.2f}, "
                 f"safety·food {cosT(c0, 0, 1, T):+.2f}/{cosT(scar1, 0, 1, T):+.2f}")
lines.append(f"   policy dependence (centre, T=32): social·food A {cosT(c0, 0, 2):+.2f} vs policy B "
             f"{cosT('centre, policy B', 0, 2):+.2f}; safety·food {cosT(c0, 0, 1):+.2f} vs "
             f"{cosT('centre, policy B', 0, 1):+.2f}")
write_summary(OUT, lines)
with open(OUT / "geometry.json", "w") as fh:
    json.dump({n: {str(T): {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res[n][T].items()
                            if k in ("A", "cos", "lo", "hi", "n")} for T in SCALES} for n in names}, fh)
