"""E21 Induced geometry, retest after four fixes (spec-v2 phase B).

Fixes after E20:
  1. knowledge = bits resolved about patch types (sight resolves a patch only if
     cues are valid; probing at the patch always does) -- no longer tied to visits;
  2. social = company: the homebody ally is replaced by a companion who wanders
     the world (the forager is the ally who is good at food);
  3. 8 seeds x 128 episodes per condition (4x E20);
  4. exploratory policy families (broad, narrow, bold, cautious, social) on four
     environments, to split each relation into environment and policy shares.

Expectations (fixed before running):
  knowledge.food   high with random cues (must probe), low with valid cues (seen from afar)
  social.food      higher with a skilled forager than an unskilled one
  safety.food      negative only with threat, more so without shelter and at longer scales
  policy share     reported, not predicted
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

from cliffnpc.commons import CENTRE, GOALS, POLICIES, PRIMITIVES, SCALES, Env, environments, record_windows  # noqa: E402
from cliffnpc.geometry import (action_gradients, bootstrap, goal_metric, merge, outcome_moment,  # noqa: E402
                               outcome_weight, return_curvature, signature)

OUT = outdir("e21_geometry_v2")
SEEDS = tuple(range(8))
EPISODES = 128
PAIRS = [(0, 2, "social·food"), (0, 1, "safety·food"), (0, 3, "knowledge·food"), (1, 2, "safety·social"),
         (1, 3, "safety·knowledge"), (2, 3, "social·knowledge")]


def env_with(**kw):
    return Env(**{**CENTRE.__dict__, **kw})


GRID_ENVS = {"centre": CENTRE, "threat 0.7": env_with(threat=0.7), "unskilled forager": env_with(ally_skill=0),
             "random cues": env_with(cue_validity=0)}


def conditions():
    out = [(f"{e.name()} | broad", e, "broad") for e in environments()]
    for en, e in GRID_ENVS.items():
        for pol in POLICIES:
            if pol != "broad":
                out.append((f"{e.name()} | {pol}", e, pol))
    return out


def one(job):
    name, env, pol, seed = job
    from cliffnpc.commons import Commons
    return name, Commons(Env(**env.__dict__), seed=seed, policy=pol).run(EPISODES)


jobs = [(n, e, p, s) for n, e, p in conditions() for s in SEEDS]
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
        boot = bootstrap(lambda w: goal_metric(action_gradients(w))[1], W, n=100)
        res[name][T] = dict(A=A, cos=cos, lo=np.percentile(boot, 2.5, axis=0), hi=np.percentile(boot, 97.5, axis=0),
                            sig={kv: signature(outcome_moment(W, outcome_weight(W, (1, 1, 0.5, kv))))
                                 for kv in (0.0, 1.0)})
    res[name]["curv"] = signature(return_curvature(log))
    res[name]["mean_return"] = [float(log["r"][:, g].sum() / (log["episode"].max() + 1)) for g in range(4)]


def fmt_pair(r, a, b):
    return f"{r['cos'][a, b]:+.2f} [{r['lo'][a, b]:+.2f},{r['hi'][a, b]:+.2f}]"


lines = ["E21 Induced geometry, retest (spec-v2 phase B)", "=" * 46,
         f"{len(SEEDS)} seeds x {EPISODES} episodes x 60 s per condition. Cosines of M = A A^T "
         "[95% bootstrap CI over episodes].", ""]
for name in names:
    lines.append(f"## {name}   mean return/episode: " +
                 "  ".join(f"{g} {v:.2f}" for g, v in zip(GOALS, res[name]["mean_return"])))
    for T in SCALES:
        r = res[name][T]
        lines.append(f"   T={T:<3d} " + "  ".join(f"{lab} {fmt_pair(r, a, b)}" for a, b, lab in PAIRS))
    lines.append(f"   outcome-moment signature T=32 (knowledge 0 / 1): {res[name][32]['sig'][0.0]} / "
                 f"{res[name][32]['sig'][1.0]};  return curvature {res[name]['curv']}")
    lines.append("")

# ---- expectations
b = lambda e: f"{e.name()} | broad"  # noqa: E731
lines.append("EXPECTATIONS (fixed before running), policy broad")
for T in SCALES:
    c = lambda e, a, bb: res[b(e)][T]["cos"][a, bb]  # noqa: E731
    lines.append(f"   T={T}: knowledge·food random cues {c(env_with(cue_validity=0), 0, 3):+.2f} vs valid "
                 f"{c(CENTRE, 0, 3):+.2f};  social·food skilled {c(CENTRE, 0, 2):+.2f} vs unskilled "
                 f"{c(env_with(ally_skill=0), 0, 2):+.2f};  safety·food threat 0/0.3/0.7 "
                 + "/".join(f"{c(env_with(threat=t), 0, 1):+.2f}" for t in (0.0, 0.3, 0.7))
                 + f", no shelter {c(env_with(shelter=False), 0, 1):+.2f}")

# ---- environment vs policy shares on the 4 x 5 grid
lines += ["", "ENVIRONMENT vs POLICY (4 environments x 5 policies): share of the variance of each cosine"]
pols = list(POLICIES)
fig, axes = plt.subplots(len(SCALES), len(PAIRS), figsize=(3.3 * len(PAIRS), 3.0 * len(SCALES)))
for i, T in enumerate(SCALES):
    row = []
    for j, (a, bb, lab) in enumerate(PAIRS):
        Y = np.array([[res[f"{e.name()} | {p}"][T]["cos"][a, bb] for p in pols] for e in GRID_ENVS.values()])
        grand = Y.mean()
        ss_env = Y.shape[1] * ((Y.mean(1) - grand) ** 2).sum()
        ss_pol = Y.shape[0] * ((Y.mean(0) - grand) ** 2).sum()
        ss_tot = ((Y - grand) ** 2).sum()
        ss_int = ss_tot - ss_env - ss_pol
        row.append(f"{lab} env {ss_env / ss_tot:.2f} / policy {ss_pol / ss_tot:.2f} / interaction "
                   f"{ss_int / ss_tot:.2f}")
        ax = axes[i][j]
        ax.imshow(Y, cmap="RdBu_r", vmin=-1, vmax=1)
        for u in range(Y.shape[0]):
            for v in range(Y.shape[1]):
                ax.text(v, u, f"{Y[u, v]:+.2f}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(pols)), pols, rotation=45, fontsize=6)
        ax.set_yticks(range(len(GRID_ENVS)), list(GRID_ENVS), fontsize=6)
        ax.set_title(f"{lab}, T={T}", fontsize=7)
    lines.append(f"   T={T}: " + ";  ".join(row))
save(fig, OUT, "env_policy_grid.png")

# ---- relations across the one-factor sweep (policy broad)
fig, axes = plt.subplots(1, len(PAIRS), figsize=(4 * len(PAIRS), 4))
sweep = [b(e) for e in environments()]
for ax, (a, bb, lab) in zip(axes, PAIRS):
    for j, T in enumerate(SCALES):
        y = [res[n][T]["cos"][a, bb] for n in sweep]
        err = [[res[n][T]["cos"][a, bb] - res[n][T]["lo"][a, bb] for n in sweep],
               [res[n][T]["hi"][a, bb] - res[n][T]["cos"][a, bb] for n in sweep]]
        ax.errorbar(np.arange(len(sweep)) + 0.2 * (j - 1), y, yerr=err, fmt="o", ms=3, label=f"T={T}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(sweep)), [n.split(" | ")[0] for n in sweep], rotation=80, fontsize=6)
    ax.set_title(lab)
    ax.set_ylim(-1.05, 1.05)
axes[0].legend(fontsize=7)
save(fig, OUT, "relations_by_environment.png")
write_summary(OUT, lines)
with open(OUT / "geometry.json", "w") as fh:
    json.dump({n: {str(T): {k: v.tolist() for k, v in res[n][T].items() if k in ("A", "cos", "lo", "hi")}
                   for T in SCALES} for n in names}, fh)
