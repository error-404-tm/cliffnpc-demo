"""E22 The attainable set: the environment's own geometry (spec-v2 phase B).

E21 showed the local goal metric is mostly a property of the policy.  The
policy-independent object is the set of goal-return vectors attainable by *any*
policy, and its frontier: where you cannot gain one goal without losing another.

Per environment (the 7 of the one-factor sweep): 384 policies (theta ~ U(-1, 2)^6
plus the five family centres), 6 episodes each, small noise (sigma 0.2).  Expected
returns per policy are denoised by a quadratic regression in theta; the frontier is
the Pareto set of those predictions (goals rescaled to [0, 1] per environment).

Read off per environment:
  * goal correlations along the frontier (negative: a genuine trade-off);
  * utopia gaps per goal pair (0: one policy attains both maxima -- no trade-off);
  * curvature signature of the frontier surface (food as a quadratic function of the
    other goals on the frontier);
  * which primitives the frontier policies lean on.

Expectations (fixed before running):
  threat       makes safety.food a frontier trade-off (negative correlation, larger gap); none at threat 0
  shelter      softens the safety.food trade-off
  ally skill   makes social.food cooperative on the frontier (positive correlation, smaller gap)
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

from cliffnpc.commons import CENTRE, GOALS, POLICIES, PRIMITIVES, Env, environments  # noqa: E402
from cliffnpc.geometry import frontier_report, signature, smooth_returns  # noqa: E402

OUT = outdir("e22_frontier")
N_POLICIES = 384
EPISODES = 6
CHUNKS = 12
PAIRS = [(0, 1, "food·safety"), (0, 2, "food·social"), (0, 3, "food·knowledge"), (1, 2, "safety·social"),
         (1, 3, "safety·knowledge"), (2, 3, "social·knowledge")]


def policies(seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(-1.0, 2.0, (N_POLICIES - len(POLICIES), len(PRIMITIVES)))
    fam = np.array([POLICIES[p]["mean"] for p in POLICIES], float)
    return np.vstack([fam, th])


THETA = policies()


def one(job):
    env, chunk = job
    from cliffnpc.commons import Commons
    idx = np.arange(chunk, N_POLICIES, CHUNKS)
    c = Commons(Env(**env.__dict__), seed=1000 + chunk, noise_override=0.2)
    return env.name(), idx, c.run_policies(THETA[idx], EPISODES)


envs = environments()
R = {e.name(): np.zeros((N_POLICIES, EPISODES, len(GOALS))) for e in envs}
for name, idx, out in pmap(one, [(e, k) for e in envs for k in range(CHUNKS)]):
    R[name][idx] = out
np.savez_compressed(OUT / "returns.npz", theta=THETA, **{n.replace(".", "p"): v for n, v in R.items()})

rep = {}
for e in envs:
    n = e.name()
    J = smooth_returns(THETA, R[n])
    rep[n] = dict(smooth=frontier_report(J), raw=frontier_report(R[n].mean(1)), J=J)

# ---- figures: pairwise projections with the frontier highlighted
fig, axes = plt.subplots(len(envs), 3, figsize=(13, 3.3 * len(envs)))
for i, e in enumerate(envs):
    r = rep[e.name()]["smooth"]
    Z, P = r["Z"], r["pareto"]
    for j, (a, b, lab) in enumerate(PAIRS[:3]):
        ax = axes[i][j]
        ax.scatter(Z[~P, b], Z[~P, a], s=4, c="#bdc3c7")
        ax.scatter(Z[P, b], Z[P, a], s=8, c="#c0392b")
        ax.set_xlabel(GOALS[b], fontsize=7)
        ax.set_ylabel(GOALS[a], fontsize=7)
        ax.set_title(f"{e.name()}\n{lab}: frontier corr {r['corr'][a, b]:+.2f}, gap {r['gaps'][a, b]:.2f}",
                     fontsize=7)
save(fig, OUT, "frontier_projections.png")

# ---- summary
lines = ["E22 Attainable set and its frontier (spec-v2 phase B)", "=" * 52,
         f"{N_POLICIES} policies x {EPISODES} episodes x 60 s per environment; expected returns denoised by a "
         "quadratic in theta; goals rescaled to [0,1] per environment.", ""]
for e in envs:
    n = e.name()
    s, raw = rep[n]["smooth"], rep[n]["raw"]
    J = rep[n]["J"]
    lines.append(f"## {n}   frontier policies {s['n_pareto']} of {N_POLICIES} (raw means: {raw['n_pareto']});  "
                 f"best policy's distance to the ideal point {s['ideal_best']:.2f}")
    lines.append("   best attainable per goal (per episode): " +
                 "  ".join(f"{g} {J[:, k].max():.2f}" for k, g in enumerate(GOALS)))
    lines.append("   frontier correlation: " + "  ".join(
        f"{lab} {s['corr'][a, b]:+.2f}" for a, b, lab in PAIRS) + "   (raw: " + "  ".join(
        f"{raw['corr'][a, b]:+.2f}" for a, b, _ in PAIRS) + ")")
    lines.append("   utopia gap:            " + "  ".join(f"{lab} {s['gaps'][a, b]:.2f}" for a, b, lab in PAIRS))
    if s["curv"] is not None:
        H = s["curv"]["H"]
        ev = np.linalg.eigvalsh(H)
        lines.append(f"   frontier curvature (food over {', '.join(GOALS[g] for g in s['curv']['goals'])}): "
                     f"eigenvalues {np.round(ev, 2)}  signature {signature(H)}")
    lean = THETA[s["pareto"]].mean(0)
    lines.append("   frontier policies lean on: " + "  ".join(f"{p} {v:+.2f}" for p, v in zip(PRIMITIVES, lean)))
    lines.append("")

lines.append("EXPECTATIONS (fixed before running)")


def get(env, key, a, b):
    return rep[env.name()]["smooth"][key][a, b]


env = lambda **kw: Env(**{**CENTRE.__dict__, **kw})  # noqa: E731
lines.append("   food·safety by threat 0 / 0.3 / 0.7: corr " + " / ".join(
    f"{get(env(threat=t), 'corr', 0, 1):+.2f}" for t in (0.0, 0.3, 0.7)) + ";  gap " + " / ".join(
    f"{get(env(threat=t), 'gaps', 0, 1):.2f}" for t in (0.0, 0.3, 0.7)))
lines.append(f"   food·safety shelter / none: corr {get(CENTRE, 'corr', 0, 1):+.2f} / "
             f"{get(env(shelter=False), 'corr', 0, 1):+.2f};  gap {get(CENTRE, 'gaps', 0, 1):.2f} / "
             f"{get(env(shelter=False), 'gaps', 0, 1):.2f}")
lines.append(f"   food·social skilled / unskilled forager: corr {get(CENTRE, 'corr', 0, 2):+.2f} / "
             f"{get(env(ally_skill=0), 'corr', 0, 2):+.2f};  gap {get(CENTRE, 'gaps', 0, 2):.2f} / "
             f"{get(env(ally_skill=0), 'gaps', 0, 2):.2f}")
write_summary(OUT, lines)
with open(OUT / "frontier.json", "w") as fh:
    json.dump({n: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rep[n]["smooth"].items()
                   if k in ("corr", "gaps", "n_pareto", "ideal_best")} for n in rep}, fh)
