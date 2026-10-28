"""E23 Validating the feature harness against the known factors (spec-v2 section 8).

The harness (cliffnpc/harness.py) measures what turning one environment dial does to
the induced geometry, as paired common-random-number comparisons against the base
world, split into extent / shape / local / new dimension, with construct-check flags.

Worlds: the base (skill 1, threat 0.3, shelter 1, cue 1, scarcity 0) and two other
levels of each of the five dials.  Each world: a shared frontier panel of 192 fixed
policies x 4 episodes, and the five reference policy families x 64 episodes.

Validation checks (fixed before running; from E21/E22):
  V1  threat 0 vs base: safety stops being a live dimension (new dimension);
      threat 0.7 vs base: average safety falls (extent)
  V2  skill 0 vs base: food extent falls; food·social shape unchanged
  V3  cue 0 vs base: knowledge·food local cosine rises (broad family); knowledge extent falls
  V4  shelter 0 vs base: no food·safety shape change (confirmed null)
  V5  scarcity 1 vs base: average food falls
  V6  paired intervals narrower than unpaired
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pickle  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

from cliffnpc.commons import GOALS, SCALES  # noqa: E402
from cliffnpc.harness import (BASE, DIALS, PAIR_NAMES, Sizes, compare, merge_parts, world_for,  # noqa: E402
                              world_parts, world_run)

OUT = outdir("e23_harness_validation")
SIZES = Sizes()
SEED = 0


def worlds():
    out = {"base": BASE}
    for dial, spec in DIALS.items():
        for lv in spec.levels:
            w = world_for(dial, lv)
            if w != BASE:
                out[f"{dial}={lv:g}"] = w
    return out


W = worlds()


def run_part(job):
    name, part = job
    return name, world_run(W[name], SEED, SIZES, part)


if os.environ.get("E23_REUSE") and (OUT / "runs.pkl").exists():   # re-score saved runs
    with open(OUT / "runs.pkl", "rb") as fh:
        runs = pickle.load(fh)
else:
    parts = {}
    for name, res in pmap(run_part, [(n, p) for n in W for p in world_parts(SIZES)]):
        parts.setdefault(name, []).append(res)
    runs = {n: merge_parts(v, SIZES) for n, v in parts.items()}
    with open(OUT / "runs.pkl", "wb") as fh:
        pickle.dump(runs, fh)


def cmp_job(name):
    dial = name.split("=")[0]
    return name, compare(runs["base"], runs[name], SIZES, n_boot=100, seed=1, targets=DIALS[dial].targets)


R = dict(pmap(cmp_job, [n for n in W if n != "base"]))


def fmt_ci(d, ci, k):
    return f"{d[k]:+.3f} [{ci[0][k]:+.3f},{ci[1][k]:+.3f}]"


lines = ["E23 Feature harness validation (spec-v2 section 8)", "=" * 50,
         f"Paired CRN comparisons against the base world; frontier panel {SIZES.n_frontier} policies x "
         f"{SIZES.eps_frontier} episodes; {SIZES.eps_local} episodes per reference family; 95% paired-bootstrap "
         "intervals.", ""]
for name, r in R.items():
    fr = r["frontier"]
    lines.append(f"## base -> {name}   (meant to touch: {', '.join(DIALS[name.split('=')[0]].targets)})")
    lines.append("   extent, best:    " + "  ".join(f"{g} {fmt_ci(fr['extent']['d'], fr['extent']['ci'], k)}"
                                                    for k, g in enumerate(GOALS)))
    lines.append("   extent, average: " + "  ".join(f"{g} {fmt_ci(fr['mean']['d'], fr['mean']['ci'], k)}"
                                                    for k, g in enumerate(GOALS)))
    lines.append("   live dimensions: base " + ",".join(g for g, v in zip(GOALS, r["live_a"]) if v)
                 + " -> " + ",".join(g for g, v in zip(GOALS, r["live_b"]) if v))
    lines.append("   frontier gap (smoothed / raw): " + "  ".join(
        f"{p} {fr['gap_s']['d'][k]:+.2f}/{fr['gap_r']['d'][k]:+.2f}{'*' if r['gap_sig'][k] else ''}"
        for k, p in enumerate(PAIR_NAMES)))
    lines.append("   frontier corr (smoothed / raw): " + "  ".join(
        f"{p} {fr['corr_s']['d'][k]:+.2f}/{fr['corr_r']['d'][k]:+.2f}{'*' if r['corr_sig'][k] else ''}"
        for k, p in enumerate(PAIR_NAMES)))
    lines.append("   local: families (of 5) with a significant cosine change: " + "  ".join(
        f"{p} {r['local_count'][k]}" for k, p in enumerate(PAIR_NAMES)))
    kinds = []
    if r["extent"].any():
        kinds.append("extent (" + ",".join(g for g, v in zip(GOALS, r["extent"]) if v) + ")")
    if r["shape"].any():
        kinds.append("shape (" + ",".join(p for p, v in zip(PAIR_NAMES, r["shape"]) if v) + ")")
    if r["local_robust"].any():
        kinds.append("local (" + ",".join(p for p, v in zip(PAIR_NAMES, r["local_robust"]) if v) + ")")
    if r["new_dim"].any():
        kinds.append("new dimension (" + ",".join(g for g, v in zip(GOALS, r["new_dim"]) if v) + ")")
    lines.append("   KINDS: " + ("; ".join(kinds) if kinds else "none"))
    lines.append("   construct-check flags (touched but not targeted): " + (", ".join(r["flags"]) or "none"))
    w = r["ci_width_paired_vs_unpaired"]
    lines.append(f"   interval width paired / unpaired: extent {w['extent'][0]:.3f}/{w['extent'][1]:.3f}, "
                 f"frontier corr {w['corr_s'][0]:.3f}/{w['corr_s'][1]:.3f}")
    lines.append("")

# ---- validation checks
G = {g: k for k, g in enumerate(GOALS)}
P = {p: k for k, p in enumerate(PAIR_NAMES)}


def sig_neg(r, key, k):
    return r["frontier"][key]["ci"][1][k] < 0


def sig_pos(r, key, k):
    return r["frontier"][key]["ci"][0][k] > 0


checks = []
r = R["threat=0"]
checks.append(("V1a threat 0: safety no longer a live dimension", bool(r["live_a"][G["safety"]] and not r["live_b"][G["safety"]])))
r = R["threat=0.7"]
checks.append(("V1b threat 0.7: average safety falls", bool(sig_neg(r, "mean", G["safety"]))))
r = R["skill=0"]
checks.append(("V2a skill 0: food extent falls", bool(sig_neg(r, "extent", G["food"]) or sig_neg(r, "mean", G["food"]))))
checks.append(("V2b skill 0: food·social shape unchanged", bool(not r["shape"][P["food·social"]])))
r = R["cue=0"]
loc = r["local"]["broad"]
checks.append(("V3a cue 0: knowledge·food local cosine rises (broad, any scale)",
               bool(any(loc[T]["cos_ci"][0][P["food·knowledge"]] > 0 for T in SCALES))))
checks.append(("V3b cue 0: knowledge extent falls", bool(sig_neg(r, "extent", G["knowledge"]) or sig_neg(r, "mean", G["knowledge"]))))
r = R["shelter=0"]
checks.append(("V4 shelter 0: no food·safety shape change", bool(not r["shape"][P["food·safety"]])))
r = R["scarcity=1"]
checks.append(("V5 scarcity 1: average food falls", bool(sig_neg(r, "mean", G["food"]))))
wr = [R[n]["ci_width_paired_vs_unpaired"]["extent"] for n in R]
ratio = float(np.nanmean([a / b for a, b in wr if b > 0]))
checks.append((f"V6 paired intervals narrower than unpaired (extent width ratio {ratio:.2f})", ratio < 1.0))
lines.append("VALIDATION (fixed before running)")
lines += [f"   {'PASS' if ok else 'FAIL'}  {lab}" for lab, ok in checks]

# ---- dose-response figure: average return per goal along each dial
fig, axes = plt.subplots(1, len(DIALS), figsize=(4 * len(DIALS), 3.6))
for ax, (dial, spec) in zip(axes, DIALS.items()):
    for k, g in enumerate(GOALS):
        ys = []
        for lv in spec.levels:
            name = "base" if world_for(dial, lv) == BASE else f"{dial}={lv:g}"
            Jm = runs[name]["J"].mean(1)
            ys.append(Jm[:, k].mean() / (np.abs(runs["base"]["J"].mean(1)[:, k]).mean() + 1e-9))
        ax.plot(spec.levels, ys, "o-", label=g)
    ax.set_title(f"dial: {dial}", fontsize=9)
    ax.set_xlabel("level")
axes[0].set_ylabel("average return / base")
axes[0].legend(fontsize=7)
save(fig, OUT, "dose_response.png")
write_summary(OUT, lines)
