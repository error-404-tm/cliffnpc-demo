"""E26 Ladder rungs 2-4, screened environment-side (spec-v2 sections 8-9).

Each dial is measured against the rung-1 world (need 1) with the feature harness:
  rung 2  depletion   finite patches that respawn elsewhere (0.5, 1)
  rung 3  false_frac  share of false patches (0.1, 0.9; base 0.5)
  rung 4  allies      extra foragers of graded skill (0.5 -> one, 1 -> two)

Question: does each dial only move the frontier (extent), or does it bend it (shape /
local)?  Only frontier-bending rungs give an agent something new to choose.

Expectation (fixed before running): mostly extent dials; false_frac may touch the
knowledge relations, allies may touch safety.social (strength in numbers).
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pickle  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, pmap, write_summary  # noqa: E402

from cliffnpc.commons import GOALS, Dials  # noqa: E402
from cliffnpc.harness import DIALS, PAIR_NAMES, Sizes, compare, merge_parts, world_for, world_parts, world_run  # noqa: E402

OUT = outdir("e26_rungs2to4_screen")
SIZES = Sizes()
RUNG1 = Dials(need=1.0)
DIALS_HERE = {"depletion": (0.5, 1.0), "false_frac": (0.1, 0.9), "allies": (0.5, 1.0)}
WORLDS = {"rung1": RUNG1}
for dial, levels in DIALS_HERE.items():
    for lv in levels:
        WORLDS[f"{dial}={lv:g}"] = world_for(dial, lv, RUNG1)


def run_part(job):
    return job[0], world_run(WORLDS[job[0]], 0, SIZES, job[1])


pk = OUT / "runs.pkl"
if os.environ.get("E26_REUSE") and pk.exists():
    with open(pk, "rb") as fh:
        runs = pickle.load(fh)
else:
    parts = {}
    for name, res in pmap(run_part, [(n, p) for n in WORLDS for p in world_parts(SIZES)]):
        parts.setdefault(name, []).append(res)
    runs = {n: merge_parts(v, SIZES) for n, v in parts.items()}
    with open(pk, "wb") as fh:
        pickle.dump(runs, fh)


def cmp_job(name):
    return name, compare(runs["rung1"], runs[name], SIZES, n_boot=100, seed=1,
                         targets=DIALS[name.split("=")[0]].targets)


CMP = dict(pmap(cmp_job, [n for n in WORLDS if n != "rung1"]))


def fmt_ci(d, c, k):
    return f"{d[k]:+.3f} [{c[0][k]:+.3f},{c[1][k]:+.3f}]"


lines = ["E26 Ladder rungs 2-4, screened against the rung-1 world", "=" * 55, ""]
verdict = {}
for name, r in CMP.items():
    fr = r["frontier"]
    lines.append(f"## rung1 -> {name}   (meant to touch: {', '.join(DIALS[name.split('=')[0]].targets)})")
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
    dial = name.split("=")[0]
    bends = bool(r["shape"].any() or r["local_robust"].any())
    verdict[dial] = verdict.get(dial, False) or bends

lines.append("VERDICT: does the dial bend the frontier (shape or robust local change at either level)?")
lines += [f"   {d:<10s} {'bends' if v else 'extent only'}" for d, v in verdict.items()]
write_summary(OUT, lines)
