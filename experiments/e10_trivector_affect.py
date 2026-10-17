"""E10 Affects as trivectors: Cl(4,0) vs Cl(3,1) vs Cl(2,2).

Drives are vectors (hunger, social, explore, fight/flight); affects are the dual
grade-3 trivectors: e123 (no fight/flight) and the three fears e124, e134, e234.
Every push B the world gives splits as CB = C.B + C^B: the vector part moves
the drives, the trivector part is felt.  Environmental events (unmet hunger, a
meal, loneliness, company, injury taken or dealt, a witnessed death, false food,
the world's own push) are designed drive-space bivectors with a valence; each
contributes its change C^B and its surprise (C - C_hat)^B.  Felt affect uses the
metric g induces on trivectors (its 3x3 minors, the cofactors), normalized by
|det g|^(3/4).  Affect acts back on behaviour through the contraction C _| F as
a generator, and rivals near drive a boost in the social-threat plane (option A).
Pain-aversion learning blames each affect on the three drives it spans.

Everything else is the full-life two-tribe world of E9.  Long runs.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import collections  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

from cliffnpc import algebra as ga  # noqa: E402
from cliffnpc.affect import AFFECT_NAMES  # noqa: E402
from cliffnpc.arena import Arena, life_config  # noqa: E402

OUT = outdir("e10_trivector_affect")
T = 4000.0
SEEDS = (0, 1, 2, 3)
SIGNATURES = {"Cl(4,0)": (1.0, 1.0, 1.0, 1.0), "Cl(3,1)": (1.0, 1.0, 1.0, -1.0),
              "Cl(2,2)": (-1.0, 1.0, 1.0, -1.0)}
COL = {"Cl(4,0)": "#27ae60", "Cl(3,1)": "#2e86c1", "Cl(2,2)": "#c0392b"}
DS = 25


def fight_stats(dec, t0, t1):
    if len(dec) == 0:
        return (np.nan, 0, np.nan, 0)
    d = dec[(dec[:, 0] >= t0) & (dec[:, 0] < t1)]
    alone, grp = d[d[:, 2] == 0], d[d[:, 2] >= 2]
    f = lambda x: float(np.mean(x[:, 6] > 0)) if len(x) else np.nan  # noqa: E731
    return (f(alone), len(alone), f(grp), len(grp))


def one(job):
    name, seed = job
    ar = Arena(life_config(SIGNATURES[name], seed))
    log = ar.run(T)
    alive, fac = log["alive"], log["faction"]
    C, g = log["C"], log["g"]
    with np.errstate(all="ignore"):
        s = np.einsum("tai,taij,taj->ta", C, g, C)
        cone = s / np.einsum("tai,tai->ta", C, C)
        F = np.where(alive[..., None], log["affect"], np.nan)
        absF = np.nanmean(np.abs(F), axis=1)                    # per tick, per affect
        pain_t = np.nanmean(np.where(alive, log["pain"], np.nan), axis=1)
    ev = [e for a in ar.agents if a.ltm for e in a.ltm.events]
    rec = collections.Counter(e["origin"] for e in ev if e["what"] == "recalled")
    dec = np.array(ar.decisions)
    hours = float(alive.sum()) * ar.wcfg.dt * ar.wcfg.cog_every / 3600
    sides = [e.get("sides", {}) for e in ar.engagements]
    return dict(
        name=name, seed=seed, t=log["t"][::DS],
        pops={f: ((fac == k) & alive).sum(1)[::DS] for k, f in enumerate(ar.factions)},
        absF=absF[::DS], pain_t=pain_t[::DS],
        affect_mix=np.nanmean(absF, axis=0),
        reactive=float(np.nanmean(np.where(alive, cone < -0.05, np.nan))),
        births=len(ar.births), max_gen=max([b["generation"] for b in ar.births], default=0),
        deaths=[(d["faction"], d["cause"], len(d["witnesses"])) for d in ar.deaths],
        engagements=len(ar.engagements),
        fights_per_agent_hour=len(ar.engagements) / max(hours, 1e-9),
        outnumbered=sum(1 for sd in sides if sd and min(sd.values()) == 1 and max(sd.values()) > 1)
        / max(len(ar.engagements), 1),
        pain=float(np.nanmean(np.where(alive, log["pain"], np.nan))),
        hunger=float(np.nanmean(np.where(alive, C[:, :, 0], np.nan))),
        depleted=sum(e["what"] == "depleted" for e in ar.food_events),
        exposure=len(ar.decisions) / max(float(alive.sum()), 1.0),
        recalls=dict(rec),
        f_early=fight_stats(dec, 0, T / 4), f_late=fight_stats(dec, 3 * T / 4, T),
        f_all=fight_stats(dec, 0, T),
    )


results = pmap(one, [(n, s) for n in SIGNATURES for s in SEEDS])
by = {n: [r for r in results if r["name"] == n] for n in SIGNATURES}

fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
ax = axes[0][0]
for n, rs in by.items():
    for k, r in enumerate(rs):
        ax.plot(r["t"], r["pops"]["A"] + r["pops"]["B"], color=COL[n], lw=1, alpha=0.7,
                label=n if k == 0 else None)
ax.set_title("total population (each line one seed)")
ax.set_xlabel("t (s)")
ax.legend(fontsize=8)
ax = axes[0][1]
for n, rs in by.items():
    m = np.nanmean([r["pain_t"] for r in rs], axis=0)
    ax.plot(rs[0]["t"], m, color=COL[n], label=n)
ax.axhline(0, color="k", lw=0.5)
ax.set_title("mean felt pain (+) / reward (−), mean of seeds")
ax.set_xlabel("t (s)")
ax.legend(fontsize=8)
ax = axes[0][2]
x = np.arange(4)
for k, n in enumerate(SIGNATURES):
    mix = np.nanmean([r["affect_mix"] for r in by[n]], axis=0)
    ax.bar(x + (k - 1) * 0.27, mix / mix.sum(), 0.27, color=COL[n], label=n)
ax.set_xticks(x, AFFECT_NAMES, fontsize=8)
ax.set_title("composition of felt affect |F|")
ax.legend(fontsize=8)
ax = axes[1][0]
xs = np.arange(len(SIGNATURES))
w = 0.2
for j, (key, idx, lab) in enumerate([("f_early", 0, "alone, early"), ("f_late", 0, "alone, late"),
                                      ("f_early", 2, "w/ mates, early"), ("f_late", 2, "w/ mates, late")]):
    vals = [np.nanmean([r[key][idx] for r in by[n]]) for n in SIGNATURES]
    ax.bar(xs + (j - 1.5) * w, vals, w, label=lab)
ax.set_xticks(xs, list(SIGNATURES))
ax.set_title("fight tendency with a rival in view (first vs last quarter)")
ax.legend(fontsize=7)
ax = axes[1][1]
for k, cause in enumerate(("injury", "starvation")):
    vals = [np.mean([sum(d[1] == cause for d in r["deaths"]) for r in by[n]]) for n in SIGNATURES]
    ax.bar(xs + (k - 0.5) * 0.3, vals, 0.3, label=f"deaths: {cause}")
ax.plot(xs, [np.mean([r["births"] for r in by[n]]) for n in SIGNATURES], "k^", ms=10, label="births")
ax.set_xticks(xs, list(SIGNATURES))
ax.set_title("deaths and births per run")
ax.legend(fontsize=8)
ax = axes[1][2]
vals = [np.mean([r["fights_per_agent_hour"] for r in by[n]]) for n in SIGNATURES]
err = [np.std([r["fights_per_agent_hour"] for r in by[n]]) for n in SIGNATURES]
ax.bar(xs, vals, 0.5, yerr=err, color=[COL[n] for n in SIGNATURES])
ax.set_xticks(xs, list(SIGNATURES))
ax.set_title("fights per agent-hour")
save(fig, OUT, "trivector_affect.png")


def m(n, f):
    v = np.array([f(r) for r in by[n]], float)
    return f"{np.nanmean(v):8.2f} ± {np.nanstd(v):5.2f}"


rows = [
    ("births", lambda r: r["births"]),
    ("max generation", lambda r: r["max_gen"]),
    ("deaths: injury", lambda r: sum(d[1] == "injury" for d in r["deaths"])),
    ("deaths: starvation", lambda r: sum(d[1] == "starvation" for d in r["deaths"])),
    ("final population (both tribes)", lambda r: r["pops"]["A"][-1] + r["pops"]["B"][-1]),
    ("fights per agent-hour", lambda r: r["fights_per_agent_hour"]),
    ("fights with one side outnumbered", lambda r: r["outnumbered"]),
    ("share of time with a rival in view", lambda r: r["exposure"]),
    ("share of time reactive (timelike)", lambda r: r["reactive"]),
    ("food patches depleted", lambda r: r["depleted"]),
    ("mean pain", lambda r: r["pain"]),
    ("mean hunger coefficient", lambda r: r["hunger"]),
    *[(f"felt |F| on {nm}", (lambda r, i=i: r["affect_mix"][i])) for i, nm in enumerate(AFFECT_NAMES)],
    ("fight alone (whole run)", lambda r: r["f_all"][0]),
    ("fight w/ tribe-mates (whole run)", lambda r: r["f_all"][2]),
    ("fight alone, first quarter", lambda r: r["f_early"][0]),
    ("fight alone, last quarter", lambda r: r["f_late"][0]),
    ("fight w/ mates, first quarter", lambda r: r["f_early"][2]),
    ("fight w/ mates, last quarter", lambda r: r["f_late"][2]),
    ("recalls: experience", lambda r: r["recalls"].get("experience", 0)),
    ("recalls: imprinted", lambda r: r["recalls"].get("imprinted", 0)),
    ("recalls: inherited", lambda r: r["recalls"].get("inherited", 0)),
]
lines = ["E10 Affects as trivectors", "=========================",
         f"{len(SEEDS)} seeds x {T:.0f} s per signature, runs in parallel; two-tribe full-life world.",
         "", f"{'':36s}" + "".join(f"{n:>20s}" for n in SIGNATURES)]
for lab, f in rows:
    lines.append(f"{lab:36s}" + "".join(f"{m(n, f):>20s}" for n in SIGNATURES))
write_summary(OUT, lines)
