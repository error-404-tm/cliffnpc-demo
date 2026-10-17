"""E9 Life and signature: two identical tribes competing for finite food, Cl(3,1) vs Cl(2,2).

Every agent in a simulation shares one drive signature; the concurrent runs vary it:
  (3,1)  threat timelike (the default)
  (2,2)  hunger timelike too: starving becomes reactive, the hunger-threat plane
         turns elliptic (bounded tradeoff) and hunger-explore hyperbolic
Two tribes, A and B, start equal in number and share the same dynamics; each
regards only its own members as peers and the other tribe as rival.  Fights
between them are symmetric: a side of n against m takes m/(n+m) of the damage.
Life is on: starvation and injury kill (speed falls with injury), witnessing a
death shocks, sensitizes fear and imprints an escape memory, well-fed pairs
reproduce and children inherit memories, personality frame and part of their
fear.  Injury drives flight through the strongest reflex.  Satiety slows a little.
Food is finite: patches run out as they are eaten; new ones spawn half at two
fixed sites (equidistant from both homes) and half anywhere; agents only know
patches they have seen.  Hunger builds fast, starvation begins at 0.8, and
hungry agents search with weight hunger + 2 hunger^2.  A cubic saturation law
contains boost runaways under either signature.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import collections  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

from cliffnpc.arena import Arena, ArenaConfig, life_config  # noqa: E402

OUT = outdir("e9_life_signature")
T = 2000.0
SEEDS = (0, 1, 2, 3)
SIGNATURES = {"Cl(3,1)": (1.0, 1.0, 1.0, -1.0), "Cl(2,2)": (-1.0, 1.0, 1.0, -1.0)}
DS = 25                      # log ticks per time-series sample (5 s)
SAT_GAIN = 0.5               # cubic drive damping (CognitionConfig.sat_gain)
TRIBES = ("A", "B")


def make(sig, seed) -> ArenaConfig:
    return life_config(sig, seed, sat_gain=SAT_GAIN)


def fight_stats(dec, t0, t1):
    """Fraction of decisions with C4 > 0 (fight) when a rival is perceived,
    alone (no tribe-mates near) vs with >= 2 tribe-mates near; both tribes pooled."""
    if len(dec) == 0:
        return (np.nan, 0, np.nan, 0)
    d = dec[(dec[:, 0] >= t0) & (dec[:, 0] < t1)]
    alone, grp = d[d[:, 2] == 0], d[d[:, 2] >= 2]
    f = lambda x: float(np.mean(x[:, 6] > 0)) if len(x) else np.nan  # noqa: E731
    return (f(alone), len(alone), f(grp), len(grp))


def one(job):
    name, seed = job
    ar = Arena(make(SIGNATURES[name], seed))
    log = ar.run(T)
    alive, fac = log["alive"], log["faction"]
    pops = {f: ((fac == k) & alive).sum(1)[::DS] for k, f in enumerate(ar.factions)}
    with np.errstate(all="ignore"):
        fear = np.nanmean(np.where(alive, log["fear"], np.nan), axis=1)[::DS]
    ev = [e for a in ar.agents if a.ltm for e in a.ltm.events]
    rec = collections.Counter(e["origin"] for e in ev if e["what"] == "recalled")
    dec = np.array(ar.decisions)
    agent_hours = float(alive.sum()) * ar.wcfg.dt * ar.wcfg.cog_every / 3600
    sides = [e.get("sides", {}) for e in ar.engagements]
    return dict(
        name=name, seed=seed, t=log["t"][::DS], pops=pops, fear=fear,
        births=[(b["t"], b["faction"], b["generation"]) for b in ar.births],
        deaths=[(d["t"], d["faction"], d["cause"], len(d["witnesses"])) for d in ar.deaths],
        engagements=len(ar.engagements),
        outnumbered=sum(1 for s in sides if s and min(s.values()) == 1 and max(s.values()) > 1),
        fights_per_agent_hour=len(ar.engagements) / max(agent_hours, 1e-9),
        recalls=dict(rec),
        imprinted=sum(e.get("origin") == "imprinted" and e["what"] in ("stored", "merged") for e in ev),
        pain=float(np.nanmean(np.where(alive, log["pain"], np.nan))),
        hunger=float(np.nanmean(np.where(alive, log["C"][:, :, 0], np.nan))),
        depleted=sum(e["what"] == "depleted" for e in ar.food_events),
        fight_early=fight_stats(dec, 0, T / 3), fight_late=fight_stats(dec, 2 * T / 3, T),
    )


results = pmap(one, [(n, s) for n in SIGNATURES for s in SEEDS])
by = {n: [r for r in results if r["name"] == n] for n in SIGNATURES}
COL = {"Cl(3,1)": "#2e86c1", "Cl(2,2)": "#c0392b"}

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
ax = axes[0][0]
for n, rs in by.items():
    for k, r in enumerate(rs):
        ax.plot(r["t"], r["pops"]["A"], color=COL[n], lw=1, alpha=0.7, label=f"{n} tribe A" if k == 0 else None)
        ax.plot(r["t"], r["pops"]["B"], color=COL[n], lw=1, ls=":", alpha=0.9, label=f"{n} tribe B" if k == 0 else None)
ax.set_xlabel("t (s)")
ax.set_ylabel("population")
ax.set_title("tribe populations (each line one seed)")
ax.legend(fontsize=8)
ax = axes[0][1]
for n, rs in by.items():
    for k, r in enumerate(rs):
        ax.plot(r["t"], r["fear"], color=COL[n], lw=1, alpha=0.7, label=n if k == 0 else None)
ax.set_xlabel("t (s)")
ax.set_ylabel("mean learned fear sensitivity")
ax.set_title("fear develops from injury and witnessed deaths")
ax.legend(fontsize=8)
ax = axes[1][0]
x = np.arange(len(SIGNATURES))
for k, cause in enumerate(("starvation", "injury")):
    vals = [np.mean([sum(d[2] == cause for d in r["deaths"]) for r in by[n]]) for n in SIGNATURES]
    ax.bar(x + (k - 0.5) * 0.3, vals, 0.3, label=f"deaths: {cause}")
births = [np.mean([len(r["births"]) for r in by[n]]) for n in SIGNATURES]
ax.plot(x, births, "k^", ms=10, label="births")
ax.set_xticks(x, list(SIGNATURES))
ax.set_ylabel("per run (mean of seeds, both tribes)")
ax.set_title("deaths by cause, births")
ax.legend(fontsize=8)
ax = axes[1][1]
w = 0.18
for k, (key, lab) in enumerate([("fight_early", "early"), ("fight_late", "late")]):
    for j, (idx, ctx) in enumerate(((0, "alone"), (2, "with tribe-mates"))):
        vals = [np.nanmean([r[key][idx] for r in by[n]]) for n in SIGNATURES]
        ax.bar(x + (2 * k + j - 1.5) * w, vals, w, label=f"{lab}, {ctx}")
ax.set_xticks(x, list(SIGNATURES))
ax.set_ylabel("fraction fighting when a rival is near")
ax.set_title("fight tendency: first vs last third of the run")
ax.legend(fontsize=7)
save(fig, OUT, "life.png")


def m(n, f):
    v = np.array([f(r) for r in by[n]], float)
    return f"{np.nanmean(v):8.2f} ± {np.nanstd(v):5.2f}"


def per_tribe(field, cause=None):
    return [(f"{field} {t}", (lambda r, t=t: sum(d[1] == t and (cause is None or d[2] == cause)
                                                for d in r["deaths"])))
            for t in TRIBES]


rows = [
    ("births", lambda r: len(r["births"])),
    ("max generation", lambda r: max([b[2] for b in r["births"]], default=0)),
    ("deaths: starvation", lambda r: sum(d[2] == "starvation" for d in r["deaths"])),
    ("deaths: injury", lambda r: sum(d[2] == "injury" for d in r["deaths"])),
    *[(f"deaths, tribe {t}", (lambda r, t=t: sum(d[1] == t for d in r["deaths"]))) for t in TRIBES],
    *[(f"final population, tribe {t}", (lambda r, t=t: r["pops"][t][-1])) for t in TRIBES],
    ("a tribe extinct at end", lambda r: float(min(r["pops"]["A"][-1], r["pops"]["B"][-1]) == 0)),
    ("witness events", lambda r: sum(d[3] for d in r["deaths"])),
    ("imprinted memories", lambda r: r["imprinted"]),
    ("food patches depleted", lambda r: r["depleted"]),
    ("fights per agent-hour", lambda r: r["fights_per_agent_hour"]),
    ("fights with one side outnumbered 1:n", lambda r: r["outnumbered"] / max(r["engagements"], 1)),
    ("mean pain", lambda r: r["pain"]),
    ("mean hunger coefficient", lambda r: r["hunger"]),
    ("final fear sensitivity", lambda r: r["fear"][-1]),
    ("recalls: experience", lambda r: r["recalls"].get("experience", 0)),
    ("recalls: imprinted", lambda r: r["recalls"].get("imprinted", 0)),
    ("recalls: inherited", lambda r: r["recalls"].get("inherited", 0)),
    ("fight alone, early", lambda r: r["fight_early"][0]),
    ("fight alone, late", lambda r: r["fight_late"][0]),
    ("fight w/ tribe-mates, early", lambda r: r["fight_early"][2]),
    ("fight w/ tribe-mates, late", lambda r: r["fight_late"][2]),
]
lines = ["E9 Life and signature: two tribes", "==================================",
         f"{len(SEEDS)} seeds x {T:.0f} s per signature; one signature per simulation, runs in parallel.",
         "Two identical tribes (5 each at start), symmetric combat, finite food (half at two fixed",
         "sites, half random), life on, sequence memory, pain-aversion learning, learned fear.",
         "", f"{'':38s}" + "".join(f"{n:>20s}" for n in SIGNATURES)]
for lab, f in rows:
    lines.append(f"{lab:38s}" + "".join(f"{m(n, f):>20s}" for n in SIGNATURES))
write_summary(OUT, lines)
