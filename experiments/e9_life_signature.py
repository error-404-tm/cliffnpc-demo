"""E9 Life and signature: death, birth, inheritance, fear, under Cl(3,1) vs Cl(2,2).

Every agent in a simulation shares one drive signature; the concurrent runs vary it:
  (3,1)  threat timelike (the default)
  (2,2)  hunger timelike too: starving becomes reactive, the hunger-threat plane
         turns elliptic (bounded tradeoff) and hunger-explore hyperbolic
Life is on: starvation and injury kill (speed falls in proportion to injury),
witnessing a death shocks, sensitizes fear and imprints an escape memory of
the reaction, well-fed agents reproduce (NPCs with a partner, the anti-social
rival alone), children inherit their parents' strongest memories, personality
frame and part of their fear.  Injury drives flight through a reflex stronger
than any other channel.  Satiety slightly slows agents (food competition).
Food is finite and moves: patches run out as they are eaten and new ones appear
elsewhere; agents only know patches they have seen.

A cubic per-drive saturation law (sat_gain) is on: changing the signature turns
some designed-bounded rotations into boosts (in Cl(2,2) the foraging channel
in the hunger-explore plane), which otherwise run away to the safety cap.
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


def make(sig, seed) -> ArenaConfig:
    return life_config(sig, seed, sat_gain=SAT_GAIN)   # saturation law: contains boost runaways


def fight_stats(dec, t0, t1, faction):
    """Fraction of decisions with C4 > 0 (fight) when a rival is perceived, alone vs with allies."""
    if len(dec) == 0:
        return (np.nan, 0, np.nan, 0)
    d = dec[(dec[:, 0] >= t0) & (dec[:, 0] < t1) & (dec[:, 7] == faction)]
    alone = d[d[:, 2] == 0] if faction == 0 else d[d[:, 2] <= 1]
    grp = d[d[:, 2] >= 2] if faction == 0 else d[d[:, 2] >= 3]
    f = lambda x: float(np.mean(x[:, 6] > 0)) if len(x) else np.nan  # noqa: E731
    return (f(alone), len(alone), f(grp), len(grp))


def one(job):
    name, seed = job
    ar = Arena(make(SIGNATURES[name], seed))
    log = ar.run(T)
    alive, fac = log["alive"], log["faction"]
    npc_alive = (fac == 0) & alive
    fear = np.where(npc_alive, log["fear"], np.nan)
    with np.errstate(all="ignore"):
        fear_mean = np.nanmean(fear, axis=1)
    ev = [e for a in ar.agents if a.ltm for e in a.ltm.events]
    rec = collections.Counter(e["origin"] for e in ev if e["what"] == "recalled")
    dec = np.array(ar.decisions)
    npc_time = float(npc_alive.sum()) * ar.wcfg.dt * ar.wcfg.cog_every
    return dict(
        name=name, seed=seed,
        t=log["t"][::DS], pop_npc=npc_alive.sum(1)[::DS], pop_rival=((fac == 1) & alive).sum(1)[::DS],
        fear=fear_mean[::DS],
        births=[(b["t"], b["faction"], b["generation"]) for b in ar.births],
        deaths=[(d["t"], d["faction"], d["cause"], len(d["witnesses"])) for d in ar.deaths],
        engagements=len(ar.engagements), solo=sum(e["n_max"] == 1 for e in ar.engagements),
        npc_injury=float(sum(e["npc_taken"] for e in ar.engagements)),
        fights_per_npc_hour=len(ar.engagements) / max(npc_time, 1e-9) * 3600,
        recalls=dict(rec),
        imprinted=sum(e.get("origin") == "imprinted" and e["what"] in ("stored", "merged") for e in ev),
        pain=float(np.nanmean(np.where(npc_alive, log["pain"], np.nan))),
        hunger=float(np.nanmean(np.where(npc_alive, log["C"][:, :, 0], np.nan))),
        depleted=sum(e["what"] == "depleted" for e in ar.food_events),
        fight_early=fight_stats(dec, 0, T / 3, 0), fight_late=fight_stats(dec, 2 * T / 3, T, 0),
        rival_fight_early=fight_stats(dec, 0, T / 3, 1), rival_fight_late=fight_stats(dec, 2 * T / 3, T, 1),
    )


results = pmap(one, [(n, s) for n in SIGNATURES for s in SEEDS])
by = {n: [r for r in results if r["name"] == n] for n in SIGNATURES}
COL = {"Cl(3,1)": "#2e86c1", "Cl(2,2)": "#c0392b"}

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
ax = axes[0][0]
for n, rs in by.items():
    for k, r in enumerate(rs):
        ax.plot(r["t"], r["pop_npc"], color=COL[n], lw=1, alpha=0.6, label=f"{n} NPCs" if k == 0 else None)
        ax.plot(r["t"], r["pop_rival"], color=COL[n], lw=1, ls=":", alpha=0.8, label=f"{n} rivals" if k == 0 else None)
ax.set_xlabel("t (s)")
ax.set_ylabel("population")
ax.set_title("population (each line one seed)")
ax.legend(fontsize=8)
ax = axes[0][1]
for n, rs in by.items():
    for k, r in enumerate(rs):
        ax.plot(r["t"], r["fear"], color=COL[n], lw=1, alpha=0.7, label=n if k == 0 else None)
ax.set_xlabel("t (s)")
ax.set_ylabel("mean learned fear sensitivity (NPCs)")
ax.set_title("fear develops from injury and witnessed deaths")
ax.legend(fontsize=8)
ax = axes[1][0]
causes = ["starvation", "injury"]
x = np.arange(len(SIGNATURES))
for k, (fac, cause) in enumerate([(f, c) for f in ("npc", "rival") for c in causes]):
    vals = [np.mean([sum(d[1] == fac and d[2] == cause for d in r["deaths"]) for r in by[n]]) for n in SIGNATURES]
    ax.bar(x + (k - 1.5) * 0.2, vals, 0.2, label=f"{fac} {cause}")
births = [np.mean([len(r["births"]) for r in by[n]]) for n in SIGNATURES]
ax.plot(x, births, "k^", ms=10, label="births")
ax.set_xticks(x, list(SIGNATURES))
ax.set_ylabel("per run (mean of seeds)")
ax.set_title("deaths by faction and cause, births")
ax.legend(fontsize=7)
ax = axes[1][1]
w = 0.18
for k, (key, lab) in enumerate([("fight_early", "early"), ("fight_late", "late")]):
    for j, (idx, ctx) in enumerate(((0, "alone"), (2, "with allies"))):
        vals = [np.nanmean([r[key][idx] for r in by[n]]) for n in SIGNATURES]
        ax.bar(x + (2 * k + j - 1.5) * w, vals, w, label=f"{lab}, {ctx}")
ax.set_xticks(x, list(SIGNATURES))
ax.set_ylabel("fraction fighting when the rival is near (NPCs)")
ax.set_title("fight tendency: first vs last third of the run")
ax.legend(fontsize=7)
save(fig, OUT, "life.png")


def m(n, f):
    v = np.array([f(r) for r in by[n]], float)
    return f"{np.nanmean(v):8.2f} ± {np.nanstd(v):5.2f}"


rows = [
    ("births", lambda r: len(r["births"])),
    ("max generation", lambda r: max([b[2] for b in r["births"]], default=0)),
    ("NPC deaths: starvation", lambda r: sum(d[1] == "npc" and d[2] == "starvation" for d in r["deaths"])),
    ("NPC deaths: injury", lambda r: sum(d[1] == "npc" and d[2] == "injury" for d in r["deaths"])),
    ("rival deaths: starvation", lambda r: sum(d[1] == "rival" and d[2] == "starvation" for d in r["deaths"])),
    ("rival deaths: injury", lambda r: sum(d[1] == "rival" and d[2] == "injury" for d in r["deaths"])),
    ("rival extinct at end", lambda r: float(r["pop_rival"][-1] == 0)),
    ("witness events", lambda r: sum(d[3] for d in r["deaths"])),
    ("imprinted memories", lambda r: r["imprinted"]),
    ("fights per NPC-hour", lambda r: r["fights_per_npc_hour"]),
    ("solo fight fraction", lambda r: r["solo"] / max(r["engagements"], 1)),
    ("NPC mean pain", lambda r: r["pain"]),
    ("NPC mean hunger coefficient", lambda r: r["hunger"]),
    ("food patches depleted", lambda r: r["depleted"]),
    ("final NPC fear sensitivity", lambda r: r["fear"][-1]),
    ("recalls: experience", lambda r: r["recalls"].get("experience", 0)),
    ("recalls: imprinted", lambda r: r["recalls"].get("imprinted", 0)),
    ("recalls: inherited", lambda r: r["recalls"].get("inherited", 0)),
    ("NPC fight alone, early", lambda r: r["fight_early"][0]),
    ("NPC fight alone, late", lambda r: r["fight_late"][0]),
    ("NPC fight w/ allies, early", lambda r: r["fight_early"][2]),
    ("NPC fight w/ allies, late", lambda r: r["fight_late"][2]),
    ("rival fight alone, early", lambda r: r["rival_fight_early"][0]),
    ("rival fight alone, late", lambda r: r["rival_fight_late"][0]),
    ("rival fight vs crowd, early", lambda r: r["rival_fight_early"][2]),
    ("rival fight vs crowd, late", lambda r: r["rival_fight_late"][2]),
]
lines = ["E9 Life and signature", "=====================",
         f"{len(SEEDS)} seeds x {T:.0f} s per signature; one signature per simulation, runs in parallel.",
         "Life on (death, birth, inheritance, witnessing, injury reflex), sequence memory with loss",
         "aversion and present-affect gating, pain-aversion learning, fear learned from injury and deaths.",
         "", f"{'':30s}" + "".join(f"{n:>20s}" for n in SIGNATURES)]
for lab, f in rows:
    lines.append(f"{lab:30s}" + "".join(f"{m(n, f):>20s}" for n in SIGNATURES))
write_summary(OUT, lines)
