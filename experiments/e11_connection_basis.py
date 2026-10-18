"""E11 Learned connection and basis magnitudes, Cl(2,2).

Two tribes, full life, trivector affects, event-chain episodic memory (partial
match -> fast weak replay priming affect), metric learning (diagonal and cross
terms), and a learned connection (situation -> drive bivector).

Conditions (all in parallel):
  sweep:   starting magnitudes of the two bases that run away most -- hunger and
           fight/flight -- each in {0.7, 1.0, 1.4} (g0 = eta * scale^2), connection on
  control: connection learning off, unit magnitudes
Learning snapshots every 100 s; interrogation reports for sample agents.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json  # noqa: E402

import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary  # noqa: E402

from cliffnpc.arena import Arena, life_config  # noqa: E402
from cliffnpc.connection import FEATURES  # noqa: E402
from cliffnpc.interrogate import explain, population_table  # noqa: E402

OUT = outdir("e11_connection_basis")
T = 3000.0
SEEDS = (0, 1)
SIG = (-1.0, 1.0, 1.0, -1.0)
SCALES = (0.7, 1.0, 1.4)
DRIVES = ("hunger", "social", "explore", "fight")


def conditions():
    out = [(f"h{sh}_f{sf}", sh, sf, True) for sh in SCALES for sf in SCALES]
    out.append(("control (no connection)", 1.0, 1.0, False))
    return out


def fight_stats(dec, t0, t1):
    if len(dec) == 0:
        return (np.nan, np.nan)
    d = dec[(dec[:, 0] >= t0) & (dec[:, 0] < t1)]
    f = lambda x: float(np.mean(x[:, 6] > 0)) if len(x) else np.nan  # noqa: E731
    return (f(d[d[:, 2] == 0]), f(d[d[:, 2] >= 2]))


def one(job):
    name, sh, sf, conn, seed = job
    cfg = life_config(SIG, seed, memory_model="episodic", connection=conn)
    cfg.basis_scale = (sh, 1.0, 1.0, sf)
    cfg.snapshot_every = 100.0
    ar = Arena(cfg)
    log = ar.run(T)
    alive = log["alive"]
    hours = float(alive.sum()) * ar.wcfg.dt * ar.wcfg.cog_every / 3600
    dec = np.array(ar.decisions)
    reports = [explain(a) for a in sorted(ar.agents, key=lambda a: -a.generation)[:2]]
    return dict(
        name=name, sh=sh, sf=sf, conn=conn, seed=seed, snapshots=ar.snapshots,
        births=len(ar.births), max_gen=max([b["generation"] for b in ar.births], default=0),
        deaths_injury=sum(d["cause"] == "injury" for d in ar.deaths),
        deaths_starvation=sum(d["cause"] == "starvation" for d in ar.deaths),
        final_pop=int(alive[-1].sum()),
        fights=len(ar.engagements) / max(hours, 1e-9),
        exposure=len(ar.decisions) / max(float(alive.sum()), 1.0),
        pain=float(np.nanmean(np.where(alive, log["pain"], np.nan))),
        f_early=fight_stats(dec, 0, T / 4), f_late=fight_stats(dec, 3 * T / 4, T),
        reports=reports, table=population_table(ar),
    )


jobs = [(n, sh, sf, c, s) for n, sh, sf, c in conditions() for s in SEEDS]
results = pmap(one, jobs)
by = {}
for r in results:
    by.setdefault(r["name"], []).append(r)


def mean_of(rs, f):
    v = np.array([f(r) for r in rs], float)
    return float(np.nanmean(v)), float(np.nanstd(v))


def last_snap(r, k):
    s = [x for x in r["snapshots"] if k in x]
    return s[-1][k] if s else np.nan


# ---- heatmaps over the (hunger, fight) magnitude grid ------------------------
metrics = {
    "fights per agent-hour": lambda r: r["fights"],
    "deaths (injury + starvation)": lambda r: r["deaths_injury"] + r["deaths_starvation"],
    "courage index (final)": lambda r: last_snap(r, "courage_index"),
    "numbers index (final)": lambda r: last_snap(r, "numbers_index"),
    "fight alone, last quarter": lambda r: r["f_late"][0],
    "fight w/ mates, last quarter": lambda r: r["f_late"][1],
}
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (lab, f) in zip(axes.flat, metrics.items()):
    M = np.array([[mean_of(by[f"h{sh}_f{sf}"], f)[0] for sf in SCALES] for sh in SCALES])
    im = ax.imshow(M, cmap="viridis", origin="lower")
    ax.set_xticks(range(3), [f"fight {s}" for s in SCALES])
    ax.set_yticks(range(3), [f"hunger {s}" for s in SCALES])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
    ax.set_title(lab, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
save(fig, OUT, "basis_grid.png")

# ---- learning over time: connection on vs off, and learned basis magnitudes --------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
ax = axes[0][0]
for name, col in (("h1.0_f1.0", "#c0392b"), ("control (no connection)", "#7f8c8d")):
    for k, r in enumerate(by[name]):
        t = [s["t"] for s in r["snapshots"]]
        ax.plot(t, [s.get("courage_index", np.nan) for s in r["snapshots"]], color=col,
                alpha=0.8, label=name if k == 0 else None)
ax.axhline(0, color="k", lw=0.5)
ax.set_title("courage index over time (connection's fight push with allies − alone)")
ax.legend(fontsize=8)
ax = axes[0][1]
for name in by:
    if name.startswith("control"):
        continue
    t = [s["t"] for s in by[name][0]["snapshots"]]
    v = np.nanmean([[s.get("numbers_index", np.nan) for s in r["snapshots"]] for r in by[name]], axis=0)
    ax.plot(t, v, lw=1, label=name)
ax.axhline(0, color="k", lw=0.5)
ax.set_title("numbers index over time (fight push 3v1 − 1v3)")
ax.legend(fontsize=6, ncol=3)
ax = axes[1][0]
xs = np.arange(len(by))
names = list(by)
for d in (0, 3):
    start = [by[n][0]["sh"] if d == 0 else by[n][0]["sf"] for n in names]
    end = [np.mean([last_snap(r, "basis")[d] for r in by[n]]) for n in names]
    ax.plot(xs, start, "o", color="#7f8c8d" if d == 0 else "#bdc3c7",
            label=f"{DRIVES[d]} start")
    ax.plot(xs, end, "s", color="#c0392b" if d == 0 else "#8e44ad", label=f"{DRIVES[d]} learned")
ax.set_xticks(xs, names, rotation=60, fontsize=7)
ax.set_title("basis magnitudes: start vs learned (population mean, final)")
ax.legend(fontsize=7)
ax = axes[1][1]
W = np.nanmean([last_snap(r, "W_feature") for r in by["h1.0_f1.0"]], axis=0)
ax.bar(range(len(FEATURES)), W, color="#2c3e50")
ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, fontsize=8)
ax.set_title("learned connection strength per situation feature (h1.0_f1.0)")
save(fig, OUT, "learning.png")

# ---- interrogation reports ----------------------------------------------------------
with open(OUT / "agent_reports.txt", "w") as fh:
    for name, rs in by.items():
        fh.write(f"\n######## {name}\n")
        for r in rs:
            fh.write(f"\n--- seed {r['seed']}\n")
            fh.write("\n\n".join(r["reports"]) + "\n")
with open(OUT / "snapshots.json", "w") as fh:
    json.dump({n: [r["snapshots"] for r in rs] for n, rs in by.items()}, fh)

rows = [
    ("fights per agent-hour", lambda r: r["fights"]),
    ("deaths: injury", lambda r: r["deaths_injury"]),
    ("deaths: starvation", lambda r: r["deaths_starvation"]),
    ("births", lambda r: r["births"]),
    ("max generation", lambda r: r["max_gen"]),
    ("final population", lambda r: r["final_pop"]),
    ("mean pain", lambda r: r["pain"]),
    ("fight alone, first quarter", lambda r: r["f_early"][0]),
    ("fight alone, last quarter", lambda r: r["f_late"][0]),
    ("fight w/ mates, first quarter", lambda r: r["f_early"][1]),
    ("fight w/ mates, last quarter", lambda r: r["f_late"][1]),
    ("courage index (final)", lambda r: last_snap(r, "courage_index")),
    ("numbers index (final)", lambda r: last_snap(r, "numbers_index")),
    ("learned |hunger|", lambda r: last_snap(r, "basis")[0]),
    ("learned |fight|", lambda r: last_snap(r, "basis")[3]),
    ("mean |cross term|", lambda r: last_snap(r, "cross_abs")),
    ("metric change: diagonal", lambda r: last_snap(r, "diag_change")),
    ("metric change: cross terms", lambda r: last_snap(r, "offdiag_change")),
    ("affect from memory replay", lambda r: last_snap(r, "primed_share")),
    ("memory activations (population)", lambda r: last_snap(r, "activations")),
]
lines = ["E11 Learned connection and basis magnitudes, Cl(2,2)", "=" * 52,
         f"{len(SEEDS)} seeds x {T:.0f} s per condition, all runs in parallel.  Columns: condition;",
         "rows: mean ± std over seeds.  h = starting |hunger|, f = starting |fight/flight|.", ""]
for lab, f in rows:
    lines.append(lab)
    lines.append("   " + "   ".join(f"{n}: {mean_of(by[n], f)[0]:.3f}" for n in by))
lines += ["", "Interrogation reports: agent_reports.txt; learning snapshots: snapshots.json."]
write_summary(OUT, lines)
