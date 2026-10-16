"""E6 (Phase 3) Village: one event spreading as metric change through gossip.

30 villagers on a ring around a shared food patch: 3 episodic, 9 metric and
18 background (sharing one collective metric).  Social graph: ring lattice
plus a few long links.  At t=60 s a threat attacks one villager.  We track
each villager's metric change |g_i(t) - g_i(t_event)| against graph distance
from the villagers who actually sensed the threat, with and without gossip.

Each condition (gossip on/off) is run with and without the attack from the same
seed; the event's footprint is the difference between the two runs, which
removes the background homogenisation that gossip causes anyway.

LOD transitions are exercised during the run:
  t=180 s  a far background villager is promoted to the metric tier: it starts
           from the *current* collective metric, so it already carries the event
  t=220 s  an episodic villager is demoted: its episodes are consolidated into
           L in chronological order
"""

import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # tiny matrices: threads only add overhead
import numpy as np  # noqa: E402

from common import outdir, plt, pmap, save, write_summary

from cliffnpc import algebra as ga
from cliffnpc.config import Config
from cliffnpc.memory import BACKGROUND, EPISODIC, METRIC
from cliffnpc.village import Village, VillageConfig, graph_distances

OUT = outdir("e6_village")
T = 300.0
T_REF = 170.0          # analysis time (before the LOD transitions)
T_PROMOTE, T_DEMOTE = 180.0, 220.0
T_EVENT = 60.0
TIER_NAMES = {EPISODIC: "episodic", METRIC: "metric", BACKGROUND: "background"}


def build(gossip: bool, event: bool):
    cfg = Config()
    vc = VillageConfig(gossip_eps=0.01 if gossip else 0.0)
    v = Village(cfg, vc)
    v.target = 0
    lair = v.homes[0] + 25.0 * v.radial[0]
    v.world.threat.pos = lair.copy()
    v.world.threat.lair = lair.copy()
    v._last_threat = lair.copy()
    v.world.threat.active = True
    # no-event runs keep the threat idle (same RNG use, identical until t_event)
    v.world.threat.schedule = [(T_EVENT, T_EVENT + 15.0)] if event else []
    v.world.threat.wander_sigma = 0.0
    return v


lod = {}


def promote_far(v):
    d = np.linalg.norm(v.homes - v.homes[v.target], axis=1)
    cands = [i for i in np.argsort(-d) if v.agents[i].tier == BACKGROUND]
    i = cands[0]
    g_col = v.collective.g
    v.agents[i].promote()
    lod["promoted"] = (i, np.linalg.norm(v.agents[i].g - g_col),
                       np.linalg.norm(g_col - ga.ETA), np.linalg.norm(v.agents[i].g - ga.ETA))


def demote_episodic(v):
    i = next(k for k, a in enumerate(v.agents) if a.tier == EPISODIC)
    a = v.agents[i]
    n_eps = len(a.episodes.slots)
    folds_before = a.folds
    g_before = a.g.copy()
    a.demote(len(v.agents))
    lod["demoted"] = (i, n_eps, folds_before, np.linalg.norm(a.g - g_before))


def one_village(job):
    """One (gossip, event) village run; returns the village, its log and the LOD records."""
    gossip, event = job
    v = build(gossip, event)
    lod_events = {T_PROMOTE: promote_far, T_DEMOTE: demote_episodic} if (gossip and event) else {}
    log = v.run(T, lod_events)
    return job, v, log, dict(lod)


t0 = time.time()
runs = {}
for job, v_, log_, lod_ in pmap(one_village, [(g, e) for g in (True, False) for e in (True, False)]):
    runs[job] = (v_, log_)
    if job == (True, True):
        lod.update(lod_)
elapsed = time.time() - t0

v, log = runs[True, True]
t = log["t"]
k_ev = np.searchsorted(t, T_EVENT)
# who sensed the threat during the attack window
win = (t >= T_EVENT) & (t <= T_EVENT + 30)
d_thr = np.linalg.norm(log["pos"][win] - log["threat"][win][:, None, :], axis=2)
sensed = np.where((d_thr < v.cfg.world.sensor_radius).any(axis=0))[0].tolist()
dist = graph_distances(v.adj, sensed)
tiers0 = log["tier"][0]


def effect(gossip):
    """Footprint of the event: |g_i(t) with event - g_i(t) without|."""
    a, b = runs[gossip, True][1], runs[gossip, False][1]
    return (np.linalg.norm(a["g"] - b["g"], axis=(2, 3)),
            np.linalg.norm(a["g_col"] - b["g_col"], axis=(1, 2)))


DG, COL = effect(True)
DG0, COL0 = effect(False)
k_ref = np.searchsorted(t, T_REF)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
order = np.lexsort((np.arange(len(dist)), dist))
ax = axes[0][0]
im = ax.imshow(DG[:, order].T, aspect="auto", origin="lower",
               extent=[t[0], t[-1], -0.5, len(order) - 0.5], cmap="magma")
ax.set_yticks(range(len(order)), [f"{int(dist[i]) if np.isfinite(dist[i]) else '∞'}·{TIER_NAMES[tiers0[i]][0]}"
                                  for i in order], fontsize=6)
ax.axvline(T_EVENT, color="w", lw=0.7)
ax.set_ylabel("villager (graph distance · tier)")
ax.set_title("event footprint |g_i(event) − g_i(no event)|, with gossip")
fig.colorbar(im, ax=ax)
ax = axes[0][1]
maxd = int(np.nanmax(dist[np.isfinite(dist)]))
cols = plt.cm.viridis(np.linspace(0, 1, maxd + 1))
for dd in range(maxd + 1):
    idx = [i for i in range(len(dist)) if dist[i] == dd and tiers0[i] != BACKGROUND]
    if idx:
        ax.plot(t, DG[:, idx].mean(axis=1), color=cols[dd], label=f"distance {dd} (n={len(idx)})")
        ax.plot(t, DG0[:, idx].mean(axis=1), color=cols[dd], ls=":", lw=0.9)
ax.plot(t, COL, color="#c0392b", lw=2, label="collective (background), gossip")
ax.plot(t, COL0, color="#c0392b", lw=1, ls=":")
ax.axvline(T_EVENT, color="k", lw=0.6)
for te in (T_PROMOTE, T_DEMOTE):
    ax.axvline(te, color="#7f8c8d", lw=0.6, ls="--")
ax.set_xlabel("t (s)")
ax.set_ylabel("mean event footprint (individuals)")
ax.set_title("solid: with gossip   dotted: without")
ax.legend(fontsize=7)
ax = axes[1][0]
pos = log["pos"]
ax.plot(v.homes[:, 0], v.homes[:, 1], "o", color="#bbb", ms=4)
for i in range(len(v.agents)):
    if dist[i] == 0:
        ax.plot(pos[:, i, 0], pos[:, i, 1], lw=0.5, color="#c0392b")
ax.plot(log["threat"][:, 0], log["threat"][:, 1], color="#8e44ad", lw=1, label="threat")
for i in range(len(v.agents)):
    for j in range(i + 1, len(v.agents)):
        if v.adj[i, j]:
            ax.plot(*zip(v.homes[i], v.homes[j]), color="#ddd", lw=0.5, zorder=0)
ax.plot(*v.world.food[0], "o", color="#e67e22", ms=10)
ax.set_aspect("equal")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_title("village: homes, social graph, threat path, sensed villagers' paths")
ax = axes[1][1]
# arrival time of half the final change, per distance
half = {}
for dd in range(maxd + 1):
    idx = [i for i in range(len(dist)) if dist[i] == dd and tiers0[i] != BACKGROUND]
    if not idx:
        continue
    m = DG[:, idx].mean(axis=1)
    peak_k = np.argmax(np.where(t <= T_REF, m, -1))
    k = np.argmax((m >= 0.5 * m[peak_k]) & (t > T_EVENT))
    half[dd] = (t[k] - T_EVENT, m[k_ref], DG0[k_ref, idx].mean(), t[peak_k] - T_EVENT, m[peak_k])
ds = sorted(half)
ax.bar([d - 0.2 for d in ds], [half[d][1] for d in ds], 0.4, label="with gossip")
ax.bar([d + 0.2 for d in ds], [half[d][2] for d in ds], 0.4, label="without gossip")
ax.set_xlabel("graph distance from villagers who sensed the threat")
ax.set_ylabel(f"mean event footprint at t={T_REF:.0f} s")
ax.legend()
ax.set_title("how far the event travelled (before the LOD changes)")
save(fig, OUT, "village.png")

pi, pdiff, pcol, pnorm = lod["promoted"]
di, dn, dfold, ddg = lod["demoted"]
lines = [
    "E6 Village (Phase 3)",
    "====================",
    f"{len(v.agents)} villagers: {sum(t_ == EPISODIC for t_ in tiers0)} episodic, "
    f"{sum(t_ == METRIC for t_ in tiers0)} metric, {sum(t_ == BACKGROUND for t_ in tiers0)} background; "
    f"social graph {int(v.adj.sum() // 2)} edges; 4 runs of {T:.0f} s (gossip on/off x event on/off) in {elapsed:.0f} s wall time.",
    f"Threat attacked villager {v.target} at t={T_EVENT:.0f}s; {len(sensed)} villagers sensed it: {sensed}.",
    "",
    "Event footprint |g_i(with event) - g_i(without)| for individuals (episodic+metric), by graph distance:",
]
for d in ds:
    lines.append(f"  distance {d}: at t={T_REF:.0f}s with gossip {half[d][1]:.4f}, without {half[d][2]:.4f};  "
                 f"with gossip: half of peak after {half[d][0]:.0f} s, peak {half[d][4]:.4f} at +{half[d][3]:.0f} s")
lines += [
    f"  collective metric footprint at t={T_REF:.0f}s: with gossip {COL[k_ref]:.4f}, without {COL0[k_ref]:.4f}",
    "",
    "LOD transitions:",
    f"  promoted background villager {pi} (far side of the ring) at t={T_PROMOTE:.0f}: |g_new - g_collective| = {pdiff:.4f};"
    f" the collective had drifted |g_col - eta| = {pcol:.4f} since the start, so the newcomer starts",
    "  with the town's current temperament (its recent history), plus a small personal perturbation.",
    f"  demoted episodic villager {di} at t={T_DEMOTE:.0f}: {dn} stored episodes consolidated chronologically "
    f"(plus {dfold} folded earlier on eviction); |delta g| from the consolidation = {ddg:.4f}.",
    "",
    "Reading:",
    "  * With gossip the footprint travels outward along the social graph: it reaches the first hops",
    "    later and much smaller (see half-peak times and peaks above).  Beyond about two hops it is at",
    "    the level of behavioural noise, so timing there is not meaningful.  Reputation and fear spread",
    "    as a change in how people feel, not as stored facts.",
    "  * Gossip also comforts the victims: the directly hit villagers end up with a smaller footprint",
    "    than without gossip, because their neighbours' calmer metrics pull them back.",
    "  * Without gossip, footprints beyond distance 0 are not transmitted metric.  They come from",
    "    behaviour: villagers sense each other as peers, so the attack changes everyone's trajectories",
    "    a little (chaotic divergence), and with it their residuals.",
    "  * Background villagers cost almost nothing individually, yet they feel the event through the",
    "    collective metric, which also receives the individuals' diffusion.",
]
write_summary(OUT, lines)
