"""E8 Long-term sequence memory in the arena.

Behaviour sequences that ended in high reward (appetitive) or high pain relief
(aversive) are stored with the sense data present when they began, and replayed
as attractors when that sense data is approximately present in short-term memory.
Compared: memory on vs off, 4 seeds x 1500 s, everything else equal.  Then the
memory Clifford space: basis, signature Cl(p, q), and where memories lie in it.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from common import outdir, plt, save, write_summary  # noqa: E402

from cliffnpc.arena import NPC_FACTION, Arena, ArenaConfig  # noqa: E402
from cliffnpc.config import LTMConfig  # noqa: E402
from cliffnpc.ltm import APPETITIVE, AVERSIVE  # noqa: E402

OUT = outdir("e8_sequence_memory")
T = 1500.0
SEEDS = (0, 1, 2, 3)


def replay_outcomes(log, n_npcs, horizon_ticks=20):
    """Pain change over the `horizon` after a replay starts vs after random non-replay ticks."""
    rep = log["replay"][:, :n_npcs]
    pain = log["pain"][:, :n_npcs]
    starts, rand = [], []
    rng = np.random.default_rng(0)
    for i in range(n_npcs):
        r = rep[:, i]
        s = np.where((r[1:] >= 0) & (r[:-1] < 0))[0] + 1
        s = s[s + horizon_ticks < len(r)]
        starts += list(pain[s + horizon_ticks, i] - pain[s, i])
        quiet = np.where(r[:-horizon_ticks] < 0)[0]
        pick = rng.choice(quiet, size=min(len(quiet), max(len(s), 1)), replace=False)
        rand += list(pain[pick + horizon_ticks, i] - pain[pick, i])
    return np.array(starts), np.array(rand)


rows = {True: [], False: []}
keep = None
for use in (True, False):
    for sd in SEEDS:
        ar = Arena(ArenaConfig(seed=sd, ltm=LTMConfig() if use else None))
        log = ar.run(T)
        n = ar.cfg.n_npcs
        r = dict(
            pain=float(log["pain"][:, :n].mean()),
            engagements=len(ar.engagements),
            solo=sum(e["n_max"] == 1 for e in ar.engagements),
            npc_injury=float(sum(e["npc_taken"] for e in ar.engagements)),
            immobile=float((~log["mobile"][:, :n]).mean()),
        )
        if use:
            r["replay"] = float((log["replay"][:, :n] >= 0).mean())
            ev = [e for a in ar.agents[:n] for e in a.ltm.events]
            r["stored_app"] = sum(e["what"] == "stored" and e["kind"] == APPETITIVE for e in ev)
            r["stored_av"] = sum(e["what"] == "stored" and e["kind"] == AVERSIVE for e in ev)
            r["merged"] = sum(e["what"] == "merged" for e in ev)
            r["recalls"] = sum(e["what"] == "recalled" for e in ev)
            d_rep, d_rand = replay_outcomes(log, n)
            r["dpain_replay"], r["dpain_random"] = float(d_rep.mean()), float(d_rand.mean())
            sigs = []
            for a in ar.agents:
                sp = a.ltm.build_space()
                sigs.append(sp.signature if sp else (0, 0))
            r["signatures"] = sigs
            if sd == SEEDS[0]:
                keep = ar
        rows[use].append(r)

# ---- memory Clifford space of one NPC -------------------------------------
a0 = next(a for a in keep.agents if a.faction == NPC_FACTION and (a.ltm.slots["active"] == 1).sum() >= 4)
sp = a0.ltm.build_space()
X, v = a0.ltm.memory_vectors()
act = np.nonzero(a0.ltm.slots["active"] == 1)[0]
kinds = a0.ltm.slots["kind"][act]
Y = X @ sp.basis.T
ip = int(np.argmax(sp.eigenvalues))            # strongest approach mode
iq = int(np.argmin(sp.eigenvalues)) if (sp.signs < 0).any() else (1 if len(sp.signs) > 1 else 0)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0][0]
cols = ["#2e86c1" if lam > 0 else "#c0392b" for lam in sp.eigenvalues]
ax.bar(range(len(sp.eigenvalues)), sp.eigenvalues, color=cols)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("memory basis vector")
ax.set_ylabel("eigenvalue of Σ v_m x_m x_mᵀ")
ax.set_title(f"{a0.name}: memory Clifford space Cl{sp.signature}  (blue approach, red escape)")
ax = axes[0][1]
for k, c, lab in ((APPETITIVE, "#2e86c1", "appetitive (reward)"), (AVERSIVE, "#c0392b", "aversive (relief)")):
    m = kinds == k
    s = a0.ltm.slots["strength"][act][m]
    ax.scatter(Y[m, ip], Y[m, iq], s=30 + 60 * s / max(s.max(), 1e-9) if m.any() else 30, c=c, label=lab, alpha=0.75)
ax.axhline(0, color="k", lw=0.4)
ax.axvline(0, color="k", lw=0.4)
ax.set_xlabel(f"approach mode {ip}")
ax.set_ylabel(f"escape mode {iq}")
ax.set_title("memories in the leading approach/escape plane (size = strength)")
ax.legend(fontsize=8)
SENSE = ["food dist", "closing", "peers", "from home", "rival dist", "health", "pain", "allies"]
for ax, k, title in ((axes[1][0], APPETITIVE, "strongest appetitive memory"),
                     (axes[1][1], AVERSIVE, "strongest aversive memory")):
    m = np.nonzero(kinds == k)[0]
    if not len(m):
        ax.set_title(f"no {title.split()[1]} memory")
        continue
    j = act[m[np.argmax(a0.ltm.slots["strength"][act][m])]]
    slot = a0.ltm.slots[j]
    t = np.arange(slot["seq"].shape[0]) * keep.wcfg.dt * keep.wcfg.cog_every * a0.ltm.cfg.stride
    for d, (lab, c) in enumerate(zip(["hunger", "social", "explore", "threat"],
                                     ["#c0392b", "#2e86c1", "#27ae60", "#8e44ad"])):
        ax.plot(t, slot["seq"][:, d], color=c, label=lab)
    ax.set_xlabel("s into the sequence")
    ax.set_ylabel("drive coefficient")
    key_txt = "  ".join(f"{n}={x:+.2f}" for n, x in zip(SENSE, slot["key"]))
    ax.set_title(f"{title} (valence {slot['valence']:.2f}, used {int(slot['uses'])}x)", fontsize=10)
    ax.text(0.01, 0.01, "key: " + key_txt, transform=ax.transAxes, fontsize=6, wrap=True)
    ax.legend(fontsize=7)
save(fig, OUT, "memory_space.png")


def agg(key, use=True):
    x = np.array([r[key] for r in rows[use]], float)
    return f"{x.mean():.3f} ± {x.std():.3f}"


lines = [
    "E8 Long-term sequence memory",
    "============================",
    f"Arena, 5 NPCs + 1 rival, {len(SEEDS)} seeds x {T:.0f} s; memory on vs off (pain-aversion metric learning on in both).",
    "",
    f"{'':28s}{'memory ON':>22s}{'memory OFF':>22s}",
]
for k, lab in (("pain", "NPC mean pain"), ("engagements", "engagements"), ("solo", "solo engagements"),
               ("npc_injury", "total NPC injury"), ("immobile", "NPC immobile fraction")):
    lines.append(f"{lab:28s}{agg(k, True):>22s}{agg(k, False):>22s}")
lines += [
    "",
    "Memory activity (NPCs, per seed):",
]
for sd, r in zip(SEEDS, rows[True]):
    lines.append(f"  seed {sd}: stored {r['stored_app']} appetitive / {r['stored_av']} aversive, merged {r['merged']}, "
                 f"recalled {r['recalls']}x, replaying {r['replay']:.0%} of the time; "
                 f"pain change over 4 s after a replay starts {r['dpain_replay']:+.3f} vs {r['dpain_random']:+.3f} after random ticks")
    lines.append(f"          memory-space signatures (npc0..npc4, rival): {r['signatures']}")
lines += [
    "",
    f"Memory Clifford space of {a0.name} (seed {SEEDS[0]}): signature Cl{sp.signature}, eigenvalues {np.round(sp.eigenvalues, 3).tolist()}",
    "  Basis = leading eigenvectors of the valence-weighted second moment of stored (key, sequence) vectors;",
    "  approach memories contribute +strength, escape memories -strength, so eigenvalue signs give the signature.",
    "  Planes between two approach (or two escape) modes are elliptic; mixed approach/escape planes are hyperbolic,",
    "  the same causal structure as the drive algebra.",
]
write_summary(OUT, lines)
