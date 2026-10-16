"""E5 Personality threshold: sweep g23 across +-1; the e23 plane goes elliptic -> hyperbolic.

Metric family (unit diagonal, threat timelike):
    g = [[1, 0,   0,   0 ],
         [0, 1,   g23, c ],
         [0, g23, 1,   0 ],
         [0, c,   0,  -1 ]]
Signature caveat: with c = 0 the metric has signature (2,2) for |g23| > 1,
since the e2-e3 block becomes Lorentzian on its own.  To stay inside
signature (3,1) across the sweep the threat axis has to be entangled with
social (c = g24 = 1.2 works for |g23| < sqrt(1 + c^2) = 1.56).  Then e2 - e3
is timelike, and so is e4 (g44 = -1): "group wanderlust" becomes a second,
reactive-like axis that leans on threat.

Part 1: the e23 generator's spectrum vs g23 (growth rate = max Re eigenvalue).
Part 2: drive-space flow exp(t M23) from the same start state (bounded vs runaway).
Part 3: a full NPC parked among its peers (peer channel rotates e23), with
        g23 = 0.5 vs 1.3: bounded tradeoff vs runaway reinforcement, which
        damping and the state cap then have to catch.
"""

import numpy as np
from scipy.linalg import expm

from common import DRIVE_COLORS, DRIVE_LABELS, outdir, plt, save, write_summary

from cliffnpc import algebra as ga
from cliffnpc.cognition import NPC
from cliffnpc.config import Config
from cliffnpc.sim import Simulation

OUT = outdir("e5_threshold")
C_ENT = 1.2


def metric(g23, c=C_ENT):
    g = ga.ETA.copy()
    g[1, 2] = g[2, 1] = g23
    g[1, 3] = g[3, 1] = c
    return g


sweep = np.linspace(-1.5, 1.5, 301)
rates, types, sigs, sigs_c0 = [], [], [], []
for v in sweep:
    g = metric(v)
    M = ga.basis_generator(1, 2, g)
    rates.append(np.linalg.eigvals(M).real.max())
    types.append(ga.basis_plane_type(1, 2, g))
    sigs.append(ga.signature(g))
    sigs_c0.append(ga.signature(metric(v, 0.0)))
rates = np.array(rates)
theory = np.sqrt(np.clip(sweep ** 2 - 1, 0, None))

# Part 2: flow
start = np.array([0.0, 0.3, 0.1, 0.0])
ts = np.linspace(0, 12, 400)
flows = {}
for v in (0.0, 0.5, 0.95, 1.05, 1.3):
    M = ga.basis_generator(1, 2, metric(v))
    flows[v] = np.array([expm(t * M) @ start for t in ts])

# Part 3: NPC among peers
PARK = np.array([24.0, 54.0])


def npc_run(v):
    cfg = Config()
    cfg.cog.forget_rate = 0.0
    cfg.cog.learn_rate = 0.0   # hold personality fixed for the comparison
    cfg.cog.peer_gain = 0.4
    g = metric(v)
    L = ga.frame_from_metric(g)
    npc = NPC(cfg.cog, L=L)
    sim = Simulation(cfg, npc=npc, start=PARK, path=lambda t: PARK)
    sim.world.threat.active = False
    sim.world.haunt.flux = 0.0
    return sim.run(120.0)


npc_logs = {v: npc_run(v) for v in (0.5, 1.3)}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0][0]
ax.plot(sweep, rates, color="#2c3e50", lw=2, label="max Re λ(M₂₃)")
ax.plot(sweep, theory, "--", color="#e67e22", label="√(g23² − 1)")
ax.axvspan(-1, 1, color="#27ae60", alpha=0.08, label="elliptic")
ax.set_xlabel("g23")
ax.set_ylabel("growth rate")
bad = [v for v, sg in zip(sweep, sigs) if sg != (3, 1, 0)]
ax.set_title(f"spectrum of the e23 generator (signature (3,1) holds for all {len(sweep)} points)"
             if not bad else "spectrum (some points leave signature (3,1)!)", fontsize=10)
ax.legend(fontsize=8)
ax = axes[0][1]
for (v, tr), col in zip(flows.items(), ["#95a5a6", "#27ae60", "#2e86c1", "#e67e22", "#c0392b"]):
    ax.plot(tr[:, 1], tr[:, 2], color=col, label=f"g23={v} ({ga.basis_plane_type(1, 2, metric(v))})")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect("equal")
ax.set_xlabel("social C²")
ax.set_ylabel("explore C³")
ax.set_title("flow exp(t M₂₃) from the same start, t ∈ [0, 12]")
ax.legend(fontsize=7)
for ax, (v, lg) in zip(axes[1], npc_logs.items()):
    for i in range(4):
        ax.plot(lg["t"], lg["C"][:, i], color=DRIVE_COLORS[i], label=DRIVE_LABELS[i])
    ax.set_title(f"NPC parked among peers, g23={v} ({ga.basis_plane_type(1, 2, metric(v))})")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=7)
save(fig, OUT, "threshold.png")

regimes = {}
for v, ty in zip(sweep, types):
    regimes.setdefault(ty, []).append(float(v))
c0_bad = [float(v) for v, sg in zip(sweep, sigs_c0) if sg != (3, 1, 0)]
maxC = {v: float(np.abs(lg["C"]).max()) for v, lg in npc_logs.items()}
endC = {v: np.round(lg["C"][-1], 3) for v, lg in npc_logs.items()}
flow_max = {v: float(np.abs(tr).max()) for v, tr in flows.items()}
lines = [
    "E5 Personality threshold",
    "========================",
    "Plane type of e2^e3 over the sweep: " + "; ".join(
        f"{ty}: " + (f"|g23| < {max(abs(x) for x in vs) + 0.01:.2f}" if ty == "elliptic"
                     else "g23 = " + ", ".join(f"{x:+.2f}" for x in sorted({round(x, 2) for x in vs}))
                     if ty == "parabolic" else f"|g23| > {min(abs(x) for x in vs) - 0.01:.2f}")
        for ty, vs in regimes.items()) + "  (expected threshold |g23| = 1)",
    f"Growth rate of M23 matches sqrt(g23^2 - 1): max deviation {np.abs(rates - theory).max():.2e}.",
    f"Metric family keeps signature (3,1) at every sweep point: {not bad} (with threat-social entanglement c={C_ENT}).",
    f"Without entanglement (c=0) the signature is lost for |g23| >= {min(abs(v) for v in c0_bad):.2f} (becomes (2,2) or degenerate),",
    "  Signature (3,1) allows exactly one timelike direction in the g-orthogonal sense.  If e2^e3 is",
    "  Lorentzian, its g-orthogonal complement must be positive definite, so e4 has to lean into the",
    "  social-explore plane (here g24 != 0) and take its timelike character from that plane.  Personally:",
    "  in a group-wanderlust NPC, threat is no longer independent of the social drive.",
    "",
    "Flow exp(t M23), max |C| over t in [0,12] from |C0|=0.32:",
] + [f"  g23={v:5.2f}: {m:9.3f}" for v, m in flow_max.items()] + [
    "",
    "NPC parked among peers (peer channel drives e23 rotation), learning frozen:",
] + [f"  g23={v}: max |C^i| = {maxC[v]:.3f}, final C = {endC[v]}" for v in npc_logs] + [
    "",
    "Reading: below the threshold, peer density produces a bounded social<->explore tradeoff.  Above it",
    "the same forcing produces exponential reinforcement.  Here damping cannot hold it and the state",
    f"saturates at the safety cap (|C| = {Config().cog.state_cap}) within ~10 s: one personality number crossing 1",
    "changes the qualitative dynamics, as §2.4 predicts.  Game-design consequence: a hyperbolic",
    "personality plane needs a real saturation law (e.g. damping that grows with |C|), not just the NaN",
    "guard.  The signature caveat above is worth carrying back into the design.",
]
write_summary(OUT, lines)
