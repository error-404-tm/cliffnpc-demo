"""E3 Order dependence: trauma-then-comfort vs comfort-then-trauma.

Two runs are identical except that two events swap times:
  T (trauma)  - the threat attacks and the nearby peers scatter (20 s)
  F (comfort) - a shared meal: food appears next to the NPC and the peers
                gather close (20 s)
Both events touch two drives each (threat+social, hunger+social), so their
residuals are not aligned with one basis axis.  (With single-axis events the
updates are diagonal and commute exactly; see the summary.)
The NPC is held at a fixed spot so that the only difference between runs is
the event order (no trajectory divergence).  We compare the final metric g.

Three learning-law variants separate the sources of order dependence:
  1. default            - felt residual + forgetting toward g0
  2. no forgetting      - lambda = 0: removes recency, leaving path dependence
                          (the second event is felt through the geometry left
                          by the first, and the matrix exponentials don't commute)
  3. no forgetting, raw - lambda = 0 and residual not lowered with g: the
     residual          update is then additive to first order, so what
                          remains is the O(eps^2) exp commutator plus the
                          state dependence (sensitivities, frustration)
"""

import numpy as np

from common import outdir, plt, pmap, save, write_summary

from cliffnpc import algebra as ga
from cliffnpc.config import Config
from cliffnpc.sim import Simulation

OUT = outdir("e3_order")
SPOT = np.array([40.0, 40.0])
T_END = 400.0
W = 20.0
EARLY, LATE = 50.0, 150.0
PEERS_BASE = np.array([[8.0, 0.0], [-6.0, 6.0], [-4.0, -8.0]])
PEERS_CLOSE = 0.3 * PEERS_BASE


def run(order: str, forget: bool = True, felt: bool = True, scales: bool = True):
    cfg = Config()
    if not forget:
        cfg.cog.forget_rate = 0.0
    cfg.cog.felt_residual = felt
    if not scales:
        cfg.cog.use_sensitivity = False
        cfg.cog.frustration_gain = 0.0
    t_T, t_F = (EARLY, LATE) if order == "TF" else (LATE, EARLY)

    def hook(sim):
        t = sim.world.t
        w = sim.world
        if t_T <= t < t_T + W:          # attacked and abandoned: peers scatter
            w.food, w.peers = [], np.zeros((0, 2))
        elif t_F <= t < t_F + W:        # shared meal: food appears, peers gather close
            w.food, w.peers = [SPOT + np.array([2.0, 0.0])], SPOT + PEERS_CLOSE
        else:
            w.food, w.peers = [], SPOT + PEERS_BASE

    sim = Simulation(cfg, start=SPOT, path=lambda t: SPOT, world_hook=hook)
    sim.world.threat.schedule = [(t_T, t_T + W)]
    sim.world.threat.wander_sigma = 0.0   # identical attack geometry in both orders
    sim.world.haunt.flux = 0.0
    log = sim.run(T_END)
    return log


variants = {
    "default (felt residual + forgetting)": dict(forget=True, felt=True),
    "no forgetting": dict(forget=False, felt=True),
    "no forgetting, raw residual": dict(forget=False, felt=False),
    "... and no sensitivity/frustration scaling": dict(forget=False, felt=False, scales=False),
}
rows, logs = [], {}


def _run(job):
    order, kw = job
    return run(order, **kw)


jobs = [(order, kw) for kw in variants.values() for order in ("TF", "FT")]
results = iter(pmap(_run, jobs))
for name, kw in variants.items():
    a, b = next(results), next(results)
    logs[name] = (a, b)
    gA, gB = a["g"][-1], b["g"][-1]
    g0 = ga.ETA
    diff = np.linalg.norm(gA - gB)
    total = 0.5 * (np.linalg.norm(gA - g0) + np.linalg.norm(gB - g0))
    rows.append((name, diff, total, gA, gB))

iu = np.triu_indices(4)
labels = [f"g{i+1}{j+1}" for i, j in zip(*iu)]
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
a, b = logs["default (felt residual + forgetting)"]
for ax, (lg, title) in zip(axes[0], [(a, "T then F"), (b, "F then T")]):
    g = lg["g"]
    t = lg["t"]
    for k, (i, j) in enumerate(zip(*iu)):
        y = g[:, i, j] - ga.ETA[i, j]
        if np.abs(y).max() > 1e-3:
            ax.plot(t, y, label=labels[k])
    tT, tF = (EARLY, LATE) if title.startswith("T") else (LATE, EARLY)
    ax.axvspan(tT, tT + W, color="#8e44ad", alpha=0.15, label="threat")
    ax.axvspan(tF, tF + W, color="#e67e22", alpha=0.15, label="food")
    ax.set_title(f"{title}: g - η over time (default law)")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=7, ncol=3)
ax = axes[1][0]
x = np.arange(len(labels))
wbar = 0.2
for k, (name, diff, total, gA, gB) in enumerate(rows):
    ax.bar(x + (k - 1.5) * wbar, (gA - gB)[iu], wbar, label=name)
ax.set_xticks(x, labels, rotation=45)
ax.axhline(0, color="k", lw=0.5)
ax.set_title("final g(TF) − g(FT), per entry")
ax.legend(fontsize=7)
ax = axes[1][1]
names = [r[0] for r in rows]
ax.barh(names, [r[1] / r[2] for r in rows], color=["#2c3e50", "#2e86c1", "#95a5a6", "#d5dbdb"])
ax.set_xlabel("‖g_TF − g_FT‖_F  /  mean ‖g − η‖_F")
ax.set_title("order dependence relative to total learned change")
save(fig, OUT, "order.png")

lines = ["E3 Order dependence", "===================",
         f"NPC fixed at {SPOT.tolist()}; events at t={EARLY:.0f}s and t={LATE:.0f}s (20 s each); compared at t={T_END:.0f}s.",
         ""]
for name, diff, total, gA, gB in rows:
    lines.append(f"{name:38s} ||g_TF - g_FT||_F = {diff:.4f}   (mean ||g - eta||_F = {total:.4f}, ratio {diff/total:.1%})")
    lines.append(f"{'':38s} signature TF {ga.signature(gA)}, FT {ga.signature(gB)}")
r = {name: diff / total for name, diff, total, _, _ in rows}
k = list(r)
lines += [
    "",
    "Reading:",
    f"  * With forgetting (default) the final g depends strongly on order ({r[k[0]]:.0%} of the surviving",
    "    change), but that is mostly recency: the later event has decayed less.  That is real temporal",
    "    memory, but any leaky accumulator does the same.",
    f"  * Without forgetting, what is left is path dependence: {r[k[1]]:.1%} (felt residual), {r[k[2]]:.1%} (raw).",
    "    It comes from the state-dependent scales: the sensitivities |g^ii| (the first event changes how",
    "    strongly the second is weighted) and the frustration |C^B| (fear still lingering when the meal",
    "    arrives).",
    f"  * With those scales off as well, the difference is {r[k[3]]:.1e}.  The matrix-exponential",
    "    commutator alone gives no usable memory at slow learning rates.",
    "",
    "Why (important for §5.2): for L <- exp(eps A) L with eta-self-adjoint A, the induced change is",
    "dg = L^T (2 eta A) L eps.  If the generator's image in g, here sigma c c^T, does not depend on the",
    "current g, the flow on g is additive and commutes to first order; only O(eps^2) terms remember",
    "order.  'Trauma then comfort != comfort then trauma' therefore needs a generator that depends on the",
    "current geometry at first order (sensitivities, frustration, felt residual, or a nonlinear A(r, g)),",
    "or large single-event updates (unit test: two strong events at eps=0.3 differ by a few percent).",
    "The felt residual (r lowered with g) only matters once g has sizeable off-diagonals, which this short",
    "scripted run never builds.",
    "Also fixed while building this experiment: world time used to accumulate float error, which",
    "shifted scripted events by one physics step and faked about 10% order dependence.",
]
write_summary(OUT, lines)
