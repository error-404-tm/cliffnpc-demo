"""E2 Light-cone snap: a threat approach drives s>0 -> s<0 and a behaviour switch.

Part 1 (full sim): default world with the periodic threat.  Plot the interval
s(t) and the normalized interval s/|C|^2 with threat distance overlaid, and
the threat-directed component of the NPC's velocity (+ toward / - away).

Part 2 (mechanism, drive space only): the same constant e34 boost applied to
the same start state under three non-rotor settings:
  (a) no damping           - boosts are isometries, s is exactly conserved;
                             the state slides along a hyperbola and never
                             crosses the cone, however long the boost lasts.
  (b) isotropic damping    - still no crossing in this case.
  (c) anisotropic damping  - explore relaxes faster than threat ("fear
                             lingers"); the dominant eigenvector of
                             (boost - Gamma) becomes timelike and s goes negative.
"""

import numpy as np
from scipy.linalg import expm

from common import outdir, plt, save, write_summary

from cliffnpc import algebra as ga
from cliffnpc.config import Config
from cliffnpc.sim import Simulation

OUT = outdir("e2_lightcone")
cfg = Config()
sim = Simulation(cfg)
log = sim.run(500.0)
t, s, cone, td = log["t"], log["s"], log["cone"], log["threat_dist"]
C = log["C"]
pos = log["pos"]
tpos = log["threat_pos"]

# threat-directed velocity component (finite differences of logged positions)
vel = np.gradient(pos, t, axis=0)
to_threat = tpos - pos
u = to_threat / np.maximum(np.linalg.norm(to_threat, axis=1, keepdims=True), 1e-9)
v_threat = np.sum(vel * u, axis=1)

down = np.where((cone[1:] < 0) & (cone[:-1] >= 0))[0] + 1
up = np.where((cone[1:] >= 0) & (cone[:-1] < 0))[0] + 1

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
ax = axes[0]
ax.plot(t, cone, color="#2c3e50", lw=1.2, label="s / |C|²  (normalized interval)")
ax.plot(t, s, color="#95a5a6", lw=0.8, label="s = Cᵀ g C")
ax.axhline(0, color="k", lw=0.6)
ax.fill_between(t, -2, 2, where=cone < 0, color="#c0392b", alpha=0.12, label="reactive (timelike)")
ax.set_ylim(min(-1.0, cone.min() - 0.1), max(1.3, cone.max() + 0.1))
ax.set_ylabel("interval")
ax.legend(loc="lower left", fontsize=8)
ax2 = ax.twinx()
ax2.plot(t, td, color="#e67e22", lw=0.9, alpha=0.8)
ax2.set_ylabel("threat distance", color="#e67e22")
ax2.axhline(cfg.world.sensor_radius, color="#e67e22", ls=":", lw=0.8)
ax.set_title("s(t) with threat distance (dotted: sensor radius)")
ax = axes[1]
ax.plot(t, C[:, 3], color="#8e44ad", label="threat C⁴ (+fight / −flight)")
ax.plot(t, C[:, 2], color="#27ae60", label="explore C³")
ax.plot(t, C[:, 0], color="#c0392b", label="hunger C¹")
ax.axhline(0, color="k", lw=0.5)
ax.legend(fontsize=8, ncol=3)
ax.set_ylabel("coefficient")
ax = axes[2]
# (masked when on top of the threat, where the direction is undefined)
ax.plot(t, np.where((td < cfg.world.sensor_radius) & (td > 2.0), v_threat, np.nan), color="#2c3e50")
ax.axhline(0, color="k", lw=0.5)
ax.fill_between(t, -6, 6, where=cone < 0, color="#c0392b", alpha=0.12)
ax.set_ylabel("velocity toward threat")
ax.set_xlabel("t (s)")
ax.set_title("behaviour: + approach (fight) / − flee, while the threat is sensed and > 2 away")
save(fig, OUT, "snap.png")

# per-crossing stats.  A crossing is "threat-driven" if the threat is inside the
# sensor when it happens; otherwise it is a "transit": a spacelike drive (here
# hunger, while eating) sweeps through zero while residual fear lingers, so the
# state passes briefly through the null cone.
enc, transits = [], []
for k in down:
    nxt = up[up > k]
    dur = t[nxt[0]] - t[k] if len(nxt) else np.nan
    if td[k] > cfg.world.sensor_radius:
        transits.append((t[k], dur, C[k - 1], C[k]))
        continue
    w = (t > t[k] - 20) & (t < t[k] + 20)
    j = np.argmin(np.where(w, td, np.inf))
    pre = (t > t[k] - 6) & (t < t[k] - 2) & (td < cfg.world.sensor_radius) & (td > 2.0)
    post = (t > t[k]) & (t <= t[k] + 3) & (td > 1.0)
    enc.append((t[k], t[j], td[j], dur,
                np.nanmean(v_threat[pre]) if pre.any() else np.nan,
                np.mean(v_threat[post]) if post.any() else np.nan))

# ----- Part 2: mechanism ------------------------------------------------------
g = ga.ETA
M = ga.basis_generator(3, 2, g)       # M(e4, e3), the threat_e34 generator
phi = 0.6
C0v = np.zeros(4)
start = np.array([0.4, 0.0, -0.2, -0.05])
h = 0.01
settings = {
    "(a) no damping": np.zeros(4),
    "(b) isotropic damping 0.2": np.full(4, 0.2),
    "(c) anisotropic (explore 0.5, threat 0.04)": np.array([0.05, 0.3, 0.5, 0.04]),
}
E = expm(h * phi * M)
mech = {}
for name, gam in settings.items():
    Cm = start.copy()
    tr = [Cm.copy()]
    for _ in range(int(8 / h)):
        Cm = E @ Cm
        Cm = Cm - h * gam * (Cm - C0v)
        tr.append(Cm.copy())
    tr = np.array(tr)
    mech[name] = tr
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
tt = np.arange(len(next(iter(mech.values())))) * h
for (name, tr), col in zip(mech.items(), ["#7f8c8d", "#2e86c1", "#c0392b"]):
    sv = np.einsum("ij,jk,ik->i", tr, g, tr)
    axes[0].plot(tt, sv, label=name, color=col)
    axes[1].plot(tr[:, 2], tr[:, 3], label=name, color=col)
axes[0].axhline(0, color="k", lw=0.6)
axes[0].set_xlabel("t")
axes[0].set_ylabel("s = Cᵀ η C")
axes[0].set_title("constant e34 boost (rapidity rate 0.6) from the same start")
axes[0].legend(fontsize=8)
lim = 1.2
xs = np.linspace(-lim, lim, 2)
axes[1].plot(xs, xs, "k:", lw=0.7)
axes[1].plot(xs, -xs, "k:", lw=0.7)
axes[1].set_xlim(-lim, lim)
axes[1].set_ylim(-0.2, lim)
axes[1].set_xlabel("explore C³")
axes[1].set_ylabel("threat C⁴")
axes[1].set_title("(C³, C⁴) projection; dotted = light cone of this plane")
save(fig, OUT, "mechanism.png")
s_end = {name: float(tr[-1] @ g @ tr[-1]) for name, tr in mech.items()}
s_dev_a = np.abs(np.einsum("ij,jk,ik->i", mech["(a) no damping"], g, mech["(a) no damping"])
                 - start @ g @ start).max()

lines = [
    "E2 Light-cone snap",
    "==================",
    f"Full sim, 500 s, threat attacks every {cfg.world.threat_period:.0f} s:",
    f"  deliberative->reactive crossings: {len(down)} ({len(enc)} threat-driven, {len(transits)} transits)",
]
for k, (tc, tca, dca, dur, vpre, vpost) in enumerate(enc):
    lines.append(f"  threat #{k+1}: crossed at t={tc:.1f}s (closest approach t={tca:.1f}s, d={dca:.1f}); "
                 f"reactive for {dur:.1f}s; velocity toward threat {vpre:+.2f} before -> "
                 + (f"{vpost:+.2f} after" if np.isfinite(vpost) else "already in contact"))
lines += [
    "  Each threat approach pushes the state through the null cone a few seconds before contact, and",
    "  steering becomes threat-dominated.  This temperament fights (C⁴ > 0, approach): it was hungry and",
    "  home-leaning, and the e34 generator M(e4,e3) turns 'home' into 'fight' (the sign of C³ when the",
    "  threat arrives picks fight or flight, see coupling.CHANNELS).  It stays reactive for ~10-20 s",
    "  because threat damping is the slowest (fear lingers).",
]
for tc, dur, Cb, Ca in transits:
    lines.append(f"  transit: t={tc:.1f}s, reactive for {dur:.1f}s, hunger {Cb[0]:+.2f} -> {Ca[0]:+.2f} "
                 f"with residual threat {Ca[3]:+.2f}")
if transits:
    lines += [
        "  Transits are a second, unplanned way to snap: when eating sweeps hunger through zero while",
        "  fear from the last encounter is still decaying, the state briefly becomes threat-dominated.",
        "  It is a real knife-edge (§2.3), but the steering might want hysteresis to ignore it.",
    ]
lines += [
    "",
    "Mechanism (drive space only, constant boost for 8 time units):",
]
for name, v in s_end.items():
    lines.append(f"  {name:45s} s: {start @ g @ start:+.3f} -> {v:+.3f}")
lines += [
    f"  (a) max |s(t) - s(0)| = {s_dev_a:.1e}: a boost is an isometry, so on its own it can NEVER cross",
    "      the cone; it only drives the state toward the null direction (s/|C|^2 -> 0).",
    "  CORRECTION TO SPEC §2.3: 'boosts can carry a state across the cone' needs a non-isometric partner.",
    "  Here it is the per-drive damping: with gamma_explore > gamma_threat the dominant eigenvector of",
    "  (phi K - Gamma) in the e3-e4 plane has |C4| > |C3|, i.e. it is timelike.  Fear lingering longer",
    "  than curiosity is what makes the snap possible.",
]
write_summary(OUT, lines)
