"""E4 Holonomy: loops around the haunted disc vs equal-length loops elsewhere.

Both loops start and end at the same point P and have the same length:
  A encloses the haunted disc (curvature Phi of B_h = e23 inside it)
  B lies entirely outside it
Part 1, pure transport: net rotation of the cognitive state per loop.
Part 2, full cognition: the NPC walks 3 scripted loops (world coupling off,
so the only difference between A and B is the connection), then is released
at P with full coupling.  Does the difference persist in later behaviour?
"""

import numpy as np
from scipy.linalg import expm

from common import DRIVE_COLORS, DRIVE_LABELS, outdir, plt, save, write_summary

from cliffnpc import algebra as ga
from cliffnpc.config import Config
from cliffnpc.sim import Simulation
from cliffnpc.world import HauntedDisc

OUT = outdir("e4_holonomy")
cfg = Config()
c = np.array(cfg.world.haunt_center)
R_disc = cfg.world.haunt_radius
Phi = cfg.world.haunt_flux
R_loop = 14.0
P = c + np.array([0.0, -R_loop])                  # common start/end point
centres = {"A (encloses disc)": c, "B (outside)": P + np.array([0.0, -R_loop])}
SPEED = 3.0
N_LOOPS = 3
period = 2 * np.pi * R_loop / SPEED


def loop_path(centre):
    phase0 = np.arctan2(*(P - centre)[::-1])

    def path(t):
        th = phase0 + 2 * np.pi * min(t, N_LOOPS * period) / period
        return centre + R_loop * np.array([np.cos(th), np.sin(th)])
    return path


# ----- Part 1: pure transport -------------------------------------------------
disc = HauntedDisc(c, R_disc, Phi)
M = ga.basis_generator(*cfg.cog.holonomy_plane, ga.ETA)
part1 = {}
for name, cen in centres.items():
    path = loop_path(cen)
    ts = np.linspace(0, period, 2001)
    pts = np.array([path(t) for t in ts])
    alpha = sum(disc.segment_integral(p, q) for p, q in zip(pts[:-1], pts[1:]))
    C = np.array([0.0, 0.5, 0.0, 0.0])
    C1 = expm(alpha * M) @ C
    angle = np.degrees(np.arctan2(C1[2], C1[1]))
    part1[name] = (alpha, angle, C1)
expected = Phi * np.pi * R_disc ** 2

# ----- Part 2: full cognition -------------------------------------------------
T_LOOPS = N_LOOPS * period
T_AFTER = 300.0


def run(centre):
    cfgx = Config()

    def hook(sim):
        if sim.world.t >= T_LOOPS and sim.path is not None:
            sim.path = None               # release at P
            sim.active_channels = None    # full world coupling from now on

    sim = Simulation(cfgx, start=P, path=loop_path(centre), active_channels=set(),
                     world_hook=hook)
    sim.world.threat.active = False
    return sim.run(T_LOOPS + T_AFTER)


logs = {name: run(cen) for name, cen in centres.items()}
(nA, lA), (nB, lB) = logs.items()
t = lA["t"]
dC = np.linalg.norm(lA["C"] - lB["C"], axis=1)
dpos = np.linalg.norm(lA["pos"] - lB["pos"], axis=1)
rel = t >= T_LOOPS
i_rel = np.argmax(rel)
dC_rel = dC[i_rel]
# decay time of the state difference after release (to 1/e)
after = dC[rel]
below = np.where(after < dC_rel / np.e)[0]
tau = (t[rel][below[0]] - T_LOOPS) if len(below) else np.nan


def meal_times(lg):
    fd = np.linalg.norm(lg["pos"] - np.array(cfg.world.food[0]), axis=1)
    at = fd < cfg.cog.eat_radius
    k = np.where(at[1:] & ~at[:-1])[0] + 1
    return lg["t"][k]


mA, mB = meal_times(lA), meal_times(lB)
n_m = min(len(mA), len(mB))
meal_lag = mA[:n_m] - mB[:n_m]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0][0]
for (name, lg), col in zip(logs.items(), ["#8e44ad", "#2c3e50"]):
    pos = lg["pos"]
    ax.plot(pos[~rel, 0], pos[~rel, 1], color=col, lw=1.2, label=f"loops {name}")
    ax.plot(pos[rel, 0], pos[rel, 1], color=col, lw=0.6, ls="--", alpha=0.7, label=f"after release {name[0]}")
ax.add_patch(plt.Circle(c, R_disc, color="#8e44ad", alpha=0.18))
ax.plot(*P, "ko")
ax.annotate("P", P + np.array([1.5, -3]))
ax.set_aspect("equal")
ax.set_xlim(0, 100)
ax.set_ylim(-20, 100)
ax.legend(fontsize=7)
ax.set_title("paths (shaded: haunted disc)")
ax = axes[0][1]
ax.plot(t, lA["C"][:, 1], color=DRIVE_COLORS[1], label="social, A")
ax.plot(t, lB["C"][:, 1], color=DRIVE_COLORS[1], ls=":", label="social, B")
ax.plot(t, lA["C"][:, 2], color=DRIVE_COLORS[2], label="explore, A")
ax.plot(t, lB["C"][:, 2], color=DRIVE_COLORS[2], ls=":", label="explore, B")
ax.axvline(T_LOOPS, color="k", lw=0.7)
ax.annotate("release", (T_LOOPS + 1, ax.get_ylim()[1] * 0.9))
ax.legend(fontsize=7)
ax.set_title("e2/e3 coefficients (rotated by the connection in A)")
ax = axes[1][0]
ax.semilogy(t, dC + 1e-9, color="#c0392b", label="|C_A − C_B|")
ax.axvline(T_LOOPS, color="k", lw=0.7)
ax.set_xlabel("t (s)")
ax.legend()
ax.set_title("state difference")
ax = axes[1][1]
ax.plot(t, dpos, color="#2c3e50")
ax.axvline(T_LOOPS, color="k", lw=0.7)
ax.set_xlabel("t (s)")
ax.set_title("position difference after release (behaviour)")
save(fig, OUT, "holonomy.png")

lines = ["E4 Holonomy", "===========",
         f"Haunted disc: centre {tuple(map(float, c))}, R={R_disc}, flux Phi={Phi} of B_h=e{cfg.cog.holonomy_plane[0]+1}{cfg.cog.holonomy_plane[1]+1} "
         f"(elliptic plane, so holonomy is a rotation).",
         f"Loops: radius {R_loop}, length {2*np.pi*R_loop:.1f}, both through P={tuple(map(float, P))}.",
         "",
         "Part 1, pure transport (one loop, C = 0.5 e2):"]
for name, (alpha, angle, C1) in part1.items():
    lines.append(f"  {name:18s} connection integral = {alpha:+.6f} rad -> e2/e3 rotation {angle:+.2f} deg; C -> {np.round(C1, 4)}")
lines += [f"  expected for an enclosing loop: Phi*pi*R^2 = {expected:.6f} rad ({np.degrees(expected):.2f} deg); outside: 0",
          "",
          "Part 2, full cognition (3 loops with coupling off, then free with coupling on):",
          f"  |C_A - C_B| at release: {dC_rel:.4f}  (state norm ~{np.linalg.norm(lA['C'][i_rel]):.2f})",
          f"  max |C_A - C_B| during loops: {dC[~rel].max():.4f}",
          f"  decay of the state difference after release: 1/e in {tau:.1f} s",
          f"  |C_A - C_B| at the end (t={t[-1]:.0f}s): {dC[-1]:.2e}",
          f"  position difference at the end: {dpos[-1]:.2f}  (max after release {dpos[rel].max():.2f})",
          f"  meal times A: {np.round(mA[:n_m], 1).tolist()}",
          f"  meal times B: {np.round(mB[:n_m], 1).tolist()}",
          f"  lag A - B   : {np.round(meal_lag, 2).tolist()} s",
          "",
          "Reading:",
          "  * Holonomy is exactly topological: the result depends only on whether the loop encloses the",
          "    disc, not on its shape or length (unit tests: outside loop = identity to 1e-12, enclosing",
          "    loop = total flux, small loop inside = flux * area).",
          "  * While the NPC circles the disc, the connection acts as a slow steady rotation in e23 (in this",
          "    gauge it accrues uniformly with the winding angle), balanced against damping.  So the state",
          "    offset stays small (~0.03) and does not grow with the number of loops.",
          "  * After release the amplitude part of the difference decays within seconds (damping), but",
          "    the NPC is on the foraging limit cycle (E1), and a limit cycle's phase is neutrally stable:",
          "    a perturbation that shifts the phase is never corrected.  The haunted loops leave a",
          "    lasting shift in when the NPC eats (meal lags above): the memory of place persists in",
          "    behaviour timing, not in the drive values.",
          "  * For memory of place in the drive values themselves, write holonomy into the slow variables:",
          "    the place-rotor slots of tier-1 memory (Phase 3) or a metric update driven by holonomy.",
          ]
write_summary(OUT, lines)
