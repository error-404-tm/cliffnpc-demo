"""Live viewer for the arena (life on): many agents, birth and death, fear, memory.

    python -m cliffnpc.arena_viewer                    # Cl(3,1)
    python -m cliffnpc.arena_viewer --signature 22     # Cl(2,2): hunger timelike too
    python -m cliffnpc.arena_viewer --save arena.gif --seconds 120

World: tribe A (blue circles) and tribe B (orange squares), faded by injury, edged by regime
(blue deliberative, red reactive, grey null).  Recent deaths are black x's,
recent births green +'s.  The selected agent has a ring, its sensor circle and
its trail; the side panels show that agent's drives, light cone, learned metric
and state.  The strip chart shows population.

Keys: click an agent (or tab) to follow it, space pause, + / - speed,
r reset, q quit.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .viewer import (BAR_LIM, CONE_LEVELS, DRIVES, METRIC_LEVELS, POLES, REGIME_COLORS, TRAIL,
                     Viewer, pick_level, run_interactive)

FACTION_COLORS = {"npc": "#2e86c1", "rival": "#c0392b", "A": "#2e86c1", "B": "#d35400"}
FACTION_MARKERS = {"npc": "o", "rival": "^", "A": "o", "B": "s"}
MARK_SECONDS = 60.0       # how long death/birth markers stay
POP_SECONDS = 300.0       # population strip window
SIGNATURES = {"31": (1.0, 1.0, 1.0, -1.0), "22": (-1.0, 1.0, 1.0, -1.0)}


class ArenaViewer(Viewer):
    def __init__(self, plt, signature=SIGNATURES["31"], seed: int = 0, steps_per_frame: int = 2):
        from matplotlib.patches import Circle

        self.plt = plt
        self._Circle = Circle
        self.signature = tuple(signature)
        self.seed = seed
        self.steps_per_frame = steps_per_frame
        self.paused = False
        self.animated: list = []
        self.texts: list = []
        self._text_cache: list = []
        self._frame_no = 0
        self._bg = None
        self.dirty = True
        self._build_figure()
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        from .arena import Arena, life_config
        self.arena = Arena(life_config(self.signature, self.seed))
        self.selected = self.arena.agents[0].name
        self.trail: list = []
        self.cone_hist: list = []
        self.pop_hist: list = []
        self._place_static()

    def _sig_label(self) -> str:
        p = sum(x > 0 for x in self.signature)
        return f"Cl({p},{4 - p})"

    def _build_figure(self) -> None:
        plt = self.plt
        self.fig = fig = plt.figure(figsize=(15, 8.5), layout="constrained")
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title("cliffnpc arena viewer")
        gs = fig.add_gridspec(3, 3, width_ratios=[1.35, 1, 1], height_ratios=[1, 1, 0.55])
        self.ax_world = fig.add_subplot(gs[0:2, 0])
        self.ax_bars = fig.add_subplot(gs[0, 1])
        self.ax_cone = fig.add_subplot(gs[0, 2])
        self.ax_metric = fig.add_subplot(gs[1, 1])
        self.ax_info = fig.add_subplot(gs[1, 2])
        self.ax_strip = fig.add_subplot(gs[2, :])
        A = self._anim

        ax = self.ax_world
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("arena")
        self.food_sc = A(ax.scatter([], [], s=200, marker="o", color="#e67e22", alpha=0.85,
                                    label="food (size = stock)"))
        self.home_art, = ax.plot([], [], "P", color=FACTION_COLORS["A"], ms=13, mec="k", label="home A")
        self.lair_art, = ax.plot([], [], "P", color=FACTION_COLORS["B"], ms=13, mec="k", label="home B")
        self.sensor_art = A(self._Circle((0, 0), 30.0, fill=False, ls=":", color="#95a5a6"))
        ax.add_patch(self.sensor_art)
        self.trail_art = A(ax.plot([], [], "-", color="#bbb", lw=0.8)[0])
        self.npc_sc = A(ax.scatter([], [], s=90, marker=FACTION_MARKERS["A"], linewidths=2,
                                   label="tribe A", color=FACTION_COLORS["A"]))
        self.rival_sc = A(ax.scatter([], [], s=90, marker=FACTION_MARKERS["B"], linewidths=2,
                                     label="tribe B", color=FACTION_COLORS["B"]))
        self.sel_art = A(ax.plot([], [], "o", ms=20, mfc="none", mec="#f1c40f", mew=2.5)[0])
        self.dead_sc = A(ax.scatter([], [], s=80, marker="x", color="k", label="deaths (60 s)"))
        self.born_sc = A(ax.scatter([], [], s=90, marker="+", color="#27ae60", label="births (60 s)"))
        ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
        self.status = self._text(ax.text(0.99, 0.99, "", transform=ax.transAxes, ha="right",
                                         va="top", fontsize=8, family="monospace",
                                         bbox=dict(boxstyle="round", fc="white", alpha=0.85)))

        ax = self.ax_bars
        x = np.arange(4)
        self.bar_coef = [A(b) for b in ax.bar(x - 0.2, np.zeros(4), 0.4, color="#2c3e50",
                                               label="coefficient (does)")]
        self.bar_proj = [A(b) for b in ax.bar(x + 0.2, np.zeros(4), 0.4, color="#95a5a6",
                                               label="projection (feels)")]
        ax.set_xticks(x, DRIVES, fontsize=8)
        for i, (p, m) in enumerate(POLES):
            ax.text(i, BAR_LIM * 0.92, f"+{p}", ha="center", va="top", fontsize=6.5, color="#777")
            ax.text(i, -BAR_LIM * 0.92, f"−{m}", ha="center", va="bottom", fontsize=6.5, color="#777")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-BAR_LIM, BAR_LIM)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        self.bars_title = ax.set_title("drives")

        ax = self.ax_cone
        self.cone_fill = []
        self.cone_trail = A(ax.plot([], [], "-", color="#999", lw=0.8)[0])
        self.cone_pt = A(ax.plot([], [], "o", ms=9, mec="k")[0])
        ax.set_xlabel("|spacelike drives|  (own frame)", fontsize=8)
        ax.set_ylabel("|timelike drives|", fontsize=8)
        ax.set_title("light cone   s = x² − y²")
        self._draw_cone(1.5)

        ax = self.ax_metric
        self.metric_img = A(ax.imshow(np.zeros((4, 4)), cmap="RdBu_r", vmin=-0.1, vmax=0.1))
        ax.set_xticks(range(4), ["e1", "e2", "e3", "e4"])
        ax.set_yticks(range(4), DRIVES, fontsize=8)
        ax.set_title("learned personality  g − η")
        fig.colorbar(self.metric_img, ax=ax, shrink=0.8)

        ax = self.ax_info
        ax.axis("off")
        ax.text(0.0, 1.0, "keys\n  click / tab  select agent\n  space pause   + / - speed\n"
                "  r reset       q quit", va="top", fontsize=8, family="monospace",
                transform=ax.transAxes)
        self.info = self._text(ax.text(0.0, 0.74, "", va="top", fontsize=8, family="monospace",
                                       transform=ax.transAxes))

        ax = self.ax_strip
        self.pop_npc = A(ax.plot([], [], color=FACTION_COLORS["A"], label="tribe A")[0])
        self.pop_rival = A(ax.plot([], [], color=FACTION_COLORS["B"], label="tribe B")[0])
        ax.set_xlim(-POP_SECONDS, 0)
        ax.set_ylim(0, 21)
        ax.set_ylabel("population")
        ax.set_xlabel("seconds before now")
        ax.legend(loc="upper left", fontsize=7)

        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("draw_event", self._on_draw)

    def _draw_cone(self, lim: float) -> None:
        ax = self.ax_cone
        for art in self.cone_fill:
            art.remove()
        xs = np.linspace(0, lim, 2)
        self.cone_fill = [
            ax.fill_between(xs, xs, lim, color="#c0392b", alpha=0.10, lw=0),
            ax.fill_between(xs, 0, xs, color="#2e86c1", alpha=0.07, lw=0),
        ]
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        self.cone_lim = lim
        self.dirty = True

    def _place_static(self) -> None:
        cfg = self.arena.cfg
        self.home_art.set_data([cfg.home[0]], [cfg.home[1]])
        self.lair_art.set_data([cfg.lair[0]], [cfg.lair[1]])
        self.dirty = True

    # ------------------------------------------------------------ interaction
    def _agent(self):
        for a in self.arena.agents:
            if a.name == self.selected:
                return a
        return None

    def _select(self, name: str) -> None:
        if name != self.selected:
            self.selected = name
            self.trail, self.cone_hist = [], []

    def _on_key(self, ev) -> None:
        if ev.key == " ":
            self.paused = not self.paused
        elif ev.key == "tab":
            names = [a.name for a in self.arena.agents]
            if names:
                i = names.index(self.selected) if self.selected in names else -1
                self._select(names[(i + 1) % len(names)])
        elif ev.key == "r":
            self.reset()
        elif ev.key in ("+", "="):
            self.steps_per_frame = min(self.steps_per_frame * 2, 64)
        elif ev.key == "-":
            self.steps_per_frame = max(self.steps_per_frame // 2, 1)
        elif ev.key == "q":
            self.plt.close(self.fig)

    def _on_click(self, ev) -> None:
        if ev.inaxes is not self.ax_world or ev.xdata is None or not self.arena.agents:
            return
        p = np.array([ev.xdata, ev.ydata])
        best = min(self.arena.agents, key=lambda a: np.linalg.norm(a.pos - p))
        if np.linalg.norm(best.pos - p) < 6.0:
            self._select(best.name)

    # ---------------------------------------------------------------- frames
    def advance(self) -> None:
        if self.paused:
            return
        ar = self.arena
        for _ in range(self.steps_per_frame):
            ar.step()
            a = self._agent()
            if a is None and ar.agents:                # selected agent died: follow another
                self._select(ar.agents[0].name)
                a = self._agent()
            if a is not None:
                self.trail.append(a.pos.copy())
        del self.trail[:-TRAIL]
        n_npc = sum(a.faction == ar.factions[0] for a in ar.agents)
        self.pop_hist.append((ar.t, n_npc, len(ar.agents) - n_npc))

    def _scatter(self, sc, agents) -> None:
        import matplotlib.colors as mcolors
        if not agents:
            sc.set_offsets(np.zeros((0, 2)))
            return
        sc.set_offsets(np.array([a.pos for a in agents]))
        base = np.array(mcolors.to_rgba(FACTION_COLORS[agents[0].faction]))
        face = np.tile(base, (len(agents), 1))
        face[:, 3] = [0.2 + 0.8 * max(a.health, 0.0) for a in agents]
        sc.set_facecolors(face)
        sc.set_edgecolors([REGIME_COLORS[a.npc.regime()] for a in agents])

    def update(self) -> None:
        from . import algebra as ga
        ar = self.arena
        npcs = [a for a in ar.agents if a.faction == ar.factions[0]]
        rivals = [a for a in ar.agents if a.faction == ar.factions[1]]
        P = ar.patches
        self.food_sc.set_offsets(np.array([p["pos"] for p in P]).reshape(-1, 2))
        self.food_sc.set_sizes([40 + 260 * min(p["stock"] / p["stock0"], 1.0) if np.isfinite(p["stock0"])
                                else 200 for p in P])
        self._scatter(self.npc_sc, npcs)
        self._scatter(self.rival_sc, rivals)
        recent_d = [d for d in ar.deaths if ar.t - d["t"] < MARK_SECONDS]
        self.dead_sc.set_offsets(np.array([d["pos"] for d in recent_d]).reshape(-1, 2))
        born = {b["name"] for b in ar.births if ar.t - b["t"] < MARK_SECONDS}
        self.born_sc.set_offsets(np.array([a.pos for a in ar.agents if a.name in born]).reshape(-1, 2))

        a = self._agent()
        if a is not None:
            npc = a.npc
            col = REGIME_COLORS[npc.regime()]
            self.sel_art.set_data([a.pos[0]], [a.pos[1]])
            self.sensor_art.center = tuple(a.pos)
            tr = np.array(self.trail) if self.trail else a.pos[None]
            self.trail_art.set_data(tr[:, 0], tr[:, 1])
            coef, proj = npc.coefficients(), npc.projections()
            for b, v in zip(self.bar_coef, coef):
                b.set_height(np.clip(v, -BAR_LIM, BAR_LIM))
            for b, v in zip(self.bar_proj, proj):
                b.set_height(np.clip(v, -BAR_LIM, BAR_LIM))
            # cone in the agent's own frame: spacelike vs timelike parts
            self.cone_hist.append(npc.C.copy())
            del self.cone_hist[:-100]
            Co = (npc.L @ np.array(self.cone_hist).T).T
            timelike = np.array(npc.cfg.signature) < 0
            xs = np.linalg.norm(Co[:, ~timelike], axis=1)
            ys = np.linalg.norm(Co[:, timelike], axis=1)
            self.cone_trail.set_data(xs, ys)
            self.cone_pt.set_data([xs[-1]], [ys[-1]])
            self.cone_pt.set_markerfacecolor(col)
            lim = pick_level(CONE_LEVELS, 1.1 * max(xs.max(), ys.max()), self.cone_lim)
            if lim != self.cone_lim:
                self._draw_cone(lim)
            d = npc.g - npc.eta
            self.metric_img.set_data(d)
            cur = self.metric_img.get_clim()[1]
            m = pick_level(METRIC_LEVELS, np.abs(d).max(), cur)
            if m != cur:
                self.metric_img.set_clim(-m, m)
                self.dirty = True
            mem = ""
            if a.ltm is not None:
                s = a.ltm.slots
                act = s["active"] == 1
                from .ltm import ORIGINS
                counts = " ".join(f"{o[:3]} {int((act & (s['origin'] == k)).sum())}"
                                  for k, o in enumerate(ORIGINS))
                rep = "—"
                if a.ltm.replay_slot >= 0:
                    j = a.ltm.replay_slot
                    rep = (f"{'escape' if s['kind'][j] < 0 else 'reward'} "
                           f"({ORIGINS[int(s['origin'][j])]})")
                mem = f"memories   {counts}\nreplaying  {rep}\n"
            self.info.set_text(
                f"{a.name}  ({a.faction}, gen {a.generation})\n"
                f"parents    {', '.join(a.parents) or '—'}\n"
                f"age        {ar.t - a.born:6.0f} s\n"
                f"health     {a.health:6.2f}\n"
                f"hunger     {npc.C[0]:+6.2f}\n"
                f"threat     {npc.C[3]:+6.2f}  ({'fight' if npc.C[3] > 0 else 'flight'})\n"
                f"fear sens. {a.affect.fear_learned:6.2f}\n"
                f"pain       {a.affect.pain:+6.2f}\n"
                f"regime     {npc.regime()}\n" + mem +
                f"signature  {ga.signature(npc.g)[:2]}")
        # population strip
        if self.pop_hist:
            P = np.array(self.pop_hist)
            k = np.searchsorted(P[:, 0], P[-1, 0] - POP_SECONDS)
            ago = P[k:, 0] - P[-1, 0]
            self.pop_npc.set_data(ago, P[k:, 1])
            self.pop_rival.set_data(ago, P[k:, 2])
        causes = {c: sum(d["cause"] == c for d in ar.deaths) for c in ("starvation", "injury")}
        self.status.set_text(
            f"{self._sig_label()}   t = {ar.t:7.1f} s   x{self.steps_per_frame}"
            f"{'  PAUSED' if self.paused else ''}\n"
            f"{ar.factions[0]} {len(npcs):2d}   {ar.factions[1]} {len(rivals):2d}   births {len(ar.births)}\n"
            f"deaths: starvation {causes['starvation']}  injury {causes['injury']}\n"
            f"fights {len(ar.engagements)}   food patches {len(ar.patches)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signature", choices=sorted(SIGNATURES), default="31",
                    help="31 = Cl(3,1) (threat timelike), 22 = Cl(2,2) (hunger timelike too)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", help="render headless to this file (.gif or .mp4)")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--speed", type=int, default=2, help="physics steps per frame")
    args = ap.parse_args(argv)

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = ArenaViewer(plt, SIGNATURES[args.signature], args.seed, args.speed)
    if args.save:
        from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
        n_frames = int(args.seconds / (v.arena.wcfg.dt * args.speed))
        anim = FuncAnimation(v.fig, v.frame, frames=n_frames, blit=False)
        writer = PillowWriter(fps=args.fps) if args.save.endswith(".gif") else FFMpegWriter(fps=args.fps)
        anim.save(args.save, writer=writer, dpi=80)
        print(f"wrote {args.save} ({n_frames} frames)")
        return 0
    run_interactive(v, args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
