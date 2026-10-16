"""Live viewer for the single-NPC simulation (matplotlib, no extra dependencies).

    python -m cliffnpc.viewer                              # interactive window
    python -m cliffnpc.viewer --save run.gif --seconds 60  # headless render (.gif/.mp4)

Panels: the world, drive readouts (coefficients = what it does, projections =
what it feels), the light cone in the NPC's orthonormal frame, the learned
metric g - eta, and a strip chart of the normalized interval and frustration.

Keys (interactive):
    space  pause / resume          t  threat attacks now
    h      haunted disc on / off   r  reset
    + / -  faster / slower         left click   move the food patch
    q      quit                    right click  move home

The interactive window uses blitting: static parts (axes, ticks, food, home)
are drawn once and only the moving artists are redrawn each frame.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

REGIME_COLORS = {"deliberative": "#2e86c1", "reactive": "#c0392b", "null": "#7f8c8d"}
DRIVES = ["hunger", "social", "explore", "threat"]
POLES = [("hungry", "sated"), ("lonely", "withdrawn"), ("explore", "home"), ("fight", "flight")]
TRAIL = 400            # physics steps of world trail
STRIP_SECONDS = 60.0   # strip chart window
CONE_TRAIL = 100       # cognitive ticks in the cone trail
TEXT_EVERY = 10        # re-render the text panels every n frames (text is slow to blit)
BAR_LIM = 2.5
# Discrete scales with hysteresis: rescaling forces a full (slow) redraw, so it
# should happen rarely.
METRIC_LEVELS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
CONE_LEVELS = (1.5, 3.0, 6.0, 12.0)


def pick_level(levels, value, current):
    """Smallest level >= value; shrink only when value < 60% of the next lower level."""
    want = next((lv for lv in levels if lv >= value), levels[-1])
    if want >= current:
        return want
    lower = max((lv for lv in levels if lv < current), default=current)
    return want if value < 0.6 * lower else current
FR_LIM = 2.0


class Viewer:
    def __init__(self, plt, cfg=None, steps_per_frame: int = 4):
        from matplotlib.patches import Circle

        from .config import Config
        from .sim import Simulation

        self.plt = plt
        self._Simulation = Simulation
        self._Circle = Circle
        self.cfg = cfg or Config()
        self.steps_per_frame = steps_per_frame
        self.paused = False
        self.haunt_flux = self.cfg.world.haunt_flux
        self.animated: list = []
        self.texts: list = []          # animated texts, refreshed every TEXT_EVERY frames
        self._text_cache: list = []
        self._frame_no = 0
        self._bg = None
        self.dirty = True
        self._build_figure()
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        self.sim = self._Simulation(self.cfg)
        self.sim.world.haunt.flux = self.haunt_flux
        self.trail: list[np.ndarray] = []
        self._place_static()

    def _anim(self, artist):
        artist.set_animated(True)
        self.animated.append(artist)
        return artist

    def _text(self, artist):
        artist.set_animated(True)
        self.texts.append(artist)
        return artist

    def _build_figure(self) -> None:
        plt = self.plt
        self.fig = fig = plt.figure(figsize=(14, 8), layout="constrained")
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title("cliffnpc live viewer")
        gs = fig.add_gridspec(3, 3, width_ratios=[1.3, 1, 1], height_ratios=[1, 1, 0.6])
        self.ax_world = fig.add_subplot(gs[0:2, 0])
        self.ax_bars = fig.add_subplot(gs[0, 1])
        self.ax_cone = fig.add_subplot(gs[0, 2])
        self.ax_metric = fig.add_subplot(gs[1, 1])
        self.ax_info = fig.add_subplot(gs[1, 2])
        self.ax_strip = fig.add_subplot(gs[2, :])
        A = self._anim

        # world
        ax = self.ax_world
        size = self.cfg.world.size
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("world")
        self.food_art, = ax.plot([], [], "o", color="#e67e22", ms=12, label="food")
        self.home_art, = ax.plot([], [], "s", color="#2c3e50", ms=11, label="home")
        self.peer_art, = ax.plot([], [], "^", color="#2e86c1", ms=9, label="peers")
        self.haunt_art = self._Circle((0, 0), 1, color="#8e44ad", alpha=0.18, label="haunted disc")
        ax.add_patch(self.haunt_art)
        self.sensor_art = A(self._Circle((0, 0), self.cfg.world.sensor_radius, fill=False,
                                         ls=":", color="#95a5a6"))
        ax.add_patch(self.sensor_art)
        self.trail_art = A(ax.plot([], [], "-", color="#bbb", lw=0.8)[0])
        self.threat_art = A(ax.plot([], [], "X", color="#8e44ad", ms=13, label="threat")[0])
        self.npc_art = A(ax.plot([], [], "o", ms=11, mec="k", label="NPC")[0])
        ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
        self.status = self._text(ax.text(0.99, 0.99, "", transform=ax.transAxes, ha="right", va="top",
                                fontsize=8, family="monospace",
                                bbox=dict(boxstyle="round", fc="white", alpha=0.85)))

        # drive bars: pole names printed inside the axes, top (+) and bottom (-)
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
        ax.set_title("drives")

        # light cone (orthonormal frame): x = |spacelike part|, y = timelike part
        ax = self.ax_cone
        self.cone_fill = []
        self.cone_trail = A(ax.plot([], [], "-", color="#999", lw=0.8)[0])
        self.cone_pt = A(ax.plot([], [], "o", ms=9, mec="k")[0])
        ax.set_xlabel("|hunger, social, explore|", fontsize=8)
        ax.set_ylabel("threat (timelike)", fontsize=8)
        ax.set_title("light cone   s = x² − y²")
        ax.text(0.97, 0.03, "deliberative", transform=ax.transAxes, ha="right", fontsize=7, color="#2e86c1")
        ax.text(0.03, 0.97, "reactive", transform=ax.transAxes, va="top", fontsize=7, color="#c0392b")
        self._draw_cone(1.5)

        # metric
        ax = self.ax_metric
        self.metric_img = A(ax.imshow(np.zeros((4, 4)), cmap="RdBu_r", vmin=-0.1, vmax=0.1))
        ax.set_xticks(range(4), ["e1", "e2", "e3", "e4"])
        ax.set_yticks(range(4), ["hunger", "social", "explore", "threat"], fontsize=8)
        ax.set_title("learned personality  g − η")
        self.metric_cbar = fig.colorbar(self.metric_img, ax=ax, shrink=0.8)

        # keys + info
        ax = self.ax_info
        ax.axis("off")
        ax.text(0.0, 1.0, "keys\n"
                "  space  pause      t  threat now\n"
                "  h  haunt on/off   r  reset\n"
                "  + / -  speed      q  quit\n"
                "  left click   move food\n"
                "  right click  move home",
                va="top", fontsize=8, family="monospace", transform=ax.transAxes)
        # one text block for all numbers: text is by far the most expensive
        # artist to blit, so fewer, longer texts are much faster than many
        self.info = self._text(ax.text(0.0, 0.58, "", va="top", fontsize=8, family="monospace",
                              transform=ax.transAxes))

        # strip chart, x = seconds before now (fixed limits, so it can be blitted)
        ax = self.ax_strip
        self.strip_cone = A(ax.plot([], [], color="#2c3e50", label="s / |C|²")[0])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlim(-STRIP_SECONDS, 0)
        ax.set_ylim(-1.6, 1.6)
        ax.set_ylabel("normalized interval")
        ax.set_xlabel("seconds before now")
        self.ax_fr = ax.twinx()
        self.strip_fr = A(self.ax_fr.plot([], [], color="#e67e22", lw=0.9,
                                          label="frustration |C∧B|")[0])
        self.ax_fr.set_ylim(0, FR_LIM)
        self.ax_fr.set_ylabel("frustration", color="#e67e22")
        self.strip_reactive = None
        ax.legend(handles=[self.strip_cone, self.strip_fr], loc="upper left", fontsize=7)

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
            ax.fill_between(xs, -lim, -xs, color="#c0392b", alpha=0.10, lw=0),
            ax.fill_between(xs, -xs, xs, color="#2e86c1", alpha=0.07, lw=0),
        ]
        ax.set_xlim(0, lim)
        ax.set_ylim(-lim, lim)
        self.cone_lim = lim
        self.dirty = True

    def _place_static(self) -> None:
        w = self.sim.world
        self.food_art.set_data(*np.array(w.food).T if w.food else ([], []))
        self.home_art.set_data([w.home[0]], [w.home[1]])
        self.peer_art.set_data(*w.peers.T if len(w.peers) else ([], []))
        self.haunt_art.center = tuple(w.haunt.center)
        self.haunt_art.set_radius(w.haunt.radius)
        self.haunt_art.set_alpha(0.18 if w.haunt.flux else 0.04)
        self.dirty = True

    # ------------------------------------------------------------ interaction
    def _on_key(self, ev) -> None:
        w = self.sim.world
        if ev.key == " ":
            self.paused = not self.paused
        elif ev.key == "t":
            w.threat.force_attack(w.t)
        elif ev.key == "h":
            self.haunt_flux = 0.0 if w.haunt.flux else self.cfg.world.haunt_flux
            w.haunt.flux = self.haunt_flux
            self._place_static()
        elif ev.key == "r":
            self.reset()
        elif ev.key in ("+", "="):
            self.steps_per_frame = min(self.steps_per_frame * 2, 64)
        elif ev.key == "-":
            self.steps_per_frame = max(self.steps_per_frame // 2, 1)
        elif ev.key == "q":
            self.plt.close(self.fig)

    def _on_click(self, ev) -> None:
        if ev.inaxes is not self.ax_world or ev.xdata is None:
            return
        p = np.array([ev.xdata, ev.ydata])
        if ev.button == 1:
            self.sim.world.food = [p]
        elif ev.button == 3:
            self.sim.world.home = p
        self._place_static()

    # --------------------------------------------------------------- blitting
    def _on_draw(self, _ev) -> None:
        """After any full draw (startup, resize, layout change): recapture the
        static background and paint the animated artists on top."""
        c = self.fig.canvas
        self._bg = c.copy_from_bbox(self.fig.bbox)
        self._draw_animated(refresh_text=True)

    def _text_extent(self, t):
        r = self.fig.canvas.get_renderer()
        patch = t.get_bbox_patch()
        bb = patch.get_window_extent(r) if patch is not None else t.get_window_extent(r)
        return bb.expanded(1.0, 1.0).padded(3)

    def _draw_animated(self, refresh_text: bool | None) -> None:
        """refresh_text: True = all texts, False = none, None = staggered
        (text i is re-rendered when frame_no + i*k hits a multiple of TEXT_EVERY,
        so the expensive renders fall on different frames)."""
        for a in self.animated:
            self.fig.draw_artist(a)
        c = self.fig.canvas
        if len(self._text_cache) != len(self.texts):
            self._text_cache = [None] * len(self.texts)
            refresh_text = True
        step = max(1, TEXT_EVERY // max(len(self.texts), 1))
        for i, t in enumerate(self.texts):
            due = refresh_text if refresh_text is not None else \
                (self._frame_no + i * step) % TEXT_EVERY == 0
            if due or self._text_cache[i] is None:
                self.fig.draw_artist(t)
                # cache the rendered pixels; paste them on the frames in between
                self._text_cache[i] = c.copy_from_bbox(self._text_extent(t))
            else:
                c.restore_region(self._text_cache[i])

    def blit(self) -> None:
        c = self.fig.canvas
        self._frame_no += 1
        if self.dirty or self._bg is None:
            self.dirty = False
            c.draw()                      # triggers _on_draw
        else:
            c.restore_region(self._bg)
            self._draw_animated(refresh_text=None)
        c.blit(self.fig.bbox)
        c.flush_events()

    # ---------------------------------------------------------------- frames
    def advance(self) -> None:
        if self.paused:
            return
        for _ in range(self.steps_per_frame):
            self.sim.step()
            self.trail.append(self.sim.agent.pos.copy())
        del self.trail[:-TRAIL]

    def update(self) -> None:
        from .algebra import ETA, signature

        sim, w = self.sim, self.sim.world
        npc = sim.agent.npc
        pos = sim.agent.pos
        regime = npc.regime()
        col = REGIME_COLORS[regime]

        # world
        tr = np.array(self.trail) if self.trail else pos[None]
        self.trail_art.set_data(tr[:, 0], tr[:, 1])
        self.npc_art.set_data([pos[0]], [pos[1]])
        self.npc_art.set_markerfacecolor(col)
        self.sensor_art.center = tuple(pos)
        attacking = w.threat.attacking(w.t)
        if w.threat.active:
            self.threat_art.set_data([w.threat.pos[0]], [w.threat.pos[1]])
            self.threat_art.set_markersize(16 if attacking else 11)
            self.threat_art.set_alpha(1.0 if attacking else 0.5)

        # bars (values beyond the axis are clipped; the status box shows exact numbers)
        coef, proj = npc.coefficients(), npc.projections()
        for b, v in zip(self.bar_coef, coef):
            b.set_height(np.clip(v, -BAR_LIM, BAR_LIM))
        for b, v in zip(self.bar_proj, proj):
            b.set_height(np.clip(v, -BAR_LIM, BAR_LIM))

        # cone, using the current frame for the whole trail (one consistent picture)
        log = sim.log.rows
        Cs = np.array(log["C"][-CONE_TRAIL:]) if "C" in log else npc.C[None]
        Co = (npc.L @ Cs.T).T
        xs, ys = np.linalg.norm(Co[:, :3], axis=1), Co[:, 3]
        self.cone_trail.set_data(xs, ys)
        self.cone_pt.set_data([xs[-1]], [ys[-1]])
        self.cone_pt.set_markerfacecolor(col)
        lim = pick_level(CONE_LEVELS, 1.1 * max(xs.max(), np.abs(ys).max()), self.cone_lim)
        if lim != self.cone_lim:
            self._draw_cone(lim)

        # metric
        g = npc.g
        d = g - ETA
        self.metric_img.set_data(d)
        cur = self.metric_img.get_clim()[1]
        m = pick_level(METRIC_LEVELS, np.abs(d).max(), cur)
        if m != cur:
            self.metric_img.set_clim(-m, m)
            self.dirty = True              # colorbar ticks are static: redraw them

        # strip chart
        if "t" in log:
            t = np.array(log["t"])
            k = np.searchsorted(t, t[-1] - STRIP_SECONDS)
            ago = t[k:] - t[-1]
            cone = np.array(log["cone"][k:])
            fr = np.array(log["frustration"][k:])
            self.strip_cone.set_data(ago, cone)
            self.strip_fr.set_data(ago, np.minimum(fr, FR_LIM))
            if self.strip_reactive is not None:
                self.animated.remove(self.strip_reactive)
                self.strip_reactive.remove()
            self.strip_reactive = self._anim(self.ax_strip.fill_between(
                ago, -1.6, 1.6, where=cone < 0, color="#c0392b", alpha=0.12, lw=0))

        last = npc.last
        self.status.set_text(
            f"t = {w.t:7.1f} s   x{self.steps_per_frame}{'  PAUSED' if self.paused else ''}\n"
            f"regime {regime:>12s}\n"
            f"s = {npc.interval():+.3f}   s/|C|² = {npc.cone():+.2f}\n"
            f"C = [{', '.join(f'{c:+.2f}' for c in coef)}]\n"
            f"frustration {last.get('frustration', 0.0):.3f}")
        rows = "\n".join("  " + " ".join(f"{x:+.2f}" for x in row) for row in d)
        self.info.set_text(
            f"g − η\n{rows}\n\n"
            f"det L      {np.linalg.det(npc.L):.3f}\n"
            f"signature  {signature(g)[:2]}\n"
            f"threat     {'ATTACKING' if attacking else 'idle'}\n"
            f"haunt flux {w.haunt.flux:g}\n"
            f"in haunt   {w.haunt.contains(pos)}\n"
            f"learning   {'on' if last.get('learned', False) else 'gated'}")

    def frame(self, _i=None):
        self.advance()
        self.update()
        return self.animated + self.texts


def run_interactive(v: Viewer, fps: int) -> None:
    plt = v.plt
    timer = v.fig.canvas.new_timer(interval=max(1, 1000 // fps))

    def tick():
        if not plt.fignum_exists(v.fig.number):
            timer.stop()
            return
        v.frame()
        v.blit()

    timer.add_callback(tick)
    v._timer = timer  # keep a reference
    timer.start()
    plt.show()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", help="render headless to this file (.gif or .mp4) instead of a window")
    ap.add_argument("--seconds", type=float, default=60.0, help="simulated seconds to render with --save")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--speed", type=int, default=4, help="physics steps per frame")
    ap.add_argument("--threat-at", type=float, action="append", default=[],
                    help="force a threat attack at this sim time (repeatable)")
    args = ap.parse_args(argv)

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = Viewer(plt, steps_per_frame=args.speed)
    for ta in args.threat_at:
        v.sim.world.threat.force_attack(ta)
    if args.save:
        from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
        n_frames = int(args.seconds / (v.cfg.world.dt * args.speed))
        anim = FuncAnimation(v.fig, v.frame, frames=n_frames, blit=False)
        writer = PillowWriter(fps=args.fps) if args.save.endswith(".gif") else FFMpegWriter(fps=args.fps)
        anim.save(args.save, writer=writer, dpi=80)
        print(f"wrote {args.save} ({n_frames} frames, {args.seconds:.0f} simulated s)")
        return 0
    run_interactive(v, args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
