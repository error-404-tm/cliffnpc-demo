"""Smoke test for the live viewer (headless Agg backend)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cliffnpc.viewer import CONE_LEVELS, Viewer, pick_level  # noqa: E402


def test_viewer_frames_and_controls():
    v = Viewer(plt, steps_per_frame=8)
    for _ in range(30):
        v.frame()
        v.blit()
    t0 = v.sim.world.t
    assert t0 > 0
    # keyboard controls
    class Ev:
        def __init__(self, key=None, button=None, x=None, y=None, ax=None):
            self.key, self.button, self.xdata, self.ydata, self.inaxes = key, button, x, y, ax
    v._on_key(Ev("t"))
    assert v.sim.world.threat.attacking(v.sim.world.t)
    v._on_key(Ev(" "))
    v.frame()
    assert v.sim.world.t == t0           # paused
    v._on_key(Ev(" "))
    v._on_key(Ev("h"))
    assert v.sim.world.haunt.flux == 0.0
    v._on_click(Ev(button=1, x=10.0, y=90.0, ax=v.ax_world))
    assert np.allclose(v.sim.world.food[0], [10.0, 90.0])
    for _ in range(10):
        v.frame()
        v.blit()
    v._on_key(Ev("r"))
    assert v.sim.world.t == 0.0
    plt.close(v.fig)


def test_pick_level_hysteresis():
    assert pick_level(CONE_LEVELS, 1.0, 1.5) == 1.5
    assert pick_level(CONE_LEVELS, 2.0, 1.5) == 3.0
    assert pick_level(CONE_LEVELS, 1.4, 3.0) == 3.0      # no shrink just below the lower level
    assert pick_level(CONE_LEVELS, 0.8, 3.0) == 1.5      # shrinks once well below


def test_arena_viewer_runs_and_follows_agents():
    from cliffnpc.arena_viewer import SIGNATURES, ArenaViewer
    v = ArenaViewer(plt, SIGNATURES["22"], steps_per_frame=8)
    for _ in range(20):
        v.frame()
        v.blit()
    first = v.selected

    class Ev:
        key = "tab"
    v._on_key(Ev())
    assert v.selected != first or len(v.arena.agents) == 1
    plt.close(v.fig)
