"""Option B frame law: the whole frame is physical (orientation = holonomy of
experience); stretch saturates like velocity vs rapidity."""

import numpy as np
from scipy.linalg import expm

from cliffnpc import algebra as ga
from cliffnpc.cognition import NPC, saturate_frame
from cliffnpc.config import CognitionConfig

ETA = ga.ETA
H = 1.0


def sym(X):
    return ga.split_eta_generator(X)[1]


def test_repeated_same_direction_boosts_saturate_and_never_go_singular():
    A = sym(np.diag([0.0, 0.0, 1.0, 0.0]) + 0.0)       # stretch the explore axis
    L = np.eye(4)
    tops, steps = [], []
    for _ in range(100):
        L_new = saturate_frame(L, expm(0.05 * A) @ L, H)
        s_top = np.log(np.linalg.svd(L_new, compute_uv=False)).max()
        steps.append(s_top - (tops[-1] if tops else 0.0))
        tops.append(s_top)
        L = L_new
    assert tops[-1] < H                                   # bounded by the horizon
    assert np.linalg.cond(L) <= np.exp(2 * H) + 1e-9      # never singular
    assert steps[0] > 10 * steps[-1] > 0                  # diminishing returns, still moving
    assert np.isclose(steps[0], 0.05, rtol=1e-3)          # early shifts pass almost whole


def test_reversal_after_entrenchment_takes_as_long_as_the_buildup():
    A = sym(np.diag([0.0, 0.0, 1.0, 0.0]))
    L = np.eye(4)
    for _ in range(100):
        L = saturate_frame(L, expm(0.05 * A) @ L, H)
    for _ in range(100):
        L = saturate_frame(L, expm(-0.05 * A) @ L, H)
    assert np.allclose(L, np.eye(4), atol=1e-6)           # rapidity is additive: exact return


def test_holonomy_of_a_personality_loop_is_kept():
    """Change, change, undo, undo: the metric returns (to O(eps^3)) but the frame
    keeps a Lorentz turn ~ -eps^2 [P1, P2]; the law must not erase it."""
    rng = np.random.default_rng(0)
    P1, P2 = sym(rng.normal(size=(4, 4))), sym(rng.normal(size=(4, 4)))
    eps = 0.03
    L = np.eye(4)
    for A in (eps * P1, eps * P2, -eps * P1, -eps * P2):
        L = saturate_frame(L, expm(A) @ L, H)
    turn = L                                              # started at the identity
    assert np.linalg.norm(ga.metric_from_frame(L) - ETA) < 5 * eps ** 3 * 10
    assert np.linalg.norm(turn - np.eye(4)) > 0.5 * eps ** 2 * np.linalg.norm(P1 @ P2 - P2 @ P1)
    assert np.allclose(turn.T @ ETA @ turn, ETA, atol=5e-4)   # a Lorentz transformation


def test_commit_transports_the_inner_state():
    npc = NPC(CognitionConfig(), C=np.array([0.3, -0.2, 0.5, 0.1]))
    Co, s = npc.L @ npc.C, npc.interval()
    rng = np.random.default_rng(1)
    for _ in range(20):
        npc.commit_frame(expm(0.05 * sym(rng.normal(size=(4, 4)))) @ npc.L)
    assert np.allclose(npc.L @ npc.C, Co)                 # experiential state persists
    assert np.isclose(npc.interval(), s)                  # no jolt on the light cone
    assert not np.allclose(npc.C, [0.3, -0.2, 0.5, 0.1])  # its drive reading changed


def test_order_of_experience_is_remembered_in_the_orientation():
    rng = np.random.default_rng(2)
    P1, P2 = sym(rng.normal(size=(4, 4))), sym(rng.normal(size=(4, 4)))
    C0 = np.array([0.3, -0.2, 0.5, 0.1])

    def live(order):
        npc = NPC(CognitionConfig(), C=C0.copy())
        for P in order:
            for _ in range(10):
                npc.commit_frame(expm(0.01 * P) @ npc.L)
        return npc

    P1, P2 = P1 / np.linalg.norm(P1), P2 / np.linalg.norm(P2)
    a, b = live([P1, P2]), live([P2, P1])
    dg = np.linalg.norm(a.g - b.g)
    dL = np.linalg.norm(a.L - b.L)
    assert dL > 3 * dg                                    # order lives mostly in the frame
    assert not np.allclose(a.C, b.C, atol=1e-3)           # and shows up in the drive reading
