"""Learning gym: tasks run under every policy; the optimum knows the rule."""

import numpy as np

from cliffnpc import gym
from cliffnpc.connection import FEATURES


def test_tasks_run_under_every_policy():
    for policy in gym.POLICIES:
        assert len(gym.discrimination(0, policy, n_trials=2, reverse_at=1)) == 2
        assert len(gym.fear(0, policy, n_trials=2, extinguish_at=1)) == 2
        assert len(gym.courage(0, policy, n_trials=2, T_trial=5.0)) == 2
    assert len(gym.one_shot(0, "clifford", n_trials=3, death_at=1, split=1)) == 3


def test_optimal_never_goes_to_false_food_first():
    recs = gym.discrimination(1, "optimal", n_trials=8)
    assert all(r["first"] == "real" for r in recs)


def test_food_cue_is_one_hot_in_the_situation_vector():
    ar, a = gym.gym_arena(0, "clifford")
    gym._place_pair(ar, np.random.default_rng(0), looks=(gym.LOOK_B, gym.LOOK_B))
    x = ar.context_of(a, ar.percept(a))
    assert x[FEATURES.index("food_cue_b")] == 1.0 and x[FEATURES.index("food_cue_a")] == 0.0


def test_frozen_agent_does_not_learn():
    ar, a = gym.gym_arena(0, "frozen")
    g0, W0 = a.npc.g.copy(), a.conn.W.copy()
    gym._place_pair(ar, np.random.default_rng(0), looks=(gym.LOOK_A, gym.LOOK_B))
    ar.run(20.0)
    assert np.allclose(a.conn.W, W0) and np.allclose(a.npc.g, g0)
