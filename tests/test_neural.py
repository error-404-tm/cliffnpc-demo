"""Neural NPCs: memory layout rules, world rules, brain equivalences, PPO."""

import pytest
import torch

from cliffnpc.neural.brain import ActBuffers, Brain, NPCState
from cliffnpc.neural.layout import Layout
from cliffnpc.neural.ppo import PPOConfig, Trainer
from cliffnpc.neural.world import OBS_NAMES, VecWorld

SMALL = Layout(worlds=4, agents=8, patches=8, episode=16, tile=8)


def test_layout_requires_powers_of_two():
    with pytest.raises(ValueError):
        Layout(worlds=3)
    assert SMALL.n == 32 and len(OBS_NAMES) == SMALL.features


def test_world_writes_into_the_same_buffers_and_views_share_memory():
    w = VecWorld(SMALL, seed=0)
    obs0 = w.reset(torch.full((4, 8, 4), 0.5))
    ptr_obs, ptr_rew = obs0.data_ptr(), w.reward.data_ptr()
    for _ in range(5):
        obs, r = w.step(torch.zeros(4, 8, 2), torch.zeros(4, 8, dtype=torch.bool))
        assert obs.data_ptr() == ptr_obs and r.data_ptr() == ptr_rew          # preallocated, in place
    assert obs.is_contiguous() and obs.view(SMALL.n, SMALL.features).data_ptr() == ptr_obs


def test_eating_at_a_real_patch_raises_satiety_and_counts_as_food():
    w = VecWorld(SMALL, seed=0)
    w.reset()
    w.real[0, 0], w.stock[0, 0] = True, 5.0
    w.pos[0, 0] = w.ppos[0, 0]
    w.satiety[0, 0] = 0.1
    _, r = w.step(torch.zeros(4, 8, 2), torch.zeros(4, 8, dtype=torch.bool))
    assert r[0, 0, 0] > 0 and w.satiety[0, 0] > 0.1 - 0.05 * 0.2 + 1e-6


def test_combat_split_favours_the_bigger_side_and_death_respawns_in_place():
    w = VecWorld(SMALL, seed=0)
    w.reset()
    w.pos[0] = torch.tensor([[50.0, 50.0]] * 8)                              # everyone at one spot
    w.pos[0, 1:4] += 0.5                                                     # A: slots 0-3, B: 4-7
    w.pos[0, 5:8] = torch.tensor([90.0, 10.0])                               # 1 B fighter stays (slot 4)
    fight = torch.zeros(4, 8, dtype=torch.bool)
    fight[0, :5] = True
    w.step(torch.zeros(4, 8, 2), fight)
    assert w.hit[0, 4] > 3 * w.hit[0, 0] > 0                                 # 4 vs 1: the lone one takes most
    w.health[0, 4] = 1e-6
    w.step(torch.zeros(4, 8, 2), fight)
    assert w.health[0, 4] == 1.0 and torch.allclose(w.pos[0, 4], w.home[4])  # respawned in its slot


def test_tiling_does_not_change_the_brain_outputs():
    torch.manual_seed(0)
    br = Brain(SMALL)
    obs = torch.randn(SMALL.n, SMALL.features)
    outs = []
    for tile in (8, 32):
        br.L = Layout(worlds=4, agents=8, patches=8, episode=16, tile=tile)
        st, buf = NPCState.zeros(SMALL.n, SMALL.hidden), ActBuffers.zeros(SMALL.n, SMALL.goals)
        br.act(obs, st, buf, greedy=True)
        outs.append((buf.value.clone(), st.hidden.clone()))
    assert torch.allclose(outs[0][0], outs[1][0], atol=1e-6) and torch.allclose(outs[0][1], outs[1][1], atol=1e-6)


def test_fused_sequence_equals_step_by_step():
    torch.manual_seed(1)
    br = Brain(SMALL)
    T, B = 6, 5
    obs = torch.randn(T, B, SMALL.features)
    h = torch.zeros(B, SMALL.hidden)
    z = torch.zeros(B, SMALL.hidden)
    po_seq, v_seq = br.sequence(obs, h, z, z)
    for t in range(T):
        po, v, h = br.step(obs[t], h, z, z)
        assert torch.allclose(po, po_seq[t], atol=1e-5) and torch.allclose(v, v_seq[t], atol=1e-5)


def test_adapter_gradient_of_each_npc_depends_only_on_its_own_row():
    torch.manual_seed(2)
    br = Brain(SMALL)
    B = 6
    obs = torch.randn(1, B, SMALL.features)
    scale = torch.zeros(B, SMALL.hidden, requires_grad=True)
    shift = torch.zeros(B, SMALL.hidden, requires_grad=True)
    _, v = br.sequence(obs, torch.zeros(B, SMALL.hidden), scale, shift)
    v[0, 2, 0].backward()                                                    # NPC 2's food value only
    g = shift.grad.abs().sum(-1)
    assert g[2] > 0 and float(g.sum() - g[2]) == 0.0


def test_one_ppo_iteration_runs_and_moves_the_weights():
    tr = Trainer(SMALL, PPOConfig(block=8, epochs=1), seed=0)
    before = [p.clone() for p in tr.brain.parameters()]
    info = tr.collect()
    tr.update()
    assert info["per_npc"].shape == (SMALL.n, SMALL.goals)
    assert any(not torch.allclose(a, b) for a, b in zip(before, tr.brain.parameters()))


def test_online_adapter_updates_only_adapters_and_stays_bounded():
    from cliffnpc.neural.adapt import AdaptConfig, OnlineAdapter
    tr = Trainer(SMALL, PPOConfig(block=8, epochs=1), seed=0)
    shared = [p.clone() for p in tr.brain.parameters()]
    ad = OnlineAdapter(tr.brain, SMALL, torch.ones(SMALL.goals), AdaptConfig(window=4, max_norm=0.5))
    st, buf = NPCState.zeros(SMALL.n, SMALL.hidden), ActBuffers.zeros(SMALL.n, SMALL.goals)
    pref = torch.full((SMALL.n, SMALL.goals), 0.5)
    obs = tr.world.reset(pref.view(4, 8, 4)).view(SMALL.n, -1)
    for _ in range(12):
        ad.begin_window(st)
        tr.brain.act(obs, st, buf)
        o, r = tr.world.step(torch.tanh(buf.u).view(4, 8, 2), buf.fight.view(4, 8))
        ad.record(obs, buf.u, buf.fight, r.view(SMALL.n, -1))
        obs = o.view(SMALL.n, -1)
        if ad.ready():
            ad.update(st, obs, pref)
    assert ad.updates == 3
    assert all(torch.equal(a, b) for a, b in zip(shared, tr.brain.parameters()))        # shared frozen
    assert float(st.shift.abs().sum()) > 0 and float(st.shift.norm(dim=-1).max()) <= 0.5 + 1e-5


def test_allergic_npcs_are_poisoned_by_look_b_and_others_are_not():
    w = VecWorld(SMALL, seed=0)
    w.reset()
    w.real[0, 0], w.stock[0, 0], w.look_b[0, 0] = True, 5.0, True
    w.pos[0, 0] = w.ppos[0, 0]
    w.pos[0, 1] = w.ppos[0, 0]
    w.allergic_b[0, 0] = True                                       # slot 0 allergic, slot 1 not
    w.satiety[0, :2] = 0.5
    _, r = w.step(torch.zeros(4, 8, 2), torch.zeros(4, 8, dtype=torch.bool))
    assert r[0, 0, 0] < 0 and r[0, 0, 1] < 0 and w.health[0, 0] < 1.0      # poisoned
    assert r[0, 1, 0] > 0 and w.health[0, 1] == 1.0                         # fed
    assert bool(w.eating_b[0, 0]) and bool(w.eating_b[0, 1])


def test_input_adapter_is_per_row_and_zero_means_no_change():
    torch.manual_seed(3)
    br = Brain(SMALL)
    B = 5
    obs = torch.rand(2, B, SMALL.features)
    h, z = torch.zeros(B, SMALL.hidden), torch.zeros(B, SMALL.hidden)
    inp = torch.zeros(B, 4, SMALL.hidden, requires_grad=True)
    po0, _ = br.sequence(obs, h, z, z)
    po1, v1 = br.sequence(obs, h, z, z, inp)
    assert torch.allclose(po0, po1)                                          # zero adapter = shared brain
    v1[1, 3, 0].backward()
    g = inp.grad.flatten(1).abs().sum(-1)
    assert g[3] > 0 and float(g.sum() - g[3]) == 0.0
