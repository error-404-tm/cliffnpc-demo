"""E36 Two fixes for the individual poison test (E34).

Fix 1 -- input adapters: each NPC gets a small per-NPC matrix (4 x 64, 1 KB) that adds to
how the four food-look features enter its encoder, so it can learn a cue dependence the
shared brain lacks.
Fix 2 -- pretraining for adaptability (E35): the shared brain was pretrained with random
per-NPC allergies and fragility, so it must represent food looks and read its own harm.

Same protocol as E34 (half the NPCs allergic to look B, lives of 8 episodes, same seeds;
poisoning is now felt through the "hit" observation for every brain).  Conditions:
  e32_frozen, e32_input (fix 1), e32_both (input + hidden adapters)
  e35_frozen (fix 2, in-context only), e35_hidden, e35_both

Criteria (fixed before running):
  F1  fix 1: allergic NPCs, e32_input beats e32_frozen (eps 5-8, paired by NPC), and it
      specialises (allergic look-B eating drop minus non-allergic drop > 0)
  F2  fix 2: the poison cost (non-allergic minus allergic score, eps 5-8) is smaller for
      e35_frozen than for e32_frozen, and under e35_frozen allergic NPCs eat less at look B
      than non-allergic NPCs
  F3  adapters add to fix 2: allergic NPCs, e35_both beats e35_frozen
  F4  no adapted condition harms non-allergic NPCs vs its frozen base
"""

import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import outdir, plt, save, write_summary  # noqa: E402

from cliffnpc.neural.adapt import AdaptConfig, OnlineAdapter  # noqa: E402
from cliffnpc.neural.brain import ActBuffers, Brain, NPCState  # noqa: E402
from cliffnpc.neural.layout import GOALS, Layout, set_threads  # noqa: E402
from cliffnpc.neural.ppo import sample_preferences  # noqa: E402
from cliffnpc.neural.world import VecWorld, contested  # noqa: E402

OUT = outdir("e36_poison_fixes")
set_threads(4)
L = Layout(worlds=128, patches=4)
LIFE = 8
BRAINS = {"e32": torch.load(OUT.parent / "e32_neural_contested" / "brain.pt"),
          "e35": torch.load(OUT.parent / "e35_neural_adaptable" / "brain.pt")}
PREF = sample_preferences(L, torch.Generator().manual_seed(7))
ALLERGIC = torch.rand(L.worlds, L.agents, generator=torch.Generator().manual_seed(9)) < 0.5
CONDITIONS = {
    "e32_frozen": ("e32", None),
    "e32_input": ("e32", AdaptConfig(hidden=False, input=True)),
    "e32_both": ("e32", AdaptConfig(hidden=True, input=True)),
    "e35_frozen": ("e35", None),
    "e35_hidden": ("e35", AdaptConfig(hidden=True, input=False)),
    "e35_both": ("e35", AdaptConfig(hidden=True, input=True)),
}


def reward_scale(state) -> torch.Tensor:
    br = Brain(L)
    br.load_state_dict(state)
    w = VecWorld(L, contested(), seed=99)
    st, buf = NPCState.zeros(L.n, L.hidden), ActBuffers.zeros(L.n, L.goals)
    obs = w.reset(PREF).view(L.n, -1)
    R = torch.zeros(L.episode, L.n, L.goals)
    g = torch.Generator().manual_seed(0)
    for t in range(L.episode):
        br.act(obs, st, buf, g)
        o, r = w.step(torch.tanh(buf.u).view(L.worlds, L.agents, 2), buf.fight.view(L.worlds, L.agents))
        R[t] = r.view(L.n, -1)
        obs = o.view(L.n, -1)
    G, disc = torch.zeros(L.n, L.goals), torch.empty_like(R)
    for t in reversed(range(L.episode)):
        G = R[t] + 0.99 * G
        disc[t] = G
    return disc.reshape(-1, L.goals).std(0).clamp_min(1e-6)


RSCALE = {k: reward_scale(v) for k, v in BRAINS.items()}


def live(brain_key, acfg):
    br = Brain(L)
    br.load_state_dict(BRAINS[brain_key])
    w = VecWorld(L, contested(), seed=1234)
    w.allergic_b.copy_(ALLERGIC)
    st, buf = NPCState.zeros(L.n, L.hidden), ActBuffers.zeros(L.n, L.goals)
    ad = OnlineAdapter(br, L, RSCALE[brain_key], acfg) if acfg is not None else None
    g = torch.Generator().manual_seed(5)
    pref = PREF.view(L.n, L.goals)
    out, eat_b = torch.zeros(LIFE, L.n, L.goals), torch.zeros(LIFE, L.n)
    for ep in range(LIFE):
        obs = w.reset(PREF).view(L.n, -1)
        st.hidden.zero_()
        if ad is not None:
            ad.k = 0
        for t in range(L.episode):
            if ad is not None:
                ad.begin_window(st)
            br.act(obs, st, buf, g)
            o, r = w.step(torch.tanh(buf.u).view(L.worlds, L.agents, 2), buf.fight.view(L.worlds, L.agents))
            rr = r.view(L.n, -1)
            out[ep] += rr
            eat_b[ep] += w.eating_b.view(-1).float()
            nxt = o.view(L.n, -1)
            if ad is not None:
                ad.record(obs, buf.u, buf.fight, rr)
                if ad.ready():
                    ad.update(st, nxt, pref)
            obs = nxt
    norms = dict(shift=float(st.shift.norm(dim=-1).mean()), inp=float(st.inp.flatten(1).norm(dim=-1).mean()))
    return out, eat_b, norms


res = {}
for name, (bk, acfg) in CONDITIONS.items():
    t0 = time.perf_counter()
    res[name] = live(bk, acfg)
    print(f"{name:<11s} {time.perf_counter() - t0:.0f}s", flush=True)

al = ALLERGIC.view(-1)
pref = PREF.view(L.n, L.goals)
SPREAD = res["e32_frozen"][0][0].std(0).clamp_min(1e-6)        # one common scale for all conditions


def score(R):
    return (R / SPREAD * pref.unsqueeze(0)).sum(-1)


S = {c: score(res[c][0]) for c in CONDITIONS}


def ci(d):
    d = np.asarray(d, float)
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))


def fmt(m):
    return f"{m[0]:+.3f} ± {m[1]:.3f}"


def paired(c1, c2, m):
    return ci((S[c1][4:, m].mean(0) - S[c2][4:, m].mean(0)).numpy())


def specialisation(c):
    EB = res[c][1]
    drop = EB[0] - EB[4:].mean(0)
    a, b = drop[al].numpy(), drop[~al].numpy()
    return a.mean() - b.mean(), 1.96 * np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def poison_cost(c):
    s = S[c][4:].mean(0)
    a, b = s[al].numpy(), s[~al].numpy()
    return b.mean() - a.mean(), 1.96 * np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


lines = ["E36 Two fixes for the individual poison test", "=" * 44,
         f"{L.n} NPCs ({int(al.sum())} allergic to look B), lives of {LIFE} episodes; scores share one scale.", ""]
for gname, m in (("allergic", al), ("non-allergic", ~al)):
    lines.append(f"## {gname} NPCs")
    for c in CONDITIONS:
        R, EB, nm = res[c]
        lines.append(f"   {c:<11s} score by episode " + " ".join(f"{x:+.2f}" for x in S[c][:, m].mean(1).tolist()) +
                     "   look-B eating ticks " + " ".join(f"{x:4.1f}" for x in EB[:, m].mean(1).tolist()))
        Rm = R[4:, m].mean((0, 1))
        lines.append("               returns (eps 5-8) " + " ".join(f"{g} {v:+.2f}" for g, v in zip(GOALS, Rm.tolist())))
    lines.append("")
for c in CONDITIONS:
    pc, sp = poison_cost(c), specialisation(c)
    lines.append(f"   {c:<11s} poison cost {pc[0]:+.3f} ± {pc[1]:.3f}   specialisation {sp[0]:+.2f} ± {sp[1]:.2f}   "
                 f"adapter norms: hidden shift {res[c][2]['shift']:.2f}, input {res[c][2]['inp']:.2f}")
f1a, f1b = paired("e32_input", "e32_frozen", al), specialisation("e32_input")
pc32, pc35 = poison_cost("e32_frozen"), poison_cost("e35_frozen")
eb35 = res["e35_frozen"][1][4:].mean(0)
f2b = ci(eb35[~al].numpy()) [0] - ci(eb35[al].numpy())[0]
f3 = paired("e35_both", "e35_frozen", al)
f4 = {c: paired(c, base, ~al) for c, base in (("e32_input", "e32_frozen"), ("e32_both", "e32_frozen"),
                                              ("e35_hidden", "e35_frozen"), ("e35_both", "e35_frozen"))}
checks = [
    (f"F1 fix 1: input adapters beat frozen for allergic NPCs ({fmt(f1a)}) and specialise ({f1b[0]:+.2f} ± {f1b[1]:.2f})",
     f1a[0] - f1a[1] > 0 and f1b[0] - f1b[1] > 0),
    (f"F2 fix 2: poison cost e35 {pc35[0]:+.3f} ± {pc35[1]:.3f} vs e32 {pc32[0]:+.3f} ± {pc32[1]:.3f}; allergic eat "
     f"{f2b:+.1f} fewer look-B ticks than non-allergic under e35_frozen",
     pc35[0] + pc35[1] < pc32[0] - pc32[1] and f2b > 0),
    (f"F3 adapters add to fix 2 for allergic NPCs ({fmt(f3)})", f3[0] - f3[1] > 0),
    ("F4 non-allergic not harmed: " + ", ".join(f"{c} {fmt(v)}" for c, v in f4.items()),
     all(v[0] + v[1] >= 0 for v in f4.values())),
]
lines += ["", "CRITERIA (fixed before running)"] + [f"   {'PASS' if ok else 'FAIL'}  {lab}" for lab, ok in checks]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, (gname, m) in zip(axes, (("allergic", al), ("non-allergic", ~al))):
    for c in CONDITIONS:
        ax.plot(range(1, LIFE + 1), res[c][1][:, m].mean(1).numpy(), "o-", label=c)
    ax.set_title(f"{gname} NPCs: ticks eating at look-B patches")
    ax.set_xlabel("episode of life")
axes[0].legend(fontsize=7)
save(fig, OUT, "look_b_eating.png")
write_summary(OUT, lines)
