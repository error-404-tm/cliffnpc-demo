"""E34 Individual poison: a change that breaks the pretrained behaviour for some NPCs only.

Starting from the E32 brain, a random half of the NPCs become allergic to food with look
B: eating there gives no satiety, costs health (0.1/s) and is a food setback.  The look is
observable, the allergy is not -- so an allergic NPC must learn "avoid look B" for itself,
while non-allergic NPCs should keep eating both looks.  Pooled training of the shared
weights cannot tell the two groups apart.

Lives of 8 episodes (hidden state reset per episode, adapters persist).  Conditions:
frozen; adapt (online per-NPC adapters, lr 0.02); adapt_fast (lr 0.1, for information);
finetune (PPO on the shared weights with everyone's data after each episode).

Criteria (fixed before running; adapt = lr 0.02):
  I1  allergic NPCs: adapt beats frozen over episodes 5-8 (paired by NPC)
  I2  non-allergic NPCs are not harmed by adapt
  I3  individual specialisation: under adapt, allergic NPCs cut their look-B eating (eps
      5-8 vs ep 1) more than non-allergic NPCs do (difference in differences > 0)
  I4  allergic NPCs: adapt beats finetune over episodes 5-8
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
from cliffnpc.neural.ppo import PPOConfig, Trainer, sample_preferences  # noqa: E402
from cliffnpc.neural.world import VecWorld, contested  # noqa: E402

OUT = outdir("e34_individual_poison")
set_threads(4)
L = Layout(worlds=128, patches=4)
LIFE = 8
STATE = torch.load(OUT.parent / "e32_neural_contested" / "brain.pt")
PREF = sample_preferences(L, torch.Generator().manual_seed(7))
ALLERGIC = torch.rand(L.worlds, L.agents, generator=torch.Generator().manual_seed(9)) < 0.5
CONDITIONS = ("frozen", "adapt", "adapt_fast", "finetune")


def reward_scale() -> torch.Tensor:
    br = Brain(L)
    br.load_state_dict(STATE)
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


RSCALE = reward_scale()


def live(condition: str):
    """Per-episode returns [LIFE, N, G] and look-B eating ticks [LIFE, N]."""
    out = torch.zeros(LIFE, L.n, L.goals)
    eat_b = torch.zeros(LIFE, L.n)
    if condition == "finetune":
        tr = Trainer(L, PPOConfig(block=512, lr=3e-4), seed=0, world_cfg=contested())
        tr.brain.load_state_dict(STATE)
        tr.world.g.manual_seed(1234)
        tr.world.allergic_b.copy_(ALLERGIC)
        orig_step = tr.world.step

        def step(a, f, _ep=[0]):
            o, r = orig_step(a, f)
            eat_b[_ep[0]] += tr.world.eating_b.view(-1).float()
            return o, r
        tr.world.step = step
        for ep in range(LIFE):
            step.__defaults__[0][0] = ep
            out[ep] = tr.collect(pref=PREF)["per_npc"]
            tr.update()
        return out, eat_b, 0.0
    br = Brain(L)
    br.load_state_dict(STATE)
    w = VecWorld(L, contested(), seed=1234)
    w.allergic_b.copy_(ALLERGIC)
    st, buf = NPCState.zeros(L.n, L.hidden), ActBuffers.zeros(L.n, L.goals)
    ad = None
    if condition.startswith("adapt"):
        ad = OnlineAdapter(br, L, RSCALE, AdaptConfig(lr=0.1 if condition == "adapt_fast" else 0.02))
    g = torch.Generator().manual_seed(5)
    pref = PREF.view(L.n, L.goals)
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
    return out, eat_b, (float(st.shift.norm(dim=-1).mean()) if ad is not None else 0.0)


res = {}
for c in CONDITIONS:
    t0 = time.perf_counter()
    res[c] = live(c)
    print(f"{c:<10s} {time.perf_counter() - t0:.0f}s", flush=True)

al = ALLERGIC.view(-1)
pref = PREF.view(L.n, L.goals)
SPREAD = res["frozen"][0][0].std(0).clamp_min(1e-6)


def score(R):
    return (R / SPREAD * pref.unsqueeze(0)).sum(-1)


def ci(d):
    d = np.asarray(d, float)
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))


def fmt(m):
    return f"{m[0]:+.3f} ± {m[1]:.3f}"


lines = ["E34 Individual poison (allergy to look B in half the NPCs)", "=" * 57,
         f"{L.n} NPCs ({int(al.sum())} allergic), lives of {LIFE} episodes; brain from E32.", ""]
S = {c: score(res[c][0]) for c in CONDITIONS}
for gname, m in (("allergic", al), ("non-allergic", ~al)):
    lines.append(f"## {gname} NPCs")
    for c in CONDITIONS:
        R, EB, _ = res[c]
        lines.append(f"   {c:<10s} score by episode " + " ".join(f"{x:+.2f}" for x in S[c][:, m].mean(1).tolist()) +
                     "   look-B eating ticks by episode " + " ".join(f"{x:4.1f}" for x in EB[:, m].mean(1).tolist()))
        Rm = R[4:, m].mean((0, 1))
        lines.append(f"              returns (eps 5-8) " + " ".join(f"{g} {v:+.2f}" for g, v in zip(GOALS, Rm.tolist())))
    lines.append("")


def diff(c1, c2, m):
    return (S[c1][4:, m].mean(0) - S[c2][4:, m].mean(0)).numpy()


def did(c):
    EB = res[c][1]
    drop = EB[0] - EB[4:].mean(0)                       # [N] reduction in look-B eating
    return drop[al].mean().item() - drop[~al].mean().item(), drop


i1, i2, i4 = ci(diff("adapt", "frozen", al)), ci(diff("adapt", "frozen", ~al)), ci(diff("adapt", "finetune", al))
d_adapt, drop = did("adapt")
# CI for the difference in differences: two independent groups of NPCs
dd = (drop[al].numpy(), drop[~al].numpy())
dd_h = 1.96 * np.sqrt(dd[0].var(ddof=1) / len(dd[0]) + dd[1].var(ddof=1) / len(dd[1]))
d_ft, _ = did("finetune")
d_fast, _ = did("adapt_fast")
lines.append(f"specialisation (look-B eating drop, allergic minus non-allergic): adapt {d_adapt:+.2f} ± {dd_h:.2f}, "
             f"adapt_fast {d_fast:+.2f}, finetune {d_ft:+.2f}")
lines.append(f"adapt_fast vs frozen: allergic {fmt(ci(diff('adapt_fast', 'frozen', al)))}, non-allergic "
             f"{fmt(ci(diff('adapt_fast', 'frozen', ~al)))}; adapter shift norms: adapt {res['adapt'][2]:.2f}, "
             f"adapt_fast {res['adapt_fast'][2]:.2f}")
lines.append(f"finetune vs frozen: allergic {fmt(ci(diff('finetune', 'frozen', al)))}, non-allergic "
             f"{fmt(ci(diff('finetune', 'frozen', ~al)))}")
checks = [
    (f"I1 allergic: adapt beats frozen ({fmt(i1)})", i1[0] - i1[1] > 0),
    (f"I2 non-allergic not harmed ({fmt(i2)})", i2[0] + i2[1] >= 0),
    (f"I3 adapt specialises by individual ({d_adapt:+.2f} ± {dd_h:.2f})", d_adapt - dd_h > 0),
    (f"I4 allergic: adapt beats finetune ({fmt(i4)})", i4[0] - i4[1] > 0),
]
lines += ["", "CRITERIA (fixed before running)"] + [f"   {'PASS' if ok else 'FAIL'}  {lab}" for lab, ok in checks]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (gname, m) in zip(axes, (("allergic", al), ("non-allergic", ~al))):
    for c, col in zip(CONDITIONS, ("#7f8c8d", "#c0392b", "#e67e22", "#2980b9")):
        ax.plot(range(1, LIFE + 1), res[c][1][:, m].mean(1).numpy(), "o-", color=col, label=c)
    ax.set_title(f"{gname} NPCs: ticks spent eating at look-B patches")
    ax.set_xlabel("episode of life")
axes[0].legend()
save(fig, OUT, "look_b_eating.png")
write_summary(OUT, lines)
