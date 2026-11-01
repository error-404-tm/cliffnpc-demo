"""E33 Online per-NPC adaptation during play (shared brain frozen, adapters learn).

Starts from the brain pretrained in the contested world (E32).  Each NPC keeps a fixed
personality for its "life" of 8 consecutive episodes (hidden state reset per episode,
adapters persist).  Two changes the pretrained brain never saw:

  fragile    a random half of the NPCs secretly take 3x damage (individual: only that
             NPC's own experience tells it)
  cue flip   patch looks mean the opposite (cue validity -0.6) for everyone

Conditions: frozen (no learning); adapt (online adapters, cliffnpc/neural/adapt.py);
finetune (PPO on the shared weights with all NPCs' pooled data, after each episode --
a reference, not a per-NPC method).  Scores: preference-weighted returns, each goal
divided by its spread across NPCs (frozen, episode 1).

Criteria (fixed before running):
  A1  fragile NPCs: adapt beats frozen over episodes 5-8 (paired by NPC)
  A2  robust NPCs are not harmed by adaptation (paired difference not significantly < 0)
  A3  cue flip: adapt beats frozen over episodes 5-8 (paired by NPC)
  A4  cost of online adaptation and adapter sizes reported
"""

import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import outdir, plt, save, write_summary  # noqa: E402

from cliffnpc.neural.adapt import OnlineAdapter  # noqa: E402
from cliffnpc.neural.brain import ActBuffers, Brain, NPCState  # noqa: E402
from cliffnpc.neural.layout import GOALS, Layout, set_threads  # noqa: E402
from cliffnpc.neural.ppo import PPOConfig, Trainer, sample_preferences  # noqa: E402
from cliffnpc.neural.world import VecWorld, contested  # noqa: E402

OUT = outdir("e33_online_adaptation")
set_threads(4)
L = Layout(worlds=128, patches=4)
LIFE = 8
BRAIN = OUT.parent / "e32_neural_contested" / "brain.pt"
STATE = torch.load(BRAIN)
PREF = sample_preferences(L, torch.Generator().manual_seed(7))                  # [W, K, G], fixed per NPC
FRAGILE = torch.rand(L.worlds, L.agents, generator=torch.Generator().manual_seed(8)) < 0.5


def world_cfg(scenario):
    c = contested()
    if scenario == "cue_flip":
        c.cue = -0.6
    return c


def reward_scale() -> torch.Tensor:
    """Per-goal spread of discounted returns under the pretrained brain (base world)."""
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
    G = torch.zeros(L.n, L.goals)
    disc = torch.empty_like(R)
    for t in reversed(range(L.episode)):
        G = R[t] + 0.99 * G
        disc[t] = G
    return disc.reshape(-1, L.goals).std(0).clamp_min(1e-6)


RSCALE = reward_scale()


def live(scenario: str, condition: str) -> tuple[torch.Tensor, dict]:
    """Returns per-episode returns [LIFE, N, G] and cost / adapter stats."""
    cfg = world_cfg(scenario)
    stats = dict(adapt_seconds=0.0, ticks=0, scale_norm=0.0, shift_norm=0.0)
    out = torch.zeros(LIFE, L.n, L.goals)
    if condition == "finetune":
        tr = Trainer(L, PPOConfig(block=512, lr=3e-4), seed=0, world_cfg=cfg)
        tr.brain.load_state_dict(STATE)
        tr.world.g.manual_seed(1234)
        if scenario == "fragile":
            tr.world.fragility[FRAGILE] = 3.0
        for ep in range(LIFE):
            out[ep] = tr.collect(pref=PREF)["per_npc"]
            tr.update()
        return out, stats
    br = Brain(L)
    br.load_state_dict(STATE)
    w = VecWorld(L, cfg, seed=1234)
    if scenario == "fragile":
        w.fragility[FRAGILE] = 3.0
    st, buf = NPCState.zeros(L.n, L.hidden), ActBuffers.zeros(L.n, L.goals)
    ad = OnlineAdapter(br, L, RSCALE) if condition == "adapt" else None
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
            nxt = o.view(L.n, -1)
            if ad is not None:
                ad.record(obs, buf.u, buf.fight, rr)
                if ad.ready():
                    s = ad.update(st, nxt, pref)
                    stats["scale_norm"], stats["shift_norm"] = s["scale_norm"], s["shift_norm"]
            obs = nxt
            stats["ticks"] += 1
    if ad is not None:
        stats["adapt_seconds"] = ad.seconds
    return out, stats


res, cost = {}, {}
for scenario in ("fragile", "cue_flip"):
    for condition in ("frozen", "adapt", "finetune"):
        t0 = time.perf_counter()
        res[(scenario, condition)], cost[(scenario, condition)] = live(scenario, condition)
        print(f"{scenario:<9s} {condition:<9s} {time.perf_counter() - t0:.0f}s", flush=True)

pref = PREF.view(L.n, L.goals)


def score(R):
    """[LIFE, N]: preference-weighted returns, each goal over its spread (frozen, episode 1)."""
    return (R / SPREAD * pref.unsqueeze(0)).sum(-1)


SPREAD = {}
lines = ["E33 Online per-NPC adaptation (shared brain frozen)", "=" * 51,
         f"{L.n} NPCs, lives of {LIFE} episodes; brain from E32.", ""]
fr = FRAGILE.view(-1)
checks = []
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for si, scenario in enumerate(("fragile", "cue_flip")):
    SPREAD = res[(scenario, "frozen")][0].std(0).clamp_min(1e-6)
    S = {c: score(res[(scenario, c)]) for c in ("frozen", "adapt", "finetune")}
    groups = (("fragile NPCs", fr), ("robust NPCs", ~fr)) if scenario == "fragile" else (("all NPCs", torch.ones_like(fr)),)
    lines.append(f"## {scenario}")
    for gname, m in groups:
        for c in ("frozen", "adapt", "finetune"):
            per_ep = S[c][:, m].mean(1)
            R = res[(scenario, c)][4:, m].mean((0, 1))
            lines.append(f"   {gname:<13s} {c:<9s} score by episode " + " ".join(f"{x:+.2f}" for x in per_ep.tolist()) +
                         "   returns (eps 5-8) " + " ".join(f"{gg} {v:+.2f}" for gg, v in zip(GOALS, R.tolist())))
        d = (S["adapt"][4:, m].mean(0) - S["frozen"][4:, m].mean(0)).numpy()
        m_, h_ = d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        lines.append(f"   {gname:<13s} adapt - frozen (eps 5-8, paired by NPC): {m_:+.3f} ± {h_:.3f}")
        if scenario == "fragile" and gname == "fragile NPCs":
            checks.append((f"A1 fragile NPCs: adapt beats frozen ({m_:+.3f} ± {h_:.3f})", m_ - h_ > 0))
        elif scenario == "fragile":
            checks.append((f"A2 robust NPCs not harmed ({m_:+.3f} ± {h_:.3f})", m_ + h_ >= 0))
        else:
            checks.append((f"A3 cue flip: adapt beats frozen ({m_:+.3f} ± {h_:.3f})", m_ - h_ > 0))
        ax = axes[si if scenario == "cue_flip" else (0 if gname == "fragile NPCs" else 1)]
        for c, col in (("frozen", "#7f8c8d"), ("adapt", "#c0392b"), ("finetune", "#2980b9")):
            ax.plot(range(1, LIFE + 1), S[c][:, m].mean(1).numpy(), "o-", color=col, label=c)
        ax.set_title(f"{scenario}: {gname}")
        ax.set_xlabel("episode of life")
    lines.append("")
axes[0].legend()
save(fig, OUT, "adaptation.png")
c = cost[("fragile", "adapt")]
ms = 1e3 * c["adapt_seconds"] / c["ticks"]
lines.append(f"cost: adapter updates {ms:.2f} ms per tick for {L.n} NPCs ({ms * 1000 / L.n:.2f} ms per tick per 1000 NPCs; "
             f"one update every 16 ticks); adapter norms at the end: scale {c['scale_norm']:.2f}, shift {c['shift_norm']:.2f} "
             f"(cap 2.0); adapter memory {2 * L.hidden * 4} bytes per NPC")
checks.append(("A4 cost and adapter sizes reported", True))
lines += ["", "CRITERIA (fixed before running)"] + [f"   {'PASS' if ok else 'FAIL'}  {lab}" for lab, ok in checks]
write_summary(OUT, lines)
