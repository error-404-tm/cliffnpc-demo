"""E35 Pretraining for adaptability: individual variation during pretraining.

As E32 (contested world, 300 iterations), but every episode each NPC is re-randomised:
allergic to look A or look B with probability 0.5 (half each) and fragile (3x damage)
with probability 0.3.  Allergy and fragility are unobserved; poisoning is felt through the
"hit" observation like an injury.  The shared brain must therefore represent food looks
and read its own harm -- learning, within a life, which look to avoid (RL^2-style), and
giving per-NPC adapters something to work with.  Extra check:
  P5  allergic NPCs eat less of their poisonous look than of the other one (trained)

---- E32 description follows ----
E32 Neural NPCs: pretraining in a contested world (tribes meet at food).

As E31, but patches are few (4 per world), small (stock 3) and central between the two
homes (world.contested()), so foraging costs safety and staying safe costs food -- a
scripted check showed the trade-off (forage: food +0.33, safety -0.18; forage-but-flee:
food +0.21, safety 0).  300 iterations.  Same criteria as E31.

---- E31 description follows ----
E31 Neural NPCs: pretraining a shared recurrent brain in the batched world (PyTorch, CPU).

World: cliffnpc/neural/world.py -- 128 worlds x 8 agents (two tribes, all brain-driven,
self-play), 8 patches each; food with stock and false patches, need, knowledge, combat
with the n/(n+m) rule, shelter at home, company.  Brain: ~27.5k parameters, GRU(64),
policy + four goal-value heads, preference-conditioned (each NPC sees its omega).
Training: PPO, per-goal GAE, advantages scalarised by each NPC's own preference.

Criteria (fixed before running):
  P1  it learns: final greedy score above the first iteration's (CI over NPCs)
  P2  it beats a random policy, for every preference profile
  P3  preference conditioning works with one network: forager more fed than cautious,
      cautious safer than forager, curious more knowledgeable than balanced
  P4  throughput and memory reported
"""

import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import outdir, plt, save, write_summary  # noqa: E402

from cliffnpc.neural.layout import GOALS, Layout, set_threads  # noqa: E402
from cliffnpc.neural.ppo import PPOConfig, Trainer  # noqa: E402
from cliffnpc.neural.world import contested  # noqa: E402
from cliffnpc.phase_c import PREFERENCES  # noqa: E402

OUT = outdir("e35_neural_adaptable")
set_threads(4)
L = Layout(worlds=128, patches=4)
ITERS = int(os.environ.get("E35_ITERS", "300"))
cfg = PPOConfig(block=512, lr=6e-4)
tr = Trainer(L, cfg, seed=0, world_cfg=contested(), individual=dict(p_allergic=0.5, p_fragile=0.3))
PROFILES = {k: torch.tensor(v / v.sum() * 2.0, dtype=torch.float32) for k, v in PREFERENCES.items()}


def evaluate(policy: str, episodes: int = 4) -> dict:
    """Per profile: mean episode goal returns over all NPCs (every NPC gets that profile)."""
    out = {}
    for name, w in PROFILES.items():
        pref = w.expand(L.worlds, L.agents, L.goals).clone()
        R = []
        for _ in range(episodes):
            if policy == "random":
                obs = tr.world.reset(pref)
                tot = torch.zeros(L.worlds, L.agents, L.goals)
                g = torch.Generator().manual_seed(len(R))
                for _ in range(L.episode):
                    _, r = tr.world.step(torch.rand(L.worlds, L.agents, 2, generator=g) * 2 - 1,
                                         torch.rand(L.worlds, L.agents, generator=g) < 0.3)
                    tot += r
                R.append(tot.view(L.n, L.goals))
            else:
                info = tr.collect(pref=pref, greedy=True)
                R.append(info["per_npc"].clone())
        Rm = torch.cat(R)
        score = Rm @ w
        out[name] = dict(returns=Rm.mean(0), score=float(score.mean()),
                         score_ci=float(1.96 * score.std() / np.sqrt(len(score))))
    return out


hist = []
first = evaluate("greedy", episodes=2)
rand = evaluate("random", episodes=2)
t_collect = t_update = 0.0
t_start = time.perf_counter()
for it in range(ITERS):
    t0 = time.perf_counter()
    info = tr.collect()
    t1 = time.perf_counter()
    st = tr.update()
    t2 = time.perf_counter()
    t_collect += t1 - t0
    t_update += t2 - t1
    hist.append(dict(it=it, score=float(info["score"]), returns=info["returns"].tolist(), **st))
    if it % 10 == 0 or it == ITERS - 1:
        print(f"it {it:3d} score {hist[-1]['score']:6.2f} returns {[round(x, 2) for x in hist[-1]['returns']]} "
              f"v {st['v']:.3f} kl {st['kl']:.4f} ent {st['ent']:.2f}  ({t1 - t0:.1f}s + {t2 - t1:.1f}s)", flush=True)
wall = time.perf_counter() - t_start
final = evaluate("greedy", episodes=4)
torch.save(tr.brain.state_dict(), OUT / "brain.pt")

steps = ITERS * L.episode * L.n
lines = ["E35 Pretraining for adaptability (individual allergies and fragility)", "=" * 68,
         f"{L.worlds} worlds x {L.agents} agents = {L.n} NPCs; {ITERS} iterations x {L.episode} ticks; "
         f"brain {tr.brain.n_params()} parameters ({tr.brain.n_params() * 4 / 1024:.0f} KB); "
         f"persistent state {L.bytes_per_npc()} bytes per NPC.",
         f"throughput: collection (world + brain inference) {steps / t_collect / 1e3:.0f}k NPC-steps/s; "
         f"training {steps / t_update / 1e3:.0f}k NPC-steps/s; {steps / 1e6:.0f}M NPC-steps in {wall / 60:.1f} min.", ""]
lines.append("Per profile: score (omega . returns), and returns per goal (food, safety, social, knowledge)")
for name in PROFILES:
    for lab, res in (("random", rand), ("start", first), ("trained", final)):
        r = res[name]
        lines.append(f"   {name:<9s} {lab:<8s} score {r['score']:+7.2f} ± {r['score_ci']:.2f}   " +
                     " ".join(f"{g} {v:+.2f}" for g, v in zip(GOALS, r["returns"].tolist())))
f = {k: v["returns"] for k, v in final.items()}
checks = [
    ("P1 learns: final score above the start for every profile",
     all(final[k]["score"] - final[k]["score_ci"] > first[k]["score"] + first[k]["score_ci"] for k in PROFILES)),
    ("P2 beats a random policy for every profile",
     all(final[k]["score"] - final[k]["score_ci"] > rand[k]["score"] + rand[k]["score_ci"] for k in PROFILES)),
    (f"P3 preferences differentiate (forager food {f['forager'][0]:.2f} vs cautious {f['cautious'][0]:.2f}; cautious "
     f"safety {f['cautious'][1]:.2f} vs forager {f['forager'][1]:.2f}; curious knowledge {f['curious'][3]:.2f} vs "
     f"balanced {f['balanced'][3]:.2f})",
     bool(f["forager"][0] > f["cautious"][0] and f["cautious"][1] > f["forager"][1] and f["curious"][3] > f["balanced"][3])),
]
def allergy_check(episodes=4):
    """Allergic NPCs: ticks eating their poisonous look vs the other look (greedy, trained)."""
    pois = other = 0.0
    w = tr.world
    orig = w.step
    acc = dict(p=0.0, o=0.0)

    def step(a, f):
        o, r = orig(a, f)
        eb = w.eating_b
        d_ap = torch.cdist(w.pos, w.ppos)
        at = d_ap < w.c.eat_radius
        ea = (at & (w.real & (w.stock > 0))[:, None, :] & ~w.look_b[:, None, :]).any(-1)
        acc["p"] += float(((eb & w.allergic_b) | (ea & w.allergic_a)).sum())
        acc["o"] += float(((ea & w.allergic_b) | (eb & w.allergic_a)).sum())
        return o, r
    w.step = step
    for _ in range(episodes):
        tr.collect(greedy=True)
    w.step = orig
    return acc["p"], acc["o"]


pz, ot = allergy_check()
checks.append((f"P5 allergic NPCs avoid their poisonous look (eating ticks: poisonous {pz:.0f} vs other {ot:.0f})", pz < ot))
lines += ["", "CRITERIA (fixed before running)"] + [f"   {'PASS' if ok else 'FAIL'}  {lab}" for lab, ok in checks]

fig, axes = plt.subplots(1, 5, figsize=(22, 3.6))
its = [h["it"] for h in hist]
axes[0].plot(its, [h["score"] for h in hist])
axes[0].set_title("training score (own preferences)")
for k, g in enumerate(GOALS):
    axes[k + 1].plot(its, [h["returns"][k] for h in hist])
    axes[k + 1].set_title(f"mean episode return: {g}")
for ax in axes:
    ax.set_xlabel("iteration")
save(fig, OUT, "training.png")
write_summary(OUT, lines)
