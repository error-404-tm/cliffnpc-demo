"""Interrogating the learning process.

Functions that ask an agent (or a whole arena) what it has learned and why:

    situations()                  canonical test situations (context vectors)
    drive_response(agent, x)      dC/dt the agent's learned connection produces in situation x
    courage(agent)                fight/flight push in company vs alone vs outnumbered
    response_table(agent)         every drive's learned push in every canonical situation
    connection_report(agent)      strongest learned (feature, plane) entries, per-feature norms,
                                  plane types under the agent's signature
    connection_holonomy(agent)    path dependence of the habits around a loop of situations
    metric_report(agent)          basis magnitudes, cross terms, deviation from temperament
                                  (diagonal vs off-diagonal), conditioning, felt-affect matrix
    affect_report(agent)          felt affect per trivector, share of affect from replayed memory
    memory_report(agent)          stored chains (readable), activations, what they primed
    ledger(agent)                 cumulative learning: metric diagonal / cross-term change,
                                  connection updates from pain vs reward, per-feature |dW|
    explain(agent)                all of the above as one readable text block
    population_snapshot(arena)    population means of the key learning indicators
    population_table(arena)       one row per living agent

All functions are read-only; nothing here changes the agents.
"""

from __future__ import annotations

import numpy as np

from . import algebra as ga
from .affect import AFFECT_NAMES, TrivectorAffect, felt_matrix
from .connection import FEATURES, PLANE_NAMES

DRIVES = ("hunger", "social", "explore", "fight")
REF_STATE = np.array([0.3, 0.0, 0.0, 0.3])   # mildly hungry, mildly ready to fight


def situations() -> dict[str, np.ndarray]:
    """Canonical situations as context vectors (FEATURES order)."""
    def ctx(food=0.0, rival_near=0.0, closing=0.0, allies=0, rivals=0, injured=0.0):
        x = np.zeros(len(FEATURES))
        x[:8] = [food, rival_near, closing, min(allies / 4, 1), min(rivals / 4, 1),
                 min(rivals / (1 + allies) / 2, 1), injured, min(rivals * allies / 4, 1)]
        return x
    return {
        "nothing": ctx(),
        "food, alone": ctx(food=0.8),
        "rival, alone": ctx(rival_near=0.6, closing=0.4, rivals=1),
        "rival, with 2 allies": ctx(rival_near=0.6, closing=0.4, rivals=1, allies=2),
        "outnumbered 3:1": ctx(rival_near=0.7, closing=0.5, rivals=3),
        "group 3 vs 1": ctx(rival_near=0.7, closing=0.5, rivals=1, allies=3),
        "rival at food, alone": ctx(food=0.8, rival_near=0.6, closing=0.3, rivals=1),
        "injured, rival near": ctx(rival_near=0.6, closing=0.4, rivals=1, injured=0.6),
    }


def drive_response(agent, x: np.ndarray, C: np.ndarray | None = None) -> np.ndarray:
    """dC/dt the learned connection produces in situation x, at state C (default: a
    fixed reference state, so agents are compared on equal terms)."""
    if agent.conn is None:
        return np.zeros(4)
    return agent.conn.response(x, REF_STATE if C is None else C, agent.npc.g)


def courage(agent, C: np.ndarray | None = None) -> dict:
    """Fight(+)/flight(-) push from the learned connection in key situations, and the
    courage index: push with allies minus push alone (> 0 = learned to fight together)."""
    S = situations()
    f = {k: float(drive_response(agent, S[k], C)[3]) for k in
         ("rival, alone", "rival, with 2 allies", "outnumbered 3:1", "group 3 vs 1")}
    f["courage_index"] = f["rival, with 2 allies"] - f["rival, alone"]
    f["numbers_index"] = f["group 3 vs 1"] - f["outnumbered 3:1"]
    return f


def response_table(agent, C: np.ndarray | None = None) -> dict:
    return {name: dict(zip(DRIVES, map(float, drive_response(agent, x, C))))
            for name, x in situations().items()}


def connection_report(agent, n: int = 6) -> dict:
    if agent.conn is None:
        return {}
    c = agent.conn
    eta = agent.npc.eta
    types = {pn: ga.basis_plane_type(i, j, eta) for pn, (i, j) in zip(PLANE_NAMES, ga.PLANES)}
    return dict(
        strongest=[(f, pn, round(v, 4), types[pn]) for f, pn, v in c.strongest(n)],
        feature_norms={f: float(np.linalg.norm(c.W[k])) for k, f in enumerate(FEATURES)},
        plane_norms={pn: float(np.linalg.norm(c.W[:, i, j])) for pn, (i, j) in
                     zip(PLANE_NAMES, ga.PLANES)},
        plane_types=types,
    )


def connection_holonomy(agent, loop: list[str] | None = None, dt: float = 1.0) -> dict:
    """Transport around a loop of situations (default: approach food, meet a rival,
    allies arrive, rival leaves, back to nothing).  ||H - I|| measures path dependence."""
    if agent.conn is None:
        return {}
    S = situations()
    loop = loop or ["nothing", "food, alone", "rival at food, alone", "rival, with 2 allies", "nothing"]
    H = agent.conn.holonomy([S[k] for k in loop], agent.npc.g, dt)
    return dict(loop=loop, deviation=float(np.linalg.norm(H - np.eye(4))),
                fight_row=[float(v) for v in H[3]])


def metric_report(agent) -> dict:
    npc = agent.npc
    g, g0, eta = npc.g, npc.g0, npc.eta
    d = g - g0
    off = d - np.diag(np.diag(d))
    return dict(
        basis_magnitudes={k: float(np.sqrt(abs(g[i, i]))) for i, k in enumerate(DRIVES)},
        start_magnitudes={k: float(np.sqrt(abs(g0[i, i]))) for i, k in enumerate(DRIVES)},
        cross_terms={f"{DRIVES[i]}.{DRIVES[j]}": float(g[i, j]) for i, j in ga.PLANES},
        diag_change=float(np.abs(np.diag(d)).sum()),
        offdiag_change=float(np.abs(off).sum() / 2),
        cond=float(np.linalg.cond(g)),
        signature=ga.signature(g)[:2],
        plane_types={f"{DRIVES[i]}^{DRIVES[j]}": ga.basis_plane_type(i, j, g) for i, j in ga.PLANES},
        felt_matrix=felt_matrix(g).round(3).tolist(),
    )


def affect_report(agent) -> dict:
    a = agent.affect
    out = dict(pain=float(a.pain), felt=dict(zip(AFFECT_NAMES, map(float, a.F)))
               if isinstance(a, TrivectorAffect) else {})
    if isinstance(a, TrivectorAffect):
        tot = a.src_real + a.src_primed
        out["primed_share"] = float(a.src_primed / tot) if tot > 0 else 0.0
    return out


def memory_report(agent, n: int = 5, recent: int = 5) -> dict:
    ep = agent.episodic
    if ep is None:
        return {}
    s = ep.slots
    act = np.nonzero(s["active"] == 1)[0]
    top = act[np.argsort(-s["strength"][act])][:n]
    return dict(summary=ep.summary(), top=[ep.describe(int(j)) for j in top],
                recent_activations=ep.activations[-recent:])


def ledger(agent) -> dict:
    L = dict(agent.ledger)
    if agent.conn is not None:
        c = agent.conn
        L.update(conn_updates=c.updates, conn_pain_updates=c.pain_updates,
                 conn_reward_updates=c.reward_updates,
                 conn_dW_by_feature={f: float(v) for f, v in zip(FEATURES, c.dW_abs)})
    return L


def explain(agent) -> str:
    lines = [f"== {agent.name} (tribe {agent.faction}, gen {agent.generation}, "
             f"health {agent.health:.2f})"]
    m = metric_report(agent)
    lines.append("basis magnitudes " + "  ".join(f"{k} {v:.2f} (start {m['start_magnitudes'][k]:.2f})"
                                                 for k, v in m["basis_magnitudes"].items()))
    lines.append("cross terms      " + "  ".join(f"{k} {v:+.2f}" for k, v in m["cross_terms"].items()))
    lines.append(f"metric change    diagonal {m['diag_change']:.3f}, cross terms {m['offdiag_change']:.3f}, "
                 f"cond {m['cond']:.1f}")
    if agent.conn is not None:
        c = courage(agent)
        lines.append("connection       " + "  ".join(f"{k}: {v:+.3f}" for k, v in c.items()))
        for f, pn, v, ty in connection_report(agent)["strongest"]:
            lines.append(f"   {f:>14s} -> {pn:<16s} {v:+.3f}  ({ty})")
        hol = connection_holonomy(agent)
        lines.append(f"habit holonomy   ||H - I|| = {hol['deviation']:.3f} around {' > '.join(hol['loop'])}")
    a = affect_report(agent)
    if a.get("felt"):
        lines.append("felt affect      " + "  ".join(f"{k} {v:+.2f}" for k, v in a["felt"].items())
                     + f"   (from memory {a.get('primed_share', 0):.0%})")
    mem = memory_report(agent)
    if mem:
        lines.append(f"memories         {mem['summary']}")
        for d in mem["top"]:
            lines.append("   " + d)
    lines.append(f"ledger           {ledger(agent)}")
    return "\n".join(lines)


def population_snapshot(arena) -> dict:
    ags = arena.agents
    if not ags:
        return dict(t=arena.t, n=0)
    cour = [courage(a) for a in ags if a.conn is not None]
    mets = [metric_report(a) for a in ags]
    affs = [affect_report(a) for a in ags]
    snap = dict(
        t=arena.t, n=len(ags),
        basis=np.mean([[m["basis_magnitudes"][k] for k in DRIVES] for m in mets], axis=0).tolist(),
        cross_abs=float(np.mean([np.mean(np.abs(list(m["cross_terms"].values()))) for m in mets])),
        diag_change=float(np.mean([m["diag_change"] for m in mets])),
        offdiag_change=float(np.mean([m["offdiag_change"] for m in mets])),
        pain=float(np.mean([a["pain"] for a in affs])),
        primed_share=float(np.mean([a.get("primed_share", 0.0) for a in affs])),
    )
    if cour:
        for k in ("courage_index", "numbers_index", "rival, alone", "rival, with 2 allies"):
            snap[k] = float(np.mean([c[k] for c in cour]))
        snap["W_norm"] = float(np.mean([np.linalg.norm(a.conn.W) for a in ags if a.conn is not None]))
        snap["W_feature"] = np.mean([[np.linalg.norm(a.conn.W[k]) for k in range(len(FEATURES))]
                                     for a in ags if a.conn is not None], axis=0).tolist()
    eps = [a.episodic for a in ags if a.episodic is not None]
    if eps:
        snap["memories"] = float(np.mean([e.summary()["memories"] for e in eps]))
        snap["activations"] = float(np.sum([len(e.activations) for e in eps]))
    return snap


def population_table(arena) -> list[dict]:
    rows = []
    for a in arena.agents:
        r = dict(name=a.name, tribe=a.faction, gen=a.generation, health=round(a.health, 2))
        if a.conn is not None:
            r.update({k: round(v, 3) for k, v in courage(a).items()})
        r.update({f"|{k}|": round(v, 2) for k, v in metric_report(a)["basis_magnitudes"].items()})
        rows.append(r)
    return rows
