"""Feature harness (spec-v2 section 8): measure what an environment dial does to the
induced geometry.

    DIALS                     registry: levels, base value, the goal relations a dial
                              is meant to touch (for the construct check)
    world_run(d, seed)        raw data for one world: a shared frontier panel of fixed
                              policies, plus the reference policy families for local
                              geometry; episodes are common-random-number aligned
    fingerprint(run)          extent, shape (smoothed and raw), local cosines and
                              magnitudes, record signature, live dimensions
    compare(run_a, run_b)     paired-bootstrap effects b - a, split into extent /
                              shape / local / new dimension, with construct-check flags
                              and paired-vs-unpaired interval widths

Effects are read as paired differences: the two worlds see the same patches,
noise and ambush draws, so differences come from the dial, not from luck.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .commons import GOALS, POLICIES, PRIMITIVES, SCALES, Commons, Dials, record_windows
from .geometry import (action_gradients, frontier_report, goal_metric, outcome_moment, outcome_weight,
                       signature, smooth_returns)

PAIRS = [(i, j) for i in range(len(GOALS)) for j in range(i + 1, len(GOALS))]
PAIR_NAMES = [f"{GOALS[i]}·{GOALS[j]}" for i, j in PAIRS]
BASE = Dials()


@dataclass
class DialSpec:
    levels: tuple
    targets: tuple                      # goal pairs the dial is meant to change
    note: str = ""


DIALS = {
    "skill": DialSpec((0.0, 0.5, 1.0), ("food·social",), "forager knows patch types"),
    "threat": DialSpec((0.0, 0.3, 0.7), ("food·safety", "safety·social", "safety·knowledge"), "ambush at patches"),
    "shelter": DialSpec((0.0, 0.5, 1.0), ("food·safety", "safety·social"), "rival-free disc at home"),
    "cue": DialSpec((0.0, 0.5, 1.0), ("food·knowledge", "safety·knowledge", "social·knowledge"), "looks tell types"),
    "scarcity": DialSpec((0.0, 0.5, 1.0), ("food·safety", "food·social", "food·knowledge"), "fewer, smaller patches"),
    "need": DialSpec((0.0, 0.5, 1.0), ("food·safety", "food·social", "food·knowledge"), "rung 1: hunger grows, starving hurts"),
}


@dataclass
class Sizes:
    n_frontier: int = 192               # policies in the shared frontier panel
    eps_frontier: int = 4               # episodes per frontier policy
    eps_local: int = 64                 # episodes per reference family
    families: tuple = tuple(POLICIES)


def frontier_panel(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fam = np.array([POLICIES[p]["mean"] for p in POLICIES], float)
    return np.vstack([fam, rng.uniform(-1.0, 2.0, (n - len(fam), len(PRIMITIVES)))])


def world_parts(sizes: Sizes = Sizes(), chunks: int = 4) -> list[str]:
    """Work units for one world (run in parallel, then merge_parts)."""
    return [f"frontier:{k}/{chunks}" for k in range(chunks)] + [f"local:{f}" for f in sizes.families]


def world_run(d: Dials, seed: int, sizes: Sizes = Sizes(), part: str = "all") -> dict:
    """Raw data for one world.  part: "all", "frontier", "frontier:k/K" (every K-th
    panel policy from k), or "local:<family>".  Same part, same draws in every world."""
    out = {}
    if part == "all" or part.startswith("frontier"):
        k, K = (0, 1) if ":" not in part else map(int, part.split(":")[1].split("/"))
        idx = np.arange(k, sizes.n_frontier, K)
        c = Commons(d, seed=seed * 1000 + k, noise_override=0.2)
        out["J_idx"] = idx
        out["J_part"] = c.run_policies(frontier_panel(sizes.n_frontier)[idx], sizes.eps_frontier)
        if K == 1:
            out["J"] = out.pop("J_part")
            out.pop("J_idx")
    for fam in sizes.families:
        if part in ("all", f"local:{fam}"):
            out[f"log:{fam}"] = Commons(d, seed=seed, policy=fam).run(sizes.eps_local)
    return out


def merge_parts(parts: list[dict], sizes: Sizes = Sizes()) -> dict:
    out = {}
    J = np.zeros((sizes.n_frontier, sizes.eps_frontier, len(GOALS)))
    for p in parts:
        if "J_part" in p:
            J[p["J_idx"]] = p["J_part"]
        out.update({k: v for k, v in p.items() if k.startswith("log:")})
    out["J"] = J
    return out


# ------------------------------------------------------------------ fingerprint
def _frontier_stats(J_eps: np.ndarray, theta: np.ndarray, shape_goals: np.ndarray | None = None) -> dict:
    """shape_goals: goals the frontier's shape is computed on (others held constant), so
    two worlds whose live dimensions differ are compared on the goals live in both."""
    Jm = J_eps.mean(1)
    Js = J_eps if shape_goals is None else np.where(shape_goals, J_eps, 0.0)
    sm = frontier_report(smooth_returns(theta, Js))
    raw = frontier_report(Js.mean(1))
    return dict(
        extent=Jm.max(0),                                   # best attainable (raw means)
        mean=Jm.mean(0),                                    # average over the panel
        spread=Jm.std(0),                                   # live dimension if > 0
        corr_s=np.array([sm["corr"][i, j] for i, j in PAIRS]),
        gap_s=np.array([sm["gaps"][i, j] for i, j in PAIRS]),
        corr_r=np.array([raw["corr"][i, j] for i, j in PAIRS]),
        gap_r=np.array([raw["gaps"][i, j] for i, j in PAIRS]),
    )


def _local_stats(log: dict) -> dict:
    out = {}
    for T in SCALES:
        W = record_windows(log, T)
        A = action_gradients(W)
        _, cos = goal_metric(A)
        out[T] = dict(cos=np.array([cos[i, j] for i, j in PAIRS]), mag=np.linalg.norm(A, axis=1))
    return out


def fingerprint(run: dict, sizes: Sizes = Sizes()) -> dict:
    theta = frontier_panel(sizes.n_frontier)
    fp = dict(frontier=_frontier_stats(run["J"], theta), local={})
    for fam in sizes.families:
        fp["local"][fam] = _local_stats(run[f"log:{fam}"])
    W = record_windows(run["log:broad"], 32)
    fp["signature"] = signature(outcome_moment(W, outcome_weight(W)))
    fp["live"] = fp["frontier"]["spread"] > 1e-9
    return fp


# ------------------------------------------------------------------ paired comparison
def _episode_subset(log: dict, eps: np.ndarray) -> dict:
    """The ticks of the chosen episodes, with episode ids made unique per draw."""
    parts = {k: [] for k in log}
    for n, e in enumerate(eps):
        m = log["episode"] == e
        for k, v in log.items():
            parts[k].append(np.full(m.sum(), n) if k == "episode" else v[m])
    return {k: np.concatenate(v) for k, v in parts.items()}


def _ci(x: np.ndarray) -> tuple:
    x = x[~np.isnan(x)] if np.ndim(x) == 1 else x
    return (np.nanpercentile(x, 2.5, axis=0), np.nanpercentile(x, 97.5, axis=0))


def compare(run_a: dict, run_b: dict, sizes: Sizes = Sizes(), n_boot: int = 100, seed: int = 0,
            targets: tuple = ()) -> dict:
    """Effects of going from world a to world b, with paired-bootstrap 95% intervals."""
    rng = np.random.default_rng(seed)
    theta = frontier_panel(sizes.n_frontier)
    live = (run_a["J"].mean(1).std(0) > 1e-9) & (run_b["J"].mean(1).std(0) > 1e-9)   # shape on shared goals
    fa, fb = _frontier_stats(run_a["J"], theta, live), _frontier_stats(run_b["J"], theta, live)
    fa["spread"], fb["spread"] = run_a["J"].mean(1).std(0), run_b["J"].mean(1).std(0)
    point = {k: fb[k] - fa[k] for k in fa}
    P = sizes.n_frontier
    boots = {k: [] for k in fa}
    unpaired = {"extent": [], "corr_s": []}
    for _ in range(n_boot):
        idx = rng.integers(0, P, P)                          # same policies in both worlds
        sa = _frontier_stats(run_a["J"][idx], theta[idx], live)
        sb = _frontier_stats(run_b["J"][idx], theta[idx], live)
        for k in fa:
            boots[k].append(sb[k] - sa[k])
        idx2 = rng.integers(0, P, P)                         # independent draws: unpaired width
        sb2 = _frontier_stats(run_b["J"][idx2], theta[idx2], live)
        unpaired["extent"].append(sb2["extent"] - sa["extent"])
        unpaired["corr_s"].append(sb2["corr_s"] - sa["corr_s"])
    fr = {k: dict(d=point[k], ci=_ci(np.array(boots[k]))) for k in fa}
    width = {k: (float(np.nanmean(np.subtract(*_ci(np.array(boots[k]))[::-1]))),
                 float(np.nanmean(np.subtract(*_ci(np.array(unpaired[k]))[::-1])))) for k in unpaired}

    local = {}
    n_eps = sizes.eps_local
    for fam in sizes.families:
        la, lb = run_a[f"log:{fam}"], run_b[f"log:{fam}"]
        A0, B0 = _local_stats(la), _local_stats(lb)
        draws = {T: {"cos": [], "mag": []} for T in SCALES}
        for _ in range(max(20, n_boot // 3)):
            eps = rng.integers(0, n_eps, n_eps)              # same episodes in both worlds
            sa, sb = _local_stats(_episode_subset(la, eps)), _local_stats(_episode_subset(lb, eps))
            for T in SCALES:
                draws[T]["cos"].append(sb[T]["cos"] - sa[T]["cos"])
                draws[T]["mag"].append(sb[T]["mag"] - sa[T]["mag"])
        local[fam] = {T: dict(dcos=B0[T]["cos"] - A0[T]["cos"], cos_ci=_ci(np.array(draws[T]["cos"])),
                              dmag=B0[T]["mag"] - A0[T]["mag"], mag_ci=_ci(np.array(draws[T]["mag"])),
                              cos_a=A0[T]["cos"], cos_b=B0[T]["cos"])
                      for T in SCALES}

    # ---- the four kinds of effect
    sig = lambda lo, hi: (np.nan_to_num(lo) > 0) | (np.nan_to_num(hi) < 0)  # noqa: E731
    extent = sig(*fr["extent"]["ci"]) | sig(*fr["mean"]["ci"])       # best or average attainable moved
    live_a, live_b = fa["spread"] > 1e-9, fb["spread"] > 1e-9
    new_dim = live_a != live_b
    gap_sig = sig(*fr["gap_s"]["ci"]) & (np.sign(fr["gap_s"]["d"]) == np.sign(fr["gap_r"]["d"]))
    corr_sig = sig(*fr["corr_s"]["ci"]) & (np.sign(fr["corr_s"]["d"]) == np.sign(fr["corr_r"]["d"]))
    shape = gap_sig | corr_sig
    loc_count = np.zeros(len(PAIRS), int)                     # families with a significant cosine change
    for fam in sizes.families:
        hit = np.zeros(len(PAIRS), bool)
        for T in SCALES:
            hit |= sig(*local[fam][T]["cos_ci"])
        loc_count += hit
    local_robust = loc_count >= 2
    touched = [PAIR_NAMES[k] for k in range(len(PAIRS)) if shape[k] or local_robust[k]]
    flags = [p for p in touched if targets and p not in targets]
    return dict(frontier=fr, local=local, extent=extent, new_dim=new_dim, shape=shape, gap_sig=gap_sig,
                corr_sig=corr_sig, local_count=loc_count, local_robust=local_robust, touched=touched,
                flags=flags, ci_width_paired_vs_unpaired=width, live_a=live_a, live_b=live_b)


def world_for(dial: str, level: float, base: Dials = BASE) -> Dials:
    return replace(base, **{dial: float(level)})


__all__ = ["DIALS", "BASE", "PAIRS", "PAIR_NAMES", "Sizes", "DialSpec", "frontier_panel", "world_run",
           "world_parts", "merge_parts",
           "fingerprint", "compare", "world_for"]
