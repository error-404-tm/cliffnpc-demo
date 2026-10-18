"""Episodic memory as event chains, recalled by partial match, replayed as weak,
fast re-experience that primes the affect vector.

Events
    Discrete things that happen to an agent (a rival comes into view, contact,
    an injury, a meal, a witnessed death, ...).  Each event type has a designed
    drive-space bivector and a valence (+ pain, - reward); the same bivectors
    the trivector affect model uses.  The live affect feels events as C ^ B.

Formation
    When pain drops sharply (relief, weighted by loss aversion) or turns into
    clear pleasure (reward), or the agent witnesses a death, the events of the
    last `window` seconds are stored as a chain, ending in an outcome event.

Activation (episodic route)
    The agent's recent events (a short-term event buffer) are matched against
    every contiguous segment of every stored chain.  If the last `min_match` (or
    more) recent events match a segment (same types, similar magnitude and
    context), the memory fires: "this is how it began last time".

Replay
    The rest of the chain -- what came next -- is replayed weakly
    (replay_strength) and fast (replay_speed times real time): its events are
    re-injected into the affect computation as if they happened again, at
    reduced strength.  Replay therefore primes affect (anticipatory fear,
    anticipatory reward); behaviour follows from affect through the contraction,
    never directly.

Layout: an event record is 64 bytes, a chain record 2048 bytes, all buffers on
64-byte boundaries.  Every activation is logged for interrogation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .affect import FORAGE_PLANE, SOCIAL_PLANE, THREAT_PLANES
from .config import EpisodicConfig
from .ltm import aligned_zeros, is_pow2, next_pow2

# event type -> (designed bivector, valence)
EVENT_TYPES = {
    "rival_seen":     (THREAT_PLANES, +0.3),
    "contact":        (THREAT_PLANES, +1.0),
    "injury":         (THREAT_PLANES, +1.0),
    "triumph":        (THREAT_PLANES, -1.0),
    "death_seen":     (THREAT_PLANES, +3.0),
    "food_seen":      (FORAGE_PLANE, -0.3),
    "meal":           (FORAGE_PLANE, -1.0),
    "false_food":     (FORAGE_PLANE, +0.5),
    "company":        (SOCIAL_PLANE, -0.5),
    "alone":          (SOCIAL_PLANE, +0.5),
    "relief":         (THREAT_PLANES * 0.0, -1.0),     # outcome markers (no bivector)
    "reward":         (FORAGE_PLANE * 0.0, -1.0),
}
TYPE_IDS = {name: k for k, name in enumerate(EVENT_TYPES)}
TYPE_NAMES = list(EVENT_TYPES)
APPETITIVE, AVERSIVE = 1, -1
CTX_DIM = 16                     # = len(connection.FEATURES); float16 keeps an event at 64 B


def event_dtype() -> np.dtype:
    base = np.dtype([("type", "<i4"), ("mag", "<f4"), ("dt", "<f4"), ("pad", "<f4"),
                     ("ctx", "<f2", (CTX_DIM,))], align=True)
    return np.dtype({"names": base.names, "formats": [base.fields[n][0] for n in base.names],
                     "offsets": [base.fields[n][1] for n in base.names],
                     "itemsize": next_pow2(base.itemsize)})


def chain_dtype(cfg: EpisodicConfig) -> np.dtype:
    base = np.dtype([("events", event_dtype(), (cfg.chain_len,)), ("length", "<i4"),
                     ("kind", "<i4"), ("origin", "<i4"), ("active", "<i4"),
                     ("strength", "<f4"), ("valence", "<f4"), ("t", "<f4"), ("uses", "<f4"),
                     ("last_fired", "<f4")], align=True)
    return np.dtype({"names": base.names, "formats": [base.fields[n][0] for n in base.names],
                     "offsets": [base.fields[n][1] for n in base.names],
                     "itemsize": next_pow2(base.itemsize)})


ORIGINS = ("experience", "imprinted", "inherited")


@dataclass
class EventMemory:
    cfg: EpisodicConfig = field(default_factory=EpisodicConfig)

    def __post_init__(self):
        c = self.cfg
        for name in ("n_slots", "chain_len", "buffer_len"):
            if not is_pow2(getattr(c, name)):
                raise ValueError(f"EpisodicConfig.{name} must be a power of two")
        self.buffer = aligned_zeros((c.buffer_len,), event_dtype())
        self.buf_t = aligned_zeros((c.buffer_len,), np.float64)    # absolute times
        self.n_events = 0
        self.slots = aligned_zeros((c.n_slots,), chain_dtype(c))
        self.t = 0.0
        self.replay_queue: list[tuple] = []   # (due time, type id, magnitude, memory slot)
        self.activations: list[dict] = []     # interrogation log
        self.stored: list[dict] = []
        self.pain_hist: list[tuple] = []      # (t, pain) for outcome detection
        self._cool = 0.0
        self._matched_at = -1                 # n_events at the last match attempt

    # --- short-term event buffer ------------------------------------------------
    def observe_event(self, name: str, magnitude: float, ctx: np.ndarray, t: float) -> None:
        k = self.n_events % self.cfg.buffer_len
        prev_t = self.buf_t[(self.n_events - 1) % self.cfg.buffer_len] if self.n_events else t
        self.buffer[k]["type"] = TYPE_IDS[name]
        self.buffer[k]["mag"] = magnitude
        self.buffer[k]["dt"] = t - prev_t
        self.buffer[k]["ctx"] = ctx
        self.buf_t[k] = t
        self.n_events += 1

    def recent(self, since: float | None = None, n: int | None = None) -> np.ndarray:
        """Recent events, oldest first (optionally only those after `since`)."""
        L = self.cfg.buffer_len
        m = min(self.n_events, L)
        idx = (np.arange(self.n_events - m, self.n_events)) % L
        ev, ts = self.buffer[idx], self.buf_t[idx]
        if since is not None:
            keep = ts >= since
            ev, ts = ev[keep], ts[keep]
        if n is not None:
            ev, ts = ev[-n:], ts[-n:]
        return ev

    # --- formation ------------------------------------------------------------
    def observe_pain(self, pain: float, t: float) -> dict | None:
        """Track pain; store the recent event chain on relief or reward."""
        c = self.cfg
        self.t = t
        self.pain_hist.append((t, pain))
        while self.pain_hist and self.pain_hist[0][0] < t - c.window:
            self.pain_hist.pop(0)
        if t < self._cool or len(self.pain_hist) < 4:
            return None
        ps = np.array([p for _, p in self.pain_hist])
        relief = max(ps[:-1].max(), 0.0) - max(ps[-1], 0.0)
        reward = max(-ps[-1], 0.0) if ps[0] - ps[-1] > 0 else 0.0
        w_relief, w_reward = c.loss_aversion * relief, reward
        if w_relief >= c.relief_threshold and w_relief >= w_reward:
            return self._form(AVERSIVE, w_relief, "relief")
        if w_reward >= c.reward_threshold:
            return self._form(APPETITIVE, w_reward, "reward")
        return None

    def imprint(self, valence: float, t: float) -> dict | None:
        """Forced memory (e.g. a witnessed death): the chain up to now."""
        self.t = t
        return self._form(AVERSIVE, valence, "relief", origin="imprinted")

    def _form(self, kind, strength, outcome, origin="experience") -> dict | None:
        c = self.cfg
        if not c.store:
            return None
        ev = self.recent(since=self.t - c.window)
        if len(ev) < 2:
            return None
        ev = ev[-(c.chain_len - 1):]
        chain = np.zeros(len(ev) + 1, dtype=event_dtype())
        chain[:-1] = ev
        chain[-1]["type"] = TYPE_IDS[outcome]
        chain[-1]["mag"] = strength
        chain[-1]["ctx"] = ev[-1]["ctx"]
        self._cool = self.t + 2.0
        return self.store(chain, kind, strength, origin)

    def store(self, chain: np.ndarray, kind: int, strength: float, origin: str = "experience") -> dict:
        c = self.cfg
        s = self.slots
        n = len(chain)
        act = np.nonzero(s["active"] == 1)[0]
        for j in act:                                     # merge with a near-identical chain
            if s["length"][j] == n and s["kind"][j] == kind and \
                    np.array_equal(s["events"][j]["type"][:n], chain["type"]) and \
                    self._chain_similarity(s["events"][j][:n], chain) >= c.merge_threshold:
                w0 = float(s["strength"][j])
                w = w0 + strength
                s["events"][j]["mag"][:n] = (w0 * s["events"][j]["mag"][:n] + strength * chain["mag"]) / w
                s["events"][j]["ctx"][:n] = (w0 * s["events"][j]["ctx"][:n].astype(np.float32)   # f16 storage
                                            + strength * chain["ctx"].astype(np.float32)) / w
                s["strength"][j] = w
                rec = dict(t=self.t, what="merged", slot=int(j), kind=kind, origin=origin,
                           types=[TYPE_NAMES[k] for k in chain["type"]])
                self.stored.append(rec)
                return rec
        free = np.nonzero(s["active"] == 0)[0]
        j = int(free[0]) if len(free) else int(np.argmin(s["strength"]))
        s[j] = 0
        s["events"][j][:n] = chain
        s["length"][j] = n
        s["kind"][j] = kind
        s["origin"][j] = ORIGINS.index(origin)
        s["strength"][j] = strength
        s["valence"][j] = strength * (-1 if kind == APPETITIVE else 1)
        s["t"][j] = self.t
        s["last_fired"][j] = -1e9
        s["active"][j] = 1
        rec = dict(t=self.t, what="stored", slot=j, kind=kind, origin=origin,
                   types=[TYPE_NAMES[k] for k in chain["type"]])
        self.stored.append(rec)
        return rec

    # --- matching ---------------------------------------------------------------
    def _event_similarity(self, a, b) -> float:
        if a["type"] != b["type"]:
            return 0.0
        m = min(a["mag"], b["mag"]) / max(a["mag"], b["mag"], 1e-9)
        d2 = float(np.sum((a["ctx"].astype(np.float32) - b["ctx"].astype(np.float32)) ** 2)) / self.cfg.context_scale ** 2
        return 0.5 * m + 0.5 * float(np.exp(-d2))

    def _chain_similarity(self, x, y) -> float:
        return float(np.mean([self._event_similarity(a, b) for a, b in zip(x, y)]))

    def match(self) -> tuple | None:
        """Best (slot, end index j, score, m) such that the last m recent events match
        chain[j-m+1 .. j] and something remains after j."""
        c = self.cfg
        recent = self.recent(n=c.chain_len)
        if len(recent) < c.min_match:
            return None
        s = self.slots
        best = None
        for slot in np.nonzero(s["active"] == 1)[0]:
            if self.t - s["last_fired"][slot] < c.refractory:
                continue
            n = int(s["length"][slot])
            ch = s["events"][slot][:n]
            for j in range(c.min_match - 1, n - 1):        # something must come after j
                m_max = min(j + 1, len(recent))
                for m in range(m_max, c.min_match - 1, -1):
                    seg, cur = ch[j - m + 1:j + 1], recent[-m:]
                    if not np.array_equal(seg["type"], cur["type"]):
                        continue
                    score = self._chain_similarity(seg, cur)
                    if score >= c.match_threshold:
                        key = (m, score * float(s["strength"][slot]))
                        if best is None or key > best[0]:
                            best = (key, int(slot), j, score, m)
                        break
        return None if best is None else best[1:]

    # --- activation & replay ------------------------------------------------------
    def step(self, t: float) -> list[tuple]:
        """Try to activate a memory; return the replayed events due now as
        (valence, bivector) pairs at replay strength, for the affect computation."""
        c = self.cfg
        self.t = t
        # a new event is the only thing that can create a new partial match
        hit = None
        if self.n_events != self._matched_at:
            self._matched_at = self.n_events
            hit = self.match()
        if hit is not None:
            slot, j, score, m = hit
            s = self.slots
            n = int(s["length"][slot])
            ch = s["events"][slot][:n]
            tt = t
            for e in ch[j + 1:]:                          # what came next, compressed in time
                tt += float(e["dt"]) / c.replay_speed
                self.replay_queue.append((tt, int(e["type"]), float(e["mag"]), slot))
            s["last_fired"][slot] = t
            s["uses"][slot] += 1
            self.activations.append(dict(
                t=t, slot=slot, origin=ORIGINS[int(s["origin"][slot])], kind=int(s["kind"][slot]),
                matched=[TYPE_NAMES[k] for k in ch[j - m + 1:j + 1]["type"]],
                primed=[TYPE_NAMES[k] for k in ch[j + 1:]["type"]], score=score))
        due = [q for q in self.replay_queue if q[0] <= t]
        self.replay_queue = [q for q in self.replay_queue if q[0] > t]
        act = self.slots["active"] == 1
        self.slots["strength"][act] *= (1.0 - c.decay)
        out = []
        for _, k, mag, _slot in due:
            B, v = EVENT_TYPES[TYPE_NAMES[k]]
            if np.any(B):
                out.append((np.sign(v), c.replay_strength * abs(v) * mag * B))
        return out

    # --- inheritance ----------------------------------------------------------------
    def inherit_from(self, parent: "EventMemory", k: int, factor: float) -> int:
        s = parent.slots
        act = np.nonzero(s["active"] == 1)[0]
        best = act[np.argsort(-s["strength"][act])][:k]
        for j in best:
            n = int(s["length"][j])
            self.store(s["events"][j][:n].copy(), int(s["kind"][j]),
                       float(s["strength"][j]) * factor, origin="inherited")
        return len(best)

    # --- interrogation -------------------------------------------------------------------
    def describe(self, slot: int) -> str:
        s = self.slots[slot]
        n = int(s["length"])
        steps = " -> ".join(f"{TYPE_NAMES[e['type']]}({e['mag']:.2f})" for e in s["events"][:n])
        kind = "reward" if s["kind"] == APPETITIVE else "relief/dread"
        return (f"[{slot}] {kind}, {ORIGINS[int(s['origin'])]}, strength {s['strength']:.2f}, "
                f"used {int(s['uses'])}x: {steps}")

    def summary(self) -> dict:
        s = self.slots
        act = s["active"] == 1
        return dict(
            memories=int(act.sum()),
            by_origin={o: int((act & (s["origin"] == k)).sum()) for k, o in enumerate(ORIGINS)},
            activations=len(self.activations),
            activations_by_origin={o: sum(a["origin"] == o for a in self.activations) for o in ORIGINS},
            mean_chain=float(s["length"][act].mean()) if act.any() else 0.0,
        )
