"""Long-term memory of rewarding behaviour sequences, recalled as attractors.

What is stored
    When the agent's pain (affect.py) drops sharply (relief) or turns into
    clear pleasure (reward), the preceding behaviour sequence (its drive state C
    sampled seq_len times) is stored together with the sense data that was
    present when the sequence began.  Relief memories are *aversive* (escape),
    reward memories are *appetitive* (approach).

How it is recalled
    Every tick the short-term memory (STM, the last stm_len sense vectors) is
    compared with the stored sense keys.  A modern-Hopfield attractor settles
    from the best-matching STM entry into one memory's basin; if that memory's
    key is approximately present in STM (activation above threshold) its
    behaviour sequence is replayed.  Replay forces the drives through a
    generator in the plane C ^ C_target (a rotation, or a boost if the target is
    across the light cone), so memory acts like any other world channel.

Memory Clifford space
    Stored memories (key + sequence, padded to vec_dim) span a space whose
    valence-weighted second moment  G = sum_m v_m x_m x_m^T  (v > 0 appetitive,
    v < 0 aversive) is an indefinite symmetric form.  Its leading eigenvectors are
    the memory basis; the signs of their eigenvalues give the signature, so the
    basis spans Cl(p, q) with p approach modes and q escape modes, mirroring the
    drive algebra (spacelike homeostatic drives vs timelike threat).

Layout
    Every buffer length and struct size is a power of two, and every buffer is
    allocated on a 64-byte boundary.  A memory slot is one 512-byte record.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import LTMConfig

APPETITIVE, AVERSIVE = 1, -1
ALIGN = 64


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def aligned_zeros(shape, dtype, align: int = ALIGN) -> np.ndarray:
    """Zeroed array whose data pointer is a multiple of `align` bytes."""
    dtype = np.dtype(dtype)
    n = int(np.prod(shape)) * dtype.itemsize
    raw = np.zeros(n + align, dtype=np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off:off + n].view(dtype).reshape(shape)


def memory_dtype(cfg: LTMConfig) -> np.dtype:
    """One memory slot, padded to a power-of-two record size."""
    base = np.dtype([
        ("key", "<f4", (cfg.sense_dim,)),
        ("seq", "<f4", (cfg.seq_len, 4)),
        ("modes", "<f4", (cfg.n_modes,)),
        ("valence", "<f4"),
        ("strength", "<f4"),
        ("t", "<f4"),
        ("uses", "<f4"),
        ("kind", "<i4"),
        ("active", "<i4"),
    ], align=True)
    return np.dtype({"names": base.names,
                     "formats": [base.fields[n][0] for n in base.names],
                     "offsets": [base.fields[n][1] for n in base.names],
                     "itemsize": next_pow2(base.itemsize)})


def check_pow2(cfg: LTMConfig) -> None:
    for name in ("sense_dim", "stm_len", "seq_len", "hist_len", "stride", "n_slots",
                 "n_modes", "vec_dim"):
        if not is_pow2(getattr(cfg, name)):
            raise ValueError(f"LTMConfig.{name} must be a power of two")
    if cfg.hist_len < cfg.seq_len:
        raise ValueError("hist_len must be >= seq_len")
    if cfg.sense_dim + 4 * cfg.seq_len > cfg.vec_dim:
        raise ValueError("vec_dim too small for key + sequence")


# ---------------------------------------------------------------------------
# Memory Clifford space
# ---------------------------------------------------------------------------

@dataclass
class MemorySpace:
    basis: np.ndarray        # (k, vec_dim) orthonormal rows
    eigenvalues: np.ndarray  # (k,) signed: > 0 approach mode, < 0 escape mode
    signs: np.ndarray        # (k,) +1 / -1: the metric diag of Cl(p, q)

    @property
    def signature(self) -> tuple[int, int]:
        return int((self.signs > 0).sum()), int((self.signs < 0).sum())

    def coords(self, x: np.ndarray) -> np.ndarray:
        return self.basis @ x

    def interval(self, y: np.ndarray) -> float:
        """y^T diag(signs) y: > 0 approach-dominated, < 0 escape-dominated."""
        return float(y @ (self.signs * y))

    def plane_type(self, i: int, j: int) -> str:
        """(e_i ^ e_j)^2 = -s_i s_j: same signs elliptic, opposite hyperbolic."""
        return "elliptic" if self.signs[i] * self.signs[j] > 0 else "hyperbolic"


# ---------------------------------------------------------------------------
# Sequence memory
# ---------------------------------------------------------------------------

@dataclass
class SequenceMemory:
    cfg: LTMConfig = field(default_factory=LTMConfig)

    def __post_init__(self):
        c = self.cfg
        check_pow2(c)
        f4 = np.float32
        self.stm = aligned_zeros((c.stm_len, c.sense_dim), f4)
        self.beh = aligned_zeros((c.hist_len, 4), f4)
        self.pain = aligned_zeros((c.hist_len,), f4)
        self.sense_hist = aligned_zeros((c.hist_len, c.sense_dim), f4)
        self.slots = aligned_zeros((c.n_slots,), memory_dtype(c))
        self.stm_n = self.hist_n = 0
        self.tick = 0
        self.cool = 0
        self.replay_slot = -1
        self.replay_step = 0
        self.activation = 0.0
        self.t = 0.0
        self.events: list[dict] = []      # stored / merged / recalled records
        self.space: MemorySpace | None = None

    # --- ring helpers --------------------------------------------------------
    def _chrono(self, buf: np.ndarray, n_filled: int, count: int) -> np.ndarray:
        """Last `count` entries of a ring buffer, oldest first."""
        L = len(buf)
        end = n_filled % L
        idx = (np.arange(end - count, end)) % L
        return buf[idx]

    # --- observe -------------------------------------------------------------
    def observe(self, sense: np.ndarray, C: np.ndarray, pain: float, t: float) -> dict | None:
        """Record one cognitive tick; every `stride` ticks also sample behaviour,
        and store a sequence if it ended in relief or reward."""
        c = self.cfg
        self.t = t
        self.stm[self.stm_n % c.stm_len] = sense
        self.stm_n += 1
        self.tick += 1
        if self.tick % c.stride:
            return None
        k = self.hist_n % c.hist_len
        self.beh[k] = C
        self.pain[k] = pain
        self.sense_hist[k] = sense
        self.hist_n += 1
        act = self.slots["active"] == 1
        self.slots["strength"][act] *= (1.0 - c.decay)
        if self.cool:
            self.cool -= 1
            return None
        if self.hist_n < c.seq_len:
            return None
        pains = self._chrono(self.pain, self.hist_n, c.seq_len)
        # relief: pain removed (only the positive part counts); reward: pleasure gained.
        # Loss aversion: the pain side weighs loss_aversion times as much.
        relief = float(max(pains[:-1].max(), 0.0) - max(pains[-1], 0.0))
        reward = float(max(-pains[-1], 0.0)) if pains[0] - pains[-1] > 0.5 * c.form_threshold else 0.0
        w_relief, w_reward = c.loss_aversion * relief, reward
        if max(w_relief, w_reward) < c.form_threshold:
            return None
        kind, val = (AVERSIVE, w_relief) if w_relief >= w_reward else (APPETITIVE, w_reward)
        seq = self._chrono(self.beh, self.hist_n, c.seq_len)
        key = self._chrono(self.sense_hist, self.hist_n, c.seq_len)[0]
        self.cool = c.cooldown
        return self.store(key, seq, kind, val)

    # --- store ---------------------------------------------------------------
    def store(self, key: np.ndarray, seq: np.ndarray, kind: int, valence: float) -> dict:
        """valence: the *weighted* valence (loss aversion already applied); it
        becomes the memory's strength."""
        c = self.cfg
        s = self.slots
        act = np.nonzero(s["active"] == 1)[0]
        if len(act):
            dk = np.linalg.norm(s["key"][act] - key, axis=1)
            ds = np.sqrt(np.mean((s["seq"][act] - seq) ** 2, axis=(1, 2)))
            same = (dk < c.merge_radius) & (ds < 2 * c.merge_radius) & (s["kind"][act] == kind)
            if same.any():
                j = act[np.argmin(np.where(same, dk, np.inf))]
                w0, w1 = float(s["strength"][j]), valence
                s["key"][j] = (w0 * s["key"][j] + w1 * key) / (w0 + w1)
                s["seq"][j] = (w0 * s["seq"][j] + w1 * seq) / (w0 + w1)
                s["strength"][j] = w0 + w1
                s["valence"][j] = max(float(s["valence"][j]), valence)
                s["t"][j] = self.t
                ev = dict(t=self.t, what="merged", slot=int(j), kind=kind, valence=valence)
                self.events.append(ev)
                return ev
        free = np.nonzero(s["active"] == 0)[0]
        j = int(free[0]) if len(free) else int(np.argmin(s["strength"]))
        s[j] = 0
        s["key"][j] = key
        s["seq"][j] = seq
        s["valence"][j] = valence
        s["strength"][j] = valence
        s["t"][j] = self.t
        s["kind"][j] = kind
        s["active"][j] = 1
        ev = dict(t=self.t, what="stored", slot=j, kind=kind, valence=valence,
                  evicted=not len(free))
        self.events.append(ev)
        return ev

    # --- recall --------------------------------------------------------------
    def recall(self) -> np.ndarray | None:
        """Continue an ongoing replay, or start one if a stored key is
        approximately present in STM.  Returns the current target drive state."""
        c = self.cfg
        s = self.slots
        if self.replay_slot >= 0:
            j, k = self.replay_slot, self.replay_step // c.stride
            if k >= c.seq_len:
                self.replay_slot = -1
                self.activation = 0.0
            else:
                self.replay_step += 1
                return s["seq"][j][k].astype(float)
        act = np.nonzero(s["active"] == 1)[0]
        n_stm = min(self.stm_n, c.stm_len)
        if len(act) == 0 or n_stm == 0:
            self.activation = 0.0
            return None
        stm = self.stm[:n_stm]
        keys = s["key"][act]
        # loss aversion: escape memories tolerate loss_aversion x the squared mismatch
        tol = np.where(s["kind"][act] == AVERSIVE, c.loss_aversion, 1.0) * c.match_radius ** 2
        d2 = ((stm[:, None, :] - keys[None, :, :]) ** 2).sum(-1) / tol
        support = np.exp(-d2.min(axis=0))              # how present each key is in STM
        cue = stm[np.unravel_index(np.argmin(d2), d2.shape)[0]].astype(float)
        logw = np.log(np.maximum(s["strength"][act], 1e-6))
        q = cue
        for _ in range(c.hopfield_iters):              # settle into a basin
            e = -((keys - q) ** 2).sum(-1) / tol
            w = np.exp(c.hopfield_beta * e + logw - (c.hopfield_beta * e + logw).max())
            w /= w.sum()
            q = w @ keys
        win = int(np.argmax(w))
        self.activation = float(support[win])
        if self.activation < c.activation:
            return None
        j = int(act[win])
        s["uses"][j] += 1
        self.replay_slot, self.replay_step = j, 1
        self.events.append(dict(t=self.t, what="recalled", slot=j, kind=int(s["kind"][j]),
                                activation=self.activation))
        return s["seq"][j][0].astype(float)

    # --- memory Clifford space ----------------------------------------------
    def memory_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """(vectors (n, vec_dim) unit length, signed weights) of active memories."""
        c = self.cfg
        s = self.slots
        act = np.nonzero(s["active"] == 1)[0]
        X = aligned_zeros((len(act), c.vec_dim), np.float64)
        X[:, :c.sense_dim] = s["key"][act]
        X[:, c.sense_dim:c.sense_dim + 4 * c.seq_len] = s["seq"][act].reshape(len(act), 4 * c.seq_len)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        X /= np.maximum(n, 1e-12)
        v = s["kind"][act] * s["strength"][act]
        return X, v.astype(float)

    def build_space(self) -> MemorySpace | None:
        X, v = self.memory_vectors()
        if len(X) == 0:
            return None
        G = (X * v[:, None]).T @ X
        lam, U = np.linalg.eigh(G)
        order = np.argsort(-np.abs(lam))
        k = min(self.cfg.n_modes, int((np.abs(lam) > 1e-9 * np.abs(lam).max()).sum()))
        idx = order[:k]
        space = MemorySpace(U[:, idx].T.copy(), lam[idx].copy(), np.sign(lam[idx]))
        act = np.nonzero(self.slots["active"] == 1)[0]
        coords = X @ space.basis.T
        self.slots["modes"][act] = 0
        self.slots["modes"][act, :k] = coords
        self.space = space
        return space
