"""Memory layout for batched neural NPCs -- the design rules every module follows.

The hardware target is CPU (this machine: 12-core Ivy Bridge Xeon, AVX only, 32 KB L1d
and ~256 KB L2 per core).  Throughput on CPU is set by cache behaviour, not FLOPs, so:

1. Structure of arrays.  Every per-NPC quantity is ONE contiguous float32 tensor indexed
   by NPC slot; there are no Python objects per NPC.  Slots are world-major:
   slot = world * K + agent, so a world's agents are adjacent in memory and
   tensor.view(W, K, ...) and tensor.view(N, ...) are free reinterpretations.
2. Fixed capacities, powers of two.  Worlds W, agents per world K, patches per world P,
   features F, hidden units H are fixed at construction.  Nothing is resized during a
   run: death is a respawn into the same slot, a depleted patch is re-seeded in place.
3. Preallocate, then write in place.  Observation, hidden-state, scratch and rollout
   buffers are allocated once; the hot path uses in-place ops and out= arguments, so
   steady-state steps allocate (almost) nothing and the allocator never fragments.
4. Batch, and tile for the cache.  The brain runs one matrix multiply per layer for a
   whole tile of NPCs.  Its weights (~110 KB) stay resident in L2 while NPC rows stream
   past.  Tiles of ~1024 NPCs keep a tile's activations near cache size (measured best
   on this machine); larger batches spill and slow down.
5. Time-major rollouts, contiguous minibatches.  Rollout buffers are [T, N, ...]; PPO
   minibatches are contiguous blocks of NPC slots (a random permutation of blocks, not of
   individual slots), so every read is a run of adjacent rows.
6. Per-NPC adaptation is per-row.  Each NPC's adapter (a scale and shift on the hidden
   units) is a row of a contiguous [N, H] tensor, applied elementwise; one batched
   backward pass yields every NPC's adapter gradient at once.
7. Threads.  Intra-op threads hurt on tiny matrices here (12 threads ran ~10x slower than
   4); default 4.  Scaling past that is done by sharding worlds across processes, not
   threads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GOALS = ("food", "safety", "social", "knowledge")


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


@dataclass(frozen=True)
class Layout:
    worlds: int = 256               # W
    agents: int = 8                 # K per world (half tribe A, half tribe B)
    patches: int = 8                # P per world
    features: int = 32              # F, observation width
    hidden: int = 64                # H, recurrent units
    goals: int = 4                  # G
    episode: int = 256              # ticks per episode (0.2 s each)
    tile: int = 1024                # NPCs per compute tile

    def __post_init__(self):
        for name in ("worlds", "agents", "patches", "features", "hidden", "tile", "episode"):
            if not _pow2(getattr(self, name)):
                raise ValueError(f"{name} must be a power of two")
        if self.agents % 2:
            raise ValueError("agents per world must be even (two tribes)")

    @property
    def n(self) -> int:
        return self.worlds * self.agents

    def bytes_per_npc(self) -> int:
        """Persistent per-NPC state in the world and brain (float32 / bool), excluding
        rollout buffers: position, velocity, health, satiety, known bits per patch,
        observation, hidden state, adapters, preferences."""
        floats = 2 + 2 + 1 + 1 + self.patches + self.features + self.hidden + 2 * self.hidden + self.goals
        return 4 * floats


def set_threads(n: int = 4) -> None:
    torch.set_num_threads(n)
