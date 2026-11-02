"""A batched two-tribe world in PyTorch: W independent worlds stepped together.

Everything is structure-of-arrays (see layout.py).  Per world: K agents (tribe A in the
first half of the slots, tribe B in the second), P food patches.  Rules follow the
commons / arena ladder (rungs 0-8 in one world):

  food      patches hold stock, deplete and are re-seeded elsewhere; a fraction are
            false (look like food, give nothing); each patch has a look (A or B) that
            tells its type with probability (1 + cue) / 2
  need      satiety decays; eating real food raises it; starving (satiety < 0.3) is a
            food setback
  knowledge bits about patch types: a look carries 1 - H((1 + cue) / 2) bits when the patch
            is first seen, probing at the patch resolves the rest
  combat    agents of different tribes in contact (< 1.5) fight; a side of n against m
            takes m / (n + m) of the damage, shared among its n; agents with the fight flag
            within join radius 4 count toward their side; no damage inside one's own home
            disc (shelter); health 0 is death: a respawn at home (a big safety setback)
  social    an ally within peer radius 15

Every agent is controlled by the brain (all are learners; self-play between tribes).
Each tick produces a goal-progress vector per agent (food, safety, social, knowledge) and
an observation row of F = 32 features (OBS_NAMES).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .layout import Layout

OBS_NAMES = (
    "bias", "hunger", "health", "home_dx", "home_dy", "home_dist",
    "food_dx", "food_dy", "food_near", "food_look_a", "food_look_b", "food_known_false", "food_known_real",
    "ally_dx", "ally_dy", "ally_near", "allies_count",
    "enemy_dx", "enemy_dy", "enemy_near", "enemies_count", "in_contact",
    "pref_food", "pref_safety", "pref_social", "pref_knowledge", "hit",
    "food2_dx", "food2_dy", "food2_near", "food2_look_a", "food2_look_b",
)


@dataclass
class WorldConfig:
    size: float = 100.0
    dt: float = 0.2                    # s per tick
    speed: float = 5.0                 # max speed, units / s
    sensor: float = 30.0
    eat_radius: float = 2.0
    peer_radius: float = 15.0
    contact: float = 1.5
    join: float = 4.0
    damage_rate: float = 0.15          # health / s per fight cluster
    home_radius: float = 8.0
    stock: float = 6.0                 # per patch (~60 meal-seconds for one eater)
    take_rate: float = 0.1             # stock / s per eater
    false_frac: float = 0.5
    cue: float = 0.6                   # look validity: truthful with probability 0.8
    eat_gain: float = 0.2              # satiety / s
    satiety_decay: float = 0.05        # / s (rung-1 need at 1)
    starve_below: float = 0.3
    starve_rate: float = 0.05
    death_setback: float = 1.0         # safety progress at death
    patch_box: tuple = (10.0, 90.0, 10.0, 90.0)   # where patches spawn: x0, x1, y0, y1
    # per-NPC fragility (damage multiplier) is a world property set via VecWorld.fragility
    poison_rate: float = 0.1           # health / s while an allergic NPC eats a poisoned look
    nausea_rate: float = 0.2           # food setback / s while doing so


def contested() -> "WorldConfig":
    """Patches few, small and central, between the two homes: the tribes meet at food."""
    return WorldConfig(patch_box=(40.0, 60.0, 30.0, 70.0), stock=3.0)


def _h2(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))


class VecWorld:
    """W worlds x K agents x P patches, stepped as one batch."""

    def __init__(self, layout: Layout, cfg: WorldConfig = WorldConfig(), seed: int = 0):
        self.L, self.c = layout, cfg
        W, K, P = layout.worlds, layout.agents, layout.patches
        self.g = torch.Generator().manual_seed(seed)
        f32 = dict(dtype=torch.float32)
        # ---- agent state, world-major slots (views [W, K, ...] and [N, ...] share memory)
        self.pos = torch.zeros(W, K, 2, **f32)
        self.vel = torch.zeros(W, K, 2, **f32)
        self.health = torch.ones(W, K, **f32)
        self.satiety = torch.zeros(W, K, **f32)
        self.known = torch.zeros(W, K, P, **f32)          # bits known per patch
        self.seen = torch.zeros(W, K, P, dtype=torch.bool)
        self.hit = torch.zeros(W, K, **f32)
        self.pref = torch.zeros(W, K, layout.goals, **f32)
        self.fragility = torch.ones(W, K, **f32)         # per-NPC damage multiplier (unobserved)
        self.allergic_b = torch.zeros(W, K, dtype=torch.bool)   # per-NPC: look B is poison (unobserved)
        self.eating_b = torch.zeros(W, K, dtype=torch.bool)     # this tick: eating at a look-B patch
        tribe = torch.zeros(K, dtype=torch.bool)
        tribe[K // 2:] = True
        self.tribe = tribe                                 # False: A, True: B (static)
        self.same = tribe[:, None] == tribe[None, :]       # [K, K]
        self.eye = torch.eye(K, dtype=torch.bool)
        homes = torch.tensor([[20.0, 50.0], [80.0, 50.0]])
        self.home = homes[tribe.long()]                    # [K, 2]
        # ---- patches
        self.ppos = torch.zeros(W, P, 2, **f32)
        self.stock = torch.zeros(W, P, **f32)
        self.real = torch.zeros(W, P, dtype=torch.bool)
        self.look_b = torch.zeros(W, P, dtype=torch.bool)  # look A (False) or B (True)
        # ---- outputs (preallocated)
        self.obs = torch.zeros(W, K, layout.features, **f32)
        self.reward = torch.zeros(W, K, layout.goals, **f32)
        self.t = 0

    # ------------------------------------------------------------ helpers
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g)

    def _seed_patches(self, mask: torch.Tensor) -> None:
        """Re-seed the patches where mask [W, P] is set: new place, type, look, stock."""
        n = int(mask.sum())
        if n == 0:
            return
        c = self.c
        x0, x1, y0, y1 = c.patch_box
        xy = self._rand(n, 2)
        self.ppos[mask] = torch.stack([xy[:, 0] * (x1 - x0) + x0, xy[:, 1] * (y1 - y0) + y0], -1)
        real = self._rand(n) >= c.false_frac
        truthful = self._rand(n) < (1 + c.cue) / 2
        self.real[mask] = real
        self.look_b[mask] = ~real ^ ~truthful              # truthful: A real, B false; else swapped
        self.stock[mask] = c.stock
        m = mask[:, None, :].expand_as(self.known)
        self.known[m] = 0.0
        self.seen[m] = False

    def reset(self, pref: torch.Tensor | None = None) -> torch.Tensor:
        W, K = self.L.worlds, self.L.agents
        self.pos.copy_(self.home.expand(W, K, 2) + (self._rand(W, K, 2) - 0.5) * 6)
        self.vel.zero_()
        self.health.fill_(1.0)
        self.satiety.copy_(self._rand(W, K) * 0.5)
        self.hit.zero_()
        self._seed_patches(torch.ones(W, self.L.patches, dtype=torch.bool))
        if pref is not None:
            self.pref.copy_(pref)
        self.t = 0
        self._observe()
        return self.obs

    # ------------------------------------------------------------ step
    def step(self, act_vel: torch.Tensor, fight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """act_vel [W, K, 2] in [-1, 1] per axis (speed fraction), fight [W, K] bool.
        Returns (obs [W, K, F], reward [W, K, G]); both are preallocated buffers."""
        c, L = self.c, self.L
        dt = c.dt
        r = self.reward
        r.zero_()
        # move (norm <= 1), slower when hurt
        v = act_vel.clone()
        nrm = v.norm(dim=-1, keepdim=True).clamp_min(1.0)
        v.div_(nrm).mul_((c.speed * (0.5 + 0.5 * self.health)).unsqueeze(-1))
        self.vel.copy_(v)
        self.pos.add_(v * dt).clamp_(0.0, c.size)
        # food: eaters at real patches with stock
        d_ap = torch.cdist(self.pos, self.ppos)                       # [W, K, P]
        at = d_ap < c.eat_radius
        eatable = at & (self.real & (self.stock > 0))[:, None, :]
        at_b = (eatable & self.look_b[:, None, :]).any(-1)            # eating where the look is B
        self.eating_b.copy_(at_b)
        poisoned = at_b & self.allergic_b                             # allergic NPCs: no food, harm
        eating = eatable.any(-1) & ~poisoned                          # [W, K]
        gain = torch.minimum(1.0 - self.satiety, torch.full_like(self.satiety, c.eat_gain * dt)) * eating
        self.satiety.add_(gain)
        r[..., 0].add_(gain)
        if bool(poisoned.any()):
            pz = poisoned.float()
            self.health.sub_(c.poison_rate * dt * pz)
            r[..., 1].sub_(c.poison_rate * dt * pz)
            r[..., 0].sub_(c.nausea_rate * dt * pz)
        n_eaters = at.float().sum(1)                                  # [W, P] (false patches too: probing)
        self.stock.sub_(c.take_rate * dt * n_eaters * self.real)
        self.satiety.sub_(c.satiety_decay * dt).clamp_(0.0, 1.0)
        starving = (c.starve_below - self.satiety).clamp_min(0.0)
        r[..., 0].sub_(c.starve_rate * dt * starving)
        # knowledge: sight (a look's information), then probing resolves the rest
        visible = d_ap <= c.sensor
        new_seen = visible & ~self.seen
        sight_bits = float(1.0 - _h2(torch.tensor((1 + c.cue) / 2)))
        add = torch.minimum(new_seen.float() * sight_bits, 1.0 - self.known)
        self.known.add_(add)
        self.seen |= new_seen
        probe = at.float() * (1.0 - self.known)
        self.known.add_(probe)
        r[..., 3].add_(add.sum(-1) + probe.sum(-1))
        # combat
        d_aa = torch.cdist(self.pos, self.pos)                        # [W, K, K]
        enemy = ~self.same                                            # [K, K]
        home_safe = (self.pos - self.home).norm(dim=-1) < c.home_radius
        contact = (d_aa < c.contact) & enemy & ~self.eye
        in_contact = contact.any(-1)
        engaged_flag = fight | in_contact
        near = d_aa < c.join
        n_side = 1.0 + ((near & self.same & ~self.eye) & engaged_flag[:, None, :]).float().sum(-1)
        m_side = ((near & enemy) & engaged_flag[:, None, :]).float().sum(-1)
        engaged = (in_contact | (fight & (m_side > 0))) & ~home_safe
        dmg = c.damage_rate * dt * m_side / (n_side + m_side).clamp_min(1e-6) / n_side * engaged * self.fragility
        self.health.sub_(dmg)
        self.hit.copy_(dmg)
        r[..., 1].sub_(dmg)
        # death: respawn at home in the same slot
        dead = self.health <= 0.0
        if bool(dead.any()):
            r[..., 1].sub_(c.death_setback * dead)
            self.health[dead] = 1.0
            self.satiety[dead] = 0.5
            idx = dead.nonzero(as_tuple=True)
            self.pos[idx] = self.home[idx[1]]
        # social
        allies_near = ((d_aa < c.peer_radius) & self.same & ~self.eye).any(-1)
        r[..., 2].add_(allies_near.float() * dt)
        # depleted patches are re-seeded elsewhere
        self._seed_patches(self.real & (self.stock <= 0))
        self.t += 1
        self._observe(d_ap, d_aa, in_contact)
        return self.obs, r

    # ------------------------------------------------------------ observation
    def _observe(self, d_ap=None, d_aa=None, in_contact=None) -> None:
        c, o = self.c, self.obs
        W, K, P = self.L.worlds, self.L.agents, self.L.patches
        if d_ap is None:
            d_ap = torch.cdist(self.pos, self.ppos)
            d_aa = torch.cdist(self.pos, self.pos)
            in_contact = torch.zeros(W, K, dtype=torch.bool)
        R = c.sensor
        o.zero_()
        o[..., 0] = 1.0
        o[..., 1] = 1.0 - self.satiety
        o[..., 2] = self.health
        hv = self.home - self.pos
        hd = hv.norm(dim=-1, keepdim=True)
        o[..., 3:5] = hv / hd.clamp_min(1e-6)
        o[..., 5] = (hd.squeeze(-1) / 50.0).clamp_max(1.0)
        # two nearest visible patches
        dv = torch.where(d_ap <= R, d_ap, torch.full_like(d_ap, 1e9))
        vals, idx = dv.topk(2, dim=-1, largest=False)                  # [W, K, 2]
        for j, base in ((0, 6), (1, 27)):
            ok = vals[..., j] < 1e8
            pi = idx[..., j]
            pp = torch.gather(self.ppos, 1, pi.reshape(W, -1, 1).expand(-1, -1, 2)).reshape(W, K, 2)
            vec = pp - self.pos
            nrm = vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            o[..., base:base + 2] = vec / nrm * ok.unsqueeze(-1)
            o[..., base + 2] = (1.0 - vals[..., j] / R).clamp_min(0.0) * ok
            lb = torch.gather(self.look_b, 1, pi.reshape(W, -1)).reshape(W, K)
            o[..., base + 3] = (~lb & ok).float()
            o[..., base + 4] = (lb & ok).float()
            if j == 0:
                kn = torch.gather(self.known, 2, pi.unsqueeze(-1)).squeeze(-1)
                rl = torch.gather(self.real, 1, pi.reshape(W, -1)).reshape(W, K)
                full = (kn > 0.999) & ok
                o[..., 11] = (full & ~rl).float()
                o[..., 12] = (full & rl).float()
        # nearest ally and enemy
        for same, base, cnt in ((True, 13, 16), (False, 17, 20)):
            mask = (self.same if same else ~self.same) & ~self.eye
            dd = torch.where(mask & (d_aa <= R), d_aa, torch.full_like(d_aa, 1e9))
            v, i = dd.min(-1)
            ok = v < 1e8
            op = torch.gather(self.pos, 1, i.unsqueeze(-1).expand(-1, -1, 2))
            vec = op - self.pos
            o[..., base:base + 2] = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-6) * ok.unsqueeze(-1)
            o[..., base + 2] = (1.0 - v / R).clamp_min(0.0) * ok
            radius = c.peer_radius if same else R
            o[..., cnt] = ((mask & (d_aa <= radius)).float().sum(-1) / 4.0).clamp_max(1.0)
        o[..., 21] = in_contact.float()
        o[..., 22:26] = self.pref
        o[..., 26] = (self.hit * 50.0).clamp_max(1.0)
