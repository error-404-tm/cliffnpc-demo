# cliffnpc: Clifford-algebra NPC cognition, first demo

An experimental demo of the NPC-cognition half of the Clifford-algebra game
engine described in `../clifford-npc-spec.md`. The NPC's drives live in
**Cl(3,1)** (hunger, social and explore are spacelike; threat is timelike). Its
personality is its own metric `g = Lᵀ η L`. World events push it through
bivector generators. Everything is plain NumPy/SciPy with 4×4 matrices, runs
headless, and saves its plots to files.

```
cliffnpc/
  algebra.py    metric/frames, bivector generators, plane types, rotor step + rapidity clamp,
                trivectors (frustration), Gram minors / sensitivities
  cognition.py  NPC state, dynamics step, readouts, prediction/residuals, metric learning
  world.py      2D world: food, home, peers, periodically attacking threat, haunted disc
  coupling.py   sensor -> invariants -> low-rank coupling channels (Omega_ext)
  steering.py   cognition -> steering readout (isolated, swappable)
  sim.py        single-NPC main loop (multirate, optional surprise gating), logging
  memory.py     Phase 3: episodes, place rotors, collective metric, LOD transitions, gossip
  village.py    Phase 3: many tiered NPCs, social graph
  viewer.py     live viewer (matplotlib window, or headless GIF/MP4 render)
  config.py     every tunable constant
experiments/    e1..e6, each writes experiments/out/<name>/{*.png, summary.txt}
tests/          39 tests (Phase 1 geometry, Phase 3 memory, viewer smoke test)
```

## Running it

```bash
cd /home/joe/projects/cliffnpc-demo
.venv/bin/python -m pytest -q          # 39 tests, ~8 s
experiments/run_all.sh                 # all experiments; e6 takes ~4 min
.venv/bin/python experiments/e2_lightcone.py   # or one at a time
```

The venv was created with `python3 -m venv .venv && .venv/bin/pip install numpy
scipy matplotlib pytest` (Python 3.14, NumPy 2.5, SciPy 1.18).

## Live viewer

```bash
.venv/bin/python -m cliffnpc.viewer                    # interactive window (needs a display)
.venv/bin/python -m cliffnpc.viewer --save run.gif --seconds 60 --threat-at 20   # headless
```

Panels:
* **World:** the NPC is coloured by regime (blue deliberative, red reactive, grey
  null), with its sensor radius, trail, food, home, peers, the haunted disc, and the
  threat (large while attacking).
* **Drives:** each drive's coefficient ("does") next to its projection ("feels").
* **Light cone:** the state in the NPC's own orthonormal frame, so
  s = x² − y² exactly. x is the size of the hunger/social/explore part and y is threat.
* **Learned personality:** g − η as a heatmap, with the numbers in the info panel.
* **Strip chart:** the last 60 s of s/|C|² (reactive stretches shaded) and frustration |C∧B|.

Controls: `space` pause, `t` threat attacks now, `h` haunted disc on/off, `r` reset,
`+`/`-` speed, `q` quit, left click moves the food, right click moves home.

It uses only matplotlib (TkAgg). The interactive mode blits: static parts are
drawn once, and only the moving artists are redrawn each frame. Scale changes are
quantized so full redraws stay rare. Text is by far the slowest thing to draw
here (~30 ms per block), so the two text panels re-render every 10th frame and
their cached pixels are pasted in between. That gives about 18 fps on this
machine (a full redraw is ~400 ms). `--save` renders every frame in full, so a
60 s GIF takes a few minutes.

## Conventions (fixed in `algebra.py`)

* A drive vector is its coefficient array `C = (C¹..C⁴)` in the semantic basis.
  The **coefficient** readout (`C·eⁱ`, which drives action) is `C` itself. The
  **projection** readout (`C·eᵢ`, what the NPC feels) is `g @ C`.
* `L` maps semantic coordinates to an orthonormal frame: `C_o = L C`.
* Generator of `a∧b`: `v ↦ (v·a) b − (v·b) a`, i.e. `M = b aᵀg − a bᵀg`; for a general
  bivector array β, `M = −β g`. With this sign, `exp(tM(e1,e3))` runs
  hungry → explore → sated → home, and `M²` restricted to the plane equals `(a∧b)²`
  (both are tested).
* Rotor steps are exact matrix exponentials. The rapidity clamp is applied to the
  boost block in the orthonormal frame, so a clamped step is still an exact
  isometry.

## Decisions on the spec's OPEN points (all easy to change)

| Spec item | Choice | Where |
|---|---|---|
| Metric update `A(r)` (§5.2) | Read the sensitivity-weighted residual as a **covector** `c`, lowered with the current metric (`c = |r|·g r/|g r|`, the "felt" residual), then `A = ½ σ η v vᵀ` with `v = L⁻ᵀc` and `σ = sign(cᵀg⁻¹c)`. A is η-self-adjoint (no gauge part). To first order `g ← g + ε σ c cᵀ`, a Fisher/precision-like accumulation. Learning rate is scaled by `1 + |C∧B|` (frustration), and drive i is weighted by `|g^ii|`. | `cognition.salience_generator`, `felt_covector`, `learn_frame` |
| Decay toward `L₀` | `A = η(H − η)/2` with `H = L⁻ᵀ g₀ L⁻¹`. It is gauge-invariant, needs no `logm`, and gives `g ← g + λ(g₀ − g)` to first order. | `toward_metric_generator`, `forget_frame` |
| Signature safety | `L ← exp(εA) L` is always invertible, so signature (3,1) is preserved exactly and `det L` never crosses 0. In addition, σ makes `det(g + tσccᵀ)` grow in magnitude, so even the first-order target never changes signature. | tests |
| Steering readout | Linear blend of unit vectors weighted by the coefficients: hunger → food (seen, else remembered), social ± → toward/away from the peer centroid, explore + → wander, − → home, threat ± → approach/flee. Threat weight × `(1 + 3ρ)`, other drives × `(1 − 0.8ρ)`, where ρ = sigmoid of `−(s/|C|²)/0.1`. | `steering.py` |
| Frustration norm | Euclidean norm of the `C∧B` components in the orthonormal frame `L`. The Lorentz-invariant signed square is logged too (it is indefinite, so null frustration would read as zero). | `algebra.trivector_norms` |
| Regime | Classified on the normalized interval `s/|C|²`, so a small resting state isn't reported as "null" just for being small. Raw `s` is logged too. | `algebra.normalized_interval` |
| Haunted-disc connection | Vortex gauge: `a = Φ/2(−y,x)` inside, `ΦR²/(2r²)(−y,x)` outside. Line integrals are computed exactly per segment (swept area inside, polar-angle change outside), so holonomy is exactly topological. | `world.HauntedDisc` |
| Prediction | `"ema"` (leaky persistence, default) or `"cv"` (constant velocity). Scalar invariants live in the additive group, so `X̂ ← exp(η log R) X̂` reduces to `X̂ + ηR`. | `NPC.drive_residual` |
| Gossip `log(L_j L_i⁻¹)` | Replaced by the gauge-invariant first-order generator `toward_metric_generator(L_i, g_j)`. A Lorentz gauge change of `L_j` would otherwise show up as fake disagreement, and `logm` is ill-conditioned. | `memory.gossip_step` |
| Place rotors | One angle per grid cell (there is a single haunted bivector), with LRU eviction and a small replay on re-entry. | `memory.PlaceMemory` |

Spec additions I had to make: an **eating** term (`dC¹ −= rate·(C¹ − sated)·dt` at the
food; affine like hunger drift), **per-drive damping rates**, a remembered food
location for steering, a deadband on threat closing speed, and a threat that
retreats after contact.

## Results (numbers from `experiments/out/*/summary.txt`)

**Phase 1 tests.** All required tests pass: `Mᵀg + gM = 0` for random (3,1)
metrics; rotor steps (clamped or not) preserve `s` to 1e-9; plane classification,
including the g23 threshold; metric updates preserve (3,1); so(η) updates are pure
gauge; swapping updates changes L; the cofactor identity; outside-loop holonomy =
identity.

**E1 Foraging.** Yes, a limit cycle emerges, but **only through the world loop**.
Rotation + drift + damping is an affine linear ODE, so in drive space alone it
can only spiral into a fixed point; the open-loop run shows exactly that. Closed
loop: period 43.4 ± 1.7 s, hunger amplitude 1.72. A different initial state
converges to the same cycle (44.4 s, 1.79).

**E2 Light-cone snap.** Every threat approach (4 of 4) carries the state from
s > 0 to s < 0 a few seconds before contact. Steering becomes threat-dominated
(this temperament fights) and stays reactive for 16–24 s. **Correction to §2.3:**
boosts are isometries and can never cross the cone on their own (`|Δs| = 1e-13`
under a pure boost). The crossing needs a non-isometric partner, here per-drive
damping with γ_explore > γ_threat ("fear lingers longer than curiosity"). That
makes the dominant eigenvector of (boost − Γ) timelike. A second, unplanned path
showed up: **transits**, where eating sweeps hunger through zero while fear is
still decaying and the state briefly snaps.

**E3 Order dependence.** The final g depends on event order: ‖Δg‖ is 84% of the
surviving change with forgetting on. That is mostly **recency**, though.
Without forgetting the path dependence is 1.5–1.8%, and it comes from the
state-dependent scales (|g^ii| sensitivities, frustration). With those off it
drops to 3e-5. **Finding for §5.2:** a Lie-group update `L ← exp(εA)L` whose image
in g doesn't depend on g induces an *additive* flow on g, so non-commutativity
only appears at O(ε²). Real "trauma then comfort ≠ comfort then trauma" needs a
generator that depends on the current geometry at first order, or large
single-event updates.

**E4 Holonomy.** Exactly topological: an enclosing loop gives Φπ R² = 0.804 rad
(46.08°) whatever its shape, and an outside loop gives exactly 0. With dynamics
on, damping undoes the imprint in the drive values within ~4 s. But the NPC
is on the foraging limit cycle, whose phase is neutrally stable, so the haunted
loops leave a **lasting ~2 s shift in meal times**. Memory of place persists in
behaviour timing, not in the drives.

**E5 Personality threshold.** e2∧e3 is elliptic for |g23| < 1, parabolic at ±1 and
hyperbolic beyond. The growth rate matches √(g23² − 1) to 4e-16. **Signature
caveat:** with unit diagonal and threat ⟂ social/explore, |g23| > 1 forces
signature (2,2). To stay in (3,1), threat has to lean into the social–explore
plane (here g24 = 1.2). A parked NPC among peers shows a bounded tradeoff at
g23 = 0.5, and at 1.3 a runaway that damping can't hold (it saturates at the
safety cap).

**E6 Village (Phase 3).** 30 villagers (3 episodic, 9 metric, 18 background) on a
small-world graph. The attack is sensed by 3 villagers, and the event's footprint
is measured against a no-event twin run. With gossip it moves outward hop by hop:
the time to half of peak is 5, 17, 25, 43 and ~75 s at distances 0–5. The size
falls sharply after one hop (peak 0.15 at distance 0, 0.014 at distance 1, ~0.004
beyond, with only 1–5 individuals per distance). Beyond 2 hops the gossip signal is about the size of the behavioural
noise (without gossip, the attack still perturbs everyone's trajectories a
little). Gossip also *comforts* the victims: their footprint at t=170 s is 0.029
with gossip vs 0.083 without. Promotion instantiates from the current collective
metric (|g_new − g_col| = 0.03). Demotion consolidates 8 episodes chronologically.

**Surprise-gated compute.** With the `"cv"` predictor and `surprise_gate=0.05`, 44%
of cognitive ticks are skipped with identical meals and cone crossings over
600 s. At 0.1 the NPC misses snaps: a threat closing at constant speed is
*predictable*, and therefore skipped. The gate should not cover threat-closing.

## Known limitations and next steps

* Tuning is fragile. The foraging cycle and the snap both depend on the damping
  anisotropy and on eating, and all constants are in `config.py`.
* Learning is slow and nearly additive (E3). To get strong temporal memory, try a
  generator whose direction depends on g at first order, or event-sized updates.
* Place-memory replay exists, but no experiment demonstrates it persisting yet.
* A hyperbolic personality plane needs a real saturation law; the state cap is
  only a NaN guard.
* Not done: Cl(3,1,1) null-direction drift, reconstructive recall, a village
  mode for the viewer, porting to a GA library (Kingdon).
