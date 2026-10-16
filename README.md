# cliffnpc: Clifford-algebra NPC cognition, first demo

An experimental demo of the NPC-cognition half of the Clifford-algebra game
engine described in `../clifford-npc-spec.md`. The NPC's drives live in
**Cl(3,1)** (hunger, social and explore are spacelike; threat is timelike). Its
personality is its own frame `L`, with metric `g = Lᵀ η L`. World events push it
through bivector generators. Everything is plain NumPy/SciPy with 4×4 matrices,
runs headless, and saves its plots to files.

Beyond the spec's phases, the project now has:
* an **arena** where an NPC pack and an anti-social rival compete for food;
* a **pain/reward Clifford space** whose prediction error drives learning;
* a **frame law** that keeps the frame's orientation (the holonomy of experience)
  and stops personality from going singular;
* a **long-term sequence memory** that replays rewarding or relieving behaviour as
  attractors, with loss aversion and gating by present pain.

See the sections from [Frame law](#frame-law-option-b) on.

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
  affect.py     pain/reward space Cl(3,0), pain prediction error, pain-aversion learning
  arena.py      NPC pack + anti-social rival competing for food; combat physics, immobility
  ltm.py        long-term sequence memory: attractor recall, loss aversion, memory Clifford space
  config.py     every tunable constant
experiments/    e1..e6, e8, each writes experiments/out/<name>/{*.png, summary.txt}
tests/          67 tests
```

## Running it

```bash
cd /home/joe/projects/cliffnpc-demo
.venv/bin/python -m pytest -q          # 67 tests, ~20 s
experiments/run_all.sh                 # all experiments in parallel, ~3 min on 12 cores
.venv/bin/python experiments/e2_lightcone.py   # or one at a time
```

Runs never draw anything, so they go as fast as the CPU allows: the single NPC
runs ~100× real time, the 6-agent arena ~20×, the 30-agent village ~5×. The
bottleneck is Python/NumPy overhead on many tiny 4×4 operations, not arithmetic.

`run_all.sh` starts all experiments at once, and each experiment spreads its
independent runs (seeds, conditions) over worker processes (`experiments/common.py:
pmap`). Processes, not threads: this Python build has the GIL, and one simulation's
agents are too few to split. Every run is seeded, so parallel results are identical
to serial ones; checked for E3, E6 and E8. The full suite takes ~3 min instead of
~25. Set `CLIFFNPC_WORKERS=1` to run serially. Each experiment's output goes to
`experiments/out/<name>/run.log`.

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
* `L` maps semantic coordinates to the agent's orthonormal (experiential) frame:
  `C_o = L C`. Under the frame law the whole of `L` is physical, not just `g`.
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
| Learning in the arena | Pain aversion (see [Pain and reward](#pain-and-reward-affectpy)) instead of the salience law above. | `affect.py` |
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

All numbers are from the current model (frame law included); E1–E6 were re-run
after it.

**Phase 1 tests.** All required tests pass: `Mᵀg + gM = 0` for random (3,1)
metrics; rotor steps (clamped or not) preserve `s` to 1e-9; plane classification,
including the g23 threshold; metric updates preserve (3,1); so(η) updates are pure
gauge; swapping updates changes L; the cofactor identity; outside-loop holonomy =
identity.

**E1 Foraging.** Yes, a limit cycle emerges, but **only through the world loop**.
Rotation + drift + damping is an affine linear ODE, so in drive space alone it
can only spiral into a fixed point; the open-loop run shows exactly that. Closed
loop: period 44.0 ± 1.6 s, hunger amplitude 1.76. A different initial state
converges to the same cycle (43.5 s, 1.73).

**E2 Light-cone snap.** Every threat approach (4 of 4) carries the state from
s > 0 to s < 0 a few seconds before contact. Steering becomes threat-dominated
(this temperament fights) and stays reactive for 10–22 s. **Correction to §2.3:**
boosts are isometries and can never cross the cone on their own (`|Δs| = 1e-13`
under a pure boost). The crossing needs a non-isometric partner, here per-drive
damping with γ_explore > γ_threat ("fear lingers longer than curiosity"). That
makes the dominant eigenvector of (boost − Γ) timelike. A second, unplanned path
showed up: **transits**, where eating sweeps hunger through zero while fear is
still decaying and the state briefly snaps.

**E3 Order dependence.** The final g depends on event order: ‖Δg‖ is 84% of the
surviving change with forgetting on. That is mostly **recency**, though.
Without forgetting the path dependence is 1.5–1.7%, and it comes from the
state-dependent scales (|g^ii| sensitivities, frustration). With those off it
drops to 4e-4 (the frame law's nonlinearity adds a little; before it, 3e-5). **Finding for §5.2:** a Lie-group update `L ← exp(εA)L` whose image
in g doesn't depend on g induces an *additive* flow on g, so non-commutativity
only appears at O(ε²). Real "trauma then comfort ≠ comfort then trauma" needs a
generator that depends on the current geometry at first order, or large
single-event updates. **Revised under the frame law:** the missing order
information was not lost. Metric changes don't close (`[p, p] ⊂ so(3,1)`), so the
O(ε²) commutator of two experiences lands in the frame's *orientation*, not in g.
That orientation used to be discarded as gauge. With the frame law it is kept,
and a test shows that the same experiences in a different order change the frame
several times more than the metric.

**E4 Holonomy.** Exactly topological: an enclosing loop gives Φπ R² = 0.804 rad
(46.08°) whatever its shape, and an outside loop gives exactly 0. With dynamics
on, damping undoes the imprint in the drive values within ~4 s. But the NPC
is on the foraging limit cycle, whose phase is neutrally stable, so the haunted
loops leave a **lasting shift in meal times** that never decays back to zero
(between −3.8 and +2.4 s over the next 300 s). Memory of place persists in
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
is measured against a no-event twin run. With gossip it moves outward: the time
to half of peak is 5 s at distance 0, 18 s at distance 1 and 52 s at distance 2.
The size falls sharply after one hop (peak 0.17 at distance 0, 0.016 at distance
1, 0.003–0.008 beyond, with only 1–5 individuals per distance). Beyond 2 hops the
gossip signal is about the size of the behavioural noise, so its timing there
isn't meaningful (without gossip, the attack still perturbs everyone's
trajectories a little). Gossip also *comforts* the victims: their footprint at
t=170 s is 0.049 with gossip vs 0.085 without. Promotion instantiates from the
current collective metric (|g_new − g_col| = 0.05). Demotion consolidates 8
episodes chronologically.

**Surprise-gated compute.** With the `"cv"` predictor and `surprise_gate=0.05`, 44%
of cognitive ticks are skipped with identical meals and cone crossings over
600 s. At 0.1 the NPC misses snaps: a threat closing at constant speed is
*predictable*, and therefore skipped. The gate should not cover threat-closing.

## Frame law (option B)

Metric-changing generators (η-self-adjoint, **p**) and Lorentz generators
(**so(3,1)**) split gl(4) in the Cartan pattern: `[p, p] ⊂ so(3,1)` and
`[so, p] ⊂ p`, both checked numerically. So the space of personalities is the
curved symmetric space GL(4)/O(3,1). A loop of personality changes returns the
metric (to third order) but turns the frame by a Lorentz transformation of about
−ε²[P₁, P₂]: the analogue of the Thomas–Wigner rotation.

The first version treated that turn as pure gauge and re-gauged it away. That
broke two things: it discarded the order information, and gauge-dependent
quantities (the frustration norm and the rapidity clamp) silently changed with
the drift. Option B keeps it. The whole frame is physical, and its orientation is
the holonomy of experience.

What can go singular is the frame's *stretch* (its singular values). The frame
law applies relativity's rapidity/velocity relationship to it:

* each singular direction accumulates an unbounded experience rapidity φ;
* the expressed log-stretch is `s = H·tanh(φ/H)` (here H = 1);
* rotations, including the holonomy, pass through untouched;
* old and new stretches are paired by direction, not sorted order, so crossings
  don't swap rapidities.

Consequences, all tested (`tests/test_frame_law.py`):

| Property | Result |
|---|---|
| Never singular | cond(L) ≤ e^{2H} for any history |
| Early shifts | pass almost whole (a first push of 0.05 is expressed as 0.05) |
| Diminishing returns | after 100 same-direction pushes each moves the frame < 1/10 as much as the first |
| Entrenchment | undoing a trait takes as many experiences as building it (rapidities add) |
| Order memory | the same experiences in a different order change the frame several times more than g |

The agent's inner state lives in its experiential frame: when L changes,
`C_o = L C` is kept and its drive reading changes (`NPC.commit_frame`). The
light-cone interval is preserved exactly, so learning never jolts the regime.

**Open question:** in the drive basis the dynamics are `dC/dt = −β g C`, which
depend only on g. So the orientation currently acts only at commit time (via
state transport), and damping then erases it. For it to shape behaviour lastingly,
some reference has to live in the experiential frame; the temperament baseline is
the natural candidate.

## Arena (`arena.py`)

Five NPCs and one rival compete for one food patch. Both factions run the same
Cl(3,1) cognition and differ only in temperament. The rival is withdrawn (social
baseline < 0) and finds crowding painful.

Combat is world physics; it is never told to the agents:
* contact between a mobile NPC and the mobile rival is a mutual attack;
* every fighting NPC (C⁴ > 0) within 4 units of the rival joins it;
* with n attackers and rival strength k = 2, the rival takes n/(n+k) of the damage
  and the attackers share the rest. Alone, an NPC takes 2/3; four together take
  1/12 each;
* below 0.3 health an agent is immobile until it recovers above 0.6. It can't move
  or eat, and nobody targets it meanwhile. Injured agents also move slower.

Social homeostasis (off outside the arena): solitude raises loneliness and
company satisfies it, the same way hunger drift and eating work.

## Pain and reward (`affect.py`)

A second Clifford space, Cl(3,0), with basis hunger, social and injury. Each
axis is bipolar, + pain / − reward:

| Axis | + pain | − reward |
|---|---|---|
| hunger | hungry | sated |
| social | lonely (or crowded, for the anti-social rival) | in company |
| injury | injury taken | injury dealt to a rival (triumph) |

The drive state and body map into it every tick. The pain prediction error is
measured against a **slow** baseline (~100 s). A fast baseline made every injury
produce an equal and opposite "relief" error afterwards, so lessons cancelled.

**Pain-aversion learning.** The dynamics are `dC/dt = −β g C`, linear in g, so the
metric's effect on the active action tendencies s is
`E = sym((β s) Cᵀ)` (checked against finite differences). Each pain axis is tied to
the drive it is about (hunger → e1, social → e2, injury → e4), each drive axis
keeps its own eligibility trace, and the metric moves against pain:
`g ← g − rate·Σₖ vₖ Tₖ`. Pain after an action dampens it in that context; relief
or triumph amplifies it. Guards: learning may never change any drive's causal
character (hunger/social/explore stay spacelike, threat timelike), and cond(g) is
capped.

**Status:** the intended lesson ("fight in company, flee when alone") emerges in
some seeds (in one, lone NPCs go from fighting 100% of the time to 0% while groups
still fight 77%), but it is not reliable across seeds. Likely cause: hunger,
loneliness and injury lessons share the same metric entries (cross-talk).

## Long-term sequence memory (`ltm.py`)

**Storage.** When pain drops sharply (relief → an *aversive* memory) or turns into
clear pleasure (reward → an *appetitive* memory), the agent stores its last 16
drive states (~13 s). With it go the sense snapshot from when the sequence began
and a profile of which affect axes it relieved or rewarded. Repeats merge into
one, stronger memory.

**Loss aversion (λ = 2).** A relief of x counts as 2x when forming a memory: it is
twice as strong and needs only half the pain. When triggering, escape memories
tolerate twice the squared key/STM mismatch (a √2 wider basin).

**Recall.** Each tick, short-term memory (the last 32 sense vectors) is matched
against the stored keys. A modern-Hopfield attractor settles into one memory's
basin. Activation is weighted by **present affect** along the memory's profile:
* a relief memory needs that specific pain to be happening now, and its replay
  stops if the pain goes away;
* a reward memory is boosted by present reward, with a floor of 0.5.

Replay drives the state through a generator in the plane `C ∧ C_target` (a
rotation, or a boost across the light cone), like any other world channel.

**Layout.** Every size is a power of two:
* 8 sense features, a 32-tick STM and 16-step sequences;
* 64 slots, each one 512-byte record;
* a 4-axis affect profile and memory vectors padded to 128;
* every buffer allocated on a 64-byte boundary.

A test checks all of it.

**Memory Clifford space.** The basis is the 8 leading eigenvectors of
`Σ v_m x_m x_mᵀ` over stored (key, sequence) vectors, with v = +strength for
approach memories and −strength for escape memories. The eigenvalue signs give a
signature Cl(p, q). Two approach (or two escape) modes span an elliptic plane;
mixed planes are hyperbolic. It is identified, but nothing in the dynamics uses it
yet.

**E8 results** (arena, 4 seeds × 1500 s, gating on):

| | λ = 2 | λ = 1 | memory off |
|---|---|---|---|
| NPC mean pain | −0.20 ± 0.09 | −0.19 ± 0.02 | −0.17 ± 0.03 |
| fights | 14.8 ± 5.2 | 13.5 ± 5.9 | 10.8 ± 5.4 |
| total NPC injury | 6.3 ± 2.7 | 5.6 ± 2.4 | 3.7 ± 1.8 |
| time replaying | 7% | 7% | – |
| escape memories stored / recalled | 95 / 17 | 44 / 6 | – |
| relief recalls with no pain present | 0% | 0% | – |

* **Replays work.** Pain falls by 0.38–0.65 in the 4 s after a replay starts,
  against about 0 at random moments, in all 4 seeds.
* **Loss aversion shows.** Escape memories are stored about 2× and recalled about
  3× as often as with λ = 1. Memory spaces lean escape-heavy (e.g. Cl(3,5), and
  Cl(0,8) for one rival).
* **History.** Before present-pain gating, agents replayed 45% of the time,
  relief recalls were 209 per run, replays gave no immediate relief, and fights
  and injury were about 2× the no-memory baseline. Gating fixed the first three.
* **Caveat.** Fights and injury are still above baseline. A relief sequence is the
  ~13 s that *ended* in relief, so after an injury it often contains the fight
  itself.

## Known limitations and next steps

* Tuning is fragile. The foraging cycle and the snap both depend on the damping
  anisotropy and on eating, and all constants are in `config.py`.
* The frame's orientation is recorded but doesn't yet feed back into the dynamics
  (see [Frame law](#frame-law-option-b)).
* Pain-aversion learning finds the intended courage lesson only in some seeds;
  the pain axes cross-talk through shared metric entries.
* Relief sequences still include the fight that preceded the relief. Next: trim
  them to what ended the pain, and add avoidance memories (pain anticipated and
  dodged).
* The memory Clifford space is identified but unused by the dynamics.
* Place-memory replay exists, but the village never feeds it (the haunted disc is
  off there).
* A hyperbolic personality plane needs a real saturation law; the state cap is
  only a NaN guard.
* Not done: Cl(3,1,1) null-direction drift, reconstructive recall, an arena or
  village mode for the viewer, porting to a GA library (Kingdon).
