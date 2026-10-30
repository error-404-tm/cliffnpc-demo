# cliffnpc: Clifford-algebra NPC cognition, first demo

An experimental demo of the NPC-cognition half of the Clifford-algebra game
engine described in `../clifford-npc-spec.md`. An NPC's four drives (hunger,
social, explore, fight/flight) are a vector in a Clifford algebra Cl(p,q). The
signature is configurable: the single-NPC experiments use **Cl(3,1)** (threat
timelike), and the current arena work uses **Cl(2,2)** (hunger and fight/flight
timelike). Its personality is its own frame `L`, with metric `g = Lᵀ η L`. World
events push it through bivector generators. Everything is plain NumPy/SciPy with
4×4 matrices, runs headless, and saves its plots to files.

Beyond the spec's phases, the project now has:
* a **frame law** that keeps the frame's orientation (the holonomy of experience)
  and stops personality from going singular;
* a **two-tribe arena** with finite food, combat physics, death, birth and
  inheritance;
* **affects as trivectors**, the grade-3 dual of the drives, with three kinds of fear;
* **memory**: an event-chain episodic memory (partial match → fast, weak replay
  that primes affect), and an older drive-sequence memory;
* a **learned connection** (situation → drive generator), a **TD critic**, and
  **Pavlovian valence** (learned value → wanting);
* a **learning gym** of controlled tasks with baselines (E12–E18), which is where
  the current findings come from.

Start with [Current state](#current-state-october-2026) for the short version.

## Current state (October 2026)

What works, from the gym experiments (20 seeds each, against frozen, random, a
linear actor-critic and a hand-coded optimum):

| Ability | Mechanism that delivers it | Result |
|---|---|---|
| Learn which food is real | TD critic + Pavlovian valence on the drives | error 0.00 (linear AC 0.23) |
| Re-learn after the rule swaps | same | 0.02 late (linear AC stuck at 0.64) |
| Fear conditioning and extinction | same, landmark owned by fight/flight | 0.01 → 0.50, like the optimum |
| One-shot fear from a witnessed death | surprise-gated value learning | landmark preference 0.50 → 0.06 |
| Fight when outnumbering | felt-affect reward, drive valence | courage index 0.52, best payoff |

What we learned about the architecture:
* **Value learning works everywhere.** A linear critic on 16 situation features
  learns correct values in every variant tried.
* **Habits as a connection cannot change how much a drive is felt.** A bivector
  generator preserves the drive interval (d/dt C·gC = 0), so it can only redirect
  motivation. Adding the symmetric **p** part (Cartan split) gives a real but small
  gain. Learned habits are the weak channel.
* **Learned wanting is the strong channel.** The critic's value of each perceived
  stimulus, pushed onto the drive that owns it (food → hunger, rivals and danger
  landmarks → fight/flight), solves discrimination, reversal and fear.
* **How much is learned should follow the size of the value delta.** A witnessed
  death is ~50 SD of surprise; gating the critic's step by surprise turns it into
  one-shot learning.
* **Felt affect as reward needs care.** It must be relative to its running
  average (chronic pain otherwise blames whatever is in view), and valence must not
  feed back into the felt drive state (a separate wanting channel). Even then it
  produces lifelike but rigid behaviour: strong courage, fast fear, and
  **phobias** — avoided cues are never re-tested, so a slightly negative value
  hardens into permanent avoidance.

**Change of direction (spec v2, `spec-v2.md`).** After the gym, attempts to fix
the remaining problems (confidence-weighted valence, need features, bounded
valence) moved the ball little. The diagnosis: the geometry was chosen first
(named drives, a guessed signature, designed events) and learning was bolted on.
But every choice of environment, faculties and measurements already *induces* a
geometry on what an agent can do. Spec v2 measures that induced geometry first —
with a goal *vector* (food, safety, social, knowledge) and *experience records*
(controls, senses and goal progress over a window) as the basic unit — and uses
the Clifford algebra to represent what is discovered. First findings
([Induced geometry](#spec-v2-induced-geometry-commonspy-geometrypy-e20e22)):
circumstance-induced trade-offs are real and scale-dependent (safety·food ≈ −0.8
over 6–13 s under threat, zero without), but the local goal metric is mostly a
property of the agent's own behaviour (policy explains 41–95% of the variance of
each relation); the environment's own geometry is the frontier of the attainable
set.

```
cliffnpc/
  algebra.py      metric/frames, bivector generators, plane types, rotor step + rapidity clamp,
                  trivectors, Gram minors / sensitivities
  cognition.py    NPC state, dynamics step (saturation law), readouts, prediction, metric learning
  world.py        2D world: food, home, peers, periodically attacking threat, haunted disc
  coupling.py     sensor -> invariants -> low-rank coupling channels (Omega_ext)
  steering.py     cognition -> steering readout (isolated, swappable)
  sim.py          single-NPC main loop (multirate, optional surprise gating), logging
  memory.py       Phase 3: episodes, place rotors, collective metric, LOD transitions, gossip
  village.py      Phase 3: many tiered NPCs, social graph
  viewer.py       live viewer for the single NPC (window, or headless GIF/MP4)
  arena.py        two tribes (or NPC pack + rival): food, combat, life, affect, memory, learning
  arena_viewer.py live viewer for the arena
  affect.py       affect: trivector affects (current) and the older Cl(3,0) pain space
  ltm.py          drive-sequence memory: attractor recall, loss aversion, memory Clifford space
  episodic.py     event-chain memory: partial-segment match -> fast weak replay priming affect
  connection.py   learned connection, TD actor-critic, Cartan split, Pavlovian valence,
                  surprise-gated critic
  interrogate.py  read-only reports on what an agent learned and why
  gym.py          learning gym: controlled tasks x policies (E12-E19)
  commons.py      spec v2: environment factors, exploratory controllers, experience records
  geometry.py     spec v2: induced goal metric, record moments, return curvature, attainable-set frontier
  harness.py      spec v2: environment dials, fingerprints, paired CRN comparisons
  phase_c.py      spec v2: agent that learns its own induced geometry and moves to the frontier
  config.py       every tunable constant
experiments/      e1..e6, e8..e25, each writes experiments/out/<name>/{*.png, summary.txt}
tests/            133 tests
spec-v2.md        the v2 plan: discover the induced geometry instead of imposing it
```

## Running it

```bash
cd /home/joe/projects/cliffnpc-demo
.venv/bin/python -m pytest -q          # 133 tests, ~2.5 min
experiments/run_all.sh                 # all experiments in parallel (the gym ones take ~1 h)
.venv/bin/python experiments/e2_lightcone.py   # or one at a time
```

Runs never draw anything, so they go as fast as the CPU allows: the single NPC
runs ~100× real time, the 6-agent arena ~20×, the 30-agent village ~5×. The
bottleneck is Python/NumPy overhead on many tiny 4×4 operations, not arithmetic.

`run_all.sh` starts all experiments at once, and each experiment spreads its
independent runs (seeds, conditions) over worker processes (`experiments/common.py:
pmap`). Processes, not threads: this Python build has the GIL, and one simulation's
agents are too few to split. Every run is seeded, so parallel results are identical
to serial ones; checked for E3, E6 and E8. Set `CLIFFNPC_WORKERS=1` to run serially. Each experiment's output goes to
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

The arena has its own viewer (two tribes, food, fights, deaths and births):

```bash
.venv/bin/python -m cliffnpc.arena_viewer --signature 22         # interactive
.venv/bin/python -m cliffnpc.arena_viewer --save arena.gif --seconds 60
```

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

## Results: single NPC and village, E1–E6 (numbers from `experiments/out/*/summary.txt`)

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

**Two tribes** of identical agents (5 each at the start) share a 200×200 world.
They run the same cognition and regard only their own tribe as allies. One
drive signature is used for every agent in a simulation, and concurrent runs vary it.

**Food is finite and moving.** Each patch holds a stock and disappears when
eaten out. New patches spawn half at two fixed sites and half at random places,
and 30% look like food but give no satiety (false food).

**Combat is world physics**, never told to the agents. Agents of different tribes
in contact fight; fighting members (C⁴ > 0) of either side nearby join. With n on
one side and m on the other, a side takes m/(n+m) of the damage, shared equally.

**Life.** Agents starve or die of injury; speed falls with injury and slightly
with satiety. Sated, healthy adults reproduce in pairs; children inherit a blend
of their parents' frames, some memories and the learned connection. An agent that
sees another die gets a strong shock (felt injury, flight recoil) and an imprinted
memory. Each drive also obeys a cubic **saturation law** (semi-implicit): pushes
have diminishing returns and nothing blows up.

(The original arena, an NPC pack against one anti-social rival, is still the
default `ArenaConfig`.)

## Affects as trivectors (`affect.py`)

Drives are vectors; **affects are the dual trivectors**. With drives hunger (1),
social (2), explore (3), fight/flight (4), the four affect directions are e123 and
the three fears e124, e134, e234 (each containing fight/flight). Each affect is
coupled to the three drives it contains.

* **Events** are designed drive-space bivectors with a valence: unmet hunger,
  meals, loneliness, company, injury taken and dealt, witnessed death, false food,
  and the world's push. Each contributes its change C∧B and its surprise
  (C − Ĉ)∧B to the affect state P.
* **Felt affect** is F = (felt matrix) · P, where the felt matrix is the metric g
  induces on trivectors (its 3×3 minors), sign-normalised and divided by
  |det g|^¾, so it depends on the shape of the personality, not its scale.
* **Behaviour:** the contraction C ⌋ F acts as a generator on the drives (capped).
  Rivals near add a boost in the social–threat plane (the mob channel).
* **Learning:** felt pain drives pain-aversion metric learning with per-drive
  eligibility traces; blame goes to each affect's three drives.

The older Cl(3,0) pain space (hunger, social, injury axes) is still available as
`affect_model="vector"`.

**E9/E10 (two-tribe arena, 4 seeds × 2000–4000 s).** Life produces births across
6–9 generations, witnessed deaths and imprinting. Cl(2,2) is the most violent and
fertile signature, Cl(3,1) the calmest. Cl(4,0) alone shows a structural courage
pattern (fights alone 0.24, with tribe-mates 0.59), built in rather than learned.
No signature shows learning-driven change over time; that motivated the gym.

## Drive-sequence memory (`ltm.py`, older)

Superseded in the current setup by the event-chain memory below; kept for E8.

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

## Event-chain episodic memory (`episodic.py`)

Memories are **chains of discrete events** (rival seen, contact, injury, triumph,
food seen, meal, false food, company, alone, death seen), each with its situation
vector. A chain of the last 30 s is stored when pain is relieved, reward arrives
(loss aversion 2: relief counts double) or a death is witnessed (imprinting).

**Activation by partial match.** When the recent events match a contiguous
segment of a stored chain, the rest of the chain **replays** at 0.2× strength and
8× speed, as weak impulses into the affect state: the memory primes how the
agent feels, it does not move it. Records are 64 bytes per event and 2048 per
chain, all buffers 64-byte aligned. Children inherit some of their parents' chains.

## Learned connection, critic and valence (`connection.py`)

The agent's **situation** is a 16-feature vector: food near, rival near and
closing, allies, rivals, outnumbered, injured, backed (rivals × allies), the
nearest food's look (A/B, one-hot), nearness to a marked landmark, and spare slots.

* **Connection.** Each feature carries a generator; in a situation x the drives
  move by β(x) = Σ x_f W_f. Options: bivectors only (so(g), redirect drives), or
  the **Cartan split** so(g) ⊕ p, where the p part P = g⁻¹S rescales drives.
* **Learning rules.** Per-axis pain aversion (original), or a **TD actor-critic**:
  a linear critic V(x) with TD(λ), and an actor that explores with smooth noise in
  the generator space and moves along (situation × noise) by the TD error.
* **Pavlovian valence.** The critic is linear, so each perceived stimulus has its
  own learned value (its features' share of V). That value becomes approach/avoid:
  pushed onto the drive that owns the stimulus (`"drive"`), into a separate
  action-only **wanting** state (`"wanting"`), or as a reflex velocity (`"body"`).
* **Surprise-gated critic** (Pearce–Hall). The step grows with |δ| relative to its
  running spread above 8 SD, capped at 1, normalised by the eligibility trace so it
  never overshoots: rare large events are learned in one exposure.
* **Differential TD** for the felt-affect reward: reward relative to its running
  average.

`interrogate.py` reads all of this out per agent: courage and response tables,
strongest learned entries, habit holonomy, basis magnitudes and cross terms,
how much felt affect came from replay, memories, and a learning ledger.

**E11 (arena, Cl(2,2)).** Starting basis magnitudes (hunger, fight/flight in
{0.7, 1, 1.4}) wash out: metric learning carries both to ~1.2–1.5. Connection
learning fills the tribes and lowers pain, but no population-level courage
emerges; its weights saturate.

## Learning gym (`gym.py`, E12–E18)

Controlled tasks in a 100×100 world with one learner and scripted food, rivals
and allies, each run under several policies (20 seeds per pair) with pass
criteria fixed before running:

* **discrimination** — two patches on opposite sides, look A real, look B false;
* **reversal** — the looks swap meaning halfway;
* **fear v2** — one patch at a landmark brings an ambush by three rivals, then
  stops (extinction);
* **one-shot** — an ally dies at the landmark once, against a no-death control;
* **courage** — the learner with k allies meets m rivals; fighting pays iff k+1 > m.

Baselines: the same agent frozen, random, a linear actor-critic on the same
features (objective reward), and a hand-coded optimum. Controllers perceive
exactly what the agent perceives.

| Exp. | Question | Answer |
|---|---|---|
| E12 | Does the agent learn at all? | No: no task beats frozen; learning *erases* built-in courage. The linear AC learns discrimination. |
| E13 | TD actor-critic on the connection? | Critics learn correct values, but behaviour doesn't change: a bivector connection preserves C·gC, so it can't change hunger along its own axis. Also: hunger needed an aversive pole. |
| E14 | Cartan split so(g) ⊕ p? | First real gain: discrimination error −0.05 to −0.07, all of it in p. |
| E15 | Pavlovian valence? | Drive-layer valence: discrimination 0.00, reversal 0.02 late, best courage payoff, 1% defeats. Body-layer valence sign-tracks the landmark. |
| E16 | Fear task redesigned (the 1-vs-1 ambush cost nothing) | Drive valence learns (0.01) and extinguishes (0.50) like the optimum. Felt-affect reward learns fastest but never extinguishes (avoidance paradox). |
| E17 | Learning gated by the value delta? | One witnessed death: landmark preference 0.50 → 0.06 (paired −0.45 ± 0.08). Costs a little flexibility. |
| E18 | Felt-affect reward for food | Differential TD + wanting channel: discrimination 1.00 → 0.10. Still phobic. |
| E19 | Confidence-weighted valence | Objective reward + surprise gate + confidence: the best all-round agent (one-shot −0.46, fear learned in ~6.5 trials and extinguished 0.52, discrimination 0.04, reversal 0.06). The felt-affect phobia is untouched: the avoided cue stays in view, so confidence never fades. |

Negative results kept in the code (off by default): **need features** for the critic
(crediting need pain to the need) — no effect on the phobia, breaks extinction
under objective reward; **bounded, normalised valence** — one valence channel
cannot both beat hunger (skip false food) and be beatable by hunger (re-test a
feared cue), whatever the scaling.

Gym bugs found on the way and fixed: in E12–E16 the one-shot death never happened
in 11 of 20 seeds; a 1-vs-1 ambush cost nothing under the objective reward; the
actor's exploration was paired with the wrong situation; gated steps overshot
along the eligibility trace.

## Spec v2: induced geometry (`commons.py`, `geometry.py`, E20–E22)

**Commons.** One controlled learner in a 100×100 world with a forager ally, a
companion who wanders, ambushing rivals and food patches with looks. Five
environment factors, each chosen for the goal relation it should induce: forager
skill (social·food), threat at food (safety·food), a shelter rivals cannot enter,
cue validity (knowledge), scarcity. The centre plus one-factor-at-a-time changes
gives 7 environments.

**Goals and records.** Per tick the learner logs its controls, the exploration
noise, its senses (24) and a goal-progress vector: food (satiety gained), safety
(minus injury), social (an ally in range), knowledge (bits resolved about patch
types — by sight only if cues are valid, otherwise by probing). Records are windows
of 16, 32 or 64 ticks (3.2–12.8 s), laid out in powers of two.

**Exploratory controllers.** The learner moves by a weighted sum of six primitives
(toward food, toward the forager, toward the shelter, away from / toward the
nearest rival, wander), with episode weights θ and smooth noise ξ on them. Five
policy families (broad, narrow, bold, cautious, social) set the range of θ.

**Estimators.**
* the **goal metric** M = A Aᵀ, where A[i,k] is how leaning on primitive k changes
  goal i's progress over a record (noise regressed against outcomes, situation and
  θ regressed out) — the circumstance-induced inner product of goals;
* the **outcome-weighted record moment** S = E[w (x − x̄)(x − x̄)ᵀ] and its signature;
* the **return curvature** in θ;
* the **attainable-set frontier**: the Pareto set of expected goal returns over many
  policies, its goal correlations, utopia gaps and curvature signature.

**E20/E21 (local geometry; E21: 23 conditions × 1024 episodes).**
* safety·food ≈ −0.8 at 32–64 ticks under threat, exactly 0 without threat, and
  positive at 16 ticks: the cost of going out arrives later.
* knowledge·food is higher with random cues (+0.89 to +0.93: you must go there to
  learn) than with valid ones (+0.59 to +0.79).
* The forager's skill shows in *magnitude*, not angle: the food gained by leaning
  toward the forager is 1.2–4× larger with a skilled forager, while the social·food
  cosine is ≈ +0.93 either way.
* **Policy explains most of the variance of every relation** (41–95%; environment
  1–37%): under a cautious policy safety·food is −0.94, under a bold one −0.13 to
  +0.46. The local geometry is the agent's own: a tangent space at its current way of
  behaving, which moves when it learns.
* With threat, record space has few net-good directions (signature ≈ (1, 3)); without
  threat it is balanced (≈ (6, 5)).
* In E20, defining knowledge by patch visits and keeping an ally at home produced
  spurious geometry (knowledge·food ≈ +0.9 everywhere; forager skill invisible): the
  measurement choices induce geometry too.

**E22 (the environment's own geometry: the attainable set).** 384 fixed policies × 6 episodes per environment;
expected goal returns denoised by a quadratic in θ; the frontier is the Pareto set.
* **food·social is cooperative in every environment:** the frontier correlation is
  +0.90 to +0.98 (raw means agree), and the utopia gap ≈ 0 — one kind of policy,
  following the forager, attains both maxima. Forager skill changes the frontier's
  *extent*, not its shape: the best attainable food is 1.48 with a skilled forager
  against 0.84 with an unskilled one.
* **food·safety is a genuine trade-off whenever there is threat** (gap 0.44–0.58)
  and none without threat. Against expectation, the shelter does not soften it
  (gap 0.44 with, 0.48 without).
* **Frontier policies lean on the forager and on fleeing, not on seeking food**
  (forager +0.7 to +1.6, flee +0.6 to +1.0, food −0.4 to +0.3): in this world the
  efficient way to eat is to follow the one who is good at it.
* Caution: some frontier correlations (food·safety at the centre, everything about
  knowledge) and the frontier's curvature signature depend on the denoising — the
  raw-means frontier disagrees — so they are not reported as findings yet.

## Feature harness and the ladder (`harness.py`, E23)

Every environment feature is another geometric influence, so features are added
one at a time as **dials** (intensity 0–1, 0 = absent) and measured, following
`spec-v2.md` section 8:

* **Fingerprint**, computed the same way for every world: *extent* (best and average
  attainable return per goal), *shape* (frontier correlations and utopia gaps,
  smoothed and raw, compared on the goals live in both worlds), *local geometry*
  (the goal metric at five reference policies and three time scales), the record
  signature, and *live dimensions*.
* **Paired comparisons with common random numbers:** per-episode random streams for
  food, ambushes, noise and the companion consume the same draws whatever the dials,
  so two worlds differing in one dial see the same patches and draws. Paired
  intervals are ~0.72× the width of unpaired ones (up to ~2× narrower for shape).
* **Four kinds of effect:** extent, shape, local, new dimension; **construct checks**
  flag effects on relations a dial was not meant to touch.

**E23 (validation against the known factors): all 9 pre-registered checks pass.**
Most dials move the frontier rather than reshape it; only cue validity (and extreme
scarcity) change the trade-off structure. The construct checks caught a flaw in the
harness itself (comparing frontiers over different live goals) and two real
cross-effects (an unskilled forager also changes food·safety; scarcity's knowledge
effects are partly by construction).

## Phase C (`phase_c.py`)

The phase-C agent acts like the frontier panel's policies (weights θ on the six
primitives) so its result can be placed against the measured frontier. Its state is
the geometry it induces: it learns the vector action gradient A (how each control lean
moves each goal) from its own exploration against vector TD errors, and its local goal
metric M = A Aᵀ. Preferences ω (agent-specific weights on normalised goals) pick a point
on the frontier. Three rules move θ: plain gradient Aᵀω, natural gradient (the
smallest control change for the desired goal change, using the pullback metric), and
Pareto-first (improve every valued goal at once, then trade off by ω). Interrogation:
goal modes (eigenvectors of M), trade-off planes aᵢ∧aⱼ (area = independent
controllability, cosine = aligned/opposed), and the connection (how M turns as the
agent moves).

**Rung 0 (E24; 3 rules × 4 preferences × 8 seeds × 200 episodes).**
* Every rule learns (+1 to +2 normalised units), and final policies are almost never
  dominated by any of the 192 panel policies (0–4%): agents land on the frontier,
  reaching 0.85–1.00 of the panel's best for three preferences. All converge on
  "follow the forager", the efficient strategy E22 found.
* Safety-weighted agents struggle (0.54 of best): safety is driven by rare ambushes,
  so its gradient is very noisy.
* Preferences barely differentiate, and the geometry rules do not beat the plain
  gradient (natural − plain +0.07 ± 0.48, Pareto − plain +0.25 ± 0.46): at rung 0 one
  strategy dominates and six well-conditioned controls leave nothing for the geometry
  to correct. The local metric's turning is mostly estimation noise.

**Agent fixes tested at rung 0 (E24b), both rejected.** Slower forgetting (0.98)
changes nothing; a per-tick TD critic with bootstrapped advantages and a surprise gate
makes the local metric ~3× steadier (late change per 20 episodes 0.51 → 0.14–0.16) but
lowers performance (−0.11 cautious, −0.15 balanced): its value errors steer the policy.
Both stay in the code as options, off by default.

**Rung 1: internal need (E25).** The `need` dial makes hunger grow faster and starving
a setback on the food axis (0 = rung 0).
* *Environment (harness):* need is a pure **extent** dial. The best attainable food
  more than doubles at need 1 (+2.2 normalised units, +1.1 at 0.5); no shape, local
  or new-dimension effect, and no construct flags. It stretches the food axis without
  changing any trade-off, and the frontier's best policy is the same for every
  preference.
* *Agents:* they absorb the rung reasonably well, reaching 0.74–0.86 of the panel's best
  (just below the 0.8 criterion for foragers). Safety-weighted agents now do much better
  (0.82 against 0.54 at rung 0), because food now matters to them too. Pareto-first is
  the best and steadiest rule (0.85–0.86, interval ±0.04 for balanced agents); the
  natural gradient is unstable (one cautious run set falls to −0.08). Preferences still
  cannot differentiate — by construction, since one strategy remains best for all.
* Lesson for the ladder: a dial that only moves the frontier gives the agent nothing
  new to *choose*. The geometry rules should only start to matter on rungs that change
  the frontier's shape (cue validity did; competition and combat should).

## Known limitations and next steps

* **Phase C** works at rung 0 (agents reach the frontier) but its geometric parts do
  not yet pay off; the ladder (`spec-v2.md` section 9) adds the trade-offs that should
  make them matter, one measured dial at a time.
* **Phobias.** Under the felt-affect reward, avoided cues are never re-tested, so
  a slightly negative value becomes permanent avoidance. Confidence weighting, need
  features and bounded valence did not fix it.
* The surprise gate gives one-shot learning but slows extinction and reversal.
* The gym's winning mechanisms (critic, valence, surprise gate) are not yet
  tested in the full two-tribe arena.
* Landmarks are owned by fight/flight by design; a general way to bind stimuli
  to drives is open.
* The felt-affect reward was tuned in a gym whose trial structure (hunger reset,
  2 s meals) is unnatural.
* Tuning is fragile; all constants are in `config.py`.
* The frame's orientation is recorded but doesn't yet feed back into the dynamics
  (see [Frame law](#frame-law-option-b)).
* The memory Clifford space (`ltm.py`) is identified but unused.
* Not done: Cl(3,1,1) null-direction drift, reconstructive recall, porting to a
  GA library (Kingdon).
