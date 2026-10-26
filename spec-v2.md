# Spec v2: discovering the geometry instead of imposing it

Status: draft for discussion (October 2026). Builds on `../clifford-npc-spec.md`
and on everything learned in E1–E19 (see README).

## 1. Why a v2

Up to E19 the geometry was chosen first: four named drives, a signature picked
by intuition (Cl(2,2)), designed event bivectors and affects. Learning was then
bolted on, and most of the work after the learning gym went into patching the
interface between that imposed geometry and the tasks.

But every choice of environment variables, agent faculties and measurements
already *induces* a geometry on the space of what an agent can do. v2 measures
that induced geometry first, and uses the Clifford algebra as the tool to
represent and discover it, rather than as a prior.

## 2. Objects

### 2.1 Goal vector

Goals are a vector, never a scalar. Initial goal axes (k = 4):

| Axis | Progress per tick | Setback |
|---|---|---|
| food | satiety gained | false food, starvation |
| safety | — | injury taken, death |
| social | time near allies | isolation |
| knowledge | prediction error reduced (uncertainty resolved) | — |

**Knowledge is agent-specific.** Every agent records progress on all four axes,
but how much it *values* knowledge is a trait. Agents that value it and agents that
don't are compared directly (a testable hypothesis: does valuing knowledge change
the induced geometry, and does it pay in goal completions?).

Each tick produces a goal-progress vector `r_t ∈ R^k` (setbacks as negative
progress on their axis). The vector return is `G_t = Σ_j γ^j r_{t+j} ∈ R^k`.
A vector critic `V(s) ∈ R^k` (one head per goal) estimates it.

### 2.2 The basic unit: the experience record

Not a behaviour. A **record** is a fixed window of T ticks holding:

* the control signals sent: velocity command (2), fight flag (1), and any other
  actuator, per tick;
* the sense content present: the situation features (16), per tick;
* the goal-progress vector `r_t` (k), per tick, and the vector return at the
  window's end.

**Multiple scales.** Records are kept at T = 16, 32 and 64 ticks (3.2, 6.4 and
12.8 s), so we can see which goal relations live at which time scale (e.g.
safety · food may be negative over seconds and positive over a minute, if
surviving now enables eating later).

Sizes are powers of two and records are 64-byte aligned, as in the rest of the
codebase. Records are stored in a ring buffer (agent memory) and in run logs
(analysis).

**Behaviours are higher-order.** A behaviour (foraging, fleeing, following a
forager) is a region or direction in record space, characterised *after* context
is established. We do not define behaviours up front; we look for them as
structure in record space under the induced metric (section 3).

## 3. The induced geometry

### 3.1 The goal metric induced by circumstance

For each goal i, the **action gradient** `a_i = ∂Q_i/∂u` says how a change in
control changes the expected progress of goal i, in the current situation. The
circumstance-induced inner product of two goals is

    M_ij = E_s [ a_i(s) · a_j(s) ]

* "Stay near the ally who is good at finding food" advances social and food
  together: `M_social,food > 0`.
* "Don't go out" keeps you safe now but costs food: `M_safety,food < 0`.
* Goals whose progress doesn't depend on the same controls are orthogonal.

`M` is a metric on goal space that depends on the environment. It is estimated
model-free with the machinery we already have: smooth exploration noise on the
controls, vector TD errors, and the noise × TD-error correlation (the eligibility
used by the E13 actor) gives `a_i` per situation.

### 3.2 Other induced quadratic forms (to compare, not assume)

* **Outcome-weighted record moment:** `S = E[w(τ) (x_τ − x̄)(x_τ − x̄)ᵀ]` over
  record feature vectors `x_τ`, with `w` the scalarised outcome. It is indefinite
  in general: its signature separates record directions that go with goal
  progress from those that go with setbacks; null directions are net-neutral
  (natural exploration directions).
* **Curvature of the vector return:** the Hessian of each `Q_i` with respect to
  policy parameters, from perturbation runs. Its signs show where controls trade
  off (saddles) and where they synergise.

Which form is the useful one is an empirical question. Only invariants are
meaningful, since all of these depend on how records are featurised:
signature, plane types (elliptic: combine freely; hyperbolic: genuine trade-off),
eigenstructure, and how they change with the environment.

### 3.3 Geometry as a function of circumstance

The environment is a coordinate. Sweeping environment factors gives a field of
goal metrics `M(e)` over environment space, and how `M` changes from one
circumstance to the next is a connection. This is what the original spec's
connection was meant to be: learned from circumstance, not designed.

## 4. Environment factors (E): chosen for the relations they induce

Every choice of E is a different geometry. Too many factors is intractable; too
few misses relations. So each factor is chosen to create one specific goal
relation, and the set is kept small enough to sweep:

| Factor | Levels | Relation it should induce |
|---|---|---|
| ally foraging skill | equal / one ally knows the real patches | social · food > 0 |
| threat at food | none / sometimes / often | safety · food < 0 |
| shelter (rivals cannot enter) | absent / present | safety · (staying home) > 0, competing with food |
| cue validity | cues uninformative / predictive | knowledge · food, knowledge · safety > 0 |
| scarcity | rich / scarce | strengthens every food relation |

Design: a centre point plus one-factor-at-a-time sweeps (7 environments), not a
full factorial (48). Pairwise interactions only where a one-factor sweep shows a
change in signature or plane type.

Caveat: the measured geometry also depends on the policy that generates the
records. The exploratory policy is itself a choice; it is varied in phase B.

## 5. Agent faculties (F) and what measures them

| Faculty | Measured by |
|---|---|
| discrimination | goal progress on food with false patches |
| flexibility | recovery after a cue's meaning reverses |
| conditioning / extinction | setbacks avoided at predictive cues, recovery when they stop |
| one-shot learning | change after a single large setback |
| social calculation | food progress via the skilled ally; fighting only when it pays |
| explore / exploit | knowledge progress versus food progress over time |
| planning | multi-step goal completions (e.g. food beyond a threat) |

The headline outcome: goal completions per unit time, against setbacks, as
vectors over time.

## 6. Phases

* **A. Specification.** This document; agree on goals, record format and E.
* **B. Measure.** A new environment with the E factors; the experience recorder;
  exploratory controllers; vector critics; `M(e)` and the other forms per
  environment. Output: the induced geometry and how it moves with circumstance.
  No new agent dynamics.
* **C. Represent.** The Clifford algebra on the discovered basis and signature;
  metric learning so that paths from situation to goal completion are short and
  setbacks expensive (geodesics of the induced metric). Reuse `algebra.py`, the
  frame law, generators and interrogation.
* **D. Validate.** In the gym and the arena: discovered-geometry agent vs the
  current 4-drive agent vs the linear actor-critic vs the hand-coded optimum.

## 7. What carries over

The gym (tasks, baselines, pre-registered criteria), the arena, the parallel
runners, `algebra.py`, the frame law, the TD and perturbation machinery, the
surprise gate, the interrogation tools, and every finding to date. The current
agent becomes one hypothesis among several.

## 8. Open questions

* Decided: knowledge is a goal whose value is agent-specific (section 2.1).
* Decided: records at three scales, 16/32/64 ticks (section 2.2).
* How much of the measured geometry is the exploratory policy's, and how much the
  environment's?
* Does the induced signature come out indefinite, and on which axes?
