# Row-Conditioned Visual Coverage Design

## Decision

Build a first-principles research prototype for **row-conditioned visual
coverage** under standard SFT with CE.

The mechanism should condition each object row on feature-level coverage marks
derived from previously enumerated objects. The first version prioritizes
mathematical/tensor-flow correctness over decode efficiency:

```text
V = vision_encoder(image)
M_k = coverage(committed objects before row k)
V'_k = V + coverage_residual(M_k)
p(row_k | prompt, text_prefix(S_k), V'_k)
```

This design intentionally starts with visual feature tuning, not attention-bias
routing. Coverage is a learned visual state marker; it is not a hand-coded
suppression, selection, or attention policy.

Treat the current implementation as a **Direct Feature Painting Oracle**: a
scientifically clean prototype for testing whether explicit row-conditioned
visual coverage helps autoregressive object enumeration. It is not yet a
cache-efficient deployment design and must not be interpreted as one.

Primary decision record:

- `progress/directions/2026-06-05_row_conditioned_visual_coverage.md`

Next protocol layer:

- `docs/superpowers/specs/2026-06-05-row-conditioned-visual-coverage-protocol.md`

## Goals

- Test whether explicit row-conditioned coverage helps autoregressive VLM
  object enumeration in dense scenes.
- Preserve standard SFT/CE as the first objective surface.
- Support both `random_sft` and `sorted_sft` as controlled ablation modes.
- Make coverage boundary-primary and interior-weak.
- Ensure every covered visual region remains observable and recognizable.
- Keep training and rollout tensor flow equivalent for any prefix state.
- Establish clean mechanism and rollout guardrails before optimizing inference
  efficiency.

## Non-Goals

- Do not introduce residual-set, trie, set-supervision, duplicate-unlikelihood,
  EOS forcing, or fallback objectives in the first design.
- Do not use attention-bias routing as the primary mechanism.
- Do not rely on raw-pixel repainting, image re-rendering, or re-encoding the
  image after each object.
- Do not optimize for KV-cache efficiency in the first proof.
- Do not describe bbox-derived boundary coverage as exact object-contour
  coverage unless segmentation masks are introduced.
- Do not describe the interior channel as direct visual suppression. It marks
  already enumerated extent; any duplicate reduction or attention shift must be
  learned through CE.
- Do not create an OpenSpec change yet; OpenSpec is deferred until a stable
  config schema, artifact contract, metric contract, or supported runtime
  behavior is ready.

## Core Hypothesis

The model may already perceive many missed object instances, but the
autoregressive sequence lacks an explicit visual coverage state that tracks
which regions have already been enumerated. Text history alone is a weak memory
for dense same-description scenes, especially when duplicate bursts and low
recall appear together.

Coverage should provide a feature-level state marker:

```text
this visual extent has already been enumerated
```

Coverage must not mean:

```text
hide this region
this region is negative
never attend here
the next object must be elsewhere
EOS is more likely
```

The behavior should be learned implicitly through CE.

## Coverage Descriptor

For v1, coverage has only two channels per visual token:

```text
M_k(i) = [
  boundary_mass_i,
  interior_mass_i,
]
```

Do not include class text, object count, recency, confidence, or predicted
remaining-object signals in v1. These would make the first result harder to
interpret.

This is **union coverage memory**, not instance memory. All previous boxes are
accumulated into the same boundary and interior channels. V1 cannot represent
which previous object covered a token, how many previous objects covered it, or
which text/class identities those objects had. Count channels, log-count
coverage, class-summary coverage, confidence-weighted coverage, or
instance-memory variants are future ablations, not part of the first claim
path.

Coverage accumulates additively over previous objects, then is bounded:

```text
boundary_mass_i = saturate(sum_j boundary_mass_i(B_j))
interior_mass_i = saturate(sum_j interior_mass_i(B_j))
```

Do not normalize coverage as a global probability distribution over the image.
The descriptor should preserve local density without changing unrelated
regions when a new object is added elsewhere.

## Boundary And Interior Painting

The painting operation is defined on the visual-token lattice, not at exact
pixel-edge precision.

Each visual token has an image-space footprint:

```text
P_i = footprint of visual token i
B_j = previous bbox j
```

Interior coverage:

```text
inside_mass_i(B_j) = area(P_i intersect B_j) / area(P_i)
```

Boundary coverage uses a soft bbox-edge band:

```text
boundary_band_width ~= one effective visual-token footprint
boundary_mass_i(B_j) =
  area(P_i intersect boundary_band(B_j)) / area(P_i)
```

The band width should reflect the visual-token resolution, such as the
Qwen3-VL-style 32x32 patch-scale intuition, rather than the exact precision of
coord tokens.

Bbox edges should not be expected to land exactly on visual-token cell
boundaries. Edge mismatch is handled through soft fractional painting:
each token receives mass proportional to overlap area. The boundary channel is
a bbox-extent edge band on the post-merge visual-token lattice, not a true
object contour.

Boundary is stronger than interior:

```text
alpha_boundary > alpha_interior
```

Interior coverage remains weak because overlapping, contained, occluded, or
nearby objects may still need to be recognized inside previously covered boxes.

## Visual Feature Tuning

The first `Tune` operator is a simple additive residual:

```text
V'_i = V_i
     + alpha_b * boundary_mass_i * e_boundary
     + alpha_i * interior_mass_i * e_interior
```

Requirements:

- `e_boundary` and `e_interior` are learned coverage embeddings.
- `alpha_b > alpha_i`.
- The contribution is bounded.
- The default/initial behavior is near identity.
- Original visual features remain available.

Alpha caps alone do not guarantee a small perturbation because the residual
magnitude also depends on the norms of `e_boundary` and `e_interior`. Every
coverage-active training smoke should log or otherwise inspect:

```text
||delta_F_i|| / ||F_i||
```

at minimum as max and mean token ratios, plus boundary/interior embedding
norms. Conservative initialization is required. Norm regularization,
embedding-norm control, or stricter alpha caps are valid follow-up ablations if
the ratio grows large enough to dominate the original visual feature.

Forbidden first-version behavior:

```text
V'_i = coverage_only_i
V'_i = 0 for covered tokens
attention_to_i = -inf for covered tokens
```

A gated residual adapter may be a later ablation:

```text
delta_i = Gate(V_i, M_i) * Adapter(V_i, M_i)
V'_i = V_i + delta_i
```

It should not be the first claim path because it increases ambiguity between
coverage semantics and added capacity.

## Training Tensor Flow

Use prefix-state expansion.

For an ordered object sequence:

```text
state 0: coverage(empty) -> predict row 1
state 1: coverage(row 1) -> predict row 2
state 2: coverage(rows 1..2) -> predict row 3
state N: coverage(rows 1..N) -> predict assistant stop
```

Each state is a standard SFT/CE example:

```text
input:
  prompt
  image features V tuned to V'_k
  text prefix for rows before k

target:
  next object row k under shifted CE, or the ordinary assistant stop token for
  the terminal state
```

The current row and future rows must not be included in coverage.

Loss masking is a contract, not a convenience. For row-state `k`, previous
object rows are text context, the current object row is supervised, the
terminal stop row is supervised, and prefix rows must not be re-supervised
unless a later ablation says so explicitly.

The terminal state is required for train/rollout equivalence. It must use the
same chat-template assistant stop target as standard SFT, not a custom stop
loss, stop bonus, or EOS forcing rule.

For any prefix state `S_k`, training and rollout share the same conditional
form:

```text
p(row_k | prompt, text_prefix(S_k), Tune(V, coverage(S_k)))
```

The only allowed difference is the source of `S_k`:

```text
training: teacher/sample/corrupted/rollout-derived prefix
rollout: committed model predictions
```

## Random And Sorted SFT Ablations

The design must support both object-order modes:

```text
random_sft
sorted_sft
```

`random_sft` tests whether coverage supports arbitrary enumeration without a
fixed spatial order. The coverage state says what has been emitted, not where
the model should go next.

`sorted_sft` tests whether coverage helps when the next-row policy is already
partly structured by top-left or another deterministic ordering rule.

The first experimental comparison should keep the mechanism identical across
both modes:

```text
same coverage descriptor
same visual feature tuning
same CE objective
same rollout re-prefill rule
only object ordering differs
```

Reports must not collapse the two modes into one result. The two modes answer
different scientific questions.

## Rollout Tensor Flow

Use row-boundary re-prefill for the first proof.

```text
V = vision_encoder(image)  # once

for row k:
  M_k = coverage(committed rows before k)
  V'_k = Tune(V, M_k)
  re-prefill prompt + V'_k + committed text prefix
  decode the next object row
  if row is parse-valid:
    commit row and update coverage
```

This recomputes the decoder prefix at row boundaries, but does not rerun the
vision encoder.

Rationale:

- avoids stale text KV computed under a previous visual state;
- matches prefix-state teacher forcing;
- provides an oracle-like correctness path before cache-efficient variants.

Cache-compatible variants, such as appended coverage tokens or cached K/V
patching, are deferred. They should not rescue the mechanism if the faithful
re-prefill formulation fails.

Training/inference tensor-flow equivalence is claimed only under faithful
row-boundary re-prefill. Do not mix this proof path with the later
cache-compatible append-coverage-token family except in an explicit comparison.

## Evaluation Strategy

Use staged joint evidence.

Stage 1: mechanism gate.

```text
coverage off vs coverage on
same image and comparable prefix state
measure whether row-conditioned binding changes at x1/y1
```

Stage 2: rollout guardrail.

Coverage should not be interpreted as useful if duplicate reduction comes from:

- lower recall;
- worse crowded or overlap slices;
- more invalid parses;
- premature EOS;
- object-count truncation.

First positive interpretation requires both:

```text
1. coverage changes the intended row-conditioned binding surface;
2. rollout does not win by suppressing hard objects or destabilizing output.
```

Minimal ablation set:

- baseline SFT with no coverage;
- boundary-only painting;
- interior-only painting;
- boundary plus weak interior painting;
- boundary plus weak interior plus count or log-count channel;
- clean GT coverage versus corrupted-prefix coverage;
- fixed order versus random or mixed order;
- faithful re-prefill painting decode versus a later cache-compatible
  coverage-token variant.

Primary diagnostics should target the hypothesis before global metrics:

- duplicate burst rate;
- repeated same-instance emission;
- premature assistant-stop rate;
- crowded-scene recall;
- overlapping and nested-object recall;
- row-wise recovery after prefix corruption;
- residual norm ratio `||delta_F_i|| / ||F_i||`;
- coordinate-alignment overlays from norm1000 bbox through visual-token cells.

mAP, recall, and precision remain important secondary global metrics, but they
do not by themselves prove that coverage memory is the active mechanism.

## Risks

- **Patch-resolution mismatch:** exact coord-token boundaries do not survive
  into visual-token space. Boundary painting must stay soft and lattice-aware.
- **Coordinate-alignment error:** if norm1000 boxes, original image geometry,
  processor grid metadata, and post-merge token cells are misaligned, every
  downstream result becomes unreliable. This is the highest-priority debug
  validation.
- **Residual-norm drift:** alpha caps do not bound `||delta_F||` when learned
  embedding norms grow.
- **Union-memory ambiguity:** v1 cannot distinguish object identity, count, or
  class/text identity for covered regions.
- **BBox-boundary proxy:** the boundary channel marks claimed bbox extent, not
  true object contour.
- **Interior over-suppression:** weak interior coverage may still teach the
  model to ignore overlapping objects if scaled too high.
- **Teacher/rollout prefix gap:** teacher prefixes are clean; rollout prefixes
  are predicted and may contain wrong boxes.
- **Capacity confound:** if the tuning operator becomes too expressive, gains
  may come from extra capacity rather than coverage.
- **Order confound:** sorted and random SFT can produce different behavior for
  reasons unrelated to coverage.
- **Decode cost:** row-boundary re-prefill may be slow in dense scenes.
- **Metric ambiguity:** COCO-style partial annotation can make real extra
  objects look like false positives.

## OpenSpec Boundary

Do not create or update OpenSpec for this design yet.

OpenSpec becomes appropriate only after we decide to stabilize at least one of:

- a config schema for row-conditioned coverage;
- a supported training tensor-flow behavior;
- a supported rollout re-prefill behavior;
- artifact names for coverage descriptors or mechanism probes;
- metric semantics for coverage-specific diagnostics.

Until then, this document and the progress direction note are the correct
research planning surfaces.
