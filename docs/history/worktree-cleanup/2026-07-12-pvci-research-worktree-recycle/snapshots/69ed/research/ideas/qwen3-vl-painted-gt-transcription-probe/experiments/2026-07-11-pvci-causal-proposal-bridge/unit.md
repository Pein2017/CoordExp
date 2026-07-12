---
title: PVCI Causal Proposal Bridge - Correlation versus Causal Use
description: Tests whether an explicitly supervised pre-row visual-instance proposal changes free-rollout behavior only when the model must causally consume that proposal during phrase and geometry generation.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-11-pvci-causal-proposal-bridge
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - qwen3-vl
  - visual-binding
  - proposal
  - causal-bridge
  - teacher-forcing
updated: 2026-07-12
---

# PVCI Causal Proposal Bridge - Correlation versus Causal Use

## Question

Under a fixed Qwen3-VL 2B, compact detection serialization, geo-sorted
teacher-forcing policy, and matched continuation-training budget, does an
explicitly supervised **pre-row, class-agnostic visual proposal** improve
own-prefix free-rollout behavior:

1. merely because the auxiliary loss reshapes the representation; or
2. specifically because the predicted proposal is causally consumed by the
   decoder while it generates the object phrase and coordinates?

The unit tests a training principle, not a final architecture. `Proposal` is a
provisional experimental handle for a normalized map over Qwen's existing
visual tokens. It is not a standalone box/class detector, a learned slot bank,
an external proposal model, or a new textual proposal sequence.

## Executed representation-screen status — 2026-07-12

The matched A/B/C 512-step training screen completed under the frozen
contract. The in-sample v2 gate returned `in_sample_learnable` and the
complete held-out five-control panel returned `promote`; see
`representation-readiness-audit-2026-07-11.md` and
`heldout-representation-results-2026-07-12.md` for receipts and caveats.

The held-out result promotes the proposal representation as a follow-up
research handle only. Candidate C was selected by the frozen priority order
`(C,B)`; B and C both passed all 29 applicable checks, so this is not evidence
that C is superior to B. The tie-aware support policy makes q2 unattainable on
the observed cohort (`[972,972,1014]` cutpoints); q1, q3, and q4 are the
attainable strata and were required.

This unit remains `promotion_status: not_promoted` for causal behavior. No
own-prefix rollout, detection, recall, duplication, STOP, or final-architecture
claim is made by the representation screen. The next authorized phase is the
natural 320-image own-prefix causal/safety panel with RP1.10 primary and RP1.00
secondary controls.

## Decision Relevance

- Functional uncertainty: whether pure token-level CE leaves current-instance
  selection/binding underconstrained, and whether an explicit intermediate
  visual variable can reduce that ambiguity in behavior rather than only in a
  probe.
- Costly choice this result could change: whether to invest next in a causal
  visual-binding bridge, instead target commit/coverage/STOP, open stronger
  visual adaptation, or eventually consider object slots/proposals.
- Outside this unit's scope: a final detector architecture, an explicit
  coverage ledger, slots/entity registers, a new visual backbone, an external
  detector, cardinality supervision, STOP loss, EOS masking, rollout/DAgger,
  PU learning, RL, duplicate unlikelihood, and policy optimization.
- Architecture posterior rule: every terminal outcome remains
  `not_promoted`. A positive handle authorizes a narrower follow-up, not a
  production design.

## Current Evidence Boundary

The unit starts from the following bounded evidence:

- Correct visible visual designation can strongly control which object row is
  emitted. This identifies a useful causal handle, not a deployable interface.
- The long-trained CE-dominant step-4887 checkpoint exhibits a one-step
  spatial commit/anti-revisit effect. Generated RAW and canonical SNAP rows
  suppress the exact committed row in `10/10` attributable attempts, and a
  zero-coordinate-token-overlap near row in `9/10`:
  [Endogenous Commit Write-Read Bridge](../2026-07-11-pvci-endogenous-commit-write-read-bridge/).
- A native completed row can suppress its committed candidate and raise
  conditional scores for annotated-uncovered candidates in an exact-replay
  subset, but STOP rises concurrently and successful free continuations remain
  compatible with geo-sorted traversal:
  [Native Commit-to-Annotated-Uncovered Redistribution](../2026-07-10-pvci-native-commit-to-uncovered-redistribution/).
- Generic annotation-empty coordinate occupancy was not sufficient in the
  source-swap panel (`J=7/8`, `K=0/8`, `E=1/8`), but current trajectory and
  selection state remained confounded:
  [Commit-Field Annotation-Empty Source Swap](../2026-07-11-pvci-commit-field-source-swap/).
- A stable same-description control did not support a target-specific
  current-selection gate: `SelectionSpecificSwitch=0/4`, with mixed T-minus-L
  differences:
  [Selection-Aligned Prefix-State Commit Gating](../2026-07-11-pvci-selection-aligned-prefix-state-gating/).
- Geometry-sorted training modestly outperforms random ordering in prior
  experiments, especially as object count grows. This is treated as evidence
  for a useful implicit traversal/regularity prior, not proof that ordering is
  the root mechanism.
- Attention and hidden-state correlations are not accepted as causal binding
  evidence without removal, swap, or source-control interventions that change
  the real row logits or free generation.

Therefore, the live claim is not that CE learns no commit. The live claim is:

> CE-dominant training has not demonstrated a durable, object-indexed state
> that cleanly factorizes current-instance selection and causally binds the
> whole phrase/geometry row during own-prefix rollout.

This unit targets the `SELECT/BIND` portion of that uncertainty. It does not
assume that solving `SELECT/BIND` will solve `COMMIT/COVERAGE/STOP`.

## Fixed Terminology and Causal Order

For row `t`:

- `P_t`: prompt plus only the already available prior rows, ending at the legal
  boundary before any token of the current row, including before
  `<|object_ref_start|>`;
- `H_t^0`: the cloned output of text decoder layer 0 before Qwen's first
  deepstack residual addition;
- `V_t={v_i}`: the rows of `H_t^0` at the physical image-token positions
  owned by the same example/image/packed segment as `P_t`; these states retain
  the installed runtime's merged visual-token order, and neither proposal
  normalization nor pooling may cross an example, image, or packed-segment
  boundary;
- `q_t`: a normalized class-agnostic proposal map over `V_t`, computed from
  `P_t` and `V_t` only;
- `B_t`: the frozen positive visual-token footprint bag derived from the
  current annotated target box for loss/evaluation only;
- `p_t`: the proposal-pooled visual representation;
- `C_on`: the C checkpoint with its proposal-derived visual feedback active;
- `C_off`: the same C checkpoint and decode orchestration with feedback
  disabled exactly, retaining all other parameters and policies.

The permitted direction is:

```text
image visual tokens + prior-row prefix P_t
  -> predicted proposal q_t
  -> proposal-pooled visual state p_t
  -> bounded layer-0-to-layer-1 row-residual pulse
  -> current phrase and coordinates
```

The following are invalid:

- reading any current-row phrase or coordinate token before computing `q_t`;
- feeding GT boxes, GT masks, GT token IDs, GT object IDs, GT raster maps, or
  oracle pooled regions into C at inference;
- computing a proposal after the phrase and calling it next-object selection;
- using a standalone external detector, slot bank, learned object queries, or
  second visual encoder;
- applying a bridge only at the initial prompt and claiming multi-row control;
- silently retaining the bridge across rows and thereby introducing an
  undeclared persistent ledger.

## Minimal Proposal Handle

The planned minimal query-key handle is:

```text
q_query_t = W_q h(P_t)
k_t,i     = W_k v_t,i
a_t,i     = softmax over i in V_t(q_query_t dot k_t,i / sqrt(d))
p_t       = sum over i in V_t(a_t,i v_t,i)
```

For this screen, `d=256`. `h(P_t)` and `V_t` are cloned layer-0 outputs. A
parameter-free float32 layer normalization (`weight=None`, `bias=None`,
`eps=1e-5`) is applied before bias-free `W_q` and `W_k`. Query/key projections,
scaled dot products, softmax, positive-bag mass, entropy, and denominators are
computed in float32. `W_q`, `W_k`, and `U` parameters are stored in float32;
there is no trainable normalization parameter. The installed Qwen layer-0
output must be cloned because the subsequent deepstack update is in-place in
`transformers==4.57.1`.

`a_t` is normalized only over `V_t`, never over visual tokens from another
packed example or image. It is the proposal map and is not declared to be an existing attention
head or an explanation of Qwen's native attention.

### Positive-bag target

Before any training result is inspected, implementation must materialize and
hash one spatial-footprint rule mapping every merged Qwen visual token to the
resized image coordinate system. The frozen rule is:

```text
intersection_area(merged_token_cell, target_box)
------------------------------------------------ >= 0.25
 min(merged_token_cell_area, target_box_area)
```

Coordinates use the repository's existing
`round(coord_bin * image_extent / 1000)` conversion. For the current 32-pixel
merged footprint, this score-blind IoM rule was selected before model scores
were available because token-area-only overlap at the same threshold drops
`37.094%` of COCO-small boxes, whereas IoM leaves only `45/113,558`
zero-positive objects (`0.0396%`) on the frozen proposed cohort. The
supervision is positive-bag rather than outside-negative BCE:

```text
PBMass_t = sum_{i in B_t} a_t,i
L_prop   = -log(PBMass_t + epsilon)
```

with `epsilon=1e-6` and proposal-loss weight `0.10`. No equivalent alternative
is permitted in the first screen. The implementation must not tune the bag
rule, threshold, loss weight, or epsilon after viewing model scores.

`L_prop` is reduced only over rows with a nonempty `B_t`. Zero-positive rows
contribute neither numerator nor denominator and must have exactly zero
proposal-loss gradient; they remain in data/rollout counters. Every reduction
records `eligible_count`, `skipped_zero_positive_count`, condition, size/count
stratum, and the all-row denominator separately.

The footprint rule must be score-blind and use verified post-merge token
footprints. Rows with no positive token under the frozen rule are ineligible
for representation claims; they remain visible in the data and rollout
denominator and are never silently repaired. Materialization is a contract
failure if the observed zero-positive rate exceeds `1.0%` overall or `2.0%` in
any declared size/count stratum, far above the geometry-only expectations.

Every bag and proposal receipt must bind `example_id`, image identity,
packed-segment identity, row/object identity, visual-token physical interval,
and image-grid/merge metadata. A two-image packed fixture must prove that the
proposal denominator, `B_t`, pooling, and bridge update remain segment-local.

This loss says that a token is or is not part of the **current target's**
proposal support. It does not label all other image regions as semantic
background or assert that unannotated COCO regions contain no objects.

Raw bag mass is area-biased. Report at minimum:

- bag mass and `hit@K`;
- entropy and top-k concentration;
- residual bag mass relative to expected mass from bag area;
- target specificity versus same-image other boxes;
- same-area/center/aspect matched controls;
- small-object and zero-positive rates;
- overlapping and same-class bag collisions as a separate stress stratum.

An all-token-high or area-only map is a failed proposal representation even if
raw `PBMass` is numerically large.

## Minimal Causal Feedback Handle

The installed-runtime trace places literal per-row post-scatter visual-token
mutation on HOLD: standard cached generation retains decoder K/V rather than
the raw visual states needed to apply a new post-scatter delta, so that route
would require multimodal re-prefill or unverified cache surgery for every row.

The frozen first-screen handle is instead a functional `split_layer_bridge`.
Let `[s_t,e_t)` be the physical target-token interval from
`<|object_ref_start|>` through and including `<|box_end|>`. The causal input
positions that predict that row are `[s_t-1,e_t-1)`. After decoder layer 0,
the implementation clones `H_t^0`, computes `a_t` and `p_t` from the legal
boundary state at `s_t-1` and segment-local visual positions, lets Qwen apply
its normal first deepstack residual, and then injects:

```text
raw_delta_t = gamma * U(p_t)
reference_t = H_t^0[s_t-1]
ratio_t = ||raw_delta_t||_2 / (||reference_t||_2 + 1e-6)
scale_t = min(1, 0.05 / (ratio_t + 1e-6))
delta_t = raw_delta_t * scale_t
H'_j = H_j + delta_t  for j in [s_t-1, e_t-1)
```

Here `gamma=0.10`; `U` is one shared zero-initialized `2048 -> 2048` linear
projection. Projection and relative-norm-cap arithmetic run in float32 and the
final bounded delta is cast to the model hidden dtype immediately before the
out-of-place addition. `W_q`, `W_k`, and `U` are registered in all arms with
the same initialization/shape receipt; A keeps them frozen, B trains `W_q` and
`W_k` through `L_prop` while `U` stays frozen, and C trains all three under the
frozen gradient policy. This is an experimental text-residual delivery seam,
not a claim that Qwen natively implements this bridge or a final architecture.

The cap is one scalar per row over the hidden dimension only. Its reference is
the cloned pre-deepstack boundary vector from the same packed segment; sequence
length, other rows, other examples, and the number of injection positions do
not enter the cap. Record raw-delta norm, reference norm, ratio, scale,
post-cap norm, cap-active flag, and clipped-row fraction.

The bridge must:

- use predicted `a_t`, not an oracle map, in every metric-bearing C rollout;
- be applied before the current row phrase;
- remain available at every causal input position that predicts description,
  `<|box_start|>`, all coordinate slots, and `<|box_end|>`;
- be applied once per causal input position in the declared half-open row
  interval, with one logical row application receipt;
- reset and recompute at the next legal row boundary;
- record the proposal entropy, positive-bag mass, delta norm, affected token
  count, cache/re-prefill boundary, application count, and reset receipt;
- expose `C_on` and exact `C_off` behavior from the same checkpoint. C-off must
  execute the identical row-wise re-prefill/cache/reset path and application
  schedule with an exact zero/no-op delta; bypassing the bridge orchestration is
  not a valid C-off control.

The train-time owner must be a functional extension of the installed Qwen text
forward loop, not a stateful layer hook. Non-reentrant gradient checkpointing
replays decoder layers, and a hook-based prototype was observed to fire twice;
the installed deepstack path also mutates early-layer outputs. C-off must take
the identical functional loop and row schedule with an exact zero delta.

At inference, layer-0 visual-position states are captured once during
multimodal prefill. The initial prompt tail and every emitted `<|box_end|>` are
legal proposal boundaries. The controller recomputes from that boundary's
current layer-0 state, applies the new pulse while the next row is processed,
and resets at the next `<|box_end|>`. It never mutates cached visual K/V. Prior
bridge effects live only in layer-1-and-above text cache entries and therefore
do not enter the next proposal's layer-0 query; this limitation is declared as
part of the tested handle rather than hidden as a persistent ledger.

The inference controller has a fixed `max_row_tokens=32`. A nested
`<|object_ref_start|>`, terminal token before `<|box_end|>`, repeated
`<|box_end|>`, or row timeout marks the row/controller state invalid, zeros the
active pulse immediately, and records a reasoned fail-closed reset. It does not
infer a new proposal until a legal subsequent boundary exists; max-generation
cutoff clears all state. Malformed/truncated rows stay in raw rollout and safety
denominators, and no pulse may survive past the recorded invalid/reset event.

## Three Matched Training Conditions

All three arms start from the same attested geo-sorted, long-trained
step-4887 pure-CE/type-gated adapter and use the same base model, tokenizer,
special-token payload, image pipeline, prompt, serialization, training-image
manifest, row ordering, optimizer/update budget, seed, checkpoint-selection
rule, base CE/type loss, and eligible-row accounting.

| Arm | Training signal | Runtime feedback | Scientific role |
|---|---|---|---|
| A `ce_resume` | Existing matched CE/type objective only | None | Controls for more data, updates, and adapter drift. |
| B `proposal_aux` | Same base objective plus `L_prop` | Proposal measured but never fed back | Tests correlation/representation shaping. |
| C `proposal_bridge` | Same base objective plus the same `L_prop`; bridge-gradient policy frozen and logged | Predicted proposal causally feeds the current row | Tests causal consumption. |

### Arm D is explicitly postponed

No covered-region contrastive loss, coverage logit bias, or native-attention
suppression is included in A/B/C. The positive-bag softmax is already weakly
contrastive because increasing current-target mass decreases mass elsewhere.
Before introducing stronger repulsion, this unit must measure whether proposal
mass remains concentrated on previously emitted regions.

Record for every eligible row:

```text
CurrentMass_t = sum_{i in current target bag} a_t,i
CoveredMass_t = sum_i C_t(i) * a_t,i
CurrentCoveredMargin_t = target bag score - covered bag score
```

`C_t(i)` is a monitoring-only soft field derived from prior prefix boxes and
verified visual-token footprints. It is never applied to Qwen native attention,
proposal logits, or bridge features in this unit. Report raw and
overlap-adjusted covered mass, per-prior-row mass, same-class covered mass, and
mass on tokens shared by covered/current boxes.

If C learns a target-specific proposal and causally improves row binding but
covered mass and duplicate reselection remain high, a separate future Arm-D
unit may test bag-level target-versus-covered contrast. If covered mass falls
naturally, stronger repulsion is unnecessary. No observed result in this unit
may silently activate D.

Arm B must never receive proposal-derived visual feedback or bridge gradients.
If C introduces additional parameters, A and B should carry identically sized
disabled/no-op modules where the current optimizer/checkpoint contract permits;
otherwise the parameter/gradient-budget difference is an explicit confound and
can only be resolved through C's within-checkpoint `C_on/C_off` and semantic
shuffle controls.

The C bridge-gradient policy must be frozen before launch. If row CE
backpropagates through proposal and bridge, C is an end-to-end training
condition. Therefore:

- `C_off > B` is classified as an auxiliary/end-to-end training effect;
- only `C_on > C_off` with target-specific shuffle/null controls supports
  runtime causal use;
- `C > A` alone never supports a causal bridge claim.

## Frozen Teacher-Prefix Robustness Mixture

All A/B/C training arms receive the identical deterministic view manifest.
Canonical views retain the existing full-sequence supervision. Jitter and
duplicate-history conditions are dedicated **row-state views**: they serialize
historical rows plus one untouched current row, mark every historical token
and the terminal `<|im_end|>` as ignored, and supervise only the current row's
description, schema, and coordinates. This is required so corrupted history is
conditioning context rather than a target the model is taught to reproduce.

Perturbations apply only to already completed historical rows; the current
target row, image, GT proposal target, realized geo-sorted row identity, and
current-row target digest remain unchanged. Perturbed views choose a target
row uniformly by a stable hash from indices `1..N-1`; single-row images are
canonical-only. The dedicated renderer preserves the original realized order
even if jitter changes a prior box's top-left anchor. It may not re-sort after
corruption or weaken canonical `RawExample` object-ID uniqueness.

The initial screening mixture is frozen as:

```text
80% canonical prefix
15% bounded historical-coordinate jitter
 5% structured historical-row corruption
```

The condition manifest is static for this first screen so it remains compatible
with the repository's content-addressed packing cache. Condition, target row,
and perturbation are derived from stable hashes of source example identity,
policy version, and seed `20260711`; they do not vary by epoch/update and must
not depend on model scores. A/B/C must consume byte/token-identical condition
manifests. Any authored epoch/update-varying condition with a static pack cache
is a contract failure.

Within each count bin, all images with at least two rows are ranked by
`(SHA256("pvci-condition-v1|seed=20260711|image_id=<id>"), image_id)`, where
`<id>` is the unpadded decimal COCO source `image_id` (for example `9`), not
the formatted `RawExample.example_id`.
The first declared jitter quota receives jitter, the next duplicate quota
receives duplicate-history, and every remaining image is canonical; one-row
images are canonical and are excluded before this perturbed ranking. Target
row and perturbation candidates use distinct domain-separated hashes. Two
independent materializations must produce byte-identical manifests before the
manifest digest enters the pack-cache fingerprint.

The exact per-count-bin condition quotas are frozen as:

| GT count | Total | Canonical full | Jitter row-state | Duplicate row-state |
|---|---:|---:|---:|---:|
| 1-2 | 2,560 | 2,048 | 384 | 128 |
| 3-4 | 2,560 | 2,048 | 384 | 128 |
| 5-8 | 2,560 | 2,048 | 384 | 128 |
| 9-16 | 2,560 | 2,048 | 384 | 128 |
| 17-32 | 1,536 | 1,229 | 230 | 77 |
| 33+ | 512 | 410 | 77 | 25 |
| **Total** | **12,288** | **9,831** | **1,843** | **614** |

The active segment-balanced normalizer gives a row-state segment and a full
sequence equal segment weight despite different supervised-token counts. That
is accepted as the frozen robustness-mixture semantics, not hidden as token
mass matching. Report sample, visual-token, text-token, supervised-atom, EOS,
and packed-segment mass by condition; any enumeration regression is part of
the A control and safety gate.

### Bounded coordinate jitter

- choose one prior row deterministically;
- retain its phrase and wrapper tokens;
- draw center offsets independently in `[-0.15,+0.15]` times the prior box
  width/height and log-width/log-height offsets in `[-0.12,+0.12]`;
- generate up to 16 deterministic candidates, clip to valid image bounds, and
  accept the first valid nondegenerate box with IoU in `[0.55,0.90]` and at
  least one changed coordinate bin;
- record original and perturbed coordinate bins, pixel boxes, IoU, center
  displacement, scale change, and whether any coordinate token remained equal;
- on no acceptable candidate, emit a typed failure receipt and abort strict
  manifest materialization rather than silently falling back to canonical,
  replacing the sample, or retuning the range. A geometry-only preflight must
  prove that the frozen cohort has zero such failures before packing.

### Structured historical-row corruption

The initial corruption family is limited to one deterministic duplicate-history
event: repeat one already completed row once immediately before `P_t`, while
keeping the intended current target unchanged. Both historical occurrences
are ignored context and preserve the original object identity in view metadata;
the renderer uses occurrence IDs rather than constructing an invalid duplicate
`RawExample`. This tests recovery from a common off-policy duplicate without
introducing a new GT identity or declaring an unannotated region background.

Phrase/geometry swaps, wrong-object rows, row deletion, large box corruption,
and background boxes are monitoring-only counterfactual panels in this unit.
They cannot enter the first screening training mixture without a doc amendment
and a new leakage/meaning audit.

### Prefix robustness metrics

Report every representation and rollout metric separately for:

- canonical prefix;
- coordinate-jitter prefix;
- duplicate-history prefix;
- own-prefix free rollout;
- monitoring-only wrong-object and phrase/geometry counterfactual panels, when
  available.

The main result cannot be driven solely by canonical teacher prefixes. A model
that improves only under canonical history is classified as
`representation_only` or `mixed_or_bridge_unlocalized`, not a robust causal
bridge.

## Ordering and Decode Policy

### Primary unit scope

- Training/render order: `geo_sorted` in A, B, and C.
- Primary free decode: greedy with `repetition_penalty=1.10`.
- Secondary fixed decode crossover: the same checkpoint and images with
  `repetition_penalty=1.00`.
- `rp=1.10` is treated as a salvaged token-level anti-duplication heuristic,
  not a theoretically motivated object-transition mechanism.
- Neither RP setting may be selected or retuned after observing which looks
  better.

The RP crossover must report structural-token, same-class, duplicate,
termination, parser, and count effects. A result present only under RP1.10 is
`decode_specific_bridge`, not a general learned-transition claim.

### Random-order comparator

The user is training a random-order pure-CE model separately. Its exact config,
seed, checkpoint, resolved training order, and artifacts are external
provenance for this unit until they exist and are attested. It may describe the
ordering phenotype but cannot be treated as a concurrent A/B/C causal arm.

Only if geo-sorted C passes its causal gate should a separate research unit
train `random_order + proposal_bridge`. That later 2x2 interaction asks whether
an explicit proposal reduces reliance on the geo-sorted implicit traversal
prior. This unit makes no cross-order universality claim.

## Competing Hypotheses

### H1: pre-row visual proposal is learnable

- Expected signature: B improves positive-bag mass, hit/rank, and target
  specificity beyond area, position-only, prefix-depth, image-mean, and
  token-content-shuffle baselines on held-out images.
- Falsifier: no residual visual signal; all-token saturation; proposal follows
  only row depth/raster/box area; or image-content shuffling preserves it.

### H2: representation shaping is sufficient

- Expected signature: B improves own-prefix free-rollout target behavior over
  A even though the proposal is never fed back; C-off is similar to B.
- Falsifier: B learns a strong proposal readout but A and B remain
  behaviorally equivalent.
- Bounded interpretation: `representation_only` or
  `auxiliary_training_effect`, not runtime bridge causality.

### H3: proposal must be causally consumed

- Expected signature: C-on improves target-specific own-prefix behavior over
  both B and C-off; the gain attenuates under proposal removal, image-shuffled
  proposals, within-image token permutation, position-only feedback, and
  norm-matched random feedback.
- Falsifier: C-on equals C-off/B; shuffled or null feedback works equally well;
  gains occur only with canonical teacher prefixes, after phrase tokens, or
  with an oracle GT proposal.

### H4: geo-sorted/raster shortcut explains the proposal

- Expected signature: row index, prefix length, prior coordinates, token
  position, or the learned raster successor explains proposal and rollout
  behavior without image content.
- Falsifier for the shortcut: image-content and same-depth object controls move
  the target-specific proposal/row while position-only and raster controls do
  not.

### H5: positive-bag supervision is not instance-specific enough

- Expected signature: bag metrics rise, but neighboring/same-class instances
  collide; phrase and coordinates bind different objects; or duplicates rise.
- Falsifier: target-versus-neighbor specificity and full-row phrase/geometry
  consistency survive crowded same-class controls.
- Outcome: `positive_bag_insufficient`; do not automatically escalate to slots.

### H6: proposal feedback is unstable or harmful

- Expected signature: C amplifies wrong proposals, increases STOP, duplicates,
  invalid rows, or coordinate drift, or a large bridge norm acts as an
  undeclared new predictor.
- Falsifier: active C improves target behavior within norm and safety
  guardrails, while null/shuffled/norm-matched random controls do not.

### H7: proposal is not the primary free-rollout bottleneck

- Expected signature: proposal and row-level phrase/geometry binding improve,
  but predicted count, early stopping, and dense recall do not.
- Interpretation: `SELECT/BIND` improved; next work should test
  `COMMIT/COVERAGE/STOP` rather than enlarging the proposal module.

### H8: COCO partial labels prevent a directional claim

- Expected signature: apparent gain or harm is dominated by unmatched extras,
  annotation density, or ambiguous crowded regions.
- Outcome: `label_limited` or `inconclusive`, never automatic hallucination or
  successful discovery.

## Required Controls

### Representation and shortcut controls

- proposal timing receipt proving no current-row token was visible;
- example/image/packed-segment ownership receipts and a two-image packed
  isolation fixture;
- position-only and prefix-depth predictors;
- image-mean visual representation;
- image-token content shuffle preserving token positions/grid shape;
- same-area/center/aspect matched-object metric controls;
- no-prefix, one-prior-row, and two-prior-row depth strata;
- small, crowded, overlapping, and same-class strata;
- bag-area-normalized metrics and all-token saturation checks.

The representation-control generator is frozen as
`pvci-representation-controls-v1` before any 512-step score is observed.  All
controls reuse the exact native row boundary, segment-local visual support,
FP32 affine-free normalization, `W_q/W_k`, proposal dimension, and softmax.
They change only the declared query or visual-state input; no temperature or
control hyperparameter may be tuned after scores are visible.

- `native`: use the legal pre-row query and the aligned layer-0 visual states
  without modification.
- `raster_position`: replace every visual state with a fixed 2048-D
  sinusoidal encoding of that merged cell's normalized `(center_x,center_y)`.
  Frequencies are `2*pi*2^k` for `k=0..8`; the 36-D base vector is ordered
  `[sin(f_k*x),cos(f_k*x),sin(f_k*y),cos(f_k*y)]` by increasing `k`, tiled in
  that order and truncated to 2048 dimensions.  There are no learned or
  random parameters.  The native query and native `W_q/W_k` are retained.
  This is a position/raster diagnostic, not an oracle successor proposal.
- `row_depth`: replace the native query with the same fixed sinusoidal family
  applied only to `(min(prior_completed_rows,31)/31,
  min(prefix_token_count,4096)/4096)`, repeated then truncated to 2048
  dimensions.  The native visual states and native `W_q/W_k` are retained.
  `prior_completed_rows` is the number of rendered prior-row occurrences in
  the frozen row-state view (including a deliberate duplicate-history
  occurrence); `prefix_token_count` is the segment-local count through the
  legal pre-row query position, equal to `target_start-segment_start`.
  `gt_object_count`, the current target box/identity, future annotations, and
  proposal scores are forbidden inputs.  Because this synthetic query may be
  out of distribution, it is a diagnostic null and cannot by itself establish
  a residual claim.
- `image_mean`: replace every visual state by the exact FP32 mean of the
  target image's aligned visual states.  The resulting spatial proposal is
  expected to be uniform; this is an image-global/no-localization null and is
  interpreted together with the saturation and bag-area controls.
- `image_content_shuffle`: permute visual states across the unchanged merged
  cells by sorting local indices on
  `SHA256("pvci-representation-controls-v1|image_content_shuffle|<row_id>|<i>")`.
  If the permutation is identity and support has more than one cell, rotate
  once.  The query, grid, target bag, and score support remain unchanged.
  This destroys content-to-cell association; because Qwen visual states can
  carry position, it is not called a texture-only shuffle.

Every control record carries the recipe version, code fingerprint, support
and grid identity, query/visual input hashes, and a transform receipt.  An
unsupported, identity-by-accident, nonfinite, wrong-support, or provenance-
mismatched control is `contract_fail`, never a native row relabeled as a
control.  The same controls are produced for A, B, and C on identical rows.

Same-area, same-center, same-aspect, combined-geometry, and same-class
distractors are score-blind metric controls selected only from other annotated
bags after proposal generation, with stable object-ID tie breaking.  They
never alter the proposal and are decision-bearing for specificity.  A
same-depth comparison pairs cross-image rows by
observed prior-row count, prefix-length bucket, count bin, and visual-support
quartile; it may not use future object count or target identity.  Same-depth
matching is a reported nuisance diagnostic in this first screen, not an
additional promotion threshold.

Support-quartile assignment is nearest-rank and value-based. If adjacent
nearest-rank cutpoints are tied, the duplicate boundary is collapsed by the
tie-aware value assignment; equal support sizes are never split by row rank.
The gate records cutpoint boundaries plus attainable and unattainable nominal
quartiles, and requires only attainable strata. A genuinely missing attainable
stratum remains a fail-closed rerun; an unreachable nominal quartile does not
trigger a rerun by itself.
For target/candidate boxes define `d_area=abs(log((area_j+1e-6)/
(area_t+1e-6)))`, `d_center=euclidean(center_j-center_t)/image_diagonal`, and
`d_aspect=abs(log((aspect_j+1e-6)/(aspect_t+1e-6)))`.  The three individual
controls minimize their named distance; `combined_geometry` minimizes
`d_area+d_center+d_aspect`.  Remaining distances and then string object ID are
stable tie breakers.  No distance is measured in raw pixels.

### Causal bridge controls

- A, B, C-on, and exact C-off;
- zero/no-op bridge;
- proposal from another image with matched token count;
- within-image proposal-token permutation preserving the score histogram;
- norm-matched random feedback;
- bounded delta-norm and application/reset receipts;
- phrase, structure, `x1`, `y1`, `x2`, `y2`, and full-row effects reported
  separately.

Control transforms are frozen as follows and reuse the identical row schedule,
cap, application count, and reset path:

- `removal/C_off`: exact zero delta after computing the native C proposal;
- `another_image`: pair each image by stable hash within prefix-depth and
  nearest merged-token-count strata; compute the pooled state from the donor's
  visual states using the target boundary query, then apply it to the target;
- `token_permutation`: apply a stable-hash permutation to the association
  between `a_t` and same-image `V_t`, preserving score histogram and support;
- `position_only`: replace visual content in pooling with a deterministic
  2048-D sinusoidal encoding of normalized merged-cell `(x,y)` positions,
  then norm-match the pooled vector before `U`;
- `norm_matched_random`: deterministic per-example/row Rademacher vector in
  pooled-state space, scaled to the correct pooled-state L2 norm before `U`;
- `image_content_shuffle` for representation only: permute `V_t` content by
  the same stable rule while preserving physical positions and prefix.

For every negative causal control, report the paired interaction
`(C_on-C_off) - (C_control-C_off) = C_on-C_control`. Its image-bootstrap CI
lower bound must exceed `+0.02` on matched-row precision in addition to
the control's own upper-bound rule. A control that cannot be constructed for a
row is skipped with a predeclared reason, never replaced after scores are seen.

An oracle GT-raster/GT-proposal bridge may be run only as a clearly labeled,
non-deployable ceiling. It can diagnose bridge capacity but can never support
C's metric-bearing inference claim.

## Incomplete-Annotation Safeguards

- Annotated target boxes provide positive current-target supervision only.
- No loss may label all unannotated visual tokens as verified background.
- Unmatched free predictions remain `unknown/other` unless independently
  adjudicated; they are neither automatic hallucinations nor automatic
  successes.
- Annotation exhaustion is not equated with complete visible-object coverage.
- Dense-image STOP, residual count, and global background claims remain outside
  this unit.
- Primary behavioral reporting separates labeled-GT recall, duplicate/invalid
  outputs, and unmatched extras. Official COCO AP/mAR is secondary context.
- The sample manifest and strata are frozen by annotation/count metadata before
  proposal or rollout scores are inspected.

## Planned Evidence Scope

- Checkout: `/data/CoordExp/.codex/worktrees/69ed/CoordExp`.
- Current branch: `codex/continue-handoff-session`.
- Initialization checkpoint:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/checkpoint.json`.
- Historical primary baseline rollout:
  `/data/CoordExp/outputs/infer/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_step4887_val200_bsz4_temp0_rp1p10_max3084_8gpu/`.
- Historical baseline context, not a new comparison result: `AP=0.41555`,
  `AP50=0.58076`, `AP75=0.43865`, `AR100=0.50220`; F1-like localization at
  IoU 0.50 has 1,294 raw predictions, 891 localized TP, 284 localized FP, 553
  localized FN, precision `0.7583`, recall `0.6170`, and F1 `0.6804`.
- Historical baseline decode provenance: greedy temperature 0, RP1.10,
  `max_new_tokens=3084`, eight ranks, strict expected parser. Its recorded FA2
  lineage is not assumed equivalent to current SDPA exact-replay probes.
- Frozen train source:
  `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl`,
  `117,266` rows, SHA256
  `e61a7f7e2bddadeb3ffe5550f83feac1ecccc99149c7f98271113e9a3f9580f8`.
- Frozen train cohort: `12,288` image-disjoint rows with count-bin quotas
  `2,560/2,560/2,560/2,560/1,536/512` for `1-2/3-4/5-8/9-16/17-32/33+`.
  Within each bin, rank by
  `SHA256("pvci-screen-v1|seed=20260711|image_id=<id>|bin=<bin>")`, then image
  ID, where `<id>` is the unpadded decimal COCO source `image_id`. Preserve
  source-row order after membership selection. The frozen identity payload is
  the numerically sorted decimal image IDs, one per line, including a terminal
  newline. Its expected SHA256 is
  `7faf0e26f832f67b1e7eedae821e1f534a623c10b577bd89f3b754404d41b01c`.
  A/B/C must materialize the same view manifest and verify this source/cohort
  identity before packing. No outcome-adaptive expansion is allowed.
- Frozen continuation budget: `max_steps=512`, effective batch size `24`, one
  seed (`17`), final step `512` as the primary screening checkpoint, with
  diagnostic saves at `128/256/384`. `max_steps`, not authored `epochs`, is the
  schedule authority; actual packs, segments, images, and realized passes must
  be receipted. The 512-step arm is non-resumable because the current
  checkpoint owner does not save optimizer/scheduler/scaler/iterator state;
  diagnostic checkpoints are evaluation-only. An interrupted arm restarts
  from the frozen initialization and seed. One seed supports a mechanism
  screen only.
- Frozen optimization: BF16 base execution; float32 proposal/bridge/loss
  islands; language DoRA LR `2e-5`; special-token delta LR `1e-5`; proposal and
  bridge LR `5e-4`; zero weight decay; AdamW betas `(0.9,0.999)`, epsilon
  `1e-8`; cosine schedule with `0.05` warmup ratio; max gradient norm `1.0`.
  All arms warm-start the exact step-4887 DoRA and special-token payloads and
  must prove tensor equality before update 1.
- Backend scope: matched training and primary inference require exact
  Qwen3-VL-2B FA2 receipts plus the declared packed-isolation probe. SDPA is an
  additional parity/debug backend, not a silent replacement. If exact FA2 is
  unavailable or its branch proof fails, launch closes `contract_fail` rather
  than making a backend-independent claim.
- Planned primary evaluation cohort: the same fixed val200 image IDs and input
  JSONL for every arm. The score-blind dense diagnostic adds 120 disjoint
  validation images, 20 per count bin, ranked by
  `SHA256("pvci-valdiag-v1|seed=20260711|image_id=<id>|bin=<bin>")`; its expected
  image-ID manifest SHA256 is
  `e7929e3415f89183178d4d20d9de3aeaa368b05b25b161bb903bb127050c7602`.
- Executed held-out preflight:
  `/data/CoordExp/outputs/probes/coordexp_swift/pvci_proposal_bridge_heldout_preflight`.
  It contains 320 unique images, 906 row-state views, condition counts
  `320/293/293` for canonical/jitter/duplicate-history, 27 singleton images,
  and 27 explicit unavailable rows for each history-dependent condition.  The
  combined source-row SHA256 is
  `f21dc678bda7eb461ffd1b8d4f9d8889b9aafd9da955ff347ba0058ab622a3b1`,
  combined source-ID SHA256 is
  `5dd8d62e16038ceb7cb47fc12e52cafe9a1d86fa3b6c67f8160049a6e71ba9c9`,
  and full view payload/file SHA256 is
  `2554969aaebb43f86a03e85a639f28cf06c0b7ff81b2763bf4c46e2cc474b75c`.
- The executed evaluation-only held-out cache descriptor is
  `/data/CoordExp/outputs/probes/coordexp_swift/pvci_proposal_bridge_heldout_preflight/cache_descriptor.json`.
  It binds cache fingerprint
  `eda43789f8c0111be0895c58ae710b21ae68c9ff75f08a1702c0ae5ebed0f4dd`,
  manifest SHA256
  `be4e5fccc416b28dd1c4986b9ee5401c9694caa99c3b619f183a8bb255bbb746`,
  906 views/segments, 114 microsteps, 3,639 proposal rows, and zero
  zero-positive rows.  It is a diagnostic cache and cannot collide with or
  replace the frozen training cache.
- Held-out teacher-forced proposal production is frozen unsharded as
  `rank=0, world_size=1` over all 114 diagnostic microsteps for each arm.
  A/B/C arms may run concurrently on separate GPUs, but one arm/control bundle
  has one global cohort/runtime receipt.  Rank-sharded held-out artifacts are
  not gate-eligible in this unit because no canonical merge contract is
  defined.
- Representation evaluation is two-stage.  The native rank-0 training-monitor
  shard under the frozen eight-rank schedule is a score-blind, explicitly
  in-sample learnability sample and may stop the unit early; it cannot
  establish held-out binding or full-cohort coverage.  Only rank 0 writes this
  research monitor in the current runtime, so no absent rank artifacts are
  inferred or synthesized.  This stage compares matched A/B/C native proposal loss,
  target mass, hit@K, saturation, and gradient/condition receipts and does not
  reserialize the approximately 93k supervised-row proposal vectors under all
  controls.  A candidate that passes the in-sample diagnostic must be
  confirmed with the complete five-control teacher-forced panel on the frozen
  val200-plus-120 cohort before representation promotion.  Validation images
  are expanded deterministically into canonical, jitter, and
  duplicate-history views using the already frozen view policy.  Every image
  with at least two objects is evaluated under every condition so the
  conditions are paired rather than assigned 80/15/5; singleton images are
  canonical-only with explicit `condition_unavailable=no_prior_object` for
  jitter and duplicate-history.  The exact expanded view count and singleton
  count are receipted from the combined val200-plus-120 source before any
  candidate score is inspected.  The expanded manifest must be materialized
  and hashed before any candidate score is inspected.  A smaller hash-selected
  training-image pilot may validate producer mechanics only and is never a
  substitute for either stage.
- Planned artifact root:
  `/data/CoordExp/outputs/research/pvci_causal_proposal_bridge/`.
- Executed A/B/C config paths, resolved configs, checkpoint hashes,
  trainable-group receipts, and data/view identities are bound by the completed
  run roots and the v2 gate receipts cited in the executed representation
  status above. Random-order artifact identity remains external/pending and is
  not part of this representation result.

Existing research artifacts are prior evidence only. They may not substitute
for matched A/B/C artifacts or be pooled across incompatible attention backend,
RP, max-token, parser, checkpoint, or runtime roots.

## Representation Metrics

Teacher-forced/canonical-prefix metrics diagnose the learned representation but
cannot establish free-rollout behavior:

- positive-bag mass, area-adjusted residual mass, hit@K, and target token rank;
- entropy, top-k concentration, and saturation;
- target-versus-other-box and same-class specificity;
- footprint-union localization and center distance;
- phrase, coordinate-slot, and full-row log-probability changes under correct,
  removed, wrong, and shuffled proposals;
- proposal accuracy by object size, token coverage, overlap, same-class crowd,
  and prefix depth;
- visual-content residual over position-only/raster baselines;
- proposal and bridge gradient norms and effective weighted loss contribution.

Coverage-competition instrumentation is mandatory even though Arm D is
postponed:

- current-target proposal mass;
- total and overlap-adjusted prior-covered mass;
- current-versus-covered bag score margin;
- same-class prior-covered mass;
- effective proposal support size and peak/pulse strength;
- fraction of top-k proposal tokens owned by current, covered, shared,
  other-annotated, and unknown regions;
- metric changes under canonical, jittered, duplicate-history, wrong-object,
  and phrase/geometry-mismatch prefixes.

## Free-Rollout Metrics

Only own-prefix rollouts can support a behavioral conclusion. Report:

- parser validity, closed-row rate, malformed/truncated output, and every drop
  counter;
- emitted object count, natural termination, and row-boundary behavior;
- labeled-GT recall, phrase match, IoU, AP, and mAR on identical scored rows;
- recall-versus-GT-count slope and count-stratified under-enumeration;
- same-description and any-description duplicate pairs/components/bursts;
- invalid geometry/format and phrase-coordinate chimera rates;
- unmatched/other extra predictions retained as their own category;
- proposal hit, entropy, and correct/wrong/removal effects by generated row;
- current-target, prior-covered, shared-overlap, and unknown-region proposal
  mass by generated row;
- C-on versus C-off, B, and A paired by image;
- RP1.10 primary and RP1.00 secondary reported separately;
- fixed-order agreement and raster-successor dependence as diagnostics.

The metric-bearing own-prefix panel starts from the image and task prompt and
uses the model's own generated prefix.  It therefore contains no
teacher-forced jitter or duplicate-history exposure label.  The held-out
representation-v2 panel is the frozen off-policy evidence for noncanonical
teacher prefixes; the own-prefix panel must not be described as a direct test
of teacher-prefix robustness and must not fabricate such a subset after the
fact.

NMS or duplicate guards may be reported as secondary health checks but cannot
replace raw rollout behavior or rescue a failed bridge.

## Frozen Metric and Assignment Schema

Every gated metric event records key, arm/control, example/image/segment/row,
prefix condition, numerator, denominator, eligibility, skip reason, support
size, and code fingerprint. Row metrics are averaged within image before the
image-level mean/bootstrap. Missing or zero-denominator metrics remain missing
with an explicit count; they are never coerced to zero.

For a known teacher-forced target bag `B` and proposal support `V`:

```text
PBMass = sum_{i in B} a_i
AreaMass = |unique(B)| / |V|
AreaAdjustedPBMass = PBMass - AreaMass
HitAt4 = 1 if top4(a) intersects B else 0
OtherMass = mean_j sum_{i in (B_j minus B)} a_i
SpecificityMargin = PBMass - OtherMass
SpecificityRatio = (PBMass + 1e-6) / (OtherMass + 1e-6)
NormalizedEntropy = -sum_i a_i log(a_i + 1e-12) / log(|V|)
Top16Lift = sum_{i in top16(a)} a_i - min(16,|V|)/|V|
```

`OtherMass` is eligible only when at least one other annotated box has a
nonempty overlap-adjusted bag. Shared current/other tokens are removed from
the other bag and reported separately. Metrics are reported by visual support
size quartile; raw entropy/top-k values from unequal grids are not pooled
without the normalized form. Uniform/all-token saturation means
`NormalizedEntropy>0.98` and `Top16Lift<0.02`; the representation gate fails if
this occurs on more than `10%` of eligible rows.

For prefix coverage monitoring, let `C` be the union of prior rendered-box
bags. Report separately:

```text
CurrentMass = sum_{i in B} a_i
CoveredOnlyMass = sum_{i in (C minus B)} a_i
SharedMass = sum_{i in (C intersect B)} a_i
CurrentCoveredMargin = CurrentMass - CoveredOnlyMass
```

Free-rollout behavioral matching is frozen to
`src/vis/matching.py::match_row` with exact normalized description,
IoU threshold `0.50`, greedy one-to-one candidates ordered by
`(-IoU, gt_index, pred_index)`, and duplicate candidates defined by equal
normalized description plus pair IoU `>=0.30`. Record the source fingerprint.
`MatchedRowPrecision_i = matched_count_i / max(1, raw_pred_count_i)` and
`LabeledRowRecall_i = matched_count_i / GT_count_i`. The names are intentional:
precision cannot stand in for enumeration coverage. Raw duplicate rate is the fraction
of raw predicted rows whose index appears in at least one duplicate candidate;
pair/component counts are also retained.

For proposal diagnostics during rollout, every emitted valid bbox defines a
score-blind `SelfBoxBag`; report its proposal mass without GT matching. For a
canonically matched row, also report `MatchedGTBagMass`. Unmatched rows retain
`SelfBoxMass` and status `unknown/unmatched`; they never receive a selected GT
target after the fact. Synthetic tie/permutation/overlap fixtures must prove
assignment stability before scoring.

## Screening Gates

All confidence intervals use `10,000` deterministic paired image-level
bootstrap resamples, seed `20260711`, and percentile `[2.5,97.5]` bounds. Row
bootstrap is prohibited. Proposal floors are minimum-meaningful-effect design
choices because no proposal baseline exists; they are not historical empirical
estimates and may not be tuned after scores are visible.

### In-sample learnability stop

The in-sample learnability stop uses only the matched native A/B/C training
rank-0 monitor events (`rank=0, world_size=8`) from the frozen final schedule quartile, planned steps
`385..512` inclusive.  Earlier events remain trajectory diagnostics and cannot
enter this decision.  At least one of B or C must improve area-adjusted target mass
over A by `+0.05` with paired image-bootstrap CI lower bound above `+0.02`,
improve hit@4 by `+0.10` with CI lower bound above `+0.05`, keep the declared
all-token saturation rate at or below `10%`, and retain at least `50%` of its
canonical A-relative gain on the combined jitter plus duplicate-history
subset.  Intended proposal losses must be finite and nonzero, and intended
parameter updates must be evidenced by checkpoint state transitions.
Because the live trainer receipts only a global all-rank gradient norm rather
than proposal-parameter-specific norms, the executable gate uses finite,
nonzero proposal loss plus the checkpoint state-transition contract: A keeps
`W_q/W_k/U` unchanged, B changes `W_q/W_k` but not `U`, and C changes
`W_q/W_k/U`; the global all-rank gradient consensus must also be finite.
Synthetic per-parameter gradient norms are not invented.  Failure is `hold`
and stops before held-out control production.  Passing this
stage is `in_sample_learnable`, never representation promotion.

### Held-out representation promotion gate

Only after the in-sample stop passes, the complete five-control panel on the
frozen 320-image/906-view held-out cohort evaluates the stronger specificity,
shortcut-control, shuffle, and same-class/crowded requirements below.  None of
these bullets is claimed by the rank-0 training-monitor stop above.

B or C must:

- improve area-adjusted target positive-bag mass over A/controls by at least
  `+0.05` absolute, with paired 95% CI lower bound above `+0.02`;
- improve `hit@4` by at least `+0.10`, with CI lower bound above `+0.05`;
- reach target-versus-other-box specificity ratio at least `1.5` and improve
  the matched-control specificity margin by at least `+0.15`;
- retain a positive residual after bag-area, raster-position, row-depth, and
  image-mean controls;
- fail under image-content shuffle;
- avoid the normalized all-token saturation condition defined above and attain
  mean `Top16Lift>=0.10` with CI lower bound above `0.05`;
- demonstrate same-class/crowded specificity beyond chance.

The substantive residual checks are evaluated on canonical, jitter, and
duplicate-history views, not merely their presence.  Each noncanonical
condition must retain at least `50%` of the candidate's canonical
native-minus-strongest-control gain for both area-adjusted positive-bag mass
and hit@4; the combined noncanonical image-bootstrap CI lower bound must be
above zero.  Target-minus-matched-distractor specificity must be positive for
same-area, same-center, same-aspect, and combined-geometry controls, and the
same-class eligible subset remains the primary instance-binding stress test.
Missing eligible evidence is `rerun`, an observed threshold failure is
`hold`, and malformed/stale/provenance-mismatched evidence is
`contract_fail`.

Failure stops causal promotion of the chosen positive-bag handle.

### Causal-use gate

`causal_bridge_supported` requires all of:

- the representation gate passes;
- C-on improves per-image matched-row precision (description match and
  IoU>=0.50) over C-off by at least `+0.05`, with paired CI lower bound above
  `+0.02` and at least `60%` of images nonnegative;
- C-on improves labeled recall@0.50 over both C-off and B by at least `+0.03`,
  with paired CI lower bound above zero;
- correct-proposal benefit attenuates under removal, another-image,
  token-permutation, position-only, and norm-matched-random controls;
- each negative control's CI upper bound for improvement over C-off is at most
  `+0.02`;
- phrase and geometry effects bind to the same object rather than forming a
  chimera.  A chimera is a predicted row whose geometry matches a GT object at
  IoU `>=0.50` while its normalized description disagrees with that matched GT
  description.  C-on must not raise the per-image chimera rate over either
  C-off or A by more than `+0.01`, and the paired-bootstrap CI upper bound for
  each increase must be `<=+0.01`;
- labeled recall/under-enumeration improves without a material increase in
  duplicates, invalid rows, malformed output, or premature STOP;
- the effect is consistent with the already-frozen held-out representation-v2
  noncanonical-prefix evidence; this natural own-prefix panel makes no new
  teacher-prefix robustness claim;
- RP1.00 does not reveal a contradictory failure hidden by RP1.10.

The combined jitter plus duplicate-history retention requirement belongs to
the held-out representation-v2 gate and is not recomputed from natural
own-prefix rollouts. Under RP1.00, the C-on-minus-C-off CI lower bound must
exceed `-0.03`; otherwise a result passing only RP1.10 is
`decode_specific_bridge`.

### Safety/non-inferiority gate

All safety metrics must be recomputed with identical definitions across A,
C-off, and C-on. The primary attribution contrasts are paired C-on-minus-C-off
and C-on-minus-A; the historical step-4887 rollout is context only unless it is
rerun or rematerialized with the identical backend, parser, matcher, decode,
and metric code identity. C-on must satisfy against both matched controls:

- overall raw duplicate rate no more than `+3` percentage points, with paired
  CI upper bound `<=+0.03`;
- the same duplicate delta no more than `+5` points in images with `>=9` GT
  objects;
- invalid geometry/format no more than `+1` point; on val200, more than two new
  invalid events fails the screen;
- natural terminal closure at least `95%`;
- count-normalized annotated under-enumeration
  `max(0, GT_count - matched_GT_count) / GT_count` no more than `+0.03`, with
  paired CI upper bound `<=+0.03`;
- premature annotated stop rate (a naturally terminated rollout with at least
  one annotated GT unmatched by the frozen matcher) no more than `+3` points,
  with paired CI upper bound `<=+0.03`.

These STOP metrics concern annotated residuals only and are never interpreted
as complete visible-object coverage. The executed training manifest reports
the checkpoint-level view exposure as `9831/1843/614` for
canonical/jitter/duplicate-history. The frozen renderer supervises terminal
EOS only in canonical full responses, giving supervised terminal-EOS atom
counts `9831/0/0`; row-state jitter and duplicate-history views ignore terminal
EOS/newline. These are shared training exposures, not labels on natural
held-out images. Consequently the own-prefix STOP metrics are reported by
observable held-out count strata and must not be post-hoc partitioned by a
teacher-forced view label that does not exist. Historical `200/200` terminal
`<|im_end|>` is not proof that STOP was correct.

One seed can only produce a mechanism-screen result. A replicated causal claim
requires at least three matched seeds or remains explicitly provisional.

## Ordered Outcome Interpretation

Evaluate in this order and emit exactly one primary label:

1. `contract_fail`: timing, GT leakage, token-grid, cache, optimizer,
   checkpoint, precision, or artifact contract fails.
2. `underpowered_inconclusive`: the frozen eligible cohort or required strata
   cannot support the predeclared analysis.
3. `shortcut_or_row_prior`: proposal/behavior is explained by position, depth,
   raster, area, or image-mean controls.
4. `bridge_harmful`: C increases duplicates, STOP, invalidity, geometry drift,
   or otherwise breaches safety guardrails.
5. `label_limited`: unmatched/partial-label ambiguity dominates the direction.
6. `positive_bag_insufficient`: proposal mass improves but instance
   specificity or full-row binding does not.
7. `representation_only`: B/C proposal representation improves without
   metric-bearing own-prefix behavior.
8. `auxiliary_training_effect`: C-off exceeds B but C-on does not exceed C-off.
9. `decode_specific_bridge`: C passes only under RP1.10 and not RP1.00.
10. `causal_bridge_supported`: every causal-use gate passes.
11. `mixed_or_bridge_unlocalized`: eligible evidence is mixed and matches none
    of the earlier terminal labels.

Negative evidence applies to the tested proposal target, bridge lifetime, and
delivery surface only. It does not prove that all intermediate visual-instance
representations are infeasible.

## Completion Promise

This unit is complete only when one of the following holds:

- the implementation/runtime contract fails before training and the exact
  blocker is recorded as `contract_fail`; or
- A, B, and C complete the frozen training budget, all required own-prefix
  rollout and representation conditions complete under exact receipts, and one
  ordered terminal label is emitted; or
- the frozen cohort is demonstrably underpowered before model outcomes are
  interpreted and the unit closes `underpowered_inconclusive`.

Acceptable evidence requires:

- authored, resolved, runtime, and artifact contract alignment;
- exact checkpoint/base/adapter/special-token identity;
- identical A/B/C data, order, updates, CE/type objective, and selection rule;
- a future-token-leak proof at every proposal boundary;
- one attested visual-token tensor and footprint mapping;
- raw and weighted proposal-loss terms, valid counts, mask density, dtype,
  finite, gradient, distributed-reduction, and optimizer-group receipts;
- checkpoint save/load parity for the proposal/bridge state;
- bridge-on/off, null, shuffle, permutation, and random-norm controls;
- exact clean/jitter/duplicate prefix-condition manifests and per-condition
  denominators;
- row-state receipts proving `prefix_supervised_atoms=0`,
  `eos_supervised_count=0`, unchanged current-target digest, explicit
  occurrence identity for duplicate history, static-cache fingerprint binding,
  and condition-level sample/token/segment mass;
- exact RP1.10/RP1.00 decode receipts and parser/drop counters;
- all frozen denominators and skipped/failure reasons;
- raw artifacts sufficient to regenerate every table.

Insufficient evidence includes:

- auxiliary loss or proposal accuracy alone;
- attention visualization alone;
- teacher-forced/canonical-prefix improvement without own-prefix rollout;
- C versus A without B and C-off;
- GT/oracle proposal inference;
- first-token improvement without phrase and complete geometry binding;
- AP/mAP movement without parser, duplicate, count, STOP, and label-ambiguity
  accounting;
- one favorable RP setting selected after inspection;
- comparisons against legacy random/sorted artifacts with mismatched runtime or
  checkpoint lineage.

## Pre-Implementation Runtime and Innovation Gate

The current Swift stack does not yet provide a trusted train-time owner for this
mechanism:

- `src/qwen/forward.py` rejects arbitrary `inputs_embeds` and protected forward
  overrides and currently returns logits rather than an owned hidden/visual
  auxiliary surface;
- `src/losses/runner.py` owns only the existing CE, token-type, and coordinate
  terms;
- optimizer parameter-group planning rejects unmatched trainables;
- checkpoint writing currently owns adapter and special-token payloads, not a
  proposal/bridge head;
- existing `scheduled_residual_*` and `stored_post_scatter_delta.py` utilities
  establish research inference hooks, especially the
  `Qwen3VLModel.post_scatter_inputs_embeds` boundary, but do not establish a
  training integration.

Consequently, no config-only implementation or launch is authorized. Before
coding or GPU work, a separate implementation plan and innovation audit must
identify and attest:

1. the executable owner and exact tensor for spatially indexed visual states;
2. visual-token order, merge footprint, deepstack handling, image grid, and
   resized-coordinate alignment, including example/image/packed-segment-local
   proposal normalization and pooling;
3. a no-future-token proposal boundary;
4. exact functional parity of the frozen layer-0-to-layer-1 split bridge,
   including clone-before-deepstack semantics and the declared causal row
   interval;
5. proposal/bridge trainable registration, optimizer grouping, distributed
   reduction, and checkpoint/resume ownership;
6. A/B/C compute and parameter-budget comparability;
7. per-row bridge application, cache reset, lifetime, and inference parity;
8. resolved config, manifest, metric, and artifact schemas;
9. tiny deterministic loss/gradient tests, a two-image packed isolation test,
   exact C-on/C-off path-parity test, and one real two-step smoke;
10. a bounded `promote`, `hold`, or `rerun gate` launch verdict.

Historical/legacy proposal verifiers and visual-coverage plans are idea donors
only. They are not current runtime authority and must not be copied wholesale.

The installed-source/tiny-runtime audit selected the split-layer seam but did
not authorize launch. Before implementation integration, one real local
Qwen3-VL-2B inference probe must attest layer-0 clone safety, layer-1
localization, post-merge image-major ordering, SDPA and FA2 packed isolation,
and cached row-boundary recomputation/reset. Gradient-checkpoint replay cannot
be attested before the functional bridge exists; it is a hard VF4 gate after
the functional scaffold and before any full integration, VF6 smoke, or
training launch.

### Visual-feature customization safety ladder

Visual-feature customization is treated as the highest-risk implementation
surface in this unit. A hook that appears to change an image tensor can be
wrong while producing finite loss and plausible generations. Before any
proposal training is authorized, a future implementation must pass these gates
in order and stop on the first failure.

#### VF0: executable owner and tensor identity

- identify the installed Qwen3-VL module and exact forward invocation that owns
  the chosen tensor;
- distinguish vision-tower outputs, merger/aligner outputs, deepstack features,
  post-scatter language input embeddings, and decoder residual states;
- record shape, dtype, device, autograd status, module path, call count, and
  physical image-token indices from a real batch;
- prove whether the tensor is before or after spatial merge and whether its
  order is raster, temporal, image-major, or another installed-runtime order;
- fail if source reading, mocks, or historical receipts are the only evidence.

#### VF1: spatial and packed-segment ownership

- bind every visual state to example ID, image ID, packed segment, image-grid
  metadata, resize plan, merge factor, and physical token interval;
- prove that text tokens, another image, padding, and another packed example
  cannot enter the proposal denominator or receive a bridge delta;
- include one two-image packed batch and one unequal-grid batch;
- verify any deepstack/multi-level visual tensors separately rather than
  assuming one token map indexes all levels identically.

#### VF2: exact no-op parity

- install the full intended capture/intervention orchestration with a zero
  delta and disabled proposal;
- require identical logits, selected loss terms, gradients on existing
  trainables, generated tokens, parser output, and cache/re-prefill receipts
  against the unmodified path within a predeclared numerical tolerance;
- run the same parity for C-off, including identical call count and cache/reset
  path; bypassing the hook is not parity;
- fail on silent attention-backend, dtype, position, or cache changes.

#### VF3: localized synthetic intervention

- apply a deterministic synthetic delta to a frozen, explicitly named subset
  of visual positions;
- prove that only those physical visual positions change before downstream
  mixing, with unchanged text positions and unchanged other packed segments;
- verify delta norm, application count, reset, and deterministic replay;
- compare capture before/after the owner boundary so an in-place view, clone,
  detach, or overwritten tensor cannot masquerade as a successful write.

#### VF4: gradient and distributed ownership

- immediately after the functional split-loop scaffold exists, run an exact
  local 2B + step-4887 DoRA training-mode forward/backward with non-reentrant
  checkpointing before integrating the full data/loss/checkpoint surface;
- require expected decoder replay but exactly one logical bridge application
  and metric event per row, clone/version-counter safety, and no stateful train
  hook;
- require finite intended `W_q`, `W_k`, `U`, and authorized DoRA gradients.
  The exact zero/C-off path must preserve native logits, loss, and authorized
  gradients bitwise. An executed control showed that
  `torch.use_deterministic_algorithms(True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` alone do not make FA2 backward
  deterministic: installed `flash-attn==2.8.3` defaults its separate
  `deterministic` argument off. Exact VF4 therefore additionally requires
  `FLASH_ATTENTION_DETERMINISTIC=1` before Python/CUDA initialization and a
  bitwise native-vs-native backward repeat before C-off is interpreted. The
  earlier nondeterministic HOLD receipt remains part of the evidence trail;
- prove gradient flow from a deterministic tiny proposal/bridge loss to only
  the intended new projection/gate and authorized DoRA parameters;
- test zero-positive, empty-mask, nonfinite, and all-token-saturation cases;
- verify autocast/fp32 islands, gradient checkpointing/recomputation, packed
  accumulation, DDP reduction, clipping, skipped steps, and optimizer grouping;
- prohibit hidden `.detach()`, in-place autograd corruption, unmatched
  parameters, double hook application under recomputation, and rank-local loss
  denominators.

#### VF5: checkpoint and train/infer parity

- save and reload proposal/bridge parameters with byte/hash receipts and prove
  identical outputs after reload;
- prove that training and own-prefix inference use the same visual tensor
  identity, proposal normalization, bridge projection, norm cap, row lifetime,
  and reset semantics;
- verify that inference cache behavior cannot reuse a proposal from the prior
  row or omit the current row's bridge;
- fail if a training-only hook or inference-only post-scatter path silently
  defines two different mechanisms.

#### VF6: bounded real smoke

- only after VF0-VF5 pass, run one real two-step A/B/C smoke;
- require finite raw and weighted losses, nonzero intended gradients, exact
  condition receipts, checkpoint reload, and one C-on/C-off generation parity
  panel;
- treat this smoke as wiring evidence only. It cannot support proposal
  learnability or model-behavior claims.

No full pilot, GPU scaling, or metric interpretation is authorized until the
innovation audit explicitly returns a launch verdict after VF6. A failure at
any VF gate closes or amends the unit at `contract_fail`; it is not repaired by
loosening tolerances after observing model behavior.

## Planned Procedure

1. Validate this frozen amendment: numeric effect floors, cohort/seed policy,
   IoM positive-bag rule, split-layer bridge norm/lifetime, gradient policy,
   jitter ranges/IoU band, and exact A/B/C prefix-condition table.
2. Audit installed Qwen3-VL ownership and select the smallest train-time seam
   satisfying the causal order and cache contract.
3. Implement only the bounded research surface with fail-closed schema,
   optimizer, checkpoint, metric, and artifact ownership.
4. Execute VF0-VF5 in order, including deterministic
   tokenizer/span/footprint/loss/gradient/checkpoint and packed-isolation tests.
5. Run VF6, the exact two-step smoke for A/B/C. It proves wiring only and
   cannot support a model claim.
6. Train the exact matched A/B/C screening conditions on the frozen cohort.
7. Evaluate the in-sample native training-monitor diagnostic and stop if
   proposal learnability fails.  If it passes, run the frozen held-out
   val200-plus-120 teacher-forced control panel; stop unless that representation
   gate promotes B or C.
8. Run metric-bearing own-prefix A/B/C-on/C-off and causal controls under
   RP1.10 and RP1.00.
9. Apply the ordered outcome map exactly once, record the bounded architecture
   posterior, and stop. Do not automatically launch random+bridge, slots,
   coverage, or STOP work.

## Observations

- Executed observation: the bounded train-time A/B/C surface, checkpoint
  handoff, monitoring, and C-on/C-off own-prefix controller are implemented.
  Arm D remains postponed and covered-region competition remains
  monitoring-only in this screen.
- Executed observation: the authoritative eight-GPU VF6 roots are A4, B2, and
  C1 as recorded in `vf6-results-2026-07-11.md`.  All arms completed the exact
  two-step schedule on the same cache/view identity; A kept `W_q/W_k/U`
  unchanged, B changed only `W_q/W_k`, and C changed `W_q/W_k/U`.  Checkpoint
  reload hashes were rank-equal.  The historical A3/B1 artifacts are excluded
  because cached arm metadata was contaminated before the live-arm rebind fix.
- Executed observation: C-on/C-off forced replay computed the same three legal
  proposals and row resets; native and C-off were exact on all 22 compared
  logits, while C-on changed all 22 layer-1 pulse applications.  This proves
  wiring/mechanics only, not proposal learnability or behavioral benefit.
- Installed-runtime observation: literal cached per-row post-scatter mutation
  is not owned. The implemented candidate is the attested functional
  layer-0-to-1 split bridge with strict segment ownership and reset lifetime.
- Installed-runtime observation: Transformers 4.57.1 Qwen3-VL accepts
  `output_hidden_states` and `output_attentions` through public-forward
  `**kwargs`, but the first real CUDA run exposed 29 hidden-state entries and
  no attention tuple under both SDPA and FA2. Phase 0 preserves and receipts
  that partial exposure on native/C-off paths and obtains seam evidence from
  explicit layer-boundary captures. It must not fabricate an auxiliary return
  surface merely to satisfy the probe.
- Repository observation: Swift now has a bounded trainable proposal/feedback
  owner for this research unit.  The remaining pre-512 blocker is the
  claim-valid representation producer/control/evaluator path, not the forward
  or optimizer/checkpoint wiring.
- Executed observation: the producer indexed all 12,288 frozen training views
  and 93,108 supervised rows, then passed a two-microstep real C1 smoke that
  computed all five frozen controls from one Qwen forward per microstep.
  Native repeat was byte-identical, content shuffle was nonidentity,
  image-mean uniform error was `1.45e-11`, row-depth read no forbidden field,
  and hook/gradient/CUDA cleanup receipts passed.  This is producer mechanics,
  not representation evidence.
- Executed observation: the arm-neutral held-out descriptor was indexed
  end-to-end over 906 views, 320 source images, 114 microsteps, and 3,639
  proposal rows.  A real held-out GPU microstep produced all five controls
  from one Qwen forward and passed the strict evaluator.  Its gate decision
  was the expected `rerun` because the bounded smoke contained only C and one
  microstep; it did not become held-out representation evidence.
- Executed-runtime observation: the corrected deterministic 2B probe passed
  both SDPA and the real FA2 padding-free varlen branch. It attested exact
  native/C-off logit and loss parity, exact future-token invariance of the
  boundary proposal/pool/delta, packed-first versus standalone main/deepstack
  feature identity, three deepstack levels extracted from vision indices
  `[5,11,17]` and consumed after decoder layers `0,1,2`, exact FA2 segment-B
  isolation, and a next-row pulse reset at the causal `<|box_end|>` input.
- Counterexample or negative result: the first apparent runtime PASS retained
  mutable Transformers Cache objects and therefore reported the final cache
  length for every earlier step. That artifact is diagnostic-only. The
  corrected probe freezes cache length at each call and passed the exact
  sequence `1033,1034,...,1045` on both native and zero paths.
- Artifact handle:
  `/data/CoordExp/outputs/probes/coordexp_swift/proposal_bridge_runtime_attestation/execute_gpu0_cache_snapshots_20260711T125458Z/proposal_bridge_runtime_attestation.result.json`.

## Interpretation

- Proposal representation learning is real on the frozen teacher-forced
  representation panel, but the tested layer-0-to-1 pulse did not turn that
  representation into target-specific own-prefix behavior.
- Under RP1.10, C-on raises labeled recall over C-off by only `+0.0183`
  (`95% CI [+0.0082,+0.0288]`), below the predeclared `+0.03` floor, while
  precision falls `-0.2247`, raw duplicate rate rises `+0.1806`, invalid-row
  rate rises `+0.3063`, and natural closure falls from `0.9906` to `0.6594`.
- Another-image and within-image token-permutation feedback reproduce the
  active effect within uncertainty; position-only feedback reproduces part of
  it, while norm-matched random feedback is approximately null.  The pulse is
  therefore best read as a learned continuation/row prior, not a specific
  uncovered-object bridge.
- RP1.00 slightly increases annotated recall but worsens duplication.  This
  confirms RP1.10 as a consequential anti-duplication heuristic rather than a
  neutral decode setting; it does not rescue bridge specificity.
- COCO incompleteness limits the interpretation of unmatched extras, but it
  cannot explain the another-image/token-permutation equivalence or the
  controller-invalid, truncation, and duplicate regressions.
- Negative evidence is bounded to the frozen positive-bag target, one-row
  pulse lifetime, and split-layer delivery seam.  It does not reject all
  intermediate visual-instance representations.

## Research Unit Closeout

Observed: all matched training, runtime, representation, and own-prefix phases
completed.  The final behavior evaluator loaded all sixteen frozen cells
(`8` conditions x `2` repetition penalties), exactly `320` ordered request IDs
per cell, and returned a passing artifact receipt.  The behavior gate returned
`hold` with `reason=safety_noninferiority_failed`, `primary_pass=false`, and
`negative_controls_pass=false`.  See
`own-prefix-causal-behavior-results-2026-07-12.md` and the exact artifacts at
`/data/CoordExp/outputs/probes/coordexp_swift/pvci_own_prefix_sharded_panel_ABC_320_v3_20260712/`.

Closeout replay used the hardened evaluator contract: safety is checked
against both A and C-off, includes the `9+`-GT duplicate stratum, malformed
output, and the natural-closure floor, and the PASS receipt binds the emitted
image metrics, aggregate, and gate artifacts.  The stricter replay also
returned `safety_pass=false` and the same terminal decision.

Ordered terminal label: `shortcut_or_row_prior`.  This label precedes
`bridge_harmful` in the frozen outcome map because another-image and
token-permutation controls preserve the behavior.  The safety failure remains
an independently supported finding, not the primary ordered label.

Supported: the proposal readout is learnable and generalizes under the frozen
teacher-forced representation controls.  The tested learned pulse changes
rollout length and continuation behavior.

Not supported: target-specific uncovered-object redistribution, causal bridge
promotion, safe recall improvement, coverage/STOP improvement, order
robustness, a final architecture, or a production change.  Arm D and slots
remain outside this unit and were not launched.

Bounded architecture posterior: do not scale or promote this split-layer pulse.
The next unit, if authorized, should require a short controlled transition in
which a source-swapped or token-permuted candidate loses the effect before any
long rollout.  It should test target specificity and full-row phrase/geometry
binding at the causal seam, then separately ask whether a successful write can
alter the next-row native proposal.  Do not begin with a larger module.

Next decider: run that short controlled target-specific transition probe before
any new long-rollout bridge training.  If the correct proposal cannot beat
source-swap, token-permutation, position-only, and norm controls inside the
safety envelope, retire this delivery route and move to a separately bounded
COMMIT/COVERAGE/STOP unit.

Promotion decision: `not_promoted`.
