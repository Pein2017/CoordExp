---
title: Trajectory-Quality-Weighted Object-Row Self-Imitation Screen
description: A two-arm training screen that tests whether emphasizing object rows from better sampled routes improves native greedy enumeration beyond generic exposure to sampled rows.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: not_authorized
unit_id: 2026-07-21-trajectory-quality-weighted-row-self-imitation-screen
topic: qwen3-vl-dense-enumeration
status: needs_revision
evidence_status: none
updated: 2026-07-21
---

# Trajectory-Quality-Weighted Object-Row Self-Imitation Screen

## Question

> **Current status:** This draft predates the completed
> [greedy-prefix forced object-path intervention](../2026-07-21-greedy-prefix-forced-owner-path-intervention/results.md).
> Do not implement it as written. The newer evidence prioritizes exact
> greedy-prefix sampled-row counterfactuals over sampled-prefix trajectory
> imitation. Route-quality weighting and within-image shuffling may remain as
> controls in a revised unit.

Can positive-only training on complete object rows already produced by the
source model move native greedy decoding toward higher unique-object coverage,
when the training weight reflects the quality of the sampled trajectory that
produced each row?

The treatment is useful only if it improves clean greedy rollout more than an
equal-budget control that sees the same row tokens but receives deliberately
misassigned trajectory-quality weights.

## Why This Unit Exists

The preceding
[individual-trajectory versus sampled-union audit](../2026-07-21-individual-trajectory-versus-union-support-audit/results.md)
established two bounded facts on twelve human-refined development images:

1. some complete sampled routes cover strictly more verified physical owners
   than greedy at the same row budget without adding more harmful rows; and
2. useful owner support is also distributed across multiple sampled routes.

This makes sampled rows plausible training material, but does not show that
imitating them improves the policy. The twelve audited images remain
development and evaluation evidence only and must never receive training
gradients.

## Competing Explanations

### Working explanation: trajectory quality is useful credit

Rows generated inside a better route carry evidence about a locally useful
sequence of decisions. Giving those rows more training weight should make the
source model more likely to reproduce high-coverage transitions under greedy
decoding.

### Strong alternative: generic sampled-row exposure causes the effect

Any improvement may come from another pass over model-generated rows, longer
outputs, or a general change in object-row syntax. If so, preserving all row
tokens and masks while shuffling trajectory-quality weights within each image
should improve equally.

The within-image shuffle is the primary causal control. It retains the same
images, generated trajectories, complete rows, token positions, total weight,
optimizer steps, and training surface while breaking only the association
between route quality and gradient weight.

## Training Data

Build a new cohort from a separate training split:

- 256 physical training images for the first meaningful screen;
- no overlap with the twelve human-refined development images;
- one greedy trajectory and sixteen sampled trajectories per image under the
  same source checkpoint and decoding settings used by the preceding audit;
- exact generated token identifiers and exact model-generated prefixes; no
  decode-then-retokenize reconstruction;
- complete object rows only;
- a row is eligible for positive imitation only when it is the first row in
  that trajectory matched to a labeled physical owner of the same category;
- duplicate rows, malformed rows, unmatched rows, uncertain owner assignments,
  and rows with untrusted geometry receive zero gradient rather than negative
  supervision.

Because Common Objects in Context annotations are incomplete, unmatched rows
must not be treated as hallucinations or harmful negatives. Training-route
quality is computed only from positively matched unique owners and confirmed
duplicates. A small crop-enlarged audit of admitted and rejected examples is
required before model training.

## Compared Arms

### Source checkpoint

Evaluation only. The source is the geometry-sorted, description-first,
pure-cross-entropy checkpoint at step 4,887.

### Correctly weighted self-imitation

For image `i`, trajectory `r`, and eligible complete object row `j`, let
`Y_i,r,j` be the exact row-token sequence and `P_i,r,j` its actual generated
prefix. Let `q_i,r` be a bounded route-quality score derived only from matched
unique owners and confirmed duplicates at a fixed complete-row budget. Convert
the route score into a non-negative within-image weight `w_i,r` whose mean is
one.

The treatment loss is:

```text
L_treatment =
  sum over eligible rows of
  w_i,r * mean negative log probability of Y_i,r,j given image i and P_i,r,j
```

Only tokens inside the selected complete object row receive this loss. The
prefix, prior rows, image tokens, and unselected generated rows provide context
but receive no imitation gradient.

### Within-image shuffled-quality control

The control uses the identical eligible rows, exact prefixes, token masks, and
per-image weight multiset. It applies a deterministic non-identity permutation
to `w_i,r` among trajectories from the same image. The permutation and random
seed are recorded before training.

Therefore treatment and control differ only in whether high weight is attached
to the trajectory that actually earned that quality score.

## Model and Optimization Scope

- Freeze the vision tower and multimodal aligner.
- Train only the language-tower Weight-Decomposed Low-Rank Adaptation (`DoRA`)
  payload and any already-required special-token embedding rows.
- Use a token-type gate only as a stability constraint on selected row sites;
  it is not an independent scientific arm.
- Use low learning rate, gradient clipping, one epoch, and matched effective
  examples and selected tokens per optimizer update in both arms.
- Do not add an object slot, detector, coverage ledger, external teacher,
  Kullback-Leibler divergence penalty, or inference-time controller.
- Reuse the current exact-prefix `StateBank` event store, meaning the
  repository's frozen bank of token-exact rollout prefixes and candidate rows,
  together with its replay, packing, compact-logit, and streaming training
  path. Add only the missing positive-only weighted row-loss behavior and
  experiment-local data assembler.

## Primary Observation

The primary comparison is treatment minus shuffled control on clean greedy
rollout, not training loss and not fixed-prefix likelihood alone.

Read evidence in this order:

1. Exact-prefix replay confirms that the assigned row weights changed the
   intended complete-row likelihood in the expected direction.
2. Clean greedy rollout on the 256 training images tests whether the learned
   preference survives self-generated prefixes.
3. The twelve human-refined development images test behavioral transfer and
   safety without entering training or model selection.

Report unique matched physical owners, confirmed duplicates, malformed and
dropped rows, native termination, row count, entity discovery, category
retention, full-box intersection over union, center error, and box-size error.
Entity discovery and geometry quality remain separate measurements.

## Interpretation

- **Treatment beats shuffled control safely:** route-quality credit is useful;
  prepare a 1,024-image replication before any full-data training.
- **Both arms improve similarly:** sampled-row exposure or extra optimization,
  not route-quality assignment, explains the gain.
- **Exact-prefix likelihood improves but clean rollout does not:** the signal is
  learnable locally but does not transfer through the model's own trajectory.
- **Treatment increases rows but not unique owners:** the objective mainly
  strengthens generic continuation.
- **Treatment improves entity discovery while geometry worsens:** preserve the
  route treatment as a selection result, but do not claim complete detection
  improvement; geometry needs a separate coherent-box treatment.
- **Neither arm improves:** do not scale self-imitation; return to local
  remaining-object completion or an explicit compact task-state intervention.

## Stop Rule

Do not proceed beyond the 256-image screen unless correctly weighted
self-imitation exceeds the shuffled control in clean greedy unique-owner
coverage without a material rise in confirmed duplicates, malformed output,
unsupported entities, or geometry contamination. A training-loss decrease is
not a promotion signal.

## Minimal Execution Path

1. Reuse the current rollout runner to collect the independent training cohort.
2. Reuse exact integer-token row splitting from
   `scripts/research/build_inference_coordinate_boundary_state_bank.py`.
3. Build paired treatment and control replay banks with identical selected row
   sites and weight totals.
4. Run a one-process, one-step real-model smoke for both arms.
5. Audit a compact sample of admitted rows, route scores, and shuffled pairs.
6. If the smoke and audit pass, run the matched 256-image one-epoch arms.
7. Evaluate source, treatment, and control with the same greedy protocol.

## Non-Goals

- selecting a final architecture;
- proving that every sampled route is better than greedy;
- recovering the full union of all sampled objects with one trajectory;
- estimating population-level effect size from the twelve development images;
- treating unmatched predictions as hallucinations;
- full-dataset training before the 256-image causal control closes.

## Artifact Handle

Planned logical root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-trajectory-quality-weighted-row-self-imitation-screen/<run-id>/
```

No executed artifact or implementation authorization exists yet.
