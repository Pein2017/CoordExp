---
title: Exact Greedy Prefix Uncovered-Object Row Training Screen
description: A 256-image treatment screen that asks whether a sampled real object row can be made to defeat premature stopping at the exact greedy prefix and survive ordinary greedy rollout.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
training_promotion_status: not_promoted_too_sparse_and_too_late
implementation_status: complete_for_unit
unit_id: 2026-07-21-exact-greedy-prefix-uncovered-object-row-training-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_bounded_negative
updated: 2026-07-21
---

# Exact Greedy Prefix Uncovered-Object Row Training Screen

## Question

Can a small language-tower-only treatment move a complete, sampled row for a
real uncovered object above premature stopping at the **same exact greedy
prefix**, then preserve that object in ordinary clean greedy rollout?

This is a treatment-efficacy screen, not a final architecture test. It asks
whether Qwen3 Vision-Language model (`Qwen3-VL`) already contains enough
visual and prefix state for a local, better-targeted training signal to improve
set coverage without an object slot, detector, explicit ledger, or custom
inference controller.

## Evidence That Motivates the Treatment

The preceding
[forced object-path intervention](../2026-07-21-greedy-prefix-forced-owner-path-intervention/results.md)
showed that three sampled object paths can be transported into their exact
greedy states and completed by the native decoder. The required supplied depth
varies from a description token through later coordinate boundaries. A local
entity rescue can also harm the later fixed-budget owner set.

Therefore the treatment must satisfy all of the following:

1. use a model-visited exact greedy prefix rather than a sampled-history
   prefix;
2. compare the actual harmful branch with a real uncovered object branch;
3. supervise a coherent complete row rather than one universal token or one
   coordinate; and
4. admit an event only after a fixed-budget replay shows that the intervention
   is not merely reordering or trading away more useful owners.

## Frozen Source and Cohorts

- Source: geometry-sorted, description-first, pure-cross-entropy plus
  token-type-gate Weight-Decomposed Low-Rank Adaptation (`DoRA`) checkpoint at
  step 4,887.
- Training cohort: the existing 256-image training file and source greedy
  rollout artifacts from the coordinate-boundary screen.
- Independent samples: 16 low-temperature samples at each selected exact
  prefix; repetition penalty `1.0`; no beam search.
- Human-refined evaluation: the twelve dense validation images published in
  `val.norm.jsonl` and `val.coord.jsonl`; they receive no gradient.
- Common Objects in Context 80-category (`COCO-80`) labels define the current
  closed reporting vocabulary. Unmatched predictions are unknown, not
  automatic negatives.

## Initial Event Family

The first screen uses only **premature terminal events** because their negative
branch is unambiguous and a successful rescue does not displace a native next
row.

For each image:

1. Reconstruct the source greedy trajectory from exact executed token
   identifiers and verify the newly materialized prompt-token hash against the
   original source-run prompt trace for the same image.
2. At the exact prefix immediately before its native terminal token, identify
   labeled owners missed by the greedy rollout.
3. For each candidate target, conservatively prove target-specific non-coverage
   by checking every prior prediction. Reject the target if any earlier row can
   plausibly refer to it by same-class overlap, center inclusion,
   intersection-over-smaller-area, or ambiguous crowded-scene assignment.
4. From that exact prefix, independently sample at most sixteen complete
   object rows.
5. Admit a sampled row only when it is parseable, has a trusted category and
   unambiguous match to one uncovered owner, and is not a duplicate alias of
   another admitted row. Geometry is reviewed independently: trusted
   coordinate sites may train, while uncertain coordinate sites receive zero
   direct coordinate gradient.
6. Force the complete candidate row once, release native greedy decoding, and
   compare it with the native trajectory under equal row and token budgets.
7. Keep at most one event per image: a candidate that guarantees the target
   owner's addition and introduces no confirmed fixed-budget harm. Under
   target-scoped non-coverage, unknown suffix owners remain neutral rather than
   being counted as positive set value.

Unknown, ambiguous, malformed, unmatched, semantically wrong, mixed-owner, or
poor-geometry candidates receive no gradient. Officially labeled objects can
be positives; official annotation omissions are never used as negatives in
this initial automatic bank.

Full prior-row owner resolution is retained as a high-confidence stratum but
is not required for this premature-terminal event family. All unrelated prior
rows remain unknown. Target-specific non-coverage cannot support a claim about
the complete covered set, duplicate redistribution, or the best remaining
owner. Duplicate-negative events continue to require fully resolved coverage
and are outside this initial screen.

## Fixed-Budget Admission

The primary comparison budget is sixteen complete rows over the whole
trajectory, including rows already present in the exact prefix and the forced
candidate row. Receipts also report budgets eight and thirty-two when the
generated horizon permits them. The generated-token budget is likewise a
whole-trajectory budget: prefix tokens and the candidate row consume capacity
before any released suffix is generated.

A candidate is admitted only if:

- its intended uncovered owner remains in the final candidate-arm owner set;
- unique verified-owner count increases, or remains equal while a confirmed
  native harm decreases;
- no new confirmed duplicate, malformed row, or verified unsupported entity
  appears; and
- the candidate arm receives no extra row or token capacity.

Unknown suffix rows are neutral. Trajectory quality is an admission decision,
not a continuous gradient weight.

## Treatment Objective

Let `P` be the exact greedy prefix, `Y+` an admitted complete row for an
uncovered owner, and `Y-` the actual terminal token. Let `d` be their first
divergent token. The branch term is:

```text
branch_margin =
  log p(positive token at d | image, P, shared row history)
  - log p(harmful token at d | image, P, shared row history)

branch_loss = softplus(margin - branch_margin)
```

The positive row then receives coherent continuation supervision from the
divergence through row closure:

```text
entity_continuation =
  mean negative log probability over trusted schema and description sites

geometry_continuation =
  mean negative log probability over trusted coordinate sites

continuation_loss =
  mean of the non-empty entity and geometry group losses

event_loss =
  branch_loss
  + 1.0 * continuation_loss
  + token_type_gate_weight * selected_site_token_type_gate
```

The two group means prevent description length or four coordinate tokens from
silently changing event weight. The selected-site token-type gate is a
language-format stabilizer, not an independent treatment arm. All selected
logit mathematics uses 32-bit floating point.

Comparing a summed complete-row likelihood with one terminal-token likelihood
is prohibited because their horizons differ. Historical prefix tokens receive
no training loss.

## Training Contract

- One treatment only: Source checkpoint versus branch-plus-continuation
  treatment.
- Freeze the vision tower and multimodal aligner.
- Train only the language-tower DoRA payload and already-required special-token
  embedding rows.
- Learning rate: `1e-5`.
- Gradient clipping: retain the existing bounded training value.
- One full epoch over every admitted event, with no event silently dropped.
- Eight Graphics Processing Units (`GPUs`) through the existing Accelerate
  runtime.
- Choose the largest effective batch that leaves enough optimizer steps for
  distinct approximately 30%, 60%, and 90% checkpoints plus a final
  checkpoint. The admitted event count and step schedule must agree exactly.
- No canonical supervised-fine-tuning replay, Kullback-Leibler divergence,
  Gaussian coordinate smoothing, online collection, external teacher,
  additional detector, or inference-time controller.

## Smoke and Launch Order

1. Build one real exact-prefix terminal-rescue event.
2. Prove exact source-prefix replay parity and verify that one tiny update moves
   both branch margin and positive continuation likelihood in the intended
   direction.
3. Run ordinary clean greedy inference from the smoke checkpoint.
4. Build and validate the full 256-image event bank.
5. Launch the complete eight-GPU one-epoch training run.
6. Save approximately 30%, 60%, and 90% progress checkpoints and the final
   checkpoint. Intermediate checkpoints are diagnostics; the final checkpoint
   owns the promotion decision.
7. Evaluate Source and final treatment with identical ordinary greedy decoding.

## Evaluation

Primary evidence:

- sampled-only training owners recovered by clean greedy rollout;
- unique matched physical owners and false-negative rate at fixed row budgets;
- owners found per emitted row;
- confirmed duplicate, malformed, invalid-geometry, and unsupported-entity
  rates; and
- whether verified-owner gains are accompanied by improved owners per emitted
  row rather than only a larger raw row count.

Supporting evidence:

- physical-entity F1 score;
- mean Average Precision (`mAP`) and mean recall;
- category retention;
- center error, individual `x1`, `y1`, `x2`, and `y2` boundary error, and
  complete-box overlap; and
- qualitative crop-assisted review on the twelve refined dense images.

Entity discovery and exact geometry remain separate judgments. Standard COCO
metrics are supporting measurements because incomplete dense annotations can
misclassify real discoveries.

This single-arm efficacy screen cannot fully separate the branch term,
continuation term, and generic output-length mediation. A positive result may
therefore support only the combined treatment. It must not claim that the
continuation term alone caused the gain, or that output length played no role.
The owners-per-row and safety measurements are diagnostic guards, not a
matched length-control arm.

## Decision Rule

No fixed numerical improvement margin is required for this first screen.
Promote to a 1,024-image replication when the final checkpoint is directionally
promising on unique-owner coverage or false-negative rate and does not show a
clear increase in confirmed duplicates, malformed output, unsupported
entities, or geometry collapse.

Do not promote when only exact-prefix training loss improves, when rollout
merely becomes longer, or when new-owner gains are offset by comparable owner
losses or harmful output. Full-data training requires directional replication
at 1,024 images.

## Artifact Root

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-exact-greedy-prefix-uncovered-object-row-training-screen/<run-id>/
```
