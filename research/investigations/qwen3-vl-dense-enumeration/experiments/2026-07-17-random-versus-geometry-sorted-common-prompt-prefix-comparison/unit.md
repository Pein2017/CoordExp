---
title: Random versus Geometry-Sorted Supervision under Common Prompts and Prefixes
description: Planned comparison of existing checkpoint-3668 adapters under identical prompt and covered-object history conditions before any new training.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: not_authorized
unit_id: 2026-07-17-random-versus-geometry-sorted-common-prompt-prefix-comparison
topic: qwen3-vl-dense-enumeration
status: planned
evidence_status: none
updated: 2026-07-17
---

# Random versus Geometry-Sorted Supervision under Common Prompts and Prefixes

## Status and authority

This is a discussion draft. It records the next comparison and its current
boundaries. It does not authorize implementation, graphics-processing-unit
execution, new training, or a conclusion about which supervision is better.

## Question

When the image, prompt wording, already emitted objects, decoding settings, and
numerical execution are held constant, do the existing random-order and
geometry-sorted checkpoint-3668 adapters make meaningfully different next-row
decisions?

## Why this comparison comes next

Most current mechanism evidence comes from a geometry-sorted adapter. A stable
difference under identical prompts and prefixes would support the claim that
training order changes model behavior. A difference that disappears under a
common prompt would instead indicate prompt specialization. A difference only
in each adapter's own training loss could reflect the greater uncertainty of a
target order that changes between epochs.

This unit therefore uses existing checkpoints before requesting another
training run.

## Existing checkpoints

Random-order checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668
```

Geometry-sorted checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_sorted_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-sorted-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062429/checkpoint-3668
```

The pair is suitable for a historical comparison because its resolved run
settings differ mainly in object ordering and objective name. It is not a
matched replacement for the current step-4887 Weight-Decomposed Low-Rank
Adaptation (`DoRA`) route: these are older Low-Rank Adaptation (`LoRA`) rank-8
runs with one training seed.

## Strongest competing explanations

1. **Training-order effect:** the adapters learned different ways to choose and
   continue objects.
2. **Prompt-wording effect:** each adapter mainly responds best to the ordering
   instruction used with it during inference.
3. **Changing-target effect:** the historical random arm received a different
   target order in each epoch, creating inconsistent next-object targets rather
   than useful order independence.
4. **Single-run or numerical effect:** the difference is not stable across
   seeds, physical batch shapes, or floating-point precision.

## First-stage comparison

Start with eight to sixteen deliberately selected images. Include sparse and
dense scenes, repeated categories, long outputs, and at least one image already
used in the mechanism investigation. The exact images remain open for user
discussion.

For each image, compare both checkpoints under:

1. one identical geometry-sorted instruction;
2. one identical random-order instruction;
3. one instruction that does not request an ordering policy, if the current
   prompt format allows it without changing the output schema.

For controlled prefix checks, give both checkpoints the same already emitted
object set in several orders:

1. geometry-sorted order;
2. reverse order;
3. two fixed random orders.

The object set, row contents, prefix length, image, and next valid objects must
remain identical. Only the order of already emitted rows changes.

## Primary observations

Review the raw next row and short continuation before relying on aggregate
metrics. Record:

- which object is produced next;
- whether the model stops;
- whether it repeats an already emitted object;
- whether the row is valid;
- whether the description and box refer to the same visible object;
- how the four box coordinates differ;
- how much probability remains on valid, not-yet-emitted objects when that can
  be measured without changing decoding.

Use batch size one for the main comparison. Use full-model 32-bit floating
point only for a small result that would change the conclusion and is close
enough to be numerically uncertain.

## Invariants

- identical image bytes and image processing;
- identical output schema and tokenizer;
- identical prompt text within each comparison;
- identical prefix tokens within each comparison;
- identical decoding settings and seeds;
- identical parser and duplicate handling;
- raw output retained separately from any duplicate filtering;
- no new training in this unit.

## First smoke

Run one image, both checkpoints, one common instruction, and one shared prefix.
The smoke passes only if the effective checkpoint, prompt tokens, image,
decoding settings, and output attribution are recorded correctly.

## Stop and promotion rules

- If the difference disappears under a common prompt, stop and classify the
  historical result as prompt specialization rather than a stable training
  signature.
- If only each adapter's own loss differs while next-row behavior under common
  conditions does not, stop and classify the result as a target-difficulty
  difference.
- If the same checkpoint difference appears across prompt wording and shared
  prefixes, expand to a larger held-out comparison before proposing new
  training.
- Do not make a population or current-DoRA claim from this historical one-seed
  pair.
- Do not inspect hidden states unless a stable behavioral difference first
  identifies a concrete decision to explain.

## Non-goals

- no full validation benchmark;
- no new objective or architecture;
- no large hidden-state survey;
- no claim that random ordering is better or worse;
- no attempt to explain official false positives without manual review.

## Reused surfaces and expected cost

Reuse current batch inference, checkpoint loading, prefix construction, raw
output parsing, and detection visualization. Add no general framework before a
real smoke identifies a missing seam. The pilot should require inference on at
most sixteen images and short prefix-conditioned continuations.

Logical artifact root:

```text
outputs/research/qwen3-vl-dense-enumeration/2026-07-17-random-versus-geometry-sorted-common-prompt-prefix-comparison/<run-id>/
```

`<run-id>` means one immutable execution identifier and will be assigned only
after implementation and execution are authorized.

## Decisions still open for discussion

1. Which eight to sixteen images best separate sparse, dense, repeated-category,
   and long-output behavior?
2. What exact common instruction is least likely to favor either adapter?
3. Should the first pass inspect one next row or a short continuation of two to
   three rows?
4. Which already emitted object sets are physically unambiguous enough for the
   shared-prefix comparison?
5. What observation is strong enough to justify expanding beyond the pilot?
