---
title: Random versus Geometry-Sorted Historical Checkpoints under Common Prompts and Prefixes
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

# Random versus Geometry-Sorted Historical Checkpoints under Common Prompts and Prefixes

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
settings differ mainly in the configured object ordering and objective name.
However, the old runtime used that ordering setting for two coupled changes:
it changed both the order of target rows and the ordering instruction in the
system and user prompts. The pair can therefore compare two historical
training regimes, but it cannot by itself attribute a difference to target
order alone.

It is also not a matched replacement for the current step-4887
Weight-Decomposed Low-Rank Adaptation (`DoRA`) route: these are older Low-Rank
Adaptation (`LoRA`) rank-8 runs with one training seed.

## What has already been tried

The historical sorted-versus-random no-newline mechanism smoke already gave
both checkpoints the geometry-sorted readout instruction. It examined 64
teacher-prefix states and found mixed differences: the geometry-sorted
checkpoint was more favorable to continuing instead of ending, while the
random-order checkpoint had more strict `x1` (left boundary coordinate) hits.
The associated free rollouts also contained many malformed rows and did not
support a clean behavioral conclusion.

That smoke covers one corner of the proposed comparison. It did not:

1. test both checkpoints under both ordering instructions;
2. hold the emitted object set fixed while changing only the order of its rows;
3. separate one-next-row behavior from later rollout drift;
4. establish that any difference is stable enough to justify retraining.

The next stage must reuse those artifacts as prior evidence rather than claim
that common-prompt behavior has never been tested.

## What this stage can decide

It can decide:

1. whether a checkpoint difference remains when inference prompt and prefix are
   identical;
2. whether either checkpoint is unusually dependent on the instruction that
   matches its training;
3. whether either checkpoint is more sensitive to the textual order of the
   same emitted objects;
4. whether any difference begins at the immediate next row or appears only
   after rollout has continued.

It cannot decide:

1. whether target-row ordering alone caused the difference;
2. whether the effect repeats across training seeds;
3. whether the result transfers to the current `DoRA` training route;
4. which internal representation carries the difference.

## Strongest competing explanations

1. **Historical-training effect:** the two coupled target-and-prompt regimes
   left different ways to choose and continue objects.
2. **Prompt-wording effect:** each adapter mainly responds best to the ordering
   instruction used with it during inference.
3. **Changing-target effect:** the historical random arm received a different
   target order in each epoch, creating inconsistent next-object targets rather
   than useful order independence.
4. **Single-run or numerical effect:** the difference is not stable across
   seeds, physical batch shapes, or floating-point precision.

## First-stage comparison

Use twelve deliberately selected images:

- three sparse images;
- three dense images with repeated categories;
- three images where low-temperature sampling recovered an object missed by
  greedy decoding;
- three images already used in the recent mechanism work.

An image may satisfy more than one group, but the final set should still cover
all four behaviors.

Run two separate panels because they answer different questions.

### Panel A: same prompt, natural rollout

For each image, compare both checkpoints under the same:

1. geometry-sorted instruction;
2. unrestricted-order instruction.

The prompt token identifiers must be byte-for-byte identical between
checkpoints within each comparison. Do not add an order-neutral prompt in the
first smoke: it would be new wording for both checkpoints and could introduce a
third prompt effect. Add it only if the two existing instructions leave an
unresolved prompt interaction.

Keep the raw greedy rollout. This panel asks whether the two checkpoints retain
different natural behavior after prompt wording is controlled.

### Panel B: same emitted objects, one next row

Use six of the twelve images whose object ownership is visually unambiguous.
At one early and one middle prefix depth, use the unrestricted-order instruction
and give both checkpoints exactly the same teacher rows for the emitted object
set in three orders:

1. geometry-sorted order;
2. reverse order;
3. one fixed random order.

The object set, row contents, prefix length, image, and next valid objects must
remain identical. Only the order of already emitted rows changes. Generate one
complete next row and then stop. A second fixed random order is added only if a
result depends on the first random order.

This panel asks whether either checkpoint mainly reacts to which objects have
already appeared, or to the exact textual order in which those rows appeared.
The unrestricted-order instruction is used here because a reversed prefix would
directly contradict the geometry-sorted instruction and would no longer isolate
prefix-row order.

### Conditional follow-up

Only for two to four cases where the checkpoints disagree, generate two or
three additional rows or run a small low-temperature sample. Do not sample the
whole panel before a concrete disagreement exists.

## Primary observations

Review the raw rows and images before relying on aggregate metrics. Record:

- which object is produced next;
- whether the model stops;
- whether it repeats an already emitted object;
- whether the row is valid;
- whether the description and box refer to the same visible object;
- how the four box coordinates differ;
- whether changing only the prefix row order changes the selected next object.

If the existing scoring path can do so without new model hooks, also compare
the probability of ending, already emitted objects, and valid not-yet-emitted
objects. This probability analysis is useful but does not block the first
behavioral smoke.

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

Run one image, both checkpoints, both existing ordering instructions, and one
shared prefix. The smoke passes only if the effective checkpoint, exact prompt
tokens, image, prefix tokens, decoding settings, and output attribution are
recorded correctly.

## Stop and promotion rules

- If each checkpoint is better only under the instruction matching its
  training, classify the main effect as prompt specialization.
- If the checkpoints behave similarly under both common instructions and under
  shared prefixes, stop and do not claim a stable training-order effect.
- If only each adapter's own loss differs while next-row behavior under common
  conditions does not, stop and classify the result as a target-difficulty
  difference.
- If the same checkpoint difference appears under both prompt wordings and in
  Panel B, expand to a held-out comparison before proposing new training.
- If the difference appears only after several generated rows but not in the
  first next row, treat it as rollout drift or accumulated prefix sensitivity,
  not as evidence of a different immediate object-selection rule.
- Do not make a population or current-DoRA claim from this historical one-seed
  pair.
- Do not attribute a surviving checkpoint difference specifically to target
  row order: the historical training prompt changed with it. A future training
  comparison must keep prompt text fixed while changing target serialization.
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
real smoke identifies a missing seam. The first pass is 24 natural rollouts per
instruction plus at most 72 one-row prefix calls. Run each physical model call
with batch size one; parallelize independent calls across available devices.

Logical artifact root:

```text
outputs/research/qwen3-vl-dense-enumeration/2026-07-17-random-versus-geometry-sorted-common-prompt-prefix-comparison/<run-id>/
```

`<run-id>` means one immutable execution identifier and will be assigned only
after implementation and execution are authorized.

## Decisions still open for discussion

1. Which twelve images best cover the four declared behavior groups?
2. Which six images and two prefix depths have physically unambiguous emitted
   and remaining objects?
3. What repeated checkpoint difference is strong enough to justify a larger
   held-out comparison?
