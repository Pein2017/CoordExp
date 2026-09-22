---
title: Human-Refined Greedy Set Completion under Controlled Ground-Truth Prefix Assistance
description: An exploratory twelve-image causal study of how much correct serialized history is sufficient for native greedy decoding to enumerate the remaining trusted physical objects, with stopping and one-row interventions used only to diagnose failure.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-22-human-refined-greedy-set-completion-conditions
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-22
---

# Human-Refined Greedy Set Completion under Controlled Ground-Truth Prefix Assistance

## Question

For the frozen geometry-sorted pure-cross-entropy checkpoint, what is the least
ground-truth-derived prefix assistance after which native greedy decoding can
enumerate the remaining trusted physical objects? When it cannot complete the
remaining set, is the first observable obstruction stopping, object selection,
prefix-route dependence, description-to-instance binding, or bounding-box
geometry?

The unit is exploratory. It collects a structured behavior surface and mines
failure families before deciding whether any condition meets a later promotion
requirement. Failure to reach complete coverage does not invalidate the unit.

## Competing Explanations

### Native set completion is present but requires a suitable task state

The model may already recognize most physical objects, while its ordinary
self-generated prefix fails to place the decoder in a state from which all
remaining objects are jointly reachable. A sufficiently informative but still
compact correct prefix should then permit native greedy suffix completion.

### Completion is blocked by stopping rather than object access

Remaining object rows may be locally accessible, but the native terminal token
wins before they are emitted. Temporarily suppressing terminal output should
then recover valid remaining owners without requiring further object hints.

### Completion is blocked by object selection or row coherence

Preventing terminal output may produce duplicates, malformed rows, unsupported
entities, or mixed geometry rather than new owners. A forced trusted row may
change the suffix, but it may replace another stable owner rather than add to
the final set.

### The trusted object is not recoverable from the fixed visual input

Even when only one trusted object remains, a correct prefix and a non-binding
token horizon may fail to produce a coherent owner row. This raises a visual
resolution, representation, or description-to-geometry limitation rather than
a traversal-only explanation.

## Evidence Scope

### Model

- base model:
  `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`;
- adapter family: description-first, geometry-sorted, pure cross-entropy plus
  token-type gate, Weight-Decomposed Low-Rank Adaptation;
- checkpoint:
  `/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_acceler8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/checkpoint.json`;
- full-model 32-bit floating point for conclusion-bearing logits and decoding;
- Hugging Face backend with Scaled Dot Product Attention;
- physical batch size `1` inside each independent GPU process;
- repetition penalty `1.0`;
- no image resize introduced by this unit.

The random-order checkpoint and trained treatment checkpoints are outside the
primary unit. They may be used only in a later replication after the source
mechanism is interpretable.

### Human-refined development images

The unit uses image identifiers:

```text
1584, 2685, 4134, 5001, 6040, 7511,
10707, 13348, 13923, 14038, 14439, 16228
```

Ground-truth authority:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
```

The twelve rows contain 346 Common Objects in Context 80-category annotations,
with 15 through 50 objects per image. They are development and mechanism cases,
not a blind benchmark and not training data.

### Object and geometry semantics

The primary behavior is unique physical-owner and category discovery. Exact
bounding-box extent is analyzed separately because an entity may be discovered
with shifted, partial, oversized, or neighboring-instance-contaminated
geometry.

Automatic matching provides confident and review-needed candidate links. It
must not classify every unmatched prediction as hallucination. Crop-enlarged
review owns ambiguous physical-owner judgments in promoted examples.

Objects supplied inside a forced ground-truth prefix are context, not model
discoveries. Completion is measured only on the objects left for free suffix
generation.

## Controlled Prefixes

### Ground-truth orders

Each image uses precomputed complete ground-truth rows under:

1. geometry-sorted order;
2. reverse geometry-sorted order;
3. category-grouped order with a deterministic within-category geometry sort;
4. fixed seeded random orders.

Random orders are divided into discovery and later confirmation seeds. A
single favorable shuffle is evidence of reachability, not a sufficient general
condition.

These schedules answer a practical question: whether a particular complete
ground-truth serialization places the model in a state from which it can finish
that schedule's remaining objects. They are not, by themselves, causal tests of
row order because different schedules can leave different physical objects for
the suffix.

### Same-remaining-set order controls

For order-causal comparisons, freeze the remaining-owner set to the suffix of
the geometry-sorted schedule at each tested depth. Reorder only the forced
complement. When at least two rows are forced, preserve the same final forced
row across arms and permute only the earlier rows. This keeps the image,
covered-owner set, remaining-owner set, prefix length, and immediate previous
row fixed while varying earlier row order.

The policy schedules and same-remaining-set controls answer different
questions and remain separate in analysis.

### Remaining-object depths

For an image with `N` trusted objects, force the prefix that leaves:

```text
N, 16, 8, 4, 2, or 1
```

objects for free suffix generation, skipping values greater than `N`. The
`N` condition is the clean no-ground-truth-prefix baseline. Broad depths run
first; intermediate depths are added around observed transitions instead of
enumerating every prefix length.

## Decode Conditions

### Native greedy suffix

After the declared prefix, release ordinary greedy decoding with the native
terminal decision. This condition owns claims about native completion.

### Terminal-suppressed diagnostic suffix

At the identical prefix, prevent terminal output until the declared complete-
row budget is exhausted, then permit terminal closure. This is a diagnostic
intervention only. It can separate early stopping from the absence of a valid
next row, but it cannot establish native completion.

Use two terminal diagnostics:

1. suppress only the first terminal choice at a row boundary, then restore
   native stopping immediately;
2. suppress every row-boundary terminal choice until the complete-row budget
   is reached.

Only the first diagnostic can show that a valid remaining row was locally
available immediately behind the native stop. The repeated suppression arm is
an extended-generation probe and must record how many terminal decisions were
overridden.

### Complete-row budgets

For `m` remaining trusted objects, report both:

- strict budget: `m` generated complete rows;
- relaxed diagnostic budget: `m + max(4, ceil(0.25 * m))` generated complete
  rows.

The generated-token ceiling is deliberately non-binding and is fixed at 4,096
new tokens before conclusion-bearing collection. Executed evidence requires
zero token-limit terminations. The
complete-row budget, native terminal token, malformed-row policy, and token
ceiling remain separate recorded termination causes.

## Primary Observations

For every image, order, remaining-object depth, stopping condition, and row
budget, retain:

- complete free suffix and raw token identifiers;
- terminal reason and generated complete-row count;
- trusted remaining owners discovered at least once;
- forced-context owners repeated in the suffix;
- duplicate, malformed, unsupported, unknown, and review-needed rows;
- confident entity coverage and review-expanded entity coverage;
- description-to-owner agreement;
- per-coordinate and complete-box geometry for confidently matched owners;
- the first suffix location where a trusted completion becomes unavailable;
- exact checkpoint, prompt, tokenizer, backend, batch, precision, and decode
  identities.

Physical-owner coverage is computed with a global one-to-one assignment
between generated rows and trusted entities. Ambiguous rows produce a
conservative lower bound and a review-expanded upper bound; they cannot credit
two owners. Any ambiguity that changes complete versus incomplete status
requires crop-enlarged review.

The primary exploratory view is a completion curve over the number of objects
left for native generation. Aggregate Average Precision and mean Average
Precision are secondary summaries.

## One-Row Causal Replay

Mine informative failed or divergent suffix states only after the broad grid.
At one exact native greedy prefix compare:

1. the native next row;
2. one trusted remaining row reachable in another order or stopping condition;
3. a second trusted remaining row when available;
4. one already covered owner row;
5. a category- and length-matched control when available.

Force exactly one complete row, release native greedy decoding, and preserve an
identical remaining complete-row budget. Measure whether the intervention:

- adds a new owner without losing a stable owner;
- produces one-for-one owner replacement;
- selectively suppresses a nearby or same-category owner;
- changes the native terminal position;
- broadly reroutes the suffix;
- or preserves entity discovery while degrading geometry.

The forced row itself is never counted as model discovery.

Before interpreting a forced-row contrast, force the exact native row through
the same append-and-release path. Its released suffix must match the natural
suffix exactly at the raw-token level. The native-row no-op, trusted remaining
row, covered row, and matched control receive identical post-branch row and
token opportunities. Report downstream coverage both including and excluding
the branch row for every arm.

## Teacher-Forced Reference

Teacher-forced sequence likelihood and complete-row likelihood are explanatory
references only. They do not own the conclusion because greedy decoding can
fail at one local token even when the row-average score is high.

For mined states, prefer phase-specific and token-specific records:

- terminal versus row-start margin;
- first category-distinguishing token margin;
- first owner-distinguishing coordinate margin;
- `x1`, `y1`, `x2`, and `y2` target ranks and margins;
- weakest greedy-path margin across the trusted candidate row.

Global shuffled-sequence likelihood may be recorded, but it must not rank a
condition as successful or override free suffix behavior.

## Progressive Execution

1. Validate exact checkpoint, prompt tokens, ground-truth rows, order
   construction, 32-bit floating-point loading, batch-one execution, same-call
   raw-token replay parity, and zero-truncation behavior.
2. Run clean native greedy baselines for all twelve images with a non-binding
   token ceiling.
3. Run a sparse prefix-completion grid and the paired terminal-suppressed
   diagnostic on representative dense and mixed-category images.
4. Expand the completion grid across all twelve images and mine transition
   depths, order-sensitive cases, and failure families.
5. Run one-row causal replay only for cases whose outcome distinguishes
   stopping, additive coverage, owner replacement, local inhibition, route
   change, or geometry failure.
6. Add teacher-forced local margins only for selected causal states.
7. Perform crop-enlarged review of conclusion-bearing unmatched or mixed-owner
   predictions and close the unit with a bounded result.

## Non-Goals

- no training, reinforcement learning, beam search, or architecture change;
- no object slots, external detector, external teacher, coverage carrier, or
  visual-mask intervention;
- no claim that a favorable ground-truth prefix is deployable;
- no population prevalence estimate from twelve selected images;
- no requirement that a predeclared success threshold be met before failed or
  partial observations are analyzed;
- no replacement of native greedy rollout evidence with likelihood.

## Stop Rule

Stop this unit when the twelve-image completion surface and its material
termination failures are recorded, the most informative failure families have
at least one matched one-row causal replay or an explicit reason they are not
replayable, and remaining uncertainty requires a qualitatively new
intervention such as visual designation, learned coverage state, or training.

Because the depth grid is sparse and completion need not be monotone, report
the `least tested assistance` rather than a mathematical minimum. Densify every
intermediate depth around any apparent transition and record reversals before
using stronger language.

## Artifact Handle

Logical root:

```text
outputs/research/qwen3-vl-dense-enumeration/
  2026-07-22-human-refined-greedy-set-completion-conditions/<run-id>/
```

Durable resolved root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-22-human-refined-greedy-set-completion-conditions/<run-id>/
```

Each immutable run retains a compact receipt with source commit or dirty-state
identity, checkpoint and input identities, condition keys, raw outputs,
parser/failure status, zero-truncation status, and the primary sample-level
coverage primitives.

## Result

The executed evidence and final interpretation are recorded in
[results.md](results.md). The unit found that terminal suppression is locally
useful but globally insufficient, earlier prefix order changes later routing
even under an identical covered-owner set and last row, and the first
owner-discriminative coordinate depends on the remaining candidates rather
than occupying one universal token position.
