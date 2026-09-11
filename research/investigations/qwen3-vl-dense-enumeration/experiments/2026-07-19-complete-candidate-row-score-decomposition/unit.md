---
title: Complete Candidate-Row Score Decomposition under Paired Prefix States
description: A four-image, full-model float32 study of where same-covered-set initial orders and their downstream route states change real uncovered, competing, covered, and terminal continuations.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_experiment_local
unit_id: 2026-07-19-complete-candidate-row-score-decomposition
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_bounded_panel
updated: 2026-07-19
---

# Complete Candidate-Row Score Decomposition under Paired Prefix States

Execution is complete. See [results](results.md).

## Decision Question

The preceding short-horizon study proved that two prefixes containing the same
complete object rows in different orders can produce different future physical
owners. This unit asks where that difference first becomes trainable:

1. Does prefix order already change the relative score of candidate paths at
   the first shared continuation boundary?
2. Is the change concentrated in the object description, in the four box
   coordinates, or only in the decision to continue instead of emit the
   terminal token?
3. When the initial boundary is not sufficient, do route-specific generated
   rows create a later score difference that explains premature termination or
   recurrence of a covered object?

The answer selects the next training surface. This unit does not select a
final architecture.

## Model and Numerical Contract

Use the description-first, geometry-sorted, pure-cross-entropy plus token-type
gate Weight-Decomposed Low-Rank Adaptation checkpoint at step 4,887:

```text
/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/
qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/
checkpoints/step-4887/checkpoint.json
```

Use Hugging Face full-model 32-bit floating-point forward computation and
32-bit floating-point log-softmax score accumulation. Scores come from raw,
unmodified language-model-head logits. Repetition penalty, sampling filters,
terminal suppression, and coordinate smoothing are excluded.

The exact multimodal prompt, image, frozen prefix tokens, and generated
route-specific prefix tokens come from the completed Stage 2 artifacts. Never
decode and retokenize a prefix or candidate row.

## Score Definition

For exact candidate row tokens `y_1 ... y_K` under image and prefix `P`, score:

```text
log p(y_k | image, P, y_1 ... y_(k-1))
```

Report raw sums and means for:

1. the object-row-start token;
2. object-description tokens;
3. box structure plus the four coordinate tokens;
4. each of `x1`, `y1`, `x2`, and `y2` separately; and
5. the complete row.

For every selected token, also report its vocabulary rank under the raw
32-bit floating-point logits. Here `x1`, `y1`, `x2`, and `y2` are respectively
the left, top, right, and bottom box boundaries.

At the prefix boundary, separately report:

```text
log p(<|object_ref_start|>) - log p(<|im_end|>)
```

`<|im_end|>` is the terminal token in this task. A one-token terminal score
must never be numerically compared with an unnormalized full-row sum.
Candidates are not normalized into a new probability distribution.

## Frozen Four-Image Matrix

The compact scientific case registry and source-token hashes are in
[cases.json](cases.json). The expanded manifest consumed by the scorer is in
[manifest.json](manifest.json).

### Image 18380: benign route freedom

At the initial boundary, compare the current geometry-sorted prefix with the
seeded shuffled prefix while scoring:

- the real uncovered person `gt_0019` reached first by the current route;
- the real uncovered cup `gt_0018` reached first by the alternative route; and
- the already frozen-prefix cup `gt_0017`.

This is the clean control for two valid first choices that reconverge to the
same strict physical-owner set within four rows.

### Image 9400: imperfect-geometry diagnostic control

At the initial boundary, compare the current prefix with the reversed-earlier-
rows prefix while scoring:

- the generated cup row for `gt_0021`;
- the generated mouse row for `gt_0018`;
- the generated laptop row, which lands on a real laptop or screen region but
  is oversized and does not have a strict owner assignment; and
- the frozen-prefix cup `gt_0016`.

The laptop row is diagnostic only. It cannot authorize a training decision.

### Image 9590: remaining object versus terminal output

At the route-specific boundary immediately before generated row 3, compare
the current prefix with the mild adjacent-swap prefix while scoring:

- the real remaining spoon `gt_0026`, reached only by the current route;
- the real uncovered cup `gt_0022`;
- the recently emitted cup `gt_0024`; and
- the recently generated spoon row shared by both trajectories.

The alternative route emitted the terminal token at this boundary. This is the
cleanest case for distinguishing a continuation-margin failure from a
candidate-row ranking failure.

### Image 19109: same-category covered and uncovered competition

At the route-specific boundary immediately before generated row 3, compare
the current prefix with the reversed-earlier-rows prefix while scoring:

- the uncovered motorcycle `gt_0015`;
- the current route's generated but strictly unmatched motorcycle row;
- the alternative route's generated motorcycle row for frozen-prefix entity
  `gt_0021`; and
- frozen-prefix motorcycle `gt_0017`.

All candidates share the description `motorcycle`. Phrase scores therefore
cannot identify the physical owner by themselves. This case tests whether the
coordinate continuation ranks covered and uncovered physical regions
differently. Dense overlap makes exact physical-owner interpretation
diagnostic rather than population-level evidence.

## Competing Explanations and Predictions

### Immediate candidate-path ranking failure

A verified uncovered path is already below a covered or terminal path at the
earliest token that distinguishes them, or that local margin changes sign
across the two exact prefix states. The first training candidate is an
own-prefix, paired branch-ranking loss. It must allow multiple valid uncovered
positives rather than imitate one canonical next row. A complete-row term may
be used only as a secondary transcription-consistency signal because later
easy tokens can hide the branch that actually controls greedy decoding.

### Description-to-instance or coordinate binding failure

Same-description candidates remain indistinguishable through the description
but separate sharply at one or more coordinate positions. The first treatment
must bind the chosen description to a particular visual instance before or
during coordinate generation. Independent coordinate smoothing is not
supported by this outcome.

### Continuation-only failure

Candidate-row scores remain similar, but only the row-start versus terminal
margin flips. A narrow continuation calibration may be testable, but only if
it does not increase unsupported rows or duplicates.

### Later rollout-state failure

The initial candidate ranking is stable, but the paired route-specific
prefixes separate after generated rows are appended. The next treatment must
use short own-rollout states and future unique physical-object coverage; an
immediate teacher-prefix ranking loss is insufficient.

### No stable signature

The four cases disagree without a physically interpretable pattern. Do not add
a new loss or architecture from this unit.

## Interpretation Safeguards

- A candidate unmatched to official annotations is not automatically a
  hallucination. Entity existence and geometry quality remain separate.
- The generated laptop and dense unmatched motorcycle rows are diagnostic
  controls, not trusted positive targets.
- Exact row likelihood is length-sensitive. Compare the same frozen candidate
  across prefixes, and report both sums and means; do not rank unlike rows by
  a hidden normalization rule.
- One candidate's high score does not prove the model has a complete covered
  set or a persistent ledger.
- A different valid next object is not an error when the short horizon still
  recovers the same physical-owner set.

## Minimal Implementation Boundary

Add one experiment-local scorer and bounded tests. Reuse the existing current
inference runtime, exact position-identifier construction, and row-phase score
functions. The manifest may resolve tokens from immutable Stage 2 artifacts by
an exact selector plus an expected hash; it must not introduce a general query
language or modify `src/`.

Run one image per process so the four cases can share separate graphics
processing units. Physical batch size one is acceptable because the panel is
small and full-model 32-bit floating point is conclusion-critical.

## Stop Rule

Stop after:

1. all four declared boundaries are scored under both exact prefix states;
2. all candidate and prefix token hashes match the Stage 2 sources;
3. row-start, description, coordinate, per-coordinate rank, closure, and
   complete-row scores are present;
4. the interpretation is reconciled with the existing enlarged visual review;
   and
5. one training surface is selected or the result is explicitly classified as
   having no stable signature.

Do not expand to more images, layers, attention maps, or training arms in this
unit. Do not draft an implementation OpenSpec until the result is known.

## Artifact Root

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-19-complete-candidate-row-score-decomposition/
```
