---
title: Matched Random-Order-Trained versus Geometry-Sorted-Trained Prefix-Order Screen
description: A matched checkpoint comparison of whether training-row order changes sensitivity to earlier prefix order, local covered-object suppression, and complete candidate-row scores.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
training_promotion_status: small_screen_supported_separate_authorization_required
implementation_status: complete_for_unit
unit_id: 2026-07-20-matched-random-sorted-prefix-order-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_bounded_checkpoint_conditional
updated: 2026-07-20
---

# Matched Random-Order-Trained versus Geometry-Sorted-Trained Prefix-Order Screen

## Question

Does randomizing complete object-row order during pure cross-entropy training
reduce harmful dependence on the order of already emitted rows, while
preserving local suppression of covered physical objects and access to valid
uncovered objects?

The alternative is that prefix-path dependence is mainly native to
autoregressive decoding. Under that explanation, both checkpoints remain
order-sensitive, or the random-order-trained checkpoint merely spreads
probability across more routes without improving reliable set completion.

## Checkpoints and the Changed Factor

The two checkpoints share the same base Qwen3-VL model, literal system prompt,
literal user prompt, training data source, description-first row format,
Weight-Decomposed Low-Rank Adaptation rank, token-type gate, optimizer-scale
settings, seed, and nominal training length. Their intended training difference
is the order of complete assistant object rows:

- geometry-sorted-trained checkpoint: complete rows are sorted by geometry;
- random-order-trained checkpoint: complete rows are randomly ordered.

The training runs were produced from different source commits and packing-cache
versions. Existing receipts strongly support equivalent examples, transforms,
packing shape, and supervision content apart from row order, but this pilot is
checkpoint-conditional evidence rather than a publication-grade causal estimate
of ordering policy.

At inference, both checkpoints receive the same literal prompt, exact frozen
prefix token identifiers, image, generation policy, and case specification.
The inference-side `object_ordering` configuration remains geometry-sorted for
both arms because the experiment appends already frozen row tokens and must not
introduce an additional input difference.

## Panel One: Common Physical Objects under Different Earlier Orders

Reuse the five previously activated cases without rebuilding them:

- image `18380`, prefix depth `6`;
- image `9400`, prefix depth `10`;
- image `9590`, prefix depth `10`;
- image `19109`, prefix depths `6` and `10`.

For every existing nonredundant comparison, run one greedy continuation and
paired samples with seeds `101` through `104`, temperature `0.4`, top-p `0.95`,
repetition penalty `1.0`, Hugging Face full-model 32-bit floating point, and a
one-complete-row horizon.

Promote a prefix pair to complete candidate-row scoring when either checkpoint
shows at least one of:

1. different greedy strict physical owners;
2. different strict physical owners in at least two of four paired samples;
3. a greedy difference between a valid row and terminal output, covered-owner
   recurrence, or malformed output; or
4. the same-direction terminal, covered-owner recurrence, or malformed-output
   difference in at least two of four paired samples.

A pair activated by either checkpoint is scored under both checkpoints. A
noncanonical order that is almost always malformed or terminal is generic
corruption and is not promoted as an object-routing case.

## Panel Two: Human-Relabelled Image 2299 Coverage Controls

Reuse the four fixed person tuples from the earlier same-covered-set study.
For each tuple, compare:

- `A then B then C`;
- `B then A then C`;
- `B then C`, which omits `A`;
- `A then C`, which omits `B`.

Run one greedy continuation and paired samples with seeds `101` through `108`
under the same decode settings as Panel One. A tuple is coverage-qualified only
when both omitted owners reappear at least once in their corresponding removal
controls. Without this symmetric activation, an order change is route evidence
but not evidence about a covered-object state.

## Complete Candidate-Row Scoring

For promoted prefix pairs, score the same candidate set under both checkpoints
with full-model 32-bit floating point and raw language-model-head likelihood:

- one verified uncovered row reached by either order arm;
- the competing valid row;
- one covered or repeated row when recurrence occurs; and
- terminal output.

Keep row start, description, box structure, `x1`, `y1`, `x2`, `y2`, row close,
and complete-row score separate. Candidate rows are scored independently; their
scores are not normalized into one artificial candidate distribution.

## Interpretation and Stop Rule

- Lower order sensitivity with preserved coverage supports random ordering as
  a better training base.
- Lower sensitivity with poorer coverage means randomization removed a useful
  traversal scaffold without learning set completion.
- Similar sensitivity supports native autoregressive path dependence rather
  than a geometry-sorted-only artifact.
- Broader sampled support with worse greedy behavior supports a ranking or
  own-prefix calibration problem rather than further randomization.

Stop after both panels and the promoted candidate-row scores are verified. Do
not launch four-row expansion, training, layer scanning, or architecture work
from this unit.

## Artifact Root

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-20-matched-random-sorted-prefix-order-screen/
```

The run retains checkpoint and special-token identities, config fingerprints,
case and prefix hashes, paired seeds, raw generated rows, parsing and matching
status, the scoring-selection receipt, and candidate-row score receipts.

## Closure

The unit is complete. See [results.md](results.md) for the matched-checkpoint
verdict, methodological limits, and the recommended small transition-
calibration training screen. No architecture or training run was launched by
this unit.
