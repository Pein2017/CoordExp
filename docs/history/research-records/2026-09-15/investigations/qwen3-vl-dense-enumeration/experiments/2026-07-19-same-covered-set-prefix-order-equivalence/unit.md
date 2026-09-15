---
title: Same Covered Physical-Object Set under Different Earlier Prefix Orders
description: A bounded fixed-prefix test of whether changing only the order of earlier committed rows changes the next physical-object distribution when the covered set, row count, and final row are held fixed.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-19-same-covered-set-prefix-order-equivalence
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_bounded
updated: 2026-07-19
---

# Same Covered Physical-Object Set under Different Earlier Prefix Orders

## Question

For one fixed image and one model, do two valid detection prefixes that contain
the same physical objects, the same complete rows, the same row count, and the
same final row produce materially different next-object behavior solely because
the earlier rows appear in a different order?

The primary comparison is:

```text
A then B then C
B then A then C
```

Both prefixes cover the same physical set `{A, B, C}`. The complete row for
`C` is byte-identical and token-identical in both prefixes. Only the order of
`A` and `B` changes.

## Why This Unit Comes Before Training Design

Previous completed units established that the most recent row strongly changes
the next-object distribution and that more distant rows remain causally active.
They did not execute the clean comparison above:

- one completed comparison used the same covered set in different orders but
  also changed the final row, leaving recency, geometry, and owner identity
  confounded;
- another held the final row fixed but changed which earlier physical object
  had been covered;
- two later protocols proposed the exact same-set order comparison, but their
  prerequisite gates failed and the order comparison did not run.

The result narrows what information the decoder uses, but it does not by itself
select a training treatment. Strong order sensitivity establishes path
dependence; a later value and safety comparison must determine whether that
dependence is harmful, useful, or merely selects a different valid traversal.
Order robustness is compatible with set-like prefix compression only when both
earlier rows are independently active under symmetric removal controls. If an
earlier row is inactive, order robustness may instead mean that the decoder
uses only the common final row.

## Competing Explanations

### Coverage-state explanation

The prefix is reduced approximately to the physical objects already covered,
plus the current final-row search location. If so, the two main prefixes should
produce similar next-object distributions.

### Order- and recency-sensitive explanation

The decoder retains the particular route by which the same set was reached.
Even with the same final row, moving `A` from two rows back to one row back may
change suppression, successor choice, geometry, or stopping.

### Generic off-policy corruption explanation

One reordered prefix may simply be less compatible with geometry-sorted
training. The paired prefixes therefore use identical complete rows and token
multisets. Symmetric coverage-removal controls test whether each exchanged
earlier row is causally active rather than silently ignored. This unit can
identify local forced-prefix order sensitivity, but it cannot by itself decide
whether that sensitivity is useful or harmful, whether it occurs on natural
own-prefix trajectories, or whether it generalizes across images.

## Scope

### Primary model

- base model:
  `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`;
- adapter: geometry-sorted description-first pure cross-entropy plus token-type
  gate Weight-Decomposed Low-Rank Adaptation checkpoint step `4887`;
- special-token embedding delta from the same checkpoint;
- full-model 32-bit floating point;
- Hugging Face backend with Scaled Dot Product Attention;
- repetition penalty `1.0`.

The Gaussian-coordinate and Ranked Probability Score checkpoint is not a
primary arm. It may be added only after the primary result is interpretable.

### Image and physical-entity authority

The exploratory cohort uses image `2299` from:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
```

This image has a near-complete human relabel of its dense people and ties. The
case specification freezes the selected physical-person identifiers and their
normalized coordinates. Unmatched generated boxes are not automatically
classified as hallucinations.

### Fixed-prefix cases

Four predeclared cases use `B = person-rank-0` and four distinct final rows.
`A` is a previously observed likely successor after `C`; the current primary
checkpoint must independently pass the coverage-removal activation gate before
an order comparison is interpreted.

The executed first pass used:

1. `A then B then C`;
2. `B then A then C`;
3. `B then C`, which leaves `A` uncovered and tests whether `A` is active.

The symmetric `A then C` control, which would leave `B` uncovered, was not
executed. It is now a required follow-up before any general set-like-state or
order-invariance claim.

The first two arms must pass exact token-multiset, row-count, final-row, image,
prompt, and model-identity checks before generation.

## Primary Observation

Free one-row behavior owns the conclusion. For each arm, collect one greedy
continuation and eight paired low-temperature samples:

- temperature `0.4`;
- top-p `0.95`;
- identical explicit seeds across arms;
- stop after one complete detection row, terminal output, malformed limit, or
  the declared token horizon.

Report:

- matched physical owner;
- recurrence of `A` and `B`;
- any already-covered physical person;
- any uncovered physical person;
- terminal, malformed, unmatched, and ambiguous geometry outcomes;
- description-to-geometry agreement;
- paired-seed owner agreement between the two order arms.

Fixed-prefix candidate-row scores are secondary. They must not overrule free
generation if the two disagree.

## Admission and Stop Rules

An individual case is informative for a set-like-state interpretation only if
both exchanged earlier rows are shown to be active under symmetric removal
controls. The executed first pass measured only removal of `A`; it can still
establish local order sensitivity and local suppression of `A`, but it cannot
support a general order-robust or set-like interpretation. A removed owner is
behaviorally live when it appears at least once across the greedy and sampled
one-row continuations. Candidate-row score changes are secondary and do not
satisfy this behavioral gate.

An immediate order effect is strong enough to extend from eight to twenty-four
paired seeds when at least one of the following occurs after physical review:

- the greedy next physical owner changes;
- at least three of eight paired seeds change owner; or
- absolute recurrence of either exchanged owner differs by at least `0.25`.

Interpret the possible outcomes separately:

- both earlier rows are active and the two order arms agree: record bounded
  permutation robustness compatible with a local set-like state, not proof of
  a general covered-set memory;
- both earlier rows are active and the order arms differ: record local sequence
  dependence, then test whether the changed route helps or harms unique valid
  coverage before proposing consistency training;
- either earlier row is inactive: record that this case cannot distinguish a
  set-like state from final-row-dominated or partial-history behavior;
- candidate scores differ but free generation does not: record sub-threshold
  order sensitivity rather than behavioral equivalence.

Do not run a four-row horizon until an immediate physical-owner order effect is
established. If the effect is present in at least two cases, replicate it on two
to four additional human-reviewable images before changing a training target
or architecture. Absence of an order effect after the symmetric activation
gate permits the pure-cross-entropy own-prefix reachability study to proceed;
it does not itself authorize a set-conditioned objective.

## Visual Review

Display-only enlarged crops must be reviewed for:

- every paired seed whose order arms choose different owners;
- every output matched to either exchanged owner;
- every unmatched or low-margin box;
- every malformed output with recoverable coordinates.

Review entity existence/category separately from geometry. Geometry labels are
acceptable, shifted or incomplete, neighbor-contaminated or mixed, or
uncertain.

## Reused Infrastructure and Minimal Implementation

Reuse the current Hugging Face adapter runtime, image/prompt materialization,
seeded sampling, compact parser, and physical-box matching utilities. Add only
one experiment-local runner that appends exact row token identifiers to the
materialized base prompt and validates the paired-prefix invariants.

The experiment-local runner supports the two order arms and the `B then C`
control. It tokenizes every frozen canonical row once, stores the exact token
identifiers and hashes, and appends those identifiers directly. Generated rows
are never decoded and retokenized. The missing symmetric `A then C` control is
recorded as a limitation and next discriminator rather than silently inferred.
This remains a bounded experiment-local implementation, not a shared-interface
change.

No shared inference interface, training infrastructure, model architecture, or
OpenSpec contract is authorized by this unit.

## Non-Goals

- no training or loss comparison;
- no training prescription from order sensitivity or robustness alone;
- no claim of general order invariance from one image;
- no explicit coverage carrier, slot, query, or visual write-back;
- no false-positive, malformed-prefix, shifted-box, or duplicate-prefix study;
- no long rollout before the one-row discriminator passes;
- no population mean Average Precision estimate.

## Artifact Handle

Logical root:

```text
outputs/research/qwen3-vl-dense-enumeration/
  2026-07-19-same-covered-set-prefix-order-equivalence/<run-id>/
```

Every immutable run records the resolved inference configuration, checkpoint
and special-token identities, image and prompt digests, row and prefix token
hashes, cases, seeds, raw outputs, parser status, and matching evidence. The
executed artifacts do not record a source commit or dirty-diff checksum; this
provenance gap is disclosed in the result and must be corrected before the
runner becomes a reusable evaluation contract.

## Result

See [the executed results and bounded verdict](results.md).

## Terminology

- **Exchanged earlier owners `A` and `B`**: two human-adjudicated physical
  people whose complete rows are present in both main prefixes but appear in
  opposite order.
- **Common final-row owner `C`**: the human-adjudicated physical person whose
  byte-identical and token-identical complete row ends both main prefixes.
- **Person rank**: the zero-based position of one of the 38 human-relabeled
  people in the frozen geometry-sorted serialization for image `2299`.
- **Covered physical-object set**: the unique human-adjudicated physical
  entities represented by all complete rows in a prefix, independent of row
  order.
- **Common final row**: a byte-identical and token-identical last complete row
  shared by compared prefixes.
- **Symmetric coverage-removal activation controls**: the shorter `B then C`
  and `A then C` prefixes, which remove `A` and `B` respectively, used to verify
  that both exchanged earlier rows are active before interpreting order
  robustness as compatible with set-like prefix compression.
