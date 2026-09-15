---
title: Sorted Owner-Basin Landscape, First-Skip, and Repair-Path Dissection
description: Separates strict misses caused by box extent, same-description owner collision, route ordering, and absent usable target-localized coordinate basins on the geometry-sorted step-4887 checkpoint, then tests whether a clean skipped owner can be inserted before or after the native successor without exchanging later owners.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: completed_to_predeclared_stop_rule
unit_id: 2026-08-01-sorted-owner-basin-landscape-and-repair
topic: qwen3-vl-dense-enumeration
status: closed
evidence_status: partial_negative
updated: 2026-08-02
---

# Sorted Owner-Basin Landscape, First-Skip, and Repair-Path Dissection

## Status and Authorization Boundary

This document is the execution contract for the active research goal. It
authorizes the bounded implementation, Graphics Processing Unit inference, and
new diagnostic generation required by Tasks 0 through 9. It does **not**
authorize annotation mutation, checkpoint training, architecture promotion, or
expansion beyond the twelve-image panel.

The unit is restricted to the geometry-sorted step-`4887` checkpoint and the
twelve human-refined images. Forced-continuation rows are excluded from the
primary owner census and natural-history claims. Existing forced-continuation
artifacts may remain separate stress evidence, but they cannot admit an owner,
duplicate, or route transition into this unit.

### Execution closure — 2026-08-02

The sealed Task-4 control smoke reached a mechanical `hold` before any C
sentinel was read. Stop Rule 3 triggered because the representative strict
visible positive had a strong target basin but its frozen shape classifier
returned `part_or_whole_lobes`, not the required `localized_peak`. Stop Rule 9
also triggered because the sealed B2 registry did not prospectively bind the
required same-description, non-overlapping physical control. The observed
background, scan, and covered-owner contrasts remain diagnostics and cannot
substitute for that required matched control.
No non-C freeze receipt was
emitted; Tasks 5 through 9 were therefore not authorized or executed.

The decision-bearing closure is recorded in
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v7/lead-review-receipt.json`.
Detailed observations and claim limits are in `results.md` beside this unit.

## Decision Question

For each physical owner missed by the geometry-sorted checkpoint, which of the
following explanations is supported?

1. the model represents the owner but emits an inaccurate part, extent, or
   nearby box;
2. a foreground or previously covered same-description owner suppresses a
   distinct background owner;
3. the owner has a usable coordinate basin, but the learned sorted traversal
   skips it and does not return;
4. the checkpoint has no usable target-localized coordinate basin for that
   owner under the tested description and self-prefix contexts in the current
   full-image interface;
5. the apparent false positive or false negative is an annotation or
   adjudication problem rather than model behavior.

For clean first-skip cases, a second decision asks whether inserting the
skipped owner before the native later owner or backfilling it afterwards adds
coverage without exchanging the later owner or damaging the remaining suffix.

The unit is designed to decide which repair surface should be studied next:

- output geometry or owner binding;
- covered-owner suppression and duplicate control;
- route ordering and backfill transitions;
- or the full-image visual representation and grounding path.

## Strongest Alternatives

The strongest alternative to a visual-support failure is a probability-search
failure: a real target-localized basin exists, but ordinary greedy decoding,
the sixteen sampled trajectories, or a naive top-`N` coordinate search never
enters it. This is why the unit requires both a canonical-description-
conditioned free coordinate landscape and a Ground-Truth-targeted restricted
landscape.

The strongest alternative to same-description coverage collision is an
ordinary spatial or traversal prior. A large foreground row overlapping a
background owner is only a candidate collision; the claim requires the target
owner's basin to fall selectively after that row enters the self-prefix.

The strongest alternative to a useful repair is owner exchange. Recovering the
skipped owner while losing the native successor or another retained owner is a
route substitution, not a final-set gain.

## Frozen Scope and Artifact Authority

### Model

- Base model:
  `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- Geometry-sorted adapter:
  `/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/adapter`
- Special-token embedding delta:
  `/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/special_token_embeddings`

Model likelihoods used for decisions are computed in full-model 32-bit
floating point. Raw model logits are primary. Repetition-penalty-adjusted
policy logits at `1.0` and `1.1` are auxiliary and must be labelled as policy
scores rather than model likelihoods.

### Panel

The immutable panel input is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/evaluation-inputs/human-refined-12.coord.jsonl`

The file contains twelve images and 346 registered Ground-Truth owners. The
panel is substantially human-augmented, but it is not assumed exhaustive. A
visually supported additional owner receives an independent `aux_owner_id`; it
must not silently mutate the frozen panel inside this unit.

### Existing natural rollout artifacts

Matched-policy native greedy, repetition penalty `1.0`, temperature `0`, max
new tokens `3084`:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-29-three-checkpoint-human-refined12-max3084/sorted/greedy/greedy.json`

Sixteen sampled trajectories, repetition penalty `1.0`, temperature `0.4`,
top-p `0.95`, seeds `21001` through `21016`, max new tokens `3084`:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-29-three-checkpoint-human-refined12-max3084/sorted/sampled/shard-0.json

/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-29-three-checkpoint-human-refined12-max3084/sorted/sampled/shard-1.json
```

The existing aggregate file is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-29-three-checkpoint-human-refined12-max3084/sorted/f1-metrics.json`

It reports a descriptive native single result of `107` true positives and
`239` false negatives and a K=`16` union result of `191` true positives and
`155` false negatives. Those headline values must not be used as the primary
paired baseline because the stored single metric is derived from the standard
repetition-penalty-`1.10` inference artifact while the sampled trajectories use
repetition penalty `1.0`. Task 0 must recompute the native greedy side from the
matched-policy `greedy.json` before any cohort is frozen. This is reanalysis of
existing artifacts and requires no new inference.

The standard inference artifact remains provenance and an auxiliary policy
comparison only:

`/data/CoordExp/outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-step4887-human-refined12-hf-fp32`

The earlier absolute coordinate-confidence visualization is descriptive only:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-30-three-checkpoint-absolute-coordinate-confidence-visualization-v1/manifest.json`

Its `coord_mean` measures confidence in the chosen coordinate tokens. It is not
an object-existence probability and cannot decide whether a missed owner has a
target-localized basin.

### Decode-policy strata

This unit does not designate one repetition penalty as universally correct.
The two established settings answer different questions:

- repetition penalty `1.0` is the primary mechanism stratum because it leaves
  the decoding policy closest to the raw model and already has matched greedy
  and K=`16` natural rollout artifacts;
- repetition penalty `1.10` is the production-relevant greedy stratum because
  it has historically produced the strongest deployed greedy metrics.

Both may be used. Every comparison stays within one declared stratum; owners,
rows, gains, losses, likelihoods, and repair outcomes are never pooled across
the two settings. Raw pre-penalty FP32 logits remain the primary model readout.
Penalty-adjusted logits and decoded outcomes are policy readouts.

At one byte-identical token prefix, compute the raw forward once and derive both
penalty-adjusted policy views from that shared result. If the two policies have
already produced different prefixes, those are different recurrent states and
must not be treated as a matched-logit comparison.

The existing repetition-penalty-`1.0` K=`16` sampling policy is the default
sampling reference: temperature `0.4`, top-p `0.95`, seeds `21001` through
`21016`, and max new tokens `3084`. New bounded sampling under either penalty
is permitted only after its exact `K`, temperature, top-p, seeds, horizon, and
donor-selection rule are frozen in `landscape-decision-rules.json`.

Sampling has asymmetric evidential meaning:

- a strict or meaningful loose target row found by a registered sample is
  positive support and blocks a C verdict in that context;
- failure of any finite sampling budget to find the target supplies no evidence
  for C;
- sampling-derived rows and greedy-derived rows remain separate cohorts.

## Permanent Identity Ledger

Every entity and row keeps a stable identity across filtering, clustering,
manual review, likelihood analysis, and interventions.

### `gt_owner_id`

One permanent identifier for every frozen panel owner. It is never reassigned
after cohort construction.

Recommended form:

```text
gt:<image_id>:<original_annotation_index>
```

### `pred_row_id`

One permanent identifier for every natural rollout row, preserving checkpoint,
decode policy, seed, image, and original row index.

Recommended form:

```text
pred:sorted:<decode>:<seed>:<image_id>:<row_index_0based>
```

The stored row index is zero-based. Human-facing tables may display a separate
one-based row number, but filtering must never renumber rows.

### `diagnostic_owner_id`

One physical-entity identity used by this diagnostic. It maps to exactly one of:

- a frozen `gt_owner_id`;
- a human-adjudicated additional owner `aux:<image_id>:<ordinal>`;
- an unresolved entity `unresolved:<image_id>:<ordinal>`.

Every mapping records evidence provenance and adjudication status. An
additional or unresolved owner is never converted into Ground Truth implicitly.

## Matching and Relationship Semantics

Strict evaluation uses versioned matcher contract
`sorted-owner-basin-matcher.v1`: normalized exact description equality by
default, or one predeclared symmetric alias from the matcher's immutable alias
table; one-to-one maximum-cardinality assignment at intersection over union at
least `0.5`; then maximum total intersection-over-union; then the frozen
Ground-Truth owner index and original prediction-row index as deterministic
tie-breaks. Coordinates are compared in the frozen panel's pixel space after
the production parser's declared coordinate-bin conversion. Competing
same-description assignments with an indistinguishable optimal score are
emitted as ambiguous and remain neutral until reviewed. No learned semantic
similarity or outcome-specific alias may enter strict matching.

The frozen panel's `category_id` is interpreted only in the official gapped
COCO namespace. Evaluator-local contiguous category IDs are never joined to it.
Strict semantic matching is performed from the normalized description and
declared alias table; both category namespaces and their name bridge are
recorded in `matcher-contract.json` so a namespace mismatch fails fast rather
than changing owner counts.

Loose spatial support is a diagnostic relation, not a replacement metric. Any
tiny intersection is insufficient. Candidate evidence includes:

- intersection over prediction;
- intersection over Ground Truth;
- Ground-Truth-center-in-prediction and prediction-center-in-Ground-Truth;
- normalized center distance;
- predicted-to-Ground-Truth extent and area ratios;
- semantic compatibility.

Thresholds for these features must be calibrated and declared using reviewed
non-C examples before cohort labels are emitted. They must not appear as silent
defaults in analysis code. Every proposed C owner must retain its
`no_free_spatial_support` assignment across a predeclared threshold-perturbation
band recorded in `cohort-assignments.jsonl`.

Every prediction receives two independent labels.

`physical_relation`:

```text
new_owner
covered_owner_duplicate
partial_owner
merged_multiple_owners
nested_distinct_owner
incidental_overlap
unsupported
unresolved
```

`semantic_relation`:

```text
exact
compatible_alias
similar_semantic_drift
incompatible
```

A highly overlapping but physically distinct nested object, such as a backpack
on a person, is not a duplicate. A row may exhibit similar-semantic duplication
when it returns to the same physical basin with a nearby but drifting
description. Person face-versus-body extent is the canonical part-or-whole
stress case.

## Primary Owner Cohorts

The primary mechanism census uses matched repetition-penalty-`1.0` natural
greedy and K=`16` sampled trajectories. Repetition-penalty-`1.10` greedy and any
new registered samples form separate policy-support panels. A positive target
hit in any registered panel can block C, but no panel is merged into the
primary strict-rescued denominator.

1. **Strict-rescued**: missed by native greedy at strict matching and recovered
   by the K=`16` union at strict matching.
2. **Loose-only**: still missed strictly, but at least one semantic-compatible
   natural prediction provides meaningful partial, merged, oversized, nearby,
   or extent-shifted spatial support.
3. **No-free-spatial-support**: no semantic-compatible natural prediction in
   matched greedy or K=`16` provides meaningful spatial support.
4. **Unmatched-prediction review**: every unmatched natural prediction is
   adjudicated as an additional owner, duplicate, partial or merged owner,
   unsupported row, or unresolved.

Mechanism-facing shorthand:

- **B1 — geometry or extent mismatch**: a target owner has a semantic-compatible
  spatial footprint, but strict matching fails because localization selects a
  part, superset, neighboring extent, or nearby basin.
- **B2 — same-description multi-owner collision**: a candidate foreground or
  covered row is followed by selective suppression of a distinct
  same-description owner's basin.
- **C — no usable target-localized coordinate basin**: the high-confidence conclusion
  defined below; it cannot be assigned from the natural census alone.

Far-away persons and the three wall-mounted bowls provisionally named during
manual review form an outcome-selected C-sentinel panel. Before any sentinel is
eligible, Task 0 must seal `sentinel-registry.json` with its concrete
`gt_owner_id`, source annotation index, image ID, and the exact checkpoint,
policy, and artifact supporting the prior non-recovery statement. Sentinels
remain eligible as case studies when no clean first-skip trajectory exists, but
they never support a prevalence claim.

## Coordinate-Landscape Definition

Given image `I`, exact self-prefix `P`, and a forced canonical description
`d`, a complete box probability is autoregressive:

```text
P(b | I, P, d)
= P(x1 | I, P, d)
  P(y1 | I, P, d, x1)
  P(x2 | I, P, d, x1, y1)
  P(y2 | I, P, d, x1, y1, x2).
```

The landscape must preserve the selected-token log probability at each of
`x1`, `y1`, `x2`, and `y2`; a scalar row score alone is insufficient.

### Physical-owner basins and extent submodes

A physical-owner basin is a cluster of valid coordinate sequences referring to
one physical instance. One owner may contain multiple raw geometry modes, such
as face, torso, full body, or merged extent. These are extent submodes, not
additional owners.

For a description shared by three people, the ideal landscape contains at
least three owner-level basins. A token top-`3` or top-`10` list can remain
inside one dominant person's basin and therefore cannot establish that the
other two owner basins are absent.

### Canonical-description-conditioned free coordinate landscape

After forcing the canonical description `d`, explore the checkpoint's
high-probability coordinate tree without Ground-Truth coordinate anchors:

1. branch over high-probability `x1` tokens;
2. expand `y1` conditionally and spatially diversify or cluster anchors;
3. expand `x2,y2` into valid extent modes;
4. cluster complete boxes by physical-owner basin and extent submode.

This surface measures spontaneous coordinate accessibility conditional on the
canonical description. Its positive target hit blocks C. Its null cannot
support C because a low-probability target-localized peak may be omitted by
finite free search. Branch budgets, spatial-diversity rules, pruning thresholds,
and sampling policies must be frozen on non-C controls before any C-owner
result is read.

The initial frozen bounded-search selector is exact and score-channel-specific:

1. retain at most `64` valid `x1` bins by descending raw FP32 model log
   probability, breaking ties by ascending `x1`;
2. for each retained `x1`, retain at most `32` `y1` candidates by descending
   joint raw score `log P(x1) + log P(y1 | x1)`, breaking ties by ascending
   `y1`;
3. spatially diversify the resulting anchor pool globally. Seed with the
   highest joint-score anchor, with ascending `(x1,y1)` ties; repeatedly choose
   the candidate with maximal minimum Euclidean distance in coordinate-bin
   `(x1,y1)` space from the selected set, breaking ties by higher joint score
   and then ascending `(x1,y1)`. Stop when the best remaining distance is below
   `24` bins;
4. for each selected anchor, retain at most `16` valid `x2` bins (`x2 > x1`)
   by descending raw conditional log probability, with ascending `x2` ties;
   for each retained `x2`, choose the highest-raw-score valid `y2` (`y2 > y1`),
   with ascending `y2` ties, yielding one complete extent branch.

This is deliberately a bounded tree rather than an exhaustive joint
`x2,y2` search. Every receipt records proposed and retained counts, budget
shortfalls, and tie-breaks. A free-search null is non-evidence; only positive
target support can decide a claim. Repetition-penalty policy views are derived
from the same raw forward but never choose the primary free-tree membership.

### Ground-Truth-targeted restricted landscape

Construct a deterministic candidate bank whose membership is fixed before
reading scores. The target anchor domain spans the complete Ground-Truth
interior in `x1` and `y1`, plus a frozen size-aware margin; it is not merely a
jitter shell around the Ground-Truth top-left corner. For every admitted `x1`
anchor, score the complete `y1` vocabulary, then use a frozen deterministic
extent grid or complete conditional extent scoring.

The bank contains:

- dense deterministic target anchors over the full Ground-Truth interior plus
  margin, with frozen extent candidates;
- boxes around already covered same-description owners;
- equal-size background boxes;
- geometry-sorted or scan-compatible but visually unsupported boxes;
- reviewed part, whole, and merged-extent candidates where applicable.

This surface is primary for C. It asks whether a localized target basin exists
at all, rather than whether free decoding discovers it. Target-side sparsity
could manufacture a false C verdict, so the target domain is dense. Foils are
deterministic, equal-count and equal-weight where compared, and representative;
missing a stronger foil can only inflate target prominence and therefore acts
conservatively against C.

For a fixed anchor, `P(x1,y1)` upper-bounds the probability of every subset of
its subsequent extent continuations. The full target-anchor domain therefore
provides an absence-oriented upper-bound check without requiring global
enumeration of every four-coordinate sequence. Merged extents whose anchors lie
outside the target-localized domain, and part or whole support that fails the
target-basin predicate, are reported as B1 evidence rather than silently counted
as a target-localized owner basin.

### Landscape measurements

For each candidate basin `B_u`:

- **peak height**: maximum complete-box log probability in `B_u`;
- **basin mass**: `logsumexp` over valid candidates in `B_u` with a declared
  proposal measure;
- **peak prominence**: target peak minus the matched foil peak under the frozen
  decision functional;
- **shape**: localized peak, wide ridge, part-or-whole lobes, merged extent, or
  scan-direction ridge;
- **location error**: target owner, covered owner, neighbor, background, or
  unsupported scan position.

Peak height and peak prominence are primary. Basin mass is secondary and may
enter a decision only when candidate proposal weights are demonstrably
comparable. No universal absolute likelihood threshold is predeclared. The
exact usable-peak criterion and calibration quantile must be frozen against
matched visible true-positive and B1 controls from the same image and preferably
the same category, size, and crowding regime before any C owner is scored.
Candidate counts and proposal weights must be matched or explicitly normalized
before basin mass is compared.

Every C-eligible calibration stratum must contain at least three non-C controls:
at least one natural strict visible true positive, at least one B1 or Loose-only
owner, and at least one Strict-rescued owner whose natural greedy route skipped
a target that K=`16` recovered. The last control is required to prove that the
functional recognizes a usable but non-winning basin. A stratum that cannot
meet this minimum, after applying the predeclared category-size-crowding
fallback hierarchy, leaves its candidate owners `unresolved` rather than C.
Every C receipt records the raw margin to the frozen peak/prominence boundary
and the result across the predeclared threshold-perturbation band.

`landscape-decision-rules.json` is the single owner of the deterministic target
domain, size-aware margins, extent grid, foil membership, proposal weights,
basin assignment, free-search budget, usable-peak and prominence functional,
calibration algorithm and quantile, loose-support rules, sampling policies,
canonical-description source and alias policy, and ablation operator. For every
owner it binds the exact canonical-description text and token IDs. Structural
rules also bind any `P(x1,y1)` upper-bound pruning rule and threshold used before
extent expansion; a pruned anchor remains auditable and cannot silently support
an absence claim. Structural rules are frozen with an immutable digest after
the representative non-C smoke.
The predeclared calibration algorithm is then applied to the sealed, non-C
control registry before any C owner is scored, and its output is frozen in
`landscape-calibration-receipt.json`. Any post-freeze rule change, control-set
change, or calibration-receipt change voids every previously scored C result.

## Required Landscape Controls

1. **Positive owner controls**: natural strict true positives expected to show
   localized owner basins under the same scorer and candidate construction.
   For ablation sensitivity, prefer a difficulty-matched B1 or Loose-only owner
   of similar category, size, and crowding over an easy strict true positive.
2. **Covered-owner controls**: same-description owners already present in the
   self-prefix.
3. **Background and scan foils**: equal-size visually unsupported boxes and
   coordinates favored by the learned sorted scan.
4. **Target-region ablation**: replace only the target-region pixels with one
   operator frozen in `landscape-decision-rules.json` while preserving canvas
   size, then re-encode the full image and repeat the target landscape. The
   artifact records every operator tried on non-C controls before the freeze.
5. **Equal-area background ablation**: the same operation on a matched
   non-target region.

Whole-image swapping and crop-rescale are out of scope. The conclusion is about
the current full-image enumeration interface. Crop-rescale would change the
resolution and task surface rather than diagnose this interface.

## High-Confidence C Criterion

An owner may be labelled C only when all conditions hold:

1. matched-policy natural greedy and K=`16` classify it as
   `no_free_spatial_support`, the assignment survives the frozen threshold
   perturbation band, and neither the repetition-penalty-`1.10` greedy panel nor
   any registered sample supplies positive target support;
2. description-conditioned one-row greedy and registered bounded sampling
   provide no strict or meaningful loose target localization. A sampling null
   contributes no positive evidence for C;
3. the canonical-description-conditioned free coordinate landscape contains no
   target-owner basin. This null is necessary but does not positively support C;
4. the restricted Ground-Truth target domain contains no usable localized peak
   or prominence over covered-owner, background, and scan foils under the
   frozen decision functional;
5. matched visible true-positive, B1 or Loose-only, and Strict-rescued controls
   in the admitted calibration stratum all exhibit the expected usable local
   peaks under the frozen functional;
6. no ablation veto is triggered. A selective target-region effect blocks C;
   a null effect does not confirm C. The same operator must show adequate
   sensitivity on a difficulty-matched non-C manipulation control, while the
   equal-area background ablation does not mimic the target effect;
7. the conclusion is consistent in every required declared context: `P_pre`
   and `P_post` for a clean first-skip owner; or root, natural-stop, and the
   reference scan-position self-prefix for a sentinel or owner without a clean
   first-skip case. If that reference context is not constructible, the owner
   remains `grounding_or_order_conditioned_accessibility_unresolved` and is not
   C eligible.

If unmatched-row adjudication reveals a systematic compatible alias for the
owner category, repeat the free and restricted landscapes under that alias
before C may be assigned.

When all gates pass, the permitted conclusion is:

> Under the tested canonical description(s), declared exact self-prefix
> contexts, and the current full-image enumeration interface, the Sorted
> checkpoint has no usable target-localized coordinate basin for owner `u`.

Here, `usable` means passing the frozen peak and prominence criterion in
`landscape-decision-rules.json`. Part, whole, and merged-extent support remains
B1 evidence when it does not satisfy the target-localized basin predicate. The
unit does not claim absence of internal visual information and does not identify
which internal module lost the route to the output. It must not claim that the
vision tower, projector, decoder, or image resolution alone is the cause.

## Context Construction

All generated-history probes use exact self-prefix token identifiers. Decoded
text is never re-tokenized to reconstruct a state, and Ground-Truth-clean rows
never replace native history.

### Clean first-skip contexts

Define the canonical reference order by `(y1, x1)`. For a Ground-Truth sequence
`{a,b,c,d}` and a native rollout `{a,b,d}`, admit:

- `P_pre`: exact native rows through `{a,b}`, immediately before native `d`;
- `P_post`: exact native rows through `{a,b,d}`.

Admission requires:

- every row in `P_pre` has a clear physical-owner assignment;
- `c` is the unique earlier skipped owner under the reference order;
- native `d` is a valid new owner;
- no unresolved unmatched, duplicate, or malformed row occurs before `d`;
- repair-path cases preferably have a credible K=`16` self-generated row for
  `c`.

Fewer than one admissible case per image is acceptable. Ambiguous histories
must not be repaired synthetically.

### Sentinel contexts

Owners without a clean first-skip case use:

- root context immediately after the prompt and before the first row;
- natural-stop context immediately before the native terminal token;
- when every preceding native row has a clear owner assignment, a reference
  scan-position self-prefix immediately before the first native row whose
  `(y1,x1)` reference position follows the sentinel. This context drops only
  the unique-first-skip requirement and is not repair eligible.

These contexts support the C test but not the ordered repair comparison. If the
reference scan-position context cannot be constructed, a root-plus-stop null is
reported as `grounding_or_order_conditioned_accessibility_unresolved` and cannot
route the program specifically to a visual-grounding architecture.

### B2 contexts

For each same-description collision candidate, compare:

- the exact self-prefix immediately before the candidate covering row;
- the exact self-prefix immediately after that row.

Geometry overlap alone does not establish collision. The decision depends on
the before-to-after change in the background owner's basin relative to:

- at least one same-description, non-overlapping owner at a matched sorted-scan
  offset from the covering row; and
- where available, one different-description owner with matched spatial
  relation to the covering row.

Collision is supported only when the overlapping owner falls relative to both
controls. If matched controls do not exist, B2 remains unresolved rather than
being inferred from overlap or a selective raw decrease alone.

## Description-Conditioned Generation Ladder

The canonical row prefix is forced only through description and box start:

```text
<|object_ref_start|>{description(C)}<|object_ref_end|><|box_start|>
```

No spatial words or hidden owner identifier are supplied. When multiple
physical instances share the description, this is description-conditioned,
not physical-owner-conditioned.

Run three deterministic rungs separately at repetition penalties `1.0` and
`1.10`:

1. **Description only**: greedily generate all four coordinates.
2. **Ground-Truth `x1` anchor**: supply `x1`, then generate `y1,x2,y2`; this is
   a realization upper bound.
3. **Ground-Truth `x1,y1` anchor**: supply `x1,y1`, then generate `x2,y2`; this
   is an extent upper bound.

The landscape is the primary potential readout. A registered bounded sampling
arm may additionally generate description-conditioned one-row candidates at
either repetition penalty. Its exact policy is frozen before C scoring. A
strict or meaningful loose sampled target row blocks C; a sampling null does
not support C.

If a description-conditioned row selects an already covered same-description
owner, label it a covered-owner duplicate. If its description drifts, use the
independent physical and semantic relationship labels rather than forcing a
duplicate verdict.

Only a strict target-owner row from the description-only rung may enter the
primary full-horizon repair test. If deterministic greedy does not produce one,
a strict sampled row may enter an explicitly secondary sampled-donor repair
cohort under the same repetition-penalty stratum. The donor is selected by a
predeclared rule, such as the first strict row in ascending frozen seed order,
never by post-hoc visual preference. Meaningful loose rows are retained as
geometry evidence but cannot claim route restoration.

## First-Divergence Map

Using the same self-prefixes and frozen candidate rows, locate where the skipped
owner loses to the native successor or terminal action:

```text
STOP versus row opener
description tokens
x1
y1
x2
y2
```

Record both raw and policy-adjusted logits. This directly tests whether the
checkpoint's Ground-Truth ordering by `(y1,x1)` conflicts with its emitted
serialization `x1,y1,x2,y2`. The unit diagnoses that possibility; it does not
change output serialization or training order.

## Native Replay Admission

Before any self-prefix intervention receives causal interpretation:

- a stored greedy context must reproduce the native next row, termination, and
  suffix exactly under the current FP32 runtime and the same repetition-penalty
  stratum;
- stored sampled token sequences require teacher-forced chosen-token parity and
  span alignment, not greedy regeneration.

A context that fails native replay is an identity or runtime mismatch. It is
excluded from intervention analysis and routed to stop rule 1.

## Full-Horizon Repair Factorial

For every native-replay-admitted clean first-skip case whose description-only
rung produces a strict self-generated row `C_self`, run separately inside each
repetition-penalty stratum:

1. **Native**: preserve `P_pre`, native `D`, and the original natural suffix.
2. **Prevent-skip**: `P_pre + C_pre_self`, then release natural greedy decoding.
3. **Backfill**: `P_post + C_post_self`, then release natural greedy decoding.

All released arms terminate naturally with max new tokens `3084`. Report:

- retention of `C` and native `D`;
- gained, retained, and lost unique physical owners against Native;
- duplicate, unmatched, unsupported, and invalid rows;
- strict geometry retention;
- return to the sorted traversal rail;
- natural stop timing and termination reason.

Interpretation:

- `C` gained and `D` lost is owner exchange, not coverage improvement;
- `C` followed by `D` and a retained suffix shows positive route value
  conditional on inserting `C` before `D`;
- `D -> C` followed by repeat `D` indicates last-position or coverage
  suppression;
- `D -> C` followed by later owners shows positive route value conditional on
  backfilling `C` after `D`;
- `C` followed by STOP shows row realizability without route repair;
- additional later owners may be a favorable route reset, but generic insertion
  perturbation remains an unresolved alternative;
- malformed grammar or box collapse indicates an off-manifold intervention.

This factorial estimates insertion-conditional route value. It does not prove a
`C`-specific repair mechanism relative to arbitrary inserted rows. A future
mechanism-claiming unit must add an on-manifold control insertion, preferably a
registered natural sampled row for a different uncovered owner.

## Owner-Basin Suppression Matrix

At each admitted before-or-after context, compute the basin mass of every
same-description physical owner.

After emitting owner `g`, the ideal coverage update is:

- `g`'s basin falls strongly;
- other uncovered owner basins remain accessible.

Diagnostic patterns:

- every same-category basin falls: category-level suppression;
- a spatially overlapping same-description background basin falls relative to
  both the scan-offset-matched non-overlapping same-description foil and, where
  available, the spatially matched different-description foil: candidate
  coverage collision;
- `g`'s basin does not fall: weak coverage update and duplicate risk;
- an unrelated owner's basin rises or falls: route-state redistribution.

Natural duplication timing is summarized as consecutive, short-gap, long-gap,
or cyclic recurrence, with extent drift tracked separately. Forced rows after a
natural stop are excluded from this timing analysis.

## Required Artifacts

Authorized execution must produce at least:

```text
artifact-manifest.json
matcher-contract.json
sentinel-registry.json
landscape-decision-rules.json
landscape-calibration-receipt.json
lead-review-receipt.json
owner-ledger.jsonl
prediction-row-ledger.jsonl
owner-trajectory-matrix.jsonl
cohort-assignments.jsonl
manual-review-queue.jsonl
native-replay.jsonl
landscape-candidates.jsonl
landscape-scores.jsonl
landscape-summary.json
sampling-support.jsonl
first-skip-contexts.jsonl
description-conditioned-ladder.jsonl
repair-factorial.jsonl
basin-suppression-matrix.jsonl
first-divergence.jsonl
```

Every artifact records source digests, model identity, tokenizer identity,
precision, repetition-penalty stratum, decode policy, prompt-token digest,
image identity, and the stable owner and row identifiers. Decision-bearing rows
retain raw scores and must be reconstructible without reading a visualization.
All Task `0` through `4` artifacts use explicit schema versions and declare
their foreign keys, cardinality, nullable fields, status enums, unresolved-
neutrality behavior, coordinate space and rounding, canonical-description text
and token identity where applicable, and the upstream artifact and rule
digests on which each row depends. Execution receipts additionally record the
implementation commit plus dirty-diff digest, command line, Python, Torch,
Transformers, CUDA and device identity, and the declared numeric reproduction
tolerance.

## Execution Tasks

All tasks are initially unchecked. Implementation and bounded execution are
authorized by the active goal, subject to the gates and stop rules below.

### Task 0 — Freeze sources and establish a matched-policy baseline

- [ ] Hash the panel, model components, natural greedy artifact, and both
  sampled shards.
- [ ] Verify image IDs, prompt tokens, checkpoint identity, max-new-token limit,
  precision, repetition penalty, sampling seeds, and termination metadata.
- [ ] Recompute native greedy and K=`16` strict metrics using repetition penalty
  `1.0` on both sides without new generation.
- [ ] Seal `matcher-contract.json` with the versioned alias table, coordinate
  conversion, cardinality-first assignment, tie-break, ambiguity behavior,
  official gapped COCO category namespace and name bridge, horizon `3084`,
  greedy seed `0`, and sampled seeds `21001` through `21016`.
- [ ] Record the repetition-penalty-`1.10` production-greedy artifact as a
  separate policy stratum rather than mixing it with the primary K=`16`
  denominator.
- [ ] Report both denominators explicitly: the new matched-`1.0` cohort and the
  historical repetition-penalty-`1.10`-based greedy-missed set. Divergence from
  the prior `88` recovered-owner denominator is expected, not a contradiction.
- [ ] Seal `sentinel-registry.json` with concrete IDs, non-recovery provenance,
  policy strata, and the outcome-selected/no-prevalence declaration.

**Gate:** any identity or policy mismatch stops cohort construction until it is
resolved. The historical `107/239` versus `191/155` comparison is not promoted
as the paired baseline without this task. No sentinel is eligible before the
registry is sealed.

### Task 1 — Build permanent owner and row ledgers

- [ ] Assign every frozen owner a `gt_owner_id`.
- [ ] Assign every natural prediction its immutable `pred_row_id` and preserve
  the original row index.
- [ ] Create `diagnostic_owner_id` mappings for Ground Truth, reviewed extra
  owners, and unresolved entities.
- [ ] Build an all-owner-by-all-natural-trajectory presence, overlap, semantic,
  and strict-match matrix.
- [ ] Prove that forced-continuation rows are absent.

**Gate:** no cohort, visualization, or repair task may use a renumbered or
untraceable prediction.

### Task 2 — Freeze the strict, loose, unmatched, and sentinel cohorts

- [ ] Reproduce strict one-to-one owner matching.
- [ ] Calibrate meaningful loose-support thresholds on reviewed examples.
- [ ] Freeze the threshold-perturbation band and require a stability receipt for
  every proposed C owner.
- [ ] Assign Strict-rescued, Loose-only, and No-free-spatial-support cohorts.
- [ ] Adjudicate every natural unmatched prediction on both physical and
  semantic axes.
- [ ] Register far-person and wall-bowl sentinel owners.
- [ ] Emit the manual-review queue and explicit unresolved set.

**Gate:** an unresolved row may remain neutral but cannot decide B2, C, or a
repair gain.

### Task 3 — Implement and attest both landscape surfaces

- [ ] Implement canonical-description-conditioned free coordinate-tree
  exploration with spatial diversification and complete-box clustering.
- [ ] Implement the restricted full-Ground-Truth-interior-plus-margin target
  anchor grid, complete conditional `y1` scoring, frozen extent candidates,
  and deterministic covered-owner, background, scan, part, whole, and merged
  banks.
- [ ] Store raw `x1,y1,x2,y2` selected-token log probabilities and complete-box
  scores.
- [ ] Treat prefix-key/value reuse as provisional until one real-checkpoint
  receipt reproduces uncached full-prefix FP32 logits at root, `x1`, `y1`, and
  `x2` depths within `atol=1e-6`, `rtol=1e-5`, preserves all 28 cache layers
  after serial crop, and gives branch-order-invariant rows. Cached branching
  uses explicit Qwen multimodal position IDs and rope deltas, never
  `generate()`, shallow cache copies, concurrent use of the same model session,
  or repeated image inputs after prefill.
- [ ] When the uncached fallback is selected, permit only equal-length,
  same-context GPU batching of literal `use_cache=False` full-prefix
  reforwards. `--full-reforward-batch-size=1` remains the scalar reference.
  A live batch must preserve coordinate-domain argmax at the first three
  coordinate depths and keep every coordinate log-probability within `1e-3`
  of the scalar path; otherwise the run automatically falls back to batch
  size `1`. Receipts record requested/effective batch size, physical model
  forwards, batch histograms, padding, and the admission comparison.
- [ ] Compute peak height, normalized basin mass, prominence, shape, and basin
  identity.
- [ ] Add matched visible true-positive controls.
- [ ] Add target-region and equal-area background ablations with full-image
  re-encoding.
- [ ] Draft `landscape-decision-rules.json` from non-C controls, including all
  target/foil rules, proposal weights, calibration quantile, sampling policies,
  canonical-description and token-ID rules, loose-support rules, ablation
  operator trial log, and invalidation rule.

**Gate:** a representative positive control must show the expected localized
peak, target coverage must be dense, peak prominence must reproduce, and basin
mass may be used only where proposal measures are comparable. Raw FP32 scores
must reproduce before the landscape can adjudicate any C owner. Failure of the
cached-versus-uncached parity gate disables cache reuse and falls back to the
uncached reference; it does not relax the scoring tolerance. Batching the
uncached reference is a throughput optimization, not cache reuse, and cannot
enter a result unless its separate coordinate-behavior admission passes.

### Task 4 — Run one representative end-to-end smoke

- [ ] Trace the implementation on one strict visible positive, one
  difficulty-matched B1 owner, and one B2 control set; no C or sentinel outcome
  may be used to fit a threshold or select a rule.
- [ ] Trace the non-C controls from immutable source rows through identifiers,
  cohort relation, candidates, likelihoods, basin clustering, ablation, and
  summary.
- [ ] Independently reconstruct the conclusion-bearing values from raw rows.
- [ ] Freeze and digest the structural `landscape-decision-rules.json`; its
  calibration algorithm and quantile are fixed, but the control-derived value
  is not populated until the sealed control-only pass.
- [ ] Only after the freeze, score one sealed C-sentinel case as a blinded
  end-to-end attestation without changing any decision rule. This smoke result
  is case-level diagnostic evidence only and remains `unresolved`; it cannot
  assign C before the control-only calibration receipt is sealed.
- [ ] Exercise one native first-skip replay, one description-conditioned ladder
  case, and one eligible repair arm when available.
- [ ] Emit `lead-review-receipt.json` binding all reviewed artifact and rule
  digests, every stop-rule state, and exactly one disposition:
  `proceed_full`, `proceed_census_only`, `narrow`, or `hold`.

**Gate:** stop after this smoke for research-lead review. If the smoke and
decision-rule receipts pass, a fresh lead-review receipt authorizes automatic
continuation without another user approval. `proceed_census_only` may schedule
Task `5` and applicable closeout only; Tasks `6` through `8` require
`proceed_full`. Missing or stale receipts block continuation. Any decision-rule
change after the C attestation voids that case result and requires a new sealed
attestation.

### Task 5 — Execute the twelve-image owner-basin census

- [ ] Score the sealed non-C control registry first, apply the predeclared
  calibration algorithm, run leave-one-control-out and threshold-band
  sensitivity, and freeze `landscape-calibration-receipt.json` before reading
  any C-owner landscape.
- [ ] Score both landscape surfaces for every strict miss and all matched
  positive controls.
- [ ] Record registered greedy and sampling support separately at repetition
  penalties `1.0` and `1.10`; treat sample hits as positive and sample nulls as
  non-evidence.
- [ ] Produce per-owner B1, B2-candidate, or C-gate receipts.
- [ ] Separate physical-owner basins from extent submodes.
- [ ] Apply the seven-part C criterion without an absolute global threshold.

**Gate:** a failed positive control invalidates the affected category, size, or
context stratum rather than proving target absence.

### Task 6 — Extract clean contexts and duplication chronology

- [ ] Construct clean first-skip `P_pre` and `P_post` self-prefixes.
- [ ] Construct root, natural-stop, and reference scan-position sentinel
  contexts; emit the specified unresolved status when the reference context is
  not constructible.
- [ ] Construct before-and-after B2 contexts.
- [ ] Summarize natural duplicate recurrence by immutable row index and owner.
- [ ] Reproduce every admitted native greedy next row, terminal action, and
  suffix in the current FP32 runtime, and verify teacher-forced token parity for
  any stored sampled donor.

**Gate:** ambiguous or contaminated histories are reported but not admitted to
causal repair. A context that fails native replay or sampled-token parity is
routed to stop rule 1 and cannot receive causal interpretation.

### Task 7 — Run the description-conditioned ladder and first-divergence map

- [ ] Run the description-only, Ground-Truth-`x1`, and
  Ground-Truth-`x1,y1` deterministic rungs separately at repetition penalties
  `1.0` and `1.10`.
- [ ] Run any registered bounded sampling arm and retain its positive-support
  rows without treating a null as C evidence.
- [ ] Assign every generated box to a physical basin and semantic relation.
- [ ] Locate the first decision boundary where target `C` loses to native `D`
  or STOP.
- [ ] Compare raw and repetition-penalty-adjusted logits.

**Gate:** only strict greedy description-only target rows proceed to primary
full-horizon repair. A strict donor selected by the frozen sampled-donor rule
may proceed only to the separately reported secondary repair cohort.

### Task 8 — Run prevent-skip and backfill full-horizon repairs

- [ ] Reconfirm native replay admission, then execute Native, Prevent-skip, and
  Backfill under natural termination inside each repetition-penalty stratum.
- [ ] Report gained, retained, and lost owners; retain `C` and `D` separately.
- [ ] Report duplicates, unsupported, unmatched, invalid, geometry loss, route
  rail, and stop timing.
- [ ] Mark owner exchange explicitly.
- [ ] Record generic insertion perturbation or route reset as an unresolved
  alternative for every suffix-gain result.

**Gate:** no row-level success is promoted if final unique-owner coverage falls
or if a gained owner is paid for by loss of native `D` without a declared
tradeoff.

### Task 9 — Estimate owner-basin suppression and close the unit

- [ ] Compute every admitted before-or-after same-description suppression
  matrix.
- [ ] Separate owner-specific suppression, category suppression, failed
  coverage update, and general route redistribution.
- [ ] Close `results.md` under `Observed`, `Supported`, `Ruled Out`, `Unresolved`,
  and `Not Claimed`.
- [ ] State that the twelve-image, outcome-selected panel does not estimate C
  prevalence and that prior complete-description recovery predicts a small C
  cohort outside the named sentinels.
- [ ] Route the next study toward policy repair, geometry or binding, duplicate
  control, or visual-grounding architecture using the interpretation matrix.

## Interpretation Matrix

| Observation | Supported reading | Next research surface |
|---|---|---|
| Natural loose box plus strong target-localized landscape basin | Owner exists internally; strict miss is geometry or extent | Geometry and owner-binding training |
| Canonical-description-conditioned free landscape misses target but the restricted landscape has a prominent local basin | Bounded free search or routing fails to enter a low-probability target basin | Selector, route calibration, or basin-aware decoding |
| Target basin exists at `P_pre` and Prevent-skip retains `C,D` | Inserting `C` before `D` has positive conditional route value; generic insertion reset remains unresolved | Candidate prevent-skip training screen, not a C-specific mechanism claim |
| Target basin survives `P_post`, and Backfill yields `D,C` plus later owners | Backfilling `C` has positive conditional route value; generic insertion reset remains unresolved | Candidate backfill training screen, not a C-specific mechanism claim |
| Target basin selectively collapses after an overlapping same-description row relative to both required matched foils | Same-description owner collision is supported under the matched controls | Coverage update and physical-owner separation |
| Emitted owner's own basin remains high and natural rollout later returns | Covered-owner suppression is weak | Duplicate-aware state or objective |
| Description-only fails but Ground-Truth `x1` or `x1,y1` succeeds | Completion competence exists; failure is concentrated at initial anchoring or owner disambiguation, which this rung cannot separate | Coordinate boundary or geometry treatment |
| Both landscapes and the ladder fail under the frozen rules, matched controls including Strict-rescued owners pass, no registered sample finds support, and no ablation veto fires in every required declared context | High-confidence conditioned C | Full-image owner-grounding route; sentinel-only root-plus-stop evidence without a reference scan-position context remains grounding-versus-order unresolved |
| `C` is gained but `D` or another retained owner is lost | Owner exchange | Do not promote as set improvement |

## Unit Stop Rules

Stop or narrow the unit when any of the following occurs:

1. source identities, prompt tokens, model identity, or matched decode policies
   cannot be reconciled;
2. stable physical-owner and prediction-row identities cannot be reconstructed;
3. the representative positive control lacks a localized landscape peak;
4. free and restricted landscape scores cannot be reproduced in FP32;
5. `landscape-decision-rules.json` is missing, changes after C scoring, or
   cannot reproduce its digest and calibration receipts;
6. most candidate first-skip histories are ambiguous, in which case retain the
   owner-basin census and drop the repair factorial rather than relaxing
   admission;
7. description-only strict target rows are absent from greedy and registered
   sampling, in which case close repair
   as unsupported and use the likelihood landscapes as the decision-bearing
   result;
8. repair gains are primarily owner exchanges, geometry regressions, or
   off-manifold continuations;
9. B2 candidates lack matched foils or do not show relative before-to-after
   target-basin suppression, in which case close collision rather than
   inferring it from overlap.

The unit ends after the twelve-image panel. It does not expand to validation
200, add crop-rescale, add an unregistered or adaptively expanded sampling
policy, train a new checkpoint, or promote an architecture without a new owning
decision.

## Reporting Contract

The final report must keep separate:

- strict owner identity and strict geometry;
- natural evidence and forced intervention evidence;
- owner basins and extent submodes;
- canonical-description-conditioned free support and Ground-Truth-targeted
  restricted support;
- raw model likelihood and repetition-penalty-adjusted policy score;
- repetition-penalty-`1.0` and repetition-penalty-`1.10` strata;
- greedy-derived and sampling-derived support, with sampling nulls explicitly
  marked as non-evidence;
- gained, retained, and lost unique owners;
- Ground Truth, auxiliary reviewed owners, and unresolved entities;
- observed facts, supported interpretations, unresolved alternatives, and
  claims not made.

Loss, mean Average Precision, prediction count, coordinate confidence, or a
single sampled-union recall number cannot replace the per-owner causal and
landscape accounting defined here.
