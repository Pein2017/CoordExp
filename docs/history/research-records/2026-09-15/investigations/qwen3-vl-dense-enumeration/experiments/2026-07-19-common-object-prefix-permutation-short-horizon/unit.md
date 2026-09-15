---
title: Common Physical Objects under Different Prefix Permutations and a Short Future Horizon
description: A bounded pure-cross-entropy probe of whether earlier row order changes the next object and whether initially different valid routes converge to the same uncovered set within four rows.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_experiment_local
unit_id: 2026-07-19-common-object-prefix-permutation-short-horizon
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_bounded
updated: 2026-07-19
---

# Common Physical Objects under Different Prefix Permutations and a Short Future Horizon

## Decision Question

When a detection prefix contains exactly the same complete object rows, does
changing only their earlier order, while preserving either a common final
one-row or two-row suffix, alter:

1. the next physical object selected by Qwen3-VL; and
2. the unique uncovered objects, duplication, and stopping observed over the
   next four complete rows?

The second question determines the practical meaning of the first. A changed
next object may be harmless route freedom if both paths soon recover the same
set. It is a useful training target only if one path loses valid objects,
repeats covered objects, becomes malformed, or stops earlier.

## Why This Unit Is Needed

The completed three-row experiment showed two behaviors in the same model:

- an emitted physical person could be suppressed under either of two earlier
  orders; and
- exchanging two earlier rows while preserving the covered set and final row
  could change the next physical owner.

That experiment had one image, three-row prefixes, one-sided omission controls,
and only one generated row. It could not tell whether the changed route had any
cost for set completion. The current unit extends only the unresolved parts:

- prefixes of six and ten complete rows;
- paired common final one-row and two-row suffixes, so influence at two
  controlled history distances can be compared;
- one mild adjacent swap and four stronger fixed permutations of the same
  exact rows; and
- a gated extension from one generated row to four.

No training algorithm or final architecture is selected by this unit.

## Models and Data Sources

### Primary model under intervention

- base model:
  `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`;
- adapter: description-first, geometry-sorted, pure cross-entropy plus
  token-type gate, Weight-Decomposed Low-Rank Adaptation rank 16, checkpoint
  step 4,887;
- checkpoint:
  `/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/checkpoint.json`;
- full-model 32-bit floating point Hugging Face inference;
- repetition penalty `1.0`.

The Gaussian-coordinate plus Ranked Probability Score adapter is excluded.

### Provisional common-object cohort

Candidate prefix entities are selected from the physical ground-truth indices
that both of these completed greedy rollouts matched by category and bounding
box overlap:

1. the current geometry-sorted pure-cross-entropy step-4,887 rollout; and
2. the historical random-order pure-cross-entropy checkpoint-3,668 rollout.

The historical random checkpoint is used only to select objects that were
reachable under both rollouts and to supply one deterministic relative order.
It is not compared as a model arm and contributes no evidence about the effect
of random-order training in this unit. It differs in base snapshot, training
duration, Low-Rank Adaptation setup, prompt details, and token-type supervision.

The two rollout artifacts contain identical, index-aligned ground-truth arrays
for each selected image, but those arrays predate the current relabeled
`val.coord.jsonl`. Therefore the common-object cohort is explicitly
artifact-local and provisional. Original-image review, rather than current
dataset index equality, owns physical interpretation. The current matched
random-order run will receive an exact replication after it produces a
checkpoint.

The frozen prefix and the matching ledger have different roles. The frozen
prefix contains only the six or ten common entities selected above. Generated
rows are matched against the full artifact-local COCO-80 entity ledger for the
image. This prevents a real object outside the common-object prefix set from
being mislabeled merely because it was not eligible for prefix construction.
The full artifact-local ledger is still incomplete annotation evidence, not
exhaustive ground truth; conclusion-changing unmatched predictions require
enlarged-crop review.

### Four-image panel

The bounded panel uses:

- image `18380`: a crowded meal with people, tableware, and food;
- image `9400`: a low-light group with people and multiple computers;
- image `9590`: people and many table objects;
- image `19109`: a dense street scene with motorcycles and people.

These images have at least twelve artifact-local physical entities matched by
both source rollouts. The panel is intentionally four images: enough to avoid a
single-scene anecdote while remaining small enough for enlarged-crop review.

Image `2299`, which now has a near-complete human relabel, remains the authority
for the completed three-row result. It is not folded into this long-prefix
common-object panel because the historical artifacts contain only the earlier
22-object annotation, whereas the active relabel contains 46 objects.

## Frozen Prefix Construction

For each image and prefix depth, first select the first six or ten common
physical entities in the current geometry-sorted rollout's matched order.
Convert each artifact-local ground-truth box into one complete canonical row
and tokenize it once. The exact row token identifiers are reused in every arm.

Each depth has up to six distinct arms:

1. **Current geometry-sorted rollout order**: the selected entities in the
   order in which the current rollout produced them.
2. **Historical random relative order with fixed final two rows**: reorder only
   the earlier selected entities by their relative order in the historical
   random rollout, then append the same final two rows as arm 1.
3. **Reverse earlier rows with fixed final two rows**: reverse only the rows
   before the common suffix.
4. **Seeded shuffled earlier rows with fixed final two rows**: deterministically
   shuffle only the rows before the common suffix with seed 17.
5. **Reverse earlier rows with fixed final one row**: reverse all earlier rows
   while preserving only the final canonical row. Comparing this arm with arm
   3 tests whether an order effect survives one versus two subsequent rows.
6. **Adjacent swap before fixed final two rows**: exchange exactly the two rows
   immediately before the final two-row suffix. This is the mildest order
   intervention in the panel.

Every arm records its inversion count relative to the canonical order. The
inversion count is the number of entity pairs whose relative order differs
from arm 1. Frozen-prefix likelihood is intentionally not implemented in this
pilot; the case metadata records that this plausibility diagnostic is
unavailable. Therefore a result found only under large reversals or shuffles
must retain generic distribution shift as an alternative explanation.

If two predefined arms produce the exact same row order for one image and
depth, the duplicate arm and comparison are omitted rather than replaced with
an invented permutation. The case records the omitted arm, its duplicate, and
the reason `identical_row_order`.

Before inference, every comparison must verify:

- same image and prompt;
- same model and decoding policy;
- same row count;
- same physical covered set;
- same exact row-token multiset;
- the comparison-declared identical final one or two complete rows; and
- same paired sampling seed.

The source-rollout intersection is a case-selection rule, not an estimate of
the model's population recall. It preferentially retains easier objects and
must not be used to claim that either ordering policy is generally better.

## Stage 1: One-Row Screen

For every image, depth, and arm, generate:

- one greedy continuation; and
- four paired low-temperature continuations with seeds 101 through 104,
  temperature 0.4, and top-p 0.95.

Stop after one complete row, a terminal token, a malformed-row limit, or the
declared token limit.

Record separately:

- strict physical next owner under same-category intersection-over-union of at
  least 0.5;
- whether the owner was already in the frozen prefix;
- whether it is an uncovered entity in the full artifact-local ledger;
- the top same-category candidate, its intersection-over-union, and normalized
  center distance when strict matching fails;
- category and geometry agreement;
- terminal, malformed, unmatched, and ambiguous outcomes; and
- paired-seed owner switches relative to the current sorted-order arm.

A case is promoted to Stage 2 when at least one noncanonical arm produces:

- a different greedy physical owner;
- at least two physical-owner switches across the four paired seeds; or
- a material difference in covered-owner recurrence, terminal output, or
  malformed output that is not shared by all noncanonical arms.

If all noncanonical arms only become generally malformed or terminal, classify
the result as off-policy corruption rather than meaningful covered-set
behavior. If only the high-inversion reverse or shuffle arms change while the
adjacent swap does not, retain generic order-distribution damage as a serious
alternative explanation.

## Stage 2: Four-Row Value Test

Run only promoted image-depth comparisons. Use one greedy trajectory and eight
paired low-temperature seeds, 101 through 108. Seed once at the beginning of a
trajectory. After each complete generated row, append its exact generated token
identifiers to the prefix; never decode and retokenize it.

Over the next four rows, compare:

- number and identity of unique uncovered physical objects;
- recurrence of frozen-prefix entities;
- within-horizon duplicates;
- category/geometry disagreement;
- malformed or terminal position; and
- whether initially different owner paths reconverge to the same unique set.

The image and fixed prefix-set pair is the independent research case.
Permutations and seeds are repeated measurements, not independent images.

## Competing Outcomes

### Mostly benign path freedom

The next owner varies, but four-row unique uncovered sets, duplicate counts,
and stopping are similar. This argues against training the model to reproduce
one canonical next row or exact serialized trajectory.

### Harmful path dependence

Different earlier orders produce persistent differences in unique uncovered
objects, covered-owner recurrence, malformed output, or early stopping. This
creates a concrete target for a training intervention that rewards future
unique coverage rather than exact row order.

### Local recency only

The matched reverse intervention changes behavior with a final one-row suffix
but not with a final two-row suffix. This is evidence consistent with influence
being concentrated in the nearest one or two committed rows. Agreement in
both conditions is only a non-detection at this scope, not proof of order
invariance.

### Generic off-policy damage

All noncanonical orders lose valid-row probability and become malformed or
terminal without coherent owner changes. This means arbitrary shuffling is not
a valid model-state intervention at that depth.

### No effect but no activation evidence

The permutations agree, but earlier rows were never shown to be behaviorally
active. Record the result as uninformative about set-like memory; do not claim
order invariance.

## Matched Random-Order Replication

The newly launched random-order, description-first, pure-cross-entropy plus
token-type-gate run is live but has not produced a checkpoint. Once it does,
repeat the exact same frozen case specifications, prefixes, seeds, and horizon.
The useful between-checkpoint quantity is the difference in order sensitivity,
not the raw outcome of one preferred permutation. With one checkpoint per
training policy, any difference remains checkpoint-conditional rather than a
causal estimate of the training policy.

## Minimal Implementation Boundary

Reuse the existing experiment-local exact-prefix runner. Extend it only to:

- accept named arbitrary row orders and comparison metadata;
- validate the shared suffix and exact row multiset;
- generate one to four complete rows sequentially; and
- summarize next-owner and cumulative unique-set outcomes separately.

A small deterministic case builder may materialize the provisional common
cohort from the two rollout artifacts. Do not modify `src/inference`, the model
architecture, the training code, or the active OpenSpec.

## Stop Rule

Stop this unit after:

1. the four-image Stage-1 screen is complete;
2. every promoted case has a Stage-2 result or a documented runtime blocker;
3. all owner switches and unmatched boxes that affect the verdict have received
   enlarged-crop review; and
4. the matched-random replication is either executed from an available
   checkpoint or explicitly recorded as pending.

Do not expand to more images, permutations, layers, or training treatments
unless the four-image result leaves one specific conclusion-changing ambiguity.

The sorted-checkpoint Stage 1 and Stage 2 executions are complete. The matched
random-order replication is explicitly pending because the live training run
has not produced a checkpoint. See [results](results.md).

## Artifact Root

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-19-common-object-prefix-permutation-short-horizon/
```

Every immutable run must record source rollout digests, selected artifact-local
ground-truth indices, exact row and prefix hashes, model/checkpoint identity,
resolved inference configuration, seeds, per-row raw token identifiers,
parsing and physical-match evidence, and the source commit or dirty-diff state.
