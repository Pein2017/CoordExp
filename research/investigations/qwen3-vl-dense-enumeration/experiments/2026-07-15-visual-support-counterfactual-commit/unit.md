---
title: Visual-Support Counterfactual Commit Test
description: Fixed-encoding test of whether one phrase-and-geometry-compatible prefix transition depends on localized visual support.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-15-visual-support-counterfactual-commit
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Visual-Support Counterfactual Commit Test

## Question

At the exact Common Objects in Context image `12576` **prefix state 56
(`P56`)**, the preceding factorial found that appending the coherent `cup`
description plus left-cup geometry advances the first free row to the right cup
in all nine requests. Does that transition require localized encoded visual
support for the left cup, or can the same fixed text prefix drive it without a
detectable contribution from that support?

The test changes encoded visual features, not pixels, tokens, positions, row
text, sequence length, or decode policy. It is intentionally one-sided: a
selective target-support effect can support visual revalidation, while a null
cannot prove that the transition is purely textual because Qwen3
Vision-Language (`Qwen3-VL`) visual features are globally contextualized.

## Competing Hypotheses

### Visual transaction-consistency gate

The model accepts the appended phrase-and-geometry row as a committed object
event only when compatible visual evidence remains available at the referenced
location. Replacing left-cup support should selectively weaken the right-cup
successor transition.

### Textual geometry-sorted serialization transition

The coherent row is accepted from its textual phrase, coordinate tokens, and
geometry-sorted prefix state. Replacing local left-cup visual support should not
detectably change the right-cup successor.

### Generic feature-perturbation sensitivity

Any sufficiently structured local feature substitution changes the language
model basin. Target and unrelated-region substitutions should then damage the
transition similarly, which is inconclusive for visual object commitment.

## Frozen Three-Condition Panel

All three primary conditions use the exact same `P56` recipient token prefix,
the exact same coherent nine-token left-cup row, and the same clean recipient
image encoding outside the selected region.

1. **Clean Feature Replay**: replay cloned clean recipient features unchanged.
2. **Target-Support Donor Substitution**: replace left-cup support with
   same-position features from one frozen, same-grid donor image.
3. **Equal-Area Unrelated-Support Donor Substitution**: apply the identical
   substitution to an equal-shaped recipient region that does not overlap the
   left cup, right cup, target pizza, or another immediate successor candidate.

The target support begins as the merged visual-token cells intersecting the
left-cup box for annotation `coco-ann:678023`, expanded by one merged-token
halo. On the executed `36`-row by `27`-column merged grid, that base rectangle
is rows `10` through `20` and columns `6` through `13`, for `88` cells. The
clean upper-left control base rectangle is rows `0` through `10` and columns
`0` through `7`. Those translated rectangles share two boundary-adjacent halo
cells even though the raw object box and control are disjoint.

The executed masks therefore use deterministic symmetric overlap pruning. The
two physical overlap cells occupy control-relative positions `(10,6)` and
`(10,7)`. Those two relative positions are removed from both translated
rectangles. This removes the overlap from the control, preserves the same
relative mask shape in the target, and leaves `86` strictly disjoint cells per
condition. The corresponding removed target cells are bottom-right halo cells,
not raw left-cup support. The receipt must preserve the base masks, removed
relative positions, final indices, and disjointness check rather than
describing the intervention only by a bounding box.

The frozen donor is Common Objects in Context validation image `17436`, with
the same `864` by `1,152` dimensions and visual grid. Manual review found no
cup at either substituted location; the scene contains a church, garden,
bench, and person. Its monochrome domain difference is shared by target and
control substitutions and remains an explicit generic-perturbation risk.

For every selected index and every visual channel, the operator is:

```text
recipient feature at selected spatial index
  <- donor feature at the same spatial index
```

It is applied to the primary image embeddings and to all three DeepStack visual
feature streams. Zero vectors, distant same-image copying, changed image-token
counts, and a primary-only headline condition are outside this unit.

## Primary Observation

Run one greedy request and eight paired temperature-`0.4` requests per
condition. Sampling uses top-p nucleus threshold `0.95`, repetition penalty
`1.0`, identical seeds, and the accepted request-scoped sampler.

For each condition `c`, estimate the fraction:

```text
R(c) = requests whose first valid free row is the right cup
```

The primary paired contrast is:

```text
Visual-Support Commit Effect
  = R(equal-area unrelated-support substitution)
    - R(target-support substitution)
```

Every non-right-cup result is classified as left-cup revisit, target-pizza
fallback, another supported object, terminal action, or invalid output. A
selective shift to a valid alternative object is stronger evidence than a
shift to termination or malformed generation.

## Interpretation Rules

Evidence for a visually grounded commit requires all of the following:

- clean replay and the equal-area control retain the right cup in at least
  seven of eight sampled requests;
- target substitution loses the right cup in at least six of eight sampled
  requests;
- no paired seed shows the reverse selective pattern;
- the target-substitution failures predominantly remain valid object actions.

If target and unrelated-region substitutions damage the transition similarly,
close the unit as generic feature-perturbation sensitivity. Do not tune the
replacement magnitude within this unit.

If all three conditions retain the right cup, conclude only that the
transition does not depend detectably on the selected local post-vision support
under this operator. Do not promote a purely textual-transducer claim.

## Conclusion-Critical Invariants

Before causal interpretation, the run must establish:

- exact recipient-prefix token hash and exact coherent-row token hash;
- identical model, adapter, tokenizer, prompt, decode policy, paired seeds,
  runtime, parser, and maximum generation budget;
- identical image-token count, visual grid, positions, attention mask, dtype,
  and device;
- identical donor and recipient grid shape in the primary and all DeepStack
  feature streams;
- byte-identical recipient complement outside the selected indices;
- exact same-position substitution at every recorded selected index;
- intervention before the complete prefix, including the coherent row, is
  prefetched;
- hook restoration after every condition.

The mandatory no-op gate compares the standard clean path with the enabled
feature-replay path using unchanged cloned tensors. Greedy and every paired
sampled request must preserve generated token identifiers and canonical
float32 score traces exactly. Any no-op mismatch stops the panel.

Intervention tensors remain in the model's executed dtype to avoid changing
the forward semantics. Feature differences, norms, and recorded score traces
are computed or canonicalized in float32 where needed for stable comparison.

## Execution Outline

1. Reuse the accepted model loader, exact composed-row builder, request-scoped
   sampler, parser, first-action classifier, and compact receipt conventions.
2. Encode the clean recipient and frozen donor once. Cache and clone the
   primary and DeepStack outputs rather than rerunning the visual tower across
   conditions.
3. Run one greedy standard-clean versus cloned-clean no-op check and one greedy
   request for each target/control condition. Verify all hook, mask, grid,
   complement, dtype, and device receipts; stop on any mismatch.
4. If the greedy gate passes, run the eight-seed no-op parity check and full
   three-condition paired panel in one second invocation. Close on the
   first-action distribution. Do not add a second image or training within
   this unit.

## Evidence Identity

- Worktree: `/data/CoordExp/.worktrees/research-probes`.
- Recipient image: Common Objects in Context validation image `12576`.
- Frozen donor image: Common Objects in Context validation image `17436`,
  `/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/val2017/000000017436.jpg`.
- Recipient source record:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl`.
- Inference configuration:
  `/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml`.
- Checkpoint: step `4,887` resolved by that configuration.
- Recipient state: exact `P56` token identifiers plus the verified coherent
  nine-token left-cup row from the preceding causal-replay and factorial units.

## Artifact Handle

The logical artifact root is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-visual-support-counterfactual-commit/<immutable-run-id>/
```

It contains one compact receipt per condition, raw output bundles, selected
target and control indices, donor identity, primary and DeepStack feature
hashes, no-op parity evidence, hook counters, and one cross-condition summary.

## Non-Goals and Stop Rules

- no training, OpenSpec change, architecture promotion, attention analysis,
  residual replay, population metric, or large validation sweep;
- no claim that a local feature substitution erases all distributed evidence;
- no use of a donor or control region that has not passed manual visual review;
- stop if no-op parity fails, DeepStack index semantics cannot be aligned, the
  control overlaps an active candidate, the complement changes, or target and
  control both collapse to invalid or terminal outputs;
- a selective positive result still requires one independent exact-prefix
  replication before any training scaffold is proposed.
