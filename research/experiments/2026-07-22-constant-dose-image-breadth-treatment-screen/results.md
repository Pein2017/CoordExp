---
title: Constant-Dose Image-Breadth Treatment Results
description: A fixed-dose comparison showing strongly enriched recovery of supervised physical owners without a held-out advantage from spreading supervision across more images.
type: investigation
role: results
authority: non_normative_research
unit_id: 2026-07-22-constant-dose-image-breadth-treatment-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_bounded_no_heldout_image_breadth_advantage
architecture_promotion_status: not_promoted
updated: 2026-07-23
---

# Constant-Dose Image-Breadth Treatment Results

## Verdict

**Do not scale this unchanged positive-only complete-row treatment by adding
more images or more optimizer updates. At the frozen 992-event, 31-update dose,
greater image breadth did not show a held-out advantage or meet the predeclared
support criterion. Preserve the owner-directed credit signal, but change the
learning information in the next treatment.**

At the same 992-event and 31-update dose, spreading supervision over 496 images
instead of concentrating it in 162 nested images did not improve annotated
physical-owner coverage on the pre-admission held-out cohort. At the primary
repetition penalty 1.0, broad minus concentrated was negative in both training
seeds at Intersection over Union 0.50 and was zero or negative at 0.30. Every
paired image-level uncertainty interval included zero.

This is not evidence that the training signal was ignored. On gradient images,
owners explicitly selected by sampled-route treatment events were recovered
approximately 36 to 41 percent of the time when Source had missed them, versus
approximately 8 to 12 percent for non-selected missed owners. This strongly
enriched recovery is consistent with owner-specific uptake of the row-level
supervision, but it is not a same-owner untreated counterfactual. The failure is
that this strong local association did not become a reliable rule for expanding
the final owner set on images that supplied no gradients.

The strongest bounded interpretation is:

> Positive complete-row imitation can move a specific sampled-only physical
> owner into a clean greedy rollout, but broader image exposure alone does not
> make that local owner transition generalize into stable held-out set
> expansion. The remaining problem is not merely insufficient image variety;
> the objective still under-specifies how to gain a new owner while preserving
> the rest of the useful route.

## Executed Scope

- Source: geometry-sorted, description-first, pure-cross-entropy plus
  token-type gate checkpoint at step 4,887.
- Broad arm: 496 sampled-route events and 496 Source-preservation events from
  496 physical images.
- Concentrated arm: the same event counts and object-count-band by
  within-image-selection-rank distribution from 162 nested physical images.
- Optimization: 31 updates, effective event batch size 32, learning rate
  `1e-5`, language-tower Weight-Decomposed Low-Rank Adaptation only, two matched
  training seeds, shared frozen step 30.
- Primary decoding policy: `Source@B16`, meaning repetition penalty 1.0 and at
  most sixteen complete object rows or an earlier natural image-end.
- Primary transfer cohort: 124 globally complete cases from 128 held-out
  images. Four images were excluded because at least one of the compared panels
  could not provide an eligible bounded owner comparison.
- Owner matching: same-category deterministic one-to-one matching at
  Intersection over Union 0.30 and 0.50. Unmatched Common Objects in Context
  predictions remain review-needed and are not automatically hallucinations.

## Primary Held-Out Result at Repetition Penalty 1.0

| Intersection over Union | Arm | Seed 19 net owner change, gain / loss | Seed 23 net owner change, gain / loss | Mean change versus Source, 95-percent interval |
| --- | --- | ---: | ---: | ---: |
| 0.30 | Broad | +16, 55 / 39 | +15, 53 / 38 | +15.5, [-19.5, 45.0] |
| 0.30 | Concentrated | +16, 52 / 36 | +27, 57 / 30 | +21.5, [-4.0, 45.5125] |
| 0.50 | Broad | +13, 49 / 36 | +15, 48 / 33 | +14.0, [-17.0, 40.5] |
| 0.50 | Concentrated | +16, 51 / 35 | +25, 50 / 25 | +20.5, [-3.0, 43.5125] |

Broad minus concentrated was:

- Intersection over Union 0.30: seed-level values `0` and `-12`; mean `-6.0`
  with 95-percent interval `[-24.0, 10.5]`.
- Intersection over Union 0.50: seed-level values `-3` and `-10`; mean `-6.5`
  with 95-percent interval `[-24.0, 8.5]`.

Both treatments have positive held-out point estimates relative to Source, but
none of those uncertainty intervals excludes zero. The broad arm also fails
the predeclared requirement to exceed the concentrated arm. Almost all of the
positive point estimate comes from images with sixteen or more annotations;
the result is not uniform across density bands.

## What the Gradient Images Reveal

The broad arm does beat the concentrated arm on the 334 images that supplied
gradients only to the broad arm. This advantage is reproducible at
Intersection over Union 0.30 and concentrated in the eight-to-fifteen-object
band. It supports an image-conditioned learning component, not held-out
transfer.

The nominal broad-only cohort contains 334 images, with 325 and 326
treatment-complete cases for seeds 19 and 23. The nominal concentrated cohort
contains 162 images, with 161 and 160 treatment-complete cases for seeds 19 and
23. The percentages below use those treatment-complete denominators.

The stricter event-family attribution makes the mechanism clearer:

| Gradient scope | Treatment event family | Intersection over Union | Selected Source-missed owner recovery | Non-selected Source-missed owner recovery |
| --- | --- | ---: | ---: | ---: |
| Broad-only images, broad seed 19 / 23 | Sampled-route treatment | 0.30 | 39.8% / 39.3% | 9.8% / 9.9% |
| Broad-only images, broad seed 19 / 23 | Sampled-route treatment | 0.50 | 40.4% / 41.0% | 8.9% / 9.2% |
| Concentrated images, concentrated seed 19 / 23 | Sampled-route treatment | 0.30 | 35.9% / 40.0% | 12.2% / 10.0% |
| Concentrated images, concentrated seed 19 / 23 | Sampled-route treatment | 0.50 | 36.4% / 36.2% | 9.2% / 7.7% |

On these same matched ledgers, selected Source-preservation owners are retained
at approximately 95.4 to 96.9 percent at Intersection over Union 0.50. This is
useful but imperfect: preservation supervision reduces loss on explicitly
selected Source owners without constraining every other useful owner or the
final set as a whole.

The result therefore contains three components:

1. strongly enriched recovery of the sampled-route owner named by the training
   event;
2. broader behavior changes on the same gradient image;
3. owner exchange that remains insufficiently controlled on never-trained
   images.

It is no longer accurate to call the treatment merely a generic continuation
pulse, but it is also not a learned set-coverage algorithm.

## Repetition-Penalty Sensitivity

The first attempted repetition-penalty-1.10 panel is invalid. Its configuration
name requested 1.10, but the live collector still passed 1.0 to vLLM. Those
artifacts are preserved as failed provenance and are excluded from every
scientific comparison.

The corrected second panel records and validates repetition penalty 1.10 in
the runtime request, persisted batch artifacts, manifests, and owner ledgers.
The common comparison uses the same 124 complete held-out images across both
decoding policies and all arm-seed combinations.

| Intersection over Union | Surface | Owners at repetition penalty 1.0 | Owners at repetition penalty 1.10 | Direct change, 95-percent interval |
| --- | --- | ---: | ---: | ---: |
| 0.30 | Source | 667 | 654 | -13, [-46, 22] |
| 0.30 | Broad seed 19 / 23 | 683 / 682 | 682 / 671 | -1 / -11 |
| 0.30 | Concentrated seed 19 / 23 | 683 / 694 | 673 / 674 | -10 / -20 |
| 0.50 | Source | 620 | 604 | -16, [-47, 14] |
| 0.50 | Broad seed 19 / 23 | 633 / 635 | 634 / 623 | +1 / -12 |
| 0.50 | Concentrated seed 19 / 23 | 636 / 645 | 626 / 620 | -10 / -25 |

Every direct and mean-seed uncertainty interval includes zero. More
importantly, repetition penalty 1.10 exchanges many owner identities even when
the total count barely changes: depending on arm, seed, and threshold, roughly
65 to 100 owners are gained and a comparable number are lost relative to
repetition penalty 1.0. Three of the four treatment checkpoints have lower
absolute owner counts at each threshold; the only positive direct change is
`+1` for broad seed 19 at Intersection over Union 0.50.

The stronger treatment-minus-Source point estimates under repetition penalty
1.10 are therefore partly caused by a weaker Source baseline. Repetition
penalty 1.10 is not a clean owner-coverage improvement and cannot be treated as
only a duplicate suppressor. It changes the trajectory and the realized owner
set, and can even change the sign of broad minus concentrated for one seed.

## Human-Refined Twelve-Image Check

All twelve manually refined dense images remained eligible for every arm and
seed. No treatment panel produced a malformed row, dropped prediction,
pre-budget token-limit failure, or invalid pre-budget trajectory.

At Intersection over Union 0.30, broad seeds gained net `+6` and `+8` owners
relative to Source; concentrated seeds gained `+5` and `+3`. At 0.50, broad
gained `+2` and `+4`; concentrated gained `+2` and `+1`. Broad minus
concentrated was positive or tied, but every interval includes zero. This small
panel is useful safety evidence, not a population estimate or authorization to
scale.

## Decision

The image-breadth hypothesis is not supported. The intrinsic-owner-exchange
explanation remains the leading account, but the current evidence does not
prove that an architecture change is necessary. What has been established is
narrower and more actionable:

- complete-row supervision can target the intended physical owner;
- Source-preservation rows mostly retain the owners they explicitly name;
- neither signal defines the value of the complete final owner set;
- adding image diversity without changing that information does not establish
  held-out set expansion;
- changing repetition penalty substantially changes which owners appear and
  must remain a separately frozen research variable.

The next training unit should therefore change the information supplied at the
decision boundary. The preferred minimal treatment is a same-prefix paired
comparison in which a valid uncovered-owner row is scored above a harmful
alternative while Source owners outside that local branch are explicitly
protected in the evaluation. A compact covered-set or task-state carrier
should remain a later branch, activated only if the stronger loss cannot use
the native prefix state.

Do not run another positive-only breadth or epoch scale-up. Do not use terminal
suppression or repetition penalty as a substitute for increasing valid-owner
evidence.

## Evidence Handles

- Primary held-out comparison:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/heldout-owner-ledger-v1/frozen-step30-comparison.json`
- Gradient-cohort event-family attribution:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/gradient-owner-ledger-v1/treatment-owner-attribution-v2.json`
- Corrected repetition-penalty-1.10 held-out comparison:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/repetition-penalty-1p10-v2/heldout-owner-ledgers-v1/frozen-step30-comparison.json`
- Common repetition-penalty sensitivity comparison:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/repetition-penalty-1p10-v2/heldout-owner-ledgers-v1/repetition-penalty-sensitivity-comparison.json`
- Human-refined twelve-image comparison:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/human-refined-safety-v1/frozen-step30-comparison.json`

The event-family attribution artifact has SHA-256
`a60f979d382c7a9f5376dbadd28d310ea1273dddd8bceadad04179a14e9d2528`.
The repetition-penalty sensitivity artifact has SHA-256
`63e47cb8a775e3dbf9a0b157720efa759b7db0d4d2c3e83e09ed46201be72dea`.

## Limits

- Only two training seeds were run. Image-level bootstrap intervals do not
  estimate between-training-run variance.
- Aggregate owner counts are density-weighted; very-dense images dominate the
  held-out point estimates.
- Four held-out images are excluded from the eight-ledger common complete-case
  comparison rather than counted as empty failures.
- Common Objects in Context labels remain incomplete. The owner ledger measures
  annotated-owner matching, not every real physical entity.
- Intersection over Union 0.30 and 0.50 mix discovery and geometry. They do not
  replace the separate entity-existence and box-extent review taxonomy.
- The evidence rejects an unchanged scale-up; it does not prove that every
  positive-row objective, every native-prefix treatment, or every explicit
  task-state mechanism must fail.
