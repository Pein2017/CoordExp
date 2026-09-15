---
title: Human-Audited Rare-Object Trajectory Genealogy Manual-Review Results
description: Verified qualitative geometry evidence from 119 audited bagging candidates, with the entity-ledger gate held and trajectory replay redirected to a coordinate-coherence discriminator.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_manual_review_only
unit_id: 2026-07-16-human-audited-rare-object-trajectory-genealogy
topic: qwen3-vl-dense-enumeration
status: redirected
evidence_status: partial_verified
updated: 2026-07-16
---

# Human-Audited Rare-Object Trajectory Genealogy Manual-Review Results

## Scope and Evidence Identity

The reviewer audited all `119` unmatched Full-Image K-Rollout Independent
Bagging (`FULL_BAG_K`) candidates from the frozen five-image purposive cohort.
All candidates received an approve verdict. This means that the reviewer found
a real Common Objects in Context 80-category (`COCO-80`) entity supporting
each proposal; it does **not** mean that every phrase or box was correct.

The frozen export is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-16-human-audited-rare-object-trajectory-genealogy/
human-review-candidate-ledger-20260716a/
human-review-candidate-ledger-20260716a.review.json
```

Its SHA-256 digest is:

```text
e22255f330996e2ba7c3a1ea5827ed1e20a3445e91ed1b9d4c1b4839158d3247
```

The export contains one empty free-text comment, not an empty review. Two
approved candidates omit the structured semantic and geometry statuses but
retain informative comments. These records remain valid qualitative evidence.

The annotation gate nevertheless reports:

```text
119 approved
0 approved with a valid entity reference
119 missing entity references
2 missing structured statuses
annotation gate passed: false
```

Therefore this export is **not** a frozen physical-entity ledger. It must not
be used for unique-entity frequency, candidate-to-owner consolidation, or
trajectory-level recall. The originally planned sibling-branch and causal-row
replay waves were not entered.

## Structured Review Counts

The cohort was deliberately selected from unmatched post-merge candidates, so
the following counts describe this failure-enriched panel rather than model
prevalence.

| Image identifier | Candidates | Semantic exact | Geometry acceptable | Localization error | Multiple entities | Fragment | Ambiguous geometry |
|---|---:|---:|---:|---:|---:|---:|---:|
| `12576` | 15 | 12 | 7 | 6 | 0 | 1 | 1 |
| `19432` | 85 | 81 | 13 | 28 | 34 | 0 | 8 |
| `7816` | 4 | 3 | 0 | 1 | 2 | 0 | 1 |
| `9400` | 15 | 12 | 8 | 3 | 2 | 0 | 2 |
| **Total** | **119** | **108** | **28** | **38** | **38** | **1** | **12** |

The main signal is a large semantic-versus-geometry separation: the phrase is
usually compatible with a real entity, while the four boundaries frequently
describe a part, several neighboring instances, or a spatial compromise.

## Coordinate Semantics and Conservative Boundary Recode

The model's box order is:

```text
x1, y1, x2, y2
```

where `x1` is the left horizontal boundary, `y1` the top vertical boundary,
`x2` the right horizontal boundary, and `y2` the bottom vertical boundary.

A conservative comment-level recode retained an ambiguity bucket whenever the
review did not identify one physical owner or one directional error. It is not
a replacement annotation ledger.

| Boundary | Correct | Inward or too small | Outward or too large | Wrong instance or ambiguous | Not inferable |
|---|---:|---:|---:|---:|---:|
| `x1`, left | 20 | 11 | 8 | 56 | 24 |
| `y1`, top | 38 | 15 | 3 | 29 | 34 |
| `x2`, right | 24 | 9 | 10 | 56 | 20 |
| `y2`, bottom | 21 | 35 | 4 | 29 | 30 |

The strongest directional signal is bottom under-coverage. Among the `39`
directionally codable `y2` errors, `35` end too early and only `4` extend too
far. This signal is dominated by image `19432`, where visible chair backs and
partially occluded bodies create a visible-extent versus complete-extent
ambiguity; it must not be generalized as a population estimate.

## Representative Structured Failures

The comments reveal repeated, coherent patterns rather than uniformly noisy
rectangles:

- image `12576`: fork and spoon predictions can cover only the discriminative
  utensil head even though more of the utensil is visible;
- image `9400`: several laptop predictions localize a screen but omit the
  keyboard or body, while other boxes merge two adjacent laptops;
- image `19432`, request prefix `c994b2...`: two consecutive chair boxes share
  nearly the same vertical extent, but the second right boundary expands into
  the adjacent chair;
- image `19432`, request prefix `c9e52e...`: related boxes become progressively
  thinner, sometimes preserving a plausible top or horizontal owner while the
  bottom boundary collapses toward a chair back;
- image `19432`, request prefix `ff5660...`: one row reconstructs a coherent
  occluded chair including the inferred lower body, while later rows combine a
  horizontally plausible chair with the vertical extent of two rows.

These examples argue against one universal explanation. The same model can
produce visible-part boxes, plausible complete-object boxes, adjacent-instance
unions, and axis-wise owner mixtures from closely related states.

## What Is Supported

1. **Bagging support is not equivalent to hallucination.** Within this
   purposive unmatched cohort, every reviewed candidate had some real
   `COCO-80` entity support. This does not estimate population hallucination.
2. **Semantic retrieval is substantially more stable than complete geometry.**
   Correct category evidence can coexist with poor or multi-owner boundaries.
3. **Coordinate errors are structured and phase-dependent.** The observations
   are compatible with sequential boundary composition, latent extent modes,
   and object-part shortcuts; they are not well summarized by one scalar
   Intersection over Union score.
4. **A pure object-part-only account is too strong.** Several chair rows infer
   coherent occluded full extent, proving that complete-object completion is
   reachable in at least selected trajectories.
5. **A stable instance-owner account is not yet proven.** Many chair boxes
   cover several instances or lie between instances, and the missing entity
   references prevent owner-level consolidation.

## What Is Not Supported

- no unique-entity recall, candidate frequency, or population precision;
- no conclusion that the four boundaries are statistically independent;
- no conclusion that one latent instance owner controls all four boundaries;
- no conclusion that late rollout row index alone causes geometry decay;
- no trajectory-unlock, covered-set, or commit mechanism claim from this
  manual-review export;
- no architecture or training promotion.

Within the reviewed chair candidates, acceptable-geometry rates did not
decrease monotonically with source span index: spans `0` through `5` yielded
`6/42`, spans `6` through `11` yielded `5/29`, and spans `12+` yielded `2/14`.
This bounded check does not support a simple later-row degradation account.

## Redirection Decision

The original unit asked which trajectory transitions retrieve rare physical
entities. The review instead exposed a sharper prerequisite: before treating a
row as an object commit, determine whether its four autoregressive boundaries
form one coherent object hypothesis or a composition of independently
competitive boundaries, visible-part extent, and adjacent-instance support.

The successor is a fixed-prefix complete-box factorial with two geometry
references:

1. the primary `COCO`-style visible or modal detection extent;
2. a secondary plausible complete physical or amodal extent used only as a
   mechanism probe.

It must compare coherent visible and complete boxes against adjacent-owner,
multi-object, and boundary-hybrid controls, then release later coordinates
after progressively forcing earlier coordinates. The trajectory-genealogy
branch remains held until an entity-reference ledger is explicitly completed.

## Verification

- all `119` exported candidate records were parsed;
- headline and structured counts above were independently recounted from the
  frozen JSON export;
- every free-text comment was inspected in candidate order, using enlarged
  image crops where the visible evidence was too small at full-image scale;
- the four boundary-direction buckets were conservatively reconciled after an
  independent audit corrected overconfident owner assignments;
- the review export digest matches the durable artifact copy.
