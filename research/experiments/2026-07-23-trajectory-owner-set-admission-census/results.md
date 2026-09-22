---
title: Root-State Trajectory Owner-Set Admission Census Results
description: A complete training-split census showing that only eight images satisfy the frozen multiple-positive strict owner-set predicate, so the proposed 256-image training screen is not promoted.
type: investigation
role: results
authority: non_normative_research
unit_id: 2026-07-23-trajectory-owner-set-admission-census
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_training_promotion_rejected
architecture_promotion_status: not_promoted
updated: 2026-07-23
---

# Root-State Trajectory Owner-Set Admission Census Results

## Verdict

**Stop the proposed 256-image training promotion. Only 8 of the 2,004
Source-eligible training images satisfy the frozen primary natural-alias
predicate, versus 256 required. Do not launch the grouped set-level training
screen, and do not replace the missing supervision with a unique-row or
unique-trajectory fallback.**

The result answers the unit's frozen feasibility question. It does not choose
one causal explanation for the shortfall. Automatic attribution censors 1,626
images, while the 378 fully adjudicable images also contain few admissible
edges and few same-owner-set aliases with different valid first owners. The
current evidence therefore supports both an attribution-coverage limitation
and structural scarcity in the selected fully adjudicable subset. It does not
support extrapolating either one as the sole population-wide cause.

No training, model inference, gradient-bearing work, or graphics-processing
unit execution was performed for this census.

## Executed Scope

- Population: the exact 2,004 Source-eligible members of the frozen 2,048-image
  training split.
- Candidate group: one exact `Source@B16` continuation plus sampled indices 0
  through 15 for each image, for 17 candidates per image.
- Primary predicate: one nondominated exact physical-owner-set class with at
  least two unique projected-token serializations, at least two valid first
  owners, and membership as the higher side of an admissible strict owner-set
  inclusion edge.
- Safety: universal alias safety and lower-first-owner preservation.
- Whole-candidate exclusion: a pre-budget parser failure, unresolved or
  ambiguous owner, or any geometry-untrusted row excludes the candidate.
- Blindness boundary: development and held-out identifiers were filtered
  before route-semantic inspection. No development or held-out route summary
  was produced.

## Observed

The completed census contains 2,004 unique image records and 34,068 candidates:

| Quantity | Count |
| --- | ---: |
| Primary-admitted images | **8** |
| Required images | **256** |
| Eligible candidates | 9,427 |
| Excluded candidates | 24,641 |
| Exact projected-token duplicates | 328 |
| Owner-set outcome classes | 886 |
| Strict owner-set inclusion edges | 134 |
| Admission-eligible strict edges | 105 |
| First-owner-orphaned strict edges | 29 |
| Images with at least two eligible candidates | 709 |
| Fully adjudicable images | 378 |
| Censored images | 1,626 |

The eight primary-admitted training image identifiers are:

```text
2434 16796 69532 174740 207431 256151 294679 569960
```

Images `174740`, `207431`, `256151`, and `569960` are fully adjudicable. The
other four contain excluded candidates but still satisfy the primary predicate
using their eligible candidates.

The diagnostic image-level frontier view admits ten images. Its two additional
images, `89773` and `358484`, assemble first-owner diversity across separate
incomparable frontier classes. They do not contain the required diversity
within one exact owner-set class and therefore do not pass the frozen primary
predicate.

## Failure Structure

The primary predicate is the intersection of an admissible strict edge and a
same-class, multiple-first-owner natural alias. The observed image funnel is:

| Population | Neither | Edge only | Alias only | Both and admitted | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fully adjudicable | 355 | 12 | 7 | 4 | 378 |
| Censored with at least two eligible candidates | 259 | 56 | 12 | 4 | 331 |
| All images with at least two eligible candidates | 614 | 68 | 19 | 8 | 709 |

Within the 378 fully adjudicable images, 362 lack an admissible edge and 367
lack a same-class alias with multiple valid first owners. Multiple exact
serializations are common; different valid first owners within the same
frontier class are not. The fully adjudicable subset therefore contains real
structural scarcity under the frozen exploration panel and predicate.

Censoring is simultaneously large and overlapping:

- all 1,626 censored images contain at least one geometry-untrusted candidate;
- 1,455 also contain an unknown row;
- 272 contain a pre-budget parser or projection failure; and
- 242 contain ambiguous ownership.

At candidate level, 24,640 of 24,641 excluded candidates are geometry
untrusted, and 21,049 contain unknown rows. These labels overlap and are partly
definitional; they are not independent causal counts. An unmatched official
Common Objects in Context prediction remains unresolved rather than an
automatic hallucination.

## Supported

- The current automatically attributable trajectory panel does not supply the
  256 groups required by the proposed training screen.
- Strict owner-set inclusion and naturally different first-owner
  serializations exist, but their frozen intersection occurs in only eight
  images.
- The fully adjudicable subset has a structural bottleneck on both strict
  owner-set edges and same-class first-owner diversity.
- The censored population is too large to treat the fully adjudicable subset as
  a population-wide structural estimate.
- Training promotion is false even though the broader causal diagnosis remains
  unresolved.

## Ruled Out

- Launching the proposed 256-image grouped set-level screen from this census.
- Counting the two image-level-only frontier cases as primary supervision.
- Treating incomparable owner exchanges, singleton maximal serializations, or
  excluded candidates as valid positive preference pairs.
- Falling back to one canonical positive row or trajectory merely to fill the
  training cohort.
- Reopening the admission predicate after seeing the shortfall.

## Unresolved

- Whether train-only adjudication or improved matcher and trusted-geometry
  coverage would recover enough censored images.
- Whether the censored population has the same structural scarcity as the
  fully adjudicable subset.
- Whether broader or targeted exploration would create more strict inclusion
  edges and same-class first-owner aliases under the unchanged predicate.
- Which successor research method should resolve that causal boundary.

The numerical requirement for a future, separately frozen causal-resolution
study is explicit. With four fully adjudicable admissions fixed, the censored
population would need at least 252 admissions out of 1,626 images, or 15.50
percent, to make a 256-image screen feasible. Equivalently, it must recover at
least 248 additional admissions among the 1,622 currently nonpassing censored
images. This is a falsifiable threshold, not an estimate supplied by the
current census.

## Not Claimed

- No population-wide causal attribution to censoring or structural scarcity.
- No conclusion about annotation exhaustiveness, hallucination prevalence, or
  development and held-out route behavior.
- No trajectory-score, matched-control, loss, architecture, task-state carrier,
  or training recipe was selected.
- No claim that a future cohort of at least 256 would itself authorize
  training; it would only permit a separate training unit and review gate.
- No claim about clean-greedy transfer, model improvement, or final-set
  expansion.

## Independent Review Gate

Three blind advanced-model lanes independently reconstructed the frozen
predicate and production artifact. They agreed that:

1. the artifact is mechanically valid and heldout-blind;
2. `8 < 256` is a decision-grade stop result with no conclusion-critical
   ambiguity;
3. structural scarcity is established only for the fully adjudicable subset;
   and
4. choosing a causal explanation or successor method requires a separately
   frozen research question.

The reviews are advisory provenance. The production artifact, frozen unit, and
reproducible counts remain the scientific evidence.

## Evidence Handles

- Production root:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-23-trajectory-owner-set-admission-census/production-v1`
- Image census SHA-256:
  `c375e09df621714da82118a2cdf3d80205fd7429424b1a853f38ec60e68e1805`
- Summary SHA-256:
  `2cc526310e8bca2de11f2d9d7c2e42511a017a882455dd1ebdb671b0c0df4514`
- Source-snapshot manifest SHA-256:
  `8f6c660d643aef4c68fc551c93f19caf701f1def97d73a00fd4a1587c2989d8b`
- Receipt SHA-256:
  `d322335c5e92f3fd4860b9524b8d709a3d01fe9e1cbc625dfa1d4bf002d12fe2`
- Forward and reversed record semantic hash:
  `957ae42368f6d04c1f21a51f7f00a80770b90d44d220ec1b7d8c490f3a299f44`
- Forward and reversed summary semantic hash:
  `52fbc990371d3264eebcc4c54be34f8af0eab7c48c0da4c75ee0ceb2f8c1667d`
- Sampled manifest-set SHA-256:
  `28304194b0b5e2aa7a718b2ee080253f616a8d9cf8fde8653e335589b3179b60`
- Source manifest-set SHA-256:
  `4e979f920798dcd338c0620500d36b007f90404bb4a2852014702de93b9f521e`
- Execution-model identity SHA-256:
  `8e0cc5c679df56cd55f965b56a4e9f4a661c0277bc93248100a0d143190918ac`
- Tokenizer identity SHA-256:
  `878bc75fd27e4668788cb864bf93dee4d90dbbbbceec0f2037ba8eab75247fd1`

The published artifact has terminal status `completed`, zero failures, all
integrity checks true, identical forward and reversed semantic hashes, files
mode `0444`, directories mode `0555`, and no sibling staging residue.
