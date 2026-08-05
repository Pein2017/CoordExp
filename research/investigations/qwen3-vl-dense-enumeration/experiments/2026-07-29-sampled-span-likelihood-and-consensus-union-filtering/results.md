---
title: Sampled Span-Likelihood and Consensus Filtering Results
description: Coordinate likelihood, not trajectory consensus, is what can reject badly grounded union clusters while retaining the owners only sampling recovers; consensus is confounded with exactly those owners and rejects nothing at the pre-registered operating point.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-07-29-sampled-span-likelihood-and-consensus-union-filtering
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-29
---

# Sampled Span-Likelihood and Consensus Filtering Results

Protocol, terminology, and the frozen stop rule are owned by
[unit.md](unit.md). Every threshold and label semantic below was fixed before
any likelihood was computed.

## Decision

The pre-registered stop rule is evaluated per checkpoint, at leave-one-image-out
out-of-sample retention of at least 95% of greedy-missed union-recovered owners:

| Checkpoint | Best family | Out-of-sample rejection | Needed to pass | Verdict |
|---|---|---:|---:|---|
| Sorted | support + likelihood | **61/197 = 31.0%** | 40 | passes |
| Random | support + likelihood | **28/131 = 21.4%** | 27 | passes by one cluster |
| Permutation | coordinate only | **21/138 = 15.2%** | 28 | **fails → confident binding failure** |

For Sorted and Random the outcome is **branch 1, likelihood-filterable tail**,
and explicitly *not* branch 2. For Permutation the pre-registered rule selects
**branch 3** and threshold tuning stops there.

The result is checkpoint-specific. There is no single promoted threshold.

## The mechanism that decides it

Stage 1 measured trajectory support as the strongest single separator of
catastrophic false positives (area under the curve 0.821–0.894, above every
likelihood feature). Stage 2 reverses that reading at the operating point that
actually matters, and the reason is a confound the aggregate statistic hides:

| Checkpoint | support of greedy-missed recovered owners | support of other true positives | support of catastrophic false positives |
|---|---|---|---|
| Sorted | median 5, 5th pct 1 | median 16 | median 1 |
| Random | median 2, 5th pct 1 | median 10 | median 1 |
| Permutation | median 3, 5th pct 1 | median 11 | median 1 |

The owners that only sampling recovers sit at **the same low support as the
catastrophic tail**. Support's high area under the curve came almost entirely
from the true positives greedy had already found — the ones a union operation
does not need. A support threshold that retains 95% of the greedy-missed
recoveries is forced down to support ≥ 1 and therefore rejects **nothing**.

Out-of-sample rejection for support-only reads 0.000 (Random), 0.000
(Permutation), and 0.041 (Sorted). The Sorted 0.041 is not discriminative
rejection and should not be read as one. All 8 of its rejected clusters have
support exactly 1 and come from a single leave-one-image-out fold whose
threshold happened to clear the entire support-1 block; in that same fold the
threshold also destroyed **4 of the 6** greedy-missed recovered owners in the
held-out image, and those 4 are the whole of Sorted's owner loss at this
operating point. Support cannot separate a catastrophic support-1 cluster from a
recovered-owner support-1 cluster, because by construction they carry the same
value. The correct reading is that support-only rejects nothing on any real
basis on all three checkpoints.

Coordinate likelihood is not confounded this way. It rejects badly grounded
clusters while retaining low-support recoveries, which is why every family that
passes contains a likelihood term and why support-only fails on all three
checkpoints.

This is the load-bearing consequence of the round-2 decision to make
greedy-missed union-recovered owners the primary retention denominator rather
than all union true positives. Under the all-true-positive denominator the
support families would have looked strong and the conclusion would have
inverted.

## Observed

1. Coordinate likelihood separates catastrophic false positives within every
   checkpoint (area under the curve 0.758–0.799 at the medoid).
2. The pre-registered grammar control passes decisively. Schema-wrapper
   likelihood is saturated — median −0.000 in all three groups — and its area
   under the curve is at or below chance for Random and Permutation. The
   separation is grounding confidence, not grammar confidence.
3. Description likelihood is **inverted** for Random (0.417) and Permutation
   (0.409): badly grounded clusters are named more confidently than correct
   ones. Sorted does not show this (0.709). Adding description to coordinates
   hurts every checkpoint, and on Random no target reaches 95% retention at all.
4. Support alone rejects nothing at the operating point on any of the three
   checkpoints, for the confound reason above; Sorted's apparent 4.1% is one
   fold clearing the support-1 block indiscriminately, at a cost of 4 of 6
   recovered owners in that image.
5. Low likelihood does not accumulate later in a trajectory. Median coordinate
   likelihood by position third is −3.399 / −3.518 / −3.185 (Random) and
   −2.678 / −2.942 / −2.751 (Sorted). The "bad autoregressive basin" reading is
   unsupported.
6. Fold-wise thresholds are stable, not a single lucky operating point. Across
   the 12 leave-one-image-out folds the selected threshold for
   support + likelihood spans 0.118–0.219 (Sorted) and 0.188–0.257 (Random) in
   training-percentile units.

## Supported

Internal coordinate-token confidence carries real, non-grammatical information
about whether a sampled detection row is correctly grounded, and it is
complementary to cross-trajectory consensus rather than redundant with it. On
Sorted the combination retains 95.5% of greedy-missed recovered owners while
rejecting 31% of the catastrophic tail out of sample.

## Ruled out

- **Grammar confidence as the explanation.** The wrapper control is saturated
  and carries no signal; it cannot be producing the separation.
- **Consensus as the controlling signal (branch 2).** Support is confounded with
  the retention target and rejects nothing at the operating point.
- **Description confidence as a grounding proxy.** It is inverted on two of
  three checkpoints.
- **Late-trajectory degradation** as the source of bad rows.

## Unresolved

- **Random's margin is one cluster** (28 rejected against 27 needed), on a
  denominator of only 44 recovered owners, with a retention interval of
  [0.886, 1.000]. Treat Random as "not decided against" rather than as a pass.
- **The "best family" is selected across five candidates**, which the
  pre-registration did not correct for. Taking the maximum over five families
  inflates the reported rejection, and Random sits one cluster above the bar, so
  its pass is within the selection noise. Sorted's 31.0% clears the bar by 21
  clusters and does not depend on which family is picked — coordinate-only alone
  reaches 28.9%. Only the Sorted conclusion is robust to this.
- **Why Permutation fails** while Sorted and Random pass is not explained. Its
  catastrophic tail is not separable at the required retention by any family
  tried.
- **Spatial dispersion overfits.** Adding it moves Permutation from 26.1%
  in-sample to 6.5% out of sample and Sorted from 10.7% to 3.6%. It is not a
  usable third term on 12 images.
- Absolute log-likelihoods remain differently calibrated across checkpoints; no
  cross-checkpoint threshold transfer was attempted.

## Not claimed

No production filter, inference policy, or training change is promoted. Twelve
images with 346 owners is a small panel, and the retention denominators (88 / 44
/ 60) are small enough that the leave-one-image-out estimates carry real
variance. A single deployable threshold is explicitly **not** established: the
verdict differs by checkpoint, and Permutation fails outright.

## Follow-on

[2026-07-29 Ranking Quality Versus Usable Rejection](../2026-07-29-ranking-quality-versus-usable-rejection/results.md)
explains the apparent contradiction recorded below — support ranks best and
rejects nothing — and reaches a conclusion that constrains how this unit's
Stage 1 should be read. The area under the ROC curve reported in Stage 1 is
**not a valid ranking of filter candidates**: it is scored partly on true
positives greedy already found, whose retention is free, and support is inflated
about twice as much as coordinate likelihood by that effect. Over half of every
checkpoint's catastrophic tail also sits in a discrete atom at support 1 that no
95%-retention threshold can cut. Stage 1's ordering should be read as a
descriptive separation measurement, not as evidence about which family to
deploy; this unit's verdict was decided by Stage 2 and is unaffected.

[2026-07-29 Likelihood Filter Robustness at Equal Owner Count and Panel
Annotation Validity](../2026-07-29-likelihood-filter-robustness-and-panel-annotation-validity/results.md)
answers the first discriminator below and closes a validity threat this unit did
not name. Equalizing every retention denominator to 44 owners leaves Sorted
passing in 150/200 draws and Permutation failing in 199/200, so the checkpoint
split is not a denominator artifact. It also establishes that 170 of this
panel's 346 ground-truth objects are human-added, so the catastrophic
false-positive count here is not inflated by COCO omissions — a confound this
unit should have named and did not.

## Next discriminator

Whether the Sorted result survives a larger panel, and whether Permutation's
failure is a property of its training arm or of its smaller and lower-support
recovery set. A useful cheap probe first: does coordinate likelihood still
separate when the retention denominator is held fixed across checkpoints by
subsampling to a common recovered-owner count?

## Evidence handles

Artifact root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-29-three-checkpoint-human-refined12-max3084/likelihood-mining-v1/`

- `stage0-gates-receipt.json` — all four contract gates, 576 sequences, 9,340 spans
- `span-likelihood.jsonl` — 9,340 rows of raw-model span likelihood
- `cluster-confidence.json`, `span-labels.jsonl` — 1,852 clusters, dual labels
- `stage1-separation.json` — separation and grammar control
- `stage2-retention.json` — leave-one-image-out retention and rejection sweep
- `replay-receipt.json`, `cluster-confidence-receipt.json`

Verification not performed by the implementing lane: chosen-token logprobs
recomputed in a fresh process with the canonical
`teacher_forced_chosen_token_logprobs` for 38 spans across three images and
seeds agreed at **exactly 0.0** absolute difference; the cluster reconstruction
reproduces the pre-registration baseline in every cell; the narrowed
model-identity gate was confirmed exhaustive by perturbing four separate config
fields.
