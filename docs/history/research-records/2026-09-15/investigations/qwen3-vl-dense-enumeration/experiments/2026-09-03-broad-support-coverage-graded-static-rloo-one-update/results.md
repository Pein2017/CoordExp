---
title: Broad-Support Coverage-Graded Static RLOO One-Update Results
description: One broad-support shared-DoRA update improves the registered train248 owner-count vector over both C and the selected-eight update.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-09-03-broad-support-coverage-graded-static-rloo-one-update
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-03
---

# Broad-support coverage-graded static RLOO result

## Outcome

- Mechanics: **`MECHANICALLY_VALID`**.
- Train disposition:
  **`TRAIN_GAIN_RECORDED_BROAD_SUPPORT_COVERAGE_GRADED_RLOO`**.
- Scientific summary:
  **`BROAD_SUPPORT_STRICTLY_IMPROVES_TRAIN_COUNT_VECTOR_OVER_C_AND_SELECTED8`**.

The fresh shared, unmerged DoRA candidate improves natural-greedy annotated
owner counts on the registered 248-image training cohort at every threshold:

| threshold | C | candidate | gains | losses | net | matched-IoU mean C -> candidate |
|---|---:|---:|---:|---:|---:|---:|
| IoU50 | 1,259 | 1,266 | 11 | 4 | **+7** | 0.848663 -> 0.849692 |
| IoU60 | 1,186 | 1,197 | 12 | 1 | **+11** | 0.866920 -> 0.866906 |
| IoU80 | 896 | 902 | 10 | 4 | **+6** | 0.914607 -> 0.915145 |

The summed threshold-owner count is `3,341 -> 3,365`, net **+24**.  This is a
sum over three thresholds, not 24 distinct owners.  The authoritative analysis
is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-03-broad-support-coverage-graded-static-rloo-train248-read/analysis.json`,
SHA-256
`d81dd644cc4bc401bd9c059e4edee49c29dbcba7769a2e5d3af7bba7eab1de0a`.
It revalidates both scored artifact sets, recomputes all comparisons, and binds
the exact plan, update, cold read, config, adapter, manifests, and logs.

## What broader support changed

All 218 complete-clean K4 groups remain in the mathematical objective.  The 71
equal-reward groups have exactly zero RLOO terms and were elided from model
execution; the other 147 groups supplied 588 complete trajectories.

| train248 partition | rows | IoU50 net | IoU60 net | IoU80 net | summed net |
|---|---:|---:|---:|---:|---:|
| executed K4 groups | 147 | +6 | +8 | +5 | **+19** |
| zero-term-elided groups | 71 | 0 | 0 | 0 | **0** |
| outside complete-clean K4 support | 30 | +1 | +3 | +1 | **+5** |

The seven dense repeated book/broccoli/cup quality-monitor images contribute
`0/+1/-1`, summed zero.  Excluding them leaves the full summed **+24**, so the
gain is not driven by the annotation-quality cases the user marked as
questionable.

Against the prior selected-eight graded update, the broad candidate adds
`+6/+9/+3` owners at IoU50/60/80, or +18 summed threshold-owner hits.  Both
updates have norm about `0.01059`, but their directions have cosine only
`0.192690`; all 588 tensors differ.  At this dose, widening trajectory support
therefore changed the learned direction far more than the earlier nearby
reward-weight ablation and improved all three train count thresholds.

## Natural behavior and sensitivity

- raw decode text: 170/248 rows identical to C, 60 coordinate-only with the
  same description sequence, and 18 structural or semantic changes;
- matcher-visible predictions: `2,226 -> 2,184`;
- parser-dropped spans: `845 -> 555`, with dropped rows unchanged at 12;
- duplicate candidates: `49 -> 42`; strict physical candidates: `3 -> 2`;
- invalid predictions: `2 -> 2`;
- natural termination: `245 im_end / 3 length -> 246 im_end / 2 length`;
- ordering remains monitor-only: violating images/events move `43/77 -> 44/78`.

Image 360573 contributes +4 at each threshold by replacing a repetition-driven
45-prediction length cap with nine valid predictions and natural `im_end`.
Visual inspection shows a clear motorcycle-and-person scene with plausible
eight-object GT, not a dense book/fruit/vegetable/cup case.  Removing this row
still leaves `+3/+7/+2`, or **+12** summed, so it is important but not the sole
source of the result.

## Update mechanics

The eight-rank run completed 588 forwards/backwards and exactly one synchronized
backward per rank, followed by one AdamW step at `2.5e-6`.  Raw gradient norm
was `1.347861`, clipped to `0.999999`; all 588 language-DoRA tensors changed and
the realized update norm was `0.01059129`.  Runtime was 99.85 seconds.  Peak
allocated CUDA memory was 27.39 GiB and host RSS 11.82 GiB; peak reserved CUDA
memory was 33.08 GiB, exceeding the 30-GiB forecast without OOM.  This is a
recorded monitor miss, not scientific invalidation.

Fresh cold reload exactly reproduced all 588 saved tensors with no runtime
cast.  The final artifact remains universal base + embedding delta + unmerged
DoRA; no merged checkpoint was produced.

Before launch, one valid plan materialization hit a legacy CLI summary
`KeyError`.  That attempt loaded no model, launched no GPU work, and published
no candidate.  It is preserved as `MECHANICAL_INVALID`; the shared CLI bug was
fixed and tested before the valid run.

## Claim boundary and stop

**Observed:** one deterministic broad-support trajectory-credit update improves
all three registered owner-count thresholds on train248, including positive
movement outside the executable K4 population.

**Not established:** image-disjoint generalization, repeatability, full-scene
precision, F1/mAP, or production readiness.  Per the registered stop rule, this
round ends here.  An image-disjoint read of this unchanged adapter is a possible
separate experiment, not part of this result.
