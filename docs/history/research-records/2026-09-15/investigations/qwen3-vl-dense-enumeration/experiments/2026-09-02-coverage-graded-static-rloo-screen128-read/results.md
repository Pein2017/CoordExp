---
title: Coverage-Graded Static RLOO Image-Disjoint Screen-128 Results
description: The unchanged train-positive coverage-graded adapter remains count-negative but geometry-positive on the image-disjoint screen and is behaviorally equivalent to binary reward at this dose.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-09-02-coverage-graded-static-rloo-screen128-read
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-02
---

# Coverage-graded static RLOO screen128 result

## Outcome

- Mechanics: **`MECHANICALLY_VALID`**.
- Scientific summary:
  **`TRAIN_POSITIVE_SCREEN_COUNT_NEGATIVE_GEOMETRY_POSITIVE`**.
- Disposition: **record the train gain and screen debt, then pause**.

The unchanged shared DoRA remains positive on train248 (`+1/+2/+3`), but it
does not produce a joint count win on the registered image-disjoint screen:

| threshold | C | candidate | gains | losses | net |
|---|---:|---:|---:|---:|---:|
| IoU50 | 630 | 624 | 5 | 11 | **-6** |
| IoU60 | 598 | 594 | 5 | 9 | **-4** |
| IoU80 | 461 | 459 | 2 | 4 | **-2** |

The summed threshold-owner count is `1,689 -> 1,677`, net `-12`.  This is
screen debt for this exact update, not grounds to discard its verified train
gain or reject trajectory learning generally.

The authoritative analysis is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-coverage-graded-static-rloo-screen128-read/analysis.json`,
SHA-256
`6bfc68f116ecfbe3d124a1da46a718d271a220a6d08738c846d5e1ad46581968`.

## Geometry and natural behavior

Despite lower threshold counts, matched-owner geometry improves slightly at
every threshold.  Candidate-minus-C matched-IoU means are approximately
`+0.001811`, `+0.001097`, and `+0.000503` at IoU50/60/80.  Restricting to
owners matched by both arms also gives positive mean-IoU movement at all three
thresholds.  The count and geometry observations therefore point in different
directions rather than showing an optimizer no-op.

Natural behavior changes are bounded but include one known structural debt:

- 94/128 rows are byte-identical, 26 are coordinate-only changes, and 8 are
  structural or semantic changes;
- matcher-visible predictions move `1,387 -> 1,381`;
- duplicate candidates improve `52 -> 47`;
- invalid predictions remain `1 -> 1`;
- natural ordering events move `40 -> 42` on `21 -> 22` images and remain
  monitor-only;
- C has 128 natural `im_end` stops; the candidate has 127 plus one length cap.

The cap is image `59571`, with 123 valid predictions and 214 parser-dropped
spans.  Exact 127-row sensitivity leaves IoU50 and IoU80 unchanged at `-6`
and `-2`, and moves IoU60 from `-4` to `-5`; the cap is separate debt, not the
cause of the count loss.

## Reward-ablation conclusion

The coverage-graded and binary-reward candidates have identical screen
threshold counts, gains, and losses.  Their screen outputs are byte-identical
on 125/128 rows and differ only in coordinates on three rows.  Coverage grading
raises the candidate matched-IoU mean over binary reward by only about
`1.3e-5` to `1.8e-5`, far too little to change an owner-count decision.

**Observation:** the current static trajectory update transfers a small
geometry direction beyond train, but exchanges more annotated owners than it
adds on the screen.

**Inference:** changing `IoU50` reward to `90/5/5` is not the leverage point at
this dose.  If research resumes, the next train-first candidate should increase
the breadth of trajectory support or exposure rather than run another nearby
reward-weight tweak.

**Not established:** full precision, F1, mAP, repeatability, or production
readiness.  Unmatched valid predictions remain unknown under partial labels.
