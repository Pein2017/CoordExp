---
title: Coverage-Graded Static RLOO One-Update Results
description: One coverage-graded shared-DoRA update reproduces the prior train-wide positive owner-count vector but does not separate behaviorally from binary reward at this dose.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-09-02-coverage-graded-static-rloo-one-update
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-02
---

# Coverage-graded static RLOO result

## Outcome

- Mechanics: **`MECHANICALLY_VALID`**.
- Train disposition: **`TRAIN_GAIN_RECORDED_COVERAGE_GRADED_RLOO`**.
- Scientific summary:
  **`TRAIN_WIDE_POSITIVE_REPLICATED_BINARY_BEHAVIOR`**.

The fresh shared, unmerged DoRA candidate improves natural-greedy annotated
owner counts on the registered 248-image training cohort at all three reported
thresholds:

| threshold | C | candidate | gains | losses | net |
|---|---:|---:|---:|---:|---:|
| IoU50 | 1,259 | 1,260 | 3 | 2 | **+1** |
| IoU60 | 1,186 | 1,188 | 6 | 4 | **+2** |
| IoU80 | 896 | 899 | 8 | 5 | **+3** |

The summed threshold-owner count is `3,341 -> 3,347`, a net **+6**.  The
eight optimization images supply two hits and the other 240 images supply
four.  This is a small but real train-side positive result from one update;
it is not rejected for failing to be monotone owner-by-owner.

The authoritative analysis is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-coverage-graded-static-rloo-train248-read/analysis.json`,
SHA-256
`ed7f7eca42b423aa96868d3b9d93410074ad2664ba09e33c8e78b67787849c53`.
It binds the exact C and candidate rows, config, update and cold-read receipts,
three comparisons, panel decomposition, and natural-behavior monitors.

## Update and natural behavior

The eight-rank update completed 32 forwards/backwards and one AdamW step.  All
588 language-DoRA tensors changed; the realized update norm was `0.01059154`.
Cold unmerged reload was valid.  No merged checkpoint was produced.

Across train248, 173 rows are byte-identical to C, 61 change coordinates while
preserving descriptions and row count, and 14 change structure or semantics.
Both arms retain exactly 245 natural `im_end` rows and the same three length
caps.  Natural ordering remains a monitor: 43 images contain violations in
both arms, with adjacent violation events moving `77 -> 78`.

The positive count movement remains mixed debt rather than a preservation
guarantee:

- matcher-visible predictions: `2,226 -> 2,220`;
- duplicate candidates: `49 -> 44`;
- invalid predictions: `2 -> 3`;
- parser-dropped spans: `845 -> 849`.

Unmatched valid predictions remain unknown under partial annotation and are
not converted into false positives.

## Reward ablation finding

This run changed the objective from binary IoU50 coverage to
`0.90 IoU50 + 0.05 IoU60 + 0.05 IoU80`, but at this one-step dose it did not
improve on the prior binary-reward train metric vector: both candidates have
the same `+1/+2/+3` count result and the same gained and lost owner refs.

The two parameter updates are not byte-identical: their directions have cosine
`0.970764`, and the candidate-to-candidate difference norm is `0.00256113`
versus update norms near `0.0105915`.  Nevertheless, their panel8 natural
outputs are identical and only one of 248 train rows differs, by one coordinate
bin with no owner-count effect.  Therefore coverage grading is mechanically
distinct but **not behaviorally separated** here.  This suggests that the next
train lever should be broader trajectory support or exposure, not another
small reward-weight tweak.

## Claim boundary and next read

**Observed:** one shared internal DoRA update from eight images produces
positive net IoU50/60/80 behavior over train248, including four aggregate hits
outside its optimization panel.

**Not established:** image-disjoint generalization, repeatability, full-scene
precision, F1/mAP, or production readiness.  One separate image-disjoint
screen read of this unchanged adapter is descriptive follow-up, not a gate
that can erase the train result.
