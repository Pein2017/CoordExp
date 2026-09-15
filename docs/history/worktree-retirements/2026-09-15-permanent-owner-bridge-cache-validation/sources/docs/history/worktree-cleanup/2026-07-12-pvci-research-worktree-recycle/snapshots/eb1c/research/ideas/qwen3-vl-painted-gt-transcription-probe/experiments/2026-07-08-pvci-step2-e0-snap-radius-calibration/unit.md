---
title: PVCI Step 2 E0 Snap-Radius Calibration
description: Records no-train snap-radius calibration for the Step-484 painted stepwise adapter before anti-copy training.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-08-pvci-step2-e0-snap-radius-calibration
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags: [coordexp-swift, painted-gt, pvci, step2, calibration, snap-radius]
updated: 2026-07-08
---

# PVCI Step-2 E0 Snap-Radius Calibration

Date: 2026-07-08

## Scope

This note records the no-train Step-2 calibration for the painted visual causal intervention (PVCI) line. The adapter is the existing Step-484 painted stepwise warm-start checkpoint. The evaluation scope is target-row debug evidence on a val32-derived stepwise panel, not final COCO mAP.

Inference root:

`/data/CoordExp/outputs/painted_gt/pvci_step2/inference/snap_e0_val32`

Leakage analysis root:

`/data/CoordExp/outputs/painted_gt/pvci_step2/analysis/snap_e0_val32/leakage`

## Result Table

| variant | F1 | step_validity | closer_gt | closer_mark | pred_gt_iou | pred_mark_iou | class_match | parser_fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tight_outline_center | 0.748 | 0.910 | 0.000 | 0.000 | 0.902 | 0.902 | 0.833 | 23 |
| outline_only | 0.757 | 0.895 | 0.000 | 0.000 | 0.904 | 0.904 | 0.852 | 27 |
| jitter_mild_outline | 0.791 | 0.902 | 0.056 | 0.922 | 0.814 | 0.911 | 0.883 | 25 |
| jitter_medium_outline | 0.426 | 0.883 | 0.004 | 0.991 | 0.549 | 0.921 | 0.845 | 30 |
| jitter_large_outline | 0.070 | 0.859 | 0.032 | 0.964 | 0.329 | 0.923 | 0.736 | 36 |
| aspect_jitter_outline | 0.480 | 0.902 | 0.009 | 0.991 | 0.562 | 0.917 | 0.835 | 25 |
| grid_snapped_32_outline | 0.542 | 0.895 | 0.000 | 0.996 | 0.634 | 0.935 | 0.869 | 27 |
| grid_snapped_box_outline_center | 0.197 | 0.902 | 0.013 | 0.983 | 0.338 | 0.956 | 0.736 | 25 |
| box_1p5_outline_center | 0.209 | 0.863 | 0.023 | 0.973 | 0.492 | 0.913 | 0.824 | 35 |
| box_2p0_outline_center | 0.050 | 0.836 | 0.042 | 0.953 | 0.313 | 0.916 | 0.729 | 42 |
| wrong_aspect_outline | 0.734 | 0.926 | 0.004 | 0.987 | 0.603 | 0.908 | 0.810 | 19 |
| background_outline | 0.057 | 0.871 | 0.206 | 0.789 | 0.092 | 0.714 | 0.386 | 33 |

## Interpretation

The Step-484 painted stepwise adapter has a clear visual-mark snap behavior. When the rendered mark equals the target geometry, debug F1 is high. When the mark is anti-copy jittered, the emitted box usually moves with the visible mark rather than staying at the GT target. Medium jitter, aspect jitter, and grid-snapped 32 all show `closer_mark` near 0.99.

This is not a failure of visual control. It is evidence that the current adapter treats the rendered mark as box supervision. Step-2 should therefore train an anti-copy objective: keep the visual mark object-coupled but geometry-corrupted, while the target remains the tight GT row.

## Next Action

E1 should train a train256 anti-copy jitter mix from the Step-484 painted stepwise adapter, then evaluate:

- train256 anti-copy mix, to confirm learnability on the materialized training panel;
- val32 fixed variants, to test whether anti-copy behavior transfers beyond the training examples;
- selected val100 promotion only after val32 shows a usable route.

## Research Unit Closeout

Observed:

- The Step-484 painted stepwise adapter had clear visual-mark snap behavior
  before anti-copy training.
- When rendered mark geometry differed from GT geometry, predictions usually
  moved with the rendered mark.

Supported:

- The baseline adapter treated rendered marks as box supervision.
- Anti-copy training was necessary before claiming an object-designator route.

Not supported yet:

- Geometry-invariant object designation.
- Hidden-cursor readiness.
- Production inference behavior.

Next decider:

- Train and evaluate a train256 anti-copy jitter mix from the Step-484 painted
  stepwise adapter.

Promotion decision:

- Not promoted. This remains calibration evidence for the Step-2 anti-copy
  probe.
