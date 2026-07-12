---
title: PVCI Step 2 Anti-Copy Snap-Radius Results
description: Records the anti-copy snap-radius training and transfer results for converting painted marks from geometry-copy cues toward object designators.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-08-pvci-step2-anticopy-snap-radius-results
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags: [coordexp-swift, painted-gt, pvci, step2, anti-copy, snap-radius]
updated: 2026-07-08
---

# PVCI Step-2 Anti-Copy Snap-Radius Results

Date: 2026-07-08

## Scope

This note records the Step-2 painted visual causal intervention (PVCI) snap-radius probe.
The question is whether the Step-484 painted stepwise adapter can be trained to treat a
painted mark as an object pointer without copying the mark geometry itself.

Evidence scope:

- training slice: train256 anti-copy jitter mix, 1,923 target-step rows;
- no-train calibration: existing E0 val32 fixed-variant panel from
  [PVCI Step 2 E0 Snap-Radius Calibration](../2026-07-08-pvci-step2-e0-snap-radius-calibration/unit.md);
- transfer panel: E1 val32 fixed variants, 256 target-step rows per variant;
- selected promotion: E0/E1 val100 fixed variants, 825 target-step rows per variant.

This remains a debug target-row probe, not final COCO mAP.

## Artifacts

Training run:

`/data/CoordExp/outputs/painted_gt/pvci_step2/train_snap_radius/painted_gt_pvci_step2_snap_radius_train256_anti_copy_mix_step484_warm_start_dora_all_towers_accelerate8_ebs8_16epoch-pvci-step2-anticopy-retry3-20260708T1726Z`

Training final checkpoint:

`/data/CoordExp/outputs/painted_gt/pvci_step2/train_snap_radius/painted_gt_pvci_step2_snap_radius_train256_anti_copy_mix_step484_warm_start_dora_all_towers_accelerate8_ebs8_16epoch-pvci-step2-anticopy-retry3-20260708T1726Z/checkpoints/step-484`

Train256 replay:

`/data/CoordExp/outputs/painted_gt/pvci_step2/inference/e1_train256_anti_copy_mix_step484/train256_anti_copy_mix_step484_rp110_bs2`

Val32 E1 inference:

`/data/CoordExp/outputs/painted_gt/pvci_step2/inference/e1_val32_anticopy_step484`

Val100 E0/E1 inference:

- `/data/CoordExp/outputs/painted_gt/pvci_step2/inference/e0_val100_selected_step484`
- `/data/CoordExp/outputs/painted_gt/pvci_step2/inference/e1_val100_selected_anticopy_step484`

Leakage summaries:

- `/data/CoordExp/outputs/painted_gt/pvci_step2/analysis/e1_train256_anti_copy_mix_step484/leakage`
- `/data/CoordExp/outputs/painted_gt/pvci_step2/analysis/e1_val32_anticopy_step484/leakage`
- `/data/CoordExp/outputs/painted_gt/pvci_step2/analysis/e0_val100_selected_step484/leakage`
- `/data/CoordExp/outputs/painted_gt/pvci_step2/analysis/e1_val100_selected_anticopy_step484/leakage`

## Training Gate

E1 was trained from the Step-484 painted stepwise adapter, with all-tower DoRA and special-token embedding payload loading preserved.
Final checkpoint is `step-484`; no training warnings were recorded.

Final training metrics:

- final loss: `0.0434`;
- final token top-1: `1.000`;
- final token top-5: `1.000`;
- last-25 loss mean: `0.0450`;
- last-25 token top-1 mean: `0.9995`.

Train256 replay confirms the model learned the anti-copy target panel:

| scope | F1 | step_validity | pred_gt_iou | pred_mark_iou | closer_gt | closer_mark | class_match |
|---|---:|---:|---:|---:|---:|---:|---:|
| train256 anti-copy mix | 0.995 | 0.996 | 0.999 | 0.637 | 0.801 | 0.000 | 0.999 |

Interpretation: on the training panel, the model predicts the tight GT box rather than copying the corrupted rendered mark.

## Val32 Transfer

Compared with E0 Step-484 no-train calibration, E1 strongly reduces snap-to-mark behavior on corrupted marks.

| variant | E0 F1 | E1 F1 | delta | E0 closer_mark | E1 closer_mark | E0 pred_gt_iou | E1 pred_gt_iou | E1 class_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aspect_jitter_outline | 0.480 | 0.646 | +0.166 | 0.991 | 0.240 | 0.562 | 0.694 | 0.886 |
| background_outline | 0.057 | 0.078 | +0.021 | 0.789 | 0.779 | 0.092 | 0.097 | 0.350 |
| box_1p5_outline_center | 0.209 | 0.714 | +0.505 | 0.973 | 0.116 | 0.492 | 0.744 | 0.853 |
| box_2p0_outline_center | 0.050 | 0.574 | +0.524 | 0.953 | 0.109 | 0.313 | 0.682 | 0.827 |
| grid_snapped_32_outline | 0.542 | 0.701 | +0.159 | 0.996 | 0.155 | 0.634 | 0.747 | 0.875 |
| grid_snapped_box_outline_center | 0.197 | 0.529 | +0.332 | 0.983 | 0.135 | 0.338 | 0.595 | 0.804 |
| jitter_large_outline | 0.070 | 0.569 | +0.500 | 0.964 | 0.140 | 0.329 | 0.675 | 0.842 |
| jitter_medium_outline | 0.426 | 0.738 | +0.312 | 0.991 | 0.121 | 0.549 | 0.756 | 0.875 |
| jitter_mild_outline | 0.791 | 0.737 | -0.054 | 0.922 | 0.159 | 0.814 | 0.776 | 0.884 |
| outline_only | 0.757 | 0.710 | -0.047 | 0.000 | 0.000 | 0.904 | 0.766 | 0.877 |
| tight_outline_center | 0.748 | 0.698 | -0.050 | 0.000 | 0.000 | 0.902 | 0.752 | 0.877 |
| wrong_aspect_outline | 0.734 | 0.644 | -0.090 | 0.987 | 0.392 | 0.603 | 0.665 | 0.862 |

Group means:

| group | mean E0 F1 | mean E1 F1 | mean delta |
|---|---:|---:|---:|
| train-like anti-copy variants | 0.560 | 0.706 | +0.146 |
| hard far-geometry variants | 0.131 | 0.596 | +0.465 |
| tight/same-geometry marks | 0.753 | 0.704 | -0.049 |
| controls | 0.396 | 0.361 | -0.035 |

## Selected Val100 Promotion

The val100 selected panel uses the same val200 source override and `geo_sorted` schedule family as E0 val32.
The tight variant was materialized with Qwen no-resize preflight enabled. After this checked variant completed, the remaining selected variants were materialized with `--skip-preflight` to avoid repeated model-loading overhead; all six variants share the same slice and schedule IDs:

- slice: `painted_gt_slice_273a69f1f4a3e5d2`;
- schedule: `stepwise_geo_sorted_86698d43e7af55d7`;
- row count: 825 target-step rows per variant.

| variant | E0 F1 | E1 F1 | delta | E0 closer_mark | E1 closer_mark | E0 pred_gt_iou | E1 pred_gt_iou | E1 class_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| background_outline | 0.046 | 0.068 | +0.022 | 0.784 | 0.797 | 0.072 | 0.082 | 0.346 |
| box_2p0_outline_center | 0.032 | 0.563 | +0.531 | 0.971 | 0.136 | 0.297 | 0.650 | 0.819 |
| grid_snapped_32_outline | 0.584 | 0.700 | +0.116 | 0.996 | 0.182 | 0.633 | 0.728 | 0.875 |
| jitter_large_outline | 0.052 | 0.538 | +0.486 | 0.984 | 0.202 | 0.315 | 0.628 | 0.827 |
| jitter_medium_outline | 0.454 | 0.705 | +0.251 | 0.989 | 0.140 | 0.536 | 0.734 | 0.877 |
| tight_outline_center | 0.748 | 0.693 | -0.056 | 0.000 | 0.000 | 0.894 | 0.746 | 0.852 |

## Verdict

Route status: green for the mechanism claim, with a controlled tradeoff.

The evidence supports this statement:

> The painted mark can act as a visual object pointer, and the decoder's tendency to copy the rendered mark geometry is trainable away. After anti-copy training, the model often binds the marked object while emitting the tight GT box instead of the corrupted mark geometry.

The strongest evidence is the direction flip:

- E0 corrupted marks usually produce `closer_mark` near 0.95-0.99.
- E1 corrupted marks reduce `closer_mark` to roughly 0.11-0.20 on val32/val100 selected variants.
- F1 improves massively on hard geometry corruptions such as jitter-large and box-2.0.
- Background marks remain poor, so the intervention is not merely generic prompt adaptation or class-language memorization.

The main tradeoff is also clear:

- Tight/same-geometry marks drop by about 0.05 F1.
- E1 tight-mark pred-GT IoU drops from roughly 0.89 to 0.75 on val100.

Interpretation: anti-copy training shifts the model away from literal mark-boundary transcription and toward object-conditioned tight-box prediction. That is desirable for corrupted marks, but it slightly weakens the original "copy the exact painted boundary" behavior when the mark is already perfect.

## Next Probe Seeds

1. Test whether a mixed objective can recover tight-mark performance while preserving anti-copy robustness.
   A likely next training mix is tight marks plus anti-copy jitter, not anti-copy-only.
2. Run a small self-prefix or next-object variant only after the teacher-prefix mechanism is stable; this Step-2 result is about object-pointer/geometry binding, not coverage-ledger decoding yet.
3. If moving toward production-style training, add a mark-corruption curriculum: tight marks early, then medium jitter/grid/large jitter.
4. Keep background marks as a negative control for future visual-pointer methods.

## Caveats

- These are debug F1 and leakage metrics on stepwise teacher-prefix rows, not final one-shot detection mAP.
- Val100 materialization used preflight for the first selected variant and skipped repeated preflight for the remaining selected variants; all variants share the same materialization script, source, slice, schedule, and template identity.
- The route is promising for visual object-pointer supervision, but not yet a complete inference-time method.

## Research Unit Closeout

Observed:

- E1 anti-copy training learned the train256 anti-copy target panel.
- On val32 fixed variants, E1 strongly reduced snap-to-mark behavior for many
  corrupted marks compared with E0.
- Selected val100 promotion preserved a usable but imperfect route.

Supported:

- Anti-copy training can partially transform painted marks from geometry-copy
  cues toward object-designator cues.
- The route remains sketch-and-snap rather than a complete geometry-invariant
  object identity mechanism.

Not supported yet:

- Source-image-disjoint cursor generalization.
- Production detector coverage.
- A learned hidden cursor or reusable non-pixel cursor.

Next decider:

- Identity-conflict, corrupted wrong-object, and feature-replay probes to test
  whether the visual actuator carries row/object identity beyond same-object
  anti-copy geometry correction.

Promotion decision:

- Not promoted. This remains target-row debug research evidence and does not
  define stable training or inference behavior.
