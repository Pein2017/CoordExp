---
title: PVCI Step 1 Pre-Commit Probes
description: Tests boundary, coarseness, and geometry-leakage routes before committing to painted visual cursor training.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-08-pvci-step1-precommit-probes
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags: [coordexp-swift, painted-gt, pvci, step1, boundary, geometry-leakage]
updated: 2026-07-08
---

# PVCI Step-1 Pre-Commit Probes - 2026-07-08

## Scope

This note records the Step-1 pre-commit probe for the painted visual cursor idea in the Qwen3-VL painted-GT transcription research branch. The probe is intentionally diagnostic, not a final benchmark.

Question: when the model is asked for the next target row under teacher-prefix stepwise inference, does performance come from a true object-identity / visual-control route, or from a pixel-side boundary/geometry cursor that the model can copy?

Evidence scope:

- `val32`: full 19-variant boundary/coarseness matrix.
- `val100`: selective promotion of 6 high-value variants.
- Metric: target-row debug F1, not COCO mAP.
- Leakage metric: whether the first prediction is closer to the source GT box or the painted mark box.
- Leakage coordinate space: pixel-space `pred[0].bbox` compared with `painted_gt.marks[0].source_bbox_pixels` and `painted_gt.marks[0].bbox_pixels`.

## Fixed Surfaces

- Checkpoint adapter: `/data/CoordExp/outputs/painted_gt/train_overfit_gate/painted_gt_stepwise_teacher_prefix_geo_gate256_overfit16_warm_start_dora_all_towers_accelerate8_ebs8/checkpoints/step-484/adapter`
- Special-token embedding delta: `/data/CoordExp/outputs/painted_gt/train_overfit_gate/painted_gt_stepwise_teacher_prefix_geo_gate256_overfit16_warm_start_dora_all_towers_accelerate8_ebs8/checkpoints/step-484/special_token_embeddings`
- Source input JSONL: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl`
- Schedule: `geo_sorted`
- Condition: `stepwise__painted_correct`
- Decode: HF inference, `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.10`, `max_new_tokens=96`, `batch_size=2`
- Parser/eval mode: stepwise teacher-prefix target-row debug F1.

## Artifact Roots

- Val32 materialized examples: `/data/CoordExp/outputs/painted_gt/pvci_step1/materialized/boundary_val32`
- Val32 inference: `/data/CoordExp/outputs/painted_gt/pvci_step1/inference/boundary_val32`
- Val32 leakage analysis: `/data/CoordExp/outputs/painted_gt/pvci_step1/analysis/boundary_val32`
- Val32 route report: `/data/CoordExp/outputs/painted_gt/pvci_step1/reports/val32/pvci_step1_route_report.md`
- Val100 materialized examples: `/data/CoordExp/outputs/painted_gt/pvci_step1/materialized/boundary_val100`
- Val100 inference: `/data/CoordExp/outputs/painted_gt/pvci_step1/inference/boundary_val100`
- Val100 leakage analysis: `/data/CoordExp/outputs/painted_gt/pvci_step1/analysis/boundary_val100`
- Final route report: `/data/CoordExp/outputs/painted_gt/pvci_step1/reports/final/pvci_step1_route_report.md`
- Final route summary JSON: `/data/CoordExp/outputs/painted_gt/pvci_step1/reports/final/pvci_step1_route_summary.json`

## Val32 Boundary Matrix

All 19 variants completed with 256 rows each. The main effect is large: full or near-full boundary cues stay high, one-edge / center-only / background controls collapse.

| variant | F1 | precision | recall | step validity | closer GT | closer mark | tie | mean GT IoU | mean mark IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dashed_outline | 0.7597 | 0.7538 | 0.7656 | 0.9023 | 0.0000 | 0.0000 | 1.0000 | 0.8826 | 0.8826 |
| outline_only | 0.7573 | 0.7529 | 0.7617 | 0.8945 | 0.0000 | 0.0000 | 1.0000 | 0.9041 | 0.9041 |
| tight_outline_center | 0.7481 | 0.7423 | 0.7539 | 0.9102 | 0.0000 | 0.0000 | 1.0000 | 0.9017 | 0.9017 |
| inner_outline | 0.7354 | 0.7326 | 0.7383 | 0.9023 | 0.0260 | 0.9524 | 0.0216 | 0.6934 | 0.8871 |
| wrong_aspect_outline | 0.7340 | 0.7297 | 0.7383 | 0.9258 | 0.0042 | 0.9873 | 0.0084 | 0.6032 | 0.9084 |
| top_left_edges | 0.7171 | 0.7115 | 0.7227 | 0.8906 | 0.0000 | 0.0000 | 1.0000 | 0.8677 | 0.8677 |
| four_corners_only | 0.5546 | 0.4984 | 0.6250 | 0.8945 | 0.0000 | 0.0000 | 1.0000 | 0.7399 | 0.7399 |
| semi_transparent_fill | 0.4330 | 0.3601 | 0.5430 | 0.8828 | 0.0000 | 0.0000 | 1.0000 | 0.7126 | 0.7126 |
| box_1p5_outline_center | 0.2089 | 0.2069 | 0.2109 | 0.8633 | 0.0226 | 0.9729 | 0.0045 | 0.4917 | 0.9129 |
| grid_snapped_box_outline_center | 0.1969 | 0.1947 | 0.1992 | 0.9023 | 0.0130 | 0.9827 | 0.0043 | 0.3376 | 0.9560 |
| center_point | 0.1569 | 0.1348 | 0.1875 | 0.8242 | 0.5071 | 0.4739 | 0.0190 | 0.2387 | 0.0000 |
| shifted_single_edge | 0.1402 | 0.1329 | 0.1484 | 0.8789 | 0.6933 | 0.1689 | 0.1378 | 0.1915 | 0.1640 |
| center_blob | 0.1262 | 0.1201 | 0.1328 | 0.9141 | 0.1068 | 0.8932 | 0.0000 | 0.3292 | 0.7507 |
| bottom_edge_only | 0.0848 | 0.0774 | 0.0938 | 0.8633 | 0.0000 | 0.0000 | 1.0000 | 0.1505 | 0.1505 |
| top_edge_only | 0.0679 | 0.0657 | 0.0703 | 0.8555 | 0.0000 | 0.0000 | 1.0000 | 0.1251 | 0.1251 |
| background_outline | 0.0573 | 0.0530 | 0.0625 | 0.8711 | 0.2063 | 0.7892 | 0.0045 | 0.0921 | 0.7145 |
| right_edge_only | 0.0537 | 0.0528 | 0.0547 | 0.8086 | 0.0000 | 0.0000 | 1.0000 | 0.1487 | 0.1487 |
| box_2p0_outline_center | 0.0502 | 0.0496 | 0.0508 | 0.8359 | 0.0421 | 0.9533 | 0.0047 | 0.3125 | 0.9165 |
| left_edge_only | 0.0305 | 0.0299 | 0.0312 | 0.8555 | 0.0000 | 0.0000 | 1.0000 | 0.0768 | 0.0768 |

## Val100 Promotion

The promoted set was chosen after a read-only audit of the val32 matrix:

```text
tight_outline_center,outline_only,dashed_outline,four_corners_only,background_outline,wrong_aspect_outline
```

Promotion rationale:

- `tight_outline_center` and `outline_only`: canonical full-boundary baselines.
- `dashed_outline`: strongest partial-boundary route on val32.
- `four_corners_only`: weakest still-informative boundary route above half of outline-only F1.
- `background_outline`: required negative/control route.
- `wrong_aspect_outline`: sharpest geometry-leakage sentinel because it preserves high F1 while altering the rendered box geometry.

All 6 promoted variants completed with 825 rows each.

| variant | F1 | precision | recall | step validity | closer GT | closer mark | tie | mean GT IoU | mean mark IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| outline_only | 0.7734 | 0.7711 | 0.7758 | 0.9188 | 0.0000 | 0.0000 | 1.0000 | 0.8955 | 0.8955 |
| dashed_outline | 0.7483 | 0.7452 | 0.7515 | 0.9164 | 0.0000 | 0.0000 | 1.0000 | 0.8763 | 0.8763 |
| tight_outline_center | 0.7483 | 0.7452 | 0.7515 | 0.9236 | 0.0000 | 0.0000 | 1.0000 | 0.8942 | 0.8942 |
| wrong_aspect_outline | 0.7275 | 0.7206 | 0.7345 | 0.9321 | 0.0130 | 0.9805 | 0.0065 | 0.5981 | 0.8937 |
| four_corners_only | 0.5278 | 0.4727 | 0.5976 | 0.9176 | 0.0000 | 0.0000 | 1.0000 | 0.7287 | 0.7287 |
| background_outline | 0.0461 | 0.0420 | 0.0509 | 0.8679 | 0.2151 | 0.7835 | 0.0014 | 0.0716 | 0.7061 |

## Identity Versus Geometry Leakage

The result is not consistent with a pure object-identity route.

The clean full-outline variants are not leakage evidence by themselves because their source-object pixel box and rendered mark box coincide. They produce tie rates of `1.0000`, which means they prove the boundary cue is useful but do not distinguish object identity from mark geometry.

The strongest leakage evidence is the promoted `wrong_aspect_outline` variant. At val100 it keeps high target-row debug F1 (`0.7275`) and high step validity (`0.9321`), while predictions are rarely closer to the source GT box (`closer_gt=0.0130`) and overwhelmingly closer to the rendered wrong-aspect mark (`closer_mark=0.9805`). This means the model uses the painted boundary as a strong geometry cursor when source geometry and mark geometry conflict.

The background control stays poor (`F1=0.0461` at val100), even though its predictions are often closer to the unrelated background mark than to the source GT box. So the model is not merely solving the task from any rectangle. It needs a mark coupled to the target object region; once geometry conflict is introduced around the target, the emitted coordinates follow the rendered boundary more than the original box.

## Boundary Ablation

The boundary cue has strong structure dependence:

- Full or near-full boundary cues are high: `outline_only`, `tight_outline_center`, `dashed_outline`, `inner_outline`, `top_left_edges`, and `wrong_aspect_outline`.
- `four_corners_only` remains informative but clearly weaker.
- Single-edge cues collapse: top/bottom/left/right edge-only variants stay near the background/low-control regime.
- Center-only cues are weak compared with boundary cues.

This suggests the useful signal is not just object selection or center pointing. The model benefits from an explicit spatial boundary trace.

## Route Verdict

Primary verdict: `geometry_leakage_dominant`.

The probe shows that painted visual hints can strongly steer Qwen3-VL and can overwhelm the conservative language prior, but Step-1 does not prove a clean object-identity variable. The high-performing unperturbed outline routes are compatible with both identity and boundary following because source and mark coincide. The wrong-aspect sentinel breaks that equivalence and shows that, under conflict, the model follows the rendered boundary.

Practical constraint for the next probe: `boundary_precision_required`.

Before implementing a hidden cursor, feature-space cursor, or PICD-style internal intervention, the next probe should explicitly decide whether it wants:

- a controllable boundary cursor that is allowed to define the emitted geometry; or
- an object-identity cursor that should preserve source-object geometry even when the visual mark geometry is perturbed.

These are different mechanisms and should not be mixed in one claim.

## Interpretation

This is a positive result for visual control, but a negative result for the strongest object-identity interpretation.

What it supports:

- Qwen3-VL can use explicit painted visual information to generate object rows much better than without useful object-coupled marks.
- Boundary-like marks are much more effective than single-edge, center-only, or background controls.
- A future visual cursor mechanism is worth studying if the target is a boundary/region cursor.

What it does not support yet:

- A claim that the model has a clean current-object identity variable isolated from rendered geometry.
- A claim that the painted mark only selects the object while the model recovers the original GT box.
- A full benchmark or COCO mAP claim.

## Caveats

- Evidence is target-row debug F1, not final mAP.
- The val100 promotion set is selective, not a full 19-variant replication.
- The leakage analyzer compares the first parsed prediction with GT and painted mark geometry; it is diagnostic, not a full detection evaluator.
- These probes use the Step-484 painted-GT adapter under teacher-prefix stepwise inference, not the final self-prefix/human-annotation rollout.
- Some parser/generation failures remain, but they are not the dominant signal in this probe.

## Next Actions

Recommended next direction:

1. Treat this probe as evidence that pixel/feature boundary cursors are powerful.
2. Do not advance directly to a hidden object-identity claim.
3. Design Step-2 around boundary precision: perturb boundary geometry, mask style, and feature-space mark transplantation to decide whether the useful intervention is a visual boundary cursor or a deeper object representation.
4. If the goal becomes PICD, define whether PICD should preserve the rendered boundary geometry or recover source GT geometry under perturbed marks.

## Research Unit Closeout

Observed:

- Full or near-full boundary cues remained high-performing.
- Single-edge, center-only, and background controls collapsed.
- Geometry-leakage sentinels showed that the model often followed rendered mark
  geometry rather than source GT geometry when those differed.

Supported:

- The model consumes object-coupled boundary cues strongly.
- The visual route is not simply arbitrary rectangle copying because background
  outline controls stayed poor.
- The next necessary fork is anti-copy snap-radius training.

Not supported yet:

- A clean object-designator route independent of rendered mark geometry.
- Hidden-cursor distillation readiness.
- Production inference behavior.

Next decider:

- Step 2 snap-radius / anti-copy training and evaluation.

Promotion decision:

- Not promoted. This remains non-normative pre-commit evidence for PVCI.
