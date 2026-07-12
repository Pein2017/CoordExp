---
title: PVCI Step 0 Preparation Results
description: Records wrong-mark, held-out GT-mark, and mark-coarseness preparation probes for the painted visual cursor line.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-08-pvci-step0-preparation-results
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags: [coordexp-swift, painted-gt, pvci, step0, preparation, visual-steering]
updated: 2026-07-08
---

# PVCI Step 0 Preparation Results - 2026-07-08

## Scope

This note records the first PVCI / PICD preparation experiments requested after
the Visual Cursor Internalization review. The goal was not to implement hidden
or feature-space cursors yet. The goal was to answer three causal-preparation
questions with the existing painted-GT stepwise teacher-prefix checkpoint:

1. Does a wrong visual mark steer the model to the marked object, or merely
   disrupt decoding?
2. Does GT-mark one-by-one transcription generalize to held-out validation
   images?
3. How precise must the visual mark be before transcription collapses?

## Fixed Surfaces

- Worktree: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`
- Checkpoint adapter:
  `/data/CoordExp/outputs/painted_gt/train_overfit_gate/painted_gt_stepwise_teacher_prefix_geo_gate256_overfit16_warm_start_dora_all_towers_accelerate8_ebs8/checkpoints/step-484/adapter`
- Special-token embedding delta:
  `/data/CoordExp/outputs/painted_gt/train_overfit_gate/painted_gt_stepwise_teacher_prefix_geo_gate256_overfit16_warm_start_dora_all_towers_accelerate8_ebs8/checkpoints/step-484/special_token_embeddings`
- Held-out source JSONL:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl`
- Decode: HF backend, `temperature=0.0`, `top_p=1.0`,
  `repetition_penalty=1.10`, `max_new_tokens=96`
- Inference batch size: `2`
- Materialization preflight: skipped for these experiments to avoid extra
  Qwen processor/model setup during concurrent training. Inference itself wrote
  `image_plan.jsonl` artifacts for the executed runs.

## Artifact Roots

- Materialized held-out val100 tight conditions:
  `/data/CoordExp/outputs/painted_gt/pvci_step0/materialized/heldout_val100_tight`
- Materialized held-out val32 coarseness conditions:
  `/data/CoordExp/outputs/painted_gt/pvci_step0/materialized/coarseness_val32`
- Held-out val100 inference:
  `/data/CoordExp/outputs/painted_gt/pvci_step0/inference/heldout_val100_tight`
- Coarseness val32 inference:
  `/data/CoordExp/outputs/painted_gt/pvci_step0/inference/coarseness_val32`

## 0A - Wrong-Mark Re-Score

Target-row metric on held-out val100:

| Condition | Rows | F1 | Precision | Recall | Step Validity |
| --- | ---: | ---: | ---: | ---: | ---: |
| correct tight mark | 825 | 0.748340 | 0.745192 | 0.751515 | 0.923636 |
| unpainted same prompt | 825 | 0.143558 | 0.113540 | 0.195152 | 0.764848 |
| wrong-object mark, scored against intended target | 825 | 0.017857 | 0.017544 | 0.018182 | 0.803636 |

Wrong-mark follow analysis:

| Quantity | Value |
| --- | ---: |
| rows | 825 |
| accepted rows | 663 |
| rows with first prediction | 653 |
| first prediction IoU >= 0.5 with marked object | 636 |
| marked-object rate over rows with prediction | 0.973966 |
| first prediction IoU >= 0.5 with intended target | 3 |
| intended-target rate over rows with prediction | 0.004594 |
| mark IoU > target IoU rate | 0.995406 |
| mean mark IoU | 0.874114 |
| mean target IoU | 0.033087 |

Evidence file:
`/data/CoordExp/outputs/painted_gt/pvci_step0/inference/heldout_val100_tight/wrong_object_mark_step484_rp110_bs2/wrong_object_follow_mark_analysis.json`

Verdict: wrong marks are steering actuators, not generic disruption. When the
mark is wrong, the model confidently follows the marked object. For PVCI this
means selector errors are dangerous: they will likely become confident false
positive rows unless a verifier/reject mechanism exists.

## 0B - Held-Out GT-Mark Generalization

The held-out correct tight mark reaches F1 `0.748340`, versus held-out unpainted
F1 `0.143558`. The effect therefore generalizes strongly: the model can consume
a GT visual mark on unseen images and bind row transcription to that marked
region far better than without a mark.

However, the absolute held-out F1 is far below the train256 overfit reference
from the previous counterfactual summary:

| Scope | correct mark F1 | unpainted F1 | delta |
| --- | ---: | ---: | ---: |
| train256 overfit gate | 0.936071 | 0.133606 | +0.802465 |
| held-out val100 Step 0 | 0.748340 | 0.143558 | +0.604783 |

Verdict: the causal visual cue is real and generalizes, but the current
painted-GT teacher is not yet a high-ceiling held-out teacher. Before serious
PICD hidden-cursor distillation, it is probably worth expanding or regularizing
the painted teacher rather than distilling from only the tiny overfit teacher.

## 0C - Mark Coarseness Tolerance Curve

All coarseness variants below use the same held-out val32 slice and
`stepwise__painted_correct` target rows.

| Mark variant | Rows | F1 | Precision | Recall | Step Validity |
| --- | ---: | ---: | ---: | ---: | ---: |
| tight outline + center | 256 | 0.748062 | 0.742308 | 0.753906 | 0.910156 |
| outline only | 256 | 0.757282 | 0.752896 | 0.761719 | 0.894531 |
| semi-transparent fill | 256 | 0.433022 | 0.360104 | 0.542969 | 0.882812 |
| 1.5x box outline + center | 256 | 0.208897 | 0.206897 | 0.210938 | 0.863281 |
| grid-snapped box outline + center | 256 | 0.196911 | 0.194656 | 0.199219 | 0.902344 |
| center point | 256 | 0.156863 | 0.134831 | 0.187500 | 0.824219 |
| center blob | 256 | 0.126160 | 0.120141 | 0.132812 | 0.914062 |
| 2.0x box outline + center | 256 | 0.050193 | 0.049618 | 0.050781 | 0.835938 |

Verdict: the useful visible cue is the tight rectangle boundary, not the center
dot. Removing the center dot does not hurt; outline-only is slightly higher
than tight outline+center in this slice. Coarse region hints collapse quickly:
1.5x and grid-snapped boxes are near `0.20`, 2.0x is near `0.05`, and
center-only variants are weak.

## Implications

1. The model has learned to read an externally supplied object pointer. That
   supports the VCI/PVCI framing: current AR failures are plausibly missing or
   unstable object designation, not inability to transcribe a designated object.
2. The effective pointer is spatially precise. A future selector cannot be only
   a vague center point or large proposal unless the teacher/training procedure
   is changed to make coarse marks usable.
3. A feature-space mark should probably emulate a boundary/contour-like visual
   signal first. A blob-like or fill-like signal is less aligned with the
   current teacher behavior.
4. Wrong selectors should be expected to steer confidently to the wrong object.
   Future PVCI must include a reject/verifier or selector-quality gate before
   treating row transcription as trustworthy.
5. The Step 0 gate supports continuing the research, but suggests one
   prerequisite before hidden/feature cursor work: train or evaluate a stronger
   painted teacher on broader held-out coverage, or intentionally train a
   teacher that tolerates coarser marks.

## Next Recommended Slice

Run a small teacher-strengthening experiment rather than jumping directly to
hidden cursor transplant:

- train a painted stepwise teacher on a larger slice with tight outline-only
  and tight outline+center marks;
- include a controlled small fraction of coarser marks only if the goal is to
  make the later selector less localization-sensitive;
- keep the wrong-mark steering analysis as a required safety check after each
  teacher update;
- only start PICD hidden-cursor distillation once held-out correct-mark F1 is
  closer to the train overfit gate, or once we intentionally accept a
  lower-ceiling teacher for mechanism probing.

## Research Unit Closeout

Observed:

- Wrong-object marks steered first predictions to the marked object at very
  high rate.
- Held-out GT-mark transcription generalized strongly beyond the tiny train
  slice.
- Coarse marks degraded sharply, showing that the current teacher was
  precise-boundary dependent.

Supported:

- Painted marks are real visual steering actuators rather than generic
  disruption.
- Selector errors are likely to become confident false-positive steering.
- PVCI should separate object identity from geometry-copy behavior before
  hidden-cursor or feature-cursor work.

Not supported yet:

- A robust coarse-box selector.
- A learned non-pixel cursor.
- Production inference behavior.

Next decider:

- Step 1 boundary/coarseness probes to distinguish object identity/control
  from geometry-copy shortcut behavior.

Promotion decision:

- Not promoted. This remains non-normative preparation evidence for later PVCI
  probes.
