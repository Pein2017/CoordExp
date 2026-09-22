---
title: PVCI Identity Conflict Probe
description: Tests whether wrong-object painted marks steer row-level decoding toward the marked object or the scheduled target, with E0 and E1 comparisons.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-identity-conflict
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - painted-gt
  - pvci
updated: 2026-07-09
---

# PVCI Identity Conflict Probe

## Question

Does a wrong-object painted mark causally steer row-level decoding toward the
marked object `j`, or does the model preserve the scheduled target object `i`
from the teacher-prefix row context?

This unit narrows the previous Step-2 interpretation. The anti-copy result
proved that the model can avoid copying corrupted mark geometry for the same
object. It did not by itself prove identity/pointer dominance. The decisive
identity-conflict control uses rows where the assistant target remains object
`i`, while the visual mark is deliberately placed on object `j`.

## Evidence Scope

- Checkout or branch: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`,
  `codex/qwen3-vl-painted-gt-transcription-probe`.
- Commit or diff scope: uncommitted local additions for the identity-conflict
  analyzer and two inference configs.
- Baseline checkpoint: production step917 adapter,
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_4epoch_warmup0p1-prod8-r16a32-ebs64-warmup0p1-20260702T170007Z/checkpoints/step-917`.
- E1 anti-copy checkpoint:
  `/data/CoordExp/outputs/painted_gt/pvci_step2/train_snap_radius/painted_gt_pvci_step2_snap_radius_train256_anti_copy_mix_step484_warm_start_dora_all_towers_accelerate8_ebs8_16epoch-pvci-step2-anticopy-retry3-20260708T1726Z/checkpoints/step-484`.
- Wrong-object materialized input:
  `/data/CoordExp/outputs/painted_gt/pvci_step0/materialized/heldout_val100_tight/conditions/stepwise__wrong_object_mark/stepwise.wrong_object_mark.examples.jsonl`.
- E0 wrong-object artifact:
  `/data/CoordExp/outputs/painted_gt/pvci_step0/inference/heldout_val100_tight/wrong_object_mark_step484_rp110_bs2`.
- E1 wrong-object artifact:
  `/data/CoordExp/outputs/painted_gt/pvci_identity_conflict/inference/e1_wrong_object_val100_tight/step484_anticopy_rp110_bs2`.
- Baseline standard val200 artifact:
  `/data/CoordExp/outputs/painted_gt/predicted_ledger_baselines/painted-gt-standard-step917-val200-rp110`.
- E1 standard val200 artifact:
  `/data/CoordExp/outputs/painted_gt/pvci_identity_conflict/inference/e1_standard_unpainted_val200/step484_anticopy_rp110_bs2`.
- New analyzer:
  `src/painted_gt/identity_conflict.py` and
  `scripts/probes/painted_gt/analyze_identity_conflict.py`.
- Sample window: 825 row-level wrong-object examples from heldout val100; 200
  one-shot standard unpainted val images.
- Known limitations: wrong-object tight marks have `negative_bbox == mark_bbox`,
  so this unit separates target-vs-marked-object identity but does not separate
  marked-object source bbox from rendered mark geometry. Corrupted wrong-object
  marks remain the next decider for copy-vs-object-source separation.

## Procedure

1. Analyze the existing E0 wrong-object artifact with an explicit
   target/negative/mark metric that does not reuse the ambiguous
   `mark.source_bbox_pixels` GT-reference behavior from the previous leakage
   helper.
2. Launch E1 anti-copy checkpoint on the same wrong-object val100 input.
3. Analyze E1 with the same identity-conflict metric.
4. Run E1 on standard unpainted val200 with the baseline one-shot prompt and
   evaluate it with the CoordExp-Swift detection consumer.
5. Compare E1 standard one-shot behavior against the already-existing step917
   baseline val200 artifact.

Expected failure modes before results:

- A model that follows text/schedule should predict object `i` despite the
  wrong visual mark.
- A model with visual designator control should predict object `j`.
- A model copying rendered marks rather than object identity would need
  corrupted wrong-object marks to expose; tight marks cannot distinguish these.
- A stepwise-trained checkpoint may degrade standard one-shot enumeration even
  if it remains useful for row-level mechanism probes.

## Commands

```bash
python scripts/probes/painted_gt/analyze_identity_conflict.py \
  --input /data/CoordExp/outputs/painted_gt/pvci_step0/inference/heldout_val100_tight/wrong_object_mark_step484_rp110_bs2/gt_vs_pred.jsonl \
  --output-dir /data/CoordExp/outputs/painted_gt/pvci_step0/inference/heldout_val100_tight/wrong_object_mark_step484_rp110_bs2/identity_conflict \
  --condition-name pvci_step0_heldout_val100_tight_wrong_object_mark_step484

CUDA_VISIBLE_DEVICES=4 python -m src.infer \
  --config configs/coordexp_swift/infer/painted_gt/pvci_identity_conflict/e1_wrong_object_val100_tight_step484_rp110_bs2.yaml

python scripts/probes/painted_gt/analyze_identity_conflict.py \
  --input /data/CoordExp/outputs/painted_gt/pvci_identity_conflict/inference/e1_wrong_object_val100_tight/step484_anticopy_rp110_bs2/gt_vs_pred.jsonl \
  --output-dir /data/CoordExp/outputs/painted_gt/pvci_identity_conflict/inference/e1_wrong_object_val100_tight/step484_anticopy_rp110_bs2/identity_conflict \
  --condition-name pvci_identity_conflict_e1_wrong_object_val100_tight_step484

CUDA_VISIBLE_DEVICES=4,5 python -m src.infer \
  --config configs/coordexp_swift/infer/painted_gt/pvci_identity_conflict/e1_standard_unpainted_val200_step484_rp110_bs2.yaml

python scripts/evaluate_detection.py \
  --artifact-dir /data/CoordExp/outputs/painted_gt/pvci_identity_conflict/inference/e1_standard_unpainted_val200/step484_anticopy_rp110_bs2 \
  --out-dir /data/CoordExp/outputs/painted_gt/pvci_identity_conflict/inference/e1_standard_unpainted_val200/step484_anticopy_rp110_bs2/eval_detection
```

## Observations

### Wrong-Object Identity Conflict

| condition | rows | prediction rate | pointer candidate rate | schedule candidate rate | mean IoU to target `i` | mean IoU to negative `j` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 stepwise baseline, tight wrong-object mark | 825 | 0.8036 | 0.9744 | 0.0256 | 0.0404 | 0.8741 |
| E1 anti-copy checkpoint, tight wrong-object mark | 825 | 0.8097 | 0.9731 | 0.0269 | 0.0410 | 0.7028 |

Direct observations:

- Both E0 and E1 overwhelmingly emit geometry nearest to the marked negative
  object `j`, not the scheduled row target `i`.
- E1 preserves the wrong-mark identity steering rate but becomes less
  geometrically tight to the marked object: mean IoU to `j` drops from `0.8741`
  to `0.7028`.
- Same-description rows dominate this control (`0.7673` of rows), but the
  different-description split still shows strong pointer-candidate behavior:
  E0 `0.9021` pointer candidate rate on different-description rows.
- Tight wrong-object marks cannot distinguish “follow object `j`” from “copy
  the rendered tight box,” because the source box and rendered mark box are the
  same.

Artifact handles:

- E0 identity summary:
  `/data/CoordExp/outputs/painted_gt/pvci_step0/inference/heldout_val100_tight/wrong_object_mark_step484_rp110_bs2/identity_conflict/identity_conflict.json`.
- E1 identity summary:
  `/data/CoordExp/outputs/painted_gt/pvci_identity_conflict/inference/e1_wrong_object_val100_tight/step484_anticopy_rp110_bs2/identity_conflict/identity_conflict.json`.

### Standard One-Shot Sanity

| condition | rows | scoreable predictions | mAP | mRecall | parse status |
| --- | ---: | ---: | ---: | ---: | --- |
| step917 baseline one-shot val200 | 200 | 1250 | 0.4112 | 0.4789 | 199 accepted, 1 accepted_with_drops |
| E1 anti-copy checkpoint one-shot val200 | 200 | 258 | 0.1580 | 0.1655 | 193 accepted, 5 accepted_with_drops, 2 all_spans_dropped |

Direct observations:

- E1 anti-copy is not a healthy standard one-shot detector under the baseline
  `detect all objects` prompt.
- The primary symptom is enumeration/coverage collapse: 258 predictions versus
  1250 for the step917 baseline on the same val200 scope.
- This does not refute row-level visual designator control, but it does mean E1
  should be interpreted as a specialized row-level mechanism probe checkpoint,
  not as a replacement baseline for one-shot detection.

Artifact handles:

- Baseline metrics:
  `/data/CoordExp/outputs/painted_gt/predicted_ledger_baselines/painted-gt-standard-step917-val200-rp110/eval_detection/metrics.json`.
- E1 metrics:
  `/data/CoordExp/outputs/painted_gt/pvci_identity_conflict/inference/e1_standard_unpainted_val200/step484_anticopy_rp110_bs2/eval_detection/metrics.json`.

## Interpretation

Supported reading:

- There is strong row-level visual designator control in the wrong-object setup.
  When the text schedule asks for target `i` but the image mark points to object
  `j`, the model usually follows the marked object geometry.
- The effect is already present in the E0 stepwise baseline and remains present
  after E1 anti-copy training.
- E1 anti-copy mainly changes geometry tightness and standard one-shot
  enumeration behavior; it does not erase the wrong-mark designator route.

Alternative reading:

- In tight wrong-object controls, “marked object identity” and “rendered mark
  geometry” are not separable. The observed behavior could still be a strong
  visual-box-following route rather than an object-identity route.
- The E1 standard one-shot drop may be expected from row-level training rather
  than a general capability loss, but it still makes E1 unsuitable as a
  production one-shot detector without additional training or prompt/runtime
  adaptation.

Remaining uncertainty:

- Whether the model follows the object source behind a wrong mark when the
  rendered mark is intentionally corrupted away from object `j`.
- Whether the same identity steering persists in a self-prefix or next-object
  steering runtime where no GT target row is supplied.
- Whether a training objective can preserve standard one-shot enumeration while
  strengthening row-level visual designator control.

## Research Unit Closeout

Observed:

- Wrong-object tight marks dominate row-level geometry for both E0 and E1.
- E1 anti-copy keeps the wrong-mark steering rate near E0 but reduces geometric
  tightness.
- E1 anti-copy collapses standard one-shot val200 coverage and mAP relative to
  the production step917 baseline.

Supported:

- The current row-level decoder has a controllable visual designator route under
  teacher-prefix wrong-object conflict.
- The previous Step-2 anti-copy result should be narrowed: it shows a useful
  anti-copy/geometry correction capability, not by itself a full object-identity
  proof.

Not supported yet:

- A claim that the model follows latent object identity rather than rendered
  mark geometry in wrong-object controls.
- A claim that E1 is a generally improved detector.
- A claim that the same behavior holds under self-prefix, no-GT, production
  one-shot, or full human-annotation-style rollout.

Next decider:

- Run corrupted wrong-object controls where the row target is `i`, the marked
  source object is `j`, and the rendered mark is intentionally shifted/scaled
  away from `j`. Measure whether predictions move toward target `i`, source
  object `j`, or rendered mark geometry.

Promotion decision:

- Do not promote to docs/OpenSpec. Keep as non-normative research evidence.
  The result justifies the next corrupted wrong-object probe, not a stable
  architecture or training-policy change.
