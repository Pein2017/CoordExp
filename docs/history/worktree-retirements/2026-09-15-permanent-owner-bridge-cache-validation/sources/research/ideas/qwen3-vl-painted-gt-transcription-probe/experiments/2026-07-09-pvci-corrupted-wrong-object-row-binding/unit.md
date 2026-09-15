---
title: PVCI Corrupted Wrong-Object Row Binding Probe
description: Tests whether a wrong-object visual mark binds an entire emitted row to the marked object, or merely supplies rendered geometry that can be copied.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-corrupted-wrong-object-row-binding
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - painted-gt
  - pvci
  - row-binding
updated: 2026-07-09
---

# PVCI Corrupted Wrong-Object Row Binding Probe

## Question

When the row schedule asks for object `i`, the visual mark source is object
`j`, and the rendered mark is intentionally corrupted away from object `j`,
does Qwen3-VL emit a coherent row for object `j`, copy the corrupted rendered
mark geometry, follow the scheduled target `i`, or produce a chimera row whose
phrase/class and geometry disagree?

This unit is the next decider after
[2026-07-09 PVCI Identity Conflict](../2026-07-09-pvci-identity-conflict/unit.md).
The previous tight wrong-object control showed that both the E0 stepwise
reference and E1 anti-copy checkpoint overwhelmingly follow the wrong visual
mark rather than the scheduled target. It could not separate marked-object
identity from rendered tight-box copying because `negative_bbox == mark_bbox`.

## Evidence Scope

- Checkout or branch: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`,
  `codex/qwen3-vl-painted-gt-transcription-probe`.
- Commit or diff scope: goal-scoped local research/probe additions for this
  unit only.
- Config:
  - E0:
    `configs/coordexp_swift/infer/painted_gt/pvci_row_binding/e0_wrong_object_val100_jitter_medium_step484_rp110_bs2.yaml`;
  - E1:
    `configs/coordexp_swift/infer/painted_gt/pvci_row_binding/e1_wrong_object_val100_jitter_medium_step484_rp110_bs2.yaml`.
- Checkpoints:
  - E0 row-level reference: the same stepwise painted checkpoint/surface used
    by the prior PVCI Step 0 wrong-object controls.
  - E1 anti-copy checkpoint:
    `/data/CoordExp/outputs/painted_gt/pvci_step2/train_snap_radius/painted_gt_pvci_step2_snap_radius_train256_anti_copy_mix_step484_warm_start_dora_all_towers_accelerate8_ebs8_16epoch-pvci-step2-anticopy-retry3-20260708T1726Z/checkpoints/step-484`.
- Input source:
  `/data/CoordExp/outputs/painted_gt/pvci_row_binding/materialized/heldout_val100_jitter_medium_wrong_object/conditions/stepwise__wrong_object_mark/stepwise.wrong_object_mark.examples.jsonl`.
- Materialization quality receipt:
  `/data/CoordExp/outputs/painted_gt/pvci_row_binding/materialized/heldout_val100_jitter_medium_wrong_object/materialization_quality.json`.
- Artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_row_binding/`.
- Commands:

```bash
python scripts/probes/painted_gt/materialize_counterfactual_conditions.py \
  --input-jsonl /data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl \
  --output-root /data/CoordExp/outputs/painted_gt/pvci_row_binding/materialized/heldout_val100_jitter_medium_wrong_object \
  --schedule-kind geo_sorted \
  --size 100 \
  --seed 20260704 \
  --schedule-seed 20260704 \
  --condition stepwise__wrong_object_mark \
  --mark-coarseness-variant jitter_medium_outline \
  --skip-preflight \
  --force

CUDA_VISIBLE_DEVICES=4 python -m src.infer \
  --config configs/coordexp_swift/infer/painted_gt/pvci_row_binding/e0_wrong_object_val100_jitter_medium_step484_rp110_bs2.yaml

CUDA_VISIBLE_DEVICES=5 python -m src.infer \
  --config configs/coordexp_swift/infer/painted_gt/pvci_row_binding/e1_wrong_object_val100_jitter_medium_step484_rp110_bs2.yaml

python scripts/probes/painted_gt/analyze_identity_conflict.py \
  --input /data/CoordExp/outputs/painted_gt/pvci_row_binding/inference/e0_wrong_object_val100_jitter_medium/step484_rp110_bs2/gt_vs_pred.jsonl \
  --output-dir /data/CoordExp/outputs/painted_gt/pvci_row_binding/inference/e0_wrong_object_val100_jitter_medium/step484_rp110_bs2/row_binding \
  --condition-name pvci_row_binding_e0_wrong_object_val100_jitter_medium_step484

python scripts/probes/painted_gt/analyze_identity_conflict.py \
  --input /data/CoordExp/outputs/painted_gt/pvci_row_binding/inference/e1_wrong_object_val100_jitter_medium/step484_anticopy_rp110_bs2/gt_vs_pred.jsonl \
  --output-dir /data/CoordExp/outputs/painted_gt/pvci_row_binding/inference/e1_wrong_object_val100_jitter_medium/step484_anticopy_rp110_bs2/row_binding \
  --condition-name pvci_row_binding_e1_wrong_object_val100_jitter_medium_step484_anticopy
```
- Metrics or counters:
  - prediction/parse rate;
  - phrase/class match to scheduled target `i`, wrong source object `j`, and
    any accidental third object `k`;
  - box IoU/L1 to scheduled target `i`, wrong source object `j`, rendered
    corrupted mark, and best third-object match;
  - row outcome counts: `target_row`, `source_object_row`,
    `rendered_mark_copy`, `third_object_row`, `chimera_row`,
    `malformed_or_no_prediction`;
  - chimera rate, especially `phrase=i geometry=j`,
    `phrase=j geometry=rendered_mark`, and `phrase=i geometry=rendered_mark`;
  - rendered-mark overlap with GT objects, so accidental third-object hits are
    not misread as source-object snapping.
- Sample window: held-out val100 row-level examples, `825` stepwise
  wrong-object rows under schedule `stepwise_geo_sorted_b22e7e1ee03e0ce8`.
  Report same-description, different-description, negative-selection reason,
  rendered-mark acceptance, and rendered third-overlap splits.
- Known limitations:
  - This is still a row-level teacher/schedule conflict probe, not autonomous
    selection, coverage, or STOP evaluation.
  - Pixel painting remains an oracle intervention. Internal cursor feasibility
    is a later unit.
  - If corruption places the rendered mark on a real third object, the result
    must be classified separately instead of counted as source-object snapping.

## Procedure

1. Materialize corrupted wrong-object examples where the assistant target and
   schedule remain object `i`, the mark source is object `j`, and the rendered
   mark is shifted/scaled away from object `j` through
   `mark_coarseness_variant: jitter_medium_outline`.
2. Preserve enough provenance per row to identify:
   scheduled target `i`, source object `j`, rendered mark geometry, all GT
   objects overlapped by the rendered mark, assistant prefix, and source image.
   This unit requires `candidate_objects[]` for every row; each candidate must
   carry object id, description, bbox bins, bbox pixels, category metadata when
   available, schedule position, and relation tags such as `target_i`,
   `source_j`, `previous_prefix`, `next_schedule`, and `third_candidate`.
3. Run deterministic HF inference with the same decode surface used by the
   prior row-level PVCI controls.
4. Analyze each generated row against the scheduled target, source object,
   rendered mark, and best third object.
5. Compare E0 and E1 where feasible; E0 tests whether the route already exists,
   E1 tests whether anti-copy training helps snap away from corrupted mark
   geometry.

Primary row eligibility before inference interpretation:

- `target_object_id != negative_object_id`;
- `marks[0].object_id == negative_object_id`;
- `candidate_objects[]` is present;
- rendered mark bbox exists and differs from `source_bbox_pixels`;
- `source_to_rendered_iou <= 0.95`;
- `source_to_rendered_l1 >= 1.0`;
- rendered mark IoU to scheduled target `i` is `<= 0.50`;
- rendered mark IoU to best third object `k` is `<= 0.50`.

Rows failing these checks stay in the secondary all-row summary but do not
define the clean primary conclusion. The materialization quality receipt must
report eligible count, rejection reasons, third-overlap buckets, same-description
counts, negative-selection reasons, and source-to-rendered IoU range.

Expected failure modes before inspecting results:

- `target_row`: phrase/class and geometry follow scheduled target `i`; visual
  mark loses under conflict.
- `source_object_row`: phrase/class and geometry follow wrong source object
  `j`; visual mark behaves as an object designator.
- `rendered_mark_copy`: geometry follows the corrupted rendered mark more than
  source object `j`; the model is copying paint geometry.
- `third_object_row`: the corrupted mark overlaps another GT object and the
  model follows that object.
- `chimera_row`: phrase/class and geometry resolve to different identities,
  meaning the cursor does not bind the whole row.
- `malformed_or_no_prediction`: parser failure, no row, or no usable geometry.

Analyzer outputs must keep phrase identity and geometry identity separate. In
same-description rows, phrase identity may be ambiguous and must not be used as
class-level evidence for full row binding.

## Observations

- Direct observation: the materialized held-out val100 panel preserved the prior
  slice and schedule (`painted_gt_slice_0a5e9d8587f99f95`,
  `stepwise_geo_sorted_b22e7e1ee03e0ce8`) and produced `825`
  wrong-object rows. Every row carried `candidate_objects[]`; `774 / 825`
  rows were clean primary rows under the declared separation gate.
- Materialization quality:
  - `eligible_primary_count = 774`;
  - `eligible_primary_rate = 0.9382`;
  - `rendered_differs_from_source = 815`;
  - `with_candidate_objects = 825`;
  - `same_description = 633`, `different_description = 192`;
  - source-to-rendered IoU mean `0.5264`, min `0.3868`, max `0.9980`.
- E0 row-level reference, overall:
  - prediction rate `0.8145`;
  - rendered-mark-copy candidate rate `0.9717`;
  - pointer/source candidate rate `0.0030`;
  - source-object row rate `0.0000`;
  - rendered-mark-copy row rate `0.9717`;
  - mean IoU to source object `j`: `0.5175`;
  - mean IoU to rendered mark: `0.9068`.
- E1 anti-copy checkpoint, overall:
  - prediction rate `0.8109`;
  - rendered-mark-copy candidate rate `0.1584`;
  - pointer/source candidate rate `0.7549`;
  - source-object row rate `0.0837`;
  - rendered-mark-copy row rate `0.1584`;
  - chimera row rate `0.0239`;
  - mean IoU to source object `j`: `0.6596`;
  - mean IoU to rendered mark: `0.5064`.
- Different-description split, the clearest phrase-binding subset:
  - E0: source-object row rate `0.0000`, rendered-mark-copy row rate
    `0.9041`, target-row rate `0.0685`;
  - E1: source-object row rate `0.3916`, rendered-mark-copy row rate
    `0.1399`, target-row rate `0.1119`, chimera row rate `0.0420`.
- Same-description rows remain phrase-ambiguous by construction. In those rows,
  E1 strongly shifts geometry toward source `j` (`pointer_candidate_rate =
  0.7719`) but the analyzer correctly cannot call most of them coherent
  source-object rows.
- Artifact handle:
  `/data/CoordExp/outputs/painted_gt/pvci_row_binding/analysis/row_binding_e0_e1_jitter_medium_summary.json`.

## Interpretation

- Supported reading: the previous tight wrong-object result was ambiguous for
  E0. Under corrupted wrong-object marks, E0 mostly copies or follows the
  rendered corrupted mark geometry rather than snapping back to source object
  `j`. This means the old E0 tight follow-mark result should not be called
  object-identity binding.
- Supported reading: E1 anti-copy training transfers to wrong-object conflict.
  It greatly reduces rendered-mark copying and shifts geometry toward the
  wrong source object `j`. On different-description rows, E1 produces coherent
  source-object rows at a substantial rate (`0.3916`) while keeping a modest
  chimera rate (`0.0420`).
- Alternative reading: E1 may still be following the visual object or region
  under/near the mark rather than a fully abstract object identity variable.
  This unit proves stronger row-level visual binding than E0, but it does not
  prove a deployable internal cursor, autonomous selector, coverage ledger, or
  STOP mechanism.
- Remaining uncertainty: whether this source-object binding can be reproduced
  by a non-pixel/internal cursor, and whether a mixed training recipe can keep
  one-shot enumeration while preserving the row-level actuator.

## Research Unit Closeout

Observed:

- Corrupted wrong-object materialization is valid enough for interpretation:
  `825` rows, `774` clean primary rows, all rows with candidate-object
  metadata.
- E0 is predominantly a rendered-mark-copy or rendered-region follower under
  corrupted wrong-object marks.
- E1 anti-copy substantially changes the mechanism: geometry usually moves
  toward the wrong source object `j` instead of the corrupted rendered mark.
- The strongest full-row binding evidence is in the different-description
  subset, where E1 reaches `0.3916` source-object row rate versus `0.0000`
  for E0.

Supported:

- Visual row-binding can be trained beyond pure rendered-box copying.
- E1 is a stronger row-level source-object actuator than E0 under this
  corrupted wrong-object control.
- Phrase/geometry chimera is real but not dominant in E1's
  different-description subset (`0.0420`).

Not supported yet:

- E1 is not a general detector or one-shot production baseline.
- Pixel painting is not yet replaced by an internal cursor.
- Selection, coverage/commit, and STOP are not tested by this unit.
- Same-description rows cannot support phrase-level binding claims because the
  phrase identity is intentionally ambiguous.

Next decider:

- Run the internal/non-pixel cursor feasibility unit. The immediate question is
  whether any feature-space, pseudo-visual-token, or hidden-state cursor can
  reproduce the E1 pixel-mark source-object binding behavior without modifying
  the input pixels.

Promotion decision:

Do not promote to docs/OpenSpec unless the result changes a stable training,
inference, artifact, metric, or config contract. This unit is currently
non-normative research evidence only.
