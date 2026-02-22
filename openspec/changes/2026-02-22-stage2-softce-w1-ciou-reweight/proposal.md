## Why

Stage-2 AB currently relies on bbox SmoothL1+CIoU plus optional regularizers, but it does not include Stage-1-style coord-distribution penalties (`soft_ce`, `w1`) in the Stage-2 objective.

Recent A-only diagnostics show coordinate calibration drift while CIoU remains dominant, so we need:
- direct coord-distribution supervision in Stage-2 on supervised bbox slots,
- and a less CIoU-heavy default balance in canonical Stage-2 configs.

## What Changes

- Stage-2 AB trainer computes and logs `soft_ce` + `w1` coord-distribution terms on Stage-2-supervised coord slots (`bbox_groups_prefix` + `bbox_groups_fn`) and adds them into `loss/coord_reg`.
- Stage-2 trainer emits `loss/coord_soft_ce` and `loss/coord_w1`.
- Canonical Stage-2 base config (`configs/stage2_ab/base.yaml`) is reweighted:
  - `stage2_ab.bbox_ciou_weight: 0.5` (downweighted from implicit/default 1.0),
  - `custom.coord_soft_ce_w1.soft_ce_weight: 0.02`,
  - `custom.coord_soft_ce_w1.w1_weight: 0.02`.
- Canonical Stage-2 prod leaves (`configs/stage2_ab/prod/*.yaml`) explicitly override loss weights for production runs:
  - `stage2_ab.bbox_ciou_weight: 0.2`,
  - `custom.coord_soft_ce_w1.soft_ce_weight: 0.2`,
  - `custom.coord_soft_ce_w1.w1_weight: 0.2`.

## Impact

- Affected trainer: `src/trainers/stage2_ab_training.py`.
- Affected canonical Stage-2 config surfaces:
  - `configs/stage2_ab/base.yaml` (base/smoke defaults),
  - `configs/stage2_ab/prod/*.yaml` (explicit prod overrides).
- Validation: extend Stage-2 trainer tests to assert new terms are active and logged.
