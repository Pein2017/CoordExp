## 1. Trainer Objective Update

- [x] 1.1 Add Stage-1-style coord distribution terms (`soft_ce`, `w1`) to Stage-2 AB loss computation over supervised bbox coord slots.
- [x] 1.2 Include new terms in `loss/coord_reg` and train-step metric logging (`loss/coord_soft_ce`, `loss/coord_w1`).

## 2. Canonical Config Reweighting

- [x] 2.1 Downweight CIoU default in Stage-2 canonical base (`stage2_ab.bbox_ciou_weight`).
- [x] 2.2 Enable low-weight Stage-1 coord-distribution defaults in Stage-2 canonical base (`custom.coord_soft_ce_w1.soft_ce_weight`, `w1_weight`).
- [x] 2.3 Add explicit canonical prod overrides in `configs/stage2_ab/prod/*.yaml` for `bbox_ciou_weight`, `soft_ce_weight`, and `w1_weight` (all `0.2`).

## 3. Verification

- [x] 3.1 Add/adjust Stage-2 trainer unit test coverage for new loss terms.
- [x] 3.2 Run focused Stage-2 trainer tests.
