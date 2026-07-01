## ADDED Requirements

### Requirement: Stage-2 AB objective includes coord soft-CE and W1 terms on supervised bbox slots
Stage-2 AB trainer MUST support Stage-1-style coord distribution penalties in Stage-2 training:

- `soft_ce` and `w1` MUST be computed on coord distributions for Stage-2-supervised bbox coord slots only:
  - matched-prefix groups (`bbox_groups_prefix`),
  - and FN-injected groups (`bbox_groups_fn`).
- The coord distribution for each supervised coord slot MUST follow the same causal shift contract as other Stage-2 coord losses (coord token at position `p` uses logits at `p-1`).
- These terms MUST be aggregated into Stage-2 coord regularization (`loss/coord_reg`) and surfaced as train metrics:
  - `loss/coord_soft_ce`,
  - `loss/coord_w1`.
- The trainer MUST NOT apply these terms to unsupervised FP-only coord slots.

Weighting/config contract:
- Stage-2 uses `custom.coord_soft_ce_w1` as the config source for `soft_ce_weight`, `w1_weight`, `target_sigma`, `target_truncate`, and `temperature` when enabled.
- If `custom.coord_soft_ce_w1.enabled` is false, Stage-2 soft-CE/W1 contributions MUST be zero.

#### Scenario: Enabled coord soft-CE/W1 increases Stage-2 coord regularization
- **GIVEN** Stage-2 config has `custom.coord_soft_ce_w1.enabled: true` with non-zero `soft_ce_weight` and `w1_weight`
- **AND** a batch has supervised bbox coord slots
- **WHEN** Stage-2 computes loss
- **THEN** `loss/coord_soft_ce` and `loss/coord_w1` are positive
- **AND** both contribute to `loss/coord_reg`.

### Requirement: Canonical Stage-2 base and prod leaves declare CIoU/soft-CE/W1 weights explicitly
The canonical Stage-2 AB config surfaces MUST declare CIoU and coord-distribution weights explicitly to avoid ambiguity between inherited defaults and production-tuned overrides.

Canonical base defaults:
- `stage2_ab.bbox_smoothl1_weight: 2.0`
- `stage2_ab.bbox_ciou_weight: 0.5`

Canonical base MUST also set:
- `custom.coord_soft_ce_w1.enabled: true`
- `custom.coord_soft_ce_w1.soft_ce_weight: 0.02`
- `custom.coord_soft_ce_w1.w1_weight: 0.02`

Canonical prod overrides:
- `stage2_ab.bbox_ciou_weight: 0.2`
- `custom.coord_soft_ce_w1.soft_ce_weight: 0.2`
- `custom.coord_soft_ce_w1.w1_weight: 0.2`

#### Scenario: Canonical prod leaves pin explicit CIoU/soft-CE/W1 overrides
- **GIVEN** a canonical Stage-2 profile leaf under `configs/stage2_ab/prod/*.yaml`
- **WHEN** config is materialized through one-hop inheritance from `../base.yaml`
- **THEN** the leaf explicitly overrides effective Stage-2 loss weights with the canonical prod values above.

#### Scenario: Canonical smoke leaves inherit base CIoU/soft-CE/W1 defaults
- **GIVEN** a canonical Stage-2 profile leaf under `configs/stage2_ab/smoke/*.yaml`
- **WHEN** config is materialized through one-hop inheritance from `../base.yaml`
- **THEN** effective Stage-2 loss defaults include canonical base CIoU downweight and non-zero soft-CE/W1 terms.
