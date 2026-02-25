# rollout-matching-sft Specification (Delta)

## ADDED Requirements

### Requirement: Rollout-aligned Stage-2 uses explicit objective pipeline (no implicit defaults)
When `custom.trainer_variant: stage2_rollout_aligned`, the rollout-aligned teacher-forcing objective MUST be fully determined by the declared module pipeline under:
- `rollout_matching.pipeline.objective[]` and `rollout_matching.pipeline.diagnostics[]`.

Normative behavior:
- `rollout_matching.pipeline` MUST be present (no implicit default manifest).
- Legacy aux-loss config surfaces MUST be rejected, including `custom.coord_soft_ce_w1.*`.

#### Scenario: Missing rollout pipeline fails fast
- **WHEN** `custom.trainer_variant: stage2_rollout_aligned`
- **AND** `rollout_matching.pipeline` is absent
- **THEN** config loading fails fast before trainer init
- **AND** the error indicates `rollout_matching.pipeline` is required.

### Requirement: Rollout pipeline specs are explicit and complete (no implicit defaults)
Rollout pipeline module specs MUST be authored with explicit fields and complete module configs to prevent silent drift from default injection.

Normative behavior:
- Each entry in `rollout_matching.pipeline.objective[]` and `rollout_matching.pipeline.diagnostics[]` MUST include:
  - `name`, `enabled`, `weight`, `channels`, `config`.
- `channels` MUST be explicitly authored as a subset of `{A,B}`.
- `config` MUST include exactly the allowlisted keys for the referenced module:
  - missing required keys MUST fail fast (no implicit defaults),
  - unknown keys MUST fail fast (no escape-hatch aliases).

### Requirement: Rollout pipeline module configs are strict and canonical (no aliases)
Rollout pipeline module configs MUST be strict and MUST reject unknown keys and legacy alias keys.

Normative behavior:
- `bbox_geo.config` MUST accept only:
  - `smoothl1_weight`
  - `ciou_weight`
- `coord_reg.config` MUST accept only canonical keys, including:
  - `coord_ce_weight`
  - `soft_ce_weight`
  - `w1_weight`
  - `coord_gate_weight`
  - `text_gate_weight`
  - `temperature`
  - `target_sigma`
  - `target_truncate`
- Legacy alias keys (e.g., `bbox_smoothl1_weight`, `coord_soft_ce_weight`, `coord_w1_weight`) MUST be rejected.

#### Scenario: Alias key in rollout module config fails fast
- **WHEN** `rollout_matching.pipeline.objective[*].name=coord_reg`
- **AND** the module config contains `coord_soft_ce_weight`
- **THEN** configuration parsing fails fast
- **AND** the error indicates `soft_ce_weight` is the only accepted key.

### Requirement: Rollout-aligned Stage-2 supports text_gate via coord_reg module config
Rollout-aligned Stage-2 MUST support `text_gate` as part of `coord_reg` with a typed weight:
- `rollout_matching.pipeline.objective[*].config.text_gate_weight`

Normative behavior:
- `text_gate_weight > 0` MUST introduce a non-zero `text_gate` contribution when coord-vocab mass appears at supervised text positions under `context=rollout` (FP-neutral).

#### Scenario: FP-neutral text_gate is effective
- **WHEN** rollout-context supervision includes both FN spans and FP spans
- **AND** `text_gate_weight > 0`
- **THEN** FP spans do not contribute to the emitted `text_gate` objective atom `loss/B_coord/text_gate`
- **AND** FN text spans contribute to the `text_gate` sub-term and increase `loss/B_coord/text_gate` when they exhibit coord-vocab mass.
