# coord-aux-loss Specification

## Purpose
Define the canonical coord-token auxiliary supervision contract used in the current stack (`coord_soft_ce_w1`), including config surfaces, fail-fast guards, and logging expectations.

## Requirements
### Requirement: Canonical config surface is `coord_soft_ce_w1`
The system SHALL expose coord auxiliary supervision through `custom.coord_soft_ce_w1` with the following keys:
- `enabled` (bool; default `false`)
- `ce_weight` (float; default `0.0`)
- `soft_ce_weight` (float; default `1.0`)
- `w1_weight` (float; default `1.0`)
- `gate_weight` (float; default `1.0`)
- `text_gate_weight` (float; default `0.0`; reserved for Stage-1 profiles that explicitly supervise non-coord positions)
- `temperature` (float; default `1.0`)
- `target_sigma` (float; default `2.0`)
- `target_truncate` (int or null; default `null`)

Validation contract:
- All weights MUST be `>= 0`.
- `temperature` and `target_sigma` MUST be `> 0`.
- `target_truncate` MUST be `null` or an integer `>= 0`.
- If `enabled=true`, at least one of `ce_weight`, `soft_ce_weight`, `w1_weight`, `gate_weight`, `text_gate_weight` MUST be non-zero.

Legacy compatibility:
- `custom.coord_loss` is removed and SHALL fail fast.
- Stage-2 pipeline configs MUST NOT migrate coord auxiliary knobs into
  `coord_reg`; `coord_reg` is retired from active Stage-2 training.

#### Scenario: Invalid `coord_soft_ce_w1` values fail fast
- **GIVEN** `custom.coord_soft_ce_w1.enabled: true` and all loss weights set to `0`
- **WHEN** config validation runs
- **THEN** validation fails fast with actionable guidance.

#### Scenario: Deprecated `custom.coord_loss` does not define behavior
- **GIVEN** a config that still includes `custom.coord_loss`
- **WHEN** config validation runs
- **THEN** validation fails fast.

### Requirement: Pipeline-driven Stage-2 rejects custom coord aux config
When `stage2_rollout_correction.pipeline` is present,
`custom.coord_soft_ce_w1.*` SHALL be disallowed.

Normative behavior:
- Users MUST NOT express coord auxiliary knobs through the retired `coord_reg`
  objective module.
- If both a pipeline objective and `custom.coord_soft_ce_w1.*` are present,
  config validation MUST fail fast.

#### Scenario: Pipeline + custom coord config fails fast
- **GIVEN** `stage2_rollout_correction.pipeline` is configured
- **AND** `custom.coord_soft_ce_w1` is also configured
- **WHEN** config validation runs
- **THEN** validation fails fast.

### Requirement: Coord auxiliary loss is coord-token supervision (not geometry IoU loss)
The coord auxiliary objective SHALL supervise coord-token distributions using the coord vocabulary (`<|coord_0|>`..`<|coord_999|>`).

Normative behavior:
- The soft target distribution and W1 terms SHALL be computed through the shared `coord_soft_ce_w1` helper.
- Optional hard CE and coord-vocab gate contributions SHALL be included according to configured weights.
- Optional non-coord `text_gate` contributions MAY also be included for profiles that explicitly enable `text_gate_weight`.
- The auxiliary objective SHALL apply at coord-token positions only.
- This capability SHALL NOT define bbox/poly IoU geometry losses; decoded-box
  geometry regularizers are outside the active training objective set.

#### Scenario: Coord-token positions receive softCE/W1 supervision
- **GIVEN** coord-token targets are present in the batch
- **AND** `custom.coord_soft_ce_w1.enabled: true`
- **WHEN** loss is computed
- **THEN** coord-token supervision includes weighted `soft_ce` and `w1` contributions (plus optional CE/gate terms).

### Requirement: Runtime prerequisites are fail-fast when enabled
If coord auxiliary supervision is enabled, missing runtime prerequisites MUST fail fast.

Fail-fast examples:
- missing/invalid model logits for the active forward pass,
- missing coord-token ids from tokenizer vocab,
- coord-token ids outside model vocab size,
- inability to build coord id map.

No-op behavior:
- If enabled but the current batch has zero supervised coord-token positions, the aux term MAY be skipped for that batch (no additive loss contribution).

#### Scenario: Missing coord vocab fails fast
- **GIVEN** `coord_soft_ce_w1` is enabled
- **AND** tokenizer does not provide coord token ids
- **WHEN** loss computation runs
- **THEN** training raises a fail-fast runtime error with actionable diagnostics.

### Requirement: Logging keys are stable for coord aux observability
When coord auxiliary supervision contributes to loss, the trainer SHALL emit stable metric keys:
- `coord_softce_w1/loss`
- `coord_softce_w1/soft_ce`
- `coord_softce_w1/w1`
- `coord_softce_w1/ce` (when hard CE is active)
- `coord_softce_w1/gate` (when gate is active)
- `coord_softce_w1/text_gate` (when text gate is active)

The trainer SHALL NOT use the retired `coord_diag/*` namespace for active
coord-aux objective contracts. That namespace remains historical only.

#### Scenario: Enabled coord aux emits stable metric namespaces
- **GIVEN** a training step with coord-token supervision active
- **WHEN** metrics are logged
- **THEN** logs include `coord_softce_w1/*` keys for the computed terms.
