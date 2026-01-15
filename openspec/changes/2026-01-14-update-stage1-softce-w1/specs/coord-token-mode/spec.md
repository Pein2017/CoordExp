## ADDED Requirements

### Requirement: Coord-gated soft-label losses for Stage-1
When an opt-in Stage-1 coord-loss mode is enabled, the system SHALL support computing coordinate supervision strictly at `<|coord_*|>` token positions by:
- restricting logits to the coord vocab (coord-vocab gate) at those positions, and
- applying a per-token loss of `softCE(Gaussian kernel) + 1D W1(CDF)` against a unimodal soft target over bins.

The system SHALL apply standard full-vocab CE only to non-coord tokens (text + JSON structure) under this mode, to avoid double-supervision at coord positions.

#### Scenario: Coord tokens supervised without full-vocab CE
- **GIVEN** coord-token mode is enabled and a training sample contains `<|coord_k|>` tokens in the assistant JSON
- **AND** Stage-1 coord-gated `softCE+W1` mode is enabled
- **WHEN** the training step computes losses
- **THEN** non-coord tokens contribute to the standard CE loss
- **AND** coord-token positions contribute only to the coord-gated `softCE+W1` loss
- **AND** logits at coord-token positions are restricted to the coord vocab for the coord loss computation.

### Requirement: Single-forward Stage-1 loss composition
When Stage-1 coord-gated `softCE+W1` mode is enabled, the system MUST compute:
- the non-coord CE loss and
- the coord `softCE+W1` loss
from a **single** model forward pass.

The system SHALL ensure coord-token targets do not contribute to the base full-vocab CE loss by masking coord targets to `ignore_index` (or an equivalent mechanism with zero gradient), while still using the same forward logits to compute coord losses.

#### Scenario: No second forward is required
- **GIVEN** a training batch with at least one supervised coord token position
- **AND** Stage-1 coord-gated `softCE+W1` mode is enabled
- **WHEN** `Trainer.compute_loss` is called
- **THEN** the model forward is executed exactly once for that batch
- **AND** the base CE loss ignores coord-token targets
- **AND** coord `softCE+W1` is computed from the same forward logits.

### Requirement: Unimodal soft targets and Wasserstein-1 via CDF
The system SHALL provide utilities to build unimodal soft targets over ordered coord bins and compute 1D Wasserstein-1 loss via CDF differences so that:
- `W1(p, q) = 0` when `p == q`, and
- the loss increases when probability mass shifts farther away in bin order.

#### Scenario: W1 is zero for identical distributions
- **GIVEN** two identical normalized distributions over coord bins
- **WHEN** W1(CDF) is computed
- **THEN** the result is exactly zero (up to numerical tolerance).

## MODIFIED Requirements

### Requirement: Coord-token supervision is distributional (legacy removed)
The system SHALL treat coord-token supervision as a **distributional** token-level problem:
- `<|coord_*|>` positions are supervised via coord-gated `softCE(Gaussian kernel) + 1D W1(CDF)`,
- non-coord tokens remain supervised with the model's standard full-vocab CE,
- expectation-decoding and box-level regression losses are not supported.

The system SHALL fail fast with a clear error if legacy config keys are provided (e.g. `custom.coord_loss` or `custom.coord_expectation_metrics`), directing users to `custom.coord_soft_ce_w1`.

#### Scenario: Legacy coord-loss config rejected
- **GIVEN** a YAML config that provides `custom.coord_loss` or `custom.coord_expectation_metrics`
- **WHEN** the training config is loaded
- **THEN** the load fails with a clear error message
- **AND** the error indicates the supported replacement is `custom.coord_soft_ce_w1`.

## REMOVED Requirements

### Requirement: Expectation-decoded coordinate regression losses
**Reason**: Stage-1 retraining uses token-native distribution supervision; expectation decoding and box-level losses are removed for simplicity and stability.

**Migration**: Replace `custom.coord_loss` / `custom.coord_expectation_metrics` with `custom.coord_soft_ce_w1` and retrain Stage-1 using the single-forward distribution loss mode.
