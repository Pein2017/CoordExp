## MODIFIED Requirements

### Requirement: Planned-Step Loss Normalizers

Loss normalizers SHALL be length-invariant over the planned optimizer-step
window. The protected token-wise reducer MUST be `segment_balanced`: compute
the mean loss over eligible atoms within each eligible segment, then the mean
over eligible segments in the complete planned optimizer-step window across
accumulation and ranks. Per-term denominators MUST be computed from the
complete planned-step window, not from rank-local windows or pack-local means
averaged afterward. In distributed training, rank-local denominator
contributions MUST be gathered before backward; local contributions MUST be
scaled to compensate Accelerate's mean gradient reduction so the effective
objective is globally planned-step balanced. Segments with zero eligible atoms
MUST be excluded from that term's denominator. Protected losses MUST fail if
the complete planned-step window has zero eligible segments for the term.
Runtime and loss code MUST avoid double scaling. The resulting global
denominators and effective scale MAY be included in the wide train logging row
or explicit diagnostics, but MUST NOT require rank-local durable receipt files.

#### Scenario: Different token counts across micro-steps

- **WHEN** two micro-steps in one planned optimizer step contain different
  numbers of supervised tokens
- **THEN** protected loss normalization MUST divide by the planned-step
  `segment_balanced` denominator for the selected term
- **AND** MUST NOT average two already-normalized micro-step losses equally.

#### Scenario: Unequal rank-local segment counts

- **WHEN** distributed ranks contribute unequal numbers of eligible segments
  to one planned optimizer step
- **THEN** all ranks MUST use the same all-rank `planned_step_global`
  denominator for that term
- **AND** the effective scale MUST remain inspectable through compact logging
  or explicit test diagnostics without per-rank artifact streams.

#### Scenario: Segment-balanced differs from token-balanced

- **WHEN** two segments in one planned step have unequal eligible atom counts
- **THEN** protected token-wise losses MUST weight the two segment means
  equally under `segment_balanced`
- **AND** a single global mean over all eligible atoms MAY be emitted only as a
  diagnostic metric, not as the protected objective.

### Requirement: Loss Bundle Metrics

`LossRunner` SHALL return a `LossBundle` containing total weighted loss,
weighted per-term loss metrics, top-level `acc_top1`, top-level `acc_top5`,
selected-count diagnostics, and finite-status diagnostics. Stored per-term loss
metrics MUST be weighted values. After all-rank reduction, the complete scalar
mapping for one planned step SHALL be written together in that step's wide
`train` row in `logging.jsonl`. Non-finite values MUST retain their field names
with JSON `null` values and MUST be listed in `non_finite_fields` rather than
making the planned step unwriteable.

#### Scenario: Train logging row emitted

- **WHEN** a train step completes
- **THEN** its single logging row MUST include weighted protected-loss metrics
  and top-level `acc_top1` and `acc_top5`
- **AND** top-1/top-5 names MUST NOT be nested under a base-CE namespace.

### Requirement: Non-Finite Loss And Gradient Gates

Training SHALL separate scalar loss finite checks before backward from
gradient/overflow checks after backward. Unsafe non-finite state MUST prevent a
corrupted optimizer update. Recoverable bad examples or warnings MAY be
reported without changing the planned-step schedule. In distributed execution,
the scalar finite check MUST produce one reduced all-rank decision before any
rank calls backward. The resulting update and finite status MUST be represented
once in the rank-zero train logging row for that planned step; normal training
MUST NOT write duplicate rank-local gate receipts.

#### Scenario: Non-finite scalar loss

- **WHEN** total loss is NaN or Inf before backward
- **THEN** runtime MUST reduce the unsafe scalar status across all ranks before
  any rank calls backward
- **AND** all ranks MUST skip backward and optimizer update for that planned
  step
- **AND** runtime MUST clear accumulated gradients and maintain planned-step
  scheduler/update policy
- **AND** rank zero MUST log the synchronized unsafe/update status once.
- **AND** non-finite scalar fields MUST be represented as JSON `null` and named
  in `non_finite_fields`.

#### Scenario: Distributed gradient overflow

- **WHEN** any rank reports unsafe gradient or overflow status
- **THEN** all ranks MUST use the same global skip/update decision for that
  planned step
- **AND** rank zero MUST log that global decision once.

#### Scenario: Zero eligible protected atoms on one rank

- **WHEN** a protected loss has zero eligible atoms on one rank
- **THEN** runtime MUST include the rank-local eligible count in the same
  planned-step all-rank denominator/finite decision before any rank raises or
  calls backward
- **AND** the global decision MUST avoid distributed deadlock.
