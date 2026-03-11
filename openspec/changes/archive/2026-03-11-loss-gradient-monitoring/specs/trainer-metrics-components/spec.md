# trainer-metrics-components Specification (Delta)

## MODIFIED Requirements

### Requirement: Stable metric and batch key names
The canonical metric docs SHALL use the current training metrics page and include the `gradmon/*` diagnostics introduced by this change.

Normative behavior:
- `docs/training/METRICS.md` MUST define the canonical training keys added by this contract.
- `docs/training/STAGE2_RUNBOOK.md` MUST define the corresponding packed-step / DDP interpretation where relevant.

#### Scenario: Gradient-monitor keys are documented on the canonical metrics page
- **WHEN** `gradmon/*` metrics are emitted
- **THEN** their canonical key names are documented in `docs/training/METRICS.md`
- **AND** stale references to the retired metrics doc path are absent from the live contract.

## ADDED Requirements

### Requirement: Gradient-monitor metrics are aggregated through the existing local-first reducer policy
The trainer metrics contract SHALL treat gradient-monitor diagnostics as local-first training metrics.

Normative behavior:
- `gradmon/*` scalars are computed locally on each rank first.
- During optimizer-step logging under DDP, `gradmon/*` keys MUST be synchronized through the same step-boundary reducers used by the current training metrics.
- `gradmon/*` keys MUST be treated as mean-like gauge values unless a future change explicitly introduces counter-like `gradmon/*_count|_total|_sum|_num|_den` keys.
- `time/gradmon_s` MUST follow the existing `time/*` reducer semantics for the active trainer.

#### Scenario: DDP monitor scalars follow the same reducer family rules as other training metrics
- **WHEN** a multi-rank training run emits `gradmon/*` and `time/gradmon_s`
- **THEN** each rank computes those values locally
- **AND** the step-boundary reducer synchronizes them using the trainer’s existing mean/max policy rather than a special-case monitor reducer.
