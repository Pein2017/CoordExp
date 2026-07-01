# loss-gradient-monitoring Specification (Delta)

## ADDED Requirements

### Requirement: Monitor terms reuse existing additive loss tensors
The loss-gradient monitor SHALL reuse additive scalar tensors already produced by the active loss path.

Normative behavior:
- Stage-1 MUST source coord-only monitor terms from `CoordSoftCEW1Result`.
- Stage-2 MUST source coord/geo monitor terms from teacher-forcing objective module state and the existing A1 helper path when applicable.
- The monitor MUST NOT introduce a second implementation that reconstructs coord/text supervision spans from packed `input_ids` or labels after the fact.

#### Scenario: Packed Stage-2 forward reuses module-provided atoms
- **WHEN** a packed Stage-2 forward produces `bbox_*_contrib` and `coord_*_contrib` tensors
- **THEN** the monitor uses those tensors directly as monitor terms
- **AND** it does not perform a second packed-token position scan.

### Requirement: Packed-sequence monitoring aggregates per-packed-forward diagnostics to the optimizer step
The monitor SHALL support packed-sequence training without changing the current step-budgeted execution model.

Normative behavior:
- For trainers that may execute multiple packed forwards in one optimizer step, the monitor computes diagnostics at packed-forward scope.
- Optimizer-step monitor values MUST be produced by aggregating those packed-forward diagnostics through the trainer’s existing pending-log buffer / reducer path.
- This change MUST NOT require reconstructing a single cross-pack gradient vector for the entire optimizer step.

#### Scenario: Multiple packed forwards contribute one step-level monitor payload
- **WHEN** a Stage-2 optimizer step executes multiple packed forwards
- **THEN** each packed forward may contribute local monitor scalars
- **AND** the logged optimizer-step monitor payload is the weighted aggregate produced by the existing step reducer.

### Requirement: DDP monitor aggregation is local-first and aligned with existing reducers
Under DDP, monitor metrics SHALL be computed locally on each rank first and synchronized only through the existing optimizer-step metric reducers.

Normative behavior:
- The monitor MUST NOT introduce a separate per-term all-reduce path inside the monitoring computation.
- Each rank computes local monitor scalars for its packed forwards / micro-steps.
- The trainer’s existing step-boundary metric reduction path performs cross-rank synchronization.

#### Scenario: DDP run uses current step-boundary reducer
- **WHEN** training runs with multiple ranks
- **THEN** each rank computes monitor scalars locally
- **AND** cross-rank synchronization happens only when the trainer finalizes and reduces the optimizer-step log payload.
