# rollout-matching-sft Specification (Delta)

## ADDED Requirements

### Requirement: Rollout-aligned monitor integration reuses existing objective-pipeline atoms
Rollout-aligned Stage-2 gradient monitoring SHALL reuse the additive coord/geo tensors already produced by the rollout teacher-forcing objective pipeline.

Normative behavior:
- Monitor terms MUST be sourced from `pipeline_result.state` contrib tensors and objective-spec weights.
- The monitor MUST NOT introduce a second packed-token position gatherer for rollout-aligned packed sequences.

#### Scenario: Packed rollout-aligned forward uses pipeline atoms directly
- **WHEN** a packed rollout-aligned forward produces `bbox_*_contrib` and `coord_*_contrib` tensors
- **THEN** the monitor uses those tensors directly
- **AND** no duplicate packed-span reconstruction is required.

### Requirement: Rollout-aligned monitor payloads are buffered locally and synchronized at step log time
Rollout-aligned Stage-2 gradient-monitor payloads SHALL align with the current `PendingTrainRolloutLog` reducer path.

Normative behavior:
- Each rank computes monitor scalars locally for the packed forwards it executes.
- `PendingTrainRolloutLog` MUST carry the resulting mean-like `gradmon/*` scalars in addition to existing objective atoms / timing data.
- `_reduce_train_rollout_log_payload_global(...)` MUST synchronize the optimizer-step monitor payload across ranks.

#### Scenario: Multi-rank rollout-aligned run reduces monitor payload at log time
- **WHEN** a multi-rank rollout-aligned training step emits local `gradmon/*` values
- **THEN** each rank buffers them locally for the step
- **AND** `_reduce_train_rollout_log_payload_global(...)` performs the cross-rank synchronization when the step log is emitted.
