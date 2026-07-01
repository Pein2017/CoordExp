# stage2-ab-training Specification (Delta)

## ADDED Requirements

### Requirement: Stage-2 AB monitor integration reuses existing atomic loss tensors
Stage-2 AB gradient monitoring SHALL reuse the additive coord/geo tensors already produced by the Channel-A / Channel-B objective construction path.

Normative behavior:
- Channel-A monitor terms MUST come from:
  - `A2_coord/*` tensors exposed by the self-context pipeline state, and
  - `A1_coord/*` tensors exposed by the existing A1 helper path when those anchor terms are enabled.
- Channel-B monitor terms MUST come from the rollout-context pipeline state.
- The monitor MUST NOT add a second packed-token position gatherer for coord/text spans.

#### Scenario: Channel-B packed forward reuses rollout-context atoms
- **WHEN** a packed Channel-B forward exposes `bbox_*_contrib` and `coord_*_contrib` tensors
- **THEN** the monitor uses those tensors directly
- **AND** it does not rebuild monitor terms by re-parsing the packed sequence.

### Requirement: Stage-2 AB monitor payloads follow the step-budgeted packed reducer contract
Stage-2 AB gradient-monitor payloads SHALL align with the current step-budgeted packed execution model.

Normative behavior:
- Local monitor scalars from each packed forward are added to `_PendingStage2Log`.
- `_PendingStage2Log` MUST treat `gradmon/*` keys as mean-like values.
- Cross-rank synchronization MUST happen through `_reduce_stage2_pending_metrics_global(...)` at the optimizer-step log boundary.
- If monitor execution is gated to the final synchronized pack, that gate MUST follow the executor’s existing sync decision rather than assuming `accelerator.sync_gradients` is authoritative.

#### Scenario: Stage-2 AB DDP monitor payload is reduced at the step boundary
- **WHEN** a multi-rank Stage-2 AB step emits local `gradmon/*` scalars
- **THEN** each rank buffers those values locally in `_PendingStage2Log`
- **AND** the global train log is produced only after `_reduce_stage2_pending_metrics_global(...)` synchronizes the step payload.
