# rollout-matching-sft Specification (delta: shared coordination ownership and train-time eval transport)

## Purpose
Extend the rollout-matching Stage-2 contract so rollout-aligned DDP coordination and train-time eval transport participate in the same optimization direction as the active Stage-2 AB path.

## Requirements

## ADDED Requirements

### Requirement: Rollout-aligned distributed step coordination MUST use the shared Stage-2 coordination contract
When rollout-aligned Stage-2 runs under DDP, the trainer SHALL use the same explicit distributed-step coordination contract as the active Stage-2 path for bounded synchronization, rank-symmetric failure propagation, and optimizer-step train-log reduction ownership.

Normative behavior:
- Rollout-aligned readiness guards and optimizer-step train-log reduction MUST participate in shared Stage-2 coordination ownership rather than remain on an unrelated ad hoc coordination path.
- Any per-step distributed coordination in the rollout-aligned trainer MUST remain bounded and fail clearly rather than waiting indefinitely.
- This requirement preserves rollout-learning semantics; it changes coordination ownership, not the learning objective.

#### Scenario: Rollout-aligned trainer participates in the shared coordination seam
- **GIVEN** a rollout-aligned Stage-2 training run under DDP
- **WHEN** the trainer performs per-step distributed coordination or train-log reduction
- **THEN** those operations use the shared Stage-2 coordination contract
- **AND** the rollout-aligned path does not preserve an unrelated optimizer-step reduction ownership model.

### Requirement: Rollout-aligned train-time detection eval MUST allow rank-efficient transport without broadening its contract
When rollout-aligned train-time detection eval runs under DDP and rank 0 owns final COCO scoring, the transport path MUST allow non-zero ranks to avoid materializing the full combined payload solely for rank-0 scoring.

Normative behavior:
- Rank 0 MUST remain the owner of final heavy scoring and train-time summary emission.
- Train-time optimization MUST preserve the existing eval-step summary contract, including `eval/detection/mAP`.
- This requirement does NOT broaden eval-step into full offline evaluator artifact parity.

#### Scenario: Rollout-aligned eval keeps rank-0 summary semantics while reducing payload fan-in
- **GIVEN** rollout-aligned eval-step detection runs under DDP
- **AND** rank 0 is the owner of final COCO scoring
- **WHEN** predictions and GT data are coordinated across ranks
- **THEN** non-zero ranks are not required to materialize the full combined payload solely for rank-0 scoring
- **AND** rank 0 still emits the canonical eval-step summary metrics.
