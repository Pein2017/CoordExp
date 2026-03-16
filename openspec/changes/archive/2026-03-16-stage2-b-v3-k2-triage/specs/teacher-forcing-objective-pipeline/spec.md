# teacher-forcing-objective-pipeline Specification (Delta)

## ADDED Requirements

### Requirement: loss_dead_anchor_suppression remains the canonical B-only suppression module
The teacher-forcing objective pipeline SHALL continue to use `loss_dead_anchor_suppression` as the canonical Channel-B local suppression module for the v3 contract.

Normative behavior:

- `loss_dead_anchor_suppression` remains a valid objective module name,
- `loss_dead_anchor_suppression` MUST continue to declare `channels: [B]`,
- `loss_dead_anchor_suppression.config` MUST remain `{}` in v1,
- the runtime metadata consumed by `loss_dead_anchor_suppression` MAY now encode any dead anchor-side continuation chosen by the v3 triage stage, not only same-desc duplicate bursts.

#### Scenario: loss_dead_anchor_suppression accepts dead-anchor continuation metadata
- **WHEN** Channel-B v3 provides canonical dead-anchor first-divergence targets
- **THEN** the `loss_dead_anchor_suppression` module may consume them without requiring a new module name
- **AND** pipeline validation still treats `loss_dead_anchor_suppression` as the canonical B-only suppression module.

### Requirement: Objective-module prerequisites remain strict
The strict objective pipeline SHALL fail fast if the runtime context does not provide the canonical metadata required by the configured suppression module.

Normative behavior:

- enabling `loss_dead_anchor_suppression` without the canonical per-segment target metadata MUST fail fast,
- silent fallback from dead-anchor UL to “no suppression” is forbidden once `loss_dead_anchor_suppression` is configured.

#### Scenario: Missing dead-anchor UL metadata fails fast
- **WHEN** a Stage-2 AB v3 pipeline enables `loss_dead_anchor_suppression`
- **AND** the runtime context omits the canonical suppression targets for a rollout segment
- **THEN** the training step raises with actionable diagnostics
- **AND** training does not proceed with a silently altered objective.
