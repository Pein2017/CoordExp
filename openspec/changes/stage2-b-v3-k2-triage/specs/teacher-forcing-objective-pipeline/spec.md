# teacher-forcing-objective-pipeline Specification (Delta)

## MODIFIED Requirements

### Requirement: duplicate_ul remains the canonical B-only suppression module
The teacher-forcing objective pipeline SHALL continue to use `duplicate_ul` as the canonical Channel-B local suppression module for the v3 contract.

Normative behavior:

- `duplicate_ul` remains a valid objective module name,
- `duplicate_ul` MUST continue to declare `channels: [B]`,
- `duplicate_ul.config` MUST remain `{}` in v1,
- the runtime metadata consumed by `duplicate_ul` MAY now encode any dead anchor-side continuation chosen by the v3 triage stage, not only same-desc duplicate bursts.

#### Scenario: duplicate_ul accepts dead-anchor continuation metadata
- **WHEN** Channel-B v3 provides canonical dead-anchor first-divergence targets
- **THEN** the `duplicate_ul` module may consume them without requiring a new module name
- **AND** pipeline validation still treats `duplicate_ul` as the canonical B-only suppression module.

### Requirement: Objective-module prerequisites remain strict
The strict objective pipeline SHALL fail fast if the runtime context does not provide the canonical metadata required by the configured suppression module.

Normative behavior:

- enabling `duplicate_ul` without the canonical per-segment target metadata MUST fail fast,
- silent fallback from dead-anchor UL to “no suppression” is forbidden once `duplicate_ul` is configured.

#### Scenario: Missing dead-anchor UL metadata fails fast
- **WHEN** a Stage-2 AB v3 pipeline enables `duplicate_ul`
- **AND** the runtime context omits the canonical suppression targets for a rollout segment
- **THEN** the training step raises with actionable diagnostics
- **AND** training does not proceed with a silently altered objective.
