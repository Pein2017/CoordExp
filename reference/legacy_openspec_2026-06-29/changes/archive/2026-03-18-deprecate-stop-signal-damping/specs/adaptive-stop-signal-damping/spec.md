# adaptive-stop-signal-damping Spec Delta

This is a delta spec for archived change `2026-03-18-deprecate-stop-signal-damping`.

## REMOVED Requirements

### Requirement: Adaptive stop-signal damping is a supported authored-YAML experiment
The system MUST reject authored `token_ce.config.stop_signal_damping`
configuration with guidance to delete it, because the experiment is removed
from the supported training surface.

#### Scenario: Authored stop-signal damping config fails fast
- **WHEN** a Stage-2 or rollout-aligned pipeline authors `token_ce.config.stop_signal_damping`
- **THEN** configuration or direct token-CE runtime entry fails fast
- **AND** the error tells the operator to remove the deprecated block.

### Requirement: Active training emits stop-signal objective atoms and diagnostics
The live training contract MUST NOT emit `stop_signal_ce` objective atoms or
`stop_signal/*` diagnostics.

#### Scenario: Deprecation removes live stop-signal metrics
- **WHEN** training runs after this rollback lands
- **THEN** active logs do not contain `loss/*/stop_signal_ce`
- **AND** active logs do not contain `stop_signal/*` diagnostics.
