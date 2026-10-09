## Purpose

Improve cache affinity for context-management maintenance while preserving provider compatibility and opaque checkpoint correctness, without promising backend cache hit rates.

## ADDED Requirements

### Requirement: Compaction carries compatible session affinity
Context-management compaction SHALL preserve existing cache behavior by default. When the experimental boolean `preserveCacheAffinity` setting is explicitly enabled, it SHALL forward the originating session identity and defer retention selection to the provider's existing default and resolved environment instead of forcing no affinity. An absent or false setting and other compaction protocols MUST preserve their existing behavior. Provider authentication, supplied model configuration and cache-mode compatibility MUST remain authoritative; cache warming controls MUST NOT be interpreted as cache disabling.

#### Scenario: Compatible ordinary and compact request
- **WHEN** a session explicitly enables affinity preservation and starts context-management compaction
- **THEN** the provider receives its originating session identity with provider-default retention selection, and final diagnostics disclose the actually emitted cache fields

#### Scenario: Absent or disabled experimental setting
- **WHEN** affinity preservation is absent or explicitly false
- **THEN** compact requests retain the original no-affinity behavior

#### Scenario: Another protocol
- **WHEN** a different compaction protocol is selected
- **THEN** enabling the setting does not change that protocol's existing request contract

### Requirement: Prefix transitions are qualified without private coupling
Compaction SHALL retain authoritative system and ordered tool declarations, current projected history and compatible opaque checkpoint expansion. Tests and evidence MUST distinguish cache-affinity changes from model-visible prefix changes, including ephemeral context differences and intentional maintenance instructions. The implementation MUST NOT read another extension's private state, invoke its private hooks or freeze recalled memory to manufacture prefix reuse. Any history-preparation repair in this change MUST be confined to the opt-in context-management path and preserve default-off behavior; persisted instructions, forced effective prompts, legacy history and ordered tool deltas MUST remain authoritative.

#### Scenario: Maintenance introduces a new prefix epoch
- **WHEN** a maintenance request differs from ordinary inference or installs a new checkpoint
- **THEN** qualification identifies the preserved regions and first changed region, and subsequent compatible replay preserves its new prefix and ordered tail

### Requirement: Checkpoint and task behavior survives optimization
The optimized path MUST preserve bounded opaque artifacts, exact validated replay, persisted reload, repeated compaction, cancellation and automatic task continuation without duplicating completed tools or starting extra turns. Backend cache improvement SHALL be claimed only from a bounded hosted contrast with actual reported usage; deterministic tests alone SHALL NOT establish that claim.

#### Scenario: Reload and continuation
- **WHEN** an optimized compact request publishes a checkpoint and the session is reopened
- **THEN** compatible replay restores the checkpoint and newer input exactly once without plaintext substitution or repeated tools

#### Scenario: Hosted failure
- **WHEN** the bounded hosted qualification encounters an external or entitlement failure
- **THEN** the run stops, retains its receipt and reports hosted cache benefit as unverified without automatic retry or protocol substitution
