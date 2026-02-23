# teacher-forcing-objective-pipeline Specification (Delta)

## ADDED Requirements

### Requirement: Pipeline modules consume a teacher-forcing context contract
The system SHALL execute pipeline modules against a shared `TeacherForcingContext` contract produced by the
trainer/channel forward strategy.

Normative behavior:
- The context MUST carry the minimum data required to compute losses/metrics without re-running model forward passes.
- The context MUST be packing-safe: when sequences are packed, the context MUST preserve segment boundaries and provide
  segment-local metadata so modules can compute masks/indices without leaking across segments.
- The context SHOULD expose *derived* standardized views (e.g., segment boundaries, token-type masks, coord supervision
  groups) so modules do not need to interpret trainer-specific raw meta keys. Module implementations MUST NOT rely on
  ad-hoc per-trainer meta formats when a standardized derived view is available.
- Objective modules MUST treat missing prerequisites as a hard error (fail-fast).
- Diagnostics modules MUST be best-effort (warn once and skip on unexpected failures).

Notes:
- This capability defines pipeline execution mechanics and module contract shapes.
- Canonical loss component names, token types/contexts, and Stage-1/Stage-2 mask semantics are defined by the
  `teacher-forcing-unified-loss-registry` capability.

### Requirement: Teacher-forcing objective is declared as an ordered YAML pipeline
The system SHALL support a YAML-declared module pipeline for teacher-forcing training objectives and diagnostics.

Normative configuration shape (conceptual):
- A pipeline definition contains two ordered lists:
  - `objective`: loss-contributing modules
  - `diagnostics`: metrics-only modules
- Each module entry MUST include:
  - `name` (string; registry key)
- Each module entry MAY include:
  - `enabled` (boolean; default true)
  - `weight` (float; default 1.0)
  - `channels` (list containing `"A"` and/or `"B"`; default both)
  - `config` (mapping; module-specific knobs)

Execution semantics:
- Modules MUST execute in list order.
- Disabled modules MUST be skipped.
- A module list MUST NOT contain duplicate `name` entries (fail fast).
- Module `config` payloads MUST be validated strictly by the owning module implementation at trainer initialization:
  - unknown config keys MUST fail fast with actionable diagnostics,
  - missing optional keys MUST resolve to documented defaults.

#### Scenario: Ordered pipeline executes deterministically
- **WHEN** a pipeline defines objective modules `[m1, m2, m3]` in that order
- **THEN** the system executes `m1` before `m2` before `m3`
- **AND** any per-step metrics emitted by those modules reflect the same order of execution.

#### Scenario: Duplicate module names fail fast
- **WHEN** a pipeline contains two entries with the same `name`
- **THEN** config validation fails fast with guidance to deduplicate the list.

#### Scenario: Unknown module config key fails fast
- **WHEN** a pipeline provides a module `config` containing an unknown key for that module
- **THEN** config validation fails fast with guidance listing allowed keys for the module.


### Requirement: Module registry is strict and validated before training starts
The system SHALL resolve module names via a strict registry.

Normative behavior:
- Unknown module names MUST fail fast before the first training step.
- Registry resolution MUST be deterministic and MUST NOT depend on runtime reflection of unrelated modules.

#### Scenario: Unknown module name fails fast
- **WHEN** a pipeline references an unknown module `name`
- **THEN** training initialization fails fast
- **AND** the error message lists the unknown name and available module names.


### Requirement: Objective modules are fail-fast and diagnostics modules are best-effort
The system SHALL separate objective-changing modules from diagnostics-only modules.

Normative behavior:
- If an enabled objective module cannot be computed (missing prerequisites, misalignment, invalid config), the system
  MUST raise an error and MUST NOT silently change the training objective.
- If a diagnostics module fails unexpectedly, the system MUST continue training and SHOULD emit a warning once per
  diagnostics module name.

#### Scenario: Objective failure raises
- **WHEN** an enabled objective module encounters a missing prerequisite during a training step
- **THEN** the training step raises with actionable diagnostics
- **AND** training does not proceed with a silently altered objective.

#### Scenario: Diagnostics failure does not block training
- **WHEN** a diagnostics module throws an unexpected exception during a training step
- **THEN** training continues
- **AND** a warning is emitted indicating the diagnostics module failed.


### Requirement: Modules can be scoped to specific channels
The system SHALL support channel applicability for modules where training has multiple channels (e.g., Channel-A vs
Channel-B).

Normative behavior:
- If a module declares `channels: ["A"]`, it MUST execute only on Channel-A steps.
- If a module declares `channels: ["B"]`, it MUST execute only on Channel-B steps.
- If `channels` is omitted, the module MUST execute on both channels.

#### Scenario: Channel-scoped module is skipped on other channels
- **WHEN** a module declares `channels: ["A"]`
- **AND** the current training step is Channel-B
- **THEN** the module is skipped
- **AND** it contributes neither loss nor metrics for that step.


### Requirement: Pipeline identity is recorded for reproducibility
The system SHALL make the resolved pipeline auditable from logs and artifacts.

Normative behavior:
- The trainer MUST log the resolved ordered list of objective modules and diagnostics modules at initialization.
- The trainer MUST emit a stable pipeline checksum (derived from module names + enabled flags + module configs).

Normative checksum definition (this repo; required for implementers):
- The pipeline checksum MUST be the hex digest of `sha256` over UTF-8 bytes of a **canonical JSON** serialization of a
  fully-resolved pipeline identity object.
- The identity object MUST include, at minimum:
  - `objective`: ordered list of resolved module identity entries,
  - `diagnostics`: ordered list of resolved module identity entries,
  - `extra`: mapping (default `{}`) for trainer-specific identity fields that affect objective/metrics semantics (e.g.,
    ST bridge modes). If `extra` is used, it MUST be included in the checksum input.
- Each resolved module identity entry MUST be normalized before checksum:
  - `name: str`
  - `enabled: bool`
  - `weight: float`
  - `channels: list[str]` (if omitted in config, normalize to `["A","B"]`; if provided, normalize ordering as
    `["A","B"]` filtered to those present)
  - `config: mapping` (module-resolved config with defaults applied; unknown keys are already rejected)
- Canonical JSON requirements:
  - object keys MUST be serialized with lexicographic key ordering (`sort_keys=true`),
  - list ordering MUST be preserved (execution order is semantically meaningful),
  - serialization MUST be whitespace-free (`separators=(",", ":")`),
  - floats MUST be finite (fail fast on NaN/Inf) and MUST normalize `-0.0` to `0.0` before serialization.

#### Scenario: Pipeline checksum is logged
- **WHEN** training starts with a resolved module pipeline
- **THEN** logs include a stable pipeline checksum
- **AND** the checksum is identical across runs when the pipeline config and code version are identical.
