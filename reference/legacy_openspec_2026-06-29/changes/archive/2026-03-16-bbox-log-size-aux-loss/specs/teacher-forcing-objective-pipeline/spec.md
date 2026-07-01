# teacher-forcing-objective-pipeline Specification (Delta)

## MODIFIED Requirements

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
- The context contract MUST remain sufficient for geometry modules to compute optional decoded-box size auxiliaries from
  the same predicted/target bbox groups already used for `bbox_geo`.

Notes:
- This capability defines pipeline execution mechanics and module contract shapes.
- Canonical loss component names, token types/contexts, and Stage-1/Stage-2 mask semantics are defined by the
  `teacher-forcing-unified-loss-registry` capability.

#### Scenario: Packed forwards preserve per-segment boundaries in context views
- **WHEN** packed sequences are processed through the teacher-forcing pipeline
- **THEN** derived masks and indices remain segment-local within `TeacherForcingContext`
- **AND** no module can supervise tokens across segment boundaries.

### Requirement: Teacher-forcing objective is declared as an ordered YAML pipeline
The teacher-forcing pipeline SHALL support the explicit `loss_dead_anchor_suppression` objective module for the canonical clean-prefix Channel-B contract.

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
- `loss_dead_anchor_suppression` is a valid objective module name.
- `bbox_size_aux` is a valid objective module name.
- `loss_dead_anchor_suppression` MUST be declared with `channels: [B]`.
- `loss_dead_anchor_suppression.config` MUST be validated strictly and MUST be `{}` in v1.
- For canonical Stage-2 AB clean-prefix configs, the ordered objective list MUST place `loss_dead_anchor_suppression` after `token_ce` and before `bbox_geo`.
- When `bbox_size_aux` is enabled, canonical Stage-2 objective order MUST place it after `bbox_geo` because
  `bbox_size_aux` depends on the decoded canonicalized box state produced on the same matched supervision path.
- `bbox_size_aux.config` MUST accept the canonical decoded-box size-aux keys:
  - `log_wh_weight`
  - `log_area_weight`
  - `oversize_penalty_weight`
  - `oversize_area_frac_threshold`
  - `oversize_log_w_threshold`
  - `oversize_log_h_threshold`
  - `eps`
  - `a1_log_wh_weight`
  - `a1_log_area_weight`
  - `a1_oversize_penalty_weight`
- the `bbox_size_aux` plugin/module MUST continue to consume the current
  coord-token decode path and current `bbox_2d=[x1,y1,x2,y2]` external
  expression rather than redefining bbox parameterization.

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

#### Scenario: bbox_size_aux accepts decoded-box size-aux config keys
- **WHEN** a teacher-forcing pipeline declares `bbox_size_aux.config.log_wh_weight`
- **AND** the config contains only canonical bbox-size-aux keys
- **THEN** pipeline validation succeeds for that module config shape
- **AND** the decoded-box size auxiliary remains controlled entirely by authored YAML.

## ADDED Requirements

### Requirement: `bbox_size_aux` is a strict objective plugin that depends on `bbox_geo` decoded-box state
The strict teacher-forcing module registry SHALL include `bbox_size_aux` as a
separate objective plugin for decoded-box size supervision.

Normative behavior:

- `bbox_size_aux` MUST fail fast if the current context does not provide the
  decoded predicted / target box state it requires,
- `bbox_size_aux` MUST NOT introduce an alternate bbox expression or alternate
  coord-token decode path,
- `bbox_size_aux` MUST remain reusable across Stage-2 two-channel and
  rollout-aligned trainers through the same module contract.

#### Scenario: bbox_size_aux fails fast when bbox_geo-derived state is missing
- **WHEN** a pipeline enables `bbox_size_aux`
- **AND** the runtime context does not provide the decoded canonicalized bbox
  state required by the plugin
- **THEN** the training step raises with actionable diagnostics
- **AND** training does not proceed with a silently altered objective.
