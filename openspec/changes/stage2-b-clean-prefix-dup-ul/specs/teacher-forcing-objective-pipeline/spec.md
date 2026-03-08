# teacher-forcing-objective-pipeline Specification (Delta)

## MODIFIED Requirements

### Requirement: Teacher-forcing objective is declared as an ordered YAML pipeline
The teacher-forcing pipeline SHALL support the explicit `duplicate_ul` objective module for the canonical clean-prefix Channel-B contract.

Normative behavior:
- `duplicate_ul` is a valid objective module name.
- `duplicate_ul` MUST be declared with `channels: [B]`.
- `duplicate_ul.config` MUST be validated strictly and MUST be `{}` in v1.
- For canonical Stage-2 AB clean-prefix configs, the ordered objective list MUST place `duplicate_ul` after `token_ce` and before `bbox_geo`.

#### Scenario: duplicate_ul is accepted as a Channel-B-only objective module
- **WHEN** a teacher-forcing pipeline declares `{name: duplicate_ul, channels: [B], config: {}}`
- **THEN** pipeline validation succeeds for module naming/channel shape
- **AND** the module is eligible to run only on Channel-B steps.

### Requirement: Module registry is strict and validated before training starts
The strict teacher-forcing module registry SHALL include `duplicate_ul` as an objective module and fail fast when its prerequisites are unavailable.

Normative behavior:
- Unknown module names MUST still fail fast before the first training step.
- `duplicate_ul` MUST fail fast if the runtime context does not provide the canonical duplicate-ul supervision metadata required by the clean-prefix Channel-B contract.

#### Scenario: duplicate_ul fails fast when duplicate-ul metadata is missing
- **WHEN** a pipeline enables `duplicate_ul`
- **AND** the runtime context lacks canonical duplicate-ul supervision metadata
- **THEN** the training step raises with actionable diagnostics
- **AND** training does not proceed with a silently altered objective.
