## ADDED Requirements

### Requirement: Standard Stage-1 SFT supports compact two-knob targets
The standard Stage-1 SFT surface SHALL support compact target rendering from the
resolved compact template id and `custom.object_field_order`.

The standard SFT path MUST remain separate from rollout-only
`TeacherForcingRollin` behavior.  Selecting bbox-first compact rows MUST NOT
require migrating Stage-1 SFT into a rollout-aware trainer.

#### Scenario: Standard SFT renders geometry-first compact targets
- **GIVEN** a standard Stage-1 SFT config with
  `custom.detection_template_id: compact_object_box_closed`
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** dataset conversations are built
- **THEN** assistant target text uses the rich bbox-first compact row contract
- **AND** the run does not require `TeacherForcingRollin`.

#### Scenario: Standard SFT remains sorted for final ablation
- **GIVEN** a final ablation Stage-1 SFT config
- **WHEN** config validation runs
- **THEN** `custom.object_ordering` resolves to `sorted`
- **AND** row field order is controlled only by `custom.object_field_order`.

### Requirement: Production ablation configs pin training identity
The desc-first and bbox-first production-ablation configs SHALL pin checkpoint,
packing, trainability, and dataset identity explicitly.

Normative behavior:

- checkpoint path MUST be
  `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
- `global_max_length` MUST be `12000`.
- packing MUST be static/offline and cache-fingerprinted.
- only LLM parameters MUST be trainable unless an explicitly approved follow-up
  changes the trainability policy.

#### Scenario: Production ablation config resolves expected checkpoint and packing
- **GIVEN** either final ablation production config
- **WHEN** the materialized training config is loaded
- **THEN** the model path resolves to the natural-adjacent checkpoint
- **AND** `global_max_length` resolves to `12000`
- **AND** static packing is enabled.
