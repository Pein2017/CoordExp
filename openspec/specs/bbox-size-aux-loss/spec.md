# bbox-size-aux-loss Specification

## Purpose
Record that decoded-box size auxiliary supervision is retired from active
CoordExp training objectives.

## Requirements

### Requirement: bbox size auxiliary is removed from active training configs
The system SHALL reject `bbox_size_aux` as an active Stage-1, Stage-2, or
teacher-forcing objective module.

Normative behavior:

- `custom.bbox_size_aux` MUST fail fast when authored in active training
  configs.
- `stage2_ab.pipeline.objective[*].name=bbox_size_aux` MUST fail fast before
  trainer initialization.
- no active trainer mixin or objective module SHALL compute decoded-box
  log-width/log-height, log-area, or oversize-penalty losses.
- historical artifacts and archived notes MAY mention `bbox_size_aux`, but
  current docs and configs MUST NOT recommend it as a live training mechanism.

#### Scenario: Stage-1 custom bbox size aux is rejected
- **WHEN** a training config authors `custom.bbox_size_aux`
- **THEN** config validation fails fast
- **AND** the error says the bbox size auxiliary has been removed.

#### Scenario: Stage-2 bbox size aux module is rejected
- **WHEN** `stage2_ab.pipeline.objective[*].name=bbox_size_aux`
- **THEN** config validation fails fast before trainer init
- **AND** no bbox-size objective atoms are registered.
