# research-probe-input-preparation Specification

## Purpose

Let research profiles inspect a small explicit cohort and reuse exact multimodal inputs without loading model weights or adopting a historical experiment's population and objective.

## Requirements

### Requirement: Explicit cohort selection

The inspection entry SHALL accept either distinct explicit example IDs in requested order or a positive count with an explicit seed. It SHALL bind the actual ordered membership to the source data identity. Unknown or repeated requested IDs, duplicate source IDs, malformed source rows, and counts exceeding the population SHALL fail without replacement or repetition. The selected membership SHALL be reusable unchanged across compared arms.

#### Scenario: Ordered paired examples

- **WHEN** a caller selects two known IDs in a specified order
- **THEN** inspection reports those exact ordered members and source identity, and both arms can consume that same selection

#### Scenario: Invalid small cohort

- **WHEN** a requested ID is unknown or repeated, the source is invalid, or the count exceeds its population
- **THEN** selection fails before model loading and does not substitute another example

### Requirement: Weight-free generation input planning

For selected rows and an explicit resolved profile, preparation SHALL expose the rendered prompt, actual expanded generation token IDs, image grid and identity, realized object ordering, and a request usable by existing native preparation. It SHALL use caller-supplied processor components without loading model weights, changing the profile, or requiring a historical cohort. Pixel materialization SHALL remain explicit and independent of request planning.

#### Scenario: Independent two-image preparation

- **WHEN** a caller plans two real image rows with a processor-only frontend
- **THEN** its prompts, grids, image identities and native requests equal the existing preparation behavior and no executable model is loaded

#### Scenario: Reuse planned inputs

- **WHEN** the caller explicitly materializes the planned native requests once and passes those prepared tensors to exact replay
- **THEN** replay uses those prepared inputs under the existing device/position contract without requiring another image preparation stage

### Requirement: Separate annotated target inspection

Annotated-target preparation SHALL require an explicit maximum length and expose the existing encoded target tokens, physical supervised and ignored spans, and EOS boundary separately from the generation prefix. Generation-only preparation SHALL have no target encoding. Annotated targets SHALL preserve the existing template, geometry, ordering and encoding contracts; target preparation SHALL NOT choose a loss, reduction or scientific objective.

#### Scenario: Annotated target matches a generation prefix

- **WHEN** target inspection is requested for an admitted compact-row profile
- **THEN** the full encoding before the first supervised token equals its expanded generation prefix, EOS remains in its supervised span, and trailing ignored tokens remain outside the target

#### Scenario: Ordering changes only the relevant identity

- **WHEN** object ordering changes the assistant target without changing system/user prompt content
- **THEN** realized order and target identity reflect the change without inventing a change to generation-prefix token IDs

### Requirement: Existing profiles retain their meaning

Migrated retained profiles SHALL preserve request policies, strict population and source gates, target positions, objective and denominator choices, and persisted artifact meanings. Shared input preparation SHALL NOT interpret rewards or scientific success, restore missing trajectory fields, or require provenance checks beyond those owned by the existing consumer. Changed literal histories SHALL remain supported by native replay; tampered bound artifacts SHALL still fail at their owning consumer.

#### Scenario: Existing Source256 preparation

- **WHEN** the existing Source256 preflight and plan preparation use the shared input operation
- **THEN** their frozen population, token/media bindings, EOS treatment, artifact projections and invalid-input rejection remain unchanged

#### Scenario: Fresh small probe outside a historical cohort

- **WHEN** the inspection entry uses a valid profile with two explicitly selected rows outside the historical Source256 cohort
- **THEN** it prepares those inputs without weakening the separate Source256 preflight or creating a Source256 scientific receipt

### Requirement: Explicit lifetime and bounded preparation work

The prepared-input path SHALL keep processor/model lifetime with the caller and SHALL reuse row render and image-plan results across generation and target inspection. It SHALL not force pixel materialization in a planning-only caller or repeat native image materialization solely to produce inspection metadata. Performance reports SHALL bind measurements to fixed inputs and distinguish planning, materialization and model execution.

#### Scenario: Generation and target inspection together

- **WHEN** a caller inspects both views of the same row
- **THEN** they share its render and image plan, while distinct generation and full-target tokenization retain their own contracts

#### Scenario: Measured efficiency claim

- **WHEN** an upgrade reports preparation efficiency
- **THEN** the report includes fixed input identities, old/new operation counts and measured timings and does not imply unmeasured model-throughput or quality gains
