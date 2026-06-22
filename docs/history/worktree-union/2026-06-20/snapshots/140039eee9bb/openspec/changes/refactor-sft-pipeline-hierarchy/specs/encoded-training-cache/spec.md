## MODIFIED Requirements

### Requirement: Encoded cache is limited to explicitly eligible deterministic runs

Encoded-sample cache eligibility SHALL be determined from the normalized
training hierarchy and the actual dataset mutation behavior.

Normative behavior:

- The cache MUST NOT be treated as eligible when encoded output depends on
  runtime RNG, requested dataloader index, or epoch.
- The following paths MUST be treated as ineligible in the initial migration:
  - augmentation,
  - preprocessors that mutate encoded content at fetch time,
  - non-empty `curriculum_state`,
  - `sample_factory.target_sequence.object_ordering: random`,
  - `sample_factory.target_sequence.object_ordering: random_permutation` when
    encoded output can change across epochs,
  - fusion/mixing schedules,
  - hard-sample or epoch-varying sample plans that change dataset identity at
    fetch time.
- When `training.encoded_sample_cache.ineligible_policy=error`, startup MUST
  fail fast with actionable guidance.
- When `training.encoded_sample_cache.ineligible_policy=bypass`, training MUST
  continue without cache reuse and MUST emit an explicit bypass reason.
- Historical `custom.object_ordering` MUST NOT be used as the active cache
  eligibility source after this migration.

#### Scenario: Ineligible random target-sequence path is rejected

- **GIVEN** encoded-sample caching is enabled
- **AND** `sample_factory.target_sequence.object_ordering: random_permutation`
  can change encoded content across epochs
- **WHEN** training initializes dataset caching
- **AND** `training.encoded_sample_cache.ineligible_policy=error`
- **THEN** the system does not silently reuse encoded samples
- **AND** it fails fast with actionable guidance.

### Requirement: Encoded-sample cache identity includes compact row axes

Encoded-sample cache fingerprints SHALL include the normalized hierarchy fields
that affect tokenized sample bytes. These fields are additive to existing
encoded-cache discriminators and MUST NOT replace source, image, prompt, model,
or preprocessing identity.

For compact templates, the fingerprint MUST include at minimum:

- `pipeline.id`,
- `objective.id`,
- `detection_template.id`,
- `sample_factory.id`,
- `sample_factory.target_sequence.object_field_order`,
- prompt hash,
- `sample_factory.target_sequence.object_ordering`,
- `sample_factory.target_sequence.coordinate_surface`,
- `sample_factory.target_sequence.bbox_format`,
- `sample_factory.target_sequence.strict_parse`,
- resolved train prompt variant,
- resolved prompt hash,
- tokenizer/chat-template identity.

Existing dataset source identity, train/eval split or sample-limit identity,
model/tokenizer identity, system prompt text, preprocessing identity, offline
image-pixel budget, and the current implementation's other encoded-cache
discriminators MUST remain active where they already exist.
This change MUST NOT add a new image-root or view-store discriminator unless it
is already part of the current encoded-cache key set.

#### Scenario: Cache fingerprint changes with field order

- **GIVEN** two compact Stage-1 SFT configs that differ only by
  `sample_factory.target_sequence.object_field_order`
- **WHEN** encoded-sample cache fingerprints are computed
- **THEN** the fingerprints differ.

#### Scenario: Cache fingerprint changes with object-only closure template

- **GIVEN** two compact Stage-1 SFT configs that differ only by
  `detection_template.id`
- **AND** one value is `compact_object_closed`
- **WHEN** encoded-sample cache fingerprints are computed
- **THEN** the fingerprints differ.

#### Scenario: Cache fingerprint changes with prompt variant

- **GIVEN** two compact Stage-1 SFT configs that differ only by `prompt.variant`
- **WHEN** encoded-sample cache fingerprints are computed
- **THEN** the fingerprints differ.

#### Scenario: Existing encoded-cache discriminators remain active

- **GIVEN** two compact Stage-1 SFT configs that differ only by an existing
  non-hierarchy encoded-cache discriminator
- **WHEN** encoded-sample cache fingerprints are computed
- **THEN** the fingerprints differ
- **AND** no new image-root or view-store discriminator is required by this
  change unless it already exists in that key set.
