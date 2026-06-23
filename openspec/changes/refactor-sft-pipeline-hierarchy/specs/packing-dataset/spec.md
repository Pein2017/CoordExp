## MODIFIED Requirements

### Requirement: Static pack plans are deterministic and epoch-invariant

Static packing SHALL build deterministic pack plans from normalized dataset
identity, sample lengths, and packing knobs.

Normative behavior:

- The packed dataset wrapper MUST forward epoch changes to the underlying
  dataset before sample fetches for the new epoch.
- Datasets that resample per epoch, depend on `set_epoch` to change the sample
  schedule, or change per-index planning length MUST still be rejected
  (fail-fast) when `training.packing_mode=static`.
- Target-sequence random object ordering MAY remain compatible with static
  packing only when per-index planning length is invariant across epochs and the
  encoded/tokenized bytes consumed by the static plan remain length-compatible.

#### Scenario: Length-invariant random ordering preserves the static plan across epochs

- **GIVEN** `training.packing=true` and `training.packing_mode=static`
- **AND** the underlying dataset uses
  `sample_factory.target_sequence.object_ordering: random_permutation`
- **AND** per-index planning length is invariant across epochs
- **WHEN** the packed dataset advances from one epoch to the next
- **THEN** `raw_plan` and `aligned_plan` remain unchanged
- **AND** the underlying dataset receives the new epoch before sample fetches
- **AND** fetched samples MAY reflect the new epoch's deterministic object order
  without rebuilding the plan.

### Requirement: Static packing cache identity includes compact row axes and length

Static packing cache identity SHALL include normalized hierarchy fields that can
change packed bytes, lengths, or semantics.

The fingerprint MUST include at minimum:

- `pipeline.id`,
- `objective.id`,
- `detection_template.id`,
- `sample_factory.id`,
- `sample_factory.target_sequence.object_field_order`,
- `sample_factory.target_sequence.object_ordering`,
- `sample_factory.target_sequence.bbox_format`,
- `sample_factory.target_sequence.coordinate_surface`,
- `sample_factory.target_sequence.strict_parse`,
- prompt hash,
- resolved train prompt variant,
- tokenizer/chat-template identity,
- effective packing length.

Existing packing-plan discriminators, including dataset source identity, sample
limits, tokenizer/model identity, prompt/system text, min-fill, drop-last,
allow-single-long behavior, dataloader shuffle, offline image-pixel budget, and
the current implementation's other static-packing discriminators MUST remain
active where they already exist.
This change MUST NOT add a new image-root or view-store discriminator unless it
is already part of the current static-packing key set.
The effective packing length may be derived from legacy/raw config fields such
as `global_max_length`, but the normalized identity payload should expose the
semantic field as effective packing length.

#### Scenario: Static packing cache changes with bbox-first

- **GIVEN** two compact Stage-1 SFT configs that differ only by
  `sample_factory.target_sequence.object_field_order`
- **WHEN** static packing cache fingerprints are computed
- **THEN** the fingerprints differ.

#### Scenario: Static packing cache records 12000 global length

- **GIVEN** a final ablation Stage-1 SFT config
- **WHEN** static packing cache identity is materialized
- **THEN** the effective packing length is recorded as `12000`.

#### Scenario: Static packing cache changes with packing knobs

- **GIVEN** two configs with identical normalized hierarchy
- **AND** different static-packing min-fill or drop-last behavior
- **WHEN** static packing cache fingerprints are computed
- **THEN** the fingerprints differ.
