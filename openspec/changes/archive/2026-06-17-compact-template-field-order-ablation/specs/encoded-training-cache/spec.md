## ADDED Requirements

### Requirement: Encoded-sample cache identity includes compact row axes
Encoded-sample cache fingerprints SHALL include every resolved input that can
change assistant target bytes for compact Stage-1 SFT.

For compact templates, the fingerprint MUST include at minimum:

- resolved detection template id,
- resolved object field order,
- prompt hash,
- object instance ordering policy,
- coordinate mode and bbox format,
- tokenizer/chat-template identity.

#### Scenario: Cache fingerprint changes with field order
- **GIVEN** two compact Stage-1 SFT configs that differ only by
  `custom.object_field_order`
- **WHEN** encoded-sample cache fingerprints are computed
- **THEN** the fingerprints differ.

#### Scenario: Cache fingerprint changes with object-only closure template
- **GIVEN** two compact Stage-1 SFT configs that differ only by
  `custom.detection_template_id`
- **AND** one value is `compact_object_closed`
- **WHEN** encoded-sample cache fingerprints are computed
- **THEN** the fingerprints differ.
