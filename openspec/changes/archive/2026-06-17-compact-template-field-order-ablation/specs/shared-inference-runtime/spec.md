## ADDED Requirements

### Requirement: Shared runtime provenance carries compact row axes
The shared inference runtime SHALL include resolved compact template id and
object field order in prompt, parser, backend request, and artifact provenance.

Parser policy metadata MAY contain internal parser ids, but those ids MUST be
derived from the two resolved axes rather than independently authored config
knobs.

#### Scenario: Runtime parser policy includes field order
- **GIVEN** `detection_template.id: compact_object_box_closed`
- **AND** object field order `geometry_first`
- **WHEN** the shared runtime builds parser policy metadata
- **THEN** the metadata records the compact template id and `geometry_first`
- **AND** no independent compact parse-mode or separator knob is accepted as the
  source of truth.

#### Scenario: Comparable artifacts require both axes
- **GIVEN** a compact artifact family missing object field order provenance
- **WHEN** a comparable evaluation path loads the artifact
- **THEN** loading fails before metrics are compared
- **AND** the diagnostic names the missing object field order metadata.
