## ADDED Requirements

### Requirement: Object field order controls compact row segment order
`custom.object_field_order` SHALL control compact row semantic segment order in
addition to JSON object key order.

Normative behavior:

- `desc_first` SHALL render the description/object-ref segment before the
  coordinate/box segment.
- `geometry_first` SHALL render the coordinate/box segment before the
  description/object-ref segment.
- The same value SHALL be used for training target rendering, prompt examples,
  inference parsing policy, artifact provenance, and cache fingerprints.
- `custom.object_field_order` SHALL NOT change object instance ordering.

#### Scenario: Geometry-first compact means bbox-first for bbox rows
- **GIVEN** `custom.object_field_order: geometry_first`
- **AND** the object geometry kind is `bbox_2d`
- **WHEN** a compact row is rendered
- **THEN** the box segment appears before the object-ref segment.

#### Scenario: Desc-first compact preserves current semantic order
- **GIVEN** `custom.object_field_order: desc_first`
- **WHEN** a compact row is rendered
- **THEN** the object-ref segment appears before the box segment.

#### Scenario: Field order does not reorder objects
- **GIVEN** `custom.object_ordering: sorted`
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** multiple objects are serialized
- **THEN** object instance sequence remains sorted by the active ordering policy
- **AND** only the segment order inside each object row changes.
