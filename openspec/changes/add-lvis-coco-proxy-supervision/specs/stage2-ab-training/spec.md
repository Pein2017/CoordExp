# stage2-ab-training Specification (Delta)

## MODIFIED Requirements

### Requirement: Stage-2 AB module configs are strict and canonical (no aliases)
Stage-2 AB pipeline configs SHALL support metadata-driven object weighting
through canonical module config keys only.

Normative behavior:

- `token_ce.config` MUST accept `object_weight_mode`,
- `bbox_geo.config` MUST accept `object_weight_mode`,
- `coord_reg.config` MUST accept `object_weight_mode`,
- `object_weight_mode` MUST be one of:
  - `none`
  - `metadata`
- if `object_weight_mode=metadata` and proxy metadata is absent for a sample,
  the runtime MUST fall back to weight `1.0` rather than failing,
- unknown proxy-weight config keys or alias keys MUST fail fast.

#### Scenario: Metadata mode falls back cleanly on plain COCO samples
- **WHEN** Stage-2 AB enables `object_weight_mode=metadata`
- **AND** a sample has no proxy-supervision metadata block
- **THEN** token / bbox / coord supervision uses weight `1.0`
- **AND** the sample behaves like an ordinary non-augmented COCO sample.

### Requirement: Stage-2 AB objective application is explicit and non-redundant
Stage-2 AB SHALL keep proxy-supervision weighting local to desc and coord
families without changing global structure supervision.

Normative behavior:

- `token_ce` in metadata mode MUST apply object-local proxy weights only to
  desc-value CE,
- structure CE MUST remain global and unchanged,
- `bbox_geo` and `coord_reg` in metadata mode MUST apply object-local
  `coord_weight` to the aligned bbox / coord carriers,
- the same proxy object MUST NOT require a second alternate rendered target just
  to express its weight.

#### Scenario: Plausible object lowers desc and coord supervision only
- **WHEN** a plausible proxy object appears in a Stage-2 AB sample
- **THEN** its desc-value CE and bbox/coord supervision use the lower proxy
  weights
- **AND** the surrounding object syntax remains fully supervised as structure.
