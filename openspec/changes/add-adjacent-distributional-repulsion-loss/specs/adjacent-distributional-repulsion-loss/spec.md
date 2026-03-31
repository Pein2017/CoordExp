## ADDED Requirements

### Requirement: Adjacent distributional repulsion is an immediate-previous-object edge-band anti-copy term
The system SHALL support an optional adjacent-repulsion loss term defined on the
immediately previous object in teacher-forced object order.

Normative behavior:

- adjacency means exactly the immediately previous object in the active
  teacher-forced object order within the same sample,
- the v1 contract MUST NOT require a larger previous-object window or a global
  all-pairs field,
- the v1 contract MUST NOT allow cross-sample adjacency,
- the primary v1 signal MUST be distributional in coord-bin space,
- the term MUST be computed from the current object's coord-bin distributions
  against four edge-only decaying bands induced by the previous object's target
  box,
- each edge band MUST scale with the previous object's width or height on the
  corresponding axis,
- each edge band MUST interpret its public ratio knob as a per-edge half-width
  ratio rather than a whole-box scale factor,
- each edge band MUST use a linear taper in v1,
- each edge band MUST decay to zero at the band boundary and remain zero outside
  the band,
- the four slot overlaps MUST be aggregated into one box-copy score,
- the v1 penalty MUST activate only when that box-copy score exceeds a
  configured copy margin,
- the v1 contract MUST NOT require decoded-box CIoU-style adjacent repulsion as
  the primary definition.

#### Scenario: Missing previous object yields zero adjacent-repulsion contribution
- **WHEN** the current object has no immediately previous object in the active
  teacher-forced order
- **THEN** adjacent repulsion contributes zero for that object.

#### Scenario: Partial-edge agreement below the copy margin yields zero penalty
- **WHEN** the current object's coord distributions align with only part of the
  previous box geometry
- **AND** the aggregated box-copy score stays below the configured copy margin
- **THEN** adjacent repulsion contributes zero for that object.

### Requirement: Adjacent repulsion supports `same_desc` and `global` filter modes
The system SHALL support two adjacent-repulsion filter modes.

Normative behavior:

- `same_desc` means the adjacent-repulsion term applies only when the current
  object and immediately previous object share the same normalized description,
- `global` means the description filter is dropped,
- repo-authored defaults SHOULD use `same_desc`,
- `global` MUST remain an explicit opt-in ablation mode.

#### Scenario: `same_desc` mode suppresses cross-description adjacency
- **WHEN** adjacent repulsion is enabled in `same_desc` mode
- **AND** the current object and immediately previous object have different
  normalized descriptions
- **THEN** adjacent repulsion contributes zero for that adjacent pair.

### Requirement: Adjacent repulsion is shared across GT and rollout contexts with context-correct wording
The system SHALL define adjacent repulsion across teacher-forcing contexts using
the active canonical context vocabulary.

Normative behavior:

- Stage-1 uses `context=gt`,
- Stage-2 rollout teacher forcing uses `context=rollout`,
- deprecated `self_context` terminology MUST NOT define the contract for this
  capability.

#### Scenario: Stage-1 and Stage-2 use canonical context names
- **WHEN** adjacent repulsion is described in config, docs, or specs
- **THEN** Stage-1 uses `gt`
- **AND** Stage-2 rollout teacher forcing uses `rollout`
- **AND** deprecated `self_context` wording is absent from the adjacent
  repulsion contract.
