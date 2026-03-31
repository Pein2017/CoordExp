# teacher-forcing-unified-loss-registry Specification (Delta)

## MODIFIED Requirements

### Requirement: Canonical loss scalars are mean-like and scale-invariant
The unified loss registry SHALL support `adjacent_repulsion` as an optional
coord-side sub-term under the teacher-forcing coord-distribution family.

Normative behavior:

- `adjacent_repulsion` MUST be a mean-like scalar over contributing adjacent
  objects,
- the term MUST be computed from coord-bin distributions rather than requiring a
  decoded-box-only primary definition,
- the term MUST use the immediately previous object in the active
  teacher-forced object order as its reference object,
- the term MUST construct one decaying edge band for each of the previous
  object's four edges,
- the term MUST scale those bands by the previous object's width or height,
- the term MUST interpret the public band-width ratio as a per-edge half-width
  ratio,
- the term MUST use a linear taper in v1,
- the term MUST combine the four slot overlaps into one thresholded box-copy
  score so partial-edge agreement can still contribute zero,
- the term MUST support:
  - `same_desc`
  - `global`
  filter modes.

#### Scenario: Adjacent repulsion is mean-like over contributing adjacent pairs
- **WHEN** adjacent repulsion is enabled and multiple adjacent object pairs
  contribute in one forward
- **THEN** the reported scalar remains comparable as a mean-like value rather
  than scaling with sequence length alone.

### Requirement: Gate terms and coord-side sub-terms respect context-specific masking
The unified loss registry SHALL treat adjacent repulsion as a context-aware
coord-side term.

Normative behavior:

- in `context=gt`, adjacent repulsion uses GT teacher-forced object order,
- in `context=rollout`, adjacent repulsion uses the active edited clean
  teacher-forced object order for the rollout-context sample,
- in `context=rollout`, adjacency MUST be derived from canonical clean-order
  indices rather than supervision append order,
- adjacent repulsion MUST only apply to objects that are currently eligible for
  positive coord-side supervision in the active context,
- the immediately previous object reference MAY come from the active target order
  even when that previous object is retained context rather than a positive
  coord-supervised object.

#### Scenario: Rollout-context adjacent repulsion follows edited clean target order
- **WHEN** adjacent repulsion is enabled in `context=rollout`
- **THEN** adjacency is computed from the edited clean teacher-forced target
- **AND** the current object must belong to an active positive coord-supervised
  rollout subset before the term applies.
