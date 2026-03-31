# coord-aux-loss Specification (Delta)

## MODIFIED Requirements

### Requirement: Canonical config surface is `coord_soft_ce_w1`
The system SHALL expose coord auxiliary supervision through `custom.coord_soft_ce_w1` with the following additional keys for adjacent repulsion:

- `adjacent_repulsion_weight` (float; default `0.0`)
- `adjacent_repulsion_filter_mode` (string; default `same_desc`)
- `adjacent_repulsion_margin_ratio` (float; default `0.05`)
- `adjacent_repulsion_copy_margin` (float; default `0.8`)

Validation contract additions:

- `adjacent_repulsion_weight` MUST be `>= 0`,
- `adjacent_repulsion_margin_ratio` MUST be `>= 0`,
- `adjacent_repulsion_copy_margin` MUST lie in `[0, 1]`,
- `adjacent_repulsion_filter_mode` MUST be one of:
  - `same_desc`
  - `global`
- unsupported adjacent-repulsion nested keys MUST fail fast.

#### Scenario: Invalid adjacent repulsion config fails fast
- **GIVEN** `custom.coord_soft_ce_w1.adjacent_repulsion_weight < 0`
- **OR** `custom.coord_soft_ce_w1.adjacent_repulsion_margin_ratio < 0`
- **OR** `custom.coord_soft_ce_w1.adjacent_repulsion_copy_margin` lies outside
  `[0, 1]`
- **OR** `custom.coord_soft_ce_w1.adjacent_repulsion_filter_mode` is not one of
  the supported values
- **WHEN** config validation runs
- **THEN** validation fails fast with actionable guidance.

### Requirement: Coord auxiliary loss is coord-token supervision (not geometry IoU loss)
The coord auxiliary objective SHALL support adjacent distributional repulsion as
an additional coord-side sub-term in Stage-1 GT context.

Normative behavior:

- adjacent repulsion MUST be computed in coord-bin space rather than as a
  decoded-box-only geometry loss,
- adjacent repulsion MUST use the immediately previous object in GT
  teacher-forced order within the same sample as its reference object,
- adjacent repulsion MUST construct edge-only decaying bands around the previous
  object's target edges,
- adjacent repulsion MUST scale those bands by the previous object's width and
  height according to `adjacent_repulsion_margin_ratio`,
- adjacent repulsion MUST interpret `adjacent_repulsion_margin_ratio` as a
  per-edge half-width ratio,
- adjacent repulsion MUST use the canonical pre-sorted object order provided by
  the sample rather than inferring adjacency from flattened quartet order,
- adjacent repulsion MUST aggregate the four slot overlaps into a thresholded
  box-copy score,
- adjacent repulsion MUST respect the configured `same_desc | global` filter
  mode,
- if `same_desc` mode is enabled and Stage-1 cannot recover object-local
  description identity unambiguously for the active sample, runtime MUST fail
  fast rather than guessing.

#### Scenario: Stage-1 adjacent repulsion uses GT order and edge-band distributions
- **GIVEN** Stage-1 coord auxiliary supervision is enabled
- **AND** `adjacent_repulsion_weight > 0`
- **WHEN** loss is computed
- **THEN** the adjacent repulsion term is computed from coord distributions in
  GT object order
- **AND** it uses edge-only decaying bands derived from the previous box
- **AND** adjacency never crosses sample boundaries within a batch
- **AND** it does not require decoded-box CIoU as the primary loss definition.

### Requirement: Logging keys are stable for coord aux observability
When Stage-1 adjacent repulsion contributes to loss, the trainer SHALL emit
stable metric keys:

- `coord_softce_w1/adjacent_repulsion`

The trainer SHALL also emit stable diagnostics keys under `coord_diag/*`,
including:

- `coord_diag/adjacent_repulsion_pair_count`
- `coord_diag/adjacent_repulsion_applied_count`
- `coord_diag/adjacent_repulsion_copy_score_mean`

#### Scenario: Enabled Stage-1 adjacent repulsion emits stable metric keys
- **GIVEN** a training step with Stage-1 adjacent repulsion active
- **WHEN** metrics are logged
- **THEN** logs include `coord_softce_w1/adjacent_repulsion`
- **AND** the corresponding adjacent-pair diagnostics appear under
  `coord_diag/*`.
