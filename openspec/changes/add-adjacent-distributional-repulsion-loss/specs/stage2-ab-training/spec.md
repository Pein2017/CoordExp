# stage2-ab-training Specification (Delta)

## MODIFIED Requirements

### Requirement: Stage-2 AB module configs are strict and canonical (no aliases)
Stage-2 AB MUST support adjacent repulsion as an optional `coord_reg` config
sub-term.

Normative behavior:

- `coord_reg.config` MUST additionally accept:
  - `adjacent_repulsion_weight`
  - `adjacent_repulsion_filter_mode`
  - `adjacent_repulsion_margin_ratio`
  - `adjacent_repulsion_copy_margin`
- `adjacent_repulsion_weight` MUST be `>= 0`,
- `adjacent_repulsion_margin_ratio` MUST be `>= 0`,
- `adjacent_repulsion_copy_margin` MUST lie in `[0, 1]`,
- `adjacent_repulsion_filter_mode` MUST be one of:
  - `same_desc`
  - `global`

#### Scenario: Invalid Stage-2 adjacent repulsion coord-reg config fails fast
- **WHEN** `coord_reg.config.adjacent_repulsion_weight < 0`
- **OR** `coord_reg.config.adjacent_repulsion_margin_ratio < 0`
- **OR** `coord_reg.config.adjacent_repulsion_copy_margin` lies outside `[0, 1]`
- **OR** `coord_reg.config.adjacent_repulsion_filter_mode` is unsupported
- **THEN** config validation fails fast with actionable guidance.

### Requirement: Stage-2 Two-Channel adheres to the unified loss registry contract
Stage-2 Two-Channel training SHALL implement adjacent repulsion through the
existing `coord_reg` coord-side surface when enabled.

Normative behavior:

- adjacent repulsion MUST remain a coord-side sub-term rather than a standalone
  bbox objective module in v1,
- when enabled for rollout context, adjacent repulsion MUST use the edited clean
  teacher-forced target order,
- rollout-context adjacent repulsion MUST apply only to the current rollout
  objects that already belong to positive coord-side supervision,
- rollout-context adjacent repulsion MUST use edge-only decaying bands derived
  from the immediately previous object's bbox,
- rollout-context adjacent repulsion MUST define adjacency from canonical
  clean-order indices rather than `bbox_groups_prefix` or other supervision
  append order,
- rollout-context adjacent repulsion MUST preserve enough grouped carrier
  metadata for previous-object lookup and `same_desc | global` gating before
  `coord_reg` executes,
- rollout-context adjacent repulsion MUST combine slot overlaps into one
  thresholded box-copy score,
- rollout-context adjacent repulsion MUST support both:
  - `same_desc`
  - `global`
  filter modes.

#### Scenario: Enabled rollout-context adjacent repulsion remains under coord_reg
- **WHEN** Stage-2 AB enables adjacent repulsion for rollout context
- **THEN** the term is realized through `coord_reg`
- **AND** the trainer does not introduce a standalone new bbox module to compute
  the v1 adjacent repulsion term.
