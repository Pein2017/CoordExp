# rollout-matching-sft Specification (Delta)

## MODIFIED Requirements

### Requirement: Rollout-aligned module configs are strict and canonical
Rollout-aligned Stage-2 SHALL support adjacent repulsion as an optional
`coord_reg` config sub-term in rollout context.

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
- when enabled, rollout-context adjacent repulsion MUST follow the active
  rollout teacher-forced object order and current positive coord-side masking,
- and it MUST use edge-only decaying bands derived from the immediately
  previous object's bbox,
- and rollout adjacency MUST be defined from canonical clean-order indices
  rather than supervision append order.

#### Scenario: Rollout-aligned adjacent repulsion uses rollout context and strict coord-reg config
- **WHEN** rollout-aligned training enables `coord_reg.config.adjacent_repulsion_weight`
- **THEN** the config is validated strictly
- **AND** the adjacent repulsion term uses rollout-context object order rather
  than GT-only ordering.
