## MODIFIED Requirements

### Requirement: Coord auxiliary loss is coord-token supervision (not geometry IoU loss)
The canonical `custom.coord_soft_ce_w1` objective SHALL remain coord-token
distribution supervision.

For `stage1_set_continuation`, coord auxiliary execution is branch-local.

Additional normative behavior:
- ordinary `CoordSoftCEW1LossMixin` MUST NOT be dynamically composed for
  `stage1_set_continuation`,
- the set-continuation trainer MAY execute coord aux through a branch-local
  adapter that reuses the canonical helper,
- the adapter applies only to coord-token positions inside scored candidate
  entries,
- the adapter reduction MUST first produce a mean-like per-candidate atom, then
  uniformly average valid atoms over scored candidates,
- responsibility-weighted coord aux is not a v1 mode,
- branch-local metrics MUST be explicit set-continuation aux metrics, including
  `loss/aux_coord_soft_ce_w1` and `aux/coord_soft_ce_w1/*` counters, rather
  than silently reusing ordinary one-sequence metric meaning.

#### Scenario: Branch-local coord aux is enabled
- **GIVEN** `custom.trainer_variant: stage1_set_continuation`
- **AND** `custom.coord_soft_ce_w1.enabled: true`
- **WHEN** candidate branches are scored
- **THEN** coord aux applies to coord-token positions in scored candidate
  entries
- **AND** ordinary one-sequence coord-loss mixins are not composed.

#### Scenario: No branch adapter exists
- **GIVEN** `custom.trainer_variant: stage1_set_continuation`
- **AND** coord aux is enabled
- **AND** no branch-local coord aux adapter is available
- **WHEN** setup validates auxiliary adapters
- **THEN** setup fails with actionable diagnostics.
