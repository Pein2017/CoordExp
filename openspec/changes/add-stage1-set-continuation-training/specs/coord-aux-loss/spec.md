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

### Requirement: Coord/text vocabulary gate math is reusable but objective owned
The low-level coord-vocabulary mass helpers MAY be reused by multiple trainer
surfaces, but each trainer surface SHALL own its mask semantics.

For `stage1_set_continuation`, bidirectional token-type gating is part of the
native set-continuation objective and is not the same feature as enabling
`custom.coord_soft_ce_w1`.

Additional normative behavior:
- coord-gate math computes `-log(sum P(coord_vocab))` from full-vocabulary
  logits at coord-label objective positions,
- text-gate math computes `-log(1 - sum P(coord_vocab))` from full-vocabulary
  logits at non-coord supervised objective positions,
- helper reuse MUST NOT import ordinary one-sequence `CoordSoftCEW1LossMixin`
  semantics into set-continuation,
- branch-local `coord_soft_ce_w1` aux MAY remain disabled while the
  bidirectional token gate is enabled,
- tests MUST cover both helper math and set-continuation mask ownership.

#### Scenario: Gate helpers do not define masks
- **GIVEN** set-continuation bidirectional gating is enabled
- **WHEN** the gate loss is computed
- **THEN** the trainer supplies objective-aligned coord/text masks
- **AND** the low-level coord/text gate helper only evaluates vocabulary mass
  on those provided logits rows.

#### Scenario: Gate math uses full-vocabulary coord mass
- **GIVEN** full-vocabulary logits and a configured coord-token id set
- **WHEN** coord/text gate loss is computed
- **THEN** `p_coord` equals the sum of full-vocabulary softmax probability over
  coord ids
- **AND** coord-gate loss equals `-log(p_coord)`
- **AND** text-gate loss equals a numerically stable `-log(1 - p_coord)`
- **AND** redistributing probability inside the coord or non-coord partitions
  without changing `p_coord` does not change the corresponding gate loss.
