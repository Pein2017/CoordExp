## MODIFIED Requirements

### Requirement: Stage-1 SHALL execute bbox size aux through a plugin host
Ordinary Stage-1 bbox size auxiliary supervision SHALL continue to execute
through the reusable plugin/module host rather than bespoke loss logic.

For `stage1_set_continuation`, bbox geometry and size auxiliary execution is
branch-local.

Additional normative behavior:
- ordinary one-sequence bbox mixins MUST NOT be dynamically composed for
  `stage1_set_continuation`,
- branch-local adapters MUST reuse canonical bbox geometry/size helper logic
  where available,
- branch-local bbox aux applies only to scored candidate-entry bbox state,
- the adapter reduction MUST first produce a mean-like per-candidate atom, then
  uniformly average valid atoms over scored candidates,
- `bbox_size_aux` MAY reuse the canonical size helper directly from the
  candidate-entry coord logits and labels rather than depending on ordinary
  one-sequence bbox mixins,
- setup or loss computation MUST fail fast if required branch-local bbox
  grouping metadata is unavailable.

#### Scenario: Branch-local size aux uses canonical helper
- **GIVEN** `custom.trainer_variant: stage1_set_continuation`
- **AND** `custom.bbox_size_aux.enabled: true`
- **WHEN** candidate branches are scored
- **THEN** size aux execution routes through the branch-local adapter
- **AND** that adapter reuses the canonical bbox size auxiliary helper.

#### Scenario: Missing branch-local bbox state fails fast
- **GIVEN** `custom.trainer_variant: stage1_set_continuation`
- **AND** bbox size aux is enabled
- **AND** candidate-entry bbox grouping metadata is unavailable
- **WHEN** loss computation reaches the aux adapter
- **THEN** training fails fast with actionable diagnostics.
