## ADDED Requirements

### Requirement: Random-permutation ET-RMP-CE remains the compact production baseline/comparator
The compact recursive detection production baseline SHALL remain
`objective.variant: random_permutation_et_rmp_ce` until an explicitly promoted
successor is approved.

Normative behavior:

- the comparator config path MUST be updated to
  `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml`,
- this baseline MUST use `objective.id: teacher_forcing`,
- this baseline MUST use `detection_template.id: compact`,
- this baseline MUST use `data.object_ordering: random_permutation`,
- this baseline MUST keep `objective.state_weighting:
  uniform_permutation`,
- this baseline MUST keep `objective.normalization:
  semantic_image_bucket_balanced`,
- trie support and balance weights for this baseline MUST be authored as
  `objective.trie_support_weight` and `objective.trie_balance_weight`,
- the baseline MUST remain available as a comparator when new detection
  objective variants are introduced.

#### Scenario: Comparator config resolves to random-permutation ET-RMP-CE
- **WHEN** the comparator config
  `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml`
  is parsed
- **THEN** it resolves to `objective.variant: random_permutation_et_rmp_ce`
- **AND** it resolves to `detection_template.id: compact`
- **AND** runtime teacher-forcing support and balance weights come from the
  top-level `objective.trie_*` fields.

### Requirement: Prefix-rollin ET-RMP-CE is a compact-family ablation surface
The compact recursive detection surface SHALL support
`objective.variant: prefix_rollin_et_rmp_ce` as a compact-family ablation and
diagnostic route, not as a production baseline by default.

Normative behavior:

- `prefix_rollin_et_rmp_ce` MUST require `detection_template.id` to be one of
  the supported compact template ids,
- `prefix_rollin_et_rmp_ce` MUST reject `stage1_json_pretty`,
- `prefix_rollin_et_rmp_ce` MUST sample a GT roll-in prefix length `K`
  uniformly over `[0, object_count]` for the current route,
- roll-in prefix labels MUST be masked,
- suffix order MUST remain `same_sampled_permutation` for the checked-in route,
- support and balance weights MUST be authored under `objective.target`,
- obsolete flat support/balance aliases MUST fail fast for this variant,
- E1/E2 boundary-pressure and EOS-loosening variants MUST remain historical
  evidence only, not live latest-detection training surfaces.

#### Scenario: Prefix-rollin accepts semantic compact templates
- **GIVEN** `objective.variant: prefix_rollin_et_rmp_ce`
- **AND** `detection_template.id` is one of the supported compact template ids
- **WHEN** config validation runs
- **THEN** the template family is accepted for prefix-rollin target building.

#### Scenario: Prefix-rollin rejects JSON template
- **GIVEN** `objective.variant: prefix_rollin_et_rmp_ce`
- **AND** `detection_template.id: stage1_json_pretty`
- **WHEN** config validation runs
- **THEN** validation fails before training starts.

#### Scenario: Prefix-rollin rejects obsolete flat trie weights
- **WHEN** `objective.variant: prefix_rollin_et_rmp_ce` is authored with legacy
  flat trie-weight aliases such as `objective.trie_support_weight`
- **THEN** config validation fails fast
- **AND** the error points to the nested `objective.target` and
  `objective.boundary` surfaces.

### Requirement: Compact detection token-row adaptation covers template structural rows
Compact recursive detection SHALL treat trainable token rows as a detection
template contract rather than as a coord-only adapter contract.

Normative behavior:

- compact recursive detection MUST train the 1000 coord-token rows plus the
  structural rows required by the selected `detection_template.id` when token-row
  adaptation is enabled,
- `compact` MUST require 1002 trainable rows,
- `compact_box_closed` MUST require 1003 trainable rows,
- `compact_object_box_closed` and `compact_object_box_closed_lines` MUST require
  1004 trainable rows,
- the persisted module name MUST be `token_embeddings_adapter`,
- contract wording MUST describe this surface as token-row adaptation rather
  than coord-only adaptation,
- the `coord_geometry` token-row group MUST preserve the expected
  `<|coord_0|>` through `<|coord_999|>` token id range.

#### Scenario: Compact token rows include selected structural rows
- **WHEN** compact recursive detection enables trainable token rows
- **THEN** the trainable row set includes coord geometry rows
- **AND** it includes exactly the compact structural rows required by the
  selected `detection_template.id`.

## REMOVED Requirements

### Requirement: Random-permutation ET-RMP-CE remains the production baseline/comparator
The old compact-full baseline requirement is removed and replaced by the compact
semantic-template baseline requirement above.

**Reason**: This requirement hard-codes the old `compact_full` template id and
config path.

**Migration**: Use `Random-permutation ET-RMP-CE remains the compact production
baseline/comparator`, with `detection_template.id: compact` and the compact-named
active comparator config path.

### Requirement: Prefix-rollin ET-RMP-CE is a compact-full ablation surface
The old compact-full prefix-rollin requirement is removed and replaced by the
compact-family prefix-rollin requirement above.

**Reason**: Prefix-rollin should be tied to the semantic compact template family,
not the old `compact_full` name.

**Migration**: Use `Prefix-rollin ET-RMP-CE is a compact-family ablation
surface`.

### Requirement: Compact-full token-row adaptation covers geometry and structural rows
The old compact-full token-row requirement is removed and replaced by the
template-derived compact token-row requirement above.

**Reason**: Token-row adaptation is now derived from the selected semantic
compact template id and may require 1002, 1003, or 1004 rows.

**Migration**: Use `Compact detection token-row adaptation covers template
structural rows`.
