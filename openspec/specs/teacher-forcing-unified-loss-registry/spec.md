# teacher-forcing-unified-loss-registry Specification

## Purpose
Define the canonical teacher-forcing contexts, token types, active loss component
names, and retired-module boundaries shared across Stage-2 two-channel training
and the new typed teacher-forcing objective architecture.

## Requirements

### Requirement: Canonical contexts and token types are shared
The system SHALL use a small shared vocabulary for teacher-forcing contexts and
token roles.

Normative contexts:
- `gt`: pure GT teacher forcing.
- `rollout`: Channel-B clean-prefix teacher-forced logits under accepted rollout
  prefix plus injected FN objects.

Normative token types:
- `struct`: schema, punctuation, keys, delimiters, and other non-desc,
  non-coord content.
- `desc`: free-text object description tokens.
- `coord`: coordinate vocabulary tokens `<|coord_k|>`.
- `eos`: end token `<|im_end|>` for Qwen3-VL training/eval contracts.

#### Scenario: Stage-2 two-channel context set is explicit
- **WHEN** registry contexts are enumerated for active Stage-2 training
- **THEN** the supported contexts include `gt` and `rollout`
- **AND** `self_context` is not part of the active context contract.

### Requirement: Active loss component names are text/trie focused
The active teacher-forcing registry SHALL keep only the text/structure and
typed-trie component names needed by the current architecture.

Active component names:
- `struct_ce`: weighted hard CE over schema/structure positions.
- `desc_ce`: weighted hard CE or valid-set text supervision over description
  positions.
- `coord_token_ce`: hard CE over coord positions only when a selected hard path
  is explicitly supervised.
- `type_exclusive`: global token-type mass objective owned by the typed
  teacher-forcing objective.
- `valid_set_marginal`: next-token valid-set marginal likelihood owned by the
  typed trie objective.
- `coverage`: optional alpha-weighted within-valid coverage regularizer owned by
  the typed trie objective.
- `continuation_margin`: optional valid-continuation-vs-EOS calibration owned by
  the typed trie objective.

Removed component/module names:
- `loss_duplicate_burst_unlikelihood`
- `geo`
- `bbox_geo`
- `bbox_size_aux`
- `coord_reg`
- `coord_gate`
- `text_gate`

#### Scenario: Removed duplicate-burst objective is absent
- **WHEN** the live teacher-forcing objective registry is enumerated
- **THEN** `loss_duplicate_burst_unlikelihood` is absent
- **AND** duplicate-control diagnostics are not represented as a live objective
  component.

#### Scenario: Removed bbox/coord modules fail fast
- **WHEN** a Stage-2 or teacher-forcing config declares `bbox_geo`,
  `bbox_size_aux`, `coord_reg`, `coord_gate`, or `text_gate` as a pipeline module
- **THEN** config validation fails before trainer initialization
- **AND** the error lists the supported active module names.

### Requirement: Token-type exclusivity is a standalone typed objective concern
The registry SHALL NOT route token-type exclusivity through legacy coord
regularizer modules.

Normative behavior:
- The typed teacher-forcing objective owns global token-type mass supervision and
  wrong-type-mass diagnostics.
- Inner valid-set objectives operate on conditional probabilities within the
  correct token type and do not own type loss.
- Stage-2 two-channel legacy text objectives MUST NOT emit coord-gate,
  text-gate, decoded-box geometry, or bbox-size objective atoms.

#### Scenario: Inner objectives do not own type loss
- **WHEN** an ambiguous typed-trie node is trained
- **THEN** the global type-exclusive term is computed separately
- **AND** valid-set marginal / optional coverage are computed within the
  already-correct token-type group.

### Requirement: Canonical loss scalars are mean-like
Active `loss/<component>` scalars SHALL be comparable across packing, batch
size, and gradient-accumulation settings.

Normative behavior:
- `struct_ce`, `desc_ce`, and `coord_token_ce` MUST be weighted means over
  contributing tokens.
- `type_exclusive`, `valid_set_marginal`, `coverage`, and
  `continuation_margin` MUST be mean-like over their contributing typed-trie
  positions.
- Internal numerators/denominators MAY be kept internal or emitted only under
  explicit counter-like names such as `*_sum`, `*_count`, `*_num`, or `*_den`.
- Removed geometry/coord-regularizer names MUST NOT appear as active
  `loss/<component>` scalars.

#### Scenario: Packed sequences do not change mean-like loss scale
- **WHEN** two packed forwards contain different numbers of supervised tokens
  but identical per-token distributions
- **THEN** active `loss/<component>` scalars remain comparable mean-like values.

### Requirement: Pseudo-positive rollout spans remain text/structure aligned
This clean-prefix baseline requirement applies only when Channel-B selects
`token_ce` or `hard_sft`. Channel-B pseudo-positive current-attempt objects
SHALL be handled as accepted-prefix context, not as a way to reactivate retired
coord/bbox regularizers.

Normative behavior:
- dead current-attempt objects MUST NOT appear in the positive teacher-forced prefix.
- Retained prefix objects MAY receive global rollout-prefix structure
  supervision as defined by the Channel-B contract.
- `pseudo_positive` objects MAY remain in the edited current-attempt prefix for
  accepted structure context.
- `pseudo_positive`, shield-only, and dead-current objects MUST NOT contribute
  desc-positive supervision unless explicitly represented as `matched_clean`,
  `fn`, or `recovered_fn`.
- `pseudo_positive` current-attempt objects MUST NOT activate `bbox_geo`,
  `bbox_size_aux`, `coord_reg`, `coord_gate`, or `text_gate`.
- shield-only objects MUST NOT receive bbox/coord positive supervision unless
  they are promoted to `pseudo_positive` by full peer consensus.

#### Scenario: Pseudo-positive spans are desc-neutral
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** selected pseudo-positive current-attempt objects may remain as structure/context
  prefix content
- **AND** they do not contribute desc CE
- **AND** they do not activate retired coord/bbox regularizer terms.

### Requirement: Stage-2 text objectives and residual-state trie aliases remain separable
Stage-2 two-channel legacy objectives SHALL remain text/structure-only, while
the residual-state trie aliases SHALL own token-type exclusivity and valid-set
marginal likelihood over dynamic residual actions.

Normative behavior:
- Stage-2 AB active pipeline objective module names are limited to `token_ce`,
  `hard_sft`, `stage2_trie_ce`, and `residual_set_correction`.
- `stage2_trie_ce` and `residual_set_correction` are aliases for the same
  residual-state dynamic valid-set objective; Channel-B configs MUST select at
  most one of them.
- Residual-state trie configs MUST NOT reuse removed legacy bbox/coord module
  names or legacy candidate-trie config keys.
- The hard-SFT baseline MUST remain available as an explicit ablation surface.

#### Scenario: Hard SFT remains a baseline
- **WHEN** a config selects the hard-SFT baseline
- **THEN** the objective supervises the selected teacher path
- **AND** it does not silently enable valid-set marginal, coverage, or retired
  bbox/coord auxiliaries.

### Requirement: First-error correction atoms may target oracle tokens
Teacher-forcing target IR SHALL normally require `selected_token_id` to match
`input_ids[target_position]`, but residual first-error correction atoms MAY
target an oracle token different from the live self-prefix token when the
provenance explicitly marks that mismatch.

Normative behavior:
- A target-token mismatch is valid only when
  `provenance.allow_target_token_mismatch=true`.
- The oracle `selected_token_id` MUST still be inside `valid_token_ids`.
- Loss modules compute valid-set probability from `valid_token_ids`; they MUST
  NOT treat the live self-prefix token as positive unless it is also an oracle
  valid action.
- Residual-set metrics SHOULD count target-token mismatches and ambiguous vs
  singleton target surfaces.

#### Scenario: First-error mismatch remains explicit
- **WHEN** an atom's selected token differs from the live target-position token
- **AND** the provenance flag is absent
- **THEN** validation fails fast.
- **WHEN** the provenance flag is present
- **THEN** validation succeeds and the valid-set loss supervises the oracle
  positive set.
