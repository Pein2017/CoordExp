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
Channel-B pseudo-positive anchors SHALL be handled as accepted-prefix context,
not as a way to reactivate retired coord/bbox regularizers.

Normative behavior:
- `dead_anchor` objects MUST NOT appear in the positive teacher-forced prefix.
- Retained prefix objects MAY receive global rollout-prefix structure
  supervision as defined by the Channel-B contract.
- `pseudo_positive` objects MAY remain in the edited anchor prefix for accepted
  structure context.
- `pseudo_positive`, `shielded_anchor`, and dead unmatched anchors MUST NOT
  contribute desc-positive supervision unless explicitly represented as
  `matched_clean`, `fn`, or `recovered_fn`.
- `pseudo_positive` anchors MUST NOT activate `bbox_geo`, `bbox_size_aux`,
  `coord_reg`, `coord_gate`, or `text_gate`.

#### Scenario: Pseudo-positive spans are desc-neutral
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** selected pseudo-positive anchors may remain as structure/context
  prefix content
- **AND** they do not contribute desc CE
- **AND** they do not activate retired coord/bbox regularizer terms.

### Requirement: Stage-2 text objectives and typed-trie objectives remain separable
Stage-2 two-channel legacy objectives SHALL remain text/structure-only, while
the unified typed-trie objective SHALL own token-type exclusivity, valid-set
marginal likelihood, optional coverage, and optional continuation calibration.

Normative behavior:
- Stage-2 AB active pipeline objective module names are limited to `token_ce`,
  `hard_sft`, and `stage2_trie_ce`.
- New typed teacher-forcing objective configs MAY introduce typed-trie module
  names, but MUST NOT reuse removed legacy bbox/coord module names.
- The hard-SFT baseline MUST remain available as an explicit ablation surface.

#### Scenario: Hard SFT remains a baseline
- **WHEN** a config selects the hard-SFT baseline
- **THEN** the objective supervises the selected teacher path
- **AND** it does not silently enable valid-set marginal, coverage, or retired
  bbox/coord auxiliaries.
