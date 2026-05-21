## ADDED Requirements

### Requirement: Residual-set loss components are registry-visible

The teacher-forcing loss registry SHALL expose residual-set correction
components without reintroducing retired geometry or duplicate-specific losses.

Normative behavior:

- The registry MUST include a standalone `type_exclusive` component for global
  schema/text/coord mass supervision.
- The registry MUST explicitly map residual-set conceptual groups to existing
  registry token names where they differ: schema/control to `struct`, free text
  to `desc`, coordinate to `coord`, and STOP/EOS as the configured `<|im_end|>`
  singleton role/group.
- The registry MUST include a residual-set valid-set marginal component for
  self-prefix correction atoms.
- The registry MUST allow singleton valid-set cases to represent hard CE, EOS,
  deterministic template schema, and template-rendered newline.
- The registry MUST keep token-type exclusivity separate from inner valid-set
  objectives.
- The registry MUST NOT restore `loss_duplicate_burst_unlikelihood`,
  `bbox_geo`, `bbox_size_aux`, `coord_reg`, `coord_gate`, or `text_gate`.

#### Scenario: Type loss and inner loss are separate

- **WHEN** a residual-set atom is trained
- **THEN** type exclusivity computes global correct-token-type mass
- **AND** valid-set marginal computes the inner target over the atom's valid
  token ids
- **AND** neither component reimplements the other's responsibility.

### Requirement: Residual-set atom weighting and normalization are mean-like

The teacher-forcing loss registry SHALL keep residual-set losses mean-like and
comparable across sequences with different atom counts.

Normative behavior:

- Atom weights SHALL be provided by target builders and consumed as scalar
  weights by loss modules.
- Per-rollout sequence loss SHALL be a weighted mean over active atoms.
- Batch loss SHALL be the mean of retained sequence losses.
- Clean, dirty, GT, UL, EOS, continuation, and label-conflict groups SHALL be
  diagnostics only and MUST NOT add a second normalization layer.
- Metrics SHOULD expose active atom count, atom weight sum, raw atom loss sum,
  and sequence loss.

#### Scenario: UL and dirty weights affect atoms but not bucket means

- **WHEN** a sequence contains GT-clean, GT-dirty, and UL atoms
- **THEN** each atom uses its scalar weight
- **AND** the sequence is normalized once by total atom weight.

### Requirement: Residual-set provenance weights are explicit

The registry SHALL preserve target provenance through explicit scalar weights
and diagnostics.

Normative defaults:

- labeled GT weight: `1.0`;
- promoted UL weight: `0.5`;
- spatial wrong-description label-conflict multiplier: `0.25`;
- dirty-prefix fallback uses the configured fallback context weight;
- mixed provenance atoms use the final scalar weight provided by the target
  builder.

Normative behavior:

- UL contributions MUST be reportable separately from labeled GT.
- Spatial wrong-description conflicts MUST be reportable separately from
  ordinary wrong-desc corrections.
- Duplicate-burst diagnostics MUST NOT appear as live duplicate-specific loss.

#### Scenario: Spatial wrong-desc conflict is downweighted

- **WHEN** a high-IoU wrong-description row produces a desc correction atom
- **THEN** the atom's weight is multiplied by `label_conflict_weight`
- **AND** metrics can separate it from ordinary wrong-description corrections.

## MODIFIED Requirements

### Requirement: Stage-2 text objectives and typed-trie objectives remain separable

Stage-2 two-channel legacy objectives SHALL remain selectable baselines, while
the residual-set correction objective SHALL own self-prefix residual-set
semantics only when explicitly selected.

Normative behavior:

- Stage-2 AB active pipeline objective module names include `token_ce`,
  `hard_sft`, `stage2_trie_ce`, and `residual_set_correction`.
- `residual_set_correction` is explicit opt-in and MUST NOT silently mutate
  `token_ce`, `hard_sft`, or `stage2_trie_ce` behavior.
- New residual-set configs MUST NOT reuse removed legacy bbox/coord module
  names.
- The hard-SFT baseline MUST remain available as an explicit ablation surface.

#### Scenario: Baseline hard SFT remains clean

- **WHEN** a config selects hard SFT
- **THEN** it supervises the selected teacher path
- **AND** it does not implicitly enable residual-set correction, UL mining,
  type exclusivity, or removed bbox/coord losses.
