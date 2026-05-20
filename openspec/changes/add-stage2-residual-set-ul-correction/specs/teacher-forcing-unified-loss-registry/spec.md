## ADDED Requirements

### Requirement: Residual-set correction components are registry-visible

The teacher-forcing loss registry SHALL expose residual-set correction
components and provenance weights without reintroducing retired geometry or
duplicate-specific losses.

Normative behavior:

- Stage-2 objective module names SHALL include `residual_set_correction` when
  the residual-set correction path is enabled.
- The registry MUST include a residual-set valid-action marginal component for
  Stage-2 self-prefix correction.
- The registry MUST keep token-type exclusivity separate from inner valid-action
  objectives.
- `ul_promoted_local` MUST have a separately configurable provenance weight,
  defaulting to `0.5`.
- Labeled GT and UL-promoted contributions MUST be reportable separately.
- Ambiguous residual-set marginal atoms SHALL use the scalar atom-weight policy
  defined by the residual-set correction spec: mixed labeled/UL support uses
  weight `1.0`; UL-only support uses `lambda_ul_promoted`.
- Removed component/module names such as `loss_duplicate_burst_unlikelihood`,
  `bbox_geo`, `bbox_size_aux`, `coord_reg`, `coord_gate`, and `text_gate` MUST
  remain rejected.

#### Scenario: UL provenance weight is visible

- **WHEN** a residual-set atom is produced from `ul_promoted_local`
- **THEN** its effective loss weight reflects the configured UL provenance
  weight
- **AND** metrics can distinguish UL-promoted contribution from labeled-GT
  contribution.

#### Scenario: Mixed labeled and UL marginal is weighted deterministically

- **WHEN** one residual-set valid-action marginal includes both labeled-GT and
  UL-promoted candidates
- **THEN** the atom uses the mixed-support scalar weight defined by the
  residual-set correction spec
- **AND** labeled and UL valid masses remain separately reportable.

#### Scenario: Residual-set path does not restore duplicate loss

- **WHEN** the residual-set objective path handles a repeated-object boundary
- **THEN** the live loss registry still omits `loss_duplicate_burst_unlikelihood`
- **AND** the boundary correction uses valid-action marginal over remaining
  objects or STOP.

### Requirement: STOP and continuation diagnostics remain separate from core valid actions

The registry SHALL treat continuation-vs-STOP margin as an optional component
that does not change residual-state-machine valid actions.

Normative behavior:

- Default continuation-margin weight MUST be `0.0`.
- STOP and continuation metrics MUST be reportable even when the margin is
  disabled.
- Enabling a margin MUST NOT place STOP and continuation in the same core valid
  action set.

#### Scenario: Margin disabled still reports STOP diagnostics

- **WHEN** `lambda_continue_margin = 0.0`
- **THEN** residual-set runs may still report `log P(valid_continue)`,
  `log P(STOP)`, and `valid_vs_stop_margin`
- **AND** no continuation-margin loss contribution is emitted.

## MODIFIED Requirements

### Requirement: Stage-2 text objectives and typed-trie objectives remain separable
Stage-2 two-channel legacy objectives SHALL remain text/structure-only, while
the unified typed-trie and residual-set correction objectives SHALL own
token-type exclusivity, valid-set/valid-action marginal likelihood, optional
coverage, optional continuation calibration, and residual-set correction
semantics when explicitly selected.

Normative behavior:
- Stage-2 AB active pipeline objective module names include `token_ce`,
  `hard_sft`, `stage2_trie_ce`, and `residual_set_correction`.
- `residual_set_correction` SHALL be explicit opt-in and MUST NOT silently mutate
  the behavior of `token_ce`, `hard_sft`, or `stage2_trie_ce` baselines.
- New typed teacher-forcing objective configs MAY introduce typed-trie or
  residual-set module names, but MUST NOT reuse removed legacy bbox/coord module
  names.
- The hard-SFT baseline MUST remain available as an explicit ablation surface.

#### Scenario: Hard SFT remains a baseline
- **WHEN** a config selects the hard-SFT baseline
- **THEN** the objective supervises the selected teacher path
- **AND** it does not silently enable valid-set marginal, coverage,
  residual-set correction, or retired bbox/coord auxiliaries.

#### Scenario: Residual-set correction is explicit opt-in
- **WHEN** a config does not select `residual_set_correction`
- **THEN** Stage-2 objective behavior remains governed by the selected baseline
  module
- **AND** residual-set self-prefix correction is not enabled implicitly.
