# teacher-forcing-unified-loss-registry Specification

## Purpose
Define the canonical teacher-forcing token roles, active Stage-2 rollout
correction loss component names, and retired-module boundaries shared by the
typed teacher-forcing objective architecture.

## Requirements

### Requirement: Canonical token roles are shared
The system SHALL use a small shared vocabulary for teacher-forcing token roles.

Normative token roles:
- `struct`: schema, punctuation, keys, delimiters, and other non-desc,
  non-coord content.
- `desc`: free-text object description tokens.
- `coord`: coordinate vocabulary tokens `<|coord_k|>`.
- `eos`: end token `<|im_end|>` for Qwen3-VL training/eval contracts.

Active Stage-2 rollout correction uses rollout prefix as roll-in context and
supervises only the residual/GT correction target IR selected by
`residual_set_correction`.

### Requirement: Active Stage-2 loss component is residual-set correction
The unified Stage-2 contract SHALL enable exactly one Stage-2 objective module:
`residual_set_correction`.

Normative active component names:
- `type_exclusive`: global token-type mass objective owned by residual-set
  correction.
- `valid_set_marginal`: next-token valid-set marginal likelihood owned by
  residual-set correction.
- `coverage`: optional alpha-weighted within-valid coverage regularizer owned
  by residual-set correction.
- `continuation_margin`: optional valid-continuation-vs-EOS calibration owned by
  residual-set correction.

Removed Stage-2 module names:
- `token_ce`
- `hard_sft`
- `stage2_trie_ce`
- `loss_duplicate_burst_unlikelihood`
- `geo`
- `bbox_geo`
- `bbox_size_aux`
- `coord_reg`
- `coord_gate`
- `text_gate`

#### Scenario: Removed Stage-2 modules fail fast
- **WHEN** a `stage2_rollout_correction.pipeline.objective[]` entry declares any
  removed module name
- **THEN** config validation fails before trainer initialization
- **AND** the error tells the author to use `residual_set_correction` with
  `application.preset: rollout_self_prefix`.

### Requirement: Rollout tokens are roll-in context only
The Stage-2 rollout prefix SHALL NOT automatically become positive labels.

Normative behavior:
- residual-state `ValidAction` records define the positive next-token set at
  each correction position,
- a live rollout token that is not in the oracle valid-action set MUST NOT be
  inserted into `valid_token_ids`,
- first-error atoms MAY supervise an oracle selected token that differs from
  `input_ids[target_position]`, but only when atom provenance explicitly marks
  the target-token mismatch.

#### Scenario: Wrong live token is corrected instead of reinforced
- **GIVEN** a rollout prefix contains a wrong token at a correction position
- **WHEN** residual-set target IR is built
- **THEN** the wrong rollout token remains context only
- **AND** the oracle valid-action set excludes that wrong token unless it is
  independently valid under the residual state.
