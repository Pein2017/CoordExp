## 1. Spec And Governance

- [x] 1.1 Keep this OpenSpec change limited to stable type-family gating.
- [x] 1.2 Keep coverage ledger, continuation, geometry penalties, adapter
  inference, and 128-sample smoke launches out of this main implementation.
- [x] 1.3 Run review-convergence on the OpenSpec change and super-power plan
  before source implementation. Required reviewer lanes are OpenSpec
  governance, implementation/code fit, metric semantics, and workflow safety.
  Timeout or disconnection is unresolved, not approval; any open P0/P1 blocks
  source edits.

## 2. Config Contract

- [x] 2.1 Replace the generic enabled-only
  `objective.terms.token_type_mass` parser with a term config that accepts only
  `enabled: bool` and `weight: nonnegative finite float`.
- [x] 2.2 Reject `objective.terms.token_type_mass.mode` and any unknown key.
- [x] 2.3 Allow `objective.profile: hard_sft` to enable
  `objective.terms.token_type_mass`.
- [x] 2.4 Keep `hard_sft` rejection for
  `conditional_valid_set_likelihood`, `within_valid_coverage`, and
  `continuation_margin`.
- [x] 2.5 Persist `token_type_mass.weight` in the detection objective runtime
  payload.

## 3. Family Vocabulary

- [x] 3.1 Update compact token groups so schema includes all compact structural
  tokens known to the tokenizer: `<|object_ref_start|>`,
  `<|object_ref_end|>`, `<|box_start|>`, `<|box_end|>`, and compact row
  separators when present.
- [x] 3.2 Keep the coordinate family exactly `<|coord_0|>` through
  `<|coord_999|>`.
- [x] 3.3 Keep the stop family exactly `<|im_end|>`.
- [x] 3.4 Build the description family from tokenizer vocabulary after
  excluding schema, coord, stop, pad, unknown, and control-special tokens.
- [x] 3.5 Validate that the four active families are mutually exclusive and
  non-empty for compact teacher-forcing.

## 4. Loss Semantics

- [x] 4.1 Replace allowed-union type loss with a four-family
  logsumexp-over-family-mass objective.
- [x] 4.2 At each supervised atom, select the target family from
  `atom.selected_token_role`.
- [x] 4.3 Compute family logits as the logsumexp of full-vocabulary log-probs
  for `schema`, `coord`, `desc`, and `stop`.
- [x] 4.4 Normalize only across the four family logits, excluding pad/control
  tokens and out-of-family tokens from the type-loss denominator.
- [x] 4.5 Add `token_type_mass.weight * type_loss` to the existing atom loss
  when the term is enabled.
- [x] 4.6 Preserve ordinary hard-token CE and target-IR valid-set behavior; this
  term only changes family mass pressure.

## 5. Metrics

- [x] 5.1 Emit `teacher_forcing/loss/token_type_mass` as the raw mean type loss.
- [x] 5.2 Emit `teacher_forcing/loss/token_type_mass/contribution` as
  `weight * raw_mean`.
- [x] 5.3 Emit family-health means:
  - `teacher_forcing/type/schema_mass_at_schema`
  - `teacher_forcing/type/coord_mass_at_coord`
  - `teacher_forcing/type/desc_mass_at_desc`
  - `teacher_forcing/type/stop_mass_at_stop`
- [x] 5.4 Do not emit `_weighted` metric names for this term.

## 6. Verification

- [x] 6.1 Add unit tests for family partitioning, including closed compact
  structural tokens and Qwen stop-token separation.
- [x] 6.2 Add probability tests proving the loss rewards the selected family and
  penalizes the other three family masses.
- [x] 6.3 Add config tests for `hard_sft + token_type_mass enabled`,
  unknown-key rejection, and mode-key rejection.
- [x] 6.4 Add trainer/objective tests for raw and contribution metric keys,
  exact raw/contribution values, family-specific denominator omission, and
  out-of-family denominator exclusion.
- [x] 6.5 Add metric-contract tests for
  `REQUIRED_TEACHER_FORCING_METRIC_KEYS`, flattened metric values, and absence
  of `_weighted` keys.
- [x] 6.6 Run:
  `python -m pytest tests/test_compact_type_gate.py tests/test_teacher_forcing_objective_runner.py tests/test_teacher_forcing_metric_contract.py tests/test_teacher_forcing_config_contract.py tests/test_detection_training_config_contract.py -q`
- [x] 6.7 Run the shared objective regression guard:
  `python -m pytest tests/test_objective_runner_math.py tests/test_metric_events.py -q`
- [x] 6.8 Run `git diff --check`.
- [x] 6.9 Run a path-scoped status/diff check over only the implementation
  targets before any staging or handoff. Do not use `git add -A` or
  `git commit -a` for this split.
- [x] 6.10 Update `docs/training/METRICS.md` with the promoted metric keys and
  semantics when source implementation changes metric behavior.
- [x] 6.11 If the OpenSpec CLI becomes available, run
  `openspec validate bidirectional-type-gating-losses --type change --strict`;
  otherwise record that CLI validation is blocked by the missing binary.
