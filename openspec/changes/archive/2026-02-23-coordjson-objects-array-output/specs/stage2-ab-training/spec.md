## MODIFIED Requirements

### Requirement: Channel-B reuses rollout-matching infra (strict parse/match + mandatory FN append)
Channel-B MUST reuse the rollout-matching pipeline:
- Rollout generation MUST be configured under `rollout_matching` (backend `hf` or `vllm`).
- Parsing MUST be token-aligned and structure-aware (no re-tokenization of the rollout prefix), except for a possible token-internal cut on the final token where the trainer MAY retokenize only the final token as a shorter tokenization that decodes exactly to the original substring.
- Matching MUST be deterministic.
- FN append MUST be performed (mandatory) to ensure all GT objects are present in `Y_train`.

This pipeline SHALL operate on the model-facing assistant format defined by the CoordJSON contract (top-level `{"objects": [...]}`).
It reuses `SerializeAppend` from rollout-matching, so the appended fragment MUST follow the canonical CoordJSON formatting rules defined in the output-format contract (single-space separators `, ` and `: `; no extra whitespace or indentation) to keep the tokenization that feeds `Y_train` deterministic.

Normative robustness split:
- Predicted rollout records are untrusted: record-level contract violations MUST be handled robustly (drop invalid predicted records; do not raise).
- Deterministic stage-2 serialization (e.g., FN append of GT objects) is trusted: unexpected serialization/contract violations MUST fail fast (raise) rather than silently dropping GT.

#### Scenario: Rollout prefix + FN append produces a valid teacher-forced target
- **GIVEN** Channel-B is selected and rollout generation succeeds
- **WHEN** the trainer builds `Y_train` for teacher forcing
- **THEN** `Y_train` contains the rollout prefix (suffix-trimmed only) followed by a CoordJSON FN append fragment within the `"objects"` array.

### Requirement: Stage-2 serialized object field order follows shared config
Stage-2 AB serialization paths SHALL honor `custom.object_field_order` exactly as stage-1 serialization does.

Scope:
- Channel-A teacher-forced assistant payload construction.
- Channel-B FN append serialization path used to build `Y_train`.

Normative behavior:
- `desc_first`: per-record payload order is `desc` then concrete geometry key (`bbox_2d` or `poly`).
- `geometry_first`: per-record payload order is concrete geometry key (`bbox_2d` or `poly`) then `desc`.
- The serializer MUST NOT emit a synthetic key literally named `geometry`.
- Any deterministic CoordJSON serialization performed by stage-2 (including FN append fragments) MUST follow the canonical CoordJSON formatting rules (single-space separators `, ` and `: `; no indentation; no newlines outside strings; preserve Unicode characters).
- This requirement governs key order within each record and MUST NOT alter the object instance sequence in the `"objects"` array.

#### Scenario: Channel-A uses geometry-first payload when configured
- **GIVEN** `custom.object_field_order: geometry_first`
- **WHEN** Channel-A constructs teacher-forced assistant payload text
- **THEN** each serialized record places its concrete geometry key before `desc`.

#### Scenario: Channel-B uses geometry-first for FN append when configured
- **GIVEN** `custom.object_field_order: geometry_first`
- **AND** Channel-B appends unmatched GT objects
- **WHEN** `Y_train` is constructed
- **THEN** appended records place their concrete geometry key before `desc`
- **AND** matching/masking logic remains unchanged.

#### Scenario: Default desc-first behavior is preserved in both channels
- **GIVEN** `custom.object_field_order` is omitted
- **WHEN** Channel-A or Channel-B serializes record payloads
- **THEN** payloads remain `desc` before the concrete geometry key (`bbox_2d` or `poly`).

### Requirement: Channel-B record-level strict-drop behavior is explicit and measured
When Channel-B parses rollout predictions under the CoordJSON contract, it MUST validate each predicted record in the `"objects"` array against the CoordJSON output-format contract (record schema, geometry arity, CoordTok-only geometry arrays, non-empty `desc`, and configured `custom.object_field_order`).

Normative behavior:
- Predicted records that violate the contract MUST be treated as invalid and MUST be DROPPED (record-level salvage). The sample continues with remaining valid predicted records.
- Dropped predicted records MUST NOT participate in matching, coord-slot supervision, or any “self-context” masks.
- The trainer MUST emit strict-drop diagnostics during Channel-B construction:
  - `stage2_ab/channel_b/strict_drop/N_valid_pred`: number of predicted records retained after validation,
  - `stage2_ab/channel_b/strict_drop/N_drop_invalid`: number of predicted records dropped by validation,
  - `stage2_ab/channel_b/strict_drop/reason/<bucket>`: counts by reason bucket (e.g., `wrong_arity`, `missing_desc`, `order_violation`, `unexpected_keys`).
- Reason-bucket assignment for a record with multiple simultaneous violations MUST be deterministic: implementations SHALL use the following fixed first-match precedence list and apply exactly one reason bucket per dropped record: `unexpected_keys` > `missing_desc` > `order_violation` > `wrong_arity` > `other`.

This requirement applies to per-record validity only. Sample-level container invalidity cases are handled by the invalid-rollout fallback requirement below.

#### Scenario: Invalid predicted record is dropped and measured
- **GIVEN** Channel-B parses a rollout whose `"objects"` array contains at least one invalid record (e.g., wrong geometry arity or missing `desc`)
- **WHEN** the trainer constructs the Channel-B training targets and masks
- **THEN** the invalid record is dropped and does not participate in matching or supervision
- **AND** `stage2_ab/channel_b/strict_drop/N_drop_invalid` increases accordingly.

### Requirement: Channel-B invalid rollouts fall back deterministically (no silent skips)
When Channel-B is selected and a rollout response is sample-level invalid under the CoordJSON top-level container contract (e.g., there is no top-level `{`, the `"objects"` key is missing, `"objects"` is not an array, or extraneous top-level keys are present), the trainer MUST:
- Mark the rollout as invalid for that sample and emit `stage2_ab/channel_b/invalid_rollout` as a deterministic counter/metric key.
- Fall back to a canonical empty CoordJSON prefix of exactly `{"objects": [` (as token ids) so FN append can proceed.
- Treat the rollout as containing zero valid predicted objects for matching/supervision purposes.
- Continue training that sample by FN-appending all GT objects and running the normal teacher-forced loss (i.e., the sample is not skipped and the trainer does not raise).

This fallback MUST be deterministic given the same `response_token_ids` and tokenizer and SHALL align with the CoordJSON transpiler’s salvage parse-failure behavior (returning `{"objects": []}` for downstream strict JSON usage and recording the parse failure) so downstream components treat the sample as having zero valid predicted objects.

Normative minimum: this requirement MUST at least cover the case where the rollout response contains no top-level `{` (equivalent to the current strict parser’s “completely malformed rollout” condition).

#### Scenario: Missing opening brace falls back to `{"objects": [` and trains
- **GIVEN** Channel-B is selected for a sample
- **AND** the rollout response text contains no top-level `{`
- **WHEN** the trainer parses the rollout for matching
- **THEN** it marks the rollout invalid for that sample
- **AND** it uses `{"objects": [` as the prefix and FN-appends all GT objects
- **AND** the sample is still included in teacher-forced training.

### REMOVED Requirements

### Requirement: repeat-terminate / max-object hard stops exist
This requirement is removed from the Stage-2 AB training spec; training SHALL NOT rely on repeat-terminate / max-object hard stops as a defined contract surface.

#### Scenario: (removed) repeat-terminate is not part of the training loop
- **GIVEN** a rollout response that contains repeated text patterns
- **WHEN** Stage-2 AB training processes the rollout response
- **THEN** training does not invoke any repeat-terminate logic and relies on max-length truncation and CoordJSON salvage/strict-drop behavior.

**Reason**: This change deletes repeat-terminate and relies on max-length truncation + record-level salvage dropping for rollouts.
