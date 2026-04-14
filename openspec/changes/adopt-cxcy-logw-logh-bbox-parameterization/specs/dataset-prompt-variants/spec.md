## ADDED Requirements

### Requirement: Dense prompt variants use structured bbox-parameterization placeholders
Dense prompt variants SHALL express bbox-parameterization-dependent wording
through structured placeholders or template fragments rather than ad hoc
free-text rewriting.

Normative behavior:

- override-style dense prompt variants MUST provide explicit placeholder seams
  for:
  - bbox parameterization wording/examples
  - object field order wording/examples
- prompt resolution MUST substitute those placeholders after variant selection,
- missing required placeholders MUST fail fast with actionable diagnostics.

#### Scenario: Override variant renders cxcy-logw-logh wording
- **WHEN** an override-style prompt variant is resolved with
  `bbox_format=cxcy_logw_logh`
- **THEN** the final prompt text uses the structured cxcy-logw-logh fragment
- **AND** it does not rely on brittle free-text replacement.

#### Scenario: Missing bbox placeholder fails fast
- **WHEN** an override-style prompt variant omits the required bbox
  parameterization placeholder
- **THEN** prompt resolution fails fast before training or inference starts
- **AND** the error identifies the missing placeholder contract.

#### Scenario: Missing object-field-order placeholder fails fast
- **WHEN** an override-style prompt variant omits the required object-field-order
  placeholder
- **THEN** prompt resolution fails fast before training or inference starts
- **AND** the error identifies the missing placeholder contract.

## MODIFIED Requirements

### Requirement: Variant Registry and Deterministic Resolution
The system SHALL provide a centralized prompt-variant registry for dense
detection prompts and SHALL resolve prompt variants by explicit key with
deterministic behavior.

Prompt resolution inputs for dense prompts MUST include:
- prompt variant key
- ordering policy
- object field order
- bbox format
- coord mode

#### Scenario: Deterministic repeated resolution
- **WHEN** the same variant key is resolved repeatedly with the same prompt
  inputs
- **THEN** the resolver returns byte-identical system and user prompt text
  across calls.

### Requirement: Prompt text reflects Stage-1 cxcy-logw-logh serialization
The shared dense prompt resolver SHALL describe `cxcy_logw_logh` as
`bbox_2d: [cx, cy, u(w), u(h)]` on model-facing surfaces, with `u(*)` defined
as the shared log-size expression.

Normative behavior:

- Stage-1 dense prompts MUST resolve variant, ordering, object field order,
  bbox format, and coord mode from the active training config,
- standalone inference dense prompts MUST resolve:
  - variant from `infer.prompt_variant`
  - ordering policy from `infer.object_ordering`
  - object field order from `infer.object_field_order`
  - bbox format from `infer.bbox_format`
  - coord mode from the canonical dense resolver mode `coord_tokens`
- trainer-driven rollout/eval prompt rebuilding MUST reject
  `bbox_format=cxcy_logw_logh` for this V1 change,
- when `bbox_format=cxcy_logw_logh`, prompt wording MUST make clear that the
  third and fourth slots represent `u(w), u(h)` in the shared log-size
  expression rather than raw `x2, y2`, raw linear `w, h`, or unnormalized
  `log(w), log(h)`.

#### Scenario: Stage-1 prompt applies cxcy-logw-logh wording
- **WHEN** `custom.bbox_format=cxcy_logw_logh`
- **THEN** the Stage-1 prompt instructs the model to emit
  `bbox_2d: [cx, cy, u(w), u(h)]`
- **AND** it defines `u(*)` as the shared log-size expression.

#### Scenario: Inference prompt matches Stage-1 bbox semantics
- **WHEN** `infer.bbox_format=cxcy_logw_logh`
- **THEN** inference message construction applies the same cxcy-logw-logh
  semantics as Stage-1 prompt resolution.

#### Scenario: Rollout/eval prompt rebuilding rejects cxcy-logw-logh
- **WHEN** trainer-driven rollout/eval prompt rebuilding requests
  `bbox_format=cxcy_logw_logh`
- **THEN** prompt resolution fails fast with guidance that V1 is limited to
  Stage-1 and standalone inference.

### Requirement: Inference artifacts record the full prompt identity needed for audit
Inference artifacts SHALL include the resolved prompt identity for reproducible
auditing.

#### Scenario: Resolved config records prompt identity
- **WHEN** an inference pipeline run emits `resolved_config.json`
- **THEN** the file MUST record:
  - the resolved prompt variant
  - the resolved dense ordering policy
  - the resolved object field order
  - the resolved bbox format
  - the resolved coord mode when available
  - the resolved prompt/template hash or equivalent digest.

#### Scenario: Summary metadata records prompt identity
- **WHEN** an inference pipeline run emits summary metadata
- **THEN** the summary MUST record:
  - the resolved prompt variant
  - the resolved dense ordering policy
  - the resolved object field order
  - the resolved bbox format
  - the resolved prompt/template hash or equivalent digest.
