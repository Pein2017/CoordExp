## MODIFIED Requirements

### Requirement: Composition fidelity and behavioral diagnosis

The system SHALL resolve one immutable execution model for the configured base,
optional DoRA adapter, and optional selected-token embedding delta. Any
configured adapter or embedding delta MUST be materialized into a standard HF
checkpoint before vLLM loading. DoRA MUST be merged first and the selected-
token delta MUST be folded exactly once into the tied input/output weight.

The blocking execution receipt MUST bind and validate the current source
payload identities, tensor compatibility, target dtype, merge/fold outcomes,
tied-weight structure, standard Qwen3-VL snapshot files, absence of adapter
residue, exhaustive snapshot identity, and atomic cache publication. A
composition-fidelity or HF/vLLM behavioral-comparison receipt MUST NOT be
required for ordinary vLLM loading.

Before publication, adapter merge evidence MUST exist exactly when an adapter
identity is configured and MUST bind that identity with status `merged`.
Embedding-delta fold evidence MUST exist exactly when a delta identity is
configured and MUST bind that identity, status `folded`, one row addition, and
tied input/output storage. The complete materialization MUST also attest tied
input/output structure. Missing, unexpected, or identity-mismatched outcomes
MUST prevent cache publication.

Explicit composition probes MAY bind and validate exact merged-target,
selected-row, prompt, logit, or generated-token comparisons as optional
diagnostics. A failed or stale optional comparison MUST NOT invalidate a
structurally valid execution model unless it demonstrates that the current
materialized snapshot violates one of the blocking composition invariants.

#### Scenario: Valid composition without comparison sidecar

- **WHEN** base, DoRA, and embedding delta materialize into a structurally
  valid tied Qwen3-VL snapshot but no composition-fidelity sidecar exists
- **THEN** ordinary vLLM runtime accepts the execution-model receipt and
  proceeds to engine loading

#### Scenario: Stale behavioral comparison

- **WHEN** an old HF comparison receipt is missing, source-stale, or exceeds a
  historical numerical tolerance while the current structural receipt passes
- **THEN** the condition is recorded only when inspected diagnostically
- **AND** does not prevent ordinary vLLM execution

#### Scenario: Invalid selected-token fold

- **WHEN** selected-token ids, shapes, dtype, base/tokenizer identity, or tied
  input/output structure is incompatible during materialization
- **THEN** materialization fails and no completed execution snapshot is
  published

#### Scenario: Dynamic and materialized BF16 behavior differs

- **WHEN** current structural composition checks pass but dynamic and
  materialized HF logits or greedy ids differ
- **THEN** an explicit comparison records the behavioral difference without
  rejecting the structurally valid execution model

#### Scenario: Reloaded merged target differs

- **WHEN** a reloaded materialized DoRA target differs from the merge outcome
  bound by the current execution receipt
- **THEN** the execution model is rejected before vLLM loading

#### Scenario: Folded selected row differs after target-dtype cast

- **WHEN** a selected embedding row does not equal the current one-addition
  fold outcome after target-dtype casting
- **THEN** materialization fails before completed snapshot publication

#### Scenario: Clean cache reuses durable FP32 proof

- **WHEN** a clean checkout reuses a structurally valid content-addressed cache
  entry and a linked durable FP32 comparison is available
- **THEN** runtime may expose that comparison as diagnostic evidence
- **AND** cache acceptance remains based on the current structural receipt
