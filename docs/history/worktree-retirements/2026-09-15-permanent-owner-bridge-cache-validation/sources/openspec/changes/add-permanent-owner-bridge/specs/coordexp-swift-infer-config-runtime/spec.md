## ADDED Requirements

### Requirement: Permanent bridge inference identity and backend constraint

Inference configuration SHALL support an explicit permanent-owner-bridge payload path and strict architecture-profile identity in addition to base, adapter, and selected-embedding paths. Runtime MUST validate the bridge manifest and tensor fingerprint, model family, block count, hidden width, tokenizer identity, layer seam, slot/key/value dimensions, parameter names and shapes, and companion payload identities before generation.

A configured permanent bridge MUST use the dynamic HF backend in this change. Bridge-plus-vLLM configuration, a missing bridge payload for a bridge-bound adapter, an extra bridge payload for an incompatible adapter, or an independently overridden layer seam MUST fail before model loading. Runtime MUST NOT infer bridge paths from `checkpoint-final` metadata.

#### Scenario: Complete permanent composition
- **WHEN** a production HF config explicitly names compatible base, adapter, selected-embedding, and bridge payloads
- **THEN** validation MUST resolve one exact composition identity and record every loaded payload before generation

#### Scenario: Bridge checkpoint sent to vLLM
- **WHEN** an inference config combines a permanent bridge with the vLLM backend
- **THEN** validation MUST fail with an unsupported-backend error before materialization or GPU setup

#### Scenario: Bridge payload omitted
- **WHEN** the configured adapter or its declared companion identity requires a permanent bridge but no bridge path is present
- **THEN** runtime MUST fail rather than execute a bridge-free approximation

