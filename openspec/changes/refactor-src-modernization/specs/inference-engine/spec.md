# inference-engine Spec Delta

This is a delta spec for change `refactor-src-modernization`.

## ADDED Requirements

### Requirement: HF attention backend selection is resilient across environments
Inference-engine SHALL support resilient HF attention backend selection.
If a preferred backend is unavailable in the runtime environment, engine initialization MUST fall back to a supported backend with explicit diagnostics while preserving output contract semantics.
The selected backend (including fallback choice when applied) MUST be recorded in run artifacts via exact `summary.json` fields:
- `backend.attn_implementation_requested`
- `backend.attn_implementation_selected`

`resolved_config.json` MAY mirror these values, but `summary.json` fields are the required compatibility surface for this contract.

#### Scenario: Missing preferred attention backend falls back with warning
- **GIVEN** HF backend inference configuration prefers an unavailable attention backend
- **WHEN** model initialization runs
- **THEN** the engine selects a supported fallback backend
- **AND** emits explicit diagnostics without changing output artifact schema.

#### Scenario: Selected attention backend is captured in run artifacts
- **GIVEN** inference runs under either preferred or fallback attention backend
- **WHEN** artifacts are persisted
- **THEN** `summary.json.backend.attn_implementation_requested` and `summary.json.backend.attn_implementation_selected` are present
- **AND** operators can determine from artifacts whether fallback occurred by comparing requested vs selected values.

### Requirement: Backend runtime is selected through an explicit backend contract
Inference-engine SHALL use an explicit backend runtime contract to isolate backend-specific generation details from artifact standardization.
All backend runtimes MUST produce standardized prediction payloads consumed by shared post-processing.

#### Scenario: Backend runtime swap preserves standardized output payload
- **GIVEN** equivalent inputs and generation settings
- **WHEN** backend runtime selection changes via config
- **THEN** standardized output payload fields remain compatible with shared post-processing and artifact writers.

### Requirement: Inference error reporting remains structured and sample-scoped
Inference-engine SHALL preserve structured, per-sample error reporting in output artifacts and summary counters during internal refactor.
Internal exceptions MUST map to explicit sample error entries rather than silent skips.

#### Scenario: Sample-level generation failure is reflected in structured errors
- **GIVEN** generation fails for one sample
- **WHEN** inference continues for the batch/run
- **THEN** the failed sample includes a structured error entry
- **AND** summary counters include the failure classification.
