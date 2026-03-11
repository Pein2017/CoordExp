# rollout-matching-sft Specification (Delta)

## MODIFIED Requirements

### Requirement: Rollout-aware trainers may issue per-call decode overrides
The rollout-matching infrastructure SHALL support call-local decode overrides when a trainer needs multiple rollout policies within the same optimizer step.

Normative behavior:

- the rollout API MUST allow a caller to override decode parameters for a specific rollout request without mutating global config state,
- supported override fields MUST include at least:
  - `decode_mode`
  - `temperature`
  - `top_p`
  - `top_k`
- when no override is provided, existing global decoding semantics remain unchanged.

#### Scenario: Trainer requests greedy and stochastic rollouts in one step
- **WHEN** a rollout-aware trainer issues one anchor rollout and one explorer rollout for the same raw sample set
- **THEN** it may supply distinct per-call decode overrides for the two requests
- **AND** the rollout backend honors those request-local settings without rewriting global config state.

### Requirement: Dual-policy rollout support applies across HF and vLLM backends
When a rollout-aware trainer selects a backend/runtime combination for the canonical v3 Channel-B contract, that combination SHALL either honor the dual-policy rollout contract or fail fast before training.

Normative behavior:

- HF rollout helpers MUST honor the per-call decode overrides above,
- vLLM colocate rollout helpers MUST honor the same per-call decode overrides,
- vLLM server rollout helpers MUST honor the same per-call decode overrides,
- if a configured backend/runtime cannot honor greedy anchor plus stochastic explorer requests, trainer initialization MUST fail fast with actionable guidance.

#### Scenario: Unsupported dual-policy backend fails fast
- **WHEN** a rollout backend/runtime combination cannot honor both greedy anchor and stochastic explorer policies
- **THEN** Stage-2 trainer initialization fails fast before training starts
- **AND** the error identifies the unsupported backend/runtime requirement.
