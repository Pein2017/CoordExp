# rollout-matching-sft Spec Delta

This is a delta spec for change `add-vllm-repeat-aware-logits-processor`.

## ADDED Requirements

### Requirement: vLLM rollout backend supports repeat-aware per-sequence early termination
When rollout generation uses vLLM server backend (`custom.extra.rollout_matching.rollout_backend: vllm` in rollout-server mode), the system MUST support repeat-aware termination semantics equivalent to the existing HF repeat guard.

Normative behavior:
- This requirement scope is vLLM rollout server mode used by Stage-2 AB; colocate/non-server vLLM paths are out of scope for this change and remain unchanged.
- Repeat-aware termination MUST be controlled by `custom.extra.rollout_matching.repeat_terminate`.
- On the current vLLM V1-default stack, when `repeat_terminate.enabled: true`, vLLM rollout serving MUST activate repeat-aware processing in server mode via startup-time plugin injection (e.g., launching `swift rollout` with `--external_plugins <repo-owned-plugin>`).
  - The plugin MUST attach a repeat-aware logits processor on the server side via vLLM `SamplingParams.logits_processors` (or an equivalent vLLM-native hook) so the learner does not need to inject processors per request.
- The vLLM rollout server MUST receive the full `repeat_terminate` subtree at startup (recommended: `COORDEXP_VLLM_REPEAT_TERMINATE_CONFIG_JSON=<json>` or `COORDEXP_VLLM_REPEAT_TERMINATE_CONFIG_JSON_PATH=<path>`).
- Repeat-aware vLLM rollout termination MUST be implemented without modifying external library source code (e.g., ms-swift or vLLM). A compliant approach is to use `swift rollout --external_plugins` to import a repo-owned plugin at rollout-server startup.
- The processor MUST evaluate the configured thresholds from the full subtree (`enabled`, `min_new_tokens`, `max_consecutive_token_repeats`, `ngram_size`, `ngram_repeats`, and optional `max_object_keys`).
- Triggering repeat-aware termination for one sequence MUST NOT abort or cancel generation for unrelated sequences in the same rollout batch.
- If repeat-aware processing is required by config but cannot be activated in vLLM rollout serving, startup MUST fail fast with actionable diagnostics.

#### Scenario: Offending sequence is terminated without batch abort
- **WHEN** vLLM rollout generation receives a batch and one sequence exceeds configured repeat thresholds
- **THEN** that sequence is forced to EOS on the next decode step
- **AND** remaining sequences continue generation normally in the same batch.

#### Scenario: Config-required repeat-aware processor missing fails fast
- **WHEN** `repeat_terminate.enabled` is true and vLLM rollout server cannot load repeat-aware processing
- **THEN** rollout startup fails before training proceeds
- **AND** the error reports the missing processor activation path.

#### Scenario: vLLM V1 rollout does not rely on request-time logits processors
- **GIVEN** vLLM V1-default rollout serving
- **AND** `repeat_terminate.enabled: true`
- **WHEN** rollout requests are issued
- **THEN** repeat-aware behavior is provided by startup-loaded plugin state (server-side)
- **AND** correctness does not depend on learner-provided per-request `logits_processors` fields.

#### Scenario: Non-server vLLM paths are unchanged by this delta
- **GIVEN** a non-server/colocate vLLM rollout path
- **WHEN** this change is applied
- **THEN** no new repeat-aware contract is imposed by this delta on that path.

### Requirement: Repeat-termination contract is backend-parity and config-first
The rollout-matching contract MUST keep repeat-termination behavior config-first and backend-parity.

Normative behavior:
- The same YAML subtree (`custom.extra.rollout_matching.repeat_terminate`) MUST drive both HF and vLLM guard behavior.
- vLLM mode MUST NOT require new standalone CLI flags for repeat-aware behavior.
- Existing configs that set `repeat_terminate.enabled: true` MUST activate repeat-aware behavior in vLLM mode (i.e., vLLM MUST honor YAML when enabled; no extra knobs are required).
- Legacy “repeat_terminate is HF-only / ignored by vLLM” config or docs statements MUST be removed or updated as part of migration.

#### Scenario: Existing YAML enables repeat-aware behavior in vLLM mode
- **GIVEN** a rollout-matching config with `repeat_terminate.enabled: true`
- **WHEN** rollout backend is switched from HF to vLLM
- **THEN** repeat-aware termination remains enabled without adding new CLI parameters.
