# stage2-ab-training Spec Delta

This is a delta spec for change `add-vllm-repeat-aware-logits-processor`.

## ADDED Requirements

### Requirement: Channel-B vLLM rollouts honor repeat-aware termination settings
When Stage-2 AB Channel-B performs rollouts through vLLM rollout server backend, repeat-aware termination MUST be applied according to rollout-matching config.

Normative behavior:
- Channel-B rollout path MUST propagate the full `custom.extra.rollout_matching.repeat_terminate` subtree (`enabled`, `min_new_tokens`, `max_consecutive_token_repeats`, `ngram_size`, `ngram_repeats`, optional `max_object_keys`) into the active vLLM rollout serving startup path.
- Because the rollout server is launched as a separate process (external dependency stack), the full subtree MUST be transmitted into that server startup process.
  - Recommended compliant approach: the server launcher:
    - sets `COORDEXP_VLLM_REPEAT_TERMINATE_CONFIG_JSON=<json>` and enables injection with `COORDEXP_ENABLE_VLLM_REPEAT_TERMINATE_INJECTION=1`, and
    - launches `swift rollout` with `--external_plugins <repo-owned-plugin>` so the server can attach repeat-aware processing at startup without external library source edits.
- For this stack, Channel-B MUST NOT assume request-time logits-processor fields in rollout request payloads; repeat-aware activation is validated at rollout-server startup.
- If `repeat_terminate.enabled: true` and startup activation is unavailable, Stage-2 AB MUST fail before entering training steps.
- Channel-B MUST preserve FP/matching contracts and MUST NOT change geometry supervision semantics due to repeat-aware processing.
- Channel-B logs/metrics MUST emit concrete audit keys (as entries in the neutral trainer-metrics payload `metrics` map; see `src/metrics/payload_contract.py`):
  - `rollout/repeat_terminate_active` (0 or 1),
  - `rollout/repeat_terminate_triggered_sequences` (counter).
  - Metric meaning (normative):
    - `rollout/repeat_terminate_active`: 1 iff repeat-aware processing is active for the step under the current rollout backend/mode when `repeat_terminate.enabled: true`; otherwise 0.
    - `rollout/repeat_terminate_triggered_sequences`: number of rollout sequences in the step for which repeat-aware processing **triggered at least once** and forced EOS due to configured repeat thresholds.
      - This key MUST be derived from an explicit trigger signal produced by the repeat-aware processor/server stack (not inferred from finish-reason heuristics alone).
- The vLLM server `/infer/` response MUST expose the explicit per-sequence trigger signal in an additive-only wrapper envelope.
  - A compliant per-output schema is:
    - `{"response": <ChatCompletionResponse-dict>, "coordexp": {"repeat_terminate_triggered": 0|1}}`
  - The wrapper MUST be additive-only:
    - it MUST NOT remove or rename fields within the inner `response` payload compared to the unwrapped server output,
    - and it MUST preserve the detail fields required for strict alignment and token-aligned parsing (at minimum `prompt_token_ids` and `choices[0].token_ids` when `request_config.return_details: true`).
  - The learner MUST compute `rollout/repeat_terminate_triggered_sequences` from this wrapper signal (sum of `repeat_terminate_triggered` across sequences in the step), not from stop-reason heuristics.
- All metrics emitted via the neutral trainer-metrics payload `metrics` map (see `src/metrics/payload_contract.py`) MUST be emitted as **global** aggregates after:
  - micro-batch/gradient-accumulation aggregation to one optimizer-step payload, and
  - distributed aggregation across ranks (e.g., DDP all-reduce) when `world_size > 1`.
  - Global aggregation semantics are defined per metric family (normative):
    - counters: global sum,
    - wall-time seconds (e.g., `time/*_s`): global max,
    - boolean-style activation flags: global max,
    - rates: ratio of globally-summed numerator/denominator (never mean of rank-local ratios).
  - For this requirement's audit keys:
    - `rollout/repeat_terminate_active` MUST remain in `{0,1}` after global aggregation (a compliant approach is a global max over rank-local 0/1 values).
    - `rollout/repeat_terminate_triggered_sequences` MUST be a non-negative global counter for the step (a compliant approach is a global sum over rank-local counts).
- For auditability, Channel-B rollout steps MUST emit tail-control metrics (as entries in the neutral trainer-metrics payload `metrics` map; see `src/metrics/payload_contract.py`):
  - `rollout/gen_new_tokens_p99`,
  - `rollout/parse_truncated_rate`,
  - `rollout/parse_dropped_invalid`.
  - Metric definitions/aggregation (normative, distributed):
    - `rollout/parse_truncated_rate` MUST be computed as `(sum(num_truncated_samples) / sum(num_rollout_samples))` over ranks for the step (0 when `sum(num_rollout_samples) == 0`).
    - `rollout/gen_new_tokens_p99` MUST be computed as a global conservative proxy using only all-reduce:
      - compute rank-local p99 over that rank's rollout samples for the step,
      - then compute the global metric as `max(rank_local_p99)` via all-reduce max.
      - Rationale: this preserves a simple, reproducible global metric without requiring all-gather of variable-length lists.

#### Scenario: Channel-B run in vLLM mode uses repeat-aware termination
- **GIVEN** Stage-2 AB with Channel-B and vLLM rollout backend
- **AND** `repeat_terminate.enabled: true`
- **WHEN** a rollout sequence enters degenerate repetition
- **THEN** Channel-B rollout output is terminated early for that sequence by repeat-aware logic
- **AND** downstream parse/match training continues for the batch.

#### Scenario: Channel-B startup fails when repeat-aware contract is enabled but inactive
- **GIVEN** Stage-2 AB Channel-B with vLLM backend
- **AND** `repeat_terminate.enabled: true`
- **WHEN** rollout server startup cannot activate repeat-aware processing
- **THEN** trainer startup fails with an error that reports the missing processor activation path
- **AND** no training step is executed.

#### Scenario: Tail-control audit metrics are emitted
- **GIVEN** Stage-2 AB with Channel-B and vLLM rollout backend
- **WHEN** a Channel-B rollout step executes
- **THEN** logs include `rollout/gen_new_tokens_p99`, `rollout/parse_truncated_rate`, and `rollout/parse_dropped_invalid`.
