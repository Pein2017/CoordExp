## ADDED Requirements

### Requirement: Offline inference uses the shared inference runtime

The unified inference entrypoint SHALL route detection generation through the
shared inference runtime instead of owning separate prompt, backend, trace, or
parser implementations.

Normative behavior:

- `scripts/run_infer.py` remains the offline entrypoint;
- authored offline config remains under `infer.*`;
- offline generation MUST construct shared prompt/decode/model policy objects;
- offline artifacts MUST record prompt, decode, model identity, and score
  policy provenance as applicable;
- offline backend selection MUST use the shared backend adapter contract;
- offline inference MUST preserve the canonical output schema required by the
  existing inference-engine contract.

#### Scenario: Offline run records shared runtime fingerprints

- **GIVEN** an offline inference YAML with `infer.backend.type: hf`
- **WHEN** `scripts/run_infer.py --config ...` completes
- **THEN** `summary.json` and resolved metadata include
  `prompt_policy_fingerprint`, `decode_policy_fingerprint`, and
  `model_identity_fingerprint`
- **AND** `gt_vs_pred.jsonl` remains schema-compatible with the existing
  inference-engine output contract.

### Requirement: Offline vLLM logprob tracing follows the shared result contract

Offline inference SHALL require the shared generated-sequence trace contract
when `infer.generation.trace_logprobs` or the equivalent resolved trace flag is
enabled.

Normative behavior:

- vLLM backends MUST request generated-token logprobs for trace-required runs;
- missing or malformed vLLM logprob payloads MUST fail fast before
  metric-bearing artifacts are written;
- trace validation MUST be backend-agnostic and must not silently clip or pad
  trace arrays;
- prompt logprobs remain optional unless explicitly requested.

#### Scenario: Offline vLLM trace failure stops before metric artifacts

- **GIVEN** an offline run selects vLLM and enables generated-token logprob
  tracing
- **WHEN** the vLLM response lacks aligned generated-token logprobs
- **THEN** the run fails before writing a comparable `gt_vs_pred.jsonl`
- **AND** the diagnostic identifies the trace contract failure.
