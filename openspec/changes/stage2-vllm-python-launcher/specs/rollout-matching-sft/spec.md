# rollout-matching-sft Specification (Delta)

## MODIFIED Requirements

### Requirement: Rollout generation supports vLLM backends (colocate default, server optional)
When rollout-matching training is enabled (`custom.trainer_variant: stage2_rollout_aligned`), the system SHALL support generating rollouts using a vLLM backend, while keeping teacher-forced forward/backprop on the normal training model.

Rollout-matching settings are a first-class top-level namespace:
- Rollout-matching settings MUST be authored under top-level `rollout_matching.*`.
- Legacy placement under `custom.extra.rollout_matching.*` is unsupported and MUST fail fast with actionable guidance to migrate to `rollout_matching.*`.

Backend selection MUST be YAML-driven under `rollout_matching`:
- `rollout_backend` MUST accept `"vllm"` or `"hf"`.
- `rollout_backend` MUST default to `"vllm"`.
- `eval_rollout_backend` MUST accept `null`, `"hf"`, or `"vllm"` (case-insensitive). Missing MUST be treated as `null`.
  - When `null`/missing, evaluation rollouts MUST inherit `rollout_backend`.
  - When set to `"hf"` or `"vllm"`, evaluation rollouts MUST use `eval_rollout_backend` and MUST NOT affect training-time rollouts.

Length-coherence guardrails (fail-fast):
- If the effective rollout backend is `vllm` (training or eval), the system MUST enforce:
  - `rollout_matching.max_new_tokens < rollout_matching.vllm.max_model_len` (to avoid truncation/overflow), and
  - `rollout_matching.vllm.max_model_len >= global_max_length` (to avoid silent truncation drift between training and rollouts).

When `rollout_backend: "vllm"`:
- The trainer MUST configure vLLM from `rollout_matching.vllm` (mapping).
- The vLLM integration MUST support two modes under `rollout_matching.vllm.mode`:
  - `colocate` (default)
  - `server` (optional)
- If `rollout_matching.vllm.mode` is unset, it MUST default to `colocate` to preserve current behavior.

Common vLLM contract (both modes):
- The vLLM backend MUST return:
  - `response_token_ids` (assistant token ids, stop-trimmed),
  - `prompt_token_ids` (prompt token ids used by vLLM),
  so stage_2 can enforce strict prompt-prefix token-id alignment.
- Invalid vLLM configuration MUST fail fast with actionable guidance.

Colocate mode requirements (`rollout_matching.vllm.mode: colocate`):
- The trainer MUST use a colocated vLLM engine for rollout generation.
  - Implementation detail: this MAY be in-process or an internal colocated worker, but it MUST consume VRAM on the same GPU(s) as training.
- Default vLLM settings MUST be conservative to preserve training headroom on 4-GPU runs:
  - `gpu_memory_utilization: 0.45`
  - `tensor_parallel_size: 4`

Server mode requirements (`rollout_matching.vllm.mode: server`):
- The trainer MUST connect to an external vLLM rollout server (pre-launched) instead of instantiating a local vLLM engine.
- The trainer MUST support in-memory weight synchronization to the server (no disk checkpoint reload) so that rollouts can be generated with the latest policy parameters.
- Server mode MUST support multi-process learners (i.e., `torch.distributed` initialized with `world_size >= 1`).
- Under multi-process learners (`world_size > 1`), the trainer MUST synchronize weights in a DDP-safe way:
    - rank0-only communicator init + weight push,
    - strict ordering: barrier -> rank0 sync -> barrier,
    - all ranks MUST take the same control-flow (including early-return decisions) to avoid deadlocks.
  - `rollout_matching.vllm.sync.mode` MUST resolve to `full` (only `full` is supported in this stack).
  - If these requirements cannot be satisfied (e.g., communicator init fails, misconfigured sync mode), the trainer MUST fail fast with actionable guidance.

Server connectivity MUST be YAML-driven under `rollout_matching.vllm.server` (mapping).

Server connectivity config MUST be expressed as an explicit `servers` list:
- `servers` MUST be a non-empty list of mappings.
- Each entry MUST contain:
  - `base_url` (string)
  - `group_port` (int)

Legacy paired-list server form (`rollout_matching.vllm.server.base_url` + `rollout_matching.vllm.server.group_port`) is unsupported and MUST fail fast with actionable guidance to migrate to `rollout_matching.vllm.server.servers[]` with exactly one entry.

Common server fields:
- `timeout_s` MUST accept a float or int and MUST default to `240.0`.
  - This timeout applies to initial server reachability checks (e.g., polling `/health/`) and communicator init.
- `infer_timeout_s` MUST accept `null` or a float or int and MUST default to `null` (no timeout).
  - When set to a positive number, it MUST be used as the HTTP timeout for `/infer/` requests.
  - When `null` or <= 0, the trainer MUST NOT enforce an HTTP timeout for `/infer/` requests.

Base URL semantics (normative):
- `base_url` MUST use scheme `http` or `https`.
- `base_url` MUST include an explicit port (e.g., `http://127.0.0.1:8000`).
- `base_url` host MUST NOT be `0.0.0.0`.
- `base_url` MUST NOT include a path component (it MUST be `http(s)://<host>:<port>` with an optional trailing `/`).
- Each `base_url` MUST be a URL prefix such that the following endpoints are reachable:
  - `${base_url}/health/`
  - `${base_url}/infer/`
  - `${base_url}/get_world_size/`
  - `${base_url}/init_communicator/`

Single-server constraint (this stack):
- `rollout_matching.vllm.server.servers` MUST contain exactly 1 entry.
- Configs that provide multiple server entries MUST fail fast with actionable guidance that multi-server is unsupported in this stack.
- This limitation is about “multiple base_url endpoints”, not about GPUs:
  - multi-GPU rollouts remain supported via vLLM parallelism (e.g., `tensor_parallel_size` and derived data parallelism) within the single server.

Weight sync mode selection MUST be YAML-driven under `rollout_matching.vllm.sync`:
- `sync.mode` MUST accept only `full` (case-insensitive).
  - If unset, `sync.mode` MUST default to `full`.
- `sync.fallback_to_full` MUST accept a boolean and MAY be present for backwards-compatible config parsing, but it has no effect (only full sync is supported in this stack).

Weight sync behavior (normative):
- `sync.mode: full` MUST sync full merged weights (GRPO-style) into vLLM.
- Adapter-only sync (vLLM LoRA upload / `add_lora`) is unsupported in this stack. Any attempt to request adapter-only behavior (e.g., `sync.mode: adapter|auto` or `rollout_matching.vllm.enable_lora: true`) MUST fail fast with actionable guidance.

Determinism (server mode):
- For server mode, the trainer MUST set a deterministic `RequestConfig.seed` for each server `/infer/` call.

#### Scenario: Default behavior preserved (colocate)
- **GIVEN** rollout-matching training is enabled
- **AND** `rollout_matching.rollout_backend: vllm`
- **AND** `rollout_matching.vllm.mode` is unset
- **WHEN** one training step executes
- **THEN** the trainer uses vLLM in colocate mode
- **AND** the teacher-forced forward/backprop uses the training model
- **AND** stage_2 prompt-prefix token-id alignment checks apply unchanged.

#### Scenario: Legacy paired-list server form fails fast
- **GIVEN** `rollout_matching.vllm.mode: server`
- **AND** server connectivity is configured using the legacy paired-list form (`rollout_matching.vllm.server.base_url` / `rollout_matching.vllm.server.group_port`)
- **WHEN** configuration is parsed/materialized
- **THEN** it fails fast with guidance to migrate to `rollout_matching.vllm.server.servers[]` with exactly one entry.

#### Scenario: Multi-server config fails fast
- **GIVEN** `rollout_matching.vllm.mode: server`
- **AND** `rollout_matching.vllm.server.servers` contains 2 entries
- **WHEN** configuration is parsed/materialized
- **THEN** configuration validation fails fast with actionable guidance that only single-server is supported.

#### Scenario: Adapter-only sync is rejected (full-sync-only vLLM)
- **GIVEN** rollout-matching training is enabled
- **AND** the effective rollout backend resolves to vLLM (training or eval)
- **AND** config requests adapter-only sync (e.g., `rollout_matching.vllm.enable_lora: true` or `rollout_matching.vllm.sync.mode: adapter`)
- **WHEN** training starts (before the first vLLM rollout)
- **THEN** configuration validation fails fast with actionable guidance to use full merged-weight sync (`enable_lora: false`, `sync.mode: full`).

#### Scenario: Server mode produces token ids suitable for strict alignment
- **GIVEN** rollout-matching training is enabled
- **AND** `rollout_matching.rollout_backend: vllm`
- **AND** `rollout_matching.vllm.mode: server`
- **WHEN** one fresh-rollout step (E-step) executes
- **THEN** the trainer obtains per-sample `response_token_ids` and `prompt_token_ids` from the server
- **AND** the existing prompt-prefix sanity check is applied using those token ids
- **AND** the rest of parsing/matching/target construction proceeds unchanged.

#### Scenario: Invalid server configuration fails fast
- **GIVEN** rollout-matching training is enabled
- **AND** `rollout_matching.vllm.mode: server`
- **WHEN** the server base URL or communicator port is missing/unreachable
- **THEN** the trainer fails fast with an actionable error message
- **AND** the user can explicitly switch back to `rollout_matching.vllm.mode: colocate` or `rollout_matching.rollout_backend: hf`.

#### Scenario: vLLM backend produces token ids suitable for strict alignment
- **GIVEN** rollout-matching training is enabled
- **AND** the rollout backend is set to vLLM colocate
- **WHEN** one training step executes
- **THEN** the trainer obtains per-sample `response_token_ids` and `prompt_token_ids` from vLLM
- **AND** the existing prompt-prefix sanity check is applied using those token ids
- **AND** the rest of parsing/matching/loss computation proceeds unchanged.

#### Scenario: Invalid vLLM configuration fails fast
- **GIVEN** rollout-matching training is enabled
- **AND** the rollout backend is set to vLLM
- **WHEN** vLLM is unavailable, tensor-parallel settings are incompatible, or LoRA sync is not possible
- **THEN** the trainer fails fast with an actionable error message
- **AND** the user can explicitly switch back to HF rollout via `rollout_backend: "hf"`.

#### Scenario: Multi-process learner does not deadlock in server mode
- **GIVEN** rollout-matching training is enabled
- **AND** `rollout_matching.rollout_backend: vllm`
- **AND** `rollout_matching.vllm.mode: server`
- **AND** training is launched under `torchrun` with `world_size=2`
- **WHEN** the trainer performs a fresh-rollout step that requires a server sync
- **THEN** rank0 performs the server weight sync and other ranks do not
- **AND** all ranks proceed to issue rollout `/infer/` requests without deadlock.

#### Scenario: Non-full sync mode fails fast (all learners)
- **GIVEN** rollout-matching training is enabled
- **AND** the effective rollout backend resolves to vLLM (training or eval)
- **AND** `rollout_matching.vllm.sync.mode` is set to a non-`full` value (e.g., `adapter` or `auto`)
- **WHEN** training starts (before the first vLLM rollout)
- **THEN** configuration validation fails fast with actionable guidance to use `sync.mode: full`.

### Requirement: vLLM server mode derives per-rank request chunking from server world size
When rollout-matching training uses vLLM server mode (`rollout_matching.rollout_backend=vllm` and `rollout_matching.vllm.mode=server`), the trainer MUST derive per-rank request chunk sizing from the rollout server world size and learner DDP world size to preserve the per-rollout-GPU cap defined by `decode_batch_size`.

Semantics (normative):
- `rollout_matching.vllm.server.servers` MUST contain exactly 1 entry.
- The trainer MUST query only that server’s `${base_url}/get_world_size/` endpoint and cache one `world_size` value `S`.
- Let `W = learner_world_size` (training DDP world size).
- **Feasibility**: If `decode_batch_size * S < W`, the trainer MUST fail fast with actionable guidance (the cap cannot be preserved if every learner rank must issue at least one request concurrently).
- Otherwise, the trainer MUST derive a per-learner-rank chunk size:
  - `chunk = floor(decode_batch_size * S / W)`
- The trainer MUST NOT implement cross-server request distribution in this stack.

#### Scenario: vLLM server mode fails fast when cap is infeasible
- **GIVEN** rollout-matching training is enabled
- **AND** `rollout_backend=vllm` and `vllm.mode=server`
- **AND** the configured server reports `S=2` via `/get_world_size/`
- **AND** `decode_batch_size=1` and `W=4`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to increase rollout server world size, reduce learner world size, or increase `decode_batch_size`.

### Requirement: Server/HF rollout sampling is configured under decoding.*
Rollout decoding knobs MUST be expressed under:
- `rollout_matching.decoding` (mapping)

Supported decoding keys (v1):
- `rollout_matching.decoding.temperature` (float, `>= 0`; greedy if `== 0`)
- `rollout_matching.decoding.top_p` (float, `(0, 1]`, default `1.0`)
- `rollout_matching.decoding.top_k` (int, default `-1`)

Legacy decoding keys are removed (breaking):
- If a config provides any of:
  - `rollout_matching.temperature`
  - `rollout_matching.top_p`
  - `rollout_matching.top_k`
  the system MUST fail fast with guidance to migrate to `rollout_matching.decoding.*`.

Robustness-first sampling note:
- When sampling is enabled (e.g., `temperature > 0`) and the server backend retries/splits requests for robustness,
  strict bitwise determinism is not required; the trainer MUST log sufficient metadata to audit effective decoding behavior.

#### Scenario: Legacy decoding keys fail fast
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `rollout_matching.temperature`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to use `rollout_matching.decoding.temperature` instead.

### Requirement: vLLM rollout backend supports repeat-aware per-sequence early termination
When rollout generation uses vLLM server backend (`rollout_matching.rollout_backend: vllm` in rollout-server mode), the system MUST support repeat-aware termination semantics equivalent to the existing HF repeat guard.

Normative behavior:
- This requirement scope is vLLM rollout server mode used by Stage-2 AB; colocate/non-server vLLM paths are out of scope for this change and remain unchanged.
- Repeat-aware termination MUST be controlled by `rollout_matching.repeat_terminate`.
- On the current vLLM V1-default stack, when `rollout_matching.repeat_terminate.enabled: true`, vLLM rollout serving MUST activate repeat-aware processing in server mode via startup-time plugin injection (e.g., launching `swift rollout` with `--external_plugins <repo-owned-plugin>`).
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
- **WHEN** `rollout_matching.repeat_terminate.enabled` is true and vLLM rollout server cannot load repeat-aware processing
- **THEN** rollout startup fails before training proceeds
- **AND** the error reports the missing processor activation path.

#### Scenario: vLLM V1 rollout does not rely on request-time logits processors
- **GIVEN** vLLM V1-default rollout serving
- **AND** `rollout_matching.repeat_terminate.enabled: true`
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
- The same YAML subtree (`rollout_matching.repeat_terminate`) MUST drive both HF and vLLM guard behavior.
- vLLM mode MUST NOT require new standalone CLI flags for repeat-aware behavior.
- Existing configs that set `rollout_matching.repeat_terminate.enabled: true` MUST activate repeat-aware behavior in vLLM mode (i.e., vLLM MUST honor YAML when enabled; no extra knobs are required).
- Legacy “repeat_terminate is HF-only / ignored by vLLM” config or docs statements MUST be removed or updated as part of migration.

#### Scenario: Existing YAML enables repeat-aware behavior in vLLM mode
- **GIVEN** a rollout-matching config with `rollout_matching.repeat_terminate.enabled: true`
- **WHEN** rollout backend is switched from HF to vLLM
- **THEN** repeat-aware termination remains enabled without adding new CLI parameters.
