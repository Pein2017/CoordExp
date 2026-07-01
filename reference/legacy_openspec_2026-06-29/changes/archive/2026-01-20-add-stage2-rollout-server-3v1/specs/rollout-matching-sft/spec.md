## RENAMED Requirements
- FROM: `### Requirement: Rollout generation supports a vLLM colocate backend (default)`
- TO: `### Requirement: Rollout generation supports vLLM backends (colocate default, server optional)`

## MODIFIED Requirements

### Requirement: Rollout generation supports vLLM backends (colocate default, server optional)
When rollout-matching training is enabled (`custom.trainer_variant: rollout_matching_sft`), the system SHALL support generating rollouts using a vLLM backend, while keeping teacher-forced forward/backprop on the normal training model.

Backend selection MUST be YAML-driven under `custom.extra.rollout_matching`:
- `rollout_backend` MUST accept `"vllm"` or `"hf"`.
- `rollout_backend` MUST default to `"vllm"`.

When `rollout_backend: "vllm"`:
- The trainer MUST configure vLLM from `custom.extra.rollout_matching.vllm` (mapping).
- The vLLM integration MUST support two modes under `custom.extra.rollout_matching.vllm.mode`:
  - `colocate` (default)
  - `server` (optional)
- If `custom.extra.rollout_matching.vllm.mode` is unset, it MUST default to `colocate` to preserve current behavior.

Common vLLM contract (both modes):
- The vLLM backend MUST return:
  - `response_token_ids` (assistant token ids, stop-trimmed),
  - `prompt_token_ids` (prompt token ids used by vLLM),
  so stage_2 can enforce strict prompt-prefix token-id alignment.
- Invalid vLLM configuration MUST fail fast with actionable guidance.

Colocate mode requirements (`custom.extra.rollout_matching.vllm.mode: colocate`):
- The trainer MUST use a colocated vLLM engine for rollout generation.
  - Implementation detail: this MAY be in-process or an internal colocated worker, but it MUST consume VRAM on the same GPU(s) as training.
- Default vLLM settings MUST be conservative to preserve training headroom on 4-GPU runs:
  - `gpu_memory_utilization: 0.45`
  - `tensor_parallel_size: 4`

Server mode requirements (`custom.extra.rollout_matching.vllm.mode: server`):
- The trainer MUST connect to an external vLLM rollout server (pre-launched) instead of instantiating a local vLLM engine.
- The trainer MUST support in-memory weight synchronization to the server (no disk checkpoint reload) so that rollouts can be generated with the latest policy parameters.
- For this v1 workflow, server mode MUST only be supported when the learner runs as a single training process (`world_size == 1`).
  - If `torch.distributed` is initialized with `world_size > 1`, the trainer MUST fail fast with actionable guidance (e.g., run the learner on a single GPU, or use `custom.extra.rollout_matching.vllm.mode: colocate`).

Server connectivity MUST be YAML-driven under `custom.extra.rollout_matching.vllm.server` (mapping).
The config MUST be expressed in exactly one of these forms:

1) Preferred explicit list form:
- `servers` MUST be a non-empty list of mappings.
- Each entry MUST contain:
  - `base_url` (string)
  - `group_port` (int)

2) Legacy paired-list form:
- `base_url` MUST be a string or list of strings.
- `group_port` MUST be an int or list of ints.
- If `base_url` is a list:
  - when `group_port` is a list, it MUST be a list of the same length and pairing MUST be by index
  - when `group_port` is an int, per-server group ports MUST be derived deterministically as `group_port + i` for server index `i`
- If `base_url` is a string, `group_port` MUST be an int.

Common server fields:
- `timeout_s` MUST accept a float or int and MUST default to `240.0`.
  - This timeout applies to initial server reachability checks (e.g., polling `/health/`) and communicator init.
- `infer_timeout_s` MUST accept `null` or a float or int and MUST default to `null` (no timeout).
  - When set to a positive number, it MUST be used as the HTTP timeout for `/infer/` requests.
  - When `null` or <= 0, the trainer MUST NOT enforce an HTTP timeout for `/infer/` requests.

Base URL semantics (normative):
- Each `base_url` MUST be a URL prefix such that the following endpoints are reachable:
  - `${base_url}/health/`
  - `${base_url}/infer/`
  - `${base_url}/get_world_size/`
  - `${base_url}/init_communicator/`

Multi-server request distribution (normative):
- When multiple servers are configured, the trainer MUST distribute rollout requests deterministically.
- If `N == 0` (no rollout requests), the trainer MUST return an empty output list without issuing any server requests.
- For `N > 0` requests and `S` servers, the trainer MUST use stable contiguous chunking:
  - `chunk_size = ceil(N / S)`
  - server `i` receives requests `[i*chunk_size : (i+1)*chunk_size]` in the original request order
  - servers with empty chunks MUST receive zero requests
- The trainer MUST reassemble outputs in a way that preserves the original request order.

Weight sync mode selection MUST be YAML-driven under `custom.extra.rollout_matching.vllm.sync`:
- `sync.mode` MUST accept `full`, `adapter`, or `auto`.
- If unset, `sync.mode` MUST default to `full` (robust for multimodal + DoRA).
- `sync.fallback_to_full` MUST accept a boolean and MUST default to `true`.

Weight sync behavior (normative):
- `sync.mode: full` MUST sync full merged weights (GRPO-style) into vLLM.
- `sync.mode: adapter` MUST sync adapters only, and MUST require `custom.extra.rollout_matching.vllm.enable_lora: true`.
- `sync.mode: auto` MUST behave as:
  - `adapter` when `enable_lora: true`
  - otherwise `full`
- If adapter-only sync fails at runtime:
  - when `sync.fallback_to_full: true`, the trainer MUST emit a warning and permanently fall back to `full` sync for the remainder of the run
  - when `sync.fallback_to_full: false`, the trainer MUST fail fast with actionable guidance

Determinism (server mode):
- For server mode, the trainer MUST set a deterministic `RequestConfig.seed` for each server `/infer/` call.

Definitions (normative):
- `gas` is `gradient_accumulation_steps`.
- `micro_step_idx` is the index of the current micro-step within the gradient-accumulation window for the current optimizer step.
  - It MUST be in `[0, gas-1]`.
  - It MUST follow the dataloader order for that optimizer step.
- `request_list` is the pre-chunked list of rollout requests for the current micro-step, in dataloader order.
- `req_idx` is a request's index in `request_list` (`0..N-1`).

Seed derivation requirements:
- The seed derivation MUST be a deterministic function of at least:
  - the run seed (`training.seed`)
  - the optimizer step index (`TrainerState.global_step`)
  - `micro_step_idx`
  - `req_idx`
- For multi-server distribution, the trainer MUST set each server call's `RequestConfig.seed` using the `req_idx` of the first request in that server's chunk.

#### Scenario: Default behavior preserved (colocate)
- **GIVEN** rollout-matching training is enabled
- **AND** `custom.extra.rollout_matching.rollout_backend: vllm`
- **AND** `custom.extra.rollout_matching.vllm.mode` is unset
- **WHEN** one training step executes
- **THEN** the trainer uses vLLM in colocate mode
- **AND** the teacher-forced forward/backprop uses the training model
- **AND** stage_2 prompt-prefix token-id alignment checks apply unchanged.

#### Scenario: Multi-server pairing is strict and deterministic
- **GIVEN** `custom.extra.rollout_matching.vllm.mode: server`
- **AND** server connectivity is configured using the legacy paired-list form
- **WHEN** `base_url` is a list and `group_port` is a list with a different length
- **THEN** the trainer MUST fail fast with actionable guidance.

#### Scenario: Multi-server request distribution is deterministic
- **GIVEN** `custom.extra.rollout_matching.vllm.mode: server`
- **AND** multiple servers are configured in a fixed order
- **AND** a fixed list of rollout requests in a fixed order
- **WHEN** the trainer distributes requests to servers twice
- **THEN** it assigns the same contiguous chunks to the same server indices both times
- **AND** the reassembled outputs preserve the original request order.

#### Scenario: Adapter sync can fall back to full sync
- **GIVEN** `custom.extra.rollout_matching.vllm.sync.mode: adapter`
- **AND** `custom.extra.rollout_matching.vllm.sync.fallback_to_full: true`
- **WHEN** adapter sync fails due to a runtime incompatibility
- **THEN** the trainer logs a warning
- **AND** switches to full merged-weight sync for subsequent E-steps.

#### Scenario: Server mode produces token ids suitable for strict alignment
- **GIVEN** rollout-matching training is enabled
- **AND** `custom.extra.rollout_matching.rollout_backend: vllm`
- **AND** `custom.extra.rollout_matching.vllm.mode: server`
- **WHEN** one fresh-rollout step (E-step) executes
- **THEN** the trainer obtains per-sample `response_token_ids` and `prompt_token_ids` from the server
- **AND** the existing prompt-prefix sanity check is applied using those token ids
- **AND** the rest of parsing/matching/target construction proceeds unchanged.

#### Scenario: Invalid server configuration fails fast
- **GIVEN** rollout-matching training is enabled
- **AND** `custom.extra.rollout_matching.vllm.mode: server`
- **WHEN** the server base URL or communicator port is missing/unreachable
- **THEN** the trainer fails fast with an actionable error message
- **AND** the user can explicitly switch back to `custom.extra.rollout_matching.vllm.mode: colocate` or `custom.extra.rollout_matching.rollout_backend: hf`.

## ADDED Requirements

### Requirement: Stage_2 supports a 3v1 rollout-server + learner workflow (actors vs learner)
The stage_2 rollout-matching trainer (`custom.trainer_variant: rollout_matching_sft`) MUST support a practical single-node workflow where rollout generation runs on dedicated GPUs via a vLLM server, and teacher-forced SFT training runs on a separate GPU.

Normative workflow (one node):
- A vLLM rollout server runs on a dedicated GPU subset (e.g., 3 GPUs) and is responsible only for inference.
- A stage_2 learner runs on a separate GPU subset (e.g., 1 GPU) and is responsible for:
  - dataloader iteration
  - strict parse/match
  - `Y_train` construction (prefix + mandatory FN append)
  - post-rollout packing to `global_max_length`
  - teacher-forced forward/backward and optimizer updates
- After learner updates parameters, the learner MUST synchronize weights to the rollout server before requesting rollouts that claim to be generated by the latest policy.

Dataloader iteration requirements:
- The learner MUST be the owner of the dataloader iterator.
- The rollout server MUST NOT own or advance the dataset iterator.
- With `rollout_buffer.m_steps > 1`, the learner MUST preserve the existing accumulation-window repeater semantics (no silent skipping of dataset samples).

Multimodal request payload requirements (server mode):
- Server-mode rollout requests MUST be JSON-serializable and MUST be compatible with ms-swift's `RolloutInferRequest` contract.
- Images MUST be passed as strings (e.g., URL, base64 string, or local filesystem path).
- For this single-node workflow, local filesystem paths MUST be supported, and the learner and server MUST have access to the same paths (shared filesystem assumption).

Packed target delivery requirements:
- The system MUST NOT require transferring packed `Y_train` tensors over the network/IPC boundary in order to function.
- The only required cross-process payloads are:
  - rollout requests (messages + multimodal payload)
  - rollout outputs (token ids + decoded text for logging)
  - weight sync payloads

#### Scenario: 3 rollout GPUs + 1 learner GPU is operational
- **GIVEN** a vLLM rollout server is launched on GPUs 0-2
- **AND** the stage_2 learner is launched on GPU 3 with `custom.extra.rollout_matching.vllm.mode: server`
- **WHEN** training runs for one optimizer step
- **THEN** rollouts are generated on the rollout GPUs
- **AND** teacher-forced forward/backward runs only on the learner GPU
- **AND** weights are synchronized in-memory without requiring checkpoint reload from disk.

### Requirement: Server-mode rollouts are paper-reproducible via logged metadata
When `custom.extra.rollout_matching.vllm.mode: server` is enabled, the trainer MUST log sufficient metadata to reproduce and debug the run.

At minimum, the trainer MUST log:
- the effective server list (base URLs and group ports)
- the effective weight sync mode (`full` vs `adapter`) including any fallback events
- the per-batch rollout seed used for `RequestConfig.seed`

#### Scenario: Server mode logs reproducibility metadata
- **GIVEN** server mode is enabled
- **WHEN** the trainer performs a fresh-rollout step (E-step)
- **THEN** it logs the server endpoints, sync mode, and rollout seed for that step.
