# rollout-matching-sft Specification

## Purpose
TBD - created by archiving change 2026-01-19-update-stage2-post-rollout-packing-binpacking. Update Purpose after archive.
## Requirements
### Requirement: Stage-2 post-rollout packing selection uses deterministic ms-swift-like binpacking
When rollout-matching training is enabled (`custom.trainer_variant: rollout_matching_sft`) and post-rollout packing is enabled (`training.packing: true`), the trainer SHALL select which buffered segments are included in the packed teacher-forced forward pass using a deterministic, ms-swift-like constant-volume binpacking heuristic.

Definitions:
- A “segment” is one sample’s teacher-forced encoding of `Y_train` (rollout prefix + mandatory FN append), and is treated as an atomic unit.
- `packing_length` is the maximum packed length derived from `global_max_length` / `template.max_length` and is a hard cap per packed forward.

Selection requirements:
- The trainer SHALL maintain a rank-local buffer of pending segments.
- For each packed forward, the trainer SHALL select a subset of buffered segments whose total `encoded_len` is `<= packing_length`.
- The trainer MUST NOT split a single segment across multiple packed forwards.
- To avoid starvation, the trainer SHALL always include the oldest buffered segment in the selected subset.
- The trainer SHOULD attempt to improve fill ratio (larger total length) beyond the FIFO-greedy baseline, consistent with ms-swift’s constant-volume binpacking intent (e.g., via `binpacking.to_constant_volume`).
- The trainer MUST NOT produce a selection with a lower total selected length than the FIFO-greedy baseline for the same buffer state.
  - A compliant approach is: compute the FIFO-greedy baseline; compute a binpacking candidate constrained to include the oldest; pick whichever has higher total length; use stable tie-breaking.
- The selection MUST be deterministic: with identical buffered segments in identical insertion order and identical lengths, the selected subset and its order SHALL be identical across runs.
  - Tie-breaking MUST be stable.
  - If the selection logic encounters ties in its own scoring (e.g., equal total length), it MUST break ties deterministically, for example:
    - prefer fewer selected segments, then
    - prefer the lexicographically-smallest index set (in insertion-order indices).
- The selected subset SHOULD be ordered by insertion order (oldest-first) to minimize behavior change and keep packing deterministic.

Safety requirements:
- If any single buffered segment has `encoded_len > packing_length`, the trainer MUST fail fast at segment creation / buffer insertion time (not only when it becomes the oldest) with actionable guidance (e.g., increase `global_max_length`, reduce `max_new_tokens`, or disable packing).
- If post-rollout packing selection requires `binpacking` and the `binpacking` module is not available at runtime, the trainer MUST fail fast with actionable guidance (e.g., install `binpacking` or disable `training.packing`) rather than silently falling back to another heuristic.
- The trainer MUST preserve per-token supervision semantics under packing by maintaining correct offsets for all supervision masks/indices after packing.

The selection algorithm MUST reuse existing YAML knobs and MUST NOT require new CLI flags:
- `training.packing_buffer`
- `training.packing_min_fill_ratio` (telemetry/warn threshold)
- `training.packing_drop_last` (carry-only mode requirement remains unchanged)

#### Scenario: Multiple short segments pack efficiently under the same cap
- **GIVEN** rollout-matching training is enabled with post-rollout packing
- **AND** the buffer contains multiple segments whose individual `encoded_len` are all `< packing_length`
- **WHEN** the trainer selects segments for the next packed forward
- **THEN** it selects a subset whose total length is `<= packing_length`
- **AND** the selection includes the oldest segment
- **AND** the resulting fill ratio is at least as high as the FIFO-greedy baseline for the same buffer state.

#### Scenario: Deterministic selection
- **GIVEN** identical buffered segments in identical insertion order with identical `encoded_len`
- **WHEN** selection runs twice
- **THEN** it returns the same selected subset in the same order both times.

#### Scenario: Oversized segment fails fast
- **GIVEN** a segment with `encoded_len > packing_length`
- **WHEN** the trainer prepares the segment for post-rollout buffering / insertion
- **THEN** it raises an error that includes at least one mitigation suggestion.

#### Scenario: Missing binpacking dependency fails fast
- **GIVEN** rollout-matching training is enabled with post-rollout packing
- **AND** the runtime environment does not provide the `binpacking` module
- **WHEN** the trainer attempts to select segments for post-rollout packing
- **THEN** it raises an error that includes at least one mitigation suggestion (e.g., install `binpacking` or disable `training.packing`).

### Requirement: Post-rollout packing is supported under rollout-matching training
When rollout-matching training is enabled (`custom.trainer_variant: rollout_matching_sft`), the system SHALL support packing for the *post-rollout teacher-forced forward pass* to improve training efficiency.

Packing MUST remain YAML-driven and MUST NOT require new CLI hyperparameter flags.

When packing is enabled for rollout-matching training:
- Rollout generation MUST remain un-packed (no sequence packing during autoregressive generation).
- The trainer MUST generate rollouts first, then build each sample's teacher-forced `Y_train` sequence (rollout prefix + mandatory FN append).
- The trainer MUST perform **dynamic packing** based on the actual constructed `Y_train` lengths (after rollout + FN append), not on pre-rollout dataset length estimates.
- The trainer MUST treat each sample as an atomic segment and MUST NOT split a single sample across multiple packed forwards.
- The trainer MUST preserve per-sample loss semantics by maintaining correct per-token supervision masks after packing (offset-correct in the packed row).
- The trainer MUST maintain a rank-local carry buffer for leftover segments and reuse the existing YAML packing knobs:
  - `training.packing_buffer`, `training.packing_min_fill_ratio`, `training.packing_drop_last`
- For carry-only mode, `training.packing_drop_last` MUST be `true` (the trainer does not run "extra flush steps" after max_steps/epoch end).
- If the carry buffer would exceed the configured `packing_buffer`, the trainer MUST fail fast with actionable error text.
- The trainer MUST keep the existing sanity checks:
  - prompt prefix tokenization matches generation,
  - supervised coord indices fall within the assistant span.

#### Scenario: Packing enabled does not break rollout-matching training
- **GIVEN** a YAML config sets `custom.trainer_variant: rollout_matching_sft`
- **AND** packing is enabled for training
- **WHEN** one training step executes
- **THEN** rollouts are generated without sequence packing
- **AND** the teacher-forced forward pass uses packed sequences
- **AND** loss masking logic remains correct and validated by sanity checks.

#### Scenario: Packing is never applied during rollout generation
- **GIVEN** rollout-matching training is enabled with packing
- **WHEN** the trainer performs autoregressive generation for rollouts
- **THEN** each rollout is generated using a standard padded batch encoding (un-packed)
- **AND** the packing mechanism is applied only after the rollout is complete.

#### Scenario: Carry buffer preserves segments without splitting
- **GIVEN** rollout-matching training is enabled with packing
- **AND** the rank-local carry buffer contains leftover segments from a previous step
- **WHEN** the next training step executes
- **THEN** the trainer packs forward using a subset of segments
- **AND** any leftover segments remain buffered for future steps
- **AND** no segment is split across packed forwards.

#### Scenario: Carry buffer overflow fails fast with actionable guidance
- **GIVEN** rollout-matching training is enabled with packing
- **AND** `training.packing_buffer` is too small for the configured raw batch size / sequence lengths
- **WHEN** the trainer would exceed the configured buffer cap
- **THEN** the trainer fails fast with an error message that suggests at least one mitigation (e.g. reduce raw batch size, increase `packing_buffer`, or enable multi-pack-per-step in a future change).

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
- Server mode MUST support multi-process learners (i.e., `torch.distributed` initialized with `world_size >= 1`).
  - Under multi-process learners (`world_size > 1`), the trainer MUST synchronize weights in a DDP-safe way:
    - rank0-only communicator init + weight push,
    - strict ordering: barrier -> rank0 sync -> barrier,
    - all ranks MUST take the same control-flow (including early-return decisions) to avoid deadlocks.
  - Under multi-process learners (`world_size > 1`), `custom.extra.rollout_matching.vllm.sync.mode` MUST resolve to `full`.
  - If these requirements cannot be satisfied (e.g., communicator init fails, misconfigured sync mode), the trainer MUST fail fast with actionable guidance.

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

### Requirement: Rollout generation supports microbatched decoding within each rank
When rollout-matching training is enabled, the trainer SHALL support generating rollouts for multiple samples in a single backend decode call (HF `generate()` or vLLM `/infer/`), controlled by a YAML knob:

- `custom.extra.rollout_matching.decode_batch_size` (int)

Semantics (normative):
- `decode_batch_size` denotes the maximum number of sequences decoded per rollout GPU in one generation call.
- The trainer MUST enforce this bound for both HF and vLLM backends.

Defaulting (normative):
- If `custom.extra.rollout_matching.decode_batch_size` is unset, the implementation MUST default it to `1` (conservative).
- Higher-level experiment templates MAY set a larger default explicitly (e.g., Stage2-AB YAML under `configs/stage2_ab/**` uses `4`).

#### Scenario: Microbatching increases decode parallelism without changing outputs format
- **GIVEN** rollout-matching training is enabled
- **AND** `custom.extra.rollout_matching.decode_batch_size: M` where `M > 1`
- **WHEN** the trainer generates rollouts for a batch of `M` samples on one rank
- **THEN** the trainer performs one batched generate call for those `M` samples
- **AND** it returns per-sample `response_token_ids` suitable for strict token-aligned parsing.

### Requirement: Legacy rollout batch-size knobs fail fast
The system MUST fail fast if a config provides any legacy rollout batch-size knob under `custom.extra.rollout_matching`, with actionable guidance to migrate to the unified decode batching knob `custom.extra.rollout_matching.decode_batch_size`.

Legacy knobs (normative):
- `custom.extra.rollout_matching.rollout_generate_batch_size`
- `custom.extra.rollout_matching.rollout_infer_batch_size`

#### Scenario: rollout_generate_batch_size fails fast when present
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `custom.extra.rollout_matching.rollout_generate_batch_size`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to remove it and use `custom.extra.rollout_matching.decode_batch_size`.

#### Scenario: rollout_infer_batch_size fails fast when present
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `custom.extra.rollout_matching.rollout_infer_batch_size`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to remove it and use `custom.extra.rollout_matching.decode_batch_size`.

### Requirement: Window-aware post-rollout packing scope knob is removed
The system MUST fail fast if a config provides `custom.extra.rollout_matching.post_rollout_pack_scope` (any value), with actionable guidance to remove it.

Normative behavior:
- Post-rollout packing behavior MUST be the standardized micro-scope dynamic packing semantics.
- Window-scoped packing is not supported.

#### Scenario: post_rollout_pack_scope fails fast when present
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `custom.extra.rollout_matching.post_rollout_pack_scope`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to remove `post_rollout_pack_scope`.

### Requirement: vLLM server mode derives per-rank request chunking from server world size
When rollout-matching training uses vLLM server mode (`custom.extra.rollout_matching.rollout_backend=vllm` and `custom.extra.rollout_matching.vllm.mode=server`), the trainer MUST derive per-rank request chunk sizing from the rollout server world size and learner DDP world size to preserve the per-rollout-GPU cap defined by `decode_batch_size`.

Semantics (normative):
- The trainer MUST query each configured server’s `${base_url}/get_world_size/` endpoint and cache one `world_size` per server entry.
- Let `server_world_sizes = [s_0, s_1, ...]` be those cached values, and define:
  - `S = sum(server_world_sizes)` (total rollout inference device count across servers; DP replicas)
  - `W = learner_world_size` (training DDP world size)
- **Feasibility**: If `decode_batch_size * S < W`, the trainer MUST fail fast with actionable guidance (the cap cannot be preserved if every learner rank must issue at least one request concurrently).
- Otherwise, the trainer MUST derive a per-learner-rank chunk size:
  - `chunk = floor(decode_batch_size * S / W)`
- The trainer MUST distribute requests across servers in a capacity-aware deterministic way (proportional to `server_world_sizes`) so that rollout GPUs are not overloaded when servers are heterogeneous.

#### Scenario: vLLM server mode fails fast when cap is infeasible
- **GIVEN** rollout-matching training is enabled
- **AND** `rollout_backend=vllm` and `vllm.mode=server`
- **AND** `decode_batch_size=1`, `S=2`, and `W=4`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to increase rollout server world size, reduce learner world size, or increase `decode_batch_size`.

### Requirement: Legacy rollout_buffer configs fail fast
The system MUST fail fast if a config provides `custom.extra.rollout_matching.rollout_buffer`, with actionable guidance to remove it (no backward compatibility).

#### Scenario: rollout_buffer fails fast when present
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `custom.extra.rollout_matching.rollout_buffer`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to remove `rollout_buffer`.

### Requirement: Rollout-matching supports an offload context during colocate vLLM rollouts
When rollout-matching training uses vLLM rollouts in colocate mode, the system SHALL support an opt-in offload context
to reduce peak GPU memory usage during rollout inference.

Offload MUST be YAML-driven under `custom.extra.rollout_matching.offload`:
- `custom.extra.rollout_matching.offload.enabled` MUST accept a boolean and MUST default to `false`.
- `custom.extra.rollout_matching.offload.offload_model` MUST accept a boolean and MUST default to `false`.
- `custom.extra.rollout_matching.offload.offload_optimizer` MUST accept a boolean and MUST default to `false`.

Semantics when enabled:
- Offloading MUST occur only during rollout generation (no-grad inference).
- The trainer MUST restore model/optimizer state before teacher-forced forward/backprop.
- When rollout backend is not vLLM colocate (e.g., vLLM server mode or HF rollouts), offload settings MUST be ignored (no-op).
- When vLLM is lazily initialized, the offload context MUST cover all vLLM-side allocations required for rollout
  generation, including vLLM engine initialization and LoRA adapter loading/synchronization, not only the infer call.
- If offload is requested but cannot be applied safely under the current setup, the trainer MUST fail fast with an
  actionable error message that suggests at least one mitigation (e.g., disable offload, switch rollout backend to HF,
  or adjust DeepSpeed/ZeRO settings).

#### Scenario: Offload context does not break the training step
- **GIVEN** rollout-matching training is enabled with vLLM colocate rollouts
- **AND** offload is enabled for optimizer and/or model
- **WHEN** one training step executes
- **THEN** rollout inference completes without allocating training activations on GPU
- **AND** teacher-forced forward/backprop still executes successfully after offload restoration.

### Requirement: Rollout-matching trainer is YAML-gated
The system SHALL provide an opt-in rollout-matching training mode (alias: `stage_2`) that is enabled via YAML by setting:
- `custom.trainer_variant: rollout_matching_sft`.

When enabled, rollout-matching training SHALL be driven by YAML configuration and SHALL NOT require adding new hyperparameter CLI flags.

#### Scenario: Rollout-matching enabled via trainer_variant
- **GIVEN** a training config sets `custom.trainer_variant: rollout_matching_sft`
- **WHEN** `python -m src.sft --config <yaml>` is executed
- **THEN** training uses the rollout-matching trainer implementation
- **AND** the baseline training behaviour (alias: `stage_1`) is not used for that run.

### Requirement: Single-path training constructs one canonical teacher-forced target sequence
When rollout-matching training is enabled, the trainer SHALL implement a **single** training path that is expressed as:
- one canonical assistant token sequence per sample (`Y_train`), and
- one forward pass on that sequence, with per-token supervision masks.

There SHALL NOT exist separate training “paths” (e.g., “reordered-GT SFT” vs “self-context”); all supervision SHALL be expressed as per-token loss masks on a single teacher-forced forward pass.

The trainer SHALL construct the canonical assistant target sequence as:
- `Y_train = Y_rollout_prefix + SerializeAppend(FN_gt_objects) + EOS`

where:
- `Y_rollout_prefix` is a **prefix** of the model’s rollout assistant `response_token_ids` produced by autoregressive generation.
  - The trainer MAY perform **suffix-only trimming** to define `Y_rollout_prefix` so it is safe for append:
    - treat `<|im_end|>` as a hard stop and strip it (and any suffix after it) when present,
    - drop any trailing incomplete / invalid suffix tokens beyond the last complete predicted object boundary (e.g., when rollout is truncated mid-object),
    - drop the final top-level JSON closing brace `}` so the prefix ends in an **open** JSON object ready for append.
    - NOTE: Some tokenizers fuse closing punctuation (e.g., a single token may decode to `}}` or `}},`).
      - In those cases, the desired suffix cut can fall **inside** the final token.
      - The trainer MAY realize the cut by replacing ONLY the final token with a shorter tokenization that decodes to a strict prefix of that token’s decoded text (e.g., `}}` → `}`), while keeping all earlier token IDs unchanged.
  - The trainer SHALL NOT edit or re-tokenize any token **before** the cut boundary used for the prefix (no decode+re-encode; no pretty-printing; no key sorting).
  - Failure behavior: if the rollout does not contain an opening `{` OR the prefix cannot be made append-ready via suffix-only trimming, the trainer SHALL treat the rollout prefix as empty and use `Y_rollout_prefix = "{"` (no prefix supervision; all GT objects become FN and are appended).
- `FN_gt_objects` are the GT objects that are unmatched after matching (see matching requirement).
- `SerializeAppend(FN_gt_objects)` emits GT objects in the project’s JSON-only assistant schema (object-index JSON mapping `object_{n}` → `{desc, geometry}`) as an **append fragment**:
  - it SHALL emit **only** comma-separated `"object_{n}": {...}` entries (no outer `{}` wrapper),
  - it SHALL decide whether to emit a leading comma based on the last non-whitespace character of the decoded `Y_rollout_prefix`:
    - if the last non-whitespace character is `{` or `,`, it SHALL NOT emit a leading comma,
    - if the last non-whitespace character is `}`, it SHALL emit a leading `, `,
    - otherwise it SHALL error (the prefix is not append-ready).
  - it SHALL assign keys `object_{n}` starting from `n = max_object_index_in_prefix + 1` (or `n = 1` when no valid object index exists in the prefix),
    - `max_object_index_in_prefix` is the maximum integer `n` observed in any key matching the pattern `object_{n}` in the rollout prefix (best-effort; invalid/malformed keys are ignored),
  - it SHALL terminate by emitting the single top-level JSON closing brace `}` so `Y_train` is a valid JSON object before `EOS`.

There SHALL be exactly ONE forward pass per sample on the canonical encoding (same chat template as generation).

#### Scenario: Training uses one sequence and one forward pass
- **GIVEN** a batch under rollout-matching training
- **WHEN** the trainer executes one training step
- **THEN** it performs exactly one forward pass per sample on `Y_train`
- **AND** it computes exactly one total loss from that forward pass by applying per-token supervision masks.

### Requirement: Rollout generation returns token IDs (no grad) and selects one response
When rollout-matching training is enabled, the trainer SHALL perform an autoregressive rollout (generation) for each training sample:
- using the current model parameters,
- with gradients disabled during rollout, and
- producing both a decoded assistant response string and the corresponding assistant `response_token_ids` sequence (which defines `Y_rollout`).

The rollout decoding mode SHALL be configurable (at minimum: greedy and beam).

If decoding uses beam search, the trainer SHALL select exactly one rollout response for training: the best beam (highest logprob). Other beams MAY be logged for debugging, but SHALL NOT affect training.

#### Scenario: Beam search selects exactly one rollout response
- **GIVEN** rollout-matching training is enabled
- **AND** rollout decoding is configured as beam search
- **WHEN** a sample produces multiple beams
- **THEN** the trainer uses only the single best beam as `Y_rollout`
- **AND** all subsequent parsing, matching, and loss masking are computed from that selected beam only.

### Requirement: Mandatory FN append (recall recovery)
Unmatched GT objects (false negatives, `FN_gt_objects`) SHALL ALWAYS be appended to the end of `Y_rollout_prefix` to form `Y_train`.

Rationale (normative): GT annotations may be incomplete, but they do not hallucinate. Recall MUST be recovered via FN append, rather than suppressing unmatched GT.

#### Scenario: All GT is appended when no valid matches exist
- **GIVEN** a sample where rollout parsing yields zero usable predicted objects (or all pairs are gated out)
- **WHEN** `Y_train` is constructed
- **THEN** `FN_gt_objects` equals the full GT object set for that sample
- **AND** `SerializeAppend(FN_gt_objects)` is appended to `Y_rollout_prefix` before EOS.

### Requirement: Predicted order is defined by raw rollout text appearance (no silent reordering)
“Predicted order” SHALL be defined as the appearance order in the raw rollout string (the assistant response text decoded from `response_token_ids`).

Parsers/utilities used for rollout-matching training SHALL NOT sort JSON keys or re-serialize/pretty-print the rollout prefix in any way that changes tokenization.

If predicted objects are represented as a dict-like JSON object, the object order SHALL be defined by the order of appearance of each object’s key/value span in the raw rollout text (NOT lexicographic key order).

#### Scenario: Dict keys are not sorted for matching order
- **GIVEN** a rollout response string whose object keys appear as `"object_10": {...}, "object_2": {...}`
- **WHEN** predicted objects are enumerated for matching
- **THEN** the predicted order is `object_10` followed by `object_2` (appearance order)
- **AND** the trainer does not reorder them lexicographically.

### Requirement: Strict parsing drops invalid predicted objects (no repair)
The trainer SHALL require strict, schema-conformant predicted objects.

If a predicted object is malformed or violates schema (e.g., missing brackets, extra commas, nested/unexpected keys, wrong coord count, non-coord tokens in coord arrays, invalid geometry key, missing/empty `desc`), that object SHALL be marked invalid and DROPPED:
- invalid objects SHALL NOT participate in matching, and
- invalid objects SHALL NOT contribute to self-context supervision.

The trainer SHALL NOT perform token-inserting “JSON repair” for rollout-matching training (no adding braces/quotes, no filling missing tokens, no re-serializing JSON).

However, the trainer MAY perform **suffix-only trimming** of the rollout tokens to:
- drop trailing incomplete text when rollout is truncated, and/or
- strip a terminal `<|im_end|>` token, and/or
- drop the final top-level `}` to enable FN append under the canonical `Y_train` construction.

All tokens **before** the suffix-trim boundary SHALL remain unchanged; “dropping” affects only (a) the parsed object list and loss masks and (b) the chosen prefix cut for `Y_rollout_prefix`.

#### Scenario: One malformed object does not block training
- **GIVEN** a rollout response containing 3 object entries in appearance order
- **AND** the middle object entry is malformed (e.g., wrong coord count)
- **WHEN** strict parsing runs
- **THEN** exactly that object is marked invalid and excluded
- **AND** the other valid object entries remain eligible for matching and supervision
- **AND** training still proceeds because FN append provides teacher-forced supervision for unmatched GT.

#### Scenario: Truncated rollout tail is trimmed (suffix-only) and training proceeds
- **GIVEN** a rollout response that is truncated mid-object (no balanced JSON object is available)
- **WHEN** the trainer constructs `Y_rollout_prefix` via suffix-only trimming
- **THEN** it drops the incomplete trailing suffix and keeps only the prefix up to the last complete object boundary (or just `{` when none)
- **AND** it appends `SerializeAppend(FN_gt_objects)` and `EOS` to form a valid `Y_train`
- **AND** the training step completes without crashing.

### Requirement: Coord-slot token indices are derived from token-aligned parsing (object-level validity)
Self-context supervision SHALL NOT rely on searching for repeated `<|coord_k|>` patterns in the text.

Coord-slot token indices SHALL be obtained deterministically from parsing the rollout token sequence (or a token-aligned parse), producing for each VALID predicted object:
- `bbox_2d`: exactly 4 coord-token indices (for `[x1, y1, x2, y2]`), and
- `poly`: exactly `2N` coord-token indices with `N >= 3` and even length.

If coord-slot indices for an object are not uniquely determined / not trusted, that object SHALL be excluded from self-context supervision (object-level exclusion). The sample SHALL still train via mandatory FN append in the tail.

#### Token-aligned parsing / prefix trimming algorithm (normative sketch)
The trainer SHOULD implement a single streaming pass over the rollout assistant token IDs to determine:
- (a) a safe `Y_rollout_prefix` cut boundary (append-ready), and
- (b) per-object geometry kind + coord-token indices in **appearance order**.

Normative algorithm sketch (no string-search for coord patterns; structure-aware only):
1. **Precompute coord-token IDs**:
   - Build a `coord_id_set` from `get_coord_token_ids(tokenizer)` (size 1000).
2. **Materialize per-token decoded text pieces**:
   - For each assistant token id `t_i` in `response_token_ids`, compute `piece_i = tokenizer.decode([t_i], skip_special_tokens=False, clean_up_tokenization_spaces=False)`.
   - The trainer MUST NOT decode+re-encode the entire sequence to “normalize” spacing or quotes.
3. **Run a streaming JSON structure scanner over `piece_i` in order**:
   - Track JSON state variables across characters:
     - `in_string` + `escape` (to ignore braces/brackets inside JSON strings),
     - `brace_depth` for `{}` and `bracket_depth` for `[]`.
   - Track parse context:
     - current top-level key string (e.g., `"object_17"`),
     - current object index `n` (when inside an `"object_n": {...}` value),
     - current geometry key (`bbox_2d` or `poly`) and whether the scanner is currently inside its array value.
4. **Determine appearance-order objects and coord-token indices**:
   - When a JSON string is parsed at `brace_depth == 1` in a “expecting key” position, and the key matches `object_{n}` where `n` is an integer, start a new predicted object context in appearance order.
   - Within that object’s value dict (`brace_depth == 2`), the scanner MUST locate exactly one geometry key (`bbox_2d` or `poly`) and then enter “capture mode” on the next `[` that begins that geometry’s array.
   - While in capture mode for a geometry array, every assistant token position `i` whose token id is in `coord_id_set` SHALL be appended to that object’s `coord_token_indices` list (even if the coord token is surrounded by JSON quotes).
   - Capture mode ends when the corresponding array `]` is closed (i.e., `bracket_depth` returns to the value it had immediately before the geometry array opened).
   - After capture ends:
     - `bbox_2d` MUST have exactly 4 coord-token indices, else the object is invalid.
     - `poly` MUST have an even number of indices and at least 6 total indices, else the object is invalid.
   - If an object contains multiple geometry keys, nested/unexpected geometry keys, or the geometry array is not fully closed before the chosen prefix cut boundary, the object is invalid and MUST be dropped (no repair).
5. **Determine the append-ready `Y_rollout_prefix` cut boundary**:
   - During the same streaming scan, the trainer MUST record candidate cut positions corresponding to the **end of the last complete predicted object entry** in the top-level JSON object.
     - A candidate cut occurs after a `}` that reduces `brace_depth` from 2 → 1 (end of an `"object_n": {...}` value dict).
     - A candidate cut MAY include a following comma token if it is fused (e.g., token decodes to `},`); `SerializeAppend` MUST handle prefixes whose last non-whitespace char is `{`, `,`, or `}` as specified above.
   - The selected cut boundary SHALL be the last recorded candidate at or before the end of the rollout, after stripping any trailing end-of-turn tokens like `<|im_end|>`.
   - **Fused-suffix handling**: if the last candidate cut falls inside the final token (e.g., a token decodes to `}}` and the cut is after the first `}`), the trainer MAY replace ONLY that final token with a shorter tokenization that decodes to the needed substring (e.g., `}}` → `}`), keeping all earlier token IDs unchanged.

#### Scenario: Ambiguous coord-slot alignment excludes object from self-context supervision
- **GIVEN** a predicted object whose geometry can be parsed but whose coord token indices cannot be uniquely aligned to `response_token_ids`
- **WHEN** the trainer builds self-context supervision masks
- **THEN** that object contributes no self-context coord loss
- **AND** its GT counterpart (if any) is treated as unmatched and included in `FN_gt_objects` for tail append.

### Requirement: Matching baseline uses Hungarian assignment with dummy augmentation and maskIoU gating
Matching SHALL be done via Hungarian assignment with dummy augmentation to allow FP/FN.

MVP baseline (configurable via YAML, with defaults defined by the trainer):
- Candidate reduction: for each predicted object, the trainer SHALL compute AABB IoU against GT AABBs and select top-k candidates before expensive geometry. If AABB IoU is all zero or candidates are insufficient, a deterministic fallback SHALL be used (e.g., keep top-k by center distance).
- Geometry cost: the trainer SHALL compute `maskIoU` between predicted and GT shapes (bbox/poly rasterized to masks) and define:
  - `cost_geo(i, j) = 1 - maskIoU(i, j)`
  - `maskIoU` SHALL be computed in **norm1000 space** on a fixed virtual canvas of size `R x R` (default `R=256`), by:
    - treating `poly` as a single-ring polygon,
    - treating `bbox_2d` as its quadrilateral polygon,
    - clamping coordinates to `[0, 999]` before projection to the `R x R` canvas.
- Gating (pre-assignment): pairs with `maskIoU < threshold` SHALL be treated as infeasible (equivalently `cost = +INF`) BEFORE assignment, to avoid wrong matches.
- Dummy semantics:
  - `pred -> dummy` represents FP (low penalty / light control only),
  - `dummy -> gt` represents FN and SHALL be handled by mandatory FN append (the GT object is appended, not silently dropped).

The matching output SHALL determine:
- matched pairs eligible for self-context supervision (subject to coord-slot alignment validity), and
- `FN_gt_objects` (all GT objects not matched to a usable predicted object).

#### Scenario: Pre-assignment gating prevents wrong matches and triggers FN append
- **GIVEN** a predicted shape that has `maskIoU < threshold` with every GT shape
- **WHEN** Hungarian matching runs
- **THEN** all `pred -> gt` edges are infeasible (`+INF`) prior to assignment
- **AND** the predicted object is assigned to dummy (FP)
- **AND** all GT objects remain unmatched and are appended via `FN_gt_objects`.

### Requirement: Poly self-context targets use Sinkhorn OT with barycentric projection only
For matched pairs where poly is involved (poly<->poly, bbox_2d<->poly, poly<->bbox_2d), the trainer SHALL construct self-context coord supervision targets using OT + barycentric projection (ONLY barycentric; no mixture):

- Represent both shapes as point sets:
  - `poly`: its vertex points from parsed coord tokens (point count `N >= 3`)
  - `bbox_2d`: MVP point set = 4 corners `(x1,y1),(x2,y1),(x2,y2),(x1,y2)` derived from `[x1,y1,x2,y2]`
- Compute an OT plan `T` via Sinkhorn on a chosen cost (L1 or L2 in norm1000 space), and treat `T` as a stop-grad aligner.
- Use barycentric projection ONLY:
  - `g_hat_i = sum_j ((T_ij / sum_j T_ij) * g_j)`
- Convert each `g_hat_i` into unimodal soft labels `q(x)` and `q(y)` over coord bins.
- Apply token-level coord supervision at the predicted object’s supervised coord token indices (poly vertices, or bbox tokens) using those `q` targets.

The trainer SHALL NOT implement or reference any “mixture” target construction for this OT alignment.

#### Scenario: Poly prediction is supervised by barycentric-projected targets
- **GIVEN** a matched pair where the predicted geometry is `poly`
- **WHEN** OT+barycentric target construction runs
- **THEN** each predicted poly coord token position receives a unimodal soft target derived from barycentric projection onto the GT shape
- **AND** the resulting targets are used for coord-token supervision only (no decoded coordinate regression losses).

### Requirement: Unified loss definition uses token masks (no decoded-coordinate losses/metrics)
The trainer SHALL compute a single total loss from the logits of the ONE forward pass on `Y_train` by applying per-token supervision masks.

At ALL `<|coord_*|>` supervised positions (both: matched coord slots in the rollout prefix AND coord tokens in the appended GT tail), the trainer SHALL compute:

`L_coord = softCE(q, p) + λ * W1(p, q) + λ_gate * GateMassLeak(p_full_vocab)`

where:
- `p` is the coord-bin distribution derived from logits restricted to the coord sub-vocabulary,
- `q` is a unimodal soft target over coord bins (Gaussian-like, configured by σ), and
- `GateMassLeak` penalizes probability mass outside the coord sub-vocabulary at coord positions.

For non-coordinate tokens in the appended GT tail segment, the trainer SHALL compute standard hard CE over the full vocabulary.
For this rollout-matching rollout, the trainer SHALL ignore (mask out) CE supervision for tokens that correspond to the JSON string *value* of `desc` fields in the appended GT tail (i.e., the token span inside `"desc": "<VALUE>"`). JSON structure tokens (braces/quotes/keys/colons/commas) in the appended GT tail remain supervised by CE.

The trainer SHALL NOT compute any of the legacy decoded-coordinate losses or metrics:
- no expectation/argmax/median decoding losses,
- no L1 regression,
- no IoU/GIoU/maskIoU loss terms,
- no polygon mask losses, smoothness losses, or geometry regularizers,
- no IoU/GIoU/maskIoU metric logging.

#### Scenario: Prefix coord tokens can be supervised without supervising prefix text tokens
- **GIVEN** a sample with at least one matched predicted object in the rollout prefix
- **WHEN** losses are computed for that sample
- **THEN** coord tokens at matched coord-slot indices in the prefix contribute `L_coord`
- **AND** non-coord tokens in the rollout prefix contribute neither CE nor coord loss (they are masked out)
- **AND** appended tail non-coord tokens contribute standard CE EXCEPT `desc` value tokens, which are masked out.

### Requirement: Canonical encoding and supervision index sanity checks
The ONE teacher-forced forward pass SHALL use the exact same prompt/messages encoding (chat template + image tokens placement) as rollout generation.

Labels SHALL align to assistant response tokens only; prompt tokens MUST be `ignore_index` (or equivalent).

The trainer MUST implement two engineering sanity checks:
- (a) prompt+image prefix tokenization matches generation (e.g., `len` and/or hash of the prompt token IDs),
- (b) all supervised `coord_token_indices` fall within the assistant-label span (never into the prompt span).

#### Scenario: Supervision indices are validated against assistant span
- **GIVEN** a sample with computed coord token indices for self-context supervision in the rollout prefix
- **WHEN** the trainer builds loss masks for the forward pass
- **THEN** it asserts every supervised coord index lies within the assistant portion of the encoded sequence
- **AND** it errors clearly if any index points into the prompt/image prefix (preventing silent misalignment).

### Requirement: Training-time counters expose parsing/matching health (without geometry metrics)
The trainer SHALL expose counters that record:
- number of predicted objects parsed as valid vs invalid (dropped),
- number of objects excluded due to coord-slot alignment ambiguity,
- match rate (#matched vs #GT),
- number of FN appended objects,
- number of gating rejections,
- rollout decoding mode (greedy/beam) and any truncation flags.

These counters SHALL NOT include IoU/GIoU/maskIoU numeric metric logging (those values are used internally for matching only).

#### Scenario: Parse failures are visible without crashing
- **GIVEN** a batch where some samples have malformed rollout JSON objects
- **WHEN** the trainer processes that batch
- **THEN** invalid objects are dropped and counted
- **AND** the training step completes without crashing due to mandatory FN append supervision in the tail.
