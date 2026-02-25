# rollout-matching-sft Specification

## Purpose
Define the rollout-matching SFT trainer contract, including rollout generation backends, vLLM server/weight-sync behavior, decode microbatching, and matching/packing invariants.

## Requirements
### Requirement: Rollout-aligned Stage-2 uses explicit objective pipeline (no implicit defaults)
When `custom.trainer_variant: stage2_rollout_aligned`, the rollout-aligned teacher-forcing objective MUST be fully determined by the declared module pipeline under:
- `rollout_matching.pipeline.objective[]` and `rollout_matching.pipeline.diagnostics[]`.

Normative behavior:
- `rollout_matching.pipeline` MUST be present (no implicit default manifest).
- Legacy aux-loss config surfaces MUST be rejected, including `custom.coord_soft_ce_w1.*`.

#### Scenario: Missing rollout pipeline fails fast
- **WHEN** `custom.trainer_variant: stage2_rollout_aligned`
- **AND** `rollout_matching.pipeline` is absent
- **THEN** config loading fails fast before trainer init
- **AND** the error indicates `rollout_matching.pipeline` is required.

### Requirement: Rollout pipeline specs are explicit and complete (no implicit defaults)
Rollout pipeline module specs MUST be authored with explicit fields and complete module configs to prevent silent drift from default injection.

Normative behavior:
- Each entry in `rollout_matching.pipeline.objective[]` and `rollout_matching.pipeline.diagnostics[]` MUST include:
  - `name`, `enabled`, `weight`, `channels`, `config`.
- `channels` MUST be explicitly authored as a subset of `{A,B}`.
- `config` MUST include exactly the allowlisted keys for the referenced module:
  - missing required keys MUST fail fast (no implicit defaults),
  - unknown keys MUST fail fast (no escape-hatch aliases).

### Requirement: Rollout pipeline module configs are strict and canonical (no aliases)
Rollout pipeline module configs MUST be strict and MUST reject unknown keys and legacy alias keys.

Normative behavior:
- `bbox_geo.config` MUST accept only:
  - `smoothl1_weight`
  - `ciou_weight`
- `coord_reg.config` MUST accept only canonical keys, including:
  - `coord_ce_weight`
  - `soft_ce_weight`
  - `w1_weight`
  - `coord_gate_weight`
  - `text_gate_weight`
  - `temperature`
  - `target_sigma`
  - `target_truncate`
- Legacy alias keys (e.g., `bbox_smoothl1_weight`, `coord_soft_ce_weight`, `coord_w1_weight`) MUST be rejected.

### Requirement: Rollout-aligned Stage-2 supports text_gate via coord_reg module config
Rollout-aligned Stage-2 MUST support `text_gate` as part of `coord_reg` with a typed weight:
- `rollout_matching.pipeline.objective[*].config.text_gate_weight`

Normative behavior:
- `text_gate_weight > 0` MUST introduce a non-zero `text_gate` contribution when coord-vocab mass appears at supervised text positions under `context=rollout` (FP-neutral).

### Requirement: Stage-2 post-rollout packing selection uses deterministic ms-swift-like binpacking
When rollout-matching training is enabled (`custom.trainer_variant: stage2_rollout_aligned`) and post-rollout packing is enabled (`training.packing: true`), the trainer SHALL select which buffered segments are included in the packed teacher-forced forward pass using a deterministic, ms-swift-like constant-volume binpacking heuristic.

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
When rollout-matching training is enabled (`custom.trainer_variant: stage2_rollout_aligned`), the system SHALL support packing for the *post-rollout teacher-forced forward pass* to improve training efficiency.

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
- **GIVEN** a YAML config sets `custom.trainer_variant: stage2_rollout_aligned`
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
When rollout-matching training is enabled (`custom.trainer_variant: stage2_rollout_aligned`), the system SHALL support generating rollouts using a vLLM backend, while keeping teacher-forced forward/backprop on the normal training model.

Rollout-matching settings are a first-class top-level namespace:
- Rollout-matching settings MUST be authored under top-level `rollout_matching.*`.
- Legacy placement under `custom.extra.rollout_matching.*` is unsupported and MUST fail fast with actionable guidance to migrate to `rollout_matching.*`.

Backend selection MUST be YAML-driven under `rollout_matching`:
- `rollout_backend` MUST accept `"vllm"` or `"hf"`.
- `rollout_backend` MUST default to `"vllm"`.

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
  - Under multi-process learners (`world_size > 1`), `rollout_matching.vllm.sync.mode` MUST resolve to `full`.
  - If these requirements cannot be satisfied (e.g., communicator init fails, misconfigured sync mode), the trainer MUST fail fast with actionable guidance.

Server connectivity MUST be YAML-driven under `rollout_matching.vllm.server` (mapping).

Server connectivity config MUST be expressed as an explicit `servers` list:
- `servers` MUST be a non-empty list of mappings.
- Each entry MUST contain:
  - `base_url` (string)
  - `group_port` (int)

Legacy paired-list server form (`rollout_matching.vllm.server.base_url` + `rollout_matching.vllm.server.group_port`) is unsupported and MUST fail fast with actionable guidance to migrate to `rollout_matching.vllm.server.servers[]`.

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

Weight sync mode selection MUST be YAML-driven under `rollout_matching.vllm.sync`:
- `sync.mode` MUST accept `full`, `adapter`, or `auto`.
- If unset, `sync.mode` MUST default to `full` (robust for multimodal + DoRA).
- `sync.fallback_to_full` MUST accept a boolean and MUST default to `true`.

Weight sync behavior (normative):
- `sync.mode: full` MUST sync full merged weights (GRPO-style) into vLLM.
- `sync.mode: adapter` MUST sync adapters only, and MUST require `rollout_matching.vllm.enable_lora: true`.
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
- **AND** `rollout_matching.rollout_backend: vllm`
- **AND** `rollout_matching.vllm.mode` is unset
- **WHEN** one training step executes
- **THEN** the trainer uses vLLM in colocate mode
- **AND** the teacher-forced forward/backprop uses the training model
- **AND** stage_2 prompt-prefix token-id alignment checks apply unchanged.

#### Scenario: Multi-server pairing is strict and deterministic
- **GIVEN** `rollout_matching.vllm.mode: server`
- **AND** server connectivity is configured using the legacy paired-list form (`rollout_matching.vllm.server.base_url` / `rollout_matching.vllm.server.group_port`)
- **WHEN** configuration is parsed/materialized
- **THEN** it fails fast with guidance to migrate to `rollout_matching.vllm.server.servers[]`.

#### Scenario: Multi-server request distribution is deterministic
- **GIVEN** `rollout_matching.vllm.mode: server`
- **AND** multiple servers are configured in a fixed order
- **AND** a fixed list of rollout requests in a fixed order
- **WHEN** the trainer distributes requests to servers twice
- **THEN** it assigns the same contiguous chunks to the same server indices both times
- **AND** the reassembled outputs preserve the original request order.

#### Scenario: Adapter sync can fall back to full sync
- **GIVEN** `rollout_matching.vllm.sync.mode: adapter`
- **AND** `rollout_matching.vllm.sync.fallback_to_full: true`
- **WHEN** adapter sync fails due to a runtime incompatibility
- **THEN** the trainer logs a warning
- **AND** switches to full merged-weight sync for subsequent E-steps.

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
- **AND** `custom.extra.rollout_matching.rollout_backend: vllm`
- **AND** `custom.extra.rollout_matching.vllm.mode: server`
- **AND** training is launched under `torchrun` with `world_size=2`
- **WHEN** the trainer performs a fresh-rollout step that requires a server sync
- **THEN** rank0 performs the server weight sync and other ranks do not
- **AND** all ranks proceed to issue rollout `/infer/` requests without deadlock.


#### Scenario: Non-full sync mode fails fast under multi-process learner
- **GIVEN** rollout-matching training is enabled with server mode under `world_size > 1`
- **AND** `custom.extra.rollout_matching.vllm.sync.mode: adapter` (or `auto` resolving to adapter)
- **WHEN** training starts (before the first rollout)
- **THEN** configuration validation fails fast with actionable guidance to use `sync.mode: full`.


### Requirement: Stage_2 supports a 3v1 rollout-server + learner workflow (actors vs learner)
The stage_2 rollout-matching trainer (`custom.trainer_variant: stage2_rollout_aligned`) MUST support a practical single-node workflow where rollout generation runs on dedicated GPUs via a vLLM server, and teacher-forced SFT training runs on a separate GPU.

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
- **AND** the stage_2 learner is launched on GPU 3 with `rollout_matching.vllm.mode: server`
- **WHEN** training runs for one optimizer step
- **THEN** rollouts are generated on the rollout GPUs
- **AND** teacher-forced forward/backward runs only on the learner GPU
- **AND** weights are synchronized in-memory without requiring checkpoint reload from disk.

### Requirement: Server-mode rollouts are paper-reproducible via logged metadata
When `rollout_matching.vllm.mode: server` is enabled, the trainer MUST log sufficient metadata to reproduce and debug the run.

At minimum, the trainer MUST log:
- the effective server list (base URLs and group ports)
- the effective weight sync mode (`full` vs `adapter`) including any fallback events
- the per-batch rollout seed used for `RequestConfig.seed`

#### Scenario: Server mode logs reproducibility metadata
- **GIVEN** server mode is enabled
- **WHEN** the trainer performs a fresh-rollout step (E-step)
- **THEN** it logs the server endpoints, sync mode, and rollout seed for that step.

### Requirement: Rollout generation supports microbatched decoding within each rank
When rollout-matching training is enabled, the trainer SHALL support generating rollouts for multiple samples in a single backend decode call (HF `generate()` or vLLM `/infer/`), controlled by a normalized rollout configuration knob.

Rollout config keys and nested structures MUST be validated through schema-derived strict contracts before runtime rollout execution.

Normalized rollout knob (normative):
- `decode_batch_size` (int) in the trainer’s injected rollout config contract.

Config source semantics (normative):
- For canonical grouped authoring, rollout backend selection MUST come from `rollout_matching.rollout_backend`.
- For canonical grouped authoring, `decode_batch_size` MUST come from `rollout_matching.decode_batch_size`.
- For Stage-2 AB rollout knobs that previously existed under `custom.extra.rollout_matching.*`, canonical migration MUST be path-only relocation to `rollout_matching.*` with the same subkey names.
- Legacy Stage-2 alias keys under `custom.extra.rollout_matching.*` are unsupported and MUST fail fast with migration guidance.
- The trainer/runtime contract MUST expose a single resolved `decode_batch_size` value to rollout execution code.

Schema-derived strictness (normative):
- Rollout config key acceptance MUST be derived from typed schema definitions and enforced at config-load time.
- Unknown rollout keys (top-level or nested) MUST fail fast with dotted-path error messages.
- Unknown-key dotted paths MUST include list indices when present (e.g., `rollout_matching.vllm.server.servers[0].unknown_flag`).
- Runtime rollout validators MAY enforce execution-dependent constraints (runtime mode compatibility, numeric bounds) but MUST NOT be the long-term owner of static schema key acceptance.
- Rollout server schema supports only `rollout_matching.vllm.server.servers[]`; legacy paired-list form (`vllm.server.base_url` + `vllm.server.group_port`) is removed and MUST fail fast with migration guidance.
- Stage-2 launcher preflight MAY expose a projected `server_base_urls` array for launch wiring, but that projection MUST be derived from canonical `servers[]` entries and MUST NOT replace schema requirements for `base_url` + `group_port`.

Semantics (normative):
- `decode_batch_size` denotes the maximum number of sequences decoded per rollout GPU in one generation call.
- The trainer MUST enforce this bound for both HF and vLLM backends.
- `decode_batch_size` is the single source of truth for rollout decode/evaluation microbatching in rollout-aware trainer variants.
- `training.per_device_eval_batch_size` and similar per-device eval knobs MUST NOT independently control rollout decode/evaluation batch behavior.

Defaulting (normative):
- If `decode_batch_size` is unset, the implementation MUST default it to `1` (conservative).
- Higher-level experiment templates MAY set a larger default explicitly (e.g., Stage2-AB YAML under `configs/stage2_two_channel/**` uses `4`).

#### Scenario: Canonical Stage-2 key controls decode microbatching
- **WHEN** a Stage-2 AB config sets `rollout_matching.decode_batch_size: M` where `M > 1`
- **THEN** rollout generation uses `M` as the resolved decode batch size in trainer rollout config.

#### Scenario: Microbatching increases decode parallelism without changing outputs format
- **WHEN** rollout-matching training runs with resolved `decode_batch_size=M` where `M > 1`
- **THEN** the trainer performs batched decode calls for up to `M` samples per rollout GPU
- **AND** it returns per-sample `response_token_ids` suitable for strict token-aligned parsing.

#### Scenario: Eval per-device knobs do not override rollout decode batching
- **WHEN** rollout-matching training runs with `rollout_matching.decode_batch_size=M` and `training.per_device_eval_batch_size=N` where `M != N`
- **THEN** rollout decode/evaluation microbatching follows `M`
- **AND** `training.per_device_eval_batch_size` does not independently change rollout decode/evaluation behavior.

#### Scenario: Legacy decode key path fails fast
- **WHEN** a Stage-2 config sets `custom.extra.rollout_matching.decode_batch_size`
- **THEN** config loading fails fast with guidance to migrate to `rollout_matching.decode_batch_size`.

#### Scenario: Legacy rollout backend key path fails fast
- **WHEN** a Stage-2 config sets `custom.extra.rollout_matching.rollout_backend`
- **THEN** config loading fails fast with guidance to migrate to `rollout_matching.rollout_backend`.

#### Scenario: Unknown rollout key fails before rollout trainer execution
- **WHEN** config includes `rollout_matching.unknown_rollout_key`
- **THEN** loader fails fast during schema parsing
- **AND** rollout trainer is not constructed.

#### Scenario: Unknown nested decoding key fails before execution
- **WHEN** config includes `rollout_matching.decoding.unknown_decoding_key`
- **THEN** loader fails fast during schema parsing with nested dotted-path context.

#### Scenario: Legacy rollout server paired-list shape fails fast
- **WHEN** config uses `rollout_matching.vllm.server.base_url` and `rollout_matching.vllm.server.group_port`
- **THEN** schema loading fails fast with guidance to migrate to `rollout_matching.vllm.server.servers[]`.

### Requirement: Trainer rollout contract is source-agnostic after normalization
Rollout-matching trainer internals MUST consume a source-agnostic rollout config contract after normalization from canonical grouped keys.

Normative behavior:
- Runtime normalization MUST produce one `rollout_matching_cfg` mapping injected into rollout-aware trainers.
- Rollout execution code MUST read resolved rollout values from the injected contract and MUST NOT branch on original YAML source path.
- Normalized contract fields MUST include at least:
  - rollout backend selection,
  - vLLM mode/server/sync settings,
  - decoding parameters,
  - repeat-terminate settings,
  - matching/packing runtime knobs consumed by trainer validation and execution.

#### Scenario: Canonical grouped config drives trainer contract
- **WHEN** a config defines rollout settings under canonical grouped keys
- **THEN** trainer-side `rollout_matching_cfg` contains the normalized rollout contract used by rollout execution.

### Requirement: Rollout schema validation ownership is centralized
The rollout schema contract MUST be owned centrally by config schema parsing, and reused consistently by runtime/preflight consumers.

Normative behavior:
- One schema-driven rollout contract defines accepted rollout keys and nested structures.
- Preflight/runtime consumers read normalized rollout config from the shared loader path and MUST NOT define conflicting parallel schema ownership.
- Contract changes for rollout keys MUST be implemented by updating typed schema definitions, not by adding independent manual allowlists in each consumer.
- Runtime duplicate static-key checks may remain temporarily during migration as safety gates, but MUST be removed once loader-level parity checks pass.
- `src/sft.py` continues to inject normalized rollout configuration into trainer `rollout_matching_cfg`; schema refactor MUST keep this runtime interface stable.

#### Scenario: Preflight/runtime consume a shared validated rollout contract
- **WHEN** canonical rollout config is valid and loaded
- **THEN** both launcher preflight and trainer runtime observe the same validated rollout contract.

#### Scenario: Runtime rollout injection path stays stable
- **WHEN** rollout config passes schema validation
- **THEN** runtime still injects `rollout_matching_cfg` from the normalized loader contract
- **AND** rollout trainers do not depend on alternative legacy config source paths.

### Requirement: Legacy rollout batch-size knobs fail fast
The system MUST fail fast if a config provides any legacy rollout batch-size knob under `rollout_matching`, with actionable guidance to migrate to the unified decode batching knob `rollout_matching.decode_batch_size`.

Legacy knobs (normative):
- `rollout_matching.rollout_generate_batch_size`
- `rollout_matching.rollout_infer_batch_size`

#### Scenario: rollout_generate_batch_size fails fast when present
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `rollout_matching.rollout_generate_batch_size`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to remove it and use `rollout_matching.decode_batch_size`.

#### Scenario: rollout_infer_batch_size fails fast when present
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `rollout_matching.rollout_infer_batch_size`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to remove it and use `rollout_matching.decode_batch_size`.

### Requirement: Window-aware post-rollout packing scope knob is removed
The system MUST fail fast if a config provides `rollout_matching.post_rollout_pack_scope` (any value), with actionable guidance to remove it.

Normative behavior:
- Post-rollout packing behavior MUST be the standardized micro-scope dynamic packing semantics.
- Window-scoped packing is not supported.

#### Scenario: post_rollout_pack_scope fails fast when present
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `rollout_matching.post_rollout_pack_scope`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to remove `post_rollout_pack_scope`.

### Requirement: vLLM server mode derives per-rank request chunking from server world size
When rollout-matching training uses vLLM server mode (`rollout_matching.rollout_backend=vllm` and `rollout_matching.vllm.mode=server`), the trainer MUST derive per-rank request chunk sizing from the rollout server world size and learner DDP world size to preserve the per-rollout-GPU cap defined by `decode_batch_size`.

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
The system MUST fail fast if a config provides `rollout_matching.rollout_buffer`, with actionable guidance to remove it (no backward compatibility).

#### Scenario: rollout_buffer fails fast when present
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `rollout_matching.rollout_buffer`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to remove `rollout_buffer`.

### Requirement: Rollout-matching supports an offload context during colocate vLLM rollouts
When rollout-matching training uses vLLM rollouts in colocate mode, the system SHALL support an opt-in offload context
to reduce peak GPU memory usage during rollout inference.

Offload MUST be YAML-driven under `rollout_matching.offload`:
- `rollout_matching.offload.enabled` MUST accept a boolean and MUST default to `false`.
- `rollout_matching.offload.offload_model` MUST accept a boolean and MUST default to `false`.
- `rollout_matching.offload.offload_optimizer` MUST accept a boolean and MUST default to `false`.

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
- `custom.trainer_variant: stage2_rollout_aligned`.

When enabled, rollout-matching training SHALL be driven by YAML configuration and SHALL NOT require adding new hyperparameter CLI flags.

#### Scenario: Rollout-matching enabled via trainer_variant
- **GIVEN** a training config sets `custom.trainer_variant: stage2_rollout_aligned`
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

#### Scenario: Highest retained object key controls FN start even when object is invalid
- **GIVEN** retained rollout prefix contains `object_2` (valid) and `object_9` (invalid object body)
- **WHEN** `SerializeAppend(FN_gt_objects)` assigns new keys
- **THEN** `max_object_index_in_prefix` is `9`
- **AND** the first FN key is `object_10`.

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

### Requirement: FN append serialization honors configured object field order
When rollout-matching builds `Y_train` via mandatory FN append, each appended object payload SHALL follow `custom.object_field_order`.

Normative behavior:
- `desc_first`: append payload uses `{desc, bbox_2d}` or `{desc, poly}` depending on object geometry type.
- `geometry_first`: append payload uses `{bbox_2d, desc}` or `{poly, desc}` depending on object geometry type.
- Geometry key can be `bbox_2d` or `poly`.
- The serializer MUST NOT emit a synthetic key literally named `geometry`.

This requirement applies only to field order within each appended object payload and MUST NOT alter:
- object key numbering (`object_{n}` continuation),
- predicted object appearance-order parsing,
- matching order semantics.

#### Scenario: geometry-first changes only per-object field order in FN append
- **GIVEN** `custom.object_field_order: geometry_first`
- **AND** Channel-B has unmatched GT objects to append
- **WHEN** `SerializeAppend(FN_gt_objects)` is produced
- **THEN** each appended object places its concrete geometry key (`bbox_2d` or `poly`) before `desc`
- **AND** object keys still start at `max_object_index_in_prefix + 1`.

#### Scenario: desc-first remains baseline append layout
- **GIVEN** `custom.object_field_order` is omitted or set to `desc_first`
- **WHEN** FN append fragment is serialized
- **THEN** appended object payloads keep `desc` before the concrete geometry key (`bbox_2d` or `poly`).

### Requirement: Field-order variation is schema-equivalent for strict parsing
Strict parsing for rollout matching SHALL treat `desc_first` and `geometry_first` object payloads as schema-equivalent.

Normative behavior:
- Reordering `{desc, bbox_2d}` to `{bbox_2d, desc}` or `{desc, poly}` to `{poly, desc}` MUST NOT by itself invalidate an object.
- Existing strict checks (missing desc, invalid geometry, wrong arity, bad coord tokens, etc.) remain unchanged.

#### Scenario: geometry-first object remains valid under strict parse
- **GIVEN** a rollout object encoded as `{\"bbox_2d\": [...], \"desc\": \"...\"}`
- **WHEN** strict parsing runs
- **THEN** the object is considered valid if all existing schema constraints pass
- **AND** it is not dropped solely due to field order.

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


### Requirement: Server/HF rollout sampling is configured under decoding.*
Rollout decoding knobs MUST be expressed under:
- `custom.extra.rollout_matching.decoding` (mapping)

Supported decoding keys (v1):
- `custom.extra.rollout_matching.decoding.temperature` (float, `>= 0`; greedy if `== 0`)
- `custom.extra.rollout_matching.decoding.top_p` (float, `(0, 1]`, default `1.0`)
- `custom.extra.rollout_matching.decoding.top_k` (int, default `-1`)

Legacy decoding keys are removed (breaking):
- If a config provides any of:
  - `custom.extra.rollout_matching.temperature`
  - `custom.extra.rollout_matching.top_p`
  - `custom.extra.rollout_matching.top_k`
  the system MUST fail fast with guidance to migrate to `custom.extra.rollout_matching.decoding.*`.

Robustness-first sampling note:
- When sampling is enabled (e.g., `temperature > 0`) and the server backend retries/splits requests for robustness,
  strict bitwise determinism is not required; the trainer MUST log sufficient metadata to audit effective decoding behavior.

#### Scenario: Legacy decoding keys fail fast
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `custom.extra.rollout_matching.temperature`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to use `custom.extra.rollout_matching.decoding.temperature` instead.


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


### Requirement: Rollout-matching exposes stable submodule contracts by concern
The rollout-matching capability SHALL expose stable public contracts for parsing, matching, packing, and backend orchestration via dedicated submodules.
The trainer-facing module MAY provide compatibility re-exports during migration, but behavior ownership MUST live in the dedicated submodules.

#### Scenario: Shared parsing contract is importable without trainer class dependency
- **WHEN** a consumer imports rollout parsing contracts for validation/testing
- **THEN** it can do so without importing the trainer class implementation
- **AND** parsed output contracts remain stable for downstream use.


### Requirement: Rollout backend orchestration uses a backend interface contract
Rollout backend selection and synchronization SHALL be mediated through a backend interface contract rather than inline trainer branches.
Supported backend implementations MUST preserve existing rollout semantics and output fields required by strict parsing/matching.

#### Scenario: Backend implementation swap preserves rollout contract
- **GIVEN** the same rollout config semantics and sample inputs
- **WHEN** backend implementation is switched through the backend interface
- **THEN** returned rollout payload fields required by parsing/matching are preserved
- **AND** trainer-side supervision construction remains valid.


### Requirement: Post-rollout packing scheduling is reusable and deterministic
Post-rollout packing/window scheduling SHALL be implemented in reusable helpers consumable by rollout and Stage-2 paths.
Given identical segment inputs and ordering, helper outputs MUST be deterministic.

#### Scenario: Shared packing helper yields deterministic selection
- **GIVEN** identical segment metadata and insertion order
- **WHEN** the shared packing scheduler runs twice
- **THEN** it yields the same selected segment set and order in both runs.


### Requirement: Coord-vocab gate math reuses the shared canonical helper
When computing coord-vocab gate loss/mass terms (used as part of coord supervision and diagnostics), rollout-matching paths SHALL delegate to the shared helper used by training and metrics so numeric fences (NaN/Inf handling, clamping, temperature scaling) remain consistent across consumers.

#### Scenario: Gate-loss numeric fences are consistent across training and rollout paths
- **GIVEN** identical full-vocab logits, coord-vocab logits, and temperature
- **WHEN** the gate term is computed in training/metrics and in rollout-matching
- **THEN** the per-token gate-loss values match (up to floating-point tolerance) due to shared helper reuse.
