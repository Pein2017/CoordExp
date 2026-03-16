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

#### Scenario: Incomplete rollout module specification fails fast
- **WHEN** a rollout pipeline entry omits one of the required fields (`name`, `enabled`, `weight`, `channels`, `config`)
- **THEN** configuration parsing fails fast before trainer initialization
- **AND** diagnostics identify the missing field path.

### Requirement: Rollout pipeline module configs are strict and canonical (no aliases)
Rollout pipeline module configs MUST be strict and MUST reject unknown keys and legacy alias keys.

Normative behavior:
- `bbox_geo.config` MUST accept only:
  - `smoothl1_weight`
  - `ciou_weight`
- `bbox_size_aux.config` MUST accept only:
  - `log_wh_weight`
  - `log_area_weight`
  - `oversize_penalty_weight`
  - `oversize_area_frac_threshold`
  - `oversize_log_w_threshold`
  - `oversize_log_h_threshold`
  - `eps`
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

#### Scenario: Alias key in rollout module config fails fast
- **WHEN** `rollout_matching.pipeline.objective[*].name=bbox_size_aux`
- **AND** the module config contains `bbox_smoothl1_weight`
- **THEN** configuration parsing fails fast
- **AND** the error indicates the canonical `bbox_size_aux.config.*` key family
  must be used.

### Requirement: Rollout-aligned Stage-2 supports text_gate via coord_reg module config
Rollout-aligned Stage-2 MUST support `text_gate` as part of `coord_reg` with a typed weight:
- `rollout_matching.pipeline.objective[*].config.text_gate_weight`

Normative behavior:
- `text_gate_weight > 0` MUST introduce a non-zero `text_gate` contribution when coord-vocab mass appears at supervised text positions under `context=rollout` (FP-neutral).

#### Scenario: FP-neutral text_gate is effective
- **WHEN** rollout-context supervision includes both FN spans and FP spans
- **AND** `text_gate_weight > 0`
- **THEN** FP spans do not contribute to the emitted `text_gate` objective atom `loss/B_coord/text_gate`
- **AND** FN text spans contribute to the `text_gate` sub-term and increase `loss/B_coord/text_gate` when they exhibit coord-vocab mass.

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
- `rollout_backend` MUST default to `"hf"`.
- `eval_rollout_backend` MUST be `"vllm"` for this stack (eval-step backend is fixed).

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
- the effective weight sync mode (full merged-weight sync; adapter-only sync is unsupported in this stack)
- the per-batch rollout seed used for `RequestConfig.seed`

#### Scenario: Server mode logs reproducibility metadata
- **GIVEN** server mode is enabled
- **WHEN** the trainer performs a fresh-rollout step (E-step)
- **THEN** it logs the server endpoints, sync mode, and rollout seed for that step.

### Requirement: Rollout generation supports microbatched decoding within each rank
When rollout-matching training is enabled, the trainer SHALL support generating rollouts for multiple samples in a single backend decode call (HF `generate()` or vLLM `/infer/`), controlled by a normalized rollout configuration knob.

Rollout config keys and nested structures MUST be validated through schema-derived strict contracts before runtime rollout execution.

Normalized rollout knobs (normative):
- `channel_b_decode_batch_size` (int) in the trainer’s injected rollout config contract for training-time Channel-B rollout decode.
- `eval_decode_batch_size` (int) in the trainer’s injected rollout config contract for eval-time rollout decode.

Config source semantics (normative):
- For canonical grouped authoring, rollout backend selection MUST come from `rollout_matching.rollout_backend`.
- For canonical grouped authoring, `channel_b_decode_batch_size` MUST come from `rollout_matching.channel_b_decode_batch_size`.
- For canonical grouped authoring, `eval_decode_batch_size` MUST come from `rollout_matching.eval_decode_batch_size`.
- For Stage-2 AB rollout knobs that previously existed under `custom.extra.rollout_matching.*`, canonical migration MUST target top-level `rollout_matching.*` keys defined by the strict schema.
- Legacy Stage-2 alias keys under `custom.extra.rollout_matching.*` are unsupported and MUST fail fast with migration guidance.
- The trainer/runtime contract MUST expose resolved decode batch-size values for both rollout contexts (`channel_b_decode_batch_size`, `eval_decode_batch_size`).

Schema-derived strictness (normative):
- Rollout config key acceptance MUST be derived from typed schema definitions and enforced at config-load time.
- Unknown rollout keys (top-level or nested) MUST fail fast with dotted-path error messages.
- Unknown-key dotted paths MUST include list indices when present (e.g., `rollout_matching.vllm.server.servers[0].unknown_flag`).
- Runtime rollout validators MAY enforce execution-dependent constraints (runtime mode compatibility, numeric bounds) but MUST NOT be the long-term owner of static schema key acceptance.
- Rollout server schema supports only `rollout_matching.vllm.server.servers[]`; legacy paired-list form (`vllm.server.base_url` + `vllm.server.group_port`) is removed and MUST fail fast with migration guidance.
- Stage-2 launcher preflight MAY expose a projected `server_base_urls` array for launch wiring, but that projection MUST be derived from canonical `servers[]` entries and MUST NOT replace schema requirements for `base_url` + `group_port`.

Semantics (normative):
- `channel_b_decode_batch_size` denotes the maximum number of sequences decoded per rollout GPU in one training-time Channel-B generation call.
- `eval_decode_batch_size` denotes the maximum number of sequences decoded per rollout GPU in one eval-time generation call.
- The trainer MUST enforce these bounds for both HF and vLLM backends where applicable.
- These keys are the single source of truth for rollout decode/evaluation microbatching in rollout-aware trainer variants.
- `training.per_device_eval_batch_size` and similar per-device eval knobs MUST NOT independently control rollout decode/evaluation batch behavior.

Defaulting (normative):
- `rollout_matching.channel_b_decode_batch_size` MUST be provided explicitly.
- `rollout_matching.eval_decode_batch_size` MUST be provided explicitly.

#### Scenario: Canonical Stage-2 key controls decode microbatching
- **WHEN** a Stage-2 AB config sets `rollout_matching.channel_b_decode_batch_size: M` and `rollout_matching.eval_decode_batch_size: N` where `M > 1` and `N > 1`
- **THEN** training-time rollout generation uses `M` as the resolved decode batch size
- **AND** eval-time rollout generation uses `N` as the resolved decode batch size.

#### Scenario: Channel-B microbatching increases decode parallelism without changing outputs format
- **WHEN** rollout-matching training runs with resolved `channel_b_decode_batch_size=M` where `M > 1`
- **THEN** the trainer performs batched Channel-B decode calls for up to `M` samples per rollout GPU
- **AND** it returns per-sample `response_token_ids` suitable for strict token-aligned parsing.

#### Scenario: Eval per-device knobs do not override rollout decode batching
- **WHEN** rollout-matching training runs with `rollout_matching.eval_decode_batch_size=M` and `training.per_device_eval_batch_size=N` where `M != N`
- **THEN** eval rollout decode microbatching follows `M`
- **AND** `training.per_device_eval_batch_size` does not independently change rollout decode/evaluation behavior.

#### Scenario: Legacy decode key path fails fast
- **WHEN** a Stage-2 config sets `custom.extra.rollout_matching.channel_b_decode_batch_size`
- **THEN** config loading fails fast with guidance to migrate to canonical top-level `rollout_matching.*` decode keys.

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
The system MUST fail fast if a config provides any legacy rollout batch-size knob under `rollout_matching`, with actionable guidance to migrate to canonical decode batching keys:
- `rollout_matching.channel_b_decode_batch_size`
- `rollout_matching.eval_decode_batch_size`

Legacy knobs (normative):
- `rollout_matching.rollout_generate_batch_size`
- `rollout_matching.rollout_infer_batch_size`

#### Scenario: rollout_generate_batch_size fails fast when present
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `rollout_matching.rollout_generate_batch_size`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to remove it and use canonical decode keys.

#### Scenario: rollout_infer_batch_size fails fast when present
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `rollout_matching.rollout_infer_batch_size`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to remove it and use canonical decode keys.

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
When rollout-matching training uses vLLM server mode (`rollout_matching.rollout_backend=vllm` and `rollout_matching.vllm.mode=server`), the trainer MUST derive per-rank request chunk sizing from the rollout server world size and learner DDP world size to preserve the per-rollout-GPU cap defined by `channel_b_decode_batch_size`.

Semantics (normative):
- The trainer MUST query each configured server’s `${base_url}/get_world_size/` endpoint and cache one `world_size` per server entry.
- Let `server_world_sizes = [s_0, s_1, ...]` be those cached values, and define:
  - `S = sum(server_world_sizes)` (total rollout inference device count across servers; DP replicas)
  - `W = learner_world_size` (training DDP world size)
- **Feasibility**: If `channel_b_decode_batch_size * S < W`, the trainer MUST fail fast with actionable guidance (the cap cannot be preserved if every learner rank must issue at least one request concurrently).
- Otherwise, the trainer MUST derive a per-learner-rank chunk size:
  - `chunk = floor(channel_b_decode_batch_size * S / W)`
- The trainer MUST distribute requests across servers in a capacity-aware deterministic way (proportional to `server_world_sizes`) so that rollout GPUs are not overloaded when servers are heterogeneous.

#### Scenario: vLLM server mode fails fast when cap is infeasible
- **GIVEN** rollout-matching training is enabled
- **AND** `rollout_backend=vllm` and `vllm.mode=server`
- **AND** `channel_b_decode_batch_size=1`, `S=2`, and `W=4`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to increase rollout server world size, reduce learner world size, or increase `channel_b_decode_batch_size`.

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
- `rollout_matching.offload.offload_model` MUST accept a boolean.
- `rollout_matching.offload.offload_optimizer` MUST accept a boolean.

Defaulting (normative):
- When `rollout_matching.offload.enabled: false` (default), missing `offload_model/offload_optimizer` MUST be treated as `false`.
- When `rollout_matching.offload.enabled: true`, missing `offload_model/offload_optimizer` MUST be treated as `true` (minimize peak VRAM by default).

Semantics when enabled:
- Offloading MUST occur only during rollout generation (no-grad inference).
- The trainer MUST restore model/optimizer state before teacher-forced forward/backprop.
- When rollout backend is not vLLM colocate (e.g., vLLM server mode or HF rollouts), offload settings MUST be ignored (no-op).
- When vLLM is lazily initialized, the offload context MUST cover all vLLM-side allocations required for rollout
  generation, including vLLM engine initialization and LoRA adapter loading/synchronization, not only the infer call.
- For evaluation rollouts (`evaluate()`), offload MUST occur once per evaluation window (not once per eval batch) when eval uses vLLM colocate.
- Offloading MUST NOT be enabled under runtimes that partition or alias optimizer/model state (e.g., DeepSpeed/ZeRO). If offload is requested under such a runtime, the run MUST fail fast with actionable guidance (disable offload, switch to HF rollouts, or disable DeepSpeed/ZeRO).
- If offload is requested but cannot be applied safely under the current setup, the trainer MUST fail fast with an
  actionable error message that suggests at least one mitigation (e.g., disable offload, switch rollout backend to HF,
  or adjust DeepSpeed/ZeRO settings).

#### Scenario: Offload context does not break the training step
- **GIVEN** rollout-matching training is enabled with vLLM colocate rollouts
- **AND** offload is enabled for optimizer and/or model
- **WHEN** one training step executes
- **THEN** rollout inference completes without allocating training activations on GPU
- **AND** teacher-forced forward/backprop still executes successfully after offload restoration.

### Requirement: Colocate vLLM evaluation lifecycle is DDP-safe; sleep mode is optional (advanced)
When evaluation rollouts use vLLM in colocate mode, the system MUST preserve training correctness and MUST be DDP-safe. vLLM sleep mode is an optional/advanced optimization and MUST be disabled by default in standard colocate mode due to observed teardown incompatibilities in our environment.

Normative behavior:
- When the effective evaluation rollout backend resolves to `vllm` and `rollout_matching.vllm.mode: colocate`:
  - The system MUST NOT attempt to shutdown vLLM in-process as part of the evaluation lifecycle (DDP safety).
  - Default behavior MUST NOT require vLLM sleep mode:
    - The system MUST NOT force vLLM sleep mode enablement at engine construction time (e.g., do not unconditionally set `enable_sleep_mode=true`).
    - Absence of vLLM sleep/wake APIs MUST NOT fail the run.
  - If vLLM sleep mode is explicitly enabled for the run (advanced / operator-controlled):
    - Before issuing any evaluation rollouts, the system MUST ensure the vLLM engine is awake (call `LLMEngine.wake_up()` or a version-equivalent wake method when available and when the engine was previously slept).
    - After `evaluate()` completes, the system SHOULD call vLLM sleep at level `2` (`LLMEngine.sleep(level=2)` or a version-equivalent sleep method) to release GPU allocations between eval windows.
    - If required vLLM APIs are missing or unsupported, the run MUST fail fast with actionable guidance *before training begins*.

#### Scenario: Optional colocated vLLM sleep-after-eval lifecycle
- **GIVEN** evaluation rollouts use vLLM in colocate mode
- **AND** vLLM sleep mode is enabled for the run (advanced)
- **WHEN** `evaluate()` completes
- **THEN** the vLLM engine is transitioned to a low-GPU-memory state (recommended: sleep level `2`)
- **AND** the next `evaluate()` call wakes the vLLM engine before issuing rollouts.

### Requirement: Eval-only vLLM rollouts are robust to sample-local failures, but fail fast on engine-level failures
Evaluation is a measurement stage. The system MUST be robust to rare sample-local decode failures under vLLM, but MUST fail fast on vLLM engine-level failures (which indicate a misconfiguration or environment/runtime problem).

Normative behavior:
- When the effective evaluation rollout backend resolves to `vllm`:
  - If an individual sample fails to decode due to a sample-local runtime error, the evaluator MUST skip that sample and continue evaluation.
  - If evaluation rollouts cannot proceed due to an engine-level failure (engine init failure, missing required lifecycle APIs for the configured mode (e.g., sleep/wake when sleep mode is enabled), eval-time OOM, or other fatal runtime error), evaluation MUST fail fast with actionable guidance (no silent fallback to HF for that eval window).

#### Scenario: Per-sample decode errors are skipped but engine failures are fatal
- **GIVEN** evaluation backend resolves to `vllm`
- **WHEN** one sample decode fails but the vLLM engine remains healthy
- **THEN** evaluation skips that sample, increments `eval/runtime/vllm_decode_error_count`, and continues
- **AND WHEN** a later engine-level vLLM failure occurs
- **THEN** evaluation fails fast with an actionable error and does not fall back to HF.

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
- **GIVEN** `custom.object_field_order: desc_first`
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

### Requirement: Repeat-terminate rollout knobs are unsupported
The rollout-matching config contract MUST treat repeat-terminate keys as unsupported.

Normative behavior:
- `rollout_matching.repeat_terminate` MUST fail fast as an unknown/unsupported key.
- Legacy placement under `custom.extra.rollout_matching.*` (including repeat-terminate subtrees) MUST fail fast with migration guidance to canonical top-level rollout keys.

#### Scenario: Top-level repeat-terminate key fails fast
- **WHEN** a rollout-matching config includes `rollout_matching.repeat_terminate`
- **THEN** strict config parsing fails fast with unknown-key diagnostics.

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

### Requirement: Rollout-aligned Stage-2 supports a config-declared objective and diagnostics pipeline
When `custom.trainer_variant: stage2_rollout_aligned`, the system SHALL support a YAML-declared module pipeline for
rollout-aligned teacher-forcing objective and diagnostics.

Normative configuration shape:
- The pipeline MUST be declared under the rollout-matching namespace:
  - `rollout_matching.pipeline.objective`: ordered list of objective modules
  - `rollout_matching.pipeline.diagnostics`: ordered list of diagnostics modules
- Each module entry MUST follow the `teacher-forcing-objective-pipeline` capability contract.

Normative behavior:
- If `rollout_matching.pipeline` is provided, the trainer MUST interpret it according to the
  `teacher-forcing-objective-pipeline` capability.
- If `rollout_matching.pipeline` is absent, the trainer MUST construct and use a default pipeline that is coherent
  with the unified teacher-forcing objective semantics.
  - The default objective MUST include bbox geometry regression (`bbox_geo`) so the Stage-2 objective described in
    `docs/training/STAGE2_RUNBOOK.md` applies consistently across `stage2_rollout_aligned` and `stage2_two_channel`.
    Disabling geometry MUST be expressed explicitly via an authored pipeline (e.g., `bbox_geo.enabled=false` or
    `bbox_geo` weights set to `0`).
- Pipeline parsing and module resolution MUST be strict and MUST fail fast on unknown module names.
- The resolved module list and a stable pipeline checksum MUST be logged at trainer initialization.

#### Default Pipeline Manifest (when `rollout_matching.pipeline` is omitted)

To eliminate ambiguity, the “pipeline omitted” behavior is defined as resolving to the following manifest.

Normative default objective modules (ordered):
1) `token_ce`
2) `bbox_geo`
3) `coord_reg`

Normative default diagnostics modules (ordered):
1) `coord_diag`

Normative default module weights (all objective modules):
- `weight = 1.0`

Normative default module configs (effective values):
- `token_ce` (rollout context):
  - `rollout_fn_desc_weight = 1.0`
  - `rollout_matched_prefix_struct_weight = 1.0`
- `bbox_geo`:
  - `smoothl1_weight = 1.0`
  - `ciou_weight = 1.0`
  - decode mode is controlled by `rollout_matching.coord_decode_mode` (default `exp`)
- `coord_reg`:
  - is configured explicitly through `rollout_matching.pipeline`
  - MUST NOT source values from removed legacy aux-loss keys
  - uses only authored pipeline config values plus schema defaults for omitted module-local fields

Informative YAML expression (conceptual; values shown are symbolic):
```yaml
rollout_matching:
  # pipeline omitted => resolves to the following manifest:
  # pipeline:
  #   objective:
  #     - {name: token_ce, weight: 1.0}
  #     - {name: bbox_geo, weight: 1.0}
  #     - {name: coord_reg, weight: 1.0}
  #   diagnostics:
  #     - {name: coord_diag}
```

#### Scenario: Pipeline omitted uses latest defaults (coherent)
- **WHEN** a rollout-matching SFT config does not define `rollout_matching.pipeline`
- **THEN** training starts successfully
- **AND** the rollout-matching objective matches the Default Pipeline Manifest
- **AND** canonical `loss/<component>` keys are emitted (per `teacher-forcing-unified-loss-registry`)

#### Scenario: Pipeline is applied when provided
- **WHEN** a rollout-matching SFT config defines `rollout_matching.pipeline` with at least one objective module
- **THEN** the trainer uses that declared module pipeline
- **AND** the resolved module list and checksum are logged.

#### Scenario: Pipeline mode rejects duplicated flat objective knobs
- **WHEN** a rollout-matching SFT config defines `rollout_matching.pipeline`
- **AND** the config also defines any objective-affecting flat knobs used by the default manifest (e.g.,
  `custom.coord_soft_ce_w1.*`)
- **THEN** config validation fails fast with guidance to move those values into the declared module configs.

#### Scenario: Unknown module name fails fast
- **WHEN** a rollout-matching SFT config defines `rollout_matching.pipeline` referencing an unknown module `name`
- **THEN** training initialization fails fast with actionable diagnostics.

### Requirement: Rollout-aligned Stage-2 module names are stable and share a registry with two-channel Stage-2
Rollout-aligned Stage-2 SHALL provide a strict module registry for its pipeline modules. Module names SHALL be stable
so YAML-declared experiments remain auditable, and the module registry SHOULD be shared with the two-channel Stage-2
engine where possible to avoid duplicated implementations.

Normative minimum module names (initial set; may be extended):
- Objective modules:
  - `token_ce` (masked/weighted CE per unified loss registry; includes EOS-enforced closure supervision)
  - `coord_reg` (coord distribution regularizers; default includes softCE + W1 + gate behavior)
  - `bbox_geo` (bbox SmoothL1 + CIoU on decoded boxes under rollout context)
- Diagnostics modules:
  - `coord_diag` (coord distribution diagnostics; best-effort)

Normative behavior:
- Unknown module names MUST fail fast before training starts.
- Error messages MUST list the unknown module name and available rollout-matching module names.

#### Scenario: Unknown rollout module names are rejected with available options
- **WHEN** `rollout_matching.pipeline` references an objective module name not in the rollout registry
- **THEN** trainer initialization fails fast
- **AND** the error message includes the unknown name and the allowed rollout module names.

### Requirement: Rollout-aligned Stage-2 module configs are strict and typed
Rollout-aligned Stage-2 SHALL validate module `config` payloads strictly so experiments are reproducible and fail fast
on schema drift.

Normative behavior:
- Module `config` mappings MUST be validated at trainer initialization (before training starts).
- Unknown keys in a module `config` MUST fail fast with actionable diagnostics listing allowed keys.

Normative config schemas (minimum set; may be extended):
- `token_ce.config`:
  - `rollout_fn_desc_weight: float` (default: `1.0`)
  - `rollout_matched_prefix_struct_weight: float` (default: `1.0`)
- `bbox_geo.config`:
  - `smoothl1_weight: float` (default: `1.0`)
  - `ciou_weight: float` (default: `1.0`)
- `coord_reg.config`:
  - `coord_ce_weight: float` (default: `0.0`)
  - `soft_ce_weight: float` (default: `0.0`)
  - `w1_weight: float` (default: `0.0`)
  - `coord_gate_weight: float` (default: `0.0`)
  - `text_gate_weight: float` (default: `0.0`)
  - `temperature: float` (default: `1.0`)
  - `target_sigma: float` (default: `2.0`)
  - `target_truncate: int|null` (default: `null`)
- `coord_diag.config`:
  - (no required keys; implementations MAY accept a strict subset of the above for convenience, but MUST document them)

Note:
- `rollout_matching.pipeline` is required.
- No implicit rollout-aligned default manifest may be derived from removed legacy aux-loss keys.

#### Scenario: Unknown module config keys fail fast
- **WHEN** a rollout module config includes a key outside the module allowlist
- **THEN** initialization fails fast before the first training step
- **AND** diagnostics include the invalid key and allowed keys.

### Requirement: Rollout-aligned Stage-2 adheres to the unified loss registry contract
Rollout-aligned Stage-2 SHALL implement loss naming and masking semantics per the `teacher-forcing-unified-loss-registry`
capability.

Normative behavior:
- Teacher-forced forward passes constructed from rollout prefix + mandatory FN append MUST be treated as
  `context=rollout` for the purpose of token-type partitioning and masking.
- The unified registry MUST be able to represent the rollout-matching supervision surface as a registry
  instantiation (including ablation variants such as “FN desc unsupervised” or “matched-prefix struct CE disabled”).
- When the module pipeline is enabled, objective/diagnostics modules MUST emit metric keys consistent with the
  registry’s canonical component names.

#### Scenario: Packing-safe registry masks do not leak across packed segments
- **GIVEN** rollout-matching teacher-forced sequences are packed into a single forward pass
- **WHEN** registry masks are built for `context=rollout`
- **THEN** per-segment boundaries are respected
- **AND** no supervised indices/masks cross segment boundaries.

### Requirement: Trainer variant naming is clear and stable
To reduce public-facing confusion, the system SHALL support clear trainer variant naming for rollout-aligned Stage-2.

Normative behavior:
- The system MUST accept `custom.trainer_variant: stage2_rollout_aligned` as the canonical trainer variant string.
- The system MUST reject `custom.trainer_variant: rollout_matching_sft` (fail fast) with actionable guidance to use
  `stage2_rollout_aligned`.

#### Scenario: Legacy rollout-matching trainer alias is rejected
- **WHEN** configuration sets `custom.trainer_variant: rollout_matching_sft`
- **THEN** config validation fails fast
- **AND** the error recommends `custom.trainer_variant: stage2_rollout_aligned`.

### Requirement: Rollout-aligned Stage-2 rollout-context semantics are coherent with the two-channel Rollout channel
Rollout-aligned Stage-2 SHALL apply the same rollout-context masking semantics as the two-channel Rollout channel by default
(`docs/training/STAGE2_DESIGN.md`), so teacher-forcing objectives do not drift across code paths.

Normative behavior:
- Rollout-context token supervision MUST enforce:
  - matched prefix: `CE_struct=1`, `CE_desc=0`, `CE_coord=0`,
  - FP spans: fully masked (`0`) for CE and excluded from geometry/coord-dist losses,
  - FN injected: `CE_struct=1`, `CE_desc=1` by default (configurable weight), `CE_coord=0`,
  - closure/EOS: supervised (EOS-enforced).
- The system MUST expose typed configuration to ablate (at minimum):
  - FN `desc` supervision weight, and
  - matched-prefix struct CE weight,
  via YAML diffs (no trainer code edits).
- When these weights differ from defaults, the resolved effective values MUST be logged as part of pipeline identity so
  runs are auditable.

Recommended expression (module config on `token_ce`, strict + typed):
- `rollout_matching.pipeline.objective[].config.rollout_fn_desc_weight: float`
- `rollout_matching.pipeline.objective[].config.rollout_matched_prefix_struct_weight: float`

#### Scenario: FN `desc` supervision can be disabled via YAML (ablation)
- **WHEN** a rollout-matching SFT config sets `rollout_fn_desc_weight: 0`
- **THEN** FN-injected `desc` tokens do not contribute to token CE
- **AND** FP-neutral and EOS-enforced semantics remain intact.

### Requirement: Rollout-matching SFT exposes ST geometry decode mode as typed YAML config
Rollout-matching SFT SHALL expose a typed config knob to select geometry coord decode mode when geometry losses are
enabled (either directly in trainer code or via the pipeline).

Normative behavior:
- Config MUST be expressed under the typed rollout-matching namespace (`rollout_matching.*`) and MUST be strict
  (unknown keys fail fast).
- When the key is omitted, defaults MUST preserve current behavior.
- When enabled, ST decode MUST follow the ST semantics defined by `teacher-forcing-unified-loss-registry`.

Normative key name:
- `rollout_matching.coord_decode_mode: exp|st`

Normative mapping / identity:
- The resolved value MUST feed the same internal decode enum used by the two-channel Stage-2 engine
  (`stage2_ab.coord_decode_mode`).
- The resolved value MUST be included in the pipeline identity checksum so ST-vs-exp differences are auditable even when
  the module list is unchanged.

#### Scenario: Rollout decode mode contributes to pipeline identity
- **WHEN** two runs are identical except `rollout_matching.coord_decode_mode` (`exp` vs `st`)
- **THEN** both runs initialize successfully
- **AND** their resolved pipeline identity payloads/checksums differ.

### Requirement: Eval-step supports COCO mAP when enabled
Rollout-aligned Stage-2 SHALL support computing COCO-style bbox mAP during `eval_step` (config-driven) and MUST log the
results under stable metric keys so training runs are paper-ready.

Normative behavior:
- COCO/mAP evaluation MUST be controlled by the typed config surface:
  - `rollout_matching.eval_detection.enabled: bool`
- For Stage-2 trainers, `rollout_matching.eval_detection.enabled` MUST default to `true` so COCO `mAP` is a standard
  eval metric like `rollout/f1`. Disabling it MUST be explicit via YAML (`enabled: false`).
- The eval-step COCO evaluation implementation MUST reuse the same detection evaluator modules used by offline
  evaluation (see `openspec/specs/detection-evaluator`), rather than re-implementing COCO preparation/scoring logic in
  trainer code.
- When `rollout_matching.eval_detection.enabled=true`, the trainer MUST attempt to compute COCO bbox AP/mAP for the
  eval-step rollouts and MUST emit, at minimum:
  - `eval/detection/mAP` (float; COCO `bbox_AP` = AP@[.50:.95])
- The trainer MUST NOT emit additional COCO summary metric keys during eval-step (e.g., `rollout/bbox_AP50`,
  `rollout/bbox_AR100`, `rollout/segm_*`) beyond `eval/detection/mAP`. Full metric reports remain available via the offline
  evaluation pipeline.
- If COCO eval fails unexpectedly (missing dependencies, invalid records, etc.), the trainer MUST:
  - emit `eval/detection/mAP=0.0` (so the key is always present when enabled), and
  - surface the failure as a warning (not silent).

#### Scenario: Rollout-aligned eval reports mAP
- **GIVEN** `rollout_matching.eval_detection.enabled=true`
- **WHEN** `eval_step` runs
- **THEN** `eval/detection/mAP` is present in the eval metrics payload

### Requirement: Trainer-variant guardrails prevent ambiguous pipeline configuration
To avoid “config declared but ignored” ambiguity, the system SHALL enforce strict guardrails:

Normative behavior:
- If `custom.trainer_variant: stage2_two_channel`, the presence of `rollout_matching.pipeline` MUST fail fast with
  guidance to use `stage2_ab.pipeline`.
- If `custom.trainer_variant: stage2_rollout_aligned`, the presence of `stage2_ab.pipeline` MUST fail fast with guidance
  to use `rollout_matching.pipeline`.

#### Scenario: Stage-2 Two-Channel rejects rollout-matching pipeline keys
- **WHEN** `custom.trainer_variant=stage2_two_channel`
- **AND** `rollout_matching.pipeline` is present
- **THEN** config validation fails fast with guidance to use `stage2_ab.pipeline`.

#### Scenario: Rollout-matching rejects stage2_ab pipeline keys
- **WHEN** `custom.trainer_variant=stage2_rollout_aligned`
- **AND** `stage2_ab.pipeline` is present
- **THEN** config validation fails fast with guidance to use `rollout_matching.pipeline`.

### Requirement: Evaluation rollout backend is fixed to vLLM
For this stack, Stage-2 evaluation (`eval_step`) rollouts SHALL use vLLM only.

Normative behavior:
- `rollout_matching.eval_rollout_backend` MUST resolve to `vllm`.
- Non-`vllm` values (including `hf`, `null`, empty, or other values) MUST fail fast with actionable guidance.
- Evaluation backend inheritance from `rollout_matching.rollout_backend` is unsupported.
- `rollout_matching.rollout_backend` continues to control training-time rollout backend only.
- Evaluation rollouts MUST enforce vLLM length-coherence guardrails:
  - `rollout_matching.max_new_tokens < rollout_matching.vllm.max_model_len`,
  - `rollout_matching.vllm.max_model_len >= global_max_length`.

#### Scenario: Evaluation uses vLLM while training backend remains configurable
- **GIVEN** `rollout_matching.rollout_backend: hf`
- **AND** `rollout_matching.eval_rollout_backend: vllm`
- **WHEN** the trainer runs `evaluate()` at an `eval_step`
- **THEN** evaluation rollouts are generated via vLLM
- **AND** training-time rollouts continue to use the configured training backend.

#### Scenario: Non-vLLM eval backend fails fast
- **GIVEN** `rollout_matching.eval_rollout_backend: hf`
- **WHEN** config validation runs
- **THEN** validation fails fast with guidance that eval-step backend is fixed to `vllm`.

### Requirement: vLLM rollouts require full merged-weight sync (no adapter-only sync)
In this stack, vLLM rollouts are supported only via full merged-weight synchronization into the vLLM engine ("full sync"). Adapter-only synchronization (vLLM LoRA upload / `add_lora`) is unsupported and MUST be rejected.

Normative behavior:
- When the effective rollout backend is `vllm` (training or eval):
  - The system MUST perform a full merged-weight sync into vLLM before issuing rollouts.
  - The system MUST NOT use vLLM adapter-only sync (no `add_lora` / adapter-only upload path).
- If configuration requests vLLM adapter-only sync while the effective backend is vLLM (e.g., `rollout_matching.vllm.enable_lora: true`), the run MUST fail fast before starting rollouts with actionable guidance.

#### Scenario: Adapter-only sync is rejected for vLLM rollouts
- **GIVEN** a config where the effective rollout backend is `vllm`
- **AND** config requests vLLM LoRA / adapter-only sync for rollout weights
- **WHEN** rollout generation begins (training or eval)
- **THEN** the run fails fast with an error stating that vLLM rollouts require full merged-weight sync.

### Requirement: Eval-only colocate vLLM MAY release GPU memory after evaluation (optional sleep-after-eval)
When evaluation rollouts use vLLM in colocate mode, the system MUST preserve training correctness and MUST be DDP-safe. vLLM sleep mode is an optional/advanced optimization and MUST be disabled by default in standard colocate mode due to observed teardown incompatibilities in our environment.

Normative behavior:
- When `rollout_matching.vllm.mode: colocate` and the effective evaluation rollout backend resolves to `vllm`:
  - The system MUST NOT attempt to "shutdown" vLLM in-process as part of the evaluation lifecycle (DDP safety).
  - Default behavior MUST NOT require vLLM sleep mode:
    - The system MUST NOT force vLLM sleep mode enablement at engine construction time (e.g., do not unconditionally set `enable_sleep_mode=true`).
    - Absence of vLLM sleep/wake APIs MUST NOT fail the run.
  - If vLLM sleep mode is explicitly enabled for the run (advanced / operator-controlled):
    - The system MUST ensure the vLLM engine is awake before issuing any evaluation rollouts.
      - If the engine was previously slept, the system MUST call `LLMEngine.wake_up()` (or a version-equivalent wake method) before generating any evaluation rollouts.
    - The system SHOULD call vLLM sleep at the end of `evaluate()` to release GPU allocations between eval windows.
      - Recommended: `LLMEngine.sleep(level=2)` (or a version-equivalent sleep method).
    - The system MUST ensure vLLM is configured to support sleep mode so sleep/wake actually affects GPU allocations.
      - This MUST be enabled at engine construction time (e.g., `EngineArgs(enable_sleep_mode=true)` in vLLM 0.11.x).
      - Failure to enable sleep mode (or lack of required vLLM APIs) MUST fail fast with actionable guidance *before training begins*, so the run cannot silently proceed with incorrect memory expectations.

#### Scenario: Optional colocated vLLM sleep-after-eval lifecycle
- **GIVEN** evaluation rollouts use `vllm` with `vllm.mode: colocate`
- **AND** vLLM sleep mode is enabled for the run (advanced)
- **WHEN** `evaluate()` completes
- **THEN** vLLM engine resources are transitioned to a low-GPU-memory state (recommended: sleep level `2`)
- **AND** the next `evaluate()` call wakes the vLLM engine before issuing rollouts.

### Requirement: Optional HF offload is supported for vLLM evaluation rollouts under plain DDP (colocate vLLM)
To reduce peak GPU memory when evaluation uses vLLM colocate mode, the system SHALL support offloading HF training state under operator control **when training runs under plain DDP** (one full model replica per rank).

Normative behavior:
- The system MUST support `rollout_matching.offload.enabled: true` during evaluation-only colocate vLLM.
- If `rollout_matching.offload.enabled: true`, the system MUST offload the requested training state before issuing vLLM rollouts and MUST restore the requested state after evaluation completes.
- Offload defaults:
  - When `rollout_matching.offload.enabled: true` and `rollout_matching.offload.offload_model` is missing, it MUST default to `true`.
  - When `rollout_matching.offload.enabled: true` and `rollout_matching.offload.offload_optimizer` is missing, it MUST default to `true`.
- Offloading MUST NOT be enabled under runtimes that partition or alias optimizer/model state (e.g., DeepSpeed/ZeRO). In particular, if `deepspeed.enabled: true` and rollout offload is requested for vLLM colocate (train or eval), the run MUST fail fast before starting rollouts with actionable guidance.

#### Scenario: DeepSpeed is rejected for eval-time offload under colocate vLLM
- **GIVEN** `deepspeed.enabled: true`
- **AND** evaluation rollouts use `vllm` with `vllm.mode: colocate`
- **AND** rollout offload is enabled
- **WHEN** evaluation begins
- **THEN** the run fails fast with an error stating that eval-time offload is only supported under plain DDP (no DeepSpeed/ZeRO).

### Requirement: vLLM traced rollouts satisfy token-trace invariants for confidence scoring (eval-safe fallback)
When evaluation requests confidence scoring derived from token logprob traces, vLLM traced rollouts MUST produce a well-formed trace for every sample. If traces are invalid during evaluation, the system MUST degrade gracefully so training can continue.

Normative behavior:
- If eval-step scoring mode requires token traces (e.g., confidence post-op scoring), vLLM evaluation MUST request logprobs and MUST run in greedy mode (`temperature=0`).
- For each evaluated sample, the vLLM backend MUST return:
  - `token_ids` (generated completion token ids),
  - `token_logprobs` (list[number]),
  - `generated_token_text` (list[string]),
  and MUST satisfy `len(token_ids) == len(token_logprobs) == len(generated_token_text)`.
- If any sample violates these invariants during **evaluation**, the system MUST:
  - emit a warning (rate-limited) that confidence scoring fell back due to invalid token traces,
  - increment an explicit metric/counter `eval/runtime/trace_fallback_count` by 1 for each affected sample, and
  - continue evaluation (do not abort training), falling back to a safe score policy equivalent to `score_mode: constant` using `constant_score` for the affected evaluation window.

#### Scenario: Trace length mismatch falls back during evaluation
- **GIVEN** eval-step confidence scoring is enabled
- **AND** evaluation backend is `vllm`
- **WHEN** any vLLM rollout returns `len(token_ids) != len(token_logprobs)` or `len(token_ids) != len(generated_token_text)`
- **THEN** evaluation continues and uses constant-score fallback for confidence scoring, and a warning is emitted indicating a token-trace invariant violation.

### Requirement: Eval-only vLLM rollouts MUST skip per-sample decode failures; engine-level failures MUST fail fast
Evaluation rollouts are observability and model-quality measurement. This stack MUST be robust to rare sample-level decode failures, but MUST fail fast on engine-level failures (which indicate misconfiguration or an environment/runtime problem and should not be silently ignored).

Normative behavior:
- When the effective evaluation rollout backend resolves to `vllm`:
  - If an individual sample fails to decode/roll out due to a runtime error localized to that sample, the system MUST:
    - skip that sample (exclude it from downstream eval aggregation),
    - increment `eval/runtime/vllm_decode_error_count` by 1,
    - and continue evaluation.
  - If evaluation rollouts cannot proceed due to a vLLM engine-level failure (e.g., engine construction failure, missing/unsupported required lifecycle APIs for the configured mode (e.g., wake/sleep when sleep mode is enabled), OOM during eval, or a fatal runtime error), the system MUST:
    - fail fast with an actionable error message (do not hang),
    - and MUST NOT silently fall back to a different backend (e.g., HF) for that evaluation window.
- This best-effort behavior is scoped to **evaluation only**. Training-time rollouts (e.g., Stage-2 Channel-B) MAY continue to fail fast to preserve training semantics.
- This change intentionally does **not** define an "eval-window abort and continue training" metric path (e.g., no `eval/vllm_eval_aborted`). Engine-level vLLM failures are treated as fatal configuration/runtime errors and MUST fail fast.

#### Scenario: Per-sample decode errors are skipped but engine failures are fatal
- **GIVEN** evaluation backend resolves to `vllm`
- **WHEN** one sample decode fails but the vLLM engine remains healthy
- **THEN** evaluation skips that sample, increments `eval/runtime/vllm_decode_error_count`, and continues
- **AND WHEN** a later engine-level vLLM failure occurs
- **THEN** evaluation fails fast with an actionable error and does not fall back to HF.

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

### Requirement: Rollout-aligned training SHALL reuse the same decoded-box size auxiliaries through `bbox_size_aux`
The rollout-aligned trainer MUST reuse the same decoded-box size auxiliaries
through `bbox_size_aux` that Stage-2 AB uses.

When `custom.trainer_variant: stage2_rollout_aligned`, the rollout-aligned
teacher-forcing path SHALL support the same optional decoded-box size
auxiliaries through `bbox_size_aux` that Stage-2 AB uses.

Normative behavior:

- matched decoded-box size auxiliaries MUST reuse the shared `bbox_size_aux`
  module
  and shared decoded-box helper implementation,
- the rollout-aligned path MUST NOT fork a second geometry implementation just
  for `log_wh`, `log_area`, or oversize penalty,
- `bbox_size_aux` MUST consume the current bbox coord slots in the same public
  `bbox_2d: [x1, y1, x2, y2]` order used elsewhere in the stack,
- the default rollout-aligned behavior SHOULD mirror the conservative Stage-2
  default:
  - small `log_wh` weight only when explicitly enabled,
  - `log_area` and `oversize` off unless explicitly authored.

#### Scenario: Rollout-aligned matched geometry can include log-size aux
- **GIVEN** a rollout-aligned config with `bbox_size_aux.config.log_wh_weight > 0`
- **WHEN** rollout-context matched geometry loss is computed
- **THEN** the matched log-width/log-height auxiliary contributes through the
  shared `bbox_size_aux` implementation
- **AND** no trainer-specific duplicate geometry path is required.

