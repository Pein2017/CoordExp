# vllm-full-sync-adapter-row-materialization Specification

## Purpose

Define how CoordExp presents adapter-backed learner weights to native vLLM
full-sync without changing training-time frozen-base or adapter-only checkpoint
semantics.

Status note (2026-05-26): this capability is historical/deferred for active
unified Stage-2 rollout-correction server training. The active server rollout
contract is governed by `unify-inference-runtime` and uses official adapter
sync plus CoordExp coord-row updates rather than native full-sync
materialization.

## ADDED Requirements

### Requirement: vLLM full-sync uses a transient ordinary-weight snapshot

When native vLLM rollout full-sync is selected, the system SHALL build a
transient state dict containing only vLLM-loadable ordinary model weights that
represent the learner's effective inference policy.

Normative behavior:

- The transient sync snapshot MAY include LoRA/DoRA deltas and token-row deltas
  applied to ordinary base weight tensors.
- Temporary PEFT merge/unmerge MAY be used to build the snapshot, but sync
  preparation MUST NOT leave lasting mutation in live base-model parameters.
- The transient sync snapshot MUST NOT change checkpoint save semantics.
- Adapter-only training and saving MUST remain valid.
- Existing PEFT full-sync canonicalization owns ordinary LoRA/DoRA realization;
  this capability owns CoordExp token-row `modules_to_save` materialization
  after that canonicalization.
- Native vLLM MUST NOT receive learner-only module keys such as
  `coord_offset_adapter.*`, `modules_to_save.default.*`, `original_module`, or
  `lora_*`.

#### Scenario: Adapter-only checkpoint with vLLM full-sync

- **GIVEN** a learner with a frozen base model and trainable adapters
- **AND** Stage-2 rollout uses native vLLM full-sync
- **WHEN** the learner synchronizes weights to the vLLM server
- **THEN** the server receives ordinary weight tensors representing the
  learner's effective inference policy
- **AND** the training model still saves adapter-only checkpoints.

#### Scenario: Temporary PEFT merge is restored before training continues

- **GIVEN** sync preparation temporarily merges PEFT adapter weights to produce
  an ordinary-weight snapshot
- **WHEN** sync preparation succeeds or fails
- **THEN** the learner model returns to the adapter training state before
  subsequent training or checkpoint saving continues
- **AND** no merged base checkpoint is written by the sync path.

### Requirement: Active token-row adapters are materialized before sync

If the learner has active CoordExp token-row adapter state, the system SHALL
materialize that state into ordinary embedding/head rows before native vLLM
full-sync.

Normative behavior:

- `coord_ids` MUST select the token rows to patch.
- `embed_offset` MUST be added to `embed_tokens.weight[coord_ids]`.
- If `lm_head.weight` is present, output-row deltas MUST be added to
  `lm_head.weight[coord_ids]`.
- For tied-head adapters, `embed_offset` is the output-row delta.
- For untied-head adapters, `head_offset` is the output-row delta.
- The system MUST clone tensors before patching rows and MUST NOT update live
  model parameters in place.
- If token-row adapter state appears in the sync state dict but the active
  adapter instance cannot be discovered from the model, sync preparation MUST
  fail fast rather than filter those keys.

#### Scenario: Tied-head token-row adapter materializes into ordinary rows

- **GIVEN** an active tied-head token-row adapter with `coord_ids` and
  `embed_offset`
- **AND** the vLLM-bound state dict contains `embed_tokens.weight` and
  `lm_head.weight`
- **WHEN** the sync snapshot is built
- **THEN** the selected embedding rows contain the base rows plus
  `embed_offset`
- **AND** the selected output-head rows contain the base rows plus
  `embed_offset`
- **AND** `coord_offset_adapter.*` is not present in the vLLM-bound state dict.

#### Scenario: Tied-head adapter with missing head requires confirmed tying

- **GIVEN** an active tied-head token-row adapter
- **AND** the vLLM-bound state dict contains `embed_tokens.weight` but no
  `lm_head.weight`
- **WHEN** the model cannot confirm tied output embeddings through config or
  shared embedding storage
- **THEN** sync preparation fails fast before broadcasting weights to vLLM.

#### Scenario: Untied-head token-row adapter requires head rows

- **GIVEN** an active untied token-row adapter with `head_offset`
- **WHEN** no safe `lm_head.weight` route exists in the sync snapshot
- **THEN** sync preparation fails fast before broadcasting weights to vLLM.

### Requirement: Sync snapshot hygiene is strict

The system SHALL validate the vLLM-bound state dict before calling native vLLM
`load_weights()`.

Normative behavior:

- Forbidden learner-only key families MUST fail fast if still present after
  materialization.
- Missing required ordinary rows for active token-row adapters MUST fail fast.
- Shape and vocabulary-bound mismatches MUST fail fast.
- No silent fallback to filtering-only behavior is allowed when an active
  token-row adapter cannot be materialized.

#### Scenario: Forbidden key remains after materialization

- **GIVEN** an active `coord_offset_adapter`
- **WHEN** sync preparation leaves `coord_offset_adapter.embed_offset`,
  `modules_to_save.alt.coord_offset_adapter.embed_offset`, `original_module`,
  or a LoRA residue key in the vLLM-bound state dict
- **THEN** the run fails before vLLM `load_weights()` is called
- **AND** the error identifies the forbidden key family.

### Requirement: Native vLLM adapter-only sync without row endpoint remains out of scope

The system SHALL NOT route active CoordExp token-row adapters through native
vLLM LoRA adapter-sync unless a row-sync runtime path is available. Active
unified Stage-2 server rollout uses the ms-swift adapter-sync path plus the
patched CoordExp token-row update endpoint.

Normative behavior:

- native vLLM LoRA adapter-sync without CoordExp row-sync support MUST fail
  fast when active token-row adapter state is present;
- ms-swift server adapter sync with the patched token-row update endpoint is
  governed by `unify-inference-runtime` and is not rejected by this historical
  full-sync materialization capability;
- This change does not require vLLM server workers to host a CoordExp private
  adapter module.

#### Scenario: Native LoRA sync without row endpoint rejects active token-row adapter

- **GIVEN** a Stage-2 learner has active `coord_offset_adapter` state
- **AND** config enables native vLLM LoRA sync without the CoordExp row-update
  endpoint
- **WHEN** config/runtime validation runs
- **THEN** validation fails with guidance to use the active ms-swift server
  adapter-sync plus coord-row update path or a future native row-sync
  implementation.
