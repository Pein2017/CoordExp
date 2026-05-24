## ADDED Requirements

### Requirement: Stage-2 vLLM full-sync preserves adapter-backed learner semantics

The system MUST ensure that when Stage-2 AB uses native vLLM for rollout
generation with full-sync, the rollout server receives weights that are
semantically equivalent to the learner's current inference policy for supported
adapter state.

Normative behavior:

- Stage-2 vLLM full-sync MUST materialize supported adapter deltas into the
  vLLM-bound ordinary-weight snapshot.
- The training model MUST remain frozen-base plus adapters when configured that
  way after sync preparation completes.
- Checkpoint saving MUST remain adapter-only when the run is configured for
  adapter training.
- Stage-2 online rollout target construction MUST NOT use a vLLM policy that
  silently omits active coord/schema token-row adapters.

#### Scenario: Online residual-trie rollout sees materialized coord rows

- **GIVEN** a Stage-2 online residual-trie run starts from a compact-full
  checkpoint with active coord/schema token-row offsets
- **AND** rollout backend is native vLLM full-sync
- **WHEN** Channel-B rollout generation syncs the learner to vLLM
- **THEN** the vLLM server receives patched ordinary token rows
- **AND** Channel-B target construction does not train against rollout samples
  from a policy missing those row offsets.

### Requirement: Stage-2 vLLM full-sync remains YAML-first and fail-fast

Stage-2 vLLM full-sync behavior SHALL be configured through existing YAML
surfaces and strict validation, not through ad-hoc CLI flags.

Normative behavior:

- The default Stage-2 native vLLM path for active CoordExp token-row adapters
  MUST be `rollout_matching.vllm.sync.mode=full` with adapter-row
  materialization.
- `rollout_matching.vllm.enable_lora=true` MUST remain rejected when active
  CoordExp token-row adapter state exists unless a future row-sync contract is
  implemented.
- Unknown or unsupported vLLM sync knobs MUST fail fast.

#### Scenario: Unsupported adapter-only vLLM sync fails early

- **GIVEN** Stage-2 config enables vLLM LoRA adapter-sync
- **AND** the learner has active `coord_offset_adapter` state
- **WHEN** config/runtime validation runs
- **THEN** the run fails before launch with guidance to use full-sync
  materialization.

#### Scenario: Legacy or top-level vLLM sync alias is rejected

- **GIVEN** a Stage-2 YAML config sets `vllm.enable_lora`,
  `vllm.sync.mode`, `enable_lora`, or `sync.mode` outside the canonical
  `rollout_matching.vllm.*` subtree
- **WHEN** strict config validation runs
- **THEN** validation fails with diagnostics that point to the canonical
  `rollout_matching.vllm.enable_lora` and
  `rollout_matching.vllm.sync.mode` paths.
