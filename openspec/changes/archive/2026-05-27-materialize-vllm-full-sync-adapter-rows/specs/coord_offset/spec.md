## ADDED Requirements

### Requirement: Coord-offset adapters materialize into ordinary rows for native vLLM full-sync

The system MUST preserve coord-offset inference semantics during native vLLM
full-sync by applying active adapter offsets to ordinary model row weights in
the transient sync snapshot.

Normative behavior:

- Materialization MUST use the active adapter instance after PEFT wrapping.
- `coord_ids` MUST be used as row indices for both embedding and output-head
  materialization.
- `embed_offset` MUST patch embedding rows.
- `embed_offset` MUST patch output-head rows when the adapter is tied-head and
  `lm_head.weight` is present.
- If `lm_head.weight` is absent for a tied-head adapter, materialization MUST
  only proceed when the model confirms tied output embeddings through config or
  shared embedding storage.
- `head_offset` MUST patch output-head rows when the adapter is untied-head.
- The adapter's PEFT `modules_to_save` state MUST remain the saved checkpoint
  representation.

#### Scenario: Saved coord-offset remains adapter state

- **GIVEN** a run trains with `coord_offset_adapter` active
- **WHEN** a vLLM full-sync snapshot is materialized
- **THEN** the snapshot contains ordinary patched rows for vLLM
- **AND** the checkpoint still saves `coord_offset_adapter` through PEFT
  `modules_to_save`.

### Requirement: Coord-offset state is not silently discarded for compact coord-token rollouts

The system SHALL NOT silently skip active coord-offset state when building
rollout weights for compact coord-token generation.

Normative behavior:

- If compact coord-token generation depends on active coord-offset state, failure
  to materialize that state MUST fail fast.
- If coord-offset keys are present in the sync state dict but no active adapter
  instance can be discovered, the sync path MUST fail fast.
- A filtering-only path is invalid for active coord-offset state.
- Diagnostics SHOULD report whether coord-offset materialization was applied
  and how many rows were patched.

#### Scenario: Filtering-only would lose compact coord rows

- **GIVEN** a compact-full checkpoint declares `coord_offset_adapter` in
  `modules_to_save`
- **WHEN** vLLM full-sync preparation cannot patch the selected coord rows
- **THEN** the run fails fast
- **AND** it does not continue with those adapter tensors merely filtered out.
