## ADDED Requirements

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
