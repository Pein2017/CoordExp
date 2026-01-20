## 1. Implementation
- [x] 1.1 Extract post-rollout segment selection into a small pure helper (for stable unit tests):
  - [x] Input: `encoded_lens` in insertion order (index 0 is oldest), plus `packing_length`.
  - [x] Output: `selected_indices` in insertion order; MUST include oldest; total length MUST be <= cap.
  - [x] Selection MUST be deterministic with stable tie-breaking.
- [x] 1.2 Update stage-2 post-rollout packing selection to use constant-volume binpacking (ms-swift-like) with a FIFO baseline fallback:
  - [x] Compute the FIFO-greedy baseline selection (current behavior).
  - [x] Compute a binpacking candidate selection that is constrained to include the oldest segment.
  - [x] If the `binpacking` module is unavailable at runtime, fail fast with actionable guidance (install `binpacking` or disable `training.packing`); do not silently fall back.
  - [x] Choose the binpacking candidate only if it strictly improves total selected length; otherwise keep the FIFO baseline (guarantees "never worse than FIFO").
  - [x] Keep the invariant: total selected `encoded_len` MUST be <= `packing_length`.
- [x] 1.3 Oversized segment fail-fast:
  - [x] If a segment is created with `encoded_len > packing_length`, error immediately (on insertion / before buffering), not only when it becomes the oldest.
- [x] 1.4 Maintain existing semantics:
  - [x] No segment splitting across packed forwards.
  - [x] Carry-buffer semantics unchanged (carry-only mode remains; flush steps are not introduced).
  - [x] Existing YAML knobs continue to apply: `training.packing_buffer`, `training.packing_min_fill_ratio`, `training.packing_drop_last`.
- [x] 1.5 Instrumentation:
  - [x] Ensure `packing/post_rollout_fill`, `packing/post_rollout_segments`, `packing/post_rollout_buffer` remain logged.
  - [x] (Optional) Add `packing/post_rollout_selected_total_len` for easier debugging.

## 2. Tests
- [x] 2.1 Add a unit test covering selection correctness and determinism (against the pure helper):
  - [x] GIVEN a buffer of segments with known lengths
  - [x] WHEN selection runs multiple times
  - [x] THEN it returns identical indices/order and total length <= cap (and includes the oldest).
- [x] 2.2 Add a unit test demonstrating improved fill vs FIFO-greedy on a constructed example where the oldest is included:
  - Example (cap=10, lengths `[6, 3, 2, 2]`): FIFO-greedy gives `6+3=9`, but a better feasible subset exists `6+2+2=10`.
  - Assert: selected total length >= FIFO-greedy total length, and equals the cap for this case.
- [x] 2.3 Add a unit test covering oversized segment fail-fast timing:
  - [x] GIVEN `packing_length` and a segment with `encoded_len > packing_length`
  - [x] WHEN the trainer prepares the segment for post-rollout buffering / insertion
  - [x] THEN it errors immediately (before selection) with mitigation guidance.
- [x] 2.4 Add a unit test covering missing binpacking dependency fail-fast:
  - [x] GIVEN rollout-matching training is enabled with post-rollout packing
  - [x] AND `binpacking` cannot be imported
  - [x] WHEN selection is attempted
  - [x] THEN it fails fast with actionable guidance (install `binpacking` or disable `training.packing`).

## 3. Docs
- [x] 3.1 Update `docs/PACKING_MODE_GUIDE.md` or `docs/STAGE2_ROLLOUT_MATCHING_RUNBOOK.md` to note that stage-2 post-rollout packing uses binpacking selection (ms-swift-like).
  - [x] If updating `docs/STAGE2_ROLLOUT_MATCHING_RUNBOOK.md`, also add this change-id to its "Authoritative requirements live under:" list so readers can discover it.

## 4. Validation
- [x] 4.1 Run `openspec validate 2026-01-19-update-stage2-post-rollout-packing-binpacking --strict`.
- [x] 4.2 Run `conda run -n ms python -m pytest -q` (at minimum, tests covering stage-2 packing selection).
