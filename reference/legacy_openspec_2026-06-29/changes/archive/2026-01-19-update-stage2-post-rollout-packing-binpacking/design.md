## Context
Stage-2 rollout-matching SFT maintains a **rank-local post-rollout segment buffer** and, for each teacher-forced
forward pass, packs a subset of buffered segments into a single padding-free sequence.

The selection logic is currently localized in `src/trainers/rollout_matching_sft.py` (method `_pop_post_rollout_pack`).
Today it is FIFO-greedy (scan in insertion order, add segments while they fit). This preserves a key fairness property
("oldest segment always included") but can underfill the packed sequence when several shorter segments could fill the
remaining capacity more efficiently.

This change is **only** about the selection heuristic. Packing semantics, carry-buffer behavior, and atomic segments
remain unchanged.

## Goals / Non-Goals
Goals:
- Improve packed fill ratio (less unused capacity) when multiple buffered segments are available.
- Preserve fairness: always include the oldest buffered segment (no starvation).
- Deterministic selection with stable tie-breaking.
- Never produce a worse fill than the current FIFO-greedy heuristic.

Non-goals:
- No segment splitting across packed forwards.
- No new YAML knobs or CLI flags.
- No changes to how segments are produced (rollout/parse/match) or how packed loss masking works.

## Locked Decisions
- `packing_length` remains a hard cap derived from template/global max length.
- Oversized segment detection is **early**:
  - When a segment is produced and is about to be inserted into the post-rollout buffer, if
    `encoded_len > packing_length`, the trainer raises an error immediately with mitigations.
  - Selection re-checks this invariant defensively (assert / fail-fast) to avoid silent corruption.
- Selection is framed as: include oldest, then choose an additional subset from the remainder under the residual cap.

## Algorithm: Binpacking Candidate + FIFO Baseline Fallback
Inputs (pure selection problem):
- `encoded_lens`: list of segment lengths in insertion order (index 0 is oldest)
- `packing_length`: hard cap for the packed forward

Outputs:
- `selected_indices`: indices into `encoded_lens`, ordered by insertion order (ascending)

Steps:
1) Include the oldest segment (index `0`). If `encoded_lens[0] > packing_length`, fail fast.
2) Let `cap_rem = packing_length - encoded_lens[0]`.
3) Compute two candidates over indices `[1..n-1]`:
   - **Baseline (FIFO-greedy)**: scan in insertion order, add each segment if it fits.
   - **Binpacking candidate (ms-swift-like)**: use a constant-volume binpacking heuristic over the remaining segments
     under `cap_rem` to pick a higher-fill feasible subset. This candidate MUST NOT drop the oldest segment.
4) Compare total lengths (including the oldest segment):
   - If binpacking total > baseline total: select binpacking result.
   - Else (including ties): select baseline result (backward-compatible).
5) Ordering is always insertion order: `selected_indices` are returned sorted ascending. This keeps deterministic
   behavior even if the binpacking heuristic internally permutes items.

Determinism / tie-break rules:
- Candidate selection MUST be deterministic given identical `encoded_lens` and insertion order.
- If the binpacking heuristic can yield multiple equal-score solutions, the implementation MUST apply an explicit
  secondary tie-break, so the selected subset is unique:
  - maximize total selected length (primary score),
  - then minimize number of selected segments (secondary score),
  - then choose the lexicographically-smallest index set (tertiary score; indices are insertion-order indices).
- Always choose the FIFO baseline on equal total length between baseline and binpacking candidate.

## Dependency / Runtime Behavior
- The selection code MAY reuse the third-party `binpacking` module (as used by `src/datasets/wrappers/packed_caption.py`).
- If `binpacking` is not available at runtime, stage-2 post-rollout packing selection MUST hard-error with actionable
  guidance (e.g., install `binpacking` or disable `training.packing`) rather than silently falling back, to preserve
  experiment reproducibility.

## Pseudocode
```python
def select_post_rollout_segments(encoded_lens: list[int], packing_length: int) -> list[int]:
    if not encoded_lens:
        return []

    oldest_len = encoded_lens[0]
    if oldest_len > packing_length:
        raise ValueError("Single segment exceeds packing_length; mitigate via length knobs / disable packing.")

    # 1) FIFO-greedy baseline (current behavior).
    baseline = [0]
    used = oldest_len
    for i in range(1, len(encoded_lens)):
        seg_len = int(encoded_lens[i])
        if seg_len <= 0:
            continue
        if used + seg_len <= packing_length:
            baseline.append(i)
            used += seg_len

    # 2) Binpacking candidate over the remainder under the residual cap (oldest is pinned).
    cap_rem = packing_length - oldest_len
    if cap_rem <= 0:
        return baseline

    # NOTE: Implementation may reuse the same `binpacking` dependency used in dataset packing.
    try:
        import binpacking
    except ImportError as exc:
        raise ImportError(
            "binpacking is required for stage-2 post-rollout packing; "
            "install `binpacking` or disable `training.packing`."
        ) from exc

    # Only consider segments that can fit in the residual cap.
    items = [
        (i, int(encoded_lens[i]))
        for i in range(1, len(encoded_lens))
        if 0 < int(encoded_lens[i]) <= cap_rem
    ]
    bins = binpacking.to_constant_volume(items, cap_rem, weight_pos=1) if items else []

    # Choose the best bin by an explicit deterministic tie-break:
    #   maximize total length -> minimize #segments -> lexicographically-smallest index set.
    best_rest: list[int] = []
    best_key: tuple[int, int, list[int]] | None = None
    for b in bins:
        rest = sorted(int(idx) for idx, _ in b)
        total = int(sum(int(encoded_lens[i]) for i in rest))
        key = (-total, len(rest), rest)
        if best_key is None or key < best_key:
            best_key = key
            best_rest = rest

    candidate = [0] + best_rest
    candidate.sort()

    # Baseline-fallback rule: only switch if binpacking strictly improves total length.
    if sum(int(encoded_lens[i]) for i in candidate) > sum(int(encoded_lens[i]) for i in baseline):
        return candidate
    return baseline
```

Notes:
- `binpacking.to_constant_volume(...)` is used only as a heuristic to propose feasible bins under `cap_rem`.
  Determinism comes from stable input ordering, explicit tie-breaking across returned bins, and the FIFO baseline
  fallback rule (not from assuming the heuristic will always return a unique solution).
- The baseline-fallback rule is what guarantees “never worse than FIFO-greedy”.
