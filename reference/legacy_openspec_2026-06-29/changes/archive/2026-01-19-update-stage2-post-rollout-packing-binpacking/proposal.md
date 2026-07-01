# Change: Stage-2 Post-Rollout Packing Uses ms-swift-like Binpacking

## Why
- Stage-2 (rollout-matching SFT) already supports *post-rollout packing* inside the trainer, but the current segment-selection heuristic is FIFO-greedy and can underfill the packed sequence when multiple short segments are available.
- ms-swift’s packing wrapper uses a constant-volume binpacking heuristic that generally improves fill ratio (less padding, better throughput) while keeping the same max sequence length cap.

## What Changes
- Update stage-2 post-rollout packing **selection** to use a deterministic, ms-swift-like constant-volume binpacking heuristic (e.g., `binpacking.to_constant_volume`), while:
  - preserving the existing “segment is atomic” constraint (no splitting),
  - preserving carry-buffer semantics and existing YAML knobs,
  - keeping the CUDA-safety cap `packing_length` unchanged (derived from template/global max length; default 16000).

## Impact
- No new CLI flags. Packing remains YAML-driven via existing `training.packing_*` knobs.
- Expected behavior improvement: higher average `packing/post_rollout_fill` when multiple segments are available to pack in a single forward pass.
