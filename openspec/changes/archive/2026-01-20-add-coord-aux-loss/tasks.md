## 1. Implementation
- [x] 1.1 Add `CoordLossConfig` to `src/config/schema.py` (enable flag, weights, top_k, temperature, coord/non-coord CE weights) and plumb into `CustomConfig`.
- [x] 1.2 Extend coord-token loss helpers to support top-k expectation using ordered coord token ids (fractional top_k supported).
- [x] 1.3 Attach `coord_spans` and `loss_scale` in `build_dataset_metrics_collator` for packed + non-packed batches.
- [x] 1.4 Implement auxiliary loss computation (L1 + GIoU) with poly→bbox and line L1 only; add train/eval logging via `custom_metrics`.
- [x] 1.5 Wire the aux-loss compute hook into `src/sft.py` when `custom.coord_loss.enabled` is true.
- [x] 1.6 Update docs/config example for the new coord loss block and logging keys.
