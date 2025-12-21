# Change: Add CoordExp auxiliary loss for SFT detection pretraining

## Why
Stage-1 SFT needs continuous geometric supervision on coord tokens (L1 + GIoU) while keeping the Qwen3-VL SFT pipeline intact and packing-compatible. This improves localization quality without adding a detection head, and enables controlled ablations via config.

## What Changes
- Add a long-term config block for coord auxiliary loss (weights, top-k expectation, temperature, CE coord/non-coord weighting).
- Add top-k CoordExp expectation decoding that uses coord-token id ordering (0..999) instead of raw vocab ids.
- Attach coord spans and CE loss_scale in the collator (packed + non-packed) to support coord-only weighting and per-object spans.
- Compute and log auxiliary losses (L1 per coord token; GIoU for bbox_2d and poly→bbox; line gets L1 only) in train and eval.
- Keep behavior off by default to avoid changing existing runs.

## Impact
- Affected specs: coord-token-mode, new coord-aux-loss
- Affected code: src/config/schema.py, src/coord_tokens/loss.py, src/data_collators/dataset_metrics.py, src/sft.py, new helper for aux loss
- Backward compatibility: unchanged unless coord_loss.enabled is true
