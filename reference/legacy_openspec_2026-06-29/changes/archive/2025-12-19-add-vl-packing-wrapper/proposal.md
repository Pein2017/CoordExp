# Change: Add rank-local packed dataset wrapper for VL training

## Why
- Current `training.packing` flag is a no-op because CoordExp builds torch datasets without ms-swift’s `PackingDataset`; we waste sequence budget and GPU memory.
- Vision-language samples (prompt + image → text) need packing that respects multimodal length accounting while avoiding per-dataset metrics and minimizing truncation bias.
- A wrapper path is preferred to avoid data rewrites; we need a spec to define behavior, constraints, and integration points before coding.

## What Changes
- Introduce a rank-local packing wrapper that groups already-encoded VL samples into packed sequences using ms-swift’s bin-packing heuristic.
- Gate packing via config; enforce `per_device_train_batch_size=1` and keep effective batch through `gradient_accumulation_steps`.
- Define safeguards: min fill ratio, drop-last for underfilled packs, handling of oversized single samples, and preservation of multimodal fields (pixel_values, grids, position_ids).
- Document fusion interaction (pack after fusion schedule) and metric behavior (aggregate loss/token_acc only).

## Impact
- Affected specs: new `packing-dataset` capability (added requirements).
- Affected code (later): dataset wrappers under `src/datasets`, `sft.py` packing gate, config schema, doc update.
