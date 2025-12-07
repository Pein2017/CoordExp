# Change: Add Coord Token Modules

## Why
- We need a dedicated path for experiments where JSONL already stores coordinates as `<|coord_k|>` tokens (no runtime numeric-to-token conversion).
- Current pipeline assumes numeric pixel coords; it will reject tokenized geometries and re-normalize them, causing double handling and blocking the planned coord-token ablations.
- A scoped spec lets us add the minimal modules and flags without breaking the existing numeric workflow.

## What Changes
- Introduce a coord-token–aware data path (validation + loader) that accepts `<|coord_k|>` geometry tokens and keeps pixel-width/height context for losses.
- Provide helper utilities for coord token ↔ numeric mapping and expectation decoding to share between CE-only and CoordExp/GIoU losses.
- Add a template/dataset option to bypass bbox normalization when coord tokens are pre-quantized, while preserving the current numeric path as default.
- Add a small offline conversion utility to rewrite numeric JSONL coords to coord tokens so training needn’t convert online.

## Impact
- Affects dataset loading, template selection, and loss helpers in `src/` (no model architecture changes).
- Adds a new optional configuration mode; default numeric workflow remains unchanged.
