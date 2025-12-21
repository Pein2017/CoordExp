# Change: Refactor inference/visualization pipeline and centralize coord handling

## Why
- The current `vis_tools/vis_coordexp.py` couples model rollout, parsing/repair, visualization, and evaluation prep, causing duplicated coord/token logic and fragile error handling.
- We need a stageable pipeline: inference (raw+parsed JSONL with coord_mode metadata) as the source of truth, with evaluation and visualization able to run independently on that output.
- Coord tokens are always generated on a 0–999 grid; scaling to absolute pixels must be centralized to avoid double-scaling and to support bbox–poly IoU matching (bbox GT vs poly pred) reliably.
- Robust JSONL emission per sample (even on truncated/malformed generations) is required so batch jobs never die and ablation runs across checkpoints remain comparable.

## What Changes
- Add an inference runner CLI that performs generation only (no rendering), emits one JSON line per sample with raw text, optional parsed preds, `coord_mode` (norm1000/pixel), width/height, and constant scores=1.0, keeping order and single-image assumption.
- Centralize coord-token decode/encode and scaling (norm1000 ↔ pixel) in a reusable module used by inference, visualization, and evaluation; include line→bbox and bbox→segm helpers for mixed-geometry IoU.
- Decouple visualization to consume the inference JSONL (or reparse raw text via the shared parser) without loading the model; keep robust output even when some lines have errors.
- Update detection-evaluator parsing requirement to rely on the shared coord module instead of `vis_tools`, honor `coord_mode` to avoid double scaling, and keep bbox–poly matching feasible.

## Impact
- New capability specs for the decoupled inference pipeline and shared coord utilities; modified detection-evaluator parsing requirement.
- New/updated CLIs under `scripts/` and configs under `configs/infer/` or similar; docs describing the staged workflow.
- No change to model architecture; alias-free categories and constant scores remain; single-image contract preserved.
