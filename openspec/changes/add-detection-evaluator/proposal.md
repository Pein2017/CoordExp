# Change: Add Detection Evaluator

## Why
- Current pipeline reports only token-level metrics; there is no end-to-end detection AP/AR evaluator aligned with the CoordExp prompt/JSONL contract.
- We already generate prediction JSONL via `vis_tools/vis_coordexp.py`; a scoped evaluator would turn those into COCO metrics, surface data-quality issues (invalid JSON, degenerate boxes), and unblock grounding benchmarks.
- Design centers on reuse of the existing parser plus strict-but-permissive dropping of bad objects, constant-score exports, and optional training/offline evaluation flows.

## What Changes
- Add a config-driven offline evaluator that ingests single-image CoordExp JSONL GT + model predictions (coord tokens or ints), denormalizes coords, and exports COCO-style GT/pred files (bbox + segm for polygons; lines converted to bbox).
- Use exact string categories (no aliasing); bucket unknown desc into `unknown`; scores are fixed to 1.0 (greedy decoding) with deterministic tie ordering.
- Run COCOeval (bbox, segm) and emit diagnostics (invalid/degenerate drops, raw parse errors, size mismatches, unknown-desc rate). Add a TODO hook for polygon GIoU alongside COCOeval outputs.
- Provide a CLI under `scripts/evaluate_detection.py` (overwrites output dir) plus an optional training-time eval hook and a standalone checkpoint+JSONL mode; inference/eval expected on GPU, exposing temperature/repetition_penalty knobs consistent with `vis_tools/vis_coordexp.py`.

## Impact
- New capability spec: `detection-evaluator` (added under `openspec/changes/add-detection-evaluator/specs/...`).
- Touches shared parsing (factored from `vis_tools/vis_coordexp.py`), new `src/eval/` module, `scripts/`, `configs/eval/`, and docs (`docs/` usage guide). Tests add fixtures under `tests/fixtures/eval/` and a CI smoke test.
- No model architecture changes; default training loop unchanged unless the optional eval hook is configured.
