# Change: Add centralized inference engine for coord/text models

## Why
- Current inference is split across multiple scripts, lacks a unified schema, and cannot cleanly support both coord-token and pure-text checkpoints with the required absolute-coordinate outputs.
- Downstream visualization and evaluation need a consistent `pred.jsonl` (pixels, raw geometry preserved) plus deterministic, error-tolerant execution.

## What Changes
- Introduce a centralized `InferenceEngine` with a mandatory `--mode` switch, shared coordinate processing, and standardized output schema (absolute pixels, per-object geometry + desc, score=1.0).
- Refactor the inference CLI to use explicit generation flags (temperature/top_p/max_new_tokens/repetition_penalty/seed/device/limit/output) instead of external config files.
- Update evaluation/consumer paths to accept the new schema (polygons evaluated via COCO segmentation/mask IoU; lines tolerated but not scored) and drop legacy mixed-format fields.
- Add validation/error reporting (skip-with-error per sample, counters) and deterministic seeding.

## Impact
- Affects inference scripts, evaluation adapters, and docs under `docs/centralized_infer/`.
- Requires downstream tools (vis/eval) to read the unified schema; no legacy `predictions`/mixed coord outputs.
- Sets the contract for coord vs text modes to avoid double-scaling and schema drift.
