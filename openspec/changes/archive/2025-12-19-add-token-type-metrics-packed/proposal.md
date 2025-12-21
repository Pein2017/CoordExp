# Change: Add packing-compatible token-type metrics

## Why
- CoordExp needs fine-grained desc/coord/format token telemetry to study grounding quality, but current repo lacks the Qwen3-VL instrumentation.
- Training often uses packed sequences; token-type metrics must remain correct under packing, not just padded batches.
- We only log aggregate metrics (per-dataset telemetry is disabled in CoordExp) and must avoid NaNs when no supervised tokens exist.

## What Changes
- Add config surface `token_type_metrics` (enable flag, include/exclude datasets; default targets lvis only) and wire through sft pipeline.
- Port token-type collator + trainer mixin from Qwen3-VL, adapted to CoordExp aggregate-only logging and NaN-safe handling.
- Extend logic to work when packing is enabled (train/eval): compute per-sample token types pre-pack, concatenate to align with packed labels, and fall back gracefully when misalignment occurs.
- Document behavior, defaults, and packing limitations; add unit/contract tests for padded and packed paths.

## Impact
- New capability: token-type metrics (aggregate only) across train/eval.
- Affected areas: config schema/loader, data collator, trainer mixin, packing-aware batching, docs/tests.
- No model or checkpoint format changes; metric logging only.
