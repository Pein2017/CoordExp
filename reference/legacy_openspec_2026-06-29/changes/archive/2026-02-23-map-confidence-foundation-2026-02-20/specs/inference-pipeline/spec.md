# inference-pipeline Specification (delta: evaluation is score-aware; no fixed-score toggle)

## Purpose
Allow unified YAML pipeline runs (`scripts/run_infer.py --config <yaml>`) to enable confidence-sensitive COCO AP/mAP evaluation without any deprecated toggle that disables score honoring.

This delta updates evaluation-stage configuration behavior; all other base `inference-pipeline` requirements remain unchanged unless explicitly modified below.

## Requirements

## ADDED Requirements

### Requirement: Pipeline evaluation is score-aware and rejects fixed-score toggles
Pipeline evaluation SHALL be score-aware for COCO metrics by default (it MUST NOT provide a fixed-score mode).

The pipeline YAML MUST NOT include `eval.use_pred_score`. If this key is present, the pipeline MUST fail fast with an actionable error instructing the user to remove it.

When the pipeline runs COCO-style detection evaluation, it MUST consume the scored artifact (`gt_vs_pred_scored.jsonl`) rather than the base inference artifact (`gt_vs_pred.jsonl`).

The scored artifact path MUST be resolved from `artifacts.gt_vs_pred_scored_jsonl`.

If `artifacts.gt_vs_pred_scored_jsonl` is not configured (or the file does not exist), the pipeline MUST fail fast with an actionable error instructing the user to run the confidence post-op first.

#### Scenario: Deprecated score toggle is rejected
- **WHEN** a pipeline config includes `eval.use_pred_score`
- **THEN** the pipeline terminates with a clear error explaining that fixed-score evaluation is unsupported and the key must be removed.

### Requirement: Pipeline artifact contract for confidence workflow
For score-aware COCO workflows, the pipeline/post-op artifact contract SHALL include:
- `artifacts.pred_token_trace_jsonl`: canonical token-trace sidecar path (explicit value or deterministic default `<run_dir>/pred_token_trace.jsonl`).
- Trace records keyed by `line_idx`, with 1:1 `generated_token_text` and `token_logprobs` arrays (full generated output, no filtering).

Inference-only or f1ish-only evaluation runs MAY proceed without running confidence post-op. COCO workflows MUST produce/consume these artifacts in order: `gt_vs_pred.jsonl` + `pred_token_trace.jsonl` -> confidence post-op -> `gt_vs_pred_scored.jsonl`.

#### Scenario: Missing scored artifact fails fast
- **GIVEN** a pipeline config that requests COCO detection evaluation
- **WHEN** `gt_vs_pred_scored.jsonl` is not available
- **THEN** the pipeline terminates with a clear error instructing the user to produce `gt_vs_pred_scored.jsonl` via the confidence post-op before evaluating.

#### Scenario: f1ish-only evaluation does not require a scored artifact
- **GIVEN** a pipeline config that requests only f1ish-style (non-COCO) evaluation
- **WHEN** `gt_vs_pred_scored.jsonl` is not available
- **THEN** the pipeline continues evaluation using the base inference artifact (`gt_vs_pred.jsonl`).
