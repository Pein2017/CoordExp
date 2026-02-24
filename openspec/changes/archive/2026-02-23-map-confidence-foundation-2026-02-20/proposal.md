## Why

COCO-style AP/mAP depends on sorting predictions by a meaningful per-object confidence score. Today, CoordExp’s unified inference artifact (`gt_vs_pred.jsonl`) and detection evaluator both force prediction scores to `1.0`, making COCO ranking effectively arbitrary and preventing confidence-sensitive mAP evaluation.

We need a reproducible, offline (CPU-only) confidence pipeline that emits one sortable score per predicted object, and evaluation MUST use these scores. Fixed-score clobbering/ignoring is unsupported.

## What Changes

- Add an **offline confidence post-operation** that estimates one score per predicted object (initially `bbox_2d` only) from bbox coordinate-token log-probabilities and writes a confidence sidecar JSONL.
- Define an explicit artifact contract for trace inputs (`artifacts.pred_token_trace_jsonl`), joined by `line_idx`, with full generated-token coverage (`generated_token_text` + `token_logprobs`) and no token filtering.
- Make span matching separator-tolerant by treating bbox coord tokens as an ordered exact subsequence (separator tokens between coords are allowed); tie-break remains earliest-unused deterministic assignment.
- Add a **scored artifact** derived from `gt_vs_pred.jsonl` by merging confidence values into `pred[*].score` **and dropping any prediction objects that cannot be assigned a valid confidence-derived score**, keeping the base inference artifact immutable.
- Extend the detection evaluator to **always honor prediction scores** for COCO ranking. `gt_vs_pred_scored.jsonl` is the **mandatory** input to COCO/mAP evaluation.
- When scored predictions are empty, the COCO evaluator exports explicit zero-valued aggregate metrics (instead of omitting keys), so dashboards and regression checks remain stable.
- Keep integration **config-first** (YAML), avoid new CLI flags, preserve geometry semantics (never reorder/drop bbox coords), and reject compatibility modes that ignore or overwrite scores.

**BREAKING**: Any evaluation workflow that relies on the evaluator overwriting/ignoring prediction scores (e.g., forcing `score=1.0` regardless of input) or on deprecated toggles that disable score honoring is unsupported. Configs MUST be updated to remove such toggles (e.g., `eval.use_pred_score`).

Non-goals (this change):
- No training changes.
- No requirement that inference backends already emit logprob traces; the post-op defines the contract and is usable once traces exist.

## Capabilities

### New Capabilities
- `confidence-postop`: Offline, CPU-only post-operation that joins prediction JSONL + token-trace sidecar and emits per-object confidence outputs (including deterministic failure reasons when confidence cannot be computed).
  - Unified scoring method: geometry + descriptor fused score (`confidence_details.method="bbox_desc_fused_logprob_exp"`) with configurable fusion weights and descriptor-span handling policies.

### Modified Capabilities
- `detection-evaluator`: COCO export MUST use per-object prediction scores; fixed-score mode (`score=1.0` for all preds) is removed.
- `inference-pipeline`: Pipeline evaluation runs MUST not expose deprecated toggles that disable score honoring; evaluation is score-aware by default.

## Impact

- Affected code (expected): `src/eval/detection.py`, `scripts/evaluate_detection.py`, `src/infer/pipeline.py`, new offline modules under `src/eval/`, and a new YAML entrypoint under `scripts/`.
- Behavior: COCO AP/mAP ranking is score-aware by default; fixed-score behavior is removed.
- Artifacts: new sidecar(s) for token traces and confidence outputs; derived scored JSONL for evaluator consumption/auditability.

Note: the confidence post-op requires CoordExp-produced inference artifacts that include `raw_output_json` for coord-bin reconstruction. Artifacts from other producers are unsupported unless they provide equivalent raw payload fields.
