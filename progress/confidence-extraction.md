# Confidence Extraction Design for mAP Foundation

Last updated: 2026-02-20
Status: design-only (no code changes in this document)
Scope: inference/rollout trace extension + offline post-operation for object-level confidence

## 1) Goal and Constraints

### Goal
Build a reproducible, CPU-only, offline confidence pipeline that estimates one sortable confidence score per parsable predicted object by using bbox coordinate-token log-probabilities (`<|coord_*|>`), so COCO-style AP/mAP ranking is meaningful.

### Hard constraints
- No training/inference execution in this task.
- Config-first (YAML), avoid new CLI flags.
- Preserve geometry semantics (never reorder/drop bbox coordinates inside a matched bbox slot).
- **No backward compatibility or legacy support**: fixed-score evaluation is unsupported; evaluation MUST honor per-object prediction scores.

## 2) Current Infrastructure Snapshot (Relevant to This Design)

### Inference artifact and current score behavior
- Unified artifact is written in `src/infer/engine.py` (`InferenceEngine.infer`) as `gt_vs_pred.jsonl`.
- Each record includes:
  - `image`, `width`, `height`, `mode`, `coord_mode`
  - `gt`, `pred`
  - `raw_output_json`, `raw_special_tokens`, `raw_ends_with_im_end`, `errors`
- Two score-clobber points currently force score ranking to be meaningless:
  - `src/infer/engine.py` (`_compact_objects`): sets every emitted object score to `1.0`
  - `src/eval/detection.py` (`_prepare_pred_objects`): sets every prepared pred score to `1.0`
- Docs explicitly reflect this in `docs/eval/README.md` ("scores fixed at 1.0").

### Parsing path (text -> standardized objects)
- `src/common/prediction_parsing.py`
  - `load_prediction_dict`: salvage parse with both field orders, choose highest retained object count
  - `parse_prediction`: parse to objects with `type`, `points`, `desc`, `_had_tokens`
- `src/utils/coordjson_transpiler.py`
  - `coordjson_to_strict_json_with_meta`: strict payload + parse meta
  - internal parser tracks record geometry spans (`record_span`, `geometry_span`) in parse structures
- `src/common/coord_standardizer.py`
  - `process_prediction_text` and `process_objects` convert parsed objects into pixel-space canonical predictions

### Coord-token contract
- `src/coord_tokens/codec.py`: CoordTok regex and 0..999 contract.
- `src/common/geometry/coord_utils.py`: coercion, denorm, and clamp helpers used across infer/eval.

### Probability plumbing reality
- Inference paths currently return text only (`GenerationResult` has `text`, `error`).
- No per-token logprob sidecar/artifact currently exists.
- Stage-2 rollout generation already uses `return_dict_in_generate=True` in `src/trainers/rollout_matching_sft.py`, but does not persist per-token logprobs.

## 3) Proposed End-State Architecture

### High-level flow
1. **Trace Capture (infer/rollout)**:
   Capture generated tokens + aligned token logprobs (optionally offsets) into a sidecar JSONL.
2. **Post Operation (offline CPU)**:
   Resolve each bbox object's 4 coord-token span and compute object confidence.
3. **Evaluation Integration (mAP)**:
   Use computed confidence as `score` for COCO sorting (mandatory; fixed-score mode unsupported).

### Why sidecar (not bloating `gt_vs_pred.jsonl`)
- Keep primary artifact stable and lightweight.
- Allow confidence methods to evolve without rerunning generation.
- Keep base artifact immutable while enabling a scored artifact for evaluation (no fixed-score evaluator path).

## 4) Data Contracts (Proposed)

## 4.1 Token trace sidecar (new artifact, recommended)
Path (proposed): `artifacts.pred_token_trace_jsonl`

One row per emitted sample:

```json
{
  "line_idx": 17,
  "image": "xxx.jpg",
  "mode": "coord",
  "generated_token_ids": [151645, 151650, ...],
  "generated_token_text": ["{", "\"", "objects", ...],
  "token_logprobs": [-0.12, -0.03, ...],
  "token_offsets": [[0,1], [1,2], ...],
  "text_sha1": "..."
}
```

Notes:
- `generated_token_text` should be emitted directly at generation time to avoid tokenizer reloading in post-op.
- `line_idx` is the primary join key to `gt_vs_pred.jsonl`.
- `token_offsets` optional at first; required for robust parser-assisted char-to-token mapping.

## 4.2 Confidence output sidecar (new artifact, post-op output)
Path (proposed): `artifacts.pred_confidence_jsonl`

```json
{
  "line_idx": 17,
  "image": "xxx.jpg",
  "objects": [
    {
      "object_idx": 0,
      "type": "bbox_2d",
      "desc": "cat",
      "points": [34, 50, 120, 180],
      "confidence": 0.8732,
      "score": 0.8732,
      "kept": true,
      "confidence_details": {
        "method": "bbox_coord_mean_logprob_exp",
        "coord_token_count": 4,
        "matched_token_indices": [231, 232, 233, 234],
        "reducer": "mean_logprob",
        "mapping": "exp",
        "ambiguous_matches": 0,
        "failure_reason": null
      }
    }
  ]
}
```

## 4.3 Mandatory scored artifact for evaluator consumption
Path (proposed): `artifacts.gt_vs_pred_scored_jsonl`
- Derived by merging confidence sidecar into `pred[*].score` **and dropping any prediction object that lacks a valid confidence-derived score**.
- Adds `pred_score_source` and `pred_score_version` per record so the evaluator can reject unscored legacy artifacts deterministically.
- Keeps base artifact immutable; makes evaluation reproducible and auditable.

Additional artifact:
- `confidence_postop_summary.json`: run-level counts dropped by `failure_reason` and a kept fraction, so instrumentation failures (e.g., `missing_trace`) are visible without scanning JSONL.

## 5) Confidence Definition

### Default definition (recommended)
For each valid `bbox_2d` object:
- Extract the 4 bbox coord tokens (`x1,y1,x2,y2`) in that order.
- Let their aligned logprobs be `lp1..lp4`.
- `mean_logprob = (lp1 + lp2 + lp3 + lp4) / 4`
- `score = exp(mean_logprob)` (geometric mean probability in `(0, 1]`)

This is monotonic, bounded, and COCO-sort friendly.

### Configurable reducers/mappings (documented, deterministic)
- Reducers: `mean_logprob` (default), `sum_logprob`, `min_logprob`, `trimmed_mean`
- Mappings: `exp` (default), `none` (raw logprob), optional calibrated `sigmoid(a*x+b)`

## 6) Deterministic Span Resolution Strategy

## 6.1 Strategy A (recommended long-term): parser-assisted span mapping
- Extend transpiler surface to expose per-record coord literal spans and record spans.
- With `token_offsets`, map char spans -> token spans deterministically.
- Best correctness under repeated tokens and repeated objects.

## 6.2 Strategy B (minimal first milestone): post-hoc subsequence matching
- Recover each bbox object's coord literals from parsed geometry values:
  - `bbox_2d: [v1,v2,v3,v4]` -> `["<|coord_v1|>", ..., "<|coord_v4|>"]`
- Search this coord-token subsequence in generated token stream.
- Deterministic tie-breaks:
  1. Prefer earliest unused exact match.
  2. Preserve object order and consume matches left-to-right.
  3. If multiple remain, choose earliest and record ambiguity count.

### Object alignment policy
- Use `line_idx` join first.
- Build object list in parsed order from `raw_output_json`.
- Re-run the same standardization/filtering logic offline (via existing standardizer behavior) to align with emitted `pred` order.
- Attach confidence by deterministic object index after filtering.

## 7) Edge-Case Policies (Explicit)

Per object:
- Non-`bbox_2d`: `confidence=None`, `failure_reason="unsupported_geometry_type"`.
- Missing geometry / malformed bbox arity: `None`, reason set.
- Numeric geometry without recoverable coord-token span: `None`, reason set.
- Fewer than 4 recoverable bbox tokens: default `None` (do not fabricate partial confidence).
- Multiple candidate spans: choose earliest per policy and log ambiguity metadata.
- Any join mismatch (`trace missing`, `line_idx mismatch`, `pred count drift`): do not crash; emit `None` with deterministic reason code.

## 8) YAML-First Integration Plan

### Inference/rollout trace capture knobs
Add YAML-only options (no CLI additions):

```yaml
infer:
  generation:
    emit_token_trace: false
    trace_include_token_text: true
    trace_include_offsets: false

artifacts:
  pred_token_trace_jsonl: null
  pred_confidence_jsonl: null
  gt_vs_pred_scored_jsonl: null
```

For rollout research (optional but aligned with this foundation):

```yaml
custom:
  rollout:
    emit_token_trace: false
    trace_out_jsonl: null
```

### Evaluation knobs
```yaml
eval:
  # Fixed-score mode is unsupported; evaluator always honors pred[*].score.
  # COCO/mAP evaluation MUST consume gt_vs_pred_scored_jsonl (scored artifact) and MUST fail fast otherwise.
```

Breaking change:
- Any eval workflow that overwrites/ignores scores (e.g., forcing `score=1.0`) is unsupported.

## 9) File-Level Patch Plan (Foundation Buildout)

## Phase 1: Confidence post-op foundation (CPU-only)
- New: `src/eval/bbox_confidence.py`
  - Pure confidence math + span resolution API
- New: `src/eval/confidence_postop.py`
  - Load `gt_vs_pred.jsonl` + trace sidecar, emit confidence sidecar + `gt_vs_pred_scored.jsonl`
- New script: `scripts/postop_confidence.py`
  - YAML entrypoint for offline post operation

## Phase 2: Evaluator score honoring (mAP enablement)
- Update `src/eval/detection.py`
  - Remove unconditional `score=1.0` clobbering and always export COCO `score` from `pred[*].score`
  - Fail fast when `score` is missing/non-numeric/NaN/inf
- Update `src/infer/pipeline.py` and `scripts/evaluate_detection.py`
  - Reject legacy toggles that disable score honoring (e.g., `eval.use_pred_score`)

## Phase 3: Infer trace emission hooks
- Update `src/infer/engine.py`
  - Extend generation result contract with optional trace payload
  - Write `pred_token_trace_jsonl` when enabled
- Update artifact resolver in `src/infer/pipeline.py`
  - Resolve optional trace/confidence artifact paths

## Phase 4: Rollout trace parity (research path)
- Update `src/trainers/rollout_matching_sft.py`
  - Optional trace emission for rollout generations
  - Keep training behavior unchanged when disabled

## 10) CPU-Only Test Plan

### Unit tests (new)
- `tests/test_bbox_confidence.py`
  - Confidence math correctness for reducers/mappings
  - Deterministic tie-break on repeated coord-token sequences
  - Edge reasons (`missing_span`, `unsupported_geometry_type`, `partial_bbox_tokens`)

### Integration tests (new/extend)
- `tests/test_confidence_postop.py`
  - Synthetic `gt_vs_pred` + trace sidecar -> confidence sidecar output contract
  - Scored artifact merge correctness
- Extend `tests/test_detection_eval_output_parity.py`
  - Missing/non-numeric scores fail fast
  - COCO export honors score ordering deterministically

### Reproducibility checks
- Fixed fixture token/logprob arrays.
- Stable sorting assertions across repeated runs.

## 11) Risks and Mitigations

- **Risk: object alignment drift between raw parse and emitted pred**
  - Mitigation: deterministic re-standardization + index-based mapping after filtering.
- **Risk: repeated coord patterns cause ambiguous span matches**
  - Mitigation: left-to-right monotonic assignment + ambiguity logging.
- **Risk: unsupported backend logprob surfaces differ (HF vs vLLM)**
  - Mitigation: trace contract at sidecar boundary; backend-specific adapters feed same schema.
- **Risk: accidental behavior regression in eval**
  - Mitigation: treat as breaking; update fixtures/configs and assert strict contract in tests (fail fast on missing scores).

## 12) Proposed OpenSpec Change Folder (for implementation)

Recommended change id:
- `openspec/changes/map-confidence-foundation-2026-02-20/`

Suggested artifacts:
- `openspec/changes/map-confidence-foundation-2026-02-20/proposal.md`
- `openspec/changes/map-confidence-foundation-2026-02-20/design.md`
- `openspec/changes/map-confidence-foundation-2026-02-20/specs/confidence-postop/spec.md`
- `openspec/changes/map-confidence-foundation-2026-02-20/specs/detection-evaluator/spec.md`
- `openspec/changes/map-confidence-foundation-2026-02-20/specs/inference-pipeline/spec.md`
- `openspec/changes/map-confidence-foundation-2026-02-20/tasks.md`

## 13) Immediate Next Milestone

Build Phase 1 + Phase 2 first:
1. offline confidence extraction from trace sidecar,
2. evaluator always honors per-object score (fail fast on missing/non-numeric scores),
3. tests assert the new strict contract + deterministic score ordering.

This gives score-aware mAP ranking capability and a clean foundation for later infer/rollout trace enrichment.
