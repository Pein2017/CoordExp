## Context

CoordExp’s detection evaluator computes COCO-style metrics from the unified inference artifact `gt_vs_pred.jsonl`. COCO AP/mAP requires ranking predictions by confidence. Today, both inference and evaluation force prediction scores to a constant `1.0`, so ranking is arbitrary and mAP is not confidence-sensitive.

Separately, the proposed confidence definition relies on coord-token log-probabilities for bbox coordinate tokens (`<|coord_*|>`), and the trace sidecar contract now requires full generated-token coverage (`generated_token_text` + `token_logprobs`) keyed by `line_idx` at `artifacts.pred_token_trace_jsonl`. This design therefore treats confidence extraction as a CPU-only **offline post-operation** over artifacts and makes scored evaluation mandatory (no fixed-score mode).

Constraints:
- Config-first (YAML), avoid new CLI flags.
- Preserve geometry semantics (never drop/reorder bbox coordinates inside a matched bbox slot).
- Intentionally breaking: fixed-score evaluation is removed; evaluation consumes per-object prediction scores.

## Goals / Non-Goals

**Goals:**
- Define a deterministic, CPU-only confidence pipeline that emits **one sortable score per predicted object** (initially `bbox_2d` only).
- Keep artifact boundaries auditable:
  - keep `gt_vs_pred.jsonl` immutable,
  - write confidence as a sidecar (`pred_confidence.jsonl`),
  - write a derived “scored” JSONL (`gt_vs_pred_scored.jsonl`) that merges confidence into `pred[*].score` and drops unscorable predictions for evaluator consumption.
- Extend detection evaluation to **always** use `pred[*].score` for COCO sorting; missing/invalid scores are unsupported.
- Ensure failure modes are explicit and per-object (unsupported type, span not found, trace missing, etc.).

**Non-Goals:**
- No changes to model training objectives or inference decoding strategy.
- No requirement that inference backends already emit logprob traces in this change (trace capture can follow later).
- No confidence for non-bbox geometries in the first milestone (`poly` gets `confidence=None` in `pred_confidence.jsonl` and is dropped from `gt_vs_pred_scored.jsonl`; COCO evaluation should therefore be bbox-only until poly confidence is added).

## Decisions

1) **Sidecar-first confidence outputs (do not bloat `gt_vs_pred.jsonl`)**

- Confidence is produced as a new sidecar JSONL keyed by a deterministic join key (line index + object index).
- Rationale: keeps the unified inference artifact stable, avoids retroactive schema expansion, and makes confidence methods auditable/iterable without re-running inference.

2) **Confidence definition: unified fused confidence**

For each valid `bbox_2d` object with 4 coord tokens matched in-order:
- reduce token logprobs with `mean_logprob`,
- map to score with `exp(mean_logprob)` to get a bounded `(0, 1]` value suitable for COCO sorting.

The production method is the unified fused formulation (`confidence_details.method="bbox_desc_fused_logprob_exp"`):
- geometry score comes from bbox coord-token logprobs,
- descriptor score comes from desc-span logprobs when available,
- final score is weighted fusion via `fusion.w_geom` and `fusion.w_desc`,
- descriptor-span fallbacks are controlled by `desc_span_policy` and `empty_desc_policy`.

Rationale: monotonic, bounded, comparable across objects, and directly linked to coordinate-token likelihood.

3) **Span resolution strategy (first milestone): deterministic subsequence matching**

Given an object’s raw model-output bbox bin values `[b1, b2, b3, b4]` (from the inference artifact’s raw payload, e.g. `raw_output_json`), reconstruct the expected coord token sequence:
`["<|coord_b1|>", "<|coord_b2|>", "<|coord_b3|>", "<|coord_b4|>"]`.

Search for this subsequence in the token trace’s `generated_token_text`:
- match left-to-right in object order,
- prefer the earliest unused exact match,
- allow separator tokens between expected coord tokens (ordered exact subsequence semantics),
- record ambiguity metadata when multiple matches exist.

Do not attempt to invert pixel-space `pred[*].points` back into bins; clamp/round/denorm makes this lossy.

Rationale: requires no offsets, is CPU-only, and is deterministic. Parser-assisted char-span mapping is a follow-up for improved robustness under repeated tokens.

4) **Evaluator always honors per-object scores (no fixed-score mode)**

- The evaluator uses the numeric `pred[*].score` for COCO export/scoring unconditionally.
- Missing/non-numeric scores are treated as contract violations (fail fast) rather than being replaced with a constant.
- The confidence post-op materializes `score` deterministically for each *kept* prediction in the scored artifact. Predictions that cannot be assigned a valid confidence-derived score are dropped from the scored artifact (treated as malformed/invalid rollout output).
- If no scored predictions remain for a COCO run, evaluator metrics are emitted explicitly as zeros for the active IoU families (e.g., `bbox_AP`, `bbox_AP50`, ...), preserving stable metric keys for automation.

Rationale: COCO AP/mAP is defined over ranked detections; fixed-score outputs are not confidence-evaluable and must not be treated as supported behavior.

## Risks / Trade-offs

- **[Risk] Ambiguous token matches when coord sequences repeat** → **Mitigation**: deterministic left-to-right assignment; emit `ambiguous_matches` count and matched token indices for auditability.
- **[Risk] Object alignment drift between the parsed objects used for scoring and the emitted `pred` array** → **Mitigation**: re-run the same standardization/filtering logic in post-op and attach by deterministic `object_idx` after filtering; emit explicit mismatch reasons without crashing.
- **[Risk] Missing trace sidecar prevents confidence computation** → **Mitigation**: post-op emits `confidence=None` with a deterministic `failure_reason` and drops those predictions from `gt_vs_pred_scored.jsonl`.
- **[Risk] Dropping unscorable preds can hide instrumentation/contract failures** → **Mitigation**: post-op MUST emit a run-level summary with dropped counts by reason and a kept fraction so runs can be audited/invalidated explicitly when needed.
- **[Risk] Downstream regression in eval metrics** → **Mitigation**: treat this as a breaking change; update fixtures/configs/tests to encode the new contract and fail fast on missing scores.
